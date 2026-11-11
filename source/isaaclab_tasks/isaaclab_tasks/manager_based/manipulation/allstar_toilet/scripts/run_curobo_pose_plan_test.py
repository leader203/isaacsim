"""Isolated cuRobo pose-planning test for the XMS5 brush task."""

import argparse
import copy
import importlib
import math
from pathlib import Path
import sys
import time

ALLSTAR_TOILET_ROOT = Path(__file__).resolve().parents[1]
if str(ALLSTAR_TOILET_ROOT) not in sys.path:
    sys.path.insert(0, str(ALLSTAR_TOILET_ROOT))

from isaaclab.app import AppLauncher
from curobo_pose_plan_utils.curobo_profiles import (
    DEFAULT_WORLD_POSE_TESTS,
    is_gripper_joint_profile as _profile_is_gripper_joint_profile,
    is_prismatic_gripper_joint_profile as _profile_is_prismatic_gripper_joint_profile,
    is_sanheyi_full_prismatic_gripper_joint_profile as _profile_is_sanheyi_full_prismatic_gripper_joint_profile,
    parse_joint_names as _parse_joint_names,
    resolve_curobo_profile as _resolve_curobo_profile_from_args,
)
from curobo_pose_plan_utils.curobo_runtime_setup import (
    regenerate_curobo_robot_cfg_from_urdf as _regenerate_curobo_robot_cfg_from_urdf_impl,
    require_curobo_preflight as _require_curobo_preflight_impl,
    setup_terminal_log as _setup_terminal_log_impl,
)
from curobo_pose_plan_utils.curobo_scene_obstacle_config import (
    SceneObstacleConfig,
    scene_obstacle_config_from_args,
)

_DEFAULT_GRIPPER_JOINT1_OUTWARD_OPEN_RAD = 0.52
_GRIPPER_JOINT1_LOWER_LIMIT_RAD = -0.52
_GRIPPER_JOINT1_GRASP_15MM_RAD = -0.315
_DEFAULT_PRISMATIC_GRIPPER_OPEN_M = 0.0
_PRISMATIC_GRIPPER_GRASP_M = 0.02
_PRISMATIC_GRIPPER_MIN_M = 0.0
_PRISMATIC_GRIPPER_MAX_M = 0.025
_GRIPPER_DROP_TARGET_POS_W = [2.097, -1.10, 0.727]
_GRIPPER_DROP_TARGET_QUAT_WXYZ = [-0.004, -0.715, -0.699, 0.016]   # 改为抓取姿态(与 default_target_quat 一致)，手腕只需转 ~20°
_GRIPPER_GRASP_VEL_EPS_RAD_S = 0.01
_GRIPPER_GRASP_POS_EPS_RAD = 0.01
_PRISMATIC_GRIPPER_GRASP_VEL_EPS_M_S = 0.001
_PRISMATIC_GRIPPER_GRASP_POS_EPS_M = 0.001
_BASE_TELEOP_PRE_ARM_JOINT_POS_DEG = [-2.89, -22.96, -28.61, 0.0, -137.52, 0.0]
_BASE_TELEOP_DEFAULT_SPEED_SCALE = 150.0
_BASE_TELEOP_DEFAULT_MAX_DURATION_S = 300.0
_DIRECT_JOINT6_ROLL_WAYPOINTS = 61
_CUROBO_SCENE_CUBOID_OBSTACLES: list[dict] = []

# Paper-waste logical-attach state (paper_path → (gripper_body_name, offset_w))
# offset_w = paper_pos_w - gripper_center_w at grasp time
_PAPER_ATTACH_STATE: dict[str, tuple[str, tuple[float, float, float]]] = {}
# Saved main grab trajectory for reverse-execution return (GPT review P0)
_LAST_GRAB_TRAJECTORY = None  # type: ignore[assignment] — set to torch.Tensor after import
_LAST_GRAB_GRIPPER_POS: float | None = None
# Track which paper was actually grasped (fix: only teleport grasped papers)
_LAST_GRASPED_PAPER_PATH: str | None = None
# Logical-attach state for carried papers (GPT P1 v2: visual-only, NOT under articulation)
_CARRIED_VISUAL_STATE: dict[str, dict] = {}  # paper_id → {carried_path, local_offset, source_prim_path, state}

# ---- grab phase timing instrumentation (2026-09-14) ----
_G_TIMING: dict[str, float] = {}
_G_TIMING_ORDER: list[str] = []
_G_CAMERA_WINDOW_CALLS = 0
_G_CAMERA_WINDOW_S = 0.0
_G_CAMERA_SYNC_CALLS = 0
_G_CAMERA_SYNC_S = 0.0


def _g_timing_reset() -> None:
    global _G_TIMING, _G_TIMING_ORDER, _G_CAMERA_WINDOW_CALLS, _G_CAMERA_WINDOW_S, _G_CAMERA_SYNC_CALLS, _G_CAMERA_SYNC_S
    _G_TIMING = {}
    _G_TIMING_ORDER = []
    _G_CAMERA_WINDOW_CALLS = 0
    _G_CAMERA_WINDOW_S = 0.0
    _G_CAMERA_SYNC_CALLS = 0
    _G_CAMERA_SYNC_S = 0.0


def _g_timing_mark(phase: str, t0: float) -> float:
    """Record elapsed since ``t0`` under ``phase``; return ``time.perf_counter()`` for chaining."""
    now = time.perf_counter()
    _G_TIMING[phase] = now - t0
    if phase not in _G_TIMING_ORDER:
        _G_TIMING_ORDER.append(phase)
    return now


def _g_print_summary(header: str = "GrabTimingSummary") -> None:
    """Print per-phase timing breakdown (plan phases + execute phases + camera)."""
    plan_phases = ("pre_target_move", "target_pose_setup", "curobo_plan")
    exec_phases = ("target_traj_exec", "descend", "feedback_grasp", "lift")
    plan_t = sum(_G_TIMING.get(k, 0.0) for k in plan_phases)
    exec_t = sum(_G_TIMING.get(k, 0.0) for k in exec_phases)
    grand = plan_t + exec_t
    print(f"[{header}] total(plan+execute)={grand:.3f}s (plan={plan_t:.3f}s, execute={exec_t:.3f}s)", flush=True)

    def _print_group(title, phases, group_total):
        print(f"[{header}]   -- {title} (sum {group_total:.3f}s) --", flush=True)
        for phase in phases:
            t = _G_TIMING.get(phase, 0.0)
            pct = 100.0 * t / group_total if group_total > 0 else 0.0
            print(f"[{header}]     {phase:<22} {t:8.3f}s  {pct:5.1f}%", flush=True)

    _print_group("PLAN (cuRobo 规划)", plan_phases, plan_t)
    _print_group("EXECUTE (机械臂执行)", exec_phases, exec_t)
    if _G_CAMERA_WINDOW_CALLS:
        print(
            f"[{header}]   {'camera_windows_total':<22} {_G_CAMERA_WINDOW_S:8.3f}s  "
            f"({_G_CAMERA_WINDOW_CALLS} calls, avg {1e3 * _G_CAMERA_WINDOW_S / _G_CAMERA_WINDOW_CALLS:.2f}ms)",
            flush=True,
        )
    if _G_CAMERA_SYNC_CALLS:
        print(
            f"[{header}]   {'camera_sync_total':<22} {_G_CAMERA_SYNC_S:8.3f}s  "
            f"({_G_CAMERA_SYNC_CALLS} calls, avg {1e3 * _G_CAMERA_SYNC_S / _G_CAMERA_SYNC_CALLS:.2f}ms)",
            flush=True,
        )
_RECORDER_PRISMATIC_CUROBO_TASK_ID = (
    "Isaac-Allstar-Toilet-Mecanum0730-XMS5GripperJointPrismatic-"
    "GripperJointPrismatic-CuroboPosePlanTest-Dirty-OpenSeat-v0"
)
_SANHEYI_FULL_PRISMATIC_CUROBO_TASK_ID = (
    "Isaac-Allstar-Toilet-Mecanum0730-XMS5GripperJointPrismaticSanheyiFull-"
    "GripperJointPrismatic-CuroboPosePlanTest-Dirty-OpenSeat-v0"
)
_SWITCH_PRESERVED_RIGID_ASSET_FALLBACKS = (
    "grasp_test_cube_1",
    "grasp_test_cube_2",
    "grasp_test_cube_3",
    "trash_bin",
)

parser = argparse.ArgumentParser(description="Run an isolated cuRobo pose-planning test.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Allstar-Toilet-Mecanum0730-XMS5GripperJointPrismatic-CuroboPosePlanTest-Dirty-OpenSeat-v0",
)
parser.add_argument(
    "--target_pos",
    type=float,
    nargs=3,
    default=[1.47, -1.101, 0.07],
    metavar=("X", "Y", "Z"),
    help="Optional extra world-frame target position for tool_tip_link.",
)
parser.add_argument(
    "--target_quat_wxyz",
    type=float,
    nargs=4,
    default=None,
    metavar=("W", "X", "Y", "Z"),
    help="Optional extra world-frame target orientation for tool_tip_link.",
)
parser.add_argument(
    "--default_target_quat_wxyz",
    type=float,
    nargs=4,
    default=None,
    metavar=("W", "X", "Y", "Z"),
    help="Default world-frame target orientation used when --target_pos is provided without --target_quat_wxyz.",
)
parser.add_argument(
    "--urdf_path",
    type=str,
    default=None,
    help="URDF used by cuRobo. Defaults to the URDF matching the scene robot USD.",
)
parser.add_argument(
    "--curobo_robot_cfg_path",
    type=str,
    default=None,
    help="Output/input cuRobo robot yaml. Defaults to source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/configs/curobo/<urdf-stem>.yml.",
)
parser.add_argument(
    "--base_link",
    type=str,
    default="XMS5-R800-W4G3B4C_base",
    help="cuRobo base link.",
)
parser.add_argument(
    "--tool_frame",
    type=str,
    default="tool_tip_link",
    help="cuRobo tool frame / end-effector link.",
)
parser.add_argument(
    "--joint_names",
    type=str,
    default="joint1,joint2,joint3,joint4,joint5,joint6",
    help="Comma-separated cuRobo active joint names.",
)
parser.add_argument(
    "--plan_end_effector_roll_rad",
    type=float,
    default=None,
    help="Optional extra roll added to the final active joint after each successful pose plan.",
)
parser.add_argument(
    "--gripper_joint1_pos",
    type=float,
    default=_DEFAULT_GRIPPER_JOINT1_OUTWARD_OPEN_RAD,
    help="Raw gripper_joint1 position target in radians for gripper-joint tasks. Default fully opens outward.",
)
parser.add_argument(
    "--prismatic_gripper_open_pos",
    type=float,
    default=_DEFAULT_PRISMATIC_GRIPPER_OPEN_M,
    help="Open gripper_joint1 target in meters for prismatic gripper-joint tasks.",
)
parser.add_argument(
    "--prismatic_gripper_grasp_pos",
    type=float,
    default=_PRISMATIC_GRIPPER_GRASP_M,
    help="Grasp gripper_joint1 target in meters for prismatic gripper-joint tasks.",
)
parser.add_argument(
    "--prismatic_gripper_min",
    type=float,
    default=_PRISMATIC_GRIPPER_MIN_M,
    help="Minimum gripper_joint1 target in meters for prismatic gripper-joint tasks.",
)
parser.add_argument(
    "--prismatic_gripper_max",
    type=float,
    default=_PRISMATIC_GRIPPER_MAX_M,
    help="Maximum gripper_joint1 target in meters for prismatic gripper-joint tasks.",
)
parser.add_argument(
    "--post_target_descend_m",
    type=float,
    default=0.02,
    help=(
        "After reaching the requested target, plan and execute an extra world -Z descent before grasping. "
        "Defaults to 0.02 m for prismatic gripper tasks and 0.0 m otherwise."
    ),
)
parser.add_argument(
    "--post_target_descend_hold_steps",
    type=int,
    default=None,
    help="Extra hold steps after the post-target descent. Defaults to --final_hold_steps.",
)
parser.add_argument(
    "--gripper_grasp_max_steps",
    type=int,
    default=None,
    help=(
        "Maximum env.step() calls for gripper feedback close. "
        "Defaults to max(--final_hold_steps, --steps_per_waypoint * 25)."
    ),
)
parser.add_argument(
    "--gripper_hold_steps",
    type=int,
    default=None,
    help=(
        "Maximum env.step() calls for gripper grasp/lift/drop hold phases. "
        "Defaults to max(--final_hold_steps, --steps_per_waypoint * 25)."
    ),
)
parser.add_argument(
    "--post_grasp_lift_m",
    type=float,
    default=0.10,
    help="After grasping, plan and execute a world +Z lift by this distance in meters.",
)
parser.add_argument(
    "--initial_arm_joint_pos",
    type=float,
    nargs="+",
    default=None,
    metavar="J",
    help="Optional safe initial arm joint position used before planning and execution.",
)
parser.add_argument(
    "--initial_arm_settle_steps",
    type=int,
    default=240,
    help="Number of env.step calls used to move to --initial_arm_joint_pos before planning.",
)
parser.add_argument(
    "--pre_target_joint_pos_deg",
    type=float,
    nargs="+",
    default=None,
    metavar="J",
    help="Pre-target arm joint pose in degrees. Defaults are selected from --task unless explicitly provided.",
)
parser.add_argument(
    "--use_pre_target_joint_pose",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Move to --pre_target_joint_pos_deg before planning each target.",
)
parser.add_argument(
    "--pre_target_joint_steps",
    type=int,
    default=240,
    help="Number of env.step calls used to move to the pre-target joint pose.",
)
parser.add_argument(
    "--pose_names",
    type=str,
    default="",
    help="Comma-separated pose names to execute. Empty string runs all built-in poses.",
)
parser.add_argument(
    "--multi_pose_polynomial_seed",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Plan selected poses as one multi-point polynomial-seed cuRobo trajectory instead of separate pose plans.",
)
parser.add_argument(
    "--multi_pose_seed_mode",
    type=str,
    default="cubic",
    choices=("linear", "cubic", "quintic"),
    help="Joint-space interpolation used to seed --multi_pose_polynomial_seed.",
)
parser.add_argument(
    "--settle_timeout_s",
    type=float,
    default=8.0,
)
parser.add_argument(
    "--worker_timeout_s",
    type=float,
    default=180.0,
    help="Timeout for the isolated cuRobo planning worker.",
)
parser.add_argument(
    "--worker_poll_interval_s",
    type=float,
    default=0.1,
    help=(
        "How often to poll the isolated cuRobo worker. While waiting, the simulator continuously "
        "steps and holds the fixed worker-start joint target when one is provided."
    ),
)
parser.add_argument(
    "--worker_heartbeat_s",
    type=float,
    default=5.0,
    help="How often to print a heartbeat while waiting for the isolated cuRobo worker.",
)
parser.add_argument(
    "--arm_drive_stiffness",
    type=float,
    default=None,
    help="Runtime override for all arm joint drive stiffness values. Disabled when omitted.",
)
parser.add_argument(
    "--arm_drive_damping",
    type=float,
    default=None,
    help="Runtime override for all arm joint drive damping values. Disabled when omitted.",
)
parser.add_argument(
    "--arm_drive_effort_limit",
    type=float,
    default=None,
    help="Runtime override for all arm joint effort/force limits. Disabled when omitted.",
)
parser.add_argument(
    "--arm_drive_velocity_limit",
    type=float,
    default=None,
    help="Runtime override for all arm joint velocity limits in rad/s. Disabled when omitted.",
)
parser.add_argument(
    "--arm_drive_joint2_stiffness",
    type=float,
    default=None,
    help="Runtime override for joint2 drive stiffness after --arm_drive_stiffness is applied.",
)
parser.add_argument(
    "--arm_drive_joint2_damping",
    type=float,
    default=None,
    help="Runtime override for joint2 drive damping after --arm_drive_damping is applied.",
)
parser.add_argument(
    "--arm_drive_joint2_effort_limit",
    type=float,
    default=None,
    help="Runtime override for joint2 effort/force limit after --arm_drive_effort_limit is applied.",
)
parser.add_argument(
    "--arm_drive_joint2_velocity_limit",
    type=float,
    default=None,
    help="Runtime override for joint2 velocity limit after --arm_drive_velocity_limit is applied.",
)
parser.add_argument(
    "--arm_drive_joint3_stiffness",
    type=float,
    default=None,
    help="Runtime override for joint3 drive stiffness after --arm_drive_stiffness is applied.",
)
parser.add_argument(
    "--arm_drive_joint3_damping",
    type=float,
    default=None,
    help="Runtime override for joint3 drive damping after --arm_drive_damping is applied.",
)
parser.add_argument(
    "--arm_drive_joint3_effort_limit",
    type=float,
    default=None,
    help="Runtime override for joint3 effort/force limit after --arm_drive_effort_limit is applied.",
)
parser.add_argument(
    "--arm_drive_joint3_velocity_limit",
    type=float,
    default=None,
    help="Runtime override for joint3 velocity limit after --arm_drive_velocity_limit is applied.",
)
parser.add_argument(
    "--arm_hold_effort_bias",
    type=float,
    nargs=6,
    default=None,
    metavar="TAU",
    help=(
        "Optional feed-forward effort bias for joint1..joint6, applied only while holding a fixed arm target "
        "with zero velocity target. Defaults to all zeros."
    ),
)
parser.add_argument(
    "--arm_hold_effort_bias_joint2",
    type=float,
    default=None,
    help="Override the joint2 value in --arm_hold_effort_bias for fixed-target hold phases.",
)
parser.add_argument(
    "--arm_hold_effort_bias_joint3",
    type=float,
    default=None,
    help="Override the joint3 value in --arm_hold_effort_bias for fixed-target hold phases.",
)
parser.add_argument(
    "--arm_hold_effort_bias_joint5",
    type=float,
    default=None,
    help="Override the joint5 value in --arm_hold_effort_bias for fixed-target hold phases.",
)
parser.add_argument(
    "--arm_hold_effort_bias_min_abs_joint3",
    type=float,
    default=0.0,
    help=(
        "Only apply --arm_hold_effort_bias when abs(target joint3) is at least this value. "
        "Useful for avoiding over-compensation near the reset pose."
    ),
)
parser.add_argument(
    "--steps_per_waypoint",
    type=int,
    default=6,
    help="Number of env.step() calls used for each non-final cuRobo waypoint.",
)
parser.add_argument(
    "--trajectory_velocity_feedforward",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Set arm joint velocity targets from adjacent cuRobo waypoints during trajectory execution. "
        "Hold and final-settle phases still command zero velocity."
    ),
)
parser.add_argument(
    "--trajectory_velocity_feedforward_scale",
    type=float,
    default=1.0,
    help="Scale applied to trajectory joint velocity feedforward before clamping to arm velocity limits.",
)
parser.add_argument(
    "--waypoint_log_stride",
    type=int,
    default=5,
    help="Log actual joint tracking every N waypoints while executing a trajectory.",
)
parser.add_argument(
    "--joint_target_tolerance",
    type=float,
    default=0.015,
    help="Maximum joint error in rad before moving to the next waypoint.",
)
parser.add_argument(
    "--trajectory_final_joint_target_tolerance",
    type=float,
    default=None,
    help=(
        "Optional stricter joint-error tolerance for the final waypoint of each executed cuRobo trajectory. "
        "Defaults to --joint_target_tolerance."
    ),
)
parser.add_argument(
    "--post_trajectory_settle_steps",
    type=int,
    default=0,
    help=(
        "Extra fixed env.step() calls after each cuRobo trajectory while holding the final waypoint with "
        "zero arm velocity target. This is mainly for diagnosing residual drift after dynamic tracking."
    ),
)
parser.add_argument(
    "--post_trajectory_settle_log_stride",
    type=int,
    default=25,
    help="Log every N post-trajectory settle steps, plus the first and last step.",
)
parser.add_argument(
    "--max_steps_per_waypoint",
    type=int,
    default=100,
    help="Maximum env.step() calls allowed for the final waypoint when waiting for joint convergence. Capped at 100.",
)
parser.add_argument(
    "--return_to_initial_steps",
    type=int,
    default=120,
    help="Number of env.step() calls used to return the arm to the initial joint pose after each target.",
)
parser.add_argument(
    "--return_via_reverse_trajectory",
    action="store_true",
    default=True,
    help="Return to the initial arm pose by replaying the executed trajectory in reverse.",
)
parser.add_argument(
    "--pause_after_execute_s",
    type=float,
    default=2.0,
    help="Seconds to pause after each executed trajectory before returning to the initial pose.",
)
parser.add_argument(
    "--force_direct",
    action="store_true",
    help="Deprecated compatibility option. Direct planning is always used.",
)
parser.add_argument(
    "--plan_return_with_curobo",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Deprecated compatibility option. Return is executed without an extra cuRobo planning phase.",
)
parser.add_argument(
    "--final_hold_steps",
    type=int,
    default=240,
    help="Extra env.step() calls holding the final trajectory waypoint before measuring pose error.",
)
parser.add_argument(
    "--disable_toilet_mesh_collision",
    action="store_true",
    help="Disable cuRobo mesh collision obstacles exported from the toilet ceramic shell.",
)
parser.add_argument(
    "--enable_toilet_mesh_collision",
    action="store_true",
    help="Compatibility flag; toilet mesh obstacles are enabled unless --disable_toilet_mesh_collision is set.",
)
parser.add_argument(
    "--curobo_mesh_obstacle_mode",
    type=str,
    default="full",
    choices=("full", "simplified"),
    help="Use full exported toilet meshes or simplified proxy OBJ meshes for cuRobo scene obstacles.",
)
parser.add_argument(
    "--curobo_mesh_target_faces",
    type=int,
    default=60000,
    help="Target triangle count per toilet root when --curobo_mesh_obstacle_mode=simplified.",
)
parser.add_argument(
    "--curobo_mesh_preserve_lid_seat",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Keep ToiletLid/ToiletSeat triangles unsimplified in simplified cuRobo mesh obstacle mode.",
)
parser.add_argument(
    "--curobo_mesh_lid_seat_target_faces",
    type=int,
    default=30000,
    help=(
        "Target total ToiletLid/ToiletSeat triangle count per toilet root when using simplified cuRobo mesh "
        "obstacles and --no-curobo_mesh_preserve_lid_seat."
    ),
)
parser.add_argument(
    "--curobo_mesh_preserve_outline",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Preserve boundary, sharp-edge, and extreme outline vertices during fallback vertex clustering.",
)
parser.add_argument(
    "--curobo_mesh_feature_angle_deg",
    type=float,
    default=35.0,
    help="Dihedral angle threshold in degrees used to mark sharp outline edges for fallback vertex clustering.",
)
parser.add_argument(
    "--disable_curobo_box_collision",
    action="store_true",
    help="Disable cuRobo cuboid obstacles exported from nearby wall/partition collision box primitives.",
)
parser.add_argument(
    "--curobo_box_collision_front_depth_m",
    type=float,
    default=2.0,
    help="Include wall/partition collision boxes intersecting the robot-base +X front window up to this depth. Use <=0 for no far limit.",
)
parser.add_argument(
    "--curobo_box_collision_front_half_width_m",
    type=float,
    default=1.0,
    help="Half width in robot-base Y for the front wall/partition collision box window. Use <=0 for no width limit.",
)
parser.add_argument(
    "--curobo_box_collision_front_min_x_m",
    type=float,
    default=0.0,
    help="Minimum robot-base X for front wall/partition collision boxes. A box is kept if any part reaches this X.",
)
parser.add_argument(
    "--disable_auto_curobo_robot_cfg",
    action="store_true",
    help="Do not regenerate cuRobo robot cfg from URDF sphere collisions before planning.",
)
parser.add_argument(
    "--curobo_collision_sphere_buffer",
    type=float,
    default=0.025,
    help="Extra radius buffer in meters added by cuRobo to all robot collision spheres.",
)
parser.add_argument(
    "--curobo_lock_joints",
    type=str,
    default="",
    help=(
        "Comma-separated name=value pairs of arm joints to lock in the cuRobo robot cfg "
        "(e.g. 'joint4=0.0'). Locked joints are removed from the planning cspace and held at "
        "the given value. Empty string (default) disables locking."
    ),
)
parser.add_argument(
    "--curobo_cspace_distance_weight",
    type=str,
    default="joint4=4.0",
    help=(
        "Comma-separated name=value pairs overriding per-joint cspace_distance_weight "
        "(default 1.0 for unlisted joints). Higher = cuRobo penalises moving that joint more, "
        "which damps branch-flip / large rotations on that joint. Empty string disables."
    ),
)
parser.add_argument(
    "--curobo_collision_activation_distance",
    type=float,
    default=0.06,
    help="Distance in meters where cuRobo starts optimizing away from scene collisions.",
)
parser.add_argument(
    "--curobo_position_tolerance_m",
    type=float,
    default=0.005,
    help="cuRobo position tolerance in meters for normal pose targets.",
)
parser.add_argument(
    "--curobo_orientation_tolerance_deg",
    type=float,
    default=15.0,
    help="cuRobo orientation tolerance in degrees for normal pose targets.",
)
parser.add_argument(
    "--curobo_num_ik_seeds",
    type=int,
    default=8,
    help="Number of cuRobo IK seeds used for each pose plan.",
)
parser.add_argument(
    "--curobo_num_trajopt_seeds",
    type=int,
    default=4,
    help="Number of cuRobo trajectory optimization seeds used for each pose plan.",
)
parser.add_argument(
    "--curobo_random_seed",
    type=int,
    default=123,
    help="cuRobo random seed for reproducibility. Change it (e.g. 123/456/789) to get "
    "different IK/trajopt solutions for more varied demos.",
)
parser.add_argument(
    "--curobo_high_z_tolerance_threshold_m",
    type=float,
    default=0.5,
    help="Use relaxed cuRobo pose tolerances when any base-frame target z is above this height.",
)
parser.add_argument(
    "--curobo_high_z_position_tolerance_m",
    type=float,
    default=0.03,
    help="cuRobo position tolerance for high-z pose targets.",
)
parser.add_argument(
    "--curobo_high_z_orientation_tolerance_deg",
    type=float,
    default=15.0,
    help="cuRobo orientation tolerance in degrees for high-z pose targets.",
)
parser.add_argument(
    "--record_video",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record the observer/viewer RGB view to an mp4 file.",
)
parser.add_argument(
    "--video_output_dir",
    type=str,
    default=str(ALLSTAR_TOILET_ROOT / "outputs" / "curobo"),
    help="Directory where observer mp4 videos are written.",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=15,
    help="Frames per second for the recorded mp4.",
)
parser.add_argument(
    "--video_every_n_steps",
    type=int,
    default=10,
    help="Capture one video frame every N env.step calls.",
)
parser.add_argument(
    "--video_resolution",
    type=int,
    nargs=2,
    default=(640, 360),
    metavar=("WIDTH", "HEIGHT"),
    help="Viewer render resolution used for mp4 recording.",
)
parser.add_argument(
    "--tool_api_server",
    action="store_true",
    help="Expose cuRobo pose planning and trajectory execution through a FastAPI bridge.",
)
parser.add_argument(
    "--tool_api_host",
    type=str,
    default="127.0.0.1",
    help="Bind address for --tool_api_server.",
)
parser.add_argument(
    "--tool_api_port",
    type=int,
    default=8000,
    help="Port for --tool_api_server.",
)
parser.add_argument(
    "--record_dataset",
    type=str,
    default="",
    help="HDF5 file path to record cube-grasp demonstrations. One demo = one full SOP "
    "(grab -> move_arm_to_joints -> take_out -> move_arm_to_joints). Empty disables recording.",
)
parser.add_argument(
    "--record_cameras",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record cam_tool / cam_fixed RGBD frames into each demo (requires rendering).",
)
parser.add_argument(
    "--record_flush_every",
    type=int,
    default=64,
    help="Flush recorded frames to the HDF5 file every N frames.",
)
parser.add_argument(
    "--record_max_frames_per_demo",
    type=int,
    default=5000,
    help="Safety cap on frames per demo; longer demos are truncated with a warning.",
)
parser.add_argument(
    "--record_stride",
    type=int,
    default=4,
    help="Record one frame every N env.steps. Full-rate 460x460 dual-camera RGBD is ~3MB/"
    "step (~30-60GB per full SOP); stride=5 gives ~768 frames (~2GB) per demo (24fps).",
)
parser.add_argument(
    "--initial_robot_id",
    type=str,
    default=None,
    help="Initial robot profile for the common profile-switching ToolAPI task.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Work around NVIDIA Vulkan driver version overflow bug (>535.255):
# VkPhysicalDeviceProperties::driverVersion encodes minor 309 as 53 (8-bit wrap).
# Without this flag, gpu.foundation.plugin rejects the driver as unsupported.
_RTX_DRIVER_FIX = "--/rtx/verifyDriverVersion/enabled=false"
if _RTX_DRIVER_FIX not in args_cli.kit_args:
    if args_cli.kit_args:
        args_cli.kit_args += " " + _RTX_DRIVER_FIX
    else:
        args_cli.kit_args = _RTX_DRIVER_FIX


def _option_was_passed(*option_names: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:] for name in option_names)


def _apply_recorder_prismatic_curobo_defaults() -> None:
    if args_cli.task not in {
        _RECORDER_PRISMATIC_CUROBO_TASK_ID,
        _SANHEYI_FULL_PRISMATIC_CUROBO_TASK_ID,
    }:
        return

    defaults = {
        "max_steps_per_waypoint": (300, ("--max_steps_per_waypoint",)),
        "trajectory_velocity_feedforward": (
            True,
            ("--trajectory_velocity_feedforward", "--no-trajectory_velocity_feedforward"),
        ),
        "trajectory_velocity_feedforward_scale": (1.0, ("--trajectory_velocity_feedforward_scale",)),
        "joint_target_tolerance": (0.005, ("--joint_target_tolerance",)),
        "trajectory_final_joint_target_tolerance": (0.003, ("--trajectory_final_joint_target_tolerance",)),
        "post_trajectory_settle_steps": (40, ("--post_trajectory_settle_steps",)),
        "post_trajectory_settle_log_stride": (25, ("--post_trajectory_settle_log_stride",)),
        "arm_hold_effort_bias": ([0.0, -26.5, 13.5, 0.0, 0.0, 0.0], ("--arm_hold_effort_bias",)),
        "arm_hold_effort_bias_min_abs_joint3": (0.1, ("--arm_hold_effort_bias_min_abs_joint3",)),
        "worker_heartbeat_s": (5.0, ("--worker_heartbeat_s",)),
        "post_target_descend_m": (0.03, ("--post_target_descend_m",)),
        "post_target_descend_hold_steps": (40, ("--post_target_descend_hold_steps",)),
        "gripper_grasp_max_steps": (180, ("--gripper_grasp_max_steps",)),
        "gripper_hold_steps": (60, ("--gripper_hold_steps",)),
        "post_grasp_lift_m": (0.20, ("--post_grasp_lift_m",)),
        "prismatic_gripper_open_pos": (0.0, ("--prismatic_gripper_open_pos",)),
        "prismatic_gripper_grasp_pos": (0.020, ("--prismatic_gripper_grasp_pos",)),
        "prismatic_gripper_min": (0.0, ("--prismatic_gripper_min",)),
        "prismatic_gripper_max": (0.025, ("--prismatic_gripper_max",)),
        "enable_toilet_mesh_collision": (True, ("--enable_toilet_mesh_collision",)),
        "curobo_mesh_obstacle_mode": (("simplified"), ("--curobo_mesh_obstacle_mode",)),
        "curobo_mesh_target_faces": (500, ("--curobo_mesh_target_faces",)),
        "curobo_mesh_lid_seat_target_faces": (100, ("--curobo_mesh_lid_seat_target_faces",)),
        "curobo_mesh_preserve_outline": (
            True,
            ("--curobo_mesh_preserve_outline", "--no-curobo_mesh_preserve_outline"),
        ),
        "curobo_mesh_feature_angle_deg": (35.0, ("--curobo_mesh_feature_angle_deg",)),
        "curobo_box_collision_front_depth_m": (2.0, ("--curobo_box_collision_front_depth_m",)),
        "curobo_box_collision_front_half_width_m": (1.0, ("--curobo_box_collision_front_half_width_m",)),
        "curobo_box_collision_front_min_x_m": (0.0, ("--curobo_box_collision_front_min_x_m",)),
        "curobo_collision_sphere_buffer": (0.01, ("--curobo_collision_sphere_buffer",)),
        "curobo_collision_activation_distance": (0.06, ("--curobo_collision_activation_distance",)),
        "curobo_position_tolerance_m": (0.005, ("--curobo_position_tolerance_m",)),
        "curobo_orientation_tolerance_deg": (15.0, ("--curobo_orientation_tolerance_deg",)),
        "final_hold_steps": (80, ("--final_hold_steps",)),
        "record_video": (True, ("--record_video", "--no-record_video")),
    }
    for attr_name, (default_value, option_names) in defaults.items():
        if not _option_was_passed(*option_names):
            setattr(args_cli, attr_name, default_value)


_apply_recorder_prismatic_curobo_defaults()
args_cli.enable_cameras = (
    bool(getattr(args_cli, "enable_cameras", False))
    or bool(args_cli.record_video)
    or not args_cli.headless
)

TASK_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(__file__).resolve().parents[7]


def _setup_terminal_log() -> int:
    return _setup_terminal_log_impl(TASK_ROOT)


_LOG_FD = _setup_terminal_log()


def _require_curobo_preflight() -> None:
    _require_curobo_preflight_impl(TASK_ROOT)


_require_curobo_preflight()


def _curobo_name_value_pairs(arg_name: str) -> dict[str, float]:
    """Parse a 'name=value,name=value' CLI string into {name: float}."""
    raw = str(getattr(args_cli, arg_name, "") or "").strip()
    if not raw:
        return {}
    pairs: dict[str, float] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"{arg_name} entry must be name=value, got {chunk!r}")
        name, value = chunk.split("=", 1)
        pairs[name.strip()] = float(value)
    return pairs


def _curobo_lock_joints_from_args() -> dict[str, float]:
    return _curobo_name_value_pairs("--curobo_lock_joints")


def _curobo_cspace_distance_weight_from_args() -> dict[str, float]:
    return _curobo_name_value_pairs("--curobo_cspace_distance_weight")


def _regenerate_curobo_robot_cfg_from_urdf(
    urdf_path: str | Path,
    output_path: str | Path,
    *,
    collision_sphere_buffer: float,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
) -> Path:
    return _regenerate_curobo_robot_cfg_from_urdf_impl(
        urdf_path,
        output_path,
        script_dir=Path(__file__).resolve().parent,
        collision_sphere_buffer=collision_sphere_buffer,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
        lock_joints=_curobo_lock_joints_from_args(),
        cspace_distance_weight=_curobo_cspace_distance_weight_from_args(),
    )


def _is_gripper_joint_profile() -> bool:
    return _profile_is_gripper_joint_profile(args_cli.task, args_cli.urdf_path)


def _is_sanheyi_full_prismatic_gripper_joint_profile() -> bool:
    return _profile_is_sanheyi_full_prismatic_gripper_joint_profile(args_cli.task, args_cli.urdf_path)


def _is_prismatic_gripper_joint_profile() -> bool:
    return _profile_is_prismatic_gripper_joint_profile(args_cli.task, args_cli.urdf_path)


def _prismatic_gripper_joint1_target(value: float) -> float:
    return max(float(args_cli.prismatic_gripper_min), min(float(args_cli.prismatic_gripper_max), float(value)))


def _default_open_gripper_joint1_pos() -> float:
    if _is_prismatic_gripper_joint_profile():
        return _prismatic_gripper_joint1_target(float(args_cli.prismatic_gripper_open_pos))
    return float(_DEFAULT_GRIPPER_JOINT1_OUTWARD_OPEN_RAD)


def _default_grasp_gripper_joint1_pos() -> float:
    if _is_prismatic_gripper_joint_profile():
        return _prismatic_gripper_joint1_target(float(args_cli.prismatic_gripper_grasp_pos))
    return max(float(_GRIPPER_JOINT1_LOWER_LIMIT_RAD), float(_GRIPPER_JOINT1_GRASP_15MM_RAD))


def _post_target_descend_m() -> float:
    if args_cli.post_target_descend_m is not None:
        return max(0.0, float(args_cli.post_target_descend_m))
    return 0.02 if _is_prismatic_gripper_joint_profile() else 0.0


def _default_gripper_grasp_max_steps() -> int:
    fallback = max(int(args_cli.final_hold_steps), int(args_cli.steps_per_waypoint) * 25)
    if args_cli.gripper_grasp_max_steps is None:
        return fallback
    return max(1, int(args_cli.gripper_grasp_max_steps))


def _default_gripper_hold_steps() -> int:
    min_hold_steps = max(1, int(args_cli.steps_per_waypoint))
    fallback = max(int(args_cli.final_hold_steps), int(args_cli.steps_per_waypoint) * 25)
    if args_cli.gripper_hold_steps is None:
        return max(min_hold_steps, fallback)
    return max(min_hold_steps, int(args_cli.gripper_hold_steps))


def _scene_robot_usd_path(env_cfg) -> str | None:
    scene_cfg = getattr(env_cfg, "scene", None)
    robot_cfg = getattr(scene_cfg, "robot", None)
    spawn_cfg = getattr(robot_cfg, "spawn", None)
    usd_path = getattr(spawn_cfg, "usd_path", None)
    return str(usd_path) if usd_path else None


def _common_robot_profiles(env_cfg) -> dict:
    return dict(
        getattr(env_cfg, "robot_profiles", None)
        or getattr(getattr(env_cfg, "scene", None), "robot_profiles", {})
        or {}
    )


def _common_robot_usd_root(env_cfg) -> str | None:
    robot_usd_root = getattr(env_cfg, "robot_usd_root", None) or getattr(
        getattr(env_cfg, "scene", None), "robot_usd_root", None
    )
    if robot_usd_root:
        return str(robot_usd_root)
    profiles = _common_robot_profiles(env_cfg)
    if profiles:
        current_usd = _scene_robot_usd_path(env_cfg)
        if current_usd:
            return str(Path(current_usd).parent)
    return None


def _profile_usd_path(env_cfg, robot_id: str) -> str:
    profiles = _common_robot_profiles(env_cfg)
    if robot_id not in profiles:
        raise ValueError(f"Unknown robot_id={robot_id!r}; available={sorted(profiles)}")
    profile = profiles[robot_id]
    usd_path = str(profile.get("usd_path") or "")
    if usd_path:
        return usd_path
    usd_name = str(profile.get("usd_name") or "")
    if not usd_name:
        raise ValueError(f"Robot profile {robot_id!r} must define usd_name or usd_path.")
    usd_root = _common_robot_usd_root(env_cfg)
    if not usd_root:
        raise ValueError(f"Cannot resolve USD root for robot profile {robot_id!r}.")
    return str(Path(usd_root) / usd_name)


def _apply_common_initial_robot_profile(env_cfg) -> str | None:
    profiles = _common_robot_profiles(env_cfg)
    if not profiles:
        return None
    robot_id = str(
        args_cli.initial_robot_id
        or getattr(env_cfg, "default_robot_id", None)
        or getattr(getattr(env_cfg, "scene", None), "default_robot_id", "")
        or ""
    )
    if not robot_id:
        raise ValueError(f"Common robot task requires --initial_robot_id or default_robot_id; available={sorted(profiles)}")
    robot_cfg_factory = getattr(importlib.import_module(env_cfg.__class__.__module__), "make_common_robot_cfg", None)
    if callable(robot_cfg_factory):
        env_cfg.scene.robot = robot_cfg_factory(robot_id, prim_path="{ENV_REGEX_NS}/robot")
    else:
        env_cfg.scene.robot.spawn.usd_path = _profile_usd_path(env_cfg, robot_id)
    usd_path = _scene_robot_usd_path(env_cfg)
    setattr(env_cfg, "_allstar_initial_robot_id", robot_id)
    print(f"[CommonRobot] initial_robot_id={robot_id} startup_usd={usd_path}", flush=True)
    return robot_id


def _resolve_curobo_profile(
    robot_usd_path: str | Path | None,
) -> tuple[Path, Path, str, str, list[str], list[float], float | None, list[float]]:
    return _resolve_curobo_profile_from_args(
        args_cli,
        task_root=TASK_ROOT,
        isaaclab_root=ISAACLAB_ROOT,
        robot_usd_path=robot_usd_path,
    )


def _scene_obstacle_config() -> SceneObstacleConfig:
    return scene_obstacle_config_from_args(args_cli, task_root=TASK_ROOT)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# DEBUG: 追踪 22 秒处是谁触发了 shutdown / timeline stop
# DISABLED for now — generates too much output
if False:
    import time as _debug_time
    import traceback as _debug_traceback
    import omni.kit.app as _debug_kit_app
    import omni.timeline as _debug_timeline

    _debug_app = _debug_kit_app.get_app()
    _debug_tl = _debug_timeline.get_timeline_interface()


    def _debug_on_shutdown(event):
        print("\n[DEBUG_SHUTDOWN] Kit shutdown event at", _debug_time.time(), flush=True)
        _debug_traceback.print_stack()


    def _debug_on_timeline(event):
        print(
            "\n[DEBUG_TIMELINE] type=", event.type,
            "is_playing=", _debug_tl.is_playing(),
            "is_stopped=", _debug_tl.is_stopped(),
            "time=", _debug_time.time(),
            flush=True,
        )
        _debug_traceback.print_stack()


    _debug_shutdown_sub = _debug_app.get_shutdown_event_stream().create_subscription_to_pop(
        _debug_on_shutdown, name="debug_shutdown"
    )
    _debug_timeline_sub = _debug_tl.get_timeline_event_stream().create_subscription_to_pop(
        _debug_on_timeline, name="debug_timeline"
    )
    print("[DEBUG] shutdown + timeline listeners installed", flush=True)

import gymnasium as gym
import torch

import isaaclab_tasks.manager_based.manipulation.allstar_toilet  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms
from isaaclab_tasks.manager_based.manipulation.allstar_toilet.utils.curobo_client import (
    DeferredToolCompletion,
    SimulationToolBridge,
    ToolRegistry,
    make_curobo_pose_plan_tool,
    serve_bridge_in_thread,
)
from curobo_pose_plan_utils.curobo_tool_specs import (
    apply_brush_target_forced_safe_clearance as _apply_brush_target_forced_safe_clearance,
    apply_brush_target_safe_clearance as _apply_brush_target_safe_clearance,
    base_keyboard_teleop_spec as _base_keyboard_teleop_spec,
    brush_execute_point_clean_spec as _brush_execute_point_clean_spec,
    brush_execute_spot_clean_spec as _brush_execute_spot_clean_spec,
    default_brush_target_quat_wxyz as _default_brush_target_quat_wxyz,
    downsample_ordered_pose_tests as _downsample_ordered_pose_tests,
    execute_spot_capture_spec as _execute_spot_capture_spec,
    float_list_parameter as _float_list_parameter,
    brush_move_arm_to_joints_spec as _brush_move_arm_to_joints_spec,
    gripper_prismatic_execute_grab_spec as _gripper_prismatic_execute_grab_spec,
    gripper_prismatic_local_lift_spec as _gripper_prismatic_local_lift_spec,
    gripper_prismatic_move_arm_to_joints_spec as _gripper_prismatic_move_arm_to_joints_spec,
    gripper_prismatic_take_out_spec as _gripper_prismatic_take_out_spec,
    stretch_end_tool_spec as _stretch_end_tool_spec,
)
from curobo_pose_plan_utils.curobo_tensor_utils import (
    deg_to_rad_joint_values as _deg_to_rad_joint_values,
    format_tensor as _format_tensor,
    joint_error as _joint_error,
    pose_error as _pose_error,
    pose_to_matrix as _pose_to_matrix,
    validate_joint_value_count as _validate_joint_value_count,
)
from curobo_pose_plan_utils.curobo_scene_obstacles import (
    _collect_curobo_scene_cuboid_obstacles as _collect_curobo_scene_cuboid_obstacles_impl,
    _collect_curobo_scene_mesh_obstacles as _collect_curobo_scene_mesh_obstacles_impl,
    _ensure_raycast_wall_cuboids as _ensure_raycast_wall_cuboids_impl,
    _export_curobo_scene_collision_obj as _export_curobo_scene_collision_obj_impl,
    _runtime_toilet_lid_seat_pose_maps as _runtime_toilet_lid_seat_pose_maps_impl,
)
from curobo_pose_plan_utils.curobo_base_control import (
    _hold_base_stationary,
    _install_static_motion_base_stabilizer,
    _wait_until_static_motion_base_settled as _wait_until_static_motion_base_settled_impl,
)
from curobo_pose_plan_utils.curobo_arm_joints import (
    arm_joint_ids as _arm_joint_ids_impl,
    arm_joint_names as _arm_joint_names_impl,
    build_action as _build_action_impl,
    clamp_arm_joint_target_to_soft_limits as _clamp_arm_joint_target_to_soft_limits_impl,
    find_optional_joints as _find_optional_joints,
    set_arm_effort_target as _set_arm_effort_target_impl,
    set_arm_velocity_target as _set_arm_velocity_target_impl,
    write_arm_target as _write_arm_target_impl,
    zero_arm_velocity_target as _zero_arm_velocity_target_impl,
)
from curobo_pose_plan_utils.curobo_arm_control_config import (
    arm_drive_override_config_from_args,
    arm_hold_effort_bias_tensor as _arm_hold_effort_bias_tensor_impl,
    arm_hold_effort_config_from_args,
)
from curobo_pose_plan_utils.curobo_arm_drive import (
    apply_arm_drive_overrides as _apply_arm_drive_overrides_impl,
    apply_arm_drive_overrides_to_env_cfg as _apply_arm_drive_overrides_to_env_cfg_impl,
    arm_drive_debug_suffix as _arm_drive_debug_suffix_impl,
)
from curobo_pose_plan_utils.curobo_camera_tools import (
    CAM_Q_IN_TOOL as _CAM_Q_IN_TOOL,
    CAM_T_IN_TOOL as _CAM_T_IN_TOOL,
    reset_link5_camera_mount as _reset_link5_camera_mount_impl,
    save_spot_capture_frame as _save_spot_capture_frame_impl,
    show_camera_windows as _show_camera_windows_impl,
    sync_link5_camera as _sync_link5_camera_impl,
)
from curobo_pose_plan_utils.curobo_trajectory_utils import (
    apply_roll_delta_over_waypoint_range as _apply_roll_delta_over_waypoint_range_impl,
    multi_pose_target_waypoint_range as _multi_pose_target_waypoint_range_impl,
    trajectory_waypoint_velocity as _trajectory_waypoint_velocity_impl,
)
from curobo_pose_plan_utils.curobo_trajectory_execution import (
    execute_joint_target as _execute_joint_target_impl,
    execute_joint_target_until_converged as _execute_joint_target_until_converged_impl,
    execute_trajectory as _execute_trajectory_impl,
    execute_trajectory_waypoints_fixed_steps as _execute_trajectory_waypoints_fixed_steps_impl,
    hold_joint_target_if_close as _hold_joint_target_if_close_impl,
    trajectory_execution_config_from_args,
)
from curobo_pose_plan_utils.curobo_gripper_control import (
    execute_gripper_feedback_grasp as _execute_gripper_feedback_grasp_impl,
    gripper_joint_state as _gripper_joint_state_impl,
)
from curobo_pose_plan_utils.curobo_worker_client import (
    run_pose_plan_worker as _run_pose_plan_worker_impl,
    worker_client_config_from_args,
    write_worker_payload_debug as _write_worker_payload_debug_impl,
    shutdown_serve_worker as _shutdown_serve_worker,
)
from curobo_pose_plan_utils.curobo_video_recorder import attach_video_recorder as _attach_video_recorder_impl
from curobo_pose_plan_utils.curobo_planner_client import (
    base_worker_payload as _base_worker_payload_impl,
    plan_cspace_target as _plan_cspace_target_impl,
    plan_multi_pose_polynomial_seed as _plan_multi_pose_polynomial_seed_impl,
    plan_single_pose as _plan_single_pose_impl,
    planner_client_config_from_args,
)
from curobo_pose_plan_utils.curobo_arm_debug import (
    arm_control_debug_suffix as _arm_control_debug_suffix_impl,
    format_optional_tensor as _format_optional_tensor_impl,
)
from curobo_pose_plan_utils.curobo_tool_api_helpers import (
    build_cspace_tool_payload as _build_cspace_tool_payload,
    build_pose_tool_payload as _build_pose_tool_payload,
    execute_point_clean_results as _execute_point_clean_results,
    execute_planning_pose_result_trajectories as _execute_planning_pose_result_trajectories,
    execute_spot_clean_results as _execute_spot_clean_results,
    parse_spot_clean_pose_tests as _parse_spot_clean_pose_tests,
)
from mobile_arm_record_utils.record_mobile_arm_base import (
    _MotionBasedWheelVisualizer,
    _apply_motion_based_base,
    _legacy_operator_to_motion_based_twist,
    _make_upright_root_pose,
    _quat_wxyz_to_yaw_tensor,
)
from mobile_arm_record_utils.record_mobile_arm_input import _create_teleop

from utils.scene_visual_fixes import apply_glass_door_visual_fix


def _reset_link5_camera_mount(env) -> None:
    _reset_link5_camera_mount_impl(env)


def _sync_link5_camera(env, robot) -> tuple[torch.Tensor, torch.Tensor] | None:
    global _G_CAMERA_SYNC_CALLS, _G_CAMERA_SYNC_S
    _t_cam0 = time.perf_counter()
    _r_cam = _sync_link5_camera_impl(env, robot)
    _G_CAMERA_SYNC_CALLS += 1
    _G_CAMERA_SYNC_S += time.perf_counter() - _t_cam0
    return _r_cam


def _show_camera_windows(env) -> None:
    global _G_CAMERA_WINDOW_CALLS, _G_CAMERA_WINDOW_S
    _t_cam0 = time.perf_counter()
    _show_camera_windows_impl(env, headless=args_cli.headless)
    _G_CAMERA_WINDOW_CALLS += 1
    _G_CAMERA_WINDOW_S += time.perf_counter() - _t_cam0


def _save_spot_capture_frame(
    env,
    robot,
    *,
    case_name: str,
    arm_base_body: str,
    target_q: torch.Tensor,
) -> dict:
    return _save_spot_capture_frame_impl(
        env,
        robot,
        case_name=case_name,
        arm_base_body=arm_base_body,
        target_q=target_q,
        task_root=TASK_ROOT,
        sync_link5_camera_fn=_sync_link5_camera,
        show_camera_windows_fn=_show_camera_windows,
        body_pose_w_fn=_body_pose_w,
        pose_to_matrix_fn=_pose_to_matrix,
        arm_joint_ids_fn=_arm_joint_ids,
    )


def _wait_until_static_motion_base_settled(
    env,
    robot,
    stabilizer,
    *,
    min_settle_time_s: float = 1.5,
    max_wait_time_s: float = 8.0,
    linear_vel_threshold: float = 0.03,
    angular_vel_threshold: float = 0.08,
    z_vel_threshold: float = 0.015,
    poll_interval_steps: int = 1,
) -> bool:
    return _wait_until_static_motion_base_settled_impl(
        env,
        robot,
        stabilizer,
        step_fn=_step,
        min_settle_time_s=min_settle_time_s,
        max_wait_time_s=max_wait_time_s,
        linear_vel_threshold=linear_vel_threshold,
        angular_vel_threshold=angular_vel_threshold,
        z_vel_threshold=z_vel_threshold,
        poll_interval_steps=poll_interval_steps,
    )


def _configured_arm_joint_names() -> list[str]:
    return _parse_joint_names(args_cli.joint_names)


def _arm_joint_ids(robot) -> list[int]:
    return _arm_joint_ids_impl(robot, _configured_arm_joint_names())


def _arm_joint_names(robot) -> list[str]:
    return _arm_joint_names_impl(robot, _configured_arm_joint_names())


def _write_arm_target(robot, q_arm: torch.Tensor) -> None:
    _write_arm_target_impl(robot, q_arm, _configured_arm_joint_names())


def _build_action(robot, arm_target_q: torch.Tensor, gripper_joint1_pos: float | None = None) -> torch.Tensor:
    if getattr(robot, "_allstar_direct_tool_control", False):
        _write_arm_target(robot, arm_target_q)
        gripper_target = _default_open_gripper_joint1_pos() if gripper_joint1_pos is None else gripper_joint1_pos
        gripper1_ids, _ = _find_optional_joints(robot, "gripper_joint1")
        gripper2_ids, _ = _find_optional_joints(robot, "gripper_joint2")
        if gripper1_ids:
            target = torch.full(
                (1, len(gripper1_ids)),
                float(gripper_target),
                device=robot.device,
                dtype=robot.data.joint_pos.dtype,
            )
            robot.set_joint_position_target(target, joint_ids=gripper1_ids)
        if gripper2_ids:
            target = torch.full(
                (1, len(gripper2_ids)),
                -float(gripper_target),
                device=robot.device,
                dtype=robot.data.joint_pos.dtype,
            )
            robot.set_joint_position_target(target, joint_ids=gripper2_ids)
        action_robot = getattr(robot, "_allstar_env_action_robot", robot)
        # FIX: pass gripper_joint1_pos through — was always None, causing OPEN override of direct CLOSE
        return _build_action_impl(
            action_robot,
            arm_target_q,
            _configured_arm_joint_names(),
            default_open_gripper_joint1_pos=_default_open_gripper_joint1_pos(),
            gripper_joint1_pos=gripper_joint1_pos,
        )
    return _build_action_impl(
        robot,
        arm_target_q,
        _configured_arm_joint_names(),
        default_open_gripper_joint1_pos=_default_open_gripper_joint1_pos(),
        gripper_joint1_pos=gripper_joint1_pos,
    )


def _zero_arm_velocity_target(robot) -> None:
    _zero_arm_velocity_target_impl(robot, _configured_arm_joint_names())


def _arm_hold_effort_bias_tensor(robot, target_q: torch.Tensor | None = None) -> torch.Tensor:
    return _arm_hold_effort_bias_tensor_impl(
        robot,
        _arm_joint_ids(robot),
        arm_hold_effort_config_from_args(args_cli),
        target_q=target_q,
    )


def _arm_hold_effort_bias_requested() -> bool:
    return arm_hold_effort_config_from_args(args_cli).requested


def _set_arm_effort_target(robot, effort: torch.Tensor | None = None) -> None:
    _set_arm_effort_target_impl(robot, _configured_arm_joint_names(), effort)


def _set_arm_hold_effort_target(robot, target_q: torch.Tensor | None = None) -> torch.Tensor:
    effort = _arm_hold_effort_bias_tensor(robot, target_q=target_q)
    _set_arm_effort_target(robot, effort)
    return effort


def _set_arm_velocity_target(robot, arm_target_vel: torch.Tensor | None) -> None:
    _set_arm_velocity_target_impl(robot, _configured_arm_joint_names(), arm_target_vel)


def _clamp_arm_joint_target_to_soft_limits(robot, target_q: torch.Tensor, *, label: str) -> torch.Tensor:
    return _clamp_arm_joint_target_to_soft_limits_impl(
        robot,
        target_q,
        _configured_arm_joint_names(),
        label=label,
        format_tensor=_format_tensor,
    )


def _multi_pose_target_waypoint_range(plan_result: dict, traj_len: int) -> tuple[int, int]:
    return _multi_pose_target_waypoint_range_impl(plan_result, traj_len)


def _apply_roll_delta_over_waypoint_range(
    robot,
    traj: torch.Tensor,
    *,
    roll_rad: float,
    start_idx: int,
    end_idx: int,
    label: str,
) -> torch.Tensor:
    return _apply_roll_delta_over_waypoint_range_impl(
        robot,
        traj,
        roll_rad=roll_rad,
        start_idx=start_idx,
        end_idx=end_idx,
        label=label,
        clamp_arm_joint_target_to_soft_limits_fn=_clamp_arm_joint_target_to_soft_limits,
        format_tensor_fn=_format_tensor,
    )


def _trajectory_waypoint_velocity(
    env,
    robot,
    traj: torch.Tensor,
    waypoint_idx: int,
    step_budget: int,
) -> torch.Tensor | None:
    return _trajectory_waypoint_velocity_impl(
        env,
        robot,
        traj,
        waypoint_idx,
        step_budget,
        enabled=args_cli.trajectory_velocity_feedforward,
        scale=float(getattr(args_cli, "trajectory_velocity_feedforward_scale", 1.0)),
        arm_joint_ids_fn=_arm_joint_ids,
    )


# 固定相机亚像素抖动 tick：每次 sync 自增，保证相邻渲染的相机变换必然不同
# （render product 只会在相机变换变化时重渲染，位姿重复 → 输出空白帧 → 视频闪烁）。
_FIXED_CAM_SYNC_TICK: list[int] = [0]


def _sync_fixed_camera_pose(env) -> None:
    """固定相机每步把位姿写入 Fabric（set_world_poses），与持续出图的 cam_tool 一致。

    实测结论（2026-09-16，多轮验证）：
      1. 渲染位姿每步被写成"空视角"→ 从第 2 帧起渲染空白（第 1 帧正常）。
      2. cam_tool 能持续出图，因为它位姿每步变化；静态相机位姿不变时，
         render product 的相机变换不产生 change → 渲染器停留在空视角。
    这里给 cam_fixed 加亚毫米级位置抖动（±0.05mm ≈ 0.007px，数据不可见），
    强制每帧产生变换 change，与 cam_tool 行为一致。只调 set_world_poses（Fabric +
    sync_usd_on_fabric_write 同步 USD），不再叠加直写 USD xformOp。
    """
    _cf = env.scene.sensors.get("cam_fixed", None)
    if _cf is None:
        return
    # 抖动必须"相邻渲染必不同"：旧的 ±0.05mm 双值翻转抖动（按 _frame 奇偶）在
    # "同一物理步渲染两次"（env.step 内部渲染 + record_frame 强制渲染）的路径下
    # 会写回与上一次渲染相同的位姿 → render product 的相机变换不产生 change →
    # 跳过重渲染 → 读到空白帧（每 3 帧一张全白）。这里改用单调非重复序列：
    # 每次 sync 自增 tick，dith=(3*tick)%7-3 步进 0.15mm、包络 ±0.15mm，
    # 亚像素、有界不漂移、相邻值必不同 → 每次渲染相机变换都变化 → 每帧重渲染。
    _FIXED_CAM_SYNC_TICK[0] += 1
    _tick = _FIXED_CAM_SYNC_TICK[0]
    _dith = 0.00005 * ((3 * _tick) % 7 - 3)
    _pos_t = torch.tensor([[1.81735 + _dith, -1.24027, 2.19445]], dtype=torch.float32, device=_cf.device)
    _quat_t = torch.tensor([[0.5621, -0.4323, 0.4429, 0.5485]], dtype=torch.float32, device=_cf.device)
    try:
        _cf.set_world_poses(_pos_t, _quat_t, convention="world")
    except Exception as exc:
        print(f"[CamFixed] sync failed: {exc}", flush=True)


# ── HDF5 demo 录制全局状态 ────────────────────────────────────────────────────
# _DEMO_RECORDER：HDF5DemoRecorder 实例（--record_dataset 时在 main() 创建），否则 None。
# _RECORD_ACTIVE：工具 executor 执行期间为 True，只在这期间逐帧采样（跳过 idle 帧）。
_DEMO_RECORDER = None
_RECORD_ACTIVE = False


def _step(env, render: bool = True) -> None:
    robot = getattr(env, "_allstar_active_tool_robot", env.scene["robot"])
    stabilizer = getattr(robot, "_allstar_static_motion_base_stabilizer", None)
    restore_preserved_shared_assets = getattr(env, "_allstar_restore_preserved_shared_assets", None)
    if callable(restore_preserved_shared_assets):
        restore_preserved_shared_assets()
    if stabilizer is not None:
        stabilizer.apply("pre_low_level_step")
    _sync_link5_camera(env, robot)
    _sync_fixed_camera_pose(env)
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    if callable(restore_preserved_shared_assets):
        restore_preserved_shared_assets()
    if stabilizer is not None:
        stabilizer.apply("post_low_level_step")
    _sync_link5_camera(env, robot)
    _sync_fixed_camera_pose(env)
    # GPT P1 v2: sync carried paper visuals after every sim step (continuous follow)
    try:
        _update_carried_paper_visuals(_stage, robot)
    except Exception:
        pass  # visual sync must never crash the main loop
    if render:
        env.sim.render()
        _show_camera_windows(env)


def _install_demo_recording_step(env) -> None:
    """在 env.step 外层包装 HDF5 帧采集。

    覆盖两种模式：
      - direct 模式（common profile 任务）：env.step 已被替换为 direct_env_step → _step()。
      - 普通模式（如默认 gripper 任务）：env.step 为 ManagerBasedEnv.step()。
    用 env.step 包装而非 _step 内钩子，是因为普通模式根本不走自定义 _step()。
    执行期间（_RECORD_ACTIVE=True）每步采样一帧；录制未启用时原样返回。
    """
    if _DEMO_RECORDER is None:
        return
    original_step = env.step

    def recording_step(action):
        result = original_step(action)
        if _RECORD_ACTIVE and _DEMO_RECORDER is not None:
            try:
                # 相机渲染由 record_frame 内部按 stride 帧触发（GUI/headless 都覆盖）
                robot = getattr(env, "_allstar_active_tool_robot", env.scene["robot"])
                _DEMO_RECORDER.record_frame(env, robot, "grasp_test_cube_1")
            except Exception:
                pass  # 录制绝不让仿真崩溃
        return result

    env.step = recording_step


def _execute_joint_target(
    env,
    robot,
    arm_target_q: torch.Tensor,
    steps: int,
    *,
    gripper_joint1_pos: float | None = None,
) -> None:
    _execute_joint_target_impl(
        env,
        robot,
        arm_target_q,
        steps,
        gripper_joint1_pos=gripper_joint1_pos,
        build_action_fn=_build_action,
        sync_link5_camera_fn=_sync_link5_camera,
        zero_arm_velocity_target_fn=_zero_arm_velocity_target,
        set_arm_effort_target_fn=_set_arm_effort_target,
        show_camera_windows_fn=_show_camera_windows,
    )


def _execute_joint_target_until_converged(
    env,
    robot,
    arm_target_q: torch.Tensor,
    *,
    max_steps: int,
    min_steps: int,
    label: str,
    gripper_joint1_pos: float | None = None,
) -> tuple[int, float]:
    return _execute_joint_target_until_converged_impl(
        env,
        robot,
        arm_target_q,
        max_steps=max_steps,
        min_steps=min_steps,
        label=label,
        gripper_joint1_pos=gripper_joint1_pos,
        cfg=trajectory_execution_config_from_args(args_cli),
        build_action_fn=_build_action,
        sync_link5_camera_fn=_sync_link5_camera,
        zero_arm_velocity_target_fn=_zero_arm_velocity_target,
        set_arm_effort_target_fn=_set_arm_effort_target,
        show_camera_windows_fn=_show_camera_windows,
        arm_joint_ids_fn=_arm_joint_ids,
        joint_error_fn=_joint_error,
        format_tensor_fn=_format_tensor,
        arm_control_debug_suffix_fn=_arm_control_debug_suffix,
    )


def _hold_joint_target_if_close(
    env,
    robot,
    target_q: torch.Tensor,
    steps: int,
    *,
    label: str,
    gripper_joint1_pos: float | None = None,
) -> int:
    return _hold_joint_target_if_close_impl(
        env,
        robot,
        target_q,
        steps,
        label=label,
        gripper_joint1_pos=gripper_joint1_pos,
        cfg=trajectory_execution_config_from_args(args_cli),
        arm_joint_ids_fn=_arm_joint_ids,
        joint_error_fn=_joint_error,
        format_tensor_fn=_format_tensor,
        arm_control_debug_suffix_fn=_arm_control_debug_suffix,
        execute_joint_target_fn=_execute_joint_target,
    )


def _gripper_joint1_state(robot) -> tuple[float, float] | None:
    return _gripper_joint_state_impl(robot, "gripper_joint1")


def _gripper_joint_state_by_name(robot, joint_name: str) -> tuple[float, float] | None:
    return _gripper_joint_state_impl(robot, joint_name)


def _execute_gripper_feedback_grasp(env, robot, arm_target_q: torch.Tensor, *, label: str) -> tuple[float, bool]:
    arm_target_q = arm_target_q.detach().clone()  # GPT fix: prevent view mutation during hold
    target_q = _default_grasp_gripper_joint1_pos()
    pos_eps = _PRISMATIC_GRIPPER_GRASP_POS_EPS_M if _is_prismatic_gripper_joint_profile() else _GRIPPER_GRASP_POS_EPS_RAD
    vel_eps = _PRISMATIC_GRIPPER_GRASP_VEL_EPS_M_S if _is_prismatic_gripper_joint_profile() else _GRIPPER_GRASP_VEL_EPS_RAD_S
    max_steps = _default_gripper_grasp_max_steps()
    return _execute_gripper_feedback_grasp_impl(
        env,
        robot,
        arm_target_q,
        label=label,
        target_q=target_q,
        open_target_q=_default_open_gripper_joint1_pos(),
        pos_eps=pos_eps,
        vel_eps=vel_eps,
        max_steps=max_steps,
        build_action_fn=_build_action,
        sync_link5_camera_fn=_sync_link5_camera,
        zero_arm_velocity_target_fn=_zero_arm_velocity_target,
        show_camera_windows_fn=_show_camera_windows,
        arm_control_debug_suffix_fn=_arm_control_debug_suffix,
    )


def _execute_trajectory(
    env,
    robot,
    traj: torch.Tensor,
    *,
    gripper_joint1_pos: float | None = None,
    label: str = "trajectory",
) -> torch.Tensor:
    return _execute_trajectory_impl(
        env,
        robot,
        traj,
        gripper_joint1_pos=gripper_joint1_pos,
        label=label,
        cfg=trajectory_execution_config_from_args(args_cli),
        build_action_fn=_build_action,
        sync_link5_camera_fn=_sync_link5_camera,
        set_arm_velocity_target_fn=_set_arm_velocity_target,
        set_arm_effort_target_fn=_set_arm_effort_target,
        show_camera_windows_fn=_show_camera_windows,
        arm_joint_ids_fn=_arm_joint_ids,
        joint_error_fn=_joint_error,
        trajectory_waypoint_velocity_fn=_trajectory_waypoint_velocity,
        format_tensor_fn=_format_tensor,
        format_optional_tensor_fn=_format_optional_tensor,
        arm_control_debug_suffix_fn=_arm_control_debug_suffix,
    )


def _execute_trajectory_waypoints_fixed_steps(env, robot, traj: torch.Tensor, *, label: str) -> int:
    return _execute_trajectory_waypoints_fixed_steps_impl(
        env,
        robot,
        traj,
        label=label,
        cfg=trajectory_execution_config_from_args(args_cli),
        build_action_fn=_build_action,
        sync_link5_camera_fn=_sync_link5_camera,
        set_arm_velocity_target_fn=_set_arm_velocity_target,
        set_arm_effort_target_fn=_set_arm_effort_target,
        show_camera_windows_fn=_show_camera_windows,
        arm_joint_ids_fn=_arm_joint_ids,
        joint_error_fn=_joint_error,
        trajectory_waypoint_velocity_fn=_trajectory_waypoint_velocity,
        format_tensor_fn=_format_tensor,
        format_optional_tensor_fn=_format_optional_tensor,
        arm_control_debug_suffix_fn=_arm_control_debug_suffix,
    )


def _execute_direct_joint6_roll(
    env,
    robot,
    target_q: torch.Tensor,
    *,
    label: str,
    waypoint_count: int = _DIRECT_JOINT6_ROLL_WAYPOINTS,
) -> tuple[bool, int, float]:
    arm_ids = _arm_joint_ids(robot)
    start_q = robot.data.joint_pos[0, arm_ids].detach().clone()
    target_q = target_q.detach().to(device=robot.device, dtype=robot.data.joint_pos.dtype).reshape(-1).clone()
    if target_q.shape[0] != start_q.shape[0]:
        raise ValueError(f"{label} target_q length {target_q.shape[0]} does not match arm joint count {start_q.shape[0]}.")
    waypoint_count = max(2, int(waypoint_count))
    blend = torch.linspace(
        0.0,
        1.0,
        waypoint_count,
        device=robot.device,
        dtype=robot.data.joint_pos.dtype,
    ).reshape(-1, 1)
    roll_traj = start_q.reshape(1, -1) + blend * (target_q.reshape(1, -1) - start_q.reshape(1, -1))
    print(
        f"[DirectJoint6Roll] label={label} start_q6={float(start_q[-1].item()):.6f} "
        f"target_q6={float(target_q[-1].item()):.6f} waypoints={waypoint_count} "
        "planner=direct_no_curobo",
        flush=True,
    )
    _execute_trajectory(env, robot, roll_traj, label=label)
    actual_q = robot.data.joint_pos[0, arm_ids].detach()
    _, err_max, err = _joint_error(target_q, actual_q)
    executed = err_max.item() <= float(args_cli.joint_target_tolerance)
    print(
        f"[DirectJoint6Roll] label={label} executed={executed} "
        f"joint_err_max={err_max.item():.6f} joint_err={_format_tensor(err)}",
        flush=True,
    )
    env_step_count = waypoint_count * max(1, int(args_cli.steps_per_waypoint))
    return bool(executed), int(env_step_count), float(err_max.item())


def _execute_brush_execute_point_clean_trajectory(
    env,
    robot,
    traj: torch.Tensor,
    return_q: torch.Tensor,
    *,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    scene_mesh_obstacles: list[dict],
) -> dict:
    del return_q
    if traj.dim() > 2:
        traj = traj.reshape(-1, traj.shape[-1])
    if traj.dim() != 2 or traj.shape[0] < 1:
        raise ValueError("brush_execute_point_clean requires a non-empty 2D trajectory.")

    env_step_count = 0
    if traj.shape[0] > 1:
        env_step_count += _execute_trajectory_waypoints_fixed_steps(
            env,
            robot,
            traj[:-1],
            label="brush_execute_point_clean_pre_final",
        )

    final_q = traj[-1].detach().clone()
    final_roll_q = final_q.clone()
    final_roll_q[5] = final_roll_q[5] + math.pi
    print(
        f"[BrushRolling0512] final_waypoint_count={traj.shape[0]} "
        f"final_q6={float(final_q[5].item()):.6f} roll_q6={float(final_roll_q[5].item()):.6f}",
        flush=True,
    )
    roll_executed, roll_env_steps, roll_err_after = _execute_direct_joint6_roll(
        env=env,
        robot=robot,
        target_q=final_roll_q,
        label="brush_execute_point_clean:final_joint6_roll",
    )
    env_step_count += roll_env_steps
    return {
        "waypoint_count": int(traj.shape[0]),
        "env_step_count": env_step_count,
        "joint6_roll_executed": roll_executed,
        "returned_to_pre_target": False,
        "roll_joint_err_max": float(roll_err_after),
        "rollback_joint_err_max": None,
    }


def _tool_tip_pose_base(robot, base_link: str) -> tuple[torch.Tensor, torch.Tensor]:
    arm_base_pos_w, arm_base_quat_w = _body_pose_w(robot, base_link.replace("-", "_"))
    tool_pos_w, tool_quat_w = _tool_tip_pose_w(robot)
    return subtract_frame_transforms(
        arm_base_pos_w.reshape(1, 3),
        arm_base_quat_w.reshape(1, 4),
        tool_pos_w.reshape(1, 3),
        tool_quat_w.reshape(1, 4),
    )


def _execute_brush_execute_spot_clean_trajectory(
    env,
    robot,
    traj: torch.Tensor,
    *,
    plan_result: dict,
    urdf_path: str,
    robot_cfg_path: Path,
    tool_frame: str,
    joint_names: list[str],
    scene_mesh_obstacles: list[dict],
    base_link: str,
    final_target_base: dict,
    final_pos_tolerance_m: float,
    final_rot_tolerance_rad: float | None = None,
    joint6_roll_rad: float = math.pi,
) -> dict:
    if traj.dim() > 2:
        traj = traj.reshape(-1, traj.shape[-1])
    if traj.dim() != 2 or traj.shape[0] < 1:
        raise ValueError("brush_execute_spot_clean requires a non-empty 2D trajectory.")

    planned_traj = traj.detach().clone()
    exec_traj = planned_traj.clone()
    roll_start_idx, roll_end_idx = _multi_pose_target_waypoint_range(plan_result, int(exec_traj.shape[0]))
    roll_env_steps = 0
    if roll_start_idx < roll_end_idx:
        exec_traj = _apply_roll_delta_over_waypoint_range(
            robot,
            exec_traj,
            roll_rad=joint6_roll_rad,
            start_idx=roll_start_idx,
            end_idx=roll_end_idx,
            label="brush_execute_spot_clean:joint6_roll_over_path",
        )
        roll_target_q = exec_traj[-1].detach().clone()
        applied_roll_delta_rad = float((exec_traj[-1, -1] - planned_traj[-1, -1]).item())
        print(
            f"[BrushSpotClean] waypoints={exec_traj.shape[0]} "
            f"joint6_roll_start_waypoint={roll_start_idx} joint6_roll_end_waypoint={roll_end_idx} "
            f"roll_target_q={_format_tensor(roll_target_q)} "
            f"requested_roll_delta={joint6_roll_rad:.6f} applied_roll_delta={applied_roll_delta_rad:.6f}",
            flush=True,
        )
        _execute_trajectory(env, robot, exec_traj, label="brush_execute_spot_clean:main")
        actual_q = robot.data.joint_pos[0, _arm_joint_ids(robot)].detach()
        _, final_err_tensor, final_joint_err = _joint_error(roll_target_q, actual_q)
        final_err = float(final_err_tensor.item())
        roll_executed = final_err <= float(args_cli.joint_target_tolerance)
    else:
        roll_target_q = planned_traj[-1].detach().clone()
        roll_target_q[-1] = roll_target_q[-1] + float(joint6_roll_rad)
        applied_roll_delta_rad = float((roll_target_q[-1] - planned_traj[-1, -1]).item())
        print(
            f"[BrushSpotClean] waypoints={exec_traj.shape[0]} "
            f"joint6_roll_start_waypoint={roll_start_idx} joint6_roll_end_waypoint={roll_end_idx} "
            "roll_range_degenerate=true; falling back to final direct joint6 roll "
            f"roll_target_q={_format_tensor(roll_target_q)} "
            f"requested_roll_delta={joint6_roll_rad:.6f} applied_roll_delta={applied_roll_delta_rad:.6f}",
            flush=True,
        )
        _execute_trajectory(env, robot, exec_traj, label="brush_execute_spot_clean:main")
        roll_executed, roll_env_steps, final_err = _execute_direct_joint6_roll(
            env=env,
            robot=robot,
            target_q=roll_target_q,
            label="brush_execute_spot_clean:final_joint6_roll_degenerate",
        )
        actual_q = robot.data.joint_pos[0, _arm_joint_ids(robot)].detach()
        _, _, final_joint_err = _joint_error(roll_target_q, actual_q)
    target_pos_base = torch.tensor(
        final_target_base["target_pos"],
        device=robot.device,
        dtype=robot.data.joint_pos.dtype,
    ).reshape(1, 3)
    target_quat_base = torch.tensor(
        final_target_base["target_quat"],
        device=robot.device,
        dtype=robot.data.joint_pos.dtype,
    ).reshape(1, 4)
    pose_steps = 0
    final_pos_err = torch.tensor(float("inf"), device=robot.device, dtype=robot.data.joint_pos.dtype)
    final_rot_err_rad = torch.tensor(float("inf"), device=robot.device, dtype=robot.data.joint_pos.dtype)
    final_rot_err_deg = torch.tensor(float("inf"), device=robot.device, dtype=robot.data.joint_pos.dtype)
    actual_pos_base, actual_quat_base = _tool_tip_pose_base(robot, base_link)
    final_pos_err, final_rot_err_rad, final_rot_err_deg = _pose_error(
        target_pos_base,
        target_quat_base,
        actual_pos_base,
        actual_quat_base,
    )
    rot_ok = final_rot_tolerance_rad is None or final_rot_err_rad.item() <= final_rot_tolerance_rad
    pose_converged = final_pos_err.item() <= final_pos_tolerance_m and rot_ok
    print(
        f"[BrushSpotCleanFinalPose] converged={pose_converged} "
        f"pos_err_m={float(final_pos_err.item()):.6f} "
        f"pos_tol_m={float(final_pos_tolerance_m):.6f} "
        f"rot_err_rad={float(final_rot_err_rad.item()):.6f} "
        f"rot_err_deg={float(final_rot_err_deg.item()):.3f} "
        f"rot_tol_rad={final_rot_tolerance_rad} pose_steps={pose_steps} "
        f"joint6_roll_executed={roll_executed} "
        f"joint_err_max={final_err:.6f} joint_err={_format_tensor(final_joint_err)}",
        flush=True,
    )
    return {
        "waypoint_count": int(exec_traj.shape[0]),
        "env_step_count": (
            int(exec_traj.shape[0]) * max(1, args_cli.steps_per_waypoint)
            + int(roll_env_steps)
            + int(pose_steps)
        ),
        "joint6_roll_executed": roll_executed,
        "returned_to_pre_target": False,
        "final_joint_err_max": final_err,
        "joint6_roll_start_waypoint": int(roll_start_idx),
        "joint6_roll_end_waypoint": int(roll_end_idx),
        "joint6_roll_delta_rad": applied_roll_delta_rad,
        "final_pose_converged": pose_converged,
        "final_pos_err_m": float(final_pos_err.item()),
        "final_rot_err_rad": float(final_rot_err_rad.item()),
        "final_rot_err_deg": float(final_rot_err_deg.item()),
        "final_pose_steps": pose_steps,
    }


def _tool_tip_pose_w(robot) -> tuple[torch.Tensor, torch.Tensor]:
    body_idx = robot.data.body_names.index(args_cli.tool_frame.replace("-", "_"))
    pos = robot.data.body_pos_w[0, body_idx].detach().clone()
    quat = robot.data.body_quat_w[0, body_idx].detach().clone()
    return pos, quat


def _tool_tip_forward_axis_w(robot) -> torch.Tensor:
    _, tool_quat_w = _tool_tip_pose_w(robot)
    local_forward = torch.tensor([0.0, 0.0, 1.0], device=robot.device, dtype=robot.data.joint_pos.dtype)
    return quat_apply(tool_quat_w.reshape(1, 4), local_forward.reshape(1, 3)).reshape(3)


def _body_pose_w(robot, body_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    body_idx = robot.data.body_names.index(body_name)
    pos = robot.data.body_pos_w[0, body_idx].detach().clone()
    quat = robot.data.body_quat_w[0, body_idx].detach().clone()
    return pos, quat


def _runtime_toilet_lid_seat_pose_maps(env) -> dict[int, dict[str, object]]:
    return _runtime_toilet_lid_seat_pose_maps_impl(env)


def _collect_curobo_scene_mesh_obstacles(
    stage,
    *,
    isaac_base_body: str,
    env=None,
    base_pose_w: tuple[torch.Tensor, torch.Tensor] | None = None,
    target_positions_w: list[list[float]] | None = None,
) -> list[dict[str, str]]:
    return _collect_curobo_scene_mesh_obstacles_impl(
        stage,
        isaac_base_body=isaac_base_body,
        cfg=_scene_obstacle_config(),
        gripper_profile=_is_gripper_joint_profile(),
        env=env,
        base_pose_w=base_pose_w,
        target_positions_w=target_positions_w,
    )


def _collect_curobo_scene_cuboid_obstacles(
    stage,
    *,
    base_pose_w: tuple[torch.Tensor, torch.Tensor],
) -> list[dict]:
    return _collect_curobo_scene_cuboid_obstacles_impl(
        stage,
        cfg=_scene_obstacle_config(),
        base_pose_w=base_pose_w,
    )


def _export_curobo_scene_collision_obj(
    *,
    scene_mesh_obstacles: list[dict],
    scene_cuboid_obstacles: list[dict],
) -> str | None:
    return _export_curobo_scene_collision_obj_impl(
        cfg=_scene_obstacle_config(),
        scene_mesh_obstacles=scene_mesh_obstacles,
        scene_cuboid_obstacles=scene_cuboid_obstacles,
    )


def _sync_paper_waste_proxies(
    stage,
    base_pose_w: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Update cuRobo proxy Cube transforms to track paper rigid-body world poses.

    Called before each tool invocation so that cuRobo always sees the current
    paper-waste positions.  Collected papers are skipped.
    """

    # Map: (paper_rigid_body_prim_path, proxy_cube_prim_path)
    # Proxy is a child of the paper prim: {paper_path}/curobo_proxy
    paper_proxy_pairs: list[tuple[str, str]] = [
        ("/World/envs/env_0/paper_waste/used_wet_paper_clump_01", "/World/envs/env_0/paper_waste/used_wet_paper_clump_01/curobo_proxy"),
        ("/World/envs/env_0/paper_waste/used_wet_paper_clump_02", "/World/envs/env_0/paper_waste/used_wet_paper_clump_02/curobo_proxy"),
        ("/World/envs/env_0/paper_waste/used_wet_paper_clump_03", "/World/envs/env_0/paper_waste/used_wet_paper_clump_03/curobo_proxy"),
        ("/World/envs/env_0/paper_waste/used_flattened_tissue_01", "/World/envs/env_0/paper_waste/used_flattened_tissue_01/curobo_proxy"),
        ("/World/envs/env_0/paper_waste/used_flattened_tissue_02", "/World/envs/env_0/paper_waste/used_flattened_tissue_02/curobo_proxy"),
        ("/World/envs/env_0/paper_waste/used_flattened_tissue_03", "/World/envs/env_0/paper_waste/used_flattened_tissue_03/curobo_proxy"),
    ]

    updated = 0
    for paper_path, proxy_path in paper_proxy_pairs:
        paper_prim = stage.GetPrimAtPath(paper_path)
        proxy_prim = stage.GetPrimAtPath(proxy_path)
        if not paper_prim.IsValid() or not proxy_prim.IsValid():
            continue

        # Skip if already collected
        pw_attr = proxy_prim.GetAttribute("paper_waste:collected")
        if pw_attr and pw_attr.IsValid() and pw_attr.Get() is True:
            continue

        # Proxy is a child of the paper prim — world transform auto-inherits.
        # Nothing to update.
        updated += 1

    if updated:
        global _CUROBO_SCENE_CUBOID_OBSTACLES
        _CUROBO_SCENE_CUBOID_OBSTACLES = _collect_curobo_scene_cuboid_obstacles(
            stage, base_pose_w=base_pose_w,
        )
        print(
            f"[PaperWasteSync] {updated} proxy(s) synced, "
            f"cuboid_obstacles={len(_CUROBO_SCENE_CUBOID_OBSTACLES)}",
            flush=True,
        )


def _update_attached_paper_poses(stage, robot) -> None:
    """Write attached paper-waste prim world poses to follow the gripper body.

    Called on each idle simulation step while _PAPER_ATTACH_STATE is non-empty.
    """
    from pxr import Gf, UsdGeom

    for paper_path, (gripper_body, offset_w) in list(_PAPER_ATTACH_STATE.items()):
        paper_prim = stage.GetPrimAtPath(paper_path)
        if not paper_prim.IsValid():
            del _PAPER_ATTACH_STATE[paper_path]
            continue

        # Read gripper body world pose from Isaac
        try:
            body_idx = robot.data.body_names.index(gripper_body)
        except ValueError:
            continue
        gripper_pos = robot.data.body_pos_w[0, body_idx].detach().cpu().tolist()

        # Paper world pose = gripper world pos + offset
        paper_pos_w = (
            gripper_pos[0] + offset_w[0],
            gripper_pos[1] + offset_w[1],
            gripper_pos[2] + offset_w[2],
        )

        # Update USD prim transform
        xf = UsdGeom.Xformable(paper_prim)
        ops = xf.GetOrderedXformOps()
        if ops:
            ops[0].Set(Gf.Vec3d(*paper_pos_w))


def _prim_world_pos(stage, prim_path: str) -> tuple[float, float, float] | None:
    """World translation of a USD prim (or None if missing).

    注意：对 PhysX 动态刚体，USD authored 变换（XformCache.Default）不随仿真更新，
    读到的永远是初始位姿。仿真中的真实位姿请用 _object_world_pos（读 scene 物理数据）。
    """
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
    return (float(t[0]), float(t[1]), float(t[2]))


def _pos_in_trash_bin_aabb(env, pos_w) -> bool:
    """方块是否真的落在垃圾桶里。

    用垃圾桶的**实时位姿**构造包围盒（原来写死 (2.097,-1.095)，桶被撞走后照样判 true）。
    另外要求桶基本立着：桶的 up 轴与世界 +Z 夹角 < 30°，桶倒了直接判失败。
    """
    if pos_w is None:
        return False
    import math

    try:
        bin_asset = env.scene["trash_bin"]
        pose = bin_asset.data.root_pose_w[0]
        bin_pos = [float(pose[0]), float(pose[1]), float(pose[2])]
        qx, qy = float(pose[4]), float(pose[5])
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)  # R·(0,0,1) 的 z 分量
    except Exception as exc:
        print(f"[TrashBinCheck] 读不到垃圾桶位姿，退回标称位置: {exc}", flush=True)
        bin_pos = [2.097, -1.095, 0.010]
        up_z = 1.0

    if up_z < math.cos(math.radians(30.0)):
        print(f"[TrashBinCheck] 垃圾桶倒了(up_z={up_z:.3f}) → 判定为未入桶", flush=True)
        return False

    aabb = (
        (bin_pos[0] - 0.12, bin_pos[1] - 0.12, bin_pos[2] - 0.06),
        (bin_pos[0] + 0.12, bin_pos[1] + 0.12, bin_pos[2] + 0.40),
    )
    inside = all(aabb[0][i] <= pos_w[i] <= aabb[1][i] for i in range(3))
    print(
        f"[TrashBinCheck] bin={[round(v, 3) for v in bin_pos]} up_z={up_z:.3f} "
        f"cube={[round(v, 3) for v in pos_w]} inside={inside}",
        flush=True,
    )
    return inside


def _object_world_pos(env, prim_path: str, *, default_z_from_prim: bool = True) -> tuple[float, float, float] | None:
    """World position of a dynamic rigid object from Isaac Lab physics data.

    优先读 env.scene[asset].data.root_pose_w（PhysX 实时位姿）；场景中找不到该 asset
    或数据不可用时回退到 USD prim 变换（_prim_world_pos）。
    """
    asset_name = prim_path.rsplit("/", 1)[-1]
    try:
        asset = env.scene[asset_name]
    except Exception:
        asset = None
    if asset is not None:
        root_pose = getattr(asset.data, "root_pose_w", None)
        if root_pose is None:
            root_pose = getattr(asset.data, "root_link_pose_w", None)
        if root_pose is not None:
            xyz = root_pose[0, :3].detach().cpu().tolist()
            return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    if default_z_from_prim:
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            return _prim_world_pos(stage, prim_path)
        except Exception:
            return None
    return None


def _check_paper_grasp(
    stage, robot, paper_prim_path: str, *, grasp_threshold_m: float = 0.12,
) -> tuple[bool, tuple[float, float, float] | None]:
    """Check if the paper is close enough to the gripper to count as grasped.

    Returns (grasped: bool, offset_w: tuple | None).
    offset_w = paper_pos_w - gripper_tool_tip_pos_w
    """
    from pxr import Gf, Usd, UsdGeom

    paper_prim = stage.GetPrimAtPath(paper_prim_path)
    if not paper_prim.IsValid():
        return False, None

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_mat = cache.GetLocalToWorldTransform(paper_prim)
    paper_pos = world_mat.ExtractTranslation()

    tool_pos_w = _tool_tip_pose_w(robot)[0].detach().cpu().tolist()

    import math
    dist = math.sqrt(
        sum((float(paper_pos[i]) - tool_pos_w[i]) ** 2 for i in range(3))
    )

    print(
        f"[PaperGraspCheck] paper=({paper_pos[0]:.3f},{paper_pos[1]:.3f},{paper_pos[2]:.3f}) "
        f"tool=({tool_pos_w[0]:.3f},{tool_pos_w[1]:.3f},{tool_pos_w[2]:.3f}) "
        f"dist={dist:.4f} threshold={grasp_threshold_m:.4f}",
        flush=True,
    )

    if dist < grasp_threshold_m:
        offset_w = (
            float(paper_pos[0]) - tool_pos_w[0],
            float(paper_pos[1]) - tool_pos_w[1],
            float(paper_pos[2]) - tool_pos_w[2],
        )
        return True, offset_w

    return False, None


def _set_paper_prim_state(stage, prim_path: str, state: str, collected: bool = False) -> None:
    """Write paper_waste:* custom attributes on a USD prim."""
    from pxr import Sdf

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    for attr_name, type_name, value in [
        ("paper_waste:state", Sdf.ValueTypeNames.String, state),
        ("paper_waste:is_collected", Sdf.ValueTypeNames.Bool, collected),
    ]:
        attr = prim.GetAttribute(attr_name)
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(attr_name, type_name)
        attr.Set(value)


def _set_paper_kinematic(stage, prim_path: str, kinematic: bool) -> None:
    """Toggle kinematic mode on a paper-waste rigid body.

    When kinematic, PhysX does not apply forces — the prim is moved by the user.
    """
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return
    rigid_api = UsdPhysics.RigidBodyAPI(prim)
    attr = rigid_api.GetKinematicEnabledAttr()
    if not attr or not attr.IsValid():
        attr = rigid_api.CreateKinematicEnabledAttr()
    attr.Set(kinematic)
    print(
        f"[PaperKinematic] {prim_path} kinematic={kinematic}",
        flush=True,
    )


# ============================================================
# GPT P1 v2: Safe visual-only paper following
# Carried visual at /World/envs/env_0/runtime_visuals/ — NOT under robot articulation
# ============================================================

_CARRIED_VISUAL_ROOT = "/World/envs/env_0/runtime_visuals"


def _create_carried_visual_safe(
    stage,
    carried_path: str,
    source_prim_path: str = "",
    local_offset: tuple[float, float, float] = (0.0, 0.0, 0.03),
    radius: float = 0.035,
    initial_world_pos: tuple[float, float, float] | None = None,
) -> None:
    """Create visual-only paper proxy under runtime_visuals (NOT under robot articulation).

    If *initial_world_pos* is given, the visual starts there (smooth transition);
    otherwise it defaults to *local_offset* above origin.
    """
    from pxr import Gf, UsdGeom, UsdShade, Sdf

    # Detect paper kind for appropriate visual shape
    _kind = "used_wet_clump"
    if source_prim_path:
        _src = stage.GetPrimAtPath(source_prim_path)
        if _src.IsValid():
            _ka = _src.GetAttribute("paper_waste:kind")
            if _ka and _ka.IsValid():
                _kind = str(_ka.Get() or "used_wet_clump")

    _is_tissue = _kind in ("used_flattened_tissue", "scanned_dish_towel", "scanned_bath_towel")

    # Initial position: paper world pos if given, else default offset
    _start_pos = Gf.Vec3d(*initial_world_pos) if initial_world_pos else Gf.Vec3d(*local_offset)

    xform = UsdGeom.Xform.Define(stage, carried_path)
    xform.AddTranslateOp().Set(_start_pos)
    xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    if _is_tissue:
        # Thin box for tissue/towel
        box = UsdGeom.Cube.Define(stage, carried_path + "/carried_box")
        box.CreateSizeAttr(1.0)
        bx = UsdGeom.Xformable(box.GetPrim())
        bx.AddScaleOp().Set(Gf.Vec3f(0.06, 0.04, 0.004))
        _color = Gf.Vec3f(0.72, 0.67, 0.58)
    else:
        # Sphere lumps for clumps
        offsets = [(0.0, 0.0, 0.0), (0.012, 0.006, 0.004), (-0.010, -0.004, 0.006)]
        for i, off in enumerate(offsets):
            sphere_path = f"{carried_path}/lump_{i:02d}"
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.CreateRadiusAttr(radius * (0.75 + 0.15 * i))
            sx = UsdGeom.Xformable(sphere.GetPrim())
            sx.AddTranslateOp().Set(Gf.Vec3d(*off))
        _color = Gf.Vec3f(0.65, 0.62, 0.55)

    mat_path = f"{carried_path}/carried_material"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/preview_surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(_color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.88)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(xform.GetPrim()).Bind(mat)

    print(
        f"[PaperVisual] created {carried_path} kind={_kind} is_tissue={_is_tissue} "
        f"offset={local_offset}",
        flush=True,
    )


# ============================================================
# GPT P1 v2: unified env.step wrapper — runs runtime-visual
# updates after every simulation step, not just in the idle loop.
# ============================================================

def _step_env_with_runtime_hooks(
    env,
    action=None,
    *,
    render: bool = True,
    robot=None,
    stage=None,
) -> tuple:
    """Call env.step(action) then immediately sync carried paper visuals.

    This ensures carried visuals follow the gripper continuously during
    trajectory execution, descends, lifts, returns, and hold loops — not
    only when the tool server is idle.
    """
    if action is not None:
        obs, rew, terminated, truncated, info = env.step(action)
    else:
        obs, rew, terminated, truncated, info = env.step()

    # Sync runtime visuals after every sim step
    try:
        _s = stage or _stage
        _r = robot
        if _r is not None:
            _update_carried_paper_visuals(_s, _r)
    except Exception:
        pass  # visual sync must never crash the main loop

    return obs, rew, terminated, truncated, info


def _update_carried_paper_visuals(stage, robot) -> None:
    """Per-step: sync carried visuals smoothly toward tool_tip world pose."""
    if not _CARRIED_VISUAL_STATE:
        return
    try:
        tool_body_idx = robot.data.body_names.index("tool_tip_link")
    except ValueError:
        return
    tool_pos_w = robot.data.body_pos_w[0, tool_body_idx].detach().cpu().tolist()
    tool_quat_w = robot.data.body_quat_w[0, tool_body_idx].detach().cpu().tolist()  # w,x,y,z

    from pxr import Gf, UsdGeom

    # Smoothing factor: 0=instant snap, 1=no movement; 0.15 → smooth transition
    _LERP = 0.15

    for paper_id, state in list(_CARRIED_VISUAL_STATE.items()):
        prim = stage.GetPrimAtPath(state["carried_path"])
        if not prim.IsValid():
            _CARRIED_VISUAL_STATE.pop(paper_id, None)
            continue

        # Target pose: tool pose + local offset (rotated into world)
        offset = state.get("local_offset", (0.0, 0.0, 0.03))
        tool_rot = Gf.Rotation(Gf.Quatd(tool_quat_w[1], tool_quat_w[2], tool_quat_w[3], tool_quat_w[0]))
        offset_w = tool_rot.TransformDir(Gf.Vec3d(*offset))
        target_pos = Gf.Vec3d(tool_pos_w[0] + offset_w[0], tool_pos_w[1] + offset_w[1], tool_pos_w[2] + offset_w[2])
        target_quat = Gf.Quatf(tool_quat_w[1], tool_quat_w[2], tool_quat_w[3], tool_quat_w[0])

        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        if len(ops) < 2:
            continue

        # Current position (read from USD)
        current_pos = ops[0].Get()

        # Smooth lerp toward target
        new_pos = Gf.Vec3d(
            current_pos[0] + (target_pos[0] - current_pos[0]) * _LERP,
            current_pos[1] + (target_pos[1] - current_pos[1]) * _LERP,
            current_pos[2] + (target_pos[2] - current_pos[2]) * _LERP,
        )
        ops[0].Set(new_pos)
        ops[1].Set(target_quat)


def _logical_attach_paper_safe(stage, paper_prim_path: str) -> dict:
    """GPT P1 v2: mark source paper as grasped, create carried visual at safe location.

    Does NOT deactivate the source prim (GPU PhysX crash when SetActive(False)
    on a dynamic body mid-simulation).
    """
    paper_prim = stage.GetPrimAtPath(paper_prim_path)
    if not paper_prim.IsValid():
        raise RuntimeError(f"Invalid paper prim for attach: {paper_prim_path}")

    from pxr import Sdf
    for attr_name, type_name, value in [
        ("paper_waste:state", Sdf.ValueTypeNames.String, "grasped"),
        ("paper_waste:is_carried", Sdf.ValueTypeNames.Bool, True),
    ]:
        attr = paper_prim.GetAttribute(attr_name)
        if not attr or not attr.IsValid():
            attr = paper_prim.CreateAttribute(attr_name, type_name)
        attr.Set(value)

    # NOTE: Do NOT call paper_prim.SetActive(False) — crashes GPU PhysX.
    # Instead, hide only the visual_mesh child prim so the original paper
    # disappears while the carried visual follows the gripper.
    _vis = stage.GetPrimAtPath(paper_prim_path + "/visual_mesh")
    if _vis.IsValid():
        from pxr import UsdGeom as _UsdGeom
        _img = _UsdGeom.Imageable(_vis)
        _img.MakeInvisible()
        print(f"[PaperVisual] hid source visual: {paper_prim_path}/visual_mesh", flush=True)

    # cuRobo proxy skip is handled via paper_waste:is_carried attribute.

    paper_id = paper_prim_path.rsplit("/", 1)[-1]
    carried_path = f"{_CARRIED_VISUAL_ROOT}/carried_paper_visual_{paper_id}"

    # Read source paper world position for smooth transition start point
    from pxr import UsdGeom as _Ug, Usd as _Usd
    _cache = _Ug.XformCache(_Usd.TimeCode.Default())
    _paper_world_mat = _cache.GetLocalToWorldTransform(paper_prim)
    _paper_world_pos = _paper_world_mat.ExtractTranslation()
    _paper_world_pos_tuple = (float(_paper_world_pos[0]), float(_paper_world_pos[1]), float(_paper_world_pos[2]))
    print(f"[PaperVisual] source paper world pos=({_paper_world_pos_tuple[0]:.3f},{_paper_world_pos_tuple[1]:.3f},{_paper_world_pos_tuple[2]:.3f})", flush=True)

    _create_carried_visual_safe(
        stage, carried_path,
        source_prim_path=paper_prim_path,
        local_offset=(0.0, 0.0, 0.03),
        initial_world_pos=_paper_world_pos_tuple,  # start at paper's position
    )

    state = {
        "paper_id": paper_id,
        "source_prim_path": paper_prim_path,
        "carried_path": carried_path,
        "local_offset": (0.0, 0.0, 0.03),
        "state": "grasped",
    }
    global _CARRIED_VISUAL_STATE
    _CARRIED_VISUAL_STATE[paper_id] = state
    print(f"[PaperVisual] attach complete: {paper_prim_path} → {carried_path}", flush=True)
    return state


def _logical_detach_paper_safe(stage, paper_id: str) -> bool:
    """Remove carried visual. Return True if was carried."""
    global _CARRIED_VISUAL_STATE
    state = _CARRIED_VISUAL_STATE.pop(paper_id, None)
    if state is None:
        return False
    carried_path = state["carried_path"]
    if stage.GetPrimAtPath(carried_path).IsValid():
        stage.RemovePrim(carried_path)
    # Mark source prim collected
    src_path = state.get("source_prim_path", "")
    if src_path:
        _set_paper_prim_state(stage, src_path, "collected", True)
        # Restore source visual visibility
        _vis = stage.GetPrimAtPath(src_path + "/visual_mesh")
        if _vis.IsValid():
            from pxr import UsdGeom as _UsdGeom
            _img = _UsdGeom.Imageable(_vis)
            _img.MakeVisible()
        proxy_path = f"{src_path}/curobo_proxy"
        proxy_prim = stage.GetPrimAtPath(proxy_path)
        if proxy_prim.IsValid():
            from pxr import Sdf as _Sdf
            attr = proxy_prim.GetAttribute("paper_waste:collected")
            if not attr or not attr.IsValid():
                attr = proxy_prim.CreateAttribute("paper_waste:collected", _Sdf.ValueTypeNames.Bool)
            attr.Set(True)
    print(f"[PaperVisual] detach complete: {carried_path} collected=True", flush=True)
    return True


def _detach_paper(stage, paper_prim_path: str) -> None:
    """Detach paper from gripper, restore dynamic physics, and update state."""
    global _PAPER_ATTACH_STATE
    _PAPER_ATTACH_STATE.pop(paper_prim_path, None)

    # Restore dynamic physics so the paper falls naturally
    _set_paper_prim_state(stage, paper_prim_path, "loose", False)

    # Also update proxy prim's collected flag (proxy is a child: {paper_path}/curobo_proxy)
    proxy_path = f"{paper_prim_path}/curobo_proxy"
    proxy_prim = stage.GetPrimAtPath(proxy_path)
    if proxy_prim.IsValid():
        from pxr import Sdf
        attr = proxy_prim.GetAttribute("paper_waste:collected")
        if not attr or not attr.IsValid():
            attr = proxy_prim.CreateAttribute("paper_waste:collected", Sdf.ValueTypeNames.Bool)
        attr.Set(False)

    print(f"[PaperCleanup] detached paper={paper_prim_path}", flush=True)


def _check_paper_collected_at_prim(stage, paper_prim_path: str) -> bool:
    """Check if paper is inside the trash bin collection AABB and mark as collected."""
    from pxr import Gf, Usd, UsdGeom

    paper_prim = stage.GetPrimAtPath(paper_prim_path)
    if not paper_prim.IsValid():
        return False

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_mat = cache.GetLocalToWorldTransform(paper_prim)
    t = world_mat.ExtractTranslation()
    paper_pos_w = (float(t[0]), float(t[1]), float(t[2]))

    # Inline AABB check — trash bin collection zone
    # Trash bin at (2.097, -1.095, 0.01), half-extents ~0.12m XY, 0.40m Z
    TRASH_BIN_AABB = (
        (2.097 - 0.12, -1.095 - 0.12, -0.05),  # min
        (2.097 + 0.12, -1.095 + 0.12, 0.01 + 0.40),   # max
    )
    collected = all(
        TRASH_BIN_AABB[0][i] <= paper_pos_w[i] <= TRASH_BIN_AABB[1][i]
        for i in range(3)
    )
    print(
        f"[PaperCleanup] collection_check pos=({paper_pos_w[0]:.3f},{paper_pos_w[1]:.3f},{paper_pos_w[2]:.3f}) "
        f"collected={collected}",
        flush=True,
    )

    if collected:
        _set_paper_prim_state(stage, paper_prim_path, "collected", True)
        # Update proxy prim
        proxy_path = f"{paper_prim_path}/curobo_proxy"
        proxy_prim = stage.GetPrimAtPath(proxy_path)
        if proxy_prim.IsValid():
            from pxr import Sdf
            attr = proxy_prim.GetAttribute("paper_waste:collected")
            if not attr or not attr.IsValid():
                attr = proxy_prim.CreateAttribute("paper_waste:collected", Sdf.ValueTypeNames.Bool)
            attr.Set(True)
        print(f"[PaperCleanup] PAPER COLLECTED! {paper_prim_path}", flush=True)

    return collected


def _base_worker_payload(
    *,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    q_start: torch.Tensor,
    scene_mesh_obstacles: list[dict],
) -> dict:
    return _base_worker_payload_impl(
        urdf_path=urdf_path,
        robot_cfg_path=robot_cfg_path,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
        q_start=q_start,
        scene_mesh_obstacles=scene_mesh_obstacles,
        cfg=planner_client_config_from_args(args_cli, scene_cuboid_obstacles=_CUROBO_SCENE_CUBOID_OBSTACLES),
    )


def _format_optional_tensor(tensor: torch.Tensor | None) -> str:
    return _format_optional_tensor_impl(tensor, _format_tensor)


def _arm_control_debug_suffix(
    robot,
    *,
    target_q: torch.Tensor | None = None,
    action: torch.Tensor | None = None,
) -> str:
    return _arm_control_debug_suffix_impl(
        robot,
        target_q=target_q,
        action=action,
        arm_joint_ids_fn=_arm_joint_ids,
        joint_error_fn=_joint_error,
        format_tensor_fn=_format_tensor,
    )


def _arm_drive_override_requested() -> bool:
    return arm_drive_override_config_from_args(args_cli).requested


def _arm_drive_debug_suffix(robot) -> str:
    return _arm_drive_debug_suffix_impl(
        robot,
        _arm_joint_ids,
        _arm_joint_names,
        _format_tensor,
    )


def _apply_arm_drive_overrides(robot, *, label: str) -> None:
    _apply_arm_drive_overrides_impl(
        robot,
        label=label,
        cfg=arm_drive_override_config_from_args(args_cli),
        arm_joint_ids_fn=_arm_joint_ids,
        arm_joint_names_fn=_arm_joint_names,
        format_tensor_fn=_format_tensor,
    )


def _apply_arm_drive_overrides_to_env_cfg(env_cfg, *, label: str) -> None:
    _apply_arm_drive_overrides_to_env_cfg_impl(
        env_cfg,
        label=label,
        cfg=arm_drive_override_config_from_args(args_cli),
    )


def _run_pose_plan_worker(
    payload: dict,
    *,
    env=None,
    robot=None,
    hold_q: torch.Tensor | None = None,
    hold_gripper_joint1_pos: float | None = None,
) -> dict:
    return _run_pose_plan_worker_impl(
        payload,
        worker_script=Path(__file__).with_name("run_curobo_pose_plan_test_worker.py"),
        task_root=TASK_ROOT,
        cfg=worker_client_config_from_args(args_cli),
        env=env,
        robot=robot,
        hold_q=hold_q,
        hold_gripper_joint1_pos=hold_gripper_joint1_pos,
        build_action_fn=_build_action,
        hold_base_stationary_fn=_hold_base_stationary,
        arm_joint_ids_fn=_arm_joint_ids,
        set_arm_effort_target_fn=_set_arm_effort_target,
        joint_error_fn=_joint_error,
        format_tensor_fn=_format_tensor,
        arm_drive_debug_suffix_fn=_arm_drive_debug_suffix,
        arm_control_debug_suffix_fn=_arm_control_debug_suffix,
        step_fn=_step,
        show_camera_windows_fn=_show_camera_windows,
    )


def _write_worker_payload_debug(payload: dict, *, label: str) -> Path:
    return _write_worker_payload_debug_impl(payload, label=label, task_root=TASK_ROOT)


def _attach_video_recorder(env):
    return _attach_video_recorder_impl(
        env,
        record_video=args_cli.record_video,
        output_dir=args_cli.video_output_dir,
        fps=args_cli.video_fps,
        every_n_steps=args_cli.video_every_n_steps,
    )


def _plan_single_pose(
    *,
    case_name: str,
    phase_name: str,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    q_start: torch.Tensor,
    scene_mesh_obstacles: list[dict],
    target_pos_base: torch.Tensor,
    target_quat_base: torch.Tensor,
    target_pos_world_z: float | None = None,
    env=None,
    robot=None,
    hold_gripper_joint1_pos: float | None = None,
) -> dict:
    return _plan_single_pose_impl(
        case_name=case_name,
        phase_name=phase_name,
        urdf_path=urdf_path,
        robot_cfg_path=robot_cfg_path,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
        q_start=q_start,
        scene_mesh_obstacles=scene_mesh_obstacles,
        target_pos_base=target_pos_base,
        target_quat_base=target_quat_base,
        target_pos_world_z=target_pos_world_z,
        env=env,
        robot=robot,
        hold_gripper_joint1_pos=hold_gripper_joint1_pos,
        cfg=planner_client_config_from_args(args_cli, scene_cuboid_obstacles=_CUROBO_SCENE_CUBOID_OBSTACLES),
        run_pose_plan_worker_fn=_run_pose_plan_worker,
        write_worker_payload_debug_fn=_write_worker_payload_debug,
    )


def _plan_cspace_target(
    *,
    case_name: str,
    phase_name: str,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    q_start: torch.Tensor,
    scene_mesh_obstacles: list[dict],
    target_q: torch.Tensor,
    env=None,
    robot=None,
    hold_gripper_joint1_pos: float | None = None,
) -> dict:
    return _plan_cspace_target_impl(
        case_name=case_name,
        phase_name=phase_name,
        urdf_path=urdf_path,
        robot_cfg_path=robot_cfg_path,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
        q_start=q_start,
        scene_mesh_obstacles=scene_mesh_obstacles,
        target_q=target_q,
        env=env,
        robot=robot,
        hold_gripper_joint1_pos=hold_gripper_joint1_pos,
        cfg=planner_client_config_from_args(args_cli, scene_cuboid_obstacles=_CUROBO_SCENE_CUBOID_OBSTACLES),
        run_pose_plan_worker_fn=_run_pose_plan_worker,
        write_worker_payload_debug_fn=_write_worker_payload_debug,
    )


def _plan_and_execute_cspace_target(
    *,
    env,
    robot,
    case_name: str,
    phase_name: str,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    scene_mesh_obstacles: list[dict],
    target_q: torch.Tensor,
    hold_gripper_joint1_pos: float | None = None,
    final_settle: bool = True,
    final_hold_steps: int = 0,
) -> bool:
    arm_ids = _arm_joint_ids(robot)
    q_start = robot.data.joint_pos[0, arm_ids].detach().clone()
    result = _plan_cspace_target(
        case_name=case_name,
        phase_name=phase_name,
        urdf_path=urdf_path,
        robot_cfg_path=robot_cfg_path,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
        q_start=q_start,
        scene_mesh_obstacles=scene_mesh_obstacles,
        target_q=target_q,
        env=env,
        robot=robot,
        hold_gripper_joint1_pos=hold_gripper_joint1_pos,
    )
    if not result.get("success") or not result.get("trajectory"):
        print(
            f"[CspaceExecute] case={case_name} phase={phase_name} plan_failed; "
            f"holding current_q={_format_tensor(robot.data.joint_pos[0, arm_ids].detach())}",
            flush=True,
        )
        return False
    traj = torch.tensor(result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
    if traj.dim() > 2:
        traj = traj.reshape(-1, traj.shape[-1])
    print(
        f"[CspaceExecute] case={case_name} phase={phase_name} trajectory_shape={tuple(traj.shape)} "
        f"first={_format_tensor(traj[0])} last={_format_tensor(traj[-1])}",
        flush=True,
    )
    _execute_trajectory(
        env,
        robot,
        traj,
        gripper_joint1_pos=hold_gripper_joint1_pos,
        label=f"{case_name}:{phase_name}",
    )
    if final_settle:
        actual_q = robot.data.joint_pos[0, arm_ids].detach()
        joint_err_norm, joint_err_max, joint_err = _joint_error(target_q, actual_q)
        print(
            f"[CspaceExecute] case={case_name} phase={phase_name} final_settle_via_curobo_trajectory "
            f"target_q={_format_tensor(target_q)} actual_q={_format_tensor(actual_q)} "
            f"joint_err_norm={joint_err_norm.item():.6f} joint_err_max={joint_err_max.item():.6f} "
            f"joint_err={_format_tensor(joint_err)}",
            flush=True,
        )
    if final_hold_steps > 0:
        _hold_joint_target_if_close(
            env,
            robot,
            target_q,
            final_hold_steps,
            label=f"{case_name}:{phase_name}:final_hold",
            gripper_joint1_pos=hold_gripper_joint1_pos,
        )
    return True


def _execute_direct_joint_roll(
    *,
    env,
    robot,
    target_q: torch.Tensor,
    label: str,
    final_hold_steps: int = 0,
    gripper_joint1_pos: float | None = None,
) -> tuple[int, float]:
    arm_ids = _arm_joint_ids(robot)
    start_q = robot.data.joint_pos[0, arm_ids].detach().clone()
    target_q = target_q.detach().to(device=robot.device, dtype=robot.data.joint_pos.dtype).reshape(-1)
    target_q = _clamp_arm_joint_target_to_soft_limits(robot, target_q, label=label)
    delta = target_q - start_q
    max_joint_step_rad = 0.05
    max_delta = float(torch.max(torch.abs(delta)).item()) if delta.numel() else 0.0
    waypoint_count = max(2, int(math.ceil(max_delta / max_joint_step_rad)) + 1)
    blend = torch.linspace(0.0, 1.0, waypoint_count, device=robot.device, dtype=target_q.dtype).reshape(-1, 1)
    traj = start_q.reshape(1, -1) + blend * delta.reshape(1, -1)
    print(
        f"[DirectJointRoll] label={label} waypoint_count={waypoint_count} "
        f"max_joint_step_rad={max_joint_step_rad:.6f} start_q={_format_tensor(start_q)} "
        f"target_q={_format_tensor(target_q)} delta={_format_tensor(delta)}",
        flush=True,
    )
    _execute_trajectory(
        env,
        robot,
        traj,
        gripper_joint1_pos=gripper_joint1_pos,
        label=label,
    )
    actual_q = robot.data.joint_pos[0, arm_ids].detach()
    joint_err_norm, joint_err_max, joint_err = _joint_error(target_q, actual_q)
    print(
        f"[DirectJointRoll] label={label} complete target_q={_format_tensor(target_q)} "
        f"actual_q={_format_tensor(actual_q)} joint_err_norm={joint_err_norm.item():.6f} "
        f"joint_err_max={joint_err_max.item():.6f} joint_err={_format_tensor(joint_err)}",
        flush=True,
    )
    if final_hold_steps > 0:
        hold_steps = _hold_joint_target_if_close(
            env,
            robot,
            target_q,
            final_hold_steps,
            label=f"{label}:final_hold",
            gripper_joint1_pos=gripper_joint1_pos,
        )
    else:
        hold_steps = 0
    env_step_count = waypoint_count * max(1, int(args_cli.steps_per_waypoint)) + int(hold_steps)
    return int(env_step_count), float(joint_err_max.item())


def _log_joint_delta(label: str, target_q: torch.Tensor, actual_q: torch.Tensor) -> None:
    joint_err_norm, joint_err_max, joint_err = _joint_error(target_q, actual_q)
    print(
        f"[JointDelta] label={label} target_q={_format_tensor(target_q)} "
        f"actual_q={_format_tensor(actual_q)} joint_err_norm={joint_err_norm.item():.6f} "
        f"joint_err_max={joint_err_max.item():.6f} joint_err={_format_tensor(joint_err)}",
        flush=True,
    )


def _execute_post_target_world_descend(
    *,
    env,
    robot,
    case_name: str,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    scene_mesh_obstacles: list[dict],
    isaac_base_body: str,
    descend_m: float,
    align_xy_pos_w: torch.Tensor | None = None,
    target_z_w: float | None = None,
) -> torch.Tensor | None:
    """Plan and execute a short world -Z move from the settled tool-tip pose.

    align_xy_pos_w: optional world position whose XY the tool tip is aligned to
    while descending (used for grasp_test_cube to compensate cuRobo-vs-Isaac FK
    bias that would otherwise make the gripper close on the cube edge).

    target_z_w: optional absolute world Z for the tool tip after descend. When set,
    overrides the default (start_z - descend_m). Used for grasp_test_cube so the
    gripper fingers envelop the cube body instead of closing at its top face.
    """
    if descend_m <= 0.0 and target_z_w is None:
        return None

    arm_ids = _arm_joint_ids(robot)
    start_q = robot.data.joint_pos[0, arm_ids].detach().clone()
    start_pos_w, start_quat_w = _tool_tip_pose_w(robot)
    target_pos_w = start_pos_w.reshape(1, 3).detach().clone()
    if align_xy_pos_w is not None:
        align_xy = align_xy_pos_w.reshape(-1).detach().cpu()
        # 只对齐 XY，Z 单独处理
        target_pos_w[:, 0] = float(align_xy[0])
        target_pos_w[:, 1] = float(align_xy[1])
    if target_z_w is not None:
        target_pos_w[:, 2] = float(target_z_w)
    else:
        target_pos_w[:, 2] -= float(descend_m)
    target_quat_w = start_quat_w.reshape(1, 4).detach().clone()
    arm_base_pos_w, arm_base_quat_w = _body_pose_w(robot, isaac_base_body)
    target_pos_base, target_quat_base = subtract_frame_transforms(
        arm_base_pos_w.reshape(1, 3),
        arm_base_quat_w.reshape(1, 4),
        target_pos_w,
        target_quat_w,
    )
    print(
        f"[PostTargetDescend] case={case_name} planning world_minus_z_m={descend_m:.6f} "
        f"start_pos_w={_format_tensor(start_pos_w)} target_pos_w={_format_tensor(target_pos_w)} "
        f"start_q={_format_tensor(start_q)}",
        flush=True,
    )
    descend_result = _plan_single_pose(
        case_name=case_name,
        phase_name="post_target_descend",
        urdf_path=urdf_path,
        robot_cfg_path=robot_cfg_path,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
        q_start=start_q,
        scene_mesh_obstacles=scene_mesh_obstacles,
        target_pos_base=target_pos_base,
        target_quat_base=target_quat_base,
        target_pos_world_z=float(target_pos_w.reshape(-1)[2].item()),
        env=env,
        robot=robot,
        hold_gripper_joint1_pos=_default_open_gripper_joint1_pos(),
    )
    if not descend_result["success"]:
        print(
            f"[PostTargetDescend] case={case_name} plan_failed; grasp will use current settled target pose",
            flush=True,
        )
        return None

    descend_traj = torch.tensor(descend_result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
    if descend_traj.dim() > 2:
        descend_traj = descend_traj.reshape(-1, descend_traj.shape[-1])
    print(
        f"[PostTargetDescend] case={case_name} trajectory_shape={tuple(descend_traj.shape)} "
        f"first={_format_tensor(descend_traj[0])} last={_format_tensor(descend_traj[-1])}",
        flush=True,
    )
    _execute_trajectory(
        env,
        robot,
        descend_traj,
        gripper_joint1_pos=_default_open_gripper_joint1_pos(),
        label=f"{case_name}:post_target_descend",
    )
    final_q = descend_traj[-1].detach().clone()
    hold_steps = int(args_cli.post_target_descend_hold_steps) if args_cli.post_target_descend_hold_steps is not None else int(args_cli.final_hold_steps)
    if hold_steps > 0:
        _hold_joint_target_if_close(
            env,
            robot,
            final_q,
            hold_steps,
            label=f"{case_name}:post_target_descend_hold",
            gripper_joint1_pos=_default_open_gripper_joint1_pos(),
        )
    final_pos_w, final_quat_w = _tool_tip_pose_w(robot)
    pos_err_w, rot_err_rad_w, rot_err_deg_w = _pose_error(
        target_pos_w,
        target_quat_w,
        final_pos_w.reshape(1, 3),
        final_quat_w.reshape(1, 4),
    )
    print(
        f"[PostTargetDescend] case={case_name} complete hold_steps={hold_steps} "
        f"final_pos_w={_format_tensor(final_pos_w)} pos_err_m={pos_err_w.item():.6f} "
        f"rot_err_deg={rot_err_deg_w.item():.3f}",
        flush=True,
    )
    return final_q


def _execute_gripper_post_target_sequence(
    *,
    env,
    robot,
    case_name: str,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    scene_mesh_obstacles: list[dict],
    isaac_base_body: str,
    pre_target_q: torch.Tensor,
    post_target_pre_descend_q: torch.Tensor | None = None,
    post_target_descend_q: torch.Tensor | None = None,
) -> bool:
    """After reaching the target, grasp in place, then return while holding the grasp."""
    arm_ids = _arm_joint_ids(robot)
    hold_steps = _default_gripper_hold_steps()

    grasp_q = (
        post_target_descend_q.detach().clone()
        if post_target_descend_q is not None
        else robot.data.joint_pos[0, arm_ids].detach().clone()
    )
    if post_target_descend_q is not None:
        _hold_joint_target_if_close(
            env,
            robot,
            grasp_q,
            max(1, int(args_cli.steps_per_waypoint)),
            label=f"{case_name}:grasp_pose_open_hold",
            gripper_joint1_pos=_default_open_gripper_joint1_pos(),
        )
    grasp_pos_w, grasp_quat_w = _tool_tip_pose_w(robot)
    print(
        f"[GripperPostTarget] case={case_name} grasp_in_place "
        f"tool_tip_pos_w={_format_tensor(grasp_pos_w)} "
        f"tool_tip_quat_wxyz={_format_tensor(grasp_quat_w)} "
        f"arm_q={_format_tensor(grasp_q)}",
        flush=True,
    )

    grasp_hold_q, grasped = _execute_gripper_feedback_grasp(
        env,
        robot,
        grasp_q,
        label=f"{case_name}_feedback_grasp",
    )
    _execute_joint_target_until_converged(
        env,
        robot,
        grasp_q,
        max_steps=hold_steps,
        min_steps=max(1, int(args_cli.steps_per_waypoint)),
        label=f"{case_name}:grasp_hold",
        gripper_joint1_pos=grasp_hold_q,
    )
    state = _gripper_joint1_state(robot)
    if state is not None:
        actual_gripper_q, actual_gripper_vel = state
        joint2_state = _gripper_joint_state_by_name(robot, "gripper_joint2")
        joint2_text = ""
        if joint2_state is not None:
            actual_gripper2_q, actual_gripper2_vel = joint2_state
            joint2_text = (
                f" actual_gripper_joint2={actual_gripper2_q:.6f}"
                f" actual_gripper_joint2_vel={actual_gripper2_vel:.6f}"
            )
        print(
            f"[GripperPostTarget] case={case_name} grasp_hold_complete grasped={grasped} "
            f"hold_target_q={grasp_hold_q:.6f} actual_gripper_joint1={actual_gripper_q:.6f} "
            f"actual_gripper_vel={actual_gripper_vel:.6f}{joint2_text}",
            flush=True,
        )

    print(
        f"[GripperPostTarget] case={case_name} holding_grasp_at_target "
        f"arm_q={_format_tensor(robot.data.joint_pos[0, arm_ids].detach())}",
        flush=True,
    )

    lift_start_q = robot.data.joint_pos[0, arm_ids].detach().clone()
    lift_start_pos_w, lift_start_quat_w = _tool_tip_pose_w(robot)
    post_grasp_lift_m = max(0.0, float(args_cli.post_grasp_lift_m))

    # GPT P0: Replace cuRobo trajopt lift (~15s) with reverse-trajectory lift.
    # The saved grab trajectory describes a collision-free path from pre-target to paper.
    # Reversing the first N waypoints lifts the arm back along the same safe path.
    global _LAST_GRAB_TRAJECTORY
    if _LAST_GRAB_TRAJECTORY is not None and _LAST_GRAB_TRAJECTORY.shape[0] >= 10:
        _n_lift = min(25, _LAST_GRAB_TRAJECTORY.shape[0] // 2)
        _lift_traj = torch.flip(_LAST_GRAB_TRAJECTORY[:_n_lift], [0])
        print(
            f"[GripperLift] case={case_name} reverse_traj_lift "
            f"waypoints={_lift_traj.shape[0]} save_~15s_vs_trajopt",
            flush=True,
        )
    else:
        # Fallback: simple joint hold at current position
        _lift_traj = lift_start_q.unsqueeze(0).repeat(5, 1)
        print(
            f"[GripperLift] case={case_name} hold_in_place (no grab trajectory available)",
            flush=True,
        )

    _execute_trajectory(
        env,
        robot,
        _lift_traj,
        gripper_joint1_pos=grasp_hold_q,
        label=f"{case_name}:gripper_lift",
    )
    lift_final_q = _lift_traj[-1].detach().clone()
    lift_executed = True
    lift_error = None
    lift_final_pos_w, lift_final_quat_w = _tool_tip_pose_w(robot)
    print(
        f"[GripperLift] case={case_name} lift_complete "
        f"final_tool_pos_w={_format_tensor(lift_final_pos_w)}",
        flush=True,
    )

    return_start_q = robot.data.joint_pos[0, arm_ids].detach().clone()
    _log_joint_delta(f"{case_name}:gripper_return_start_to_grasp_q", grasp_q, return_start_q)
    if post_target_pre_descend_q is not None:
        _log_joint_delta(
            f"{case_name}:gripper_return_start_to_pre_descend_q",
            post_target_pre_descend_q,
            return_start_q,
        )
    _log_joint_delta(f"{case_name}:gripper_return_start_to_lift_q", lift_final_q, return_start_q)
    _log_joint_delta(f"{case_name}:gripper_return_start_to_pre_target", pre_target_q, return_start_q)

    returned_with_plan = _plan_and_execute_cspace_target(
        env=env,
        robot=robot,
        case_name=case_name,
        phase_name="gripper_return_pre_target_grasp_hold",
        urdf_path=urdf_path,
        robot_cfg_path=robot_cfg_path,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
        scene_mesh_obstacles=scene_mesh_obstacles,
        target_q=pre_target_q,
        hold_gripper_joint1_pos=grasp_hold_q,
        final_settle=True,
    )
    if not returned_with_plan:
        return False
    returned_q = robot.data.joint_pos[0, arm_ids].detach()
    print(
        f"[GripperPostTarget] case={case_name} returned_to_pre_target_grasp_hold "
        f"target_q={_format_tensor(pre_target_q)} actual_q={_format_tensor(returned_q)}",
        flush=True,
    )

    drop_start_q = robot.data.joint_pos[0, arm_ids].detach().clone()
    arm_base_pos_w, arm_base_quat_w = _body_pose_w(robot, isaac_base_body)
    drop_target_pos_w = torch.tensor(
        _GRIPPER_DROP_TARGET_POS_W,
        device=robot.device,
        dtype=torch.float32,
    ).reshape(1, 3)
    drop_target_quat_w = torch.tensor(
        _GRIPPER_DROP_TARGET_QUAT_WXYZ,
        device=robot.device,
        dtype=torch.float32,
    ).reshape(1, 4)
    drop_target_pos_base, drop_target_quat_base = subtract_frame_transforms(
        arm_base_pos_w.reshape(1, 3),
        arm_base_quat_w.reshape(1, 4),
        drop_target_pos_w,
        drop_target_quat_w,
    )
    print(
        f"[GripperDrop] case={case_name} planning_to_drop "
        f"target_pos_w={_GRIPPER_DROP_TARGET_POS_W} "
        f"target_quat_wxyz={_GRIPPER_DROP_TARGET_QUAT_WXYZ} "
        f"start_q={_format_tensor(drop_start_q)}",
        flush=True,
    )
    drop_result = _plan_single_pose(
        case_name=case_name,
        phase_name="drop_target",
        urdf_path=urdf_path,
        robot_cfg_path=robot_cfg_path,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
        q_start=drop_start_q,
        scene_mesh_obstacles=scene_mesh_obstacles,
        target_pos_base=drop_target_pos_base,
        target_quat_base=drop_target_quat_base,
        target_pos_world_z=float(drop_target_pos_w.reshape(-1)[2].item()),
        env=env,
        robot=robot,
        hold_gripper_joint1_pos=grasp_hold_q,
    )
    if not drop_result["success"]:
        print(
            f"[GripperDrop] case={case_name} drop_plan_failed; holding_pre_target_grasp",
            flush=True,
        )
        return False

    drop_traj = torch.tensor(drop_result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
    if drop_traj.dim() > 2:
        drop_traj = drop_traj.reshape(-1, drop_traj.shape[-1])
    print(
        f"[GripperDrop] case={case_name} trajectory_shape={tuple(drop_traj.shape)} "
        f"first={_format_tensor(drop_traj[0])} last={_format_tensor(drop_traj[-1])}",
        flush=True,
    )
    _execute_trajectory(
        env,
        robot,
        drop_traj,
        gripper_joint1_pos=grasp_hold_q,
        label=f"{case_name}:gripper_drop",
    )
    drop_final_q = drop_traj[-1].detach().clone()
    actual_drop_q = robot.data.joint_pos[0, arm_ids].detach()
    _, drop_err_max, drop_err = _joint_error(drop_final_q, actual_drop_q)
    print(
        f"[GripperDrop] case={case_name} drop_target_reached_via_curobo "
        f"target_q={_format_tensor(drop_final_q)} actual_q={_format_tensor(actual_drop_q)} "
        f"joint_err_max={drop_err_max.item():.6f} joint_err={_format_tensor(drop_err)}",
        flush=True,
    )

    open_gripper_q = _default_open_gripper_joint1_pos()
    _hold_joint_target_if_close(
        env,
        robot,
        drop_final_q,
        hold_steps,
        label=f"{case_name}:drop_open_hold",
        gripper_joint1_pos=open_gripper_q,
    )
    opened_state = _gripper_joint1_state(robot)
    if opened_state is not None:
        opened_q, opened_vel = opened_state
        print(
            f"[GripperDrop] case={case_name} opened_gripper "
            f"target_q={open_gripper_q:.6f} "
            f"actual_q={opened_q:.6f} actual_vel={opened_vel:.6f}",
            flush=True,
        )

    returned_open_with_plan = _plan_and_execute_cspace_target(
        env=env,
        robot=robot,
        case_name=case_name,
        phase_name="drop_return_pre_target_open",
        urdf_path=urdf_path,
        robot_cfg_path=robot_cfg_path,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
        scene_mesh_obstacles=scene_mesh_obstacles,
        target_q=pre_target_q,
        hold_gripper_joint1_pos=open_gripper_q,
        final_settle=True,
    )
    if not returned_open_with_plan:
        return False
    final_return_q = robot.data.joint_pos[0, arm_ids].detach()
    print(
        f"[GripperDrop] case={case_name} returned_to_pre_target_open "
        f"target_q={_format_tensor(pre_target_q)} actual_q={_format_tensor(final_return_q)}",
        flush=True,
    )
    return True


def _plan_multi_pose_polynomial_seed(
    *,
    case_name: str,
    env,
    robot=None,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    q_start: torch.Tensor,
    scene_mesh_obstacles: list[dict],
    target_poses_base: list[dict] | None = None,
    target_poses: list[dict] | None = None,
    seed_mode: str = "cubic",
) -> dict:
    """Plan one continuous trajectory from q_start through multiple target tool poses.

    This is intentionally separate from _plan_single_pose so existing pose tests
    keep their current behavior.
    """
    if target_poses_base is None:
        if robot is None:
            raise ValueError("robot is required when target_poses_base is not provided.")
        if target_poses is None:
            raise ValueError("Either target_poses_base or target_poses must be provided.")
        target_poses_base = [
            _convert_brush_target_to_base(
                robot=robot,
                base_link=base_link,
                target=target,
                default_name=f"{case_name}_target_{target_idx + 1}",
                force_safe_clearance=True,
            )
            for target_idx, target in enumerate(target_poses)
        ]

    return _plan_multi_pose_polynomial_seed_impl(
        case_name=case_name,
        env=env,
        robot=robot,
        urdf_path=urdf_path,
        robot_cfg_path=robot_cfg_path,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
        q_start=q_start,
        scene_mesh_obstacles=scene_mesh_obstacles,
        target_poses_base=target_poses_base,
        seed_mode=seed_mode,
        cfg=planner_client_config_from_args(args_cli, scene_cuboid_obstacles=_CUROBO_SCENE_CUBOID_OBSTACLES),
        run_pose_plan_worker_fn=_run_pose_plan_worker,
        hold_gripper_joint1_pos=_default_open_gripper_joint1_pos(),
    )


def _convert_brush_target_to_base(
    *,
    robot,
    base_link: str,
    target: dict,
    default_name: str,
    force_safe_clearance: bool = False,
) -> dict:
    """Apply brush point-clean target defaults and convert one target to base frame."""
    target_pos = _float_list_parameter(target, "target_pos", 3, required=True)
    if force_safe_clearance:
        target_pos, safe_clearance_m = _apply_brush_target_forced_safe_clearance(target_pos)
    else:
        target_pos, safe_clearance_m = _apply_brush_target_safe_clearance(target_pos)
    target_quat = _float_list_parameter(target, "target_quat", 4, required=False)
    target_quat_defaulted = target_quat is None
    if target_quat_defaulted:
        target_quat = _default_brush_target_quat_wxyz(target_pos)

    target_frame = str(target.get("target_frame", "world")).lower()
    if target_frame not in {"world", "base"}:
        raise ValueError("multi-pose target_frame must be either 'world' or 'base'.")

    arm_base_pos_w, arm_base_quat_w = _body_pose_w(robot, base_link.replace("-", "_"))
    converted = {
        "name": str(target.get("name", default_name)),
        "input_target_frame": target_frame,
        "target_quat_defaulted": target_quat_defaulted,
        "safe_clearance_m": safe_clearance_m,
    }
    if target_frame == "world":
        target_pos_w = torch.tensor(target_pos, device=robot.device, dtype=torch.float32).reshape(1, 3)
        target_quat_w = torch.tensor(target_quat, device=robot.device, dtype=torch.float32).reshape(1, 4)
        target_pos_base, target_quat_base = subtract_frame_transforms(
            arm_base_pos_w.reshape(1, 3),
            arm_base_quat_w.reshape(1, 4),
            target_pos_w,
            target_quat_w,
        )
        converted["target_pos"] = target_pos_base.reshape(-1).detach().cpu().tolist()
        converted["target_quat"] = target_quat_base.reshape(-1).detach().cpu().tolist()
        converted["target_pos_world_z"] = float(target_pos_w.reshape(-1)[2].item())
    else:
        converted["target_pos"] = list(target_pos)
        target_pos_base = torch.tensor(target_pos, device=robot.device, dtype=torch.float32).reshape(1, 3)
        target_pos_w, _ = combine_frame_transforms(
            arm_base_pos_w.reshape(1, 3),
            arm_base_quat_w.reshape(1, 4),
            target_pos_base,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=robot.device, dtype=torch.float32),
        )
        converted["target_pos_world_z"] = float(target_pos_w.reshape(-1)[2].item())
        if target_quat_defaulted:
            target_quat_w = torch.tensor(target_quat, device=robot.device, dtype=torch.float32).reshape(1, 4)
            _, target_quat_base = subtract_frame_transforms(
                arm_base_pos_w.reshape(1, 3),
                arm_base_quat_w.reshape(1, 4),
                target_pos_w,
                target_quat_w,
            )
            converted["target_quat"] = target_quat_base.reshape(-1).detach().cpu().tolist()
        else:
            converted["target_quat"] = list(target_quat)
    return converted


def _run_tool_api_server(
    *,
    env,
    robot,
    arm_ids: list[int],
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    scene_mesh_obstacles: list[dict],
    q_initial: torch.Tensor,
    pre_target_q: torch.Tensor,
    default_target_quat_wxyz: list[float],
) -> None:
    planner_tool = make_curobo_pose_plan_tool(timeout_s=args_cli.worker_timeout_s)
    planner_cfg = planner_client_config_from_args(
        args_cli,
        scene_cuboid_obstacles=_CUROBO_SCENE_CUBOID_OBSTACLES,
    )

    import omni.usd as _omni_usd

    _stage = _omni_usd.get_context().get_stage()

    # ── Glass door visual fix (opaque fallback for Isaac Sim 4.5) ──
    try:
        _glass_result = apply_glass_door_visual_fix(_stage, "/World/envs/env_0", enabled=True)
        print(f"[GlassDoorFix] startup: {_glass_result['bound_meshes']}/{_glass_result['found_meshes']} meshes bound", flush=True)
    except Exception as _e:
        print(f"[GlassDoorFix] startup error: {_e}", flush=True)

    def _make_phase_planner_cfg(phase: str):
        """Build a planner_cfg with wall cuboids filtered by phase.

        grasp:  all walls (full collision safety)
        lift / return / takeout:  exclude glass door panels and walls far behind the tool
        """
        if phase == "grasp":
            return planner_cfg
        # 非抓取阶段：排除玻璃门面板 + 机器人后方远处墙壁
        tool_pos_w, _ = _tool_tip_pose_w(robot)
        tool_x = float(tool_pos_w[0].item())
        tool_y = float(tool_pos_w[1].item())
        filtered = []
        dropped = []
        for c in _CUROBO_SCENE_CUBOID_OBSTACLES:
            sp = c.get("source_path", "")
            # 排除纸团代理 — 纸团是抓取目标，不是障碍物
            if "paper_waste" in sp:
                dropped.append(f"paper_waste:{c['name']}")
                continue
            # 排除玻璃门面板
            if "glass_door" in sp:
                dropped.append(f"glass_door:{c['name']}")
                continue
            # 排除机器人后方 > 0.5m 的墙壁（cuboid 中心在 tool 后方且远离）
            cx = float(c.get("center_x_m", 0))
            cy = float(c.get("center_y_m", 0))
            if phase in ("lift", "return"):
                # 只保留 tool 前方 +0.3m 到 -0.8m 范围内的墙壁
                x_dist = cx - tool_x
                y_dist = abs(cy - tool_y)
                if x_dist < -0.8 or y_dist > 1.5:
                    dropped.append(f"far:{c['name']} x_dist={x_dist:.2f} y_dist={y_dist:.2f}")
                    continue
            if phase == "takeout":
                # 投放阶段：排除机器人后方墙壁
                x_dist = cx - tool_x
                if x_dist < -1.0:
                    dropped.append(f"far:{c['name']} x_dist={x_dist:.2f}")
                    continue
            filtered.append(c)
        print(
            f"[PhaseFilter] phase={phase} total={len(_CUROBO_SCENE_CUBOID_OBSTACLES)} "
            f"kept={len(filtered)} dropped={len(dropped)} {dropped}",
            flush=True,
        )
        return planner_client_config_from_args(
            args_cli,
            scene_cuboid_obstacles=filtered,
        )
    common_robot_profiles = _common_robot_profiles(env.cfg)
    active_robot_id = str(
        getattr(env.cfg, "_allstar_initial_robot_id", None)
        or args_cli.initial_robot_id
        or getattr(env.cfg, "default_robot_id", None)
        or getattr(getattr(env.cfg, "scene", None), "default_robot_id", "")
        or ""
    )
    if common_robot_profiles and active_robot_id and active_robot_id not in common_robot_profiles:
        raise ValueError(
            f"Unknown --initial_robot_id={active_robot_id!r}; "
            f"available={sorted(common_robot_profiles)}"
        )
    common_base_root_z = float(getattr(env.cfg.scene.robot.init_state, "pos", (0.0, 0.0, 0.0))[2])

    def _common_profile_enabled() -> bool:
        return bool(common_robot_profiles)

    def _move_robot_root(robot_obj, pos_xyz: tuple[float, float, float] | None = None) -> None:
        if pos_xyz is None:
            return
        root_pose = robot_obj.data.root_link_pose_w.detach().clone()
        root_pose[:, 0:3] = torch.tensor(pos_xyz, device=robot_obj.device, dtype=root_pose.dtype).reshape(1, 3)
        robot_obj.write_root_pose_to_sim(root_pose)
        robot_obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=robot_obj.device, dtype=root_pose.dtype))

    def _active_work_root_pose(robot_obj) -> torch.Tensor:
        root_pose = robot_obj.data.default_root_state.detach().clone()[:, :7]
        root_pose[:, 0:3] = torch.tensor(
            (1.832, -0.264, common_base_root_z),
            device=robot_obj.device,
            dtype=root_pose.dtype,
        ).reshape(1, 3)
        root_pose[:, 3:7] = torch.tensor(
            (0.702153052995162, 0.0, 0.0, -0.712026045990997),
            device=robot_obj.device,
            dtype=root_pose.dtype,
        ).reshape(1, 4)
        return root_pose

    def _robot_cfg_for_profile(robot_id: str):
        robot_cfg_factory = getattr(importlib.import_module(env.cfg.__class__.__module__), "make_common_robot_cfg", None)
        if callable(robot_cfg_factory):
            return robot_cfg_factory(robot_id, prim_path=f"{env.scene.env_regex_ns}/robot")
        robot_cfg = copy.deepcopy(env.cfg.scene.robot)
        robot_cfg.prim_path = f"{env.scene.env_regex_ns}/robot"
        robot_cfg.spawn.usd_path = _profile_usd_path(env.cfg, robot_id)
        return robot_cfg

    def _replace_scene_robot(robot_id: str):
        import builtins

        import omni.timeline
        import isaaclab.sim as sim_utils

        robot_cfg = _robot_cfg_for_profile(robot_id)
        env0_robot_path = "/World/envs/env_0/robot"
        print(
            f"[CommonRobotSwitch] event=replace_begin target_robot_id={robot_id} "
            f"path={env0_robot_path} sim_playing={env.sim.is_playing()}",
            flush=True,
        )
        old_robot = env.scene.articulations.pop("robot", None)
        if old_robot is not None:
            try:
                old_robot._clear_callbacks()
            except Exception:
                pass
        if getattr(env, "_allstar_active_tool_robot", None) is old_robot:
            delattr(env, "_allstar_active_tool_robot")
        builtins.ISAACLAB_CALLBACK_EXCEPTION = None
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_stopped():
            print("[CommonRobotSwitch] event=timeline_stop_begin", flush=True)
            timeline.stop()
            print("[CommonRobotSwitch] event=timeline_stop_requested", flush=True)
        if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
            exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None
            raise exception_to_raise
        print("[CommonRobotSwitch] event=delete_old_begin", flush=True)
        sim_utils.delete_prim(env0_robot_path, stage=env.sim.stage)
        print("[CommonRobotSwitch] event=delete_old_done", flush=True)
        env.cfg.scene.robot = robot_cfg
        print("[CommonRobotSwitch] event=create_new_begin", flush=True)
        new_robot = robot_cfg.class_type(robot_cfg)
        env.scene.articulations["robot"] = new_robot
        print("[CommonRobotSwitch] event=create_new_done", flush=True)
        builtins.ISAACLAB_CALLBACK_EXCEPTION = None
        print("[CommonRobotSwitch] event=sim_reset_begin soft=False", flush=True)
        env.sim.reset(soft=False)
        print(
            f"[CommonRobotSwitch] event=sim_reset_done "
            f"stopped={env.sim.is_stopped()} playing={env.sim.is_playing()}",
            flush=True,
        )
        if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
            exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
            builtins.ISAACLAB_CALLBACK_EXCEPTION = None
            raise exception_to_raise
        if not getattr(new_robot, "is_initialized", False):
            print("[CommonRobotSwitch] event=initialize_new_begin", flush=True)
            new_robot._initialize_callback(None)
            print("[CommonRobotSwitch] event=initialize_new_done", flush=True)
            if builtins.ISAACLAB_CALLBACK_EXCEPTION is not None:
                exception_to_raise = builtins.ISAACLAB_CALLBACK_EXCEPTION
                builtins.ISAACLAB_CALLBACK_EXCEPTION = None
                raise exception_to_raise
        new_robot.reset()
        joint_pos = new_robot.data.default_joint_pos.detach().clone()
        joint_vel = torch.zeros_like(new_robot.data.default_joint_vel)
        new_robot.write_joint_state_to_sim(joint_pos, joint_vel)
        new_robot.set_joint_position_target(joint_pos)
        new_robot.set_joint_velocity_target(joint_vel)
        _apply_arm_drive_overrides(new_robot, label=f"switch_robot:{robot_id}:post_rebuild")
        _reset_link5_camera_mount(env)
        _sync_link5_camera(env, new_robot)
        print(
            f"[CommonRobotSwitch] event=replace_done target_robot_id={robot_id} "
            f"initialized={getattr(new_robot, 'is_initialized', False)}",
            flush=True,
        )
        return new_robot

    def _scene_asset_or_none(asset_name: str):
        try:
            return env.scene[asset_name]
        except Exception:
            return None

    def _snapshot_rigid_asset(asset_name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        asset = _scene_asset_or_none(asset_name)
        if asset is None or not hasattr(asset, "write_root_pose_to_sim"):
            return None
        root_pose = getattr(asset.data, "root_pose_w", None)
        if root_pose is None:
            root_pose = getattr(asset.data, "root_link_pose_w", None)
        root_vel = getattr(asset.data, "root_vel_w", None)
        if root_pose is None or root_vel is None:
            return None
        return root_pose.detach().clone(), root_vel.detach().clone()

    def _restore_rigid_asset_snapshot(
        asset_name: str,
        snapshot: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        zero_velocity: bool = True,
    ) -> None:
        if snapshot is None:
            return
        asset = _scene_asset_or_none(asset_name)
        if asset is None:
            return
        root_pose, root_vel = snapshot
        asset.write_root_pose_to_sim(root_pose)
        if zero_velocity:
            root_vel = torch.zeros_like(root_vel)
        asset.write_root_velocity_to_sim(root_vel)

    def _switch_preserved_rigid_asset_names() -> list[str]:
        names: list[str] = []
        rigid_objects = getattr(env.scene, "rigid_objects", None)
        if isinstance(rigid_objects, dict):
            names.extend(str(asset_name) for asset_name in rigid_objects.keys())
        names.extend(_SWITCH_PRESERVED_RIGID_ASSET_FALLBACKS)

        unique_names: list[str] = []
        seen: set[str] = set()
        for asset_name in names:
            if asset_name in seen:
                continue
            seen.add(asset_name)
            unique_names.append(asset_name)
        return unique_names

    def _snapshot_switch_preserved_rigid_assets() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        snapshots: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for asset_name in _switch_preserved_rigid_asset_names():
            snapshot = _snapshot_rigid_asset(asset_name)
            if snapshot is not None:
                snapshots[asset_name] = snapshot
        print(
            f"[CommonRobotSwitch] event=preserve_rigid_assets assets={sorted(snapshots)}",
            flush=True,
        )
        return snapshots

    def _restore_switch_preserved_rigid_assets(
        snapshots: dict[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        zero_velocity: bool = True,
    ) -> None:
        for asset_name, snapshot in snapshots.items():
            _restore_rigid_asset_snapshot(asset_name, snapshot, zero_velocity=zero_velocity)

    def _switch_preserved_articulation_names() -> list[str]:
        names: list[str] = []
        articulations = getattr(env.scene, "articulations", None)
        if isinstance(articulations, dict):
            names.extend(str(asset_name) for asset_name in articulations.keys() if asset_name != "robot")
        return names

    def _snapshot_articulation_asset(asset_name: str) -> dict[str, torch.Tensor] | None:
        asset = _scene_asset_or_none(asset_name)
        if asset is None or not hasattr(asset, "write_joint_state_to_sim"):
            return None
        joint_pos = getattr(asset.data, "joint_pos", None)
        joint_vel = getattr(asset.data, "joint_vel", None)
        if joint_pos is None or joint_vel is None:
            return None
        snapshot: dict[str, torch.Tensor] = {
            "joint_pos": joint_pos.detach().clone(),
            "joint_vel": joint_vel.detach().clone(),
        }
        for attr_name in ("joint_pos_target", "joint_vel_target", "joint_effort_target"):
            target = getattr(asset.data, attr_name, None)
            if target is not None:
                snapshot[attr_name] = target.detach().clone()
        return snapshot

    def _restore_articulation_asset_snapshot(
        asset_name: str,
        snapshot: dict[str, torch.Tensor] | None,
        *,
        zero_velocity: bool = True,
    ) -> None:
        if snapshot is None:
            return
        asset = _scene_asset_or_none(asset_name)
        if asset is None:
            return
        joint_pos = snapshot["joint_pos"]
        joint_vel = snapshot["joint_vel"]
        if zero_velocity:
            joint_vel = torch.zeros_like(joint_vel)
        asset.write_joint_state_to_sim(joint_pos, joint_vel)
        joint_pos_target = snapshot.get("joint_pos_target", joint_pos)
        asset.set_joint_position_target(joint_pos_target)
        joint_vel_target = snapshot.get("joint_vel_target")
        if joint_vel_target is not None and hasattr(asset, "set_joint_velocity_target"):
            if zero_velocity:
                joint_vel_target = torch.zeros_like(joint_vel_target)
            asset.set_joint_velocity_target(joint_vel_target)
        joint_effort_target = snapshot.get("joint_effort_target")
        if joint_effort_target is not None and hasattr(asset, "set_joint_effort_target"):
            asset.set_joint_effort_target(joint_effort_target)

    def _snapshot_switch_preserved_articulations() -> dict[str, dict[str, torch.Tensor]]:
        snapshots: dict[str, dict[str, torch.Tensor]] = {}
        for asset_name in _switch_preserved_articulation_names():
            snapshot = _snapshot_articulation_asset(asset_name)
            if snapshot is not None:
                snapshots[asset_name] = snapshot
        print(
            f"[CommonRobotSwitch] event=preserve_articulations assets={sorted(snapshots)}",
            flush=True,
        )
        return snapshots

    def _restore_switch_preserved_articulations(
        snapshots: dict[str, dict[str, torch.Tensor]],
        *,
        zero_velocity: bool = True,
    ) -> None:
        for asset_name, snapshot in snapshots.items():
            _restore_articulation_asset_snapshot(asset_name, snapshot, zero_velocity=zero_velocity)

    def _restore_switch_preserved_shared_assets(
        snapshots: dict[str, dict],
        *,
        zero_velocity: bool = True,
    ) -> None:
        _restore_switch_preserved_rigid_assets(
            snapshots.get("rigid_assets", {}),
            zero_velocity=zero_velocity,
        )
        _restore_switch_preserved_articulations(
            snapshots.get("articulations", {}),
            zero_velocity=zero_velocity,
        )

    def _begin_switch_preserve_shared_assets() -> dict[str, dict]:
        snapshots = {
            "rigid_assets": _snapshot_switch_preserved_rigid_assets(),
            "articulations": _snapshot_switch_preserved_articulations(),
        }
        if snapshots["rigid_assets"] or snapshots["articulations"]:
            setattr(
                env,
                "_allstar_restore_preserved_shared_assets",
                lambda snapshots=snapshots: _restore_switch_preserved_shared_assets(snapshots),
            )
            _restore_switch_preserved_shared_assets(snapshots)
        return snapshots

    def _end_switch_preserve_shared_assets(
        snapshots: dict[str, dict],
    ) -> None:
        if hasattr(env, "_allstar_restore_preserved_shared_assets"):
            delattr(env, "_allstar_restore_preserved_shared_assets")
        if snapshots.get("rigid_assets") or snapshots.get("articulations"):
            _restore_switch_preserved_shared_assets(snapshots)

    def _resolve_active_curobo_profile(robot_usd_path: str) -> tuple[Path, Path, str, str, list[str], list[float], float | None, list[float]]:
        saved_urdf_path = args_cli.urdf_path
        args_cli.urdf_path = None
        try:
            resolved = _resolve_curobo_profile(robot_usd_path)
        finally:
            args_cli.urdf_path = saved_urdf_path
        active_urdf_path, active_robot_cfg_path, active_base_link, active_tool_frame, active_joint_names, active_quat, active_roll, active_pre_target_deg = resolved
        if args_cli.disable_auto_curobo_robot_cfg:
            print(f"[CuroboRobotCfg] auto regeneration disabled; using {active_robot_cfg_path}", flush=True)
        else:
            active_robot_cfg_path = _regenerate_curobo_robot_cfg_from_urdf(
                active_urdf_path,
                active_robot_cfg_path,
                collision_sphere_buffer=args_cli.curobo_collision_sphere_buffer,
                base_link=active_base_link,
                tool_frame=active_tool_frame,
                joint_names=active_joint_names,
            )
        return (
            active_urdf_path,
            active_robot_cfg_path,
            active_base_link,
            active_tool_frame,
            active_joint_names,
            active_quat,
            active_roll,
            active_pre_target_deg,
        )

    common_direct_env_step_installed = False

    def _install_common_direct_env_step() -> None:
        nonlocal common_direct_env_step_installed
        if common_direct_env_step_installed:
            return
        setattr(env, "_allstar_manager_env_step", env.step)

        def direct_env_step(action=None):
            del action
            _step(env, render=not args_cli.headless)
            return None

        env.step = direct_env_step
        common_direct_env_step_installed = True

    def _set_active_robot_context(
        robot_id: str,
        *,
        move_to_work_pose: bool = True,
        rebuild_robot: bool = False,
    ) -> None:
        nonlocal active_robot_id, robot, arm_ids, urdf_path, robot_cfg_path
        nonlocal base_link, tool_frame, joint_names, pre_target_q, default_target_quat_wxyz
        profile = common_robot_profiles[robot_id]
        active_robot = _replace_scene_robot(robot_id) if rebuild_robot else env.scene["robot"]
        if move_to_work_pose:
            root_pose = _active_work_root_pose(active_robot)
            active_robot.write_root_pose_to_sim(root_pose)
            active_robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=active_robot.device, dtype=root_pose.dtype))

        robot_usd_path = _profile_usd_path(env.cfg, robot_id)
        (
            active_urdf_path,
            active_robot_cfg_path,
            active_base_link,
            active_tool_frame,
            active_joint_names,
            active_quat,
            _active_roll,
            active_pre_target_deg,
        ) = _resolve_active_curobo_profile(robot_usd_path)
        robot = active_robot
        setattr(robot, "_allstar_direct_tool_control", True)
        setattr(robot, "_allstar_env_action_robot", robot)
        setattr(env, "_allstar_active_tool_robot", robot)
        active_base_stabilizer = _install_static_motion_base_stabilizer(env, robot, patch_env_step=False)
        active_base_stabilizer.apply("active_context")
        active_robot_id = robot_id
        urdf_path = str(active_urdf_path)
        args_cli.urdf_path = str(active_urdf_path)
        robot_cfg_path = active_robot_cfg_path
        base_link = active_base_link
        tool_frame = active_tool_frame
        joint_names = active_joint_names
        arm_ids = _arm_joint_ids(robot)
        default_target_quat_wxyz = list(active_quat)
        _validate_joint_value_count(active_pre_target_deg, name=f"{robot_id}.pre_target_joint_pos_deg", joint_names=joint_names)
        pre_target_q = _deg_to_rad_joint_values(active_pre_target_deg, device=robot.device, dtype=robot.data.joint_pos.dtype)
        print(
            f"[CommonRobot] active_robot_id={active_robot_id} asset=robot "
            f"robot_type={profile['robot_type']} usd={robot_usd_path} urdf={urdf_path} "
            f"tools={'brush+common' if profile['robot_type'] == 'brush' else 'gripper+common'} "
            "collision_state=authored",
            flush=True,
        )

    def _active_profile() -> dict:
        if not _common_profile_enabled():
            return {"robot_type": "gripper" if _is_prismatic_gripper_joint_profile() else "brush"}
        return common_robot_profiles[active_robot_id]

    def _require_active_robot_type(required_type: str, tool_name: str) -> None:
        if not _common_profile_enabled():
            return
        active_type = str(_active_profile().get("robot_type", ""))
        if active_type != required_type:
            raise ValueError(
                f"tool {tool_name} requires robot_type={required_type}, "
                f"active_robot_id={active_robot_id} active_robot_type={active_type}"
            )

    def _wrap_tool_type(required_type: str, tool_name: str, fn):
        def wrapped(parameters: dict) -> dict:
            _require_active_robot_type(required_type, tool_name)
            return fn(parameters)

        return wrapped

    def _switch_ready_q() -> torch.Tensor:
        return _deg_to_rad_joint_values(
            list(_BASE_TELEOP_PRE_ARM_JOINT_POS_DEG),
            device=robot.device,
            dtype=robot.data.joint_pos.dtype,
        )

    def _move_active_arm_to_switch_ready(case_name: str, phase_name: str) -> dict:
        target_q = _switch_ready_q()
        moved = _plan_and_execute_cspace_target(
            env=env,
            robot=robot,
            case_name=case_name,
            phase_name=phase_name,
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            scene_mesh_obstacles=scene_mesh_obstacles,
            target_q=target_q,
            hold_gripper_joint1_pos=_current_gripper_joint1_pos_or_none(),
            final_settle=True,
        )
        actual_q = robot.data.joint_pos[0, arm_ids].detach()
        _, err_max, err = _joint_error(target_q, actual_q)
        fallback_steps = 0
        if not moved:
            fallback_steps, _ = _execute_joint_target_until_converged(
                env,
                robot,
                target_q,
                max_steps=max(int(args_cli.max_steps_per_waypoint), 300),
                min_steps=max(1, int(args_cli.steps_per_waypoint)),
                label=f"{case_name}:{phase_name}:direct_switch_ready_fallback",
                gripper_joint1_pos=_current_gripper_joint1_pos_or_none(),
            )
            actual_q = robot.data.joint_pos[0, arm_ids].detach()
            _, err_max, err = _joint_error(target_q, actual_q)
        return {
            "moved": bool(moved) or err_max.item() <= float(args_cli.joint_target_tolerance),
            "planned": bool(moved),
            "fallback_steps": int(fallback_steps),
            "target_q": target_q.detach().cpu().tolist(),
            "actual_q": actual_q.detach().cpu().tolist(),
            "joint_err_max": float(err_max.item()),
            "joint_err": err.detach().cpu().tolist(),
        }

    if _common_profile_enabled():
        _set_active_robot_context(active_robot_id, move_to_work_pose=False)
        _install_common_direct_env_step()
    # HDF5 demo 录制：在所有 env.step 包装（video/stabilizer/direct）之后套最外层，
    # 这样 direct 与普通两种模式都能逐帧采集，且不会重复计数。
    _install_demo_recording_step(env)

    def _current_gripper_joint1_pos_or_none() -> float | None:
        state = _gripper_joint1_state(robot)
        if state is None:
            return None
        return float(state[0])

    def base_keyboard_teleop_plan_tool(parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            raise ValueError("base_keyboard_teleop parameters must be a JSON object.")
        case_name = str(parameters.get("name", "base_keyboard_teleop"))
        joint_pos_deg = _float_list_parameter(parameters, "pre_arm_joint_pos_deg", 6, required=False)
        if joint_pos_deg is None:
            joint_pos_deg = list(_BASE_TELEOP_PRE_ARM_JOINT_POS_DEG)
        pre_teleop_q = _deg_to_rad_joint_values(
            joint_pos_deg,
            device=robot.device,
            dtype=robot.data.joint_pos.dtype,
        )
        payload = _build_cspace_tool_payload(
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            q_current=robot.data.joint_pos[0, arm_ids].detach(),
            scene_mesh_obstacles=scene_mesh_obstacles,
            cfg=planner_cfg,
            cspace_targets=[
                {
                    "name": f"{case_name}_pre_teleop_pose",
                    "joint_pos": pre_teleop_q.detach().cpu().tolist(),
                },
            ],
        )
        print(
            f"[BaseKeyboardTeleop] case={case_name} planning_pre_arm_q_deg={joint_pos_deg}",
            flush=True,
        )
        output = planner_tool(payload)
        output = dict(output)
        output["base_keyboard_teleop"] = {
            "case_name": case_name,
            "pre_arm_joint_pos_deg": list(joint_pos_deg),
            "pre_arm_joint_pos": pre_teleop_q.detach().cpu().tolist(),
            "speed_scale": float(parameters.get("speed_scale", _BASE_TELEOP_DEFAULT_SPEED_SCALE)),
            "max_duration_s": float(parameters.get("max_duration_s", _BASE_TELEOP_DEFAULT_MAX_DURATION_S)),
            "stop_key": str(parameters.get("stop_key", "Q")).strip().upper()[:1] or "Q",
            "keyboard_hold_sec": float(parameters.get("keyboard_hold_sec", 0.12)),
        }
        return output

    def execute_base_keyboard_teleop(tool_output: dict, parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            parameters = {}
        meta = dict(tool_output.get("base_keyboard_teleop", {}))
        case_name = str(meta.get("case_name", parameters.get("name", "base_keyboard_teleop")))
        pre_arm_q = torch.tensor(
            meta.get("pre_arm_joint_pos", _deg_to_rad_joint_values(
                list(_BASE_TELEOP_PRE_ARM_JOINT_POS_DEG),
                device=robot.device,
                dtype=robot.data.joint_pos.dtype,
            ).detach().cpu().tolist()),
            device=robot.device,
            dtype=robot.data.joint_pos.dtype,
        ).reshape(-1)
        results = tool_output.get("results", [])
        result = results[0] if results else {}
        if not result.get("success") or not result.get("trajectory"):
            print(
                f"[BaseKeyboardTeleop] pre-teleop C-space plan unavailable; not entering base teleop. "
                f"result={result}",
                flush=True,
            )
            return {"executed": False, "error": "pre-teleop C-space plan unavailable", "target_result": result}

        gripper_hold_q = _current_gripper_joint1_pos_or_none()
        pre_traj = torch.tensor(result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
        if pre_traj.dim() > 2:
            pre_traj = pre_traj.reshape(-1, pre_traj.shape[-1])
        _execute_trajectory(
            env,
            robot,
            pre_traj,
            gripper_joint1_pos=gripper_hold_q,
            label=f"{case_name}:pre_teleop_pose",
        )
        if args_cli.final_hold_steps > 0:
            _hold_joint_target_if_close(
                env,
                robot,
                pre_arm_q,
                args_cli.final_hold_steps,
                label=f"{case_name}:pre_teleop_hold",
                gripper_joint1_pos=gripper_hold_q,
            )
        actual_pre_q = robot.data.joint_pos[0, arm_ids].detach()
        _, pre_err_max, pre_err = _joint_error(pre_arm_q, actual_pre_q)
        # 放宽 base_keyboard_teleop 进入检查阈值：0.15rad（原 joint_target_tolerance=0.005rad 太严格）
        _base_teleop_enter_tolerance = 0.15
        if pre_err_max.item() > _base_teleop_enter_tolerance:
            print(
                f"[BaseKeyboardTeleop] pre-teleop arm target not reached; not entering base teleop. "
                f"target_q={_format_tensor(pre_arm_q)} actual_q={_format_tensor(actual_pre_q)} "
                f"joint_err_max={pre_err_max.item():.6f} joint_err={_format_tensor(pre_err)}",
                flush=True,
            )
            return {
                "executed": False,
                "error": "pre-teleop arm target not reached",
                "pre_arm_joint_pos": pre_arm_q.detach().cpu().tolist(),
                "actual_arm_joint_pos": actual_pre_q.detach().cpu().tolist(),
                "arm_joint_err_max": float(pre_err_max.item()),
                "arm_joint_err": pre_err.detach().cpu().tolist(),
            }

        speed_scale = float(meta.get("speed_scale", parameters.get("speed_scale", _BASE_TELEOP_DEFAULT_SPEED_SCALE)))
        max_duration_s = float(
            meta.get("max_duration_s", parameters.get("max_duration_s", _BASE_TELEOP_DEFAULT_MAX_DURATION_S))
        )
        stop_key = str(meta.get("stop_key", parameters.get("stop_key", "Q"))).strip().upper()[:1] or "Q"
        keyboard_hold_sec = float(meta.get("keyboard_hold_sec", parameters.get("keyboard_hold_sec", 0.12)))
        teleop = _create_teleop("keyboard", keyboard_hold_sec)
        motion_based_wheel_visualizer = _MotionBasedWheelVisualizer(robot, write_joint_state=True)
        motion_based_wheel_visualizer.sync_from_sim()
        motion_base_nominal_z = robot.data.root_link_pose_w[:, 2].clone()
        motion_base_yaw_target = _quat_wxyz_to_yaw_tensor(robot.data.root_link_pose_w[:, 3:7]).clone()
        last_base_pose_w, _ = _make_upright_root_pose(
            robot.data.root_link_pose_w.clone(),
            motion_base_nominal_z,
            motion_base_yaw_target,
        )
        setattr(robot, "_allstar_motion_base_teleop_active", True)
        start_wall = time.monotonic()
        env_step_count = 0
        last_report_wall = start_wall
        exit_reason = "stop_key"
        final_wheel_vel = torch.zeros((1, 4), device=robot.device, dtype=robot.data.joint_pos.dtype)
        print(
            f"[BaseKeyboardTeleop] case={case_name} entered motion-based keyboard base control "
            f"keys=W/S,A/D,Z/X stop_key={stop_key} speed_scale={speed_scale:.3f} "
            f"max_duration_s={max_duration_s:.3f}",
            flush=True,
        )
        try:
            while simulation_app.is_running():
                now_wall = time.monotonic()
                elapsed_s = now_wall - start_wall
                if max_duration_s > 0.0 and elapsed_s >= max_duration_s:
                    exit_reason = "max_duration_s"
                    break
                if stop_key and teleop.is_held(stop_key):
                    exit_reason = "stop_key"
                    break

                sensitivity = float(getattr(teleop, "SENSITIVITY", 0.05))
                vx = teleop.get_axis("W", "S") * sensitivity * speed_scale
                vy = teleop.get_axis("A", "D") * sensitivity * speed_scale
                wz = teleop.get_axis("Z", "X") * sensitivity * speed_scale
                vx_body, vy_body, wz_body = _legacy_operator_to_motion_based_twist(vx, vy, wz)
                final_wheel_vel, motion_base_yaw_target, last_base_pose_w = _apply_motion_based_base(
                    robot,
                    vx_body,
                    vy_body,
                    wz_body,
                    env.physics_dt,
                    env.device,
                    last_base_pose_w,
                    motion_base_nominal_z,
                    motion_base_yaw_target,
                )
                motion_based_wheel_visualizer.apply(final_wheel_vel, env.physics_dt)
                _sync_link5_camera(env, robot)
                _zero_arm_velocity_target(robot)
                _set_arm_effort_target(robot)
                action = _build_action(robot, pre_arm_q, gripper_joint1_pos=gripper_hold_q)
                env.step(action)
                _sync_link5_camera(env, robot)
                if not args_cli.headless:
                    _show_camera_windows(env)
                env_step_count += 1

                if now_wall - last_report_wall >= 1.0:
                    last_report_wall = now_wall
                    base_pose = robot.data.root_link_pose_w[0].detach().cpu().tolist()
                    print(
                        f"[BaseKeyboardTeleop] case={case_name} t={elapsed_s:.1f}s "
                        f"legacy_cmd=({vx:.3f},{vy:.3f},{wz:.3f}) "
                        f"body_twist=({vx_body:.3f},{vy_body:.3f},{wz_body:.3f}) "
                        f"base_xy_yaw=({base_pose[0]:.3f},{base_pose[1]:.3f},{float(motion_base_yaw_target[0].item()):.3f})",
                        flush=True,
                    )
        finally:
            setattr(robot, "_allstar_motion_base_teleop_active", False)
            teleop.close()
            motion_based_wheel_visualizer.apply(
                torch.zeros_like(final_wheel_vel),
                env.physics_dt,
            )
            _hold_base_stationary(robot)

        final_arm_q = robot.data.joint_pos[0, arm_ids].detach()
        _, arm_err_max, arm_err = _joint_error(pre_arm_q, final_arm_q)
        final_base_pose = robot.data.root_link_pose_w[0].detach().cpu().tolist()
        return {
            "executed": True,
            "pre_teleop_executed": True,
            "exit_reason": exit_reason,
            "env_step_count": env_step_count,
            "speed_scale": speed_scale,
            "max_duration_s": max_duration_s,
            "stop_key": stop_key,
            "final_base_root_pose_w": final_base_pose,
            "pre_arm_joint_pos": pre_arm_q.detach().cpu().tolist(),
            "final_arm_joint_pos": final_arm_q.detach().cpu().tolist(),
            "arm_joint_err_max": float(arm_err_max.item()),
            "arm_joint_err": arm_err.detach().cpu().tolist(),
        }

    def brush_execute_point_clean_plan_tool(parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            raise ValueError("brush_execute_point_clean parameters must be a JSON object.")
        case_name = str(parameters.get("name", "brush_execute_point_clean"))
        target_pos = _float_list_parameter(parameters, "target_pos", 3, required=True)
        target_pos, safe_clearance_m = _apply_brush_target_safe_clearance(target_pos)
        target_quat = _float_list_parameter(parameters, "target_quat", 4, required=False)
        target_quat_defaulted = target_quat is None
        if target_quat_defaulted:
            target_quat = _default_brush_target_quat_wxyz(target_pos)
        target_frame = str(parameters.get("target_frame", "world")).lower()
        if target_frame not in {"world", "base"}:
            raise ValueError("brush_execute_point_clean target_frame must be either 'world' or 'base'.")
        pose_test = {
            "name": case_name,
            "target_pos": target_pos,
            "target_quat": target_quat,
            "input_target_frame": target_frame,
            "target_quat_defaulted": target_quat_defaulted,
            "safe_clearance_m": safe_clearance_m,
        }

        if args_cli.use_pre_target_joint_pose:
            print(
                f"[BrushRolling0512] case={case_name} moving_to_planning_q={_format_tensor(pre_target_q)} "
                "via cuRobo cspace",
                flush=True,
            )
            moved_to_pre_target = _plan_and_execute_cspace_target(
                env=env,
                robot=robot,
                case_name=case_name,
                phase_name="pre_target",
                urdf_path=urdf_path,
                robot_cfg_path=robot_cfg_path,
                base_link=base_link,
                tool_frame=tool_frame,
                joint_names=joint_names,
                scene_mesh_obstacles=scene_mesh_obstacles,
                target_q=pre_target_q,
                final_settle=True,
            )
            if not moved_to_pre_target:
                raise RuntimeError(f"Failed to reach pre-target pose with cuRobo for case={case_name}.")

        arm_base_pos_w, arm_base_quat_w = _body_pose_w(robot, base_link.replace("-", "_"))
        if target_frame == "world":
            target_pos_w = torch.tensor(target_pos, device=robot.device, dtype=torch.float32).reshape(1, 3)
            target_quat_w = torch.tensor(target_quat, device=robot.device, dtype=torch.float32).reshape(1, 4)
            target_pos_base, target_quat_base = subtract_frame_transforms(
                arm_base_pos_w.reshape(1, 3),
                arm_base_quat_w.reshape(1, 4),
                target_pos_w,
                target_quat_w,
            )
            converted_pose_test = dict(pose_test)
            converted_pose_test["target_pos"] = target_pos_base.reshape(-1).detach().cpu().tolist()
            converted_pose_test["target_quat"] = target_quat_base.reshape(-1).detach().cpu().tolist()
        else:
            converted_pose_test = dict(pose_test)
            if target_quat_defaulted:
                target_pos_base = torch.tensor(target_pos, device=robot.device, dtype=torch.float32).reshape(1, 3)
                target_quat_w = torch.tensor(target_quat, device=robot.device, dtype=torch.float32).reshape(1, 4)
                target_pos_w, _ = combine_frame_transforms(
                    arm_base_pos_w.reshape(1, 3),
                    arm_base_quat_w.reshape(1, 4),
                    target_pos_base,
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=robot.device, dtype=torch.float32),
                )
                _, target_quat_base = subtract_frame_transforms(
                    arm_base_pos_w.reshape(1, 3),
                    arm_base_quat_w.reshape(1, 4),
                    target_pos_w,
                    target_quat_w,
                )
                converted_pose_test["target_quat"] = target_quat_base.reshape(-1).detach().cpu().tolist()

        payload = _build_pose_tool_payload(
            parameters,
            pose_test=converted_pose_test,
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            q_current=robot.data.joint_pos[0, arm_ids].detach(),
            scene_mesh_obstacles=scene_mesh_obstacles,
            cfg=planner_cfg,
        )
        return planner_tool(payload)

    def execute_brush_execute_point_clean(tool_output: dict, parameters: dict) -> dict:
        del parameters
        return _execute_point_clean_results(
            tool_output,
            env=env,
            robot=robot,
            pre_target_q=pre_target_q,
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            scene_mesh_obstacles=scene_mesh_obstacles,
            pre_target_executed=args_cli.use_pre_target_joint_pose,
            execute_point_clean_trajectory_fn=_execute_brush_execute_point_clean_trajectory,
        )

    def brush_execute_spot_clean_plan_tool(parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            raise ValueError("brush_execute_spot_clean parameters must be a JSON object.")
        case_name = str(parameters.get("name", "brush_execute_spot_clean"))
        target_frame = str(parameters.get("target_frame", "world")).lower()
        if target_frame not in {"world", "base"}:
            raise ValueError("brush_execute_spot_clean target_frame must be either 'world' or 'base'.")
        sampled_pose_tests = _parse_spot_clean_pose_tests(
            parameters,
            case_name=case_name,
            float_list_parameter_fn=_float_list_parameter,
            downsample_ordered_pose_tests_fn=_downsample_ordered_pose_tests,
            max_count=8,
        )

        if args_cli.use_pre_target_joint_pose:
            print(
                f"[BrushSpotClean] case={case_name} moving_to_planning_q={_format_tensor(pre_target_q)} "
                "via cuRobo cspace",
                flush=True,
            )
            moved_to_pre_target = _plan_and_execute_cspace_target(
                env=env,
                robot=robot,
                case_name=case_name,
                phase_name="pre_target",
                urdf_path=urdf_path,
                robot_cfg_path=robot_cfg_path,
                base_link=base_link,
                tool_frame=tool_frame,
                joint_names=joint_names,
                scene_mesh_obstacles=scene_mesh_obstacles,
                target_q=pre_target_q,
                final_settle=True,
            )
            if not moved_to_pre_target:
                raise RuntimeError(f"Failed to reach pre-target pose with cuRobo for case={case_name}.")

        target_poses_base = []
        for target_idx, (target_name, target_pos, target_quat) in enumerate(sampled_pose_tests):
            target = {
                "name": target_name,
                "target_frame": target_frame,
                "target_pos": target_pos,
            }
            if target_quat:
                target["target_quat"] = target_quat
            target_poses_base.append(
                _convert_brush_target_to_base(
                    robot=robot,
                    base_link=base_link,
                    target=target,
                    default_name=f"{case_name}_target_{target_idx + 1:02d}",
                    force_safe_clearance=True,
                )
            )

        result = _plan_multi_pose_polynomial_seed(
            case_name=case_name,
            env=env,
            robot=robot,
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            q_start=robot.data.joint_pos[0, arm_ids].detach().clone(),
            scene_mesh_obstacles=scene_mesh_obstacles,
            target_poses_base=target_poses_base,
            seed_mode=str(parameters.get("seed_mode", "cubic")),
        )
        return {
            "results": [result],
            "spot_clean_final_target_base": target_poses_base[-1],
        }

    def execute_brush_execute_spot_clean(tool_output: dict, parameters: dict) -> dict:
        return _execute_spot_clean_results(
            tool_output,
            parameters,
            env=env,
            robot=robot,
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            tool_frame=tool_frame,
            joint_names=joint_names,
            scene_mesh_obstacles=scene_mesh_obstacles,
            base_link=base_link,
            pre_target_executed=args_cli.use_pre_target_joint_pose,
            execute_spot_clean_trajectory_fn=_execute_brush_execute_spot_clean_trajectory,
        )

    def stretch_end_tool_plan_tool(parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            raise ValueError("stretch_end_tool parameters must be a JSON object.")
        case_name = str(parameters.get("name", "stretch_end_tool"))
        joint_pos_deg = _float_list_parameter(parameters, "joint_pos_deg", 6, required=False)
        if joint_pos_deg is None:
            joint_pos_deg = [5.7, -14.4, -20.0, 0.0, 8.6, 0.0]
        stretch_q = _deg_to_rad_joint_values(
            joint_pos_deg,
            device=robot.device,
            dtype=robot.data.joint_pos.dtype,
        )
        planning_q = pre_target_q.detach().clone()
        payload = _build_cspace_tool_payload(
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            q_current=robot.data.joint_pos[0, arm_ids].detach(),
            scene_mesh_obstacles=scene_mesh_obstacles,
            cfg=planner_cfg,
            cspace_targets=[
                {
                    "name": f"{case_name}_stretch",
                    "joint_pos": stretch_q.detach().cpu().tolist(),
                },
                {
                    "name": f"{case_name}_planning_pose",
                    "joint_pos": planning_q.detach().cpu().tolist(),
                },
            ],
        )
        print(
            f"[StretchEndTool] case={case_name} stretch_q_deg={joint_pos_deg} "
            f"planning_q={_format_tensor(planning_q)}",
            flush=True,
        )
        return planner_tool(payload)

    def execute_stretch_end_tool(tool_output: dict, parameters: dict) -> dict:
        del parameters
        return _execute_planning_pose_result_trajectories(
            tool_output,
            env=env,
            robot=robot,
            pre_target_q=pre_target_q,
            steps_per_waypoint=args_cli.steps_per_waypoint,
            final_hold_steps=args_cli.final_hold_steps,
            execute_trajectory_fn=_execute_trajectory,
            hold_joint_target_if_close_fn=_hold_joint_target_if_close,
            arm_joint_ids_fn=_arm_joint_ids,
            joint_error_fn=_joint_error,
            trajectory_label_suffix="stretch_end_tool",
            hold_label="stretch_end_tool:planning_pose_hold",
        )

    def brush_move_arm_to_joints_plan_tool(parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            raise ValueError("brush_move_arm_to_joints parameters must be a JSON object.")
        case_name = str(parameters.get("name", "brush_move_arm_to_joints"))
        planning_q = pre_target_q.detach().clone()
        payload = _build_cspace_tool_payload(
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            q_current=robot.data.joint_pos[0, arm_ids].detach(),
            scene_mesh_obstacles=scene_mesh_obstacles,
            cfg=planner_cfg,
            cspace_targets=[
                {
                    "name": f"{case_name}_planning_pose",
                    "joint_pos": planning_q.detach().cpu().tolist(),
                },
            ],
        )
        print(
            f"[BrushMoveArmToJoints] case={case_name} planning_q={_format_tensor(planning_q)}",
            flush=True,
        )
        try:
            return planner_tool(payload)
        except Exception as exc:  # noqa: BLE001 - fallback is handled by the executor.
            print(
                f"[BrushMoveArmToJoints] case={case_name} cuRobo planner failed; "
                f"will use direct joint fallback. error={exc}",
                flush=True,
            )
            return {
                "results": [
                    {
                        "name": f"{case_name}_planning_pose",
                        "success": False,
                        "trajectory": [],
                        "error": str(exc),
                    }
                ],
                "planner_error": str(exc),
            }

    def execute_brush_move_arm_to_joints(tool_output: dict, parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            parameters = {}
        case_name = str(parameters.get("name", "brush_move_arm_to_joints"))
        result = _execute_planning_pose_result_trajectories(
            tool_output,
            env=env,
            robot=robot,
            pre_target_q=pre_target_q,
            steps_per_waypoint=args_cli.steps_per_waypoint,
            final_hold_steps=args_cli.final_hold_steps,
            execute_trajectory_fn=_execute_trajectory,
            hold_joint_target_if_close_fn=_hold_joint_target_if_close,
            arm_joint_ids_fn=_arm_joint_ids,
            joint_error_fn=_joint_error,
            trajectory_label_suffix="brush_move_arm_to_joints",
            hold_label="brush_move_arm_to_joints:planning_pose_hold",
            log_result_prefix="BrushMoveArmToJointsResult",
            no_executable_message=(
                "[BrushMoveArmToJoints] cuRobo returned no executable C-space trajectory; "
                "falling back to direct joint move to planning pose."
            ),
            include_fallback=True,
            require_nonempty_2d=True,
        )
        if result.get("executed"):
            return result

        fallback_steps, fallback_err_max = _execute_direct_joint_roll(
            env=env,
            robot=robot,
            target_q=pre_target_q,
            label=f"{case_name}:direct_joint_fallback_planning_pose",
            final_hold_steps=args_cli.final_hold_steps,
        )
        actual_q = robot.data.joint_pos[0, arm_ids].detach()
        _, err_max, err = _joint_error(pre_target_q, actual_q)
        result.update(
            {
                "executed": True,
                "returned_to_planning_pose": True,
                "fallback_executed": True,
                "fallback_reason": "curobo cspace plan unavailable",
                "fallback_env_step_count": int(fallback_steps),
                "env_step_count": int(result.get("env_step_count", 0)) + int(fallback_steps),
                "final_joint_err_max": float(err_max.item()),
                "fallback_joint_err_max": float(fallback_err_max),
                "final_joint_err": err.detach().cpu().tolist(),
            }
        )
        return result

    def execute_spot_capture_plan_tool(parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            raise ValueError("execute_spot_capture parameters must be a JSON object.")
        case_name = str(parameters.get("name", "spot_capture"))
        joint_pos_deg = _float_list_parameter(parameters, "joint_pos_deg", 6, required=False)
        if joint_pos_deg is None:
            joint_pos_deg = [2.8, -2.9, -25.7, 0.0, -94.5, 0.0]
        capture_q = _deg_to_rad_joint_values(
            joint_pos_deg,
            device=robot.device,
            dtype=robot.data.joint_pos.dtype,
        )
        planning_q = pre_target_q.detach().clone()
        payload = _build_cspace_tool_payload(
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            q_current=robot.data.joint_pos[0, arm_ids].detach(),
            scene_mesh_obstacles=scene_mesh_obstacles,
            cfg=planner_cfg,
            cspace_targets=[
                {
                    "name": f"{case_name}_capture_pose",
                    "joint_pos": capture_q.detach().cpu().tolist(),
                },
                {
                    "name": f"{case_name}_planning_pose",
                    "joint_pos": planning_q.detach().cpu().tolist(),
                },
            ],
        )
        print(
            f"[ExecuteSpotCapture] case={case_name} capture_q_deg={joint_pos_deg} "
            f"planning_q={_format_tensor(planning_q)}",
            flush=True,
        )
        return planner_tool(payload)

    def execute_execute_spot_capture(tool_output: dict, parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            parameters = {}
        case_name = str(parameters.get("name", "spot_capture"))
        joint_pos_deg = _float_list_parameter(parameters, "joint_pos_deg", 6, required=False)
        if joint_pos_deg is None:
            joint_pos_deg = [2.8, -2.9, -25.7, 0.0, -94.5, 0.0]
        capture_q = _deg_to_rad_joint_values(
            joint_pos_deg,
            device=robot.device,
            dtype=robot.data.joint_pos.dtype,
        )
        results = tool_output.get("results", [])
        capture_result = results[0] if len(results) >= 1 else {}
        return_result = results[1] if len(results) >= 2 else {}
        env_step_count = 0
        waypoint_count = 0

        if capture_result.get("success") and capture_result.get("trajectory"):
            capture_traj = torch.tensor(
                capture_result["trajectory"],
                device=robot.device,
                dtype=robot.data.joint_pos.dtype,
            )
            if capture_traj.dim() > 2:
                capture_traj = capture_traj.reshape(-1, capture_traj.shape[-1])
            _execute_trajectory(env, robot, capture_traj, label=f"{case_name}:capture")
            waypoint_count += int(capture_traj.shape[0])
            env_step_count += int(capture_traj.shape[0]) * max(1, args_cli.steps_per_waypoint)
        else:
            print(
                f"[ExecuteSpotCapture] capture cuRobo plan unavailable; not moving the arm. "
                f"result={capture_result}",
                flush=True,
            )
            return {
                "executed": False,
                "error": "capture cuRobo plan unavailable",
                "capture_result": capture_result,
                "return_result": return_result,
            }

        actual_capture_q = robot.data.joint_pos[0, arm_ids].detach()
        _, capture_err_tensor, _ = _joint_error(capture_q, actual_capture_q)
        capture_err = float(capture_err_tensor.item())
        capture_info = _save_spot_capture_frame(
            env,
            robot,
            case_name=case_name,
            arm_base_body=base_link.replace("-", "_"),
            target_q=capture_q,
        )

        returned_with_curobo = False
        if return_result.get("success") and return_result.get("trajectory"):
            return_traj = torch.tensor(
                return_result["trajectory"],
                device=robot.device,
                dtype=robot.data.joint_pos.dtype,
            )
            if return_traj.dim() > 2:
                return_traj = return_traj.reshape(-1, return_traj.shape[-1])
            _execute_trajectory(env, robot, return_traj, label=f"{case_name}:return")
            waypoint_count += int(return_traj.shape[0])
            env_step_count += int(return_traj.shape[0]) * max(1, args_cli.steps_per_waypoint)
            returned_with_curobo = True
        else:
            print(
                f"[ExecuteSpotCapture] return cuRobo plan unavailable; not moving the arm without cuRobo. "
                f"result={return_result}",
                flush=True,
            )

        actual_return_q = robot.data.joint_pos[0, arm_ids].detach()
        _, return_err_tensor, _ = _joint_error(pre_target_q, actual_return_q)
        return_err = float(return_err_tensor.item())
        if args_cli.final_hold_steps > 0:
            env_step_count += _hold_joint_target_if_close(
                env,
                robot,
                pre_target_q,
                args_cli.final_hold_steps,
                label="execute_spot_capture:return_hold",
            )

        return {
            "executed": True,
            "capture_path": capture_info["capture_path"],
            "rgb_shape": capture_info["rgb_shape"],
            "depth_shape": capture_info["depth_shape"],
            "cam_in_arm_base_tf": capture_info["cam_in_arm_base_tf"],
            "returned_to_planning_pose": True,
            "returned_with_curobo": returned_with_curobo,
            "waypoint_count": waypoint_count,
            "env_step_count": env_step_count,
            "capture_joint_err_max": capture_err,
            "return_joint_err_max": return_err,
        }

    def _try_plan_with_z_offsets(
        *,
        parameters,  # ← required: passed from caller scope
        target_pos_w_nominal,
        target_quat_w,
        arm_base_pos_w,
        arm_base_quat_w,
        target_frame,
        target_quat_defaulted,
        move_to_pre_target,
        case_name,
        q_start,
        scene_cuboid_obstacles=None,
    ):
        """GPT P0 v2: Pure retry — ALL attempts go through here, -Z offsets, sentinel prints."""
        print("[PaperRetry] ENTER _try_plan_with_z_offsets", flush=True)
        # The target paper's own proxy must NOT be an obstacle, or the gripper can never
        # envelope the clump (its inflated box sits right under the grasp pose). The caller
        # passes the target-excluded obstacle list.
        if scene_cuboid_obstacles is None:
            scene_cuboid_obstacles = _CUROBO_SCENE_CUBOID_OBSTACLES
        retry_cfg = planner_client_config_from_args(args_cli, scene_cuboid_obstacles=scene_cuboid_obstacles)
        # GPT fix: try LOWER heights (user says too high)
        z_offsets = [0.0, -0.01, -0.02]  # 0.115 → 0.105 → 0.095
        for attempt_idx, dz in enumerate(z_offsets):
            dx, dy = 0.0, 0.0
            retry_pos_w = torch.as_tensor(
                [[float(target_pos_w_nominal[0, 0]) + dx,
                  float(target_pos_w_nominal[0, 1]) + dy,
                  float(target_pos_w_nominal[0, 2]) + dz]],
                device=q_start.device, dtype=q_start.dtype,
            )
            retry_pos_base, retry_quat_base = subtract_frame_transforms(
                arm_base_pos_w.reshape(1, 3), arm_base_quat_w.reshape(1, 4),
                retry_pos_w, target_quat_w,
            )
            pose_test_i = {
                "name": f"{case_name}_z{attempt_idx}",
                "target_pos": retry_pos_base.reshape(-1).detach().cpu().tolist(),
                "target_quat": retry_quat_base.reshape(-1).detach().cpu().tolist(),
                "target_pos_world_z": float(retry_pos_w.reshape(-1)[2].item()),
                "input_target_frame": target_frame,
                "target_quat_defaulted": target_quat_defaulted,
                "move_to_pre_target": move_to_pre_target,
            }
            payload_i = _build_pose_tool_payload(
                parameters,
                pose_test=pose_test_i,
                urdf_path=urdf_path, robot_cfg_path=robot_cfg_path,
                base_link=base_link, tool_frame=tool_frame, joint_names=joint_names,
                q_current=q_start,
                scene_mesh_obstacles=scene_mesh_obstacles, cfg=retry_cfg,
            )
            # Stream through the isolated worker while the sim keeps stepping and holds
            # q_start — same path as pre_target. Blocking subprocess.run here froze the
            # GUI for the whole worker runtime; this keeps the viewport alive.
            output_i = dict(_run_pose_plan_worker(
                payload_i,
                env=env,
                robot=robot,
                hold_q=q_start,
                hold_gripper_joint1_pos=_default_open_gripper_joint1_pos(),
            ))
            result_i = output_i.get("results", [{}])[0] if output_i.get("results") else {}
            success_i = bool(result_i.get("success")) and bool(result_i.get("trajectory"))
            print(
                f"[PaperRetry] attempt={attempt_idx} dz={dz:.3f} "
                f"target_z={float(retry_pos_w[0,2]):.4f} "
                f"success={success_i} stage={result_i.get('metrics',{}).get('planner_failure_stage','?')}",
                flush=True,
            )
            if success_i:
                print(f"[PaperRetry] SUCCESS at attempt {attempt_idx}", flush=True)
                return {"output": output_i, "target_pos_w": retry_pos_w, "attempt": attempt_idx}
        print("[PaperRetry] all attempts exhausted", flush=True)
        return None

    def gripper_prismatic_execute_grab_plan_tool(parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            raise ValueError("gripper_prismatic_execute_grab parameters must be a JSON object.")
        case_name = str(parameters.get("name", "gripper_prismatic_execute_grab"))
        paper_cleanup_mode = bool(parameters.get("paper_cleanup_mode", False))
        paper_prim_path = str(parameters.get("paper_prim_path", "")) if paper_cleanup_mode else ""
        _g_timing_reset()
        _t_g = time.perf_counter()

        # paper_cleanup_mode: auto-detect grasp target position (grasp_test_cube_1)
        if paper_cleanup_mode:
            from pxr import UsdGeom, Usd
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if not paper_prim_path:
                paper_prim_path = "/World/envs/env_0/grasp_test_cube_1"

            paper_prim = stage.GetPrimAtPath(paper_prim_path)
            if paper_prim.IsValid():
                cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                world_mat = cache.GetLocalToWorldTransform(paper_prim)
                t = world_mat.ExtractTranslation()
                paper_pos = [float(t[0]), float(t[1]), float(t[2])]
                # 目标高度取物体上方 3cm（cube 顶 ~0.041 → 0.05，与原验证一致）
                target_pos = [paper_pos[0], paper_pos[1], max(paper_pos[2] + 0.03, 0.05)]
                print(
                    f"[PaperCleanup] case={case_name} paper_pos={paper_pos} target_pos={target_pos}",
                    flush=True,
                )
            else:
                raise ValueError(
                    f"paper_cleanup_mode=True but grasp target prim not found at {paper_prim_path}"
                )
        else:
            target_pos = _float_list_parameter(parameters, "target_pos", 3, required=True)

        target_quat = _float_list_parameter(parameters, "target_quat", 4, required=False)
        target_quat_defaulted = target_quat is None
        if target_quat_defaulted:
            target_quat = list(default_target_quat_wxyz)
        target_frame = str(parameters.get("target_frame", "world")).lower()
        if target_frame not in {"world", "base"}:
            raise ValueError("gripper_prismatic_execute_grab target_frame must be either 'world' or 'base'.")
        # 默认 False：不再先回规划起始位姿，直接规划到 target（方块上方）。传 true 可恢复。
        move_to_pre_target = bool(parameters.get("move_to_pre_target", False))

        if move_to_pre_target:
            print(
                f"[GripperPrismaticGrab] case={case_name} moving_to_pre_target={_format_tensor(pre_target_q)} "
                "via cuRobo cspace",
                flush=True,
            )
            moved_to_pre_target = _plan_and_execute_cspace_target(
                env=env,
                robot=robot,
                case_name=case_name,
                phase_name="pre_target_open",
                urdf_path=urdf_path,
                robot_cfg_path=robot_cfg_path,
                base_link=base_link,
                tool_frame=tool_frame,
                joint_names=joint_names,
                scene_mesh_obstacles=scene_mesh_obstacles,
                target_q=pre_target_q,
                hold_gripper_joint1_pos=_default_open_gripper_joint1_pos(),
                final_settle=True,
            )
            if not moved_to_pre_target:
                if paper_cleanup_mode:
                    print("[PaperCleanup] pre-target move failed; planning from current pose", flush=True)
                else:
                    raise RuntimeError(f"Failed to reach pre-target pose for case={case_name}.")
        _t_g = _g_timing_mark("pre_target_move", _t_g)

        arm_base_pos_w, arm_base_quat_w = _body_pose_w(robot, base_link.replace("-", "_"))
        if target_frame == "world":
            target_pos_w = torch.tensor(target_pos, device=robot.device, dtype=torch.float32).reshape(1, 3)
            target_quat_w = torch.tensor(target_quat, device=robot.device, dtype=torch.float32).reshape(1, 4)
            target_pos_base, target_quat_base = subtract_frame_transforms(
                arm_base_pos_w.reshape(1, 3),
                arm_base_quat_w.reshape(1, 4),
                target_pos_w,
                target_quat_w,
            )
        else:
            target_pos_base = torch.tensor(target_pos, device=robot.device, dtype=torch.float32).reshape(1, 3)
            if target_quat_defaulted:
                target_pos_w, _ = combine_frame_transforms(
                    arm_base_pos_w.reshape(1, 3),
                    arm_base_quat_w.reshape(1, 4),
                    target_pos_base,
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=robot.device, dtype=torch.float32),
                )
                target_quat_w = torch.tensor(target_quat, device=robot.device, dtype=torch.float32).reshape(1, 4)
                _, target_quat_base = subtract_frame_transforms(
                    arm_base_pos_w.reshape(1, 3),
                    arm_base_quat_w.reshape(1, 4),
                    target_pos_w,
                    target_quat_w,
                )
            else:
                target_quat_base = torch.tensor(target_quat, device=robot.device, dtype=torch.float32).reshape(1, 4)
                target_pos_w, target_quat_w = combine_frame_transforms(
                    arm_base_pos_w.reshape(1, 3),
                    arm_base_quat_w.reshape(1, 4),
                    target_pos_base,
                    target_quat_base,
                )

        pose_test = {
            "name": case_name,
            "target_pos": target_pos_base.reshape(-1).detach().cpu().tolist(),
            "target_quat": target_quat_base.reshape(-1).detach().cpu().tolist(),
            "target_pos_world_z": float(target_pos_w.reshape(-1)[2].item()),
            "input_target_frame": target_frame,
            "target_quat_defaulted": target_quat_defaulted,
            "move_to_pre_target": move_to_pre_target,
        }
        payload = _build_pose_tool_payload(
            parameters,
            pose_test=pose_test,
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            q_current=robot.data.joint_pos[0, arm_ids].detach(),
            scene_mesh_obstacles=scene_mesh_obstacles,
            cfg=planner_cfg,
        )
        _t_g = _g_timing_mark("target_pose_setup", _t_g)

        # Exclude target paper proxy from grasp plan (avoid self-collision).
        # The retry payloads must use this filtered list too, or the gripper can never
        # envelop the clump (its inflated proxy sits right under the grasp pose).
        filtered = _CUROBO_SCENE_CUBOID_OBSTACLES
        if paper_cleanup_mode and paper_prim_path:
            target_proxy_path = f"{paper_prim_path}/curobo_proxy"
            filtered = [c for c in _CUROBO_SCENE_CUBOID_OBSTACLES if target_proxy_path not in str(c.get("source_path", ""))]
            if len(filtered) < len(_CUROBO_SCENE_CUBOID_OBSTACLES):
                print(f"[PaperCleanup] excluded target proxy {target_proxy_path} from grasp plan ({len(_CUROBO_SCENE_CUBOID_OBSTACLES)}→{len(filtered)} obstacles)", flush=True)
        # GPT P0 v2: unified retry — all attempts through _try_plan_with_z_offsets
        print("[PaperRetryCallSite] BEFORE retry call", flush=True)
        q_snapshot = robot.data.joint_pos[0, arm_ids].detach().clone()
        if paper_cleanup_mode:
            retry_result = _try_plan_with_z_offsets(
                parameters=parameters,  # ← pass from plan tool scope
                target_pos_w_nominal=target_pos_w,
                target_quat_w=target_quat_w,
                arm_base_pos_w=arm_base_pos_w,
                arm_base_quat_w=arm_base_quat_w,
                target_frame=target_frame,
                target_quat_defaulted=target_quat_defaulted,
                move_to_pre_target=move_to_pre_target,
                case_name=case_name,
                q_start=q_snapshot,
                scene_cuboid_obstacles=filtered,
            )
            print("[PaperRetryCallSite] AFTER retry call, result=", retry_result is not None, flush=True)
            if retry_result is not None:
                output = retry_result["output"]
                target_pos_w = retry_result["target_pos_w"]
            else:
                # All retries failed
                _g_timing_mark("curobo_plan", _t_g)
                _g_timing_mark("plan_total", _t_g)
                _g_print_summary("GrabTimingSummary_FAILED")
                return {"executed": False, "error": "paper retry: all Z attempts failed"}
        else:
            output = planner_tool(payload)
            output = dict(output)
        _t_g = _g_timing_mark("curobo_plan", _t_g)
        output["gripper_prismatic_target"] = {
            "case_name": case_name,
            "target_frame": target_frame,
            "target_pos_w": target_pos_w.reshape(-1).detach().cpu().tolist(),
            "target_quat_wxyz": target_quat_w.reshape(-1).detach().cpu().tolist(),
            "target_pos_base": pose_test["target_pos"],
            "target_quat_base": pose_test["target_quat"],
            "move_to_pre_target": move_to_pre_target,
            "paper_cleanup_mode": paper_cleanup_mode,
            "paper_prim_path": paper_prim_path,
        }
        _g_timing_mark("plan_total", _t_g)
        return output

    def execute_gripper_prismatic_execute_grab(tool_output: dict, parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            parameters = {}
        case_name = str(parameters.get("name", "gripper_prismatic_execute_grab"))
        results = tool_output.get("results", [])
        result = results[0] if results else {}
        if not result.get("success") or not result.get("trajectory"):
            print(
                f"[GripperPrismaticGrab] target cuRobo plan unavailable; not moving the arm. result={result}",
                flush=True,
            )
            return {"executed": False, "error": "target cuRobo plan unavailable", "target_result": result}

        _t_g = time.perf_counter()
        target_traj = torch.tensor(result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
        if target_traj.dim() > 2:
            target_traj = target_traj.reshape(-1, target_traj.shape[-1])
        # Save for reverse-execution return (GPT review P0)
        global _LAST_GRAB_TRAJECTORY, _LAST_GRAB_GRIPPER_POS, _LAST_GRASPED_PAPER_PATH
        _LAST_GRAB_TRAJECTORY = target_traj.detach().clone()
        _LAST_GRAB_GRIPPER_POS = _default_open_gripper_joint1_pos()
        _execute_trajectory(
            env,
            robot,
            target_traj,
            gripper_joint1_pos=_default_open_gripper_joint1_pos(),
            label=f"{case_name}:target",
        )
        target_final_q = target_traj[-1].detach().clone()
        # DEBUG: q_goal 直接设置实验 — 打印 cuRobo planned 末端 vs Isaac 实际末端
        _tool_body_name = "tool_tip_link"
        _tool_ids = _find_optional_joints(robot, _tool_body_name)[0] if hasattr(robot, "body_names") else []
        if robot.body_names and _tool_body_name in robot.body_names:
            _tool_idx = robot.body_names.index(_tool_body_name)
            _actual_tool_pos = robot.data.body_pos_w[0, _tool_idx].detach().cpu().tolist()
        else:
            _actual_tool_pos = None
        print(
            f"[Debug_q_goal] case={case_name} q_goal={_format_tensor(target_final_q)} "
            f"actual_tool_pos_w={_actual_tool_pos}",
            flush=True,
        )
        if args_cli.final_hold_steps > 0:
            _hold_joint_target_if_close(
                env,
                robot,
                target_final_q,
                args_cli.final_hold_steps,
                label=f"{case_name}:target_hold",
                gripper_joint1_pos=_default_open_gripper_joint1_pos(),
            )

        _t_g = _g_timing_mark("target_traj_exec", _t_g)
        target_meta = tool_output.get("gripper_prismatic_target", {})
        target_pos_w = torch.tensor(
            target_meta.get("target_pos_w", [0.0, 0.0, 0.0]),
            device=robot.device,
            dtype=torch.float32,
        ).reshape(1, 3)
        target_quat_w = torch.tensor(
            target_meta.get("target_quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
            device=robot.device,
            dtype=torch.float32,
        ).reshape(1, 4)
        actual_target_pos_w, actual_target_quat_w = _tool_tip_pose_w(robot)
        target_pos_err_w, target_rot_err_rad_w, target_rot_err_deg_w = _pose_error(
            target_pos_w,
            target_quat_w,
            actual_target_pos_w.reshape(1, 3),
            actual_target_quat_w.reshape(1, 4),
        )
        # GPT P0-2: Diagnostic — planner vs actual pose error
        _planner_metrics = result.get("metrics", {})
        _planner_pos_err = _planner_metrics.get("planner_final_pos_error_m", None)
        _planner_tool_pos = _planner_metrics.get("planner_final_tool_pos", None)
        print(
            f"[GripperPrismaticGrab] case={case_name} target_reached "
            f"pos_err_m={target_pos_err_w.item():.6f} rot_err_deg={target_rot_err_deg_w.item():.3f} "
            f"planner_pos_err_m={_planner_pos_err} "
            f"planner_tool_pos_w={_planner_tool_pos} "
            f"actual_tool_pos_w={_actual_tool_pos} "
            f"target_pos_w={target_pos_w.reshape(-1).detach().cpu().tolist()}",
            flush=True,
        )

        # GPT P0-2: Block descend when actual pose error is too large
        _max_pregrasp_err = 0.06  # 6cm — relaxed for non-primary grasp targets
        _paper_mode = bool(target_meta.get("paper_cleanup_mode", False))
        if _paper_mode and target_pos_err_w.item() > _max_pregrasp_err:
            print(
                f"[GripperPrismaticGrab] ABORT descend: actual_pos_err_m={target_pos_err_w.item():.6f} > {_max_pregrasp_err}",
                flush=True,
            )
            return {
                "executed": False,
                "error": "pregrasp_pose_too_far",
                "target_pos_err_m": float(target_pos_err_w.item()),
                "planner_pos_err_m": _planner_pos_err,
            }

        post_target_descend_q = None
        # 恢复 descend：主轨迹停在方块上方 1cm，必须再下降 --post_target_descend_m
        # (默认 0.03m) 才能让夹爪包住方块体。写死 descend_m=0 会导致夹爪在方块
        # 上方闭合、永远夹不起方块（rise=0, paper_grasped=false）。
        # 与 main() 内 _execute_gripper_post_target_sequence 的 descend 保持一致。
        descend_m = _post_target_descend_m()
        if descend_m > 0.0:
            print(f"[PostTargetDescend] descend_m={descend_m:.3f}", flush=True)
        if descend_m > 0.0:
            # grasp_test_cube：descend 时把 tool tip XY 对齐到方块中心，并把 tool tip 降到
            # 方块中心下方一点，让手指包住方块体（而非搭在方块顶面）。
            # 用 PhysX 实时位姿而非 USD 静态位姿。
            align_xy_pos_w = None
            target_z_w = None
            _paper_mode_now = bool(target_meta.get("paper_cleanup_mode", False))
            _cube_path_now = str(target_meta.get("paper_prim_path", ""))
            if _paper_mode_now and "/grasp_test_cube" in _cube_path_now:
                _cube_now = _object_world_pos(env, _cube_path_now)
                if _cube_now is not None:
                    align_xy_pos_w = torch.tensor(
                        [_cube_now[0], _cube_now[1], 0.0],
                        device=robot.device, dtype=torch.float32,
                    )
                    # 方块中心 z 再下探 1cm：手指夹持点（比 tool_tip 高 ~1cm）正好包住方块中部。
                    # 方块尺寸 (0.015,0.05,0.04)，底面 z≈0.001，下限保护避免撞地。
                    cube_center_z = float(_cube_now[2])
                    target_z_w = max(cube_center_z - 0.010, 0.005)
                    print(
                        f"[PostTargetDescend] align_xy_to_cube=({_cube_now[0]:.4f},{_cube_now[1]:.4f}) "
                        f"cube_center_z={cube_center_z:.4f} target_tool_tip_z={target_z_w:.4f}",
                        flush=True,
                    )
            post_target_descend_q = _execute_post_target_world_descend(
                env=env,
                robot=robot,
                case_name=case_name,
                urdf_path=urdf_path,
                robot_cfg_path=robot_cfg_path,
                base_link=base_link,
                tool_frame=tool_frame,
                joint_names=joint_names,
                scene_mesh_obstacles=scene_mesh_obstacles,
                isaac_base_body=base_link.replace("-", "_"),
                descend_m=descend_m,
                align_xy_pos_w=align_xy_pos_w,
                target_z_w=target_z_w,
            )
        _t_g = _g_timing_mark("descend", _t_g)

        # DEBUG: 分阶段打印 tool_tip_link body pose
        _tp2 = _tool_tip_pose_w(robot)
        print(f"[DebugPhase] after_descend tool_pos_w={_tp2[0].detach().cpu().tolist()}", flush=True)

        grasp_q = (
            post_target_descend_q.detach().clone()
            if post_target_descend_q is not None
            else robot.data.joint_pos[0, arm_ids].detach().clone()
        )
        grasp_hold_q, grasped = _execute_gripper_feedback_grasp(
            env,
            robot,
            grasp_q,
            label=f"{case_name}_feedback_grasp",
        )
        _t_g = _g_timing_mark("feedback_grasp", _t_g)
        # DEBUG: after gripper close
        _tp3 = _tool_tip_pose_w(robot)
        print(f"[DebugPhase] after_gripper_close tool_pos_w={_tp3[0].detach().cpu().tolist()} grasped={grasped}", flush=True)

        # ---- paper_cleanup_mode: check if paper/cube is grasped ----
        # 成功判定逻辑：
        #   纸团（非 grasp_test_cube）：沿用距离判定 + visual attach。
        #   grasp_test_cube：距离接近只是前提，真正的"夹住"用抬升后方块是否跟手验证
        #   （在下方 lift 之后计算），避免距离<12cm 就误报成功。
        paper_grasped = False
        paper_attach_offset: tuple[float, float, float] | None = None
        cube_pos_before_lift: tuple[float, float, float] | None = None
        paper_cleanup_mode = bool(target_meta.get("paper_cleanup_mode", False))
        paper_prim_path = str(target_meta.get("paper_prim_path", ""))
        is_grasp_cube = paper_cleanup_mode and "/grasp_test_cube" in paper_prim_path
        _stage = None
        if paper_cleanup_mode and paper_prim_path:
            import omni.usd
            _stage = omni.usd.get_context().get_stage()
            paper_near, paper_attach_offset = _check_paper_grasp(
                _stage, robot, paper_prim_path,
            )
            if is_grasp_cube:
                # grasp_test_cube：只记录接近状态 + 抬升前方块位置；是否夹住由抬升后
                # 验证决定（见 lift 之后）。不做 paper 视觉 attach（否则会生成灰色球体
                # carried 视觉，cube 会变成两处）。
                # 用 PhysX 实时位姿（env.scene 数据），不要用 USD XformCache（静态初始位姿）
                cube_pos_before_lift = _object_world_pos(env, paper_prim_path)
                print(
                    f"[GraspCube] near={paper_near} offset={paper_attach_offset} path={paper_prim_path}",
                    flush=True,
                )
            elif paper_near and paper_attach_offset is not None:
                paper_grasped = True
                _LAST_GRASPED_PAPER_PATH = paper_prim_path
                # P1 v2 ENABLED: create carried visual at safe location
                _set_paper_prim_state(_stage, paper_prim_path, "grasped", False)
                try:
                    _attached = _logical_attach_paper_safe(_stage, paper_prim_path)
                    print(
                        f"[PaperCleanup] paper_grasped=True attached={_attached.get('carried_path','')} "
                        f"offset={paper_attach_offset} path={paper_prim_path}",
                        flush=True,
                    )
                except Exception as _e_attach:
                    print(
                        f"[PaperCleanup] paper_grasped=True attach_failed={_e_attach} "
                        f"offset={paper_attach_offset} path={paper_prim_path}",
                        flush=True,
                    )
            else:
                print(
                    f"[PaperCleanup] paper_grasped=False (distance > threshold)",
                    flush=True,
                )
        # --------------------------------------------------------
        hold_steps = _default_gripper_hold_steps()
        _execute_joint_target_until_converged(
            env,
            robot,
            grasp_q,
            max_steps=hold_steps,
            min_steps=max(1, int(args_cli.steps_per_waypoint)),
            label=f"{case_name}:grasp_hold",
            gripper_joint1_pos=grasp_hold_q,
        )

        lift_start_q = robot.data.joint_pos[0, arm_ids].detach().clone()
        lift_start_pos_w, lift_start_quat_w = _tool_tip_pose_w(robot)
        print(f"[DebugPhase] before_lift tool_pos_w={lift_start_pos_w.detach().cpu().tolist()}", flush=True)
        lift_target_pos_w = lift_start_pos_w.reshape(1, 3).detach().clone()
        lift_target_pos_w[:, 2] += max(0.0, float(args_cli.post_grasp_lift_m))
        lift_target_quat_w = lift_start_quat_w.reshape(1, 4).detach().clone()
        arm_base_pos_w, arm_base_quat_w = _body_pose_w(robot, base_link.replace("-", "_"))
        lift_target_pos_base, lift_target_quat_base = subtract_frame_transforms(
            arm_base_pos_w.reshape(1, 3),
            arm_base_quat_w.reshape(1, 4),
            lift_target_pos_w,
            lift_target_quat_w,
        )
        lift_result = _plan_single_pose(
            case_name=case_name,
            phase_name="gripper_lift_world_z_grasp_hold",
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            q_start=lift_start_q,
            scene_mesh_obstacles=scene_mesh_obstacles,
            target_pos_base=lift_target_pos_base,
            target_quat_base=lift_target_quat_base,
            target_pos_world_z=float(lift_target_pos_w.reshape(-1)[2].item()),
            env=env,
            robot=robot,
            hold_gripper_joint1_pos=grasp_hold_q,
        )
        lift_executed = False
        lift_error = None
        if lift_result.get("success") and lift_result.get("trajectory"):
            lift_traj = torch.tensor(lift_result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
            if lift_traj.dim() > 2:
                lift_traj = lift_traj.reshape(-1, lift_traj.shape[-1])
            _execute_trajectory(
                env,
                robot,
                lift_traj,
                gripper_joint1_pos=grasp_hold_q,
                label=f"{case_name}:gripper_lift",
            )
            lift_final_q = lift_traj[-1].detach().clone()
            _execute_joint_target_until_converged(
                env,
                robot,
                lift_final_q,
                max_steps=hold_steps,
                min_steps=max(1, int(args_cli.steps_per_waypoint)),
                label=f"{case_name}:gripper_lift_hold",
                gripper_joint1_pos=grasp_hold_q,
            )
            lift_executed = True
        else:
            lift_error = "lift cuRobo plan unavailable"
            print(
                f"[GripperPrismaticGrab] case={case_name} lift_plan_failed; holding_grasp_at_current_pose",
                flush=True,
            )
        _t_g = _g_timing_mark("lift", _t_g)

        # ---- grasp_test_cube 真实夹持验证：抬升后方块必须跟着夹爪升起 ----
        if is_grasp_cube and cube_pos_before_lift is not None and _stage is not None:
            cube_pos_after_lift = _object_world_pos(env, paper_prim_path)
            if cube_pos_after_lift is not None:
                lift_rise_m = float(cube_pos_after_lift[2]) - float(cube_pos_before_lift[2])
                lift_commanded_m = max(0.0, float(args_cli.post_grasp_lift_m))
                lift_threshold = max(0.04, 0.5 * lift_commanded_m)
                paper_grasped = lift_rise_m > lift_threshold
                print(
                    f"[GraspCube] lift_check z_before={cube_pos_before_lift[2]:.4f} "
                    f"z_after={cube_pos_after_lift[2]:.4f} rise={lift_rise_m:.4f} "
                    f"threshold={lift_threshold:.4f} grasped={paper_grasped}",
                    flush=True,
                )
                if paper_grasped:
                    _LAST_GRASPED_PAPER_PATH = paper_prim_path
            else:
                print(f"[GraspCube] lift_check skipped: cube prim missing after lift", flush=True)
        # -------------------------------------------------------------------

        final_pos_w, final_quat_w = _tool_tip_pose_w(robot)
        # DEBUG: final output
        print(f"[DebugPhase] final_output tool_pos_w={final_pos_w.detach().cpu().tolist()} lift_executed={lift_executed}", flush=True)
        _g_timing_mark("execute_total", _t_g)
        _g_print_summary("GrabTimingSummary")
        return {
            "executed": True,
            "target_executed": True,
            "descend_executed": post_target_descend_q is not None,
            "grasped": bool(grasped),
            "paper_grasped": paper_grasped,
            "carried_papers": len(_CARRIED_VISUAL_STATE),
            "gripper_hold_target_q": float(grasp_hold_q),
            "lift_executed": lift_executed,
            "lift_error": lift_error,
            "target_pos_err_m": float(target_pos_err_w.item()),
            "target_rot_err_deg": float(target_rot_err_deg_w.item()),
            "final_tool_tip_pos_w": final_pos_w.reshape(-1).detach().cpu().tolist(),
            "final_tool_tip_quat_wxyz": final_quat_w.reshape(-1).detach().cpu().tolist(),
        }

    def gripper_prismatic_move_arm_to_joints_plan_tool(parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            raise ValueError("gripper_prismatic_move_arm_to_joints parameters must be a JSON object.")
        case_name = str(parameters.get("name", "gripper_prismatic_move_arm_to_joints"))
        joint_pos_deg = _float_list_parameter(parameters, "joint_pos_deg", 6, required=False)
        target_q = (
            _deg_to_rad_joint_values(joint_pos_deg, device=robot.device, dtype=robot.data.joint_pos.dtype)
            if joint_pos_deg is not None
            else pre_target_q.detach().clone()
        )
        payload = _build_cspace_tool_payload(
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            q_current=robot.data.joint_pos[0, arm_ids].detach(),
            scene_mesh_obstacles=scene_mesh_obstacles,
            cfg=_make_phase_planner_cfg("return"),
            cspace_targets=[
                {
                    "name": f"{case_name}_joint_target",
                    "joint_pos": target_q.detach().cpu().tolist(),
                },
            ],
        )
        output = planner_tool(payload)
        output = dict(output)
        output["gripper_prismatic_joint_target"] = {
            "case_name": case_name,
            "target_q": target_q.detach().cpu().tolist(),
            "gripper_joint1_pos": _prismatic_gripper_joint1_target(
                float(parameters.get("gripper_joint1_pos", _default_grasp_gripper_joint1_pos()))
            ),
        }
        return output

    def execute_gripper_prismatic_move_arm_to_joints(tool_output: dict, parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            parameters = {}
        case_name = str(parameters.get("name", "gripper_prismatic_move_arm_to_joints"))
        target_meta = tool_output.get("gripper_prismatic_joint_target", {})
        target_q = torch.tensor(
            target_meta.get("target_q", pre_target_q.detach().cpu().tolist()),
            device=robot.device,
            dtype=robot.data.joint_pos.dtype,
        )
        gripper_hold_q = float(target_meta.get("gripper_joint1_pos", _default_grasp_gripper_joint1_pos()))

        # GPT P0: try reverse grab trajectory before cspace planning
        global _LAST_GRAB_TRAJECTORY, _LAST_GRAB_GRIPPER_POS
        return_mode = "cspace"
        if _LAST_GRAB_TRAJECTORY is not None:
            rev_traj = torch.flip(_LAST_GRAB_TRAJECTORY, [0])
            current_q = robot.data.joint_pos[0, arm_ids].detach()
            grab_start = rev_traj[0]  # = original grab trajectory end
            dist_to_start = torch.max(torch.abs(current_q - grab_start)).item()
            # Only use reverse if we're close to the grab end position
            if dist_to_start < 0.5:  # within ~30° per joint
                return_mode = "reverse_grab"
                print(
                    f"[ReverseReturn] case={case_name} using reverse grab trajectory "
                    f"waypoints={rev_traj.shape[0]} dist_to_start={dist_to_start:.4f} "
                    f"current_q={_format_tensor(current_q)} grab_end={_format_tensor(grab_start)}",
                    flush=True,
                )
                _execute_trajectory(
                    env, robot, rev_traj,
                    gripper_joint1_pos=gripper_hold_q,
                    label=f"{case_name}:reverse_grab",
                )
                actual_q = robot.data.joint_pos[0, arm_ids].detach()
                _, err_max, err = _joint_error(target_q, actual_q)
                return {
                    "executed": True,
                    "return_mode": return_mode,
                    "gripper_joint1_pos": gripper_hold_q,
                    "target_q": target_q.detach().cpu().tolist(),
                    "actual_q": actual_q.detach().cpu().tolist(),
                    "joint_err_max": float(err_max.item()),
                    "joint_err": err.detach().cpu().tolist(),
                }

        results = tool_output.get("results", [])
        result = results[0] if results else {}
        if not result.get("success") or not result.get("trajectory"):
            print(
                f"[GripperPrismaticMoveArmToJoints] cuRobo returned no executable C-space trajectory; "
                f"result={result}",
                flush=True,
            )
            return {"executed": False, "return_mode": return_mode, "error": "cspace cuRobo plan unavailable", "target_result": result}
        traj = torch.tensor(result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
        if traj.dim() > 2:
            traj = traj.reshape(-1, traj.shape[-1])
        _execute_trajectory(
            env,
            robot,
            traj,
            gripper_joint1_pos=gripper_hold_q,
            label=f"{case_name}:joint_target",
        )
        if args_cli.final_hold_steps > 0:
            _hold_joint_target_if_close(
                env,
                robot,
                target_q,
                args_cli.final_hold_steps,
                label=f"{case_name}:joint_target_hold",
                gripper_joint1_pos=gripper_hold_q,
            )
        actual_q = robot.data.joint_pos[0, arm_ids].detach()
        _, err_max, err = _joint_error(target_q, actual_q)
        return {
            "executed": True,
            "return_mode": return_mode,
            "target_q": target_q.detach().cpu().tolist(),
            "actual_q": actual_q.detach().cpu().tolist(),
            "joint_err_max": float(err_max.item()),
            "joint_err": err.detach().cpu().tolist(),
        }

    def gripper_prismatic_take_out_plan_tool(parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            raise ValueError("gripper_prismatic_take_out parameters must be a JSON object.")
        case_name = str(parameters.get("name", "gripper_prismatic_take_out"))
        drop_target_pos = _float_list_parameter(parameters, "drop_target_pos", 3, required=False)
        if drop_target_pos is None:
            drop_target_pos = list(_GRIPPER_DROP_TARGET_POS_W)
        drop_target_quat = _float_list_parameter(parameters, "drop_target_quat", 4, required=False)
        if drop_target_quat is None:
            drop_target_quat = list(_GRIPPER_DROP_TARGET_QUAT_WXYZ)
        arm_base_pos_w, arm_base_quat_w = _body_pose_w(robot, base_link.replace("-", "_"))
        drop_target_pos_w = torch.tensor(drop_target_pos, device=robot.device, dtype=torch.float32).reshape(1, 3)
        drop_target_quat_w = torch.tensor(drop_target_quat, device=robot.device, dtype=torch.float32).reshape(1, 4)
        drop_target_pos_base, drop_target_quat_base = subtract_frame_transforms(
            arm_base_pos_w.reshape(1, 3),
            arm_base_quat_w.reshape(1, 4),
            drop_target_pos_w,
            drop_target_quat_w,
        )
        pose_test = {
            "name": case_name,
            "target_pos": drop_target_pos_base.reshape(-1).detach().cpu().tolist(),
            "target_quat": drop_target_quat_base.reshape(-1).detach().cpu().tolist(),
            "target_pos_world_z": float(drop_target_pos_w.reshape(-1)[2].item()),
        }
        payload = _build_pose_tool_payload(
            parameters,
            pose_test=pose_test,
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            q_current=robot.data.joint_pos[0, arm_ids].detach(),
            scene_mesh_obstacles=scene_mesh_obstacles,
            cfg=_make_phase_planner_cfg("takeout"),
        )
        output = planner_tool(payload)
        output = dict(output)
        output["gripper_prismatic_drop_target"] = {
            "case_name": case_name,
            "drop_target_pos_w": list(drop_target_pos),
            "drop_target_quat_wxyz": list(drop_target_quat),
        }
        return output

    def execute_gripper_prismatic_take_out(tool_output: dict, parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            parameters = {}
        case_name = str(parameters.get("name", "gripper_prismatic_take_out"))
        results = tool_output.get("results", [])
        result = results[0] if results else {}
        grasp_hold_q = _default_grasp_gripper_joint1_pos()
        open_gripper_q = _default_open_gripper_joint1_pos()
        if not result.get("success") or not result.get("trajectory"):
            print(
                f"[GripperPrismaticTakeOut] drop cuRobo plan unavailable; not moving the arm. result={result}",
                flush=True,
            )
            return {"executed": False, "error": "drop cuRobo plan unavailable", "drop_result": result}
        # grasp_test_cube：drop 前先验证方块是否仍在夹爪（可能 grab_return 半路掉出）。
        # 若已掉落，不做瞬移造假，诚实报告 cube_lost。
        global _LAST_GRASPED_PAPER_PATH, _CARRIED_VISUAL_STATE
        grasped_path = _LAST_GRASPED_PAPER_PATH
        is_grasp_cube = bool(grasped_path and "/grasp_test_cube" in grasped_path)
        cube_lost = False
        if is_grasp_cube and grasped_path:
            import math as _math
            cube_pos = _object_world_pos(env, grasped_path)
            tool_pos = _tool_tip_pose_w(robot)[0].detach().cpu().tolist()
            held_dist = _math.dist(cube_pos, tool_pos) if cube_pos else float("inf")
            # 方块中心与 tool_tip 距离 > 0.15m → 已不在夹爪内（掉落）
            cube_lost = held_dist > 0.15
            print(
                f"[GraspCube] take_out held_check dist={held_dist:.3f} lost={cube_lost}",
                flush=True,
            )
        if cube_lost:
            _LAST_GRASPED_PAPER_PATH = None
            return {
                "executed": True,
                "drop_executed": False,
                "opened_gripper": False,
                "returned_to_pre_target_open": False,
                "paper_collected": False,
                "cube_lost": True,
                "reason": "cube_dropped_before_takeout",
            }
        drop_traj = torch.tensor(result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
        if drop_traj.dim() > 2:
            drop_traj = drop_traj.reshape(-1, drop_traj.shape[-1])
        _execute_trajectory(
            env,
            robot,
            drop_traj,
            gripper_joint1_pos=grasp_hold_q,
            label=f"{case_name}:drop_target",
        )
        drop_final_q = drop_traj[-1].detach().clone()
        actual_drop_q = robot.data.joint_pos[0, arm_ids].detach()
        _, drop_err_max, drop_err = _joint_error(drop_final_q, actual_drop_q)
        hold_steps = _default_gripper_hold_steps()
        _hold_joint_target_if_close(
            env,
            robot,
            drop_final_q,
            hold_steps,
            label=f"{case_name}:drop_open_hold",
            gripper_joint1_pos=open_gripper_q,
        )

        # ---- collect paper (P1 v2, or physics-verified cube drop; NO teleport for cubes) ----
        paper_collected = False
        global _LAST_GRAB_TRAJECTORY, _LAST_GRAB_GRIPPER_POS
        # P1 v2: detach visual papers
        for paper_id in list(_CARRIED_VISUAL_STATE.keys()):
            if _logical_detach_paper_safe(_stage, paper_id):
                paper_collected = True
                src_path = _LAST_GRASPED_PAPER_PATH or ""
                if src_path:
                    _set_paper_prim_state(_stage, src_path, "collected", True)
        if is_grasp_cube and grasped_path:
            # grasp_test_cube：真实物理验证。drop 已执行（夹爪张开），补跑物理步让方块
            # 落入/落定垃圾桶，再用 PhysX 实时位姿判断是否真的进了桶。禁止瞬移造假。
            settle_steps = max(60, int(args_cli.post_trajectory_settle_steps or 0))
            for _ in range(settle_steps):
                _step(env, render=False)
            cube_pos_after = _object_world_pos(env, grasped_path)
            paper_collected = _pos_in_trash_bin_aabb(env, cube_pos_after)
            print(
                f"[GraspCube] take_out phys_check pos={cube_pos_after} "
                f"collected={paper_collected}",
                flush=True,
            )
            if paper_collected:
                _set_paper_prim_state(_stage, grasped_path, "collected", True)
        elif not _CARRIED_VISUAL_STATE and _LAST_GRASPED_PAPER_PATH:
            # Fallback: teleport grasped paper (when P1 v2 is disabled) — paper_waste only,
            # 不会命中 grasp_test_cube（上面 is_grasp_cube 分支已拦截）。
            paper_path = _LAST_GRASPED_PAPER_PATH
            from pxr import Gf, UsdGeom
            _set_paper_prim_state(_stage, paper_path, "loose", False)
            paper_prim = _stage.GetPrimAtPath(paper_path)
            if paper_prim.IsValid():
                xf = UsdGeom.Xformable(paper_prim)
                ops = xf.GetOrderedXformOps()
                bin_pos = (_GRIPPER_DROP_TARGET_POS_W[0], _GRIPPER_DROP_TARGET_POS_W[1], 0.05)
                if ops:
                    ops[0].Set(Gf.Vec3d(*bin_pos))
                print(f"[PaperCleanup] teleported paper to bin pos={bin_pos}", flush=True)
            if _check_paper_collected_at_prim(_stage, paper_path):
                paper_collected = True
        _LAST_GRASPED_PAPER_PATH = None
        # ---------------------------------------

        # 投放后是否再回规划起始位姿。默认 False：投放完就让机械臂停在原地，
        # 否则数据集/视频里会多出一段"投完又抬起来"的动作（而且会污染 demo 尾部）。
        return_to_pre_target_open = bool(parameters.get("return_to_pre_target_open", False))
        if return_to_pre_target_open:
            returned_open_with_plan = _plan_and_execute_cspace_target(
                env=env,
                robot=robot,
                case_name=case_name,
                phase_name="drop_return_pre_target_open",
                urdf_path=urdf_path,
                robot_cfg_path=robot_cfg_path,
                base_link=base_link,
                tool_frame=tool_frame,
                joint_names=joint_names,
                scene_mesh_obstacles=scene_mesh_obstacles,
                target_q=pre_target_q,
                hold_gripper_joint1_pos=open_gripper_q,
                final_settle=True,
            )
        else:
            returned_open_with_plan = False
            print(
                f"[GripperPrismaticTakeOut] case={case_name} return_to_pre_target_open=False; "
                "holding current pose after drop (no post-drop motion).",
                flush=True,
            )
        final_return_q = robot.data.joint_pos[0, arm_ids].detach()
        _, return_err_max, return_err = _joint_error(pre_target_q, final_return_q)
        # Clear saved grab trajectory after drop (paper is no longer grasped)
        global _LAST_GRAB_TRAJECTORY, _LAST_GRAB_GRIPPER_POS
        _LAST_GRAB_TRAJECTORY = None
        _LAST_GRAB_GRIPPER_POS = None
        return {
            # executed 表示"投放这一步做完了"。原来它等于 returned_open_with_plan，
            # 关掉投放后回位后会让 SOP 误判成"投放未执行"。
            "executed": True,
            "drop_executed": True,
            "opened_gripper": True,
            "returned_to_pre_target_open": bool(returned_open_with_plan),
            "paper_collected": paper_collected,
            "drop_joint_err_max": float(drop_err_max.item()),
            "drop_joint_err": drop_err.detach().cpu().tolist(),
            "return_joint_err_max": float(return_err_max.item()),
            "return_joint_err": return_err.detach().cpu().tolist(),
        }

    def gripper_prismatic_local_lift_plan_tool(parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            raise ValueError("gripper_prismatic_local_lift parameters must be a JSON object.")
        case_name = str(parameters.get("name", "gripper_prismatic_local_lift"))
        lift_height_m = max(0.01, float(parameters.get("lift_height_m", 0.18)))

        # 获取当前 tool_tip 在世界坐标系中的位姿
        tool_pos_w, tool_quat_w = _tool_tip_pose_w(robot)
        target_pos_w = tool_pos_w.reshape(1, 3).detach().clone()
        target_pos_w[:, 2] += lift_height_m
        target_quat_w = tool_quat_w.reshape(1, 4).detach().clone()

        arm_base_pos_w, arm_base_quat_w = _body_pose_w(robot, base_link.replace("-", "_"))
        target_pos_base, target_quat_base = subtract_frame_transforms(
            arm_base_pos_w.reshape(1, 3),
            arm_base_quat_w.reshape(1, 4),
            target_pos_w,
            target_quat_w,
        )

        print(
            f"[LocalLift] case={case_name} lift_height_m={lift_height_m:.3f} "
            f"current_tool_z_w={float(tool_pos_w[2].item()):.4f} "
            f"target_tool_z_w={float(target_pos_w.reshape(-1)[2].item()):.4f}",
            flush=True,
        )

        pose_test = {
            "name": case_name,
            "target_pos": target_pos_base.reshape(-1).detach().cpu().tolist(),
            "target_quat": target_quat_base.reshape(-1).detach().cpu().tolist(),
            "target_pos_world_z": float(target_pos_w.reshape(-1)[2].item()),
            "input_target_frame": "world",
        }
        payload = _build_pose_tool_payload(
            parameters,
            pose_test=pose_test,
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            q_current=robot.data.joint_pos[0, arm_ids].detach(),
            scene_mesh_obstacles=scene_mesh_obstacles,
            cfg=_make_phase_planner_cfg("lift"),
        )
        output = planner_tool(payload)
        output = dict(output)
        output["gripper_prismatic_local_lift"] = {
            "case_name": case_name,
            "lift_height_m": lift_height_m,
            "target_pos_w": target_pos_w.reshape(-1).detach().cpu().tolist(),
            "target_quat_wxyz": target_quat_w.reshape(-1).detach().cpu().tolist(),
        }
        return output

    def execute_gripper_prismatic_local_lift(tool_output: dict, parameters: dict) -> dict:
        if not isinstance(parameters, dict):
            parameters = {}
        case_name = str(parameters.get("name", "gripper_prismatic_local_lift"))
        lift_meta = tool_output.get("gripper_prismatic_local_lift", {})
        lift_height_m = float(lift_meta.get("lift_height_m", 0.18))
        results = tool_output.get("results", [])
        result = results[0] if results else {}
        if not result.get("success") or not result.get("trajectory"):
            print(
                f"[LocalLift] case={case_name} lift trajopt unavailable (walls may block the path). "
                f"result={result}",
                flush=True,
            )
            return {
                "executed": False,
                "error": "local_lift trajopt plan unavailable",
                "lift_result": result,
            }

        grasp_hold_q = _default_grasp_gripper_joint1_pos()
        lift_traj = torch.tensor(result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
        if lift_traj.dim() > 2:
            lift_traj = lift_traj.reshape(-1, lift_traj.shape[-1])
        _execute_trajectory(
            env,
            robot,
            lift_traj,
            gripper_joint1_pos=grasp_hold_q,
            label=f"{case_name}:local_lift",
        )
        lift_final_q = lift_traj[-1].detach().clone()
        hold_steps = _default_gripper_hold_steps()
        _execute_joint_target_until_converged(
            env,
            robot,
            lift_final_q,
            max_steps=hold_steps,
            min_steps=max(1, int(args_cli.steps_per_waypoint)),
            label=f"{case_name}:local_lift_hold",
            gripper_joint1_pos=grasp_hold_q,
        )

        actual_pos_w, actual_quat_w = _tool_tip_pose_w(robot)
        target_pos_w = torch.tensor(
            lift_meta.get("target_pos_w", [0.0, 0.0, 0.0]),
            device=robot.device,
            dtype=torch.float32,
        ).reshape(1, 3)
        pos_err = float((actual_pos_w.reshape(1, 3) - target_pos_w).norm(dim=1).item())
        print(
            f"[LocalLift] case={case_name} lift_height_m={lift_height_m:.3f} "
            f"actual_z_w={float(actual_pos_w[2].item()):.4f} pos_err_m={pos_err:.4f}",
            flush=True,
        )
        return {
            "executed": True,
            "lift_height_m": lift_height_m,
            "actual_tool_pos_w": actual_pos_w.detach().cpu().tolist(),
            "pos_err_m": pos_err,
        }

    def list_robot_profiles_tool(parameters: dict) -> dict:
        del parameters
        if not _common_profile_enabled():
            return {
                "common_multi_robot": False,
                "active_robot_id": None,
                "profiles": [],
            }
        return {
            "common_multi_robot": True,
            "active_robot_id": active_robot_id,
            "profiles": [
                {
                    "robot_id": profile_id,
                    "asset_name": "robot",
                    "robot_type": profile["robot_type"],
                    "usd_path": _profile_usd_path(env.cfg, profile_id),
                    "enabled_tools": (
                        ["common", "brush"] if profile["robot_type"] == "brush" else ["common", "gripper"]
                    ),
                }
                for profile_id, profile in common_robot_profiles.items()
            ],
        }

    def get_active_robot_tool(parameters: dict) -> dict:
        del parameters
        if not _common_profile_enabled():
            return {
                "common_multi_robot": False,
                "active_robot_id": None,
                "robot_type": "gripper" if _is_prismatic_gripper_joint_profile() else "brush",
                "urdf_path": str(urdf_path),
                "curobo_robot_cfg_path": str(robot_cfg_path),
            }
        profile = _active_profile()
        return {
            "common_multi_robot": True,
            "active_robot_id": active_robot_id,
            "asset_name": "robot",
            "robot_type": profile["robot_type"],
            "usd_path": _profile_usd_path(env.cfg, active_robot_id),
            "urdf_path": str(urdf_path),
            "curobo_robot_cfg_path": str(robot_cfg_path),
            "base_link": base_link,
            "tool_frame": tool_frame,
            "joint_names": list(joint_names),
            "switch_ready_arm_joint_pos_deg": list(_BASE_TELEOP_PRE_ARM_JOINT_POS_DEG),
        }

    pending_switch_job: dict | None = None

    def _complete_pending_switch_robot_job():
        nonlocal pending_switch_job
        if pending_switch_job is None:
            return None
        switch_job = pending_switch_job
        pending_switch_job = None
        request_id = switch_job["request_id"]
        preserved_state = switch_job["preserved_state"]
        target_robot_id = switch_job["target_robot_id"]
        case_name = switch_job["case_name"]
        old_robot_id = switch_job["old_robot_id"]
        old_ready = switch_job["old_ready"]
        try:
            print(
                f"[CommonRobotSwitch] event=deferred_replace_begin request_id={request_id} "
                f"old_robot_id={old_robot_id} target_robot_id={target_robot_id}",
                flush=True,
            )
            _restore_switch_preserved_shared_assets(preserved_state)
            _set_active_robot_context(target_robot_id, move_to_work_pose=True, rebuild_robot=True)
            _restore_switch_preserved_shared_assets(preserved_state)
            new_ready = _move_active_arm_to_switch_ready(case_name, "new_active_switch_ready")
            output = {
                "executed": bool(new_ready.get("moved")),
                "old_robot_id": old_robot_id,
                "active_robot_id": active_robot_id,
                "old_ready": old_ready,
                "new_ready": new_ready,
                "active_robot": get_active_robot_tool({}),
                "preserved_rigid_assets": sorted(preserved_state.get("rigid_assets", {})),
                "preserved_articulations": sorted(preserved_state.get("articulations", {})),
            }
            completed_job = bridge.complete_deferred(request_id, tool_output=output)
            print(
                f"[CommonRobotSwitch] event=deferred_replace_done request_id={request_id} "
                f"executed={output['executed']} active_robot_id={active_robot_id}",
                flush=True,
            )
            return completed_job
        except Exception as exc:  # noqa: BLE001 - surfaced through ToolAPI job state.
            completed_job = bridge.complete_deferred(request_id, error=str(exc))
            print(
                f"[CommonRobotSwitch] event=deferred_replace_failed request_id={request_id} "
                f"error={exc}",
                flush=True,
            )
            return completed_job
        finally:
            _end_switch_preserve_shared_assets(preserved_state)

    def switch_robot_tool(parameters: dict) -> dict:
        nonlocal pending_switch_job
        if not _common_profile_enabled():
            raise ValueError("switch_robot is only available in the common profile-switching cuRobo task.")
        if not isinstance(parameters, dict):
            raise ValueError("switch_robot parameters must be a JSON object.")
        target_robot_id = str(parameters.get("robot_id", "")).strip()
        if not target_robot_id:
            raise ValueError("switch_robot requires a non-empty robot_id.")
        if target_robot_id not in common_robot_profiles:
            raise ValueError(f"Unknown robot_id={target_robot_id!r}; available={sorted(common_robot_profiles)}")
        case_name = str(parameters.get("name", f"switch_robot_{target_robot_id}"))
        old_robot_id = active_robot_id
        if pending_switch_job is not None:
            raise RuntimeError("Another switch_robot job is already deferred.")
        preserved_state = _begin_switch_preserve_shared_assets()
        try:
            old_ready = _move_active_arm_to_switch_ready(case_name, "old_active_switch_ready")
            if not old_ready.get("moved"):
                _end_switch_preserve_shared_assets(preserved_state)
                return {
                    "executed": False,
                    "error": "old active robot failed to reach switch-ready arm pose",
                    "old_robot_id": old_robot_id,
                    "target_robot_id": target_robot_id,
                    "old_ready": old_ready,
                    "preserved_rigid_assets": sorted(preserved_state.get("rigid_assets", {})),
                    "preserved_articulations": sorted(preserved_state.get("articulations", {})),
                }
            _restore_switch_preserved_shared_assets(preserved_state)
            current_job = bridge.current_job()
            if current_job is None:
                raise RuntimeError("switch_robot must run through SimulationToolBridge to defer robot replacement.")
            pending_switch_job = {
                "request_id": current_job.request_id,
                "case_name": case_name,
                "old_robot_id": old_robot_id,
                "target_robot_id": target_robot_id,
                "old_ready": old_ready,
                "preserved_state": preserved_state,
            }
            print(
                f"[CommonRobotSwitch] event=deferred_replace_scheduled request_id={current_job.request_id} "
                f"old_robot_id={old_robot_id} target_robot_id={target_robot_id}",
                flush=True,
            )
            raise DeferredToolCompletion(
                {
                    "executed": False,
                    "deferred": True,
                    "old_robot_id": old_robot_id,
                    "target_robot_id": target_robot_id,
                    "old_ready": old_ready,
                    "preserved_rigid_assets": sorted(preserved_state.get("rigid_assets", {})),
                    "preserved_articulations": sorted(preserved_state.get("articulations", {})),
                }
            )
        except DeferredToolCompletion:
            raise
        except Exception:
            _end_switch_preserve_shared_assets(preserved_state)
            raise

    registry = ToolRegistry()
    registry.register("list_robot_profiles", list_robot_profiles_tool, spec={"name": "list_robot_profiles"})
    registry.register("get_active_robot", get_active_robot_tool, spec={"name": "get_active_robot"})
    registry.register("switch_robot", switch_robot_tool, spec={"name": "switch_robot"})
    registry.register(
        "brush_execute_point_clean",
        _wrap_tool_type("brush", "brush_execute_point_clean", brush_execute_point_clean_plan_tool),
        output_executor=execute_brush_execute_point_clean,
        spec=_brush_execute_point_clean_spec(),
    )
    registry.register(
        "brush_execute_spot_clean",
        _wrap_tool_type("brush", "brush_execute_spot_clean", brush_execute_spot_clean_plan_tool),
        output_executor=execute_brush_execute_spot_clean,
        spec=_brush_execute_spot_clean_spec(),
    )
    registry.register(
        "stretch_end_tool",
        stretch_end_tool_plan_tool,
        output_executor=execute_stretch_end_tool,
        spec=_stretch_end_tool_spec(),
    )
    registry.register(
        "brush_move_arm_to_joints",
        _wrap_tool_type("brush", "brush_move_arm_to_joints", brush_move_arm_to_joints_plan_tool),
        output_executor=execute_brush_move_arm_to_joints,
        spec=_brush_move_arm_to_joints_spec(),
    )
    registry.register(
        "execute_spot_capture",
        execute_spot_capture_plan_tool,
        output_executor=execute_execute_spot_capture,
        spec=_execute_spot_capture_spec(),
    )
    registry.register(
        "base_keyboard_teleop",
        base_keyboard_teleop_plan_tool,
        output_executor=execute_base_keyboard_teleop,
        spec=_base_keyboard_teleop_spec(),
    )
    def _with_demo_recording(
        output_executor,
        *,
        begin_demo: bool = False,
        phase: str | None = None,
        result_kind: str | None = None,
    ):
        """HDF5 demo 录制包装。

        - begin_demo：打开新 demo（同时收尾上一条），用于 execute_grab。
        - phase：切换到对应阶段标签（move_arm_to_joints → return，take_out → take_out）。
        - result_kind：执行结束后记录成功标签（grab / take_out）。
        执行期间置 _RECORD_ACTIVE=True，让 env.step 录制包装逐帧采样；结束后复位。
        未启用录制（_DEMO_RECORDER is None）时原样返回 executor，行为完全不变。
        """
        if _DEMO_RECORDER is None:
            return output_executor

        def wrapped(tool_output, parameters):
            global _RECORD_ACTIVE
            if begin_demo:
                _DEMO_RECORDER.begin_demo(parameters)
            elif phase:
                _DEMO_RECORDER.set_phase(phase)
            _RECORD_ACTIVE = True
            try:
                result = output_executor(tool_output, parameters)
                if result_kind:
                    _DEMO_RECORDER.note_result(result_kind, result)
                    if result_kind == "take_out":
                        # 投放做完就收尾：后面的回位/空转帧不该进数据集。
                        # 原先只在下一次 grab 的 begin_demo 里收尾，会把投放后的动作也录进去。
                        _DEMO_RECORDER.finalize_demo()
                return result
            finally:
                _RECORD_ACTIVE = False

        return wrapped

    if _common_profile_enabled() or _is_prismatic_gripper_joint_profile():
        registry.register(
            "gripper_prismatic_execute_grab",
            _wrap_tool_type("gripper", "gripper_prismatic_execute_grab", gripper_prismatic_execute_grab_plan_tool),
            output_executor=_with_demo_recording(
                execute_gripper_prismatic_execute_grab,
                begin_demo=True,
                result_kind="grab",
            ),
            spec=_gripper_prismatic_execute_grab_spec(),
        )
        registry.register(
            "gripper_prismatic_move_arm_to_joints",
            _wrap_tool_type(
                "gripper",
                "gripper_prismatic_move_arm_to_joints",
                gripper_prismatic_move_arm_to_joints_plan_tool,
            ),
            output_executor=_with_demo_recording(
                execute_gripper_prismatic_move_arm_to_joints,
                phase="return",
            ),
            spec=_gripper_prismatic_move_arm_to_joints_spec(),
        )
        registry.register(
            "gripper_prismatic_take_out",
            _wrap_tool_type("gripper", "gripper_prismatic_take_out", gripper_prismatic_take_out_plan_tool),
            output_executor=_with_demo_recording(
                execute_gripper_prismatic_take_out,
                phase="take_out",
                result_kind="take_out",
            ),
            spec=_gripper_prismatic_take_out_spec(),
        )
        registry.register(
            "gripper_prismatic_local_lift",
            _wrap_tool_type("gripper", "gripper_prismatic_local_lift", gripper_prismatic_local_lift_plan_tool),
            # 必须包录制包装：否则抬升期间 _RECORD_ACTIVE=False，一帧不采，
            # 视频会从抓取结束直接闪到投放开始（抬升过程整个丢失）。
            output_executor=_with_demo_recording(
                execute_gripper_prismatic_local_lift,
                phase="lift",
            ),
            spec=_gripper_prismatic_local_lift_spec(),
        )
    bridge = SimulationToolBridge(registry)

    # ── Temporary diagnostic tool (expanded) ──
    def _diagnose_stage_tool(parameters: dict) -> dict:
        """Comprehensive wall/door/robot/cuRobo collision diagnosis.
        Pass fix_collision=True to remove group0 self-filter at runtime."""
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf

        stage = omni.usd.get_context().get_stage()

        # ── Runtime fix: remove self-filter from collision group0 ──
        if parameters.get("fix_collision"):
            GROUP_PATH = "/World/collisions/group0"
            gp = stage.GetPrimAtPath(GROUP_PATH)
            if gp.IsValid():
                g = UsdPhysics.CollisionGroup(gp)
                fr = g.GetFilteredGroupsRel()
                before = list(fr.GetTargets())
                sp = gp.GetPath()
                after = [t for t in before if t != sp]
                fr.SetTargets(after)
                return {"fix_collision": "done", "before": [str(b) for b in before],
                        "after": [str(a) for a in after]}
            return {"fix_collision": "error", "reason": "group0 not found"}

        if parameters.get("restore_collision"):
            GROUP_PATH = "/World/collisions/group0"
            gp = stage.GetPrimAtPath(GROUP_PATH)
            if gp.IsValid():
                g = UsdPhysics.CollisionGroup(gp)
                fr = g.GetFilteredGroupsRel()
                before = list(fr.GetTargets())
                sp = gp.GetPath()
                if sp not in before:
                    fr.AddTarget(sp)
                after = list(fr.GetTargets())
                return {"restore_collision": "done", "before": [str(b) for b in before],
                        "after": [str(a) for a in after]}
            return {"restore_collision": "error", "reason": "group0 not found"}

        # ── Door material deep audit ──
        if parameters.get("door_material_debug"):
            from pxr import UsdShade
            DOORS = ("YS2HVXZVALGD2PTUJQ888888_7qcYe", "YS3UEUZVAJ2V4PTUKU888888_mdynP")
            result = {"meshes": [], "used_layers": [], "materials": []}

            # Check used layers
            for layer in stage.GetUsedLayers():
                lid = layer.identifier.lower()
                if any(k in lid for k in ("door_ys2", "door_ys3", "look", "material")):
                    result["used_layers"].append(layer.identifier)

            for prim in stage.Traverse():
                path = prim.GetPath().pathString
                if not any(d in path for d in DOORS) or not prim.IsA(UsdGeom.Mesh):
                    continue

                mesh_info = {"path": path, "bindings": [], "geometry": {}}

                # Material binding chain
                p = prim
                while p and p.IsValid():
                    for rel in p.GetRelationships():
                        name = rel.GetName()
                        if "material:binding" in name:
                            mesh_info["bindings"].append({
                                "ancestor": str(p.GetPath()),
                                "relationship": name,
                                "targets": [str(t) for t in rel.GetTargets()],
                            })
                    p = p.GetParent()

                # ComputeBoundMaterial
                try:
                    mat_api = UsdShade.MaterialBindingAPI(prim)
                    mat, rel = mat_api.ComputeBoundMaterial()
                    mesh_info["computed_material"] = str(mat.GetPath()) if mat else None
                    mesh_info["computed_relationship"] = str(rel.GetPath()) if rel else None
                except Exception as e:
                    mesh_info["computed_material"] = f"error: {e}"

                # Check target material prims
                mat_targets = []
                for b in mesh_info["bindings"]:
                    for t in b["targets"]:
                        mp = stage.GetPrimAtPath(t)
                        if mp.IsValid():
                            mat_info = {"path": t, "valid": True, "type": mp.GetTypeName(),
                                       "active": mp.IsActive(), "loaded": mp.IsLoaded()}
                            # Check shader
                            for child in mp.GetChildren():
                                if child.IsA(UsdShade.Shader):
                                    sh = UsdShade.Shader(child)
                                    mat_info["shader_id"] = sh.GetIdAttr().Get() if sh.GetIdAttr().IsValid() else "?"
                                    mat_info["shader_source"] = str(sh.GetImplementationSource())
                                    # Get sourceAsset
                                    sa = child.GetAttribute("inputs:file")
                                    if sa and sa.IsValid():
                                        mat_info["source_asset"] = str(sa.Get())
                                    sa2 = child.GetAttribute("info:mdl:sourceAsset")
                                    if sa2 and sa2.IsValid():
                                        mat_info["mdl_source_asset"] = str(sa2.Get())
                            mat_targets.append(mat_info)
                        else:
                            mat_targets.append({"path": t, "valid": False})
                mesh_info["material_targets"] = mat_targets

                # Geometry attributes
                mesh = UsdGeom.Mesh(prim)
                gprim = UsdGeom.Gprim(prim)
                pts = mesh.GetPointsAttr().Get()
                counts = mesh.GetFaceVertexCountsAttr().Get()
                idx = mesh.GetFaceVertexIndicesAttr().Get()
                mesh_info["geometry"] = {
                    "points": len(pts) if pts else 0,
                    "faces": len(counts) if counts else 0,
                    "indices": len(idx) if idx else 0,
                    "doubleSided": mesh.GetDoubleSidedAttr().Get(),
                    "orientation": str(mesh.GetOrientationAttr().Get()),
                    "displayColor": str(gprim.GetDisplayColorAttr().Get()),
                    "displayOpacity": str(gprim.GetDisplayOpacityAttr().Get()),
                }

                result["meshes"].append(mesh_info)

            return {"door_material_debug": result}

        # ── Glass door final diagnosis: runtime paths vs look bindings + transparency test ──
        if parameters.get("glass_door_final_diagnosis"):
            from pxr import Sdf, UsdShade, UsdGeom, Gf
            DOORS = ("YS2HVXZVALGD2PTUJQ888888_7qcYe", "YS3UEUZVAJ2V4PTUKU888888_mdynP")
            r = {"runtime_meshes": [], "look_layers": [], "rtx_setting": None, "cubes": {}}

            # 1. Runtime mesh paths
            for prim in stage.Traverse():
                path = prim.GetPath().pathString
                if any(d in path for d in DOORS) and prim.IsA(UsdGeom.Mesh):
                    r["runtime_meshes"].append({"path": path, "type": prim.GetTypeName()})

            # 2. Look layer identifiers (not spec traversal)
            for layer in stage.GetUsedLayers():
                lid = layer.identifier.lower()
                if ("door_ys2" in lid or "door_ys3" in lid) and "look" in lid:
                    r["look_layers"].append(layer.identifier)

            # 3. RTX setting
            try:
                import carb
                settings = carb.settings.get_settings()
                r["rtx_setting"] = settings.get("/rtx/debug/onlyOpaqueRayFlags")
            except Exception as e:
                r["rtx_setting"] = f"error: {e}"

            # 4. OmniGlass extension
            try:
                import omni.kit.app
                app = omni.kit.app.get_app()
                em = app.get_extension_manager()
                r["material_library_enabled"] = em.is_extension_enabled("omni.kit.material.library")
            except Exception as e:
                r["material_library_enabled"] = f"error: {e}"

            # 5. Transparency cube test
            CUBE_ROOT = "/World/GlassRenderDiagnosis"
            cube_names = ["OpaqueCube", "PreviewOpacityCube", "OmniGlassCube"]
            cube_colors = [(1.0, 0.1, 0.1), (0.1, 0.8, 1.0), (0.3, 0.6, 1.0)]
            for i, (name, color) in enumerate(zip(cube_names, cube_colors)):
                cp = f"{CUBE_ROOT}/{name}"
                cube = UsdGeom.Cube.Define(stage, cp)
                cube.CreateSizeAttr(0.3)
                xf = UsdGeom.Xformable(cube.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(2.0 + i * 0.5, -1.5, 1.5))
                mesh = UsdGeom.Mesh(cube.GetPrim())
                if not mesh:
                    continue  # Cube is not Mesh, use the prim directly
                mesh_prim = cube.GetPrim()
                mesh_prim.CreateAttribute("doubleSided", Sdf.ValueTypeNames.Bool).Set(True)
                mat_path = f"{CUBE_ROOT}/Mat_{name}"
                mat = UsdShade.Material.Define(stage, mat_path)
                sh = UsdShade.Shader.Define(stage, mat_path + "/Shader")
                sh.CreateIdAttr("UsdPreviewSurface")
                sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
                if i == 1:  # PreviewOpacity
                    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.35)
                    sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
                elif i == 2:  # OmniGlass attempt
                    try:
                        import numpy as np
                        from isaacsim.core.api.materials import OmniGlass
                        OmniGlass(prim_path=f"{CUBE_ROOT}/OmniGlassMat_{name}", ior=1.5, depth=0.005,
                                  thin_walled=True, color=np.array([0.3, 0.6, 1.0], dtype=np.float32))
                        r["cubes"][name] = "OmniGlass created"
                    except Exception as e:
                        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.35)
                        r["cubes"][name] = f"OmniGlass failed: {e}, using opacity fallback"
                else:
                    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
                mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
                bound, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
                r["cubes"][name] = r["cubes"].get(name, f"color={color} bound={'OK' if bound else 'FAIL'}")

            return {"glass_door_final_diagnosis": r}

        # ── Cleanup diagnosis cubes ──
        if parameters.get("cleanup_diagnosis_cubes"):
            CUBE_ROOT = "/World/GlassRenderDiagnosis"
            root_prim = stage.GetPrimAtPath(CUBE_ROOT)
            if root_prim.IsValid():
                stage.RemovePrim(CUBE_ROOT)
                return {"cleanup": f"removed {CUBE_ROOT}"}
            return {"cleanup": f"{CUBE_ROOT} not found"}

        # ── Regular diagnosis below ──
        r = {"robot": {}, "collision_groups": [], "filtered_pairs": [],
             "cuRobo_export": {}, "door_visibility": {}, "bbox": {}}

        # ── Robot colliders ──
        robot_root = stage.GetPrimAtPath("/World/envs/env_0/robot")
        if robot_root.IsValid():
            rc = {"mesh_count": 0, "collision_count": 0, "enabled_count": 0, "all_colliders": [],
                  "non_mesh_colliders": []}
            for prim in Usd.PrimRange(robot_root):
                if prim.IsA(UsdGeom.Mesh):
                    rc["mesh_count"] += 1
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        rc["collision_count"] += 1
                        try:
                            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                                rc["enabled_count"] += 1
                        except: pass
                if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsA(UsdGeom.Mesh):
                    try: en = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                    except: en = "?"
                    rc["non_mesh_colliders"].append(f"{prim.GetPath()} type={prim.GetTypeName()} enabled={en}")
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    try: en = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                    except: en = "?"
                    rc["all_colliders"].append(f"{prim.GetPath()} type={prim.GetTypeName()} enabled={en}")
            r["robot"] = rc

        # ── Collision groups ──
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.CollisionGroup):
                continue
            info = {"path": str(prim.GetPath())}
            for rn in ["collection:colliders:includes", "collection:colliders:excludes", "physics:filteredGroups"]:
                rel = prim.GetRelationship(rn)
                if rel and rel.IsValid():
                    info[rn] = [str(t) for t in rel.GetTargets()]
            inv = prim.GetAttribute("physics:invertFilteredGroups")
            if inv and inv.IsValid():
                info["invertFilteredGroups"] = inv.Get()
            r["collision_groups"].append(info)

        # ── Filtered pairs ──
        for prim in stage.Traverse():
            rel = prim.GetRelationship("physics:filteredPairs")
            if rel and rel.IsValid():
                targets = rel.GetTargets()
                if targets:
                    r["filtered_pairs"].append({str(prim.GetPath()): [str(t) for t in targets]})

        # ── cuRobo export debug ──
        for rp in ["/World/envs/env_0/room/Meshes/Base",
                   "/World/envs/env_0/room/Meshes/Base/Wall_0000",
                   "/World/envs/env_0/room/Meshes/Assets/Wall_Wall",
                   "/World/envs/env_0/room/CollisionFixes/CuroboOnlyWalls"]:
            root = stage.GetPrimAtPath(rp)
            info = {"valid": root.IsValid(), "type": str(root.GetTypeName()) if root.IsValid() else "?",
                    "descendants": 0, "meshes": 0, "total_points": 0, "total_indices": 0, "mesh_samples": []}
            if root.IsValid():
                for prim in Usd.PrimRange(root):
                    info["descendants"] += 1
                    if not prim.IsA(UsdGeom.Mesh): continue
                    info["meshes"] += 1
                    mesh = UsdGeom.Mesh(prim)
                    try:
                        pts = mesh.GetPointsAttr().Get()
                        idx = mesh.GetFaceVertexIndicesAttr().Get()
                        cnt = mesh.GetFaceVertexCountsAttr().Get()
                        npts = len(pts) if pts else 0; nidx = len(idx) if idx else 0; ncnt = len(cnt) if cnt else 0
                        info["total_points"] += npts; info["total_indices"] += nidx
                        if len(info["mesh_samples"]) < 3:
                            info["mesh_samples"].append(f"{prim.GetPath()} pts={npts} idx={nidx} faces={ncnt} instPrx={prim.IsInstanceProxy()}")
                    except Exception as e:
                        info["mesh_samples"].append(f"{prim.GetPath()} ERR:{e}")
            r["cuRobo_export"][rp] = info

        # ── Glass door visibility ──
        for dn in ("YS2HVXZVALGD2PTUJQ888888_7qcYe", "YS3UEUZVAJ2V4PTUKU888888_mdynP"):
            dv = []
            for prim in stage.Traverse():
                if dn not in prim.GetPath().pathString or not prim.IsA(UsdGeom.Mesh): continue
                img = UsdGeom.Imageable(prim)
                try: vis = img.ComputeVisibility()
                except: vis = "err"
                try: purp = img.GetPurposeAttr().Get()
                except: purp = "?"
                try:
                    mat_api = UsdShade.MaterialBindingAPI(prim)
                    mat, _ = mat_api.ComputeBoundMaterial()
                    mat_path = str(mat.GetPath()) if mat else None
                except: mat_path = "err"
                try:
                    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_,
                                    UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
                    bound = bbox_cache.ComputeWorldBound(prim)
                    box = bound.ComputeAlignedBox()
                    bb = f"min={box.GetMin()} max={box.GetMax()} size={box.GetSize()}"
                except: bb = "err"
                dv.append({"path": str(prim.GetPath()), "visibility": str(vis), "purpose": str(purp),
                           "material": mat_path, "bbox": bb, "active": prim.IsActive()})
            r["door_visibility"][f"Door_{dn}"] = dv

        # ── BBox sample ──
        try:
            wmesh = stage.GetPrimAtPath("/World/envs/env_0/room/Meshes/Base/Wall_0000/Geom/A63")
            if wmesh.IsValid():
                bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_,
                                UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
                bound = bbox_cache.ComputeWorldBound(wmesh)
                box = bound.ComputeAlignedBox()
                r["bbox"]["wall_A63"] = f"min={box.GetMin()} max={box.GetMax()} size={box.GetSize()}"
        except Exception as e: r["bbox"]["wall_A63"] = f"err:{e}"

        return r

    registry.register("_diagnose_stage", _diagnose_stage_tool,
                      spec={"name": "_diagnose_stage", "description": "Stage collision diagnosis"})

    def reset_env_tool(parameters: dict) -> dict:
        """重置环境（自动采集用，避免每次重启仿真）。

        - env.reset()：底盘/机械臂/场景回到初始状态
        - 显式把机械臂写回起始位姿（与启动那段一致，防止 reset 事件没落到默认位姿）
        - 可选把 grasp_test_cube_1 挪到指定 XY（刚体要走 write_root_pose_to_sim）
        - 最后空跑若干步让 PhysX 稳定
        """
        if not isinstance(parameters, dict):
            parameters = {}
        settle_steps = int(parameters.get("settle_steps", 30))
        env.reset()

        # 底盘归位：env.reset() 只触发 cfg 里 mode="reset" 的事件（那边只有关节/马桶/碰撞，
        # 没有任何 root 位姿的事件），所以底盘不会被重置。底盘稳定器只管高度/姿态、不管 XY
        # 位置，机械臂的反作用力会把它一点点推走，多轮累积。这里显式写回初始 root 位姿。
        root_state = robot.data.default_root_state.detach().clone()
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        root_pos0 = [round(float(v), 5) for v in root_state[0, :3]]

        _arm_ids = _arm_joint_ids(robot)
        _arm_event = getattr(getattr(env.cfg, "events", None), "init_arm_joint_position", None)
        _arm_params = getattr(_arm_event, "params", None)
        _home = _arm_params.get("default_pos") if isinstance(_arm_params, dict) else None
        if _home is None:
            _home = robot.data.default_joint_pos[0, _arm_ids].tolist()
        _home_q = torch.as_tensor(_home, device=robot.device, dtype=robot.data.joint_pos.dtype).view(1, -1)
        robot.write_joint_state_to_sim(_home_q, torch.zeros_like(_home_q), joint_ids=_arm_ids)
        robot.set_joint_position_target(_home_q, joint_ids=_arm_ids)

        cube_xy = parameters.get("cube_xy")
        cube_pos_w = None
        if cube_xy is not None and len(cube_xy) == 2:
            try:
                cube = env.scene["grasp_test_cube_1"]
            except Exception:
                cube = None
            if cube is not None:
                z = float(parameters.get("cube_z", 0.021))
                pose = torch.tensor(
                    [[float(cube_xy[0]), float(cube_xy[1]), z, 1.0, 0.0, 0.0, 0.0]],
                    device=robot.device, dtype=torch.float32,
                )
                cube.write_root_pose_to_sim(pose)
                cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=robot.device))
                cube_pos_w = [float(cube_xy[0]), float(cube_xy[1]), z]
                # ⚠️ 光写物理不够：gripper_prismatic_execute_grab 的 paper_cleanup_mode 是读
                # grasp_test_cube_1 的 **USD 世界变换**来定位抓取目标的。只改物理不改 USD，
                # 规划就会仍然指向方块的原位置 → 机械臂去空处 → 夹不到。
                try:
                    from pxr import Gf, UsdGeom
                    prim = _stage.GetPrimAtPath("/World/envs/env_0/grasp_test_cube_1")
                    if prim.IsValid():
                        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
                        if ops:
                            ops[0].Set(Gf.Vec3d(float(cube_xy[0]), float(cube_xy[1]), z))
                        else:
                            UsdGeom.Xformable(prim).AddTranslateOp().Set(
                                Gf.Vec3d(float(cube_xy[0]), float(cube_xy[1]), z)
                            )
                        print(f"[ResetEnv] USD xform of grasp_test_cube_1 set to {cube_pos_w}", flush=True)
                    else:
                        print("[ResetEnv] ⚠️ grasp_test_cube_1 prim 未找到，USD 变换没更新", flush=True)
                except Exception as exc:
                    print(f"[ResetEnv] ⚠️ 写 USD 变换失败（规划可能仍指向旧位置）: {exc}", flush=True)
            else:
                print("[ResetEnv] grasp_test_cube_1 not found in scene; cube_xy ignored.", flush=True)

        # 垃圾桶：**每次 reset 都写回标称位姿**。
        # 机械臂抓取/投放时可能把桶撞歪或撞走；桶的位置不对会让 drop_target（固定坐标）
        # 打偏、入桶判定和后续数据都失真。标称位姿 (2.097,-1.095) 与马桶碰撞体重叠约
        # 6cm，settle 时 PhysX 可能把它往外挤一点，所以复位后再 settle，并记录最终位置。
        bin_pos0 = None
        bin_nominal = None
        try:
            bin_obj = env.scene["trash_bin"]
        except Exception:
            bin_obj = None
        if bin_obj is not None:
            try:
                st = bin_obj.data.default_root_state.detach().clone()
                bin_nominal = [round(float(v), 5) for v in st[0, :3]]
                before = [round(float(v), 5) for v in bin_obj.data.root_pose_w[0, :3]]
                bin_obj.write_root_pose_to_sim(st[:, :7])
                bin_obj.write_root_velocity_to_sim(st[:, 7:])
                print(
                    f"[TrashBinReset] 垃圾桶复位 before={before} nominal={bin_nominal}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[TrashBinReset] 读/复位垃圾桶失败: {exc}", flush=True)

        for _ in range(max(1, settle_steps)):
            _step(env, render=True)

        # 报 settle 之后的实际位姿，并检查是否被马桶挤走太多
        if bin_obj is not None:
            try:
                bp = bin_obj.data.root_pose_w[0]
                bin_pos0 = [round(float(v), 5) for v in bp[:3]]
                if bin_nominal is not None:
                    drift = sum((a - b) ** 2 for a, b in zip(bin_pos0, bin_nominal)) ** 0.5
                    if drift > 0.05:
                        print(
                            f"[TrashBinReset] ⚠️ 垃圾桶 settle 后偏离标称 {drift:.3f}m "
                            f"(pos={bin_pos0}, nominal={bin_nominal})",
                            flush=True,
                        )
            except Exception:
                pass

        print(
            f"[ResetEnv] done base_pos_w={root_pos0} cube_pos_w={cube_pos_w} "
            f"bin_pos_w={bin_pos0} settle_steps={settle_steps}",
            flush=True,
        )
        return {
            "reset": True,
            "base_pos_w": root_pos0,
            "cube_pos_w": cube_pos_w,
            "bin_pos_w": bin_pos0,
            "settle_steps": settle_steps,
        }

    registry.register(
        "reset_env",
        reset_env_tool,
        spec={
            "name": "reset_env",
            "description": (
                "Reset the whole environment (env.reset + arm back to home pose + optional cube "
                "teleport) without restarting the simulator. Used between demos for auto collection."
            ),
            "parameters": {
                "cube_xy": {
                    "required": False,
                    "type": "array[float]",
                    "length": 2,
                    "description": "Optional world XY to teleport grasp_test_cube_1 to.",
                },
                "cube_z": {"required": False, "type": "float", "default": 0.021},
                "settle_steps": {"required": False, "type": "int", "default": 30},
            },
        },
    )

    server, server_thread = serve_bridge_in_thread(
        bridge,
        host=args_cli.tool_api_host,
        port=args_cli.tool_api_port,
    )
    print(
        f"[ToolAPI] listening=http://{args_cli.tool_api_host}:{args_cli.tool_api_port} "
        f"tools={registry.names()} default_target_frame=world",
        flush=True,
    )
    idle_loop_step = 0
    try:
        while simulation_app.is_running():
            job = _complete_pending_switch_robot_job()
            if job is not None:
                print(
                    f"[ToolAPI] request_id={job.request_id} tool={job.tool_name} state={job.state} "
                    f"execution_output={job.execution_output} error={job.error}",
                    flush=True,
                )
                continue
            job = bridge.process_next()
            if job is not None:
                print(
                    f"[ToolAPI] request_id={job.request_id} tool={job.tool_name} state={job.state} "
                    f"execution_output={job.execution_output} error={job.error}",
                    flush=True,
                )
                # Sync paper-waste proxy positions for the next tool call
                try:
                    _arm_base_pos_w, _arm_base_quat_w = _body_pose_w(robot, base_link.replace("-", "_"))
                    _sync_paper_waste_proxies(_stage, (_arm_base_pos_w, _arm_base_quat_w))
                except Exception as _sync_exc:
                    print(f"[PaperWasteSync] ERROR: {_sync_exc}", flush=True)
                continue
            idle_loop_step += 1
            _hold_base_stationary(robot)
            _step(env, render=not args_cli.headless)
            # GPT P1 v2: sync carried paper visuals to tool_tip world pose
            _update_carried_paper_visuals(_stage, robot)
            time.sleep(0.01)
    finally:
        server.should_exit = True
        server_thread.join(timeout=5.0)


def main() -> None:
    global _DEMO_RECORDER
    # 常驻 cuRobo worker 进程在程序退出时必须清理，否则残留进程继续占用 GPU 显存。
    import atexit

    atexit.register(_shutdown_serve_worker)

    spec = gym.spec(args_cli.task)
    env_cfg_entry = spec.kwargs["env_cfg_entry_point"]
    module_path, cls_name = env_cfg_entry.rsplit(":", 1)
    env_cfg = getattr(importlib.import_module(module_path), cls_name)()
    _apply_common_initial_robot_profile(env_cfg)
    _apply_arm_drive_overrides_to_env_cfg(env_cfg, label="pre_env_create")
    if args_cli.record_video:
        env_cfg.viewer.resolution = tuple(args_cli.video_resolution)

    # 相机渲染：
    # - 抗锯齿必须用 FXAA（纯空间）：cam_fixed 有"亚毫米抖动强制每帧重渲染"补丁，
    #   DLSS/TAA 这类时域 AA 会因时域历史被每帧重置而闪烁/变空白/出红绿伪影。
    # - spp + DL 去噪：减少直接光照噪点。spp 越高越慢越吃显存，cuRobo 只分 ~4GB，可降到 4。
    env_cfg.sim.render.antialiasing_mode = "FXAA"
    env_cfg.sim.render.samples_per_pixel = 8
    env_cfg.sim.render.enable_dl_denoiser = True

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.record_video else None)
    if args_cli.record_dataset:
        from curobo_pose_plan_utils.hdf5_demo_recorder import HDF5DemoRecorder

        _DEMO_RECORDER = HDF5DemoRecorder(
            args_cli.record_dataset,
            enable_cameras=args_cli.record_cameras,
            flush_every=args_cli.record_flush_every,
            max_frames_per_demo=args_cli.record_max_frames_per_demo,
            stride=args_cli.record_stride,
            cam_follows_tool=bool(getattr(env_cfg, "cam_follows_tool", False)),
            cam_t_in_tool=tuple(getattr(env_cfg, "cam_t_in_tool", _CAM_T_IN_TOOL)),
            cam_q_in_tool=_CAM_Q_IN_TOOL,
        )
        print(
            f"[HDF5DemoRecorder] recording enabled -> {args_cli.record_dataset} "
            f"(cameras={args_cli.record_cameras})",
            flush=True,
        )
    video_recorder = None
    try:
        video_recorder = _attach_video_recorder(env)
        env.reset()
        robot = env.scene["robot"]
        _apply_arm_drive_overrides(robot, label="post_env_reset")
        # 起始机械臂位姿：启动阶段 _step 直接 sim.step()，不经过 action manager，actuator 的
        # 位置目标缓冲保持初始值 → 机械臂被驱到 0 位（直着向上），表现为"打开环境时抬升的过程"。
        # 这里按 cfg 里 init_arm_joint_position 的 default_pos 显式写一次关节状态 + 位置目标。
        _arm_ids = _arm_joint_ids(robot)
        _arm_event = getattr(getattr(env.cfg, "events", None), "init_arm_joint_position", None)
        _arm_params = getattr(_arm_event, "params", None)
        _home_arm = _arm_params.get("default_pos") if isinstance(_arm_params, dict) else None
        if _home_arm is None:
            _home_arm = robot.data.default_joint_pos[0, _arm_ids].tolist()
        _home_arm_q = torch.as_tensor(
            _home_arm, device=robot.device, dtype=robot.data.joint_pos.dtype
        ).view(1, -1)
        robot.write_joint_state_to_sim(_home_arm_q, torch.zeros_like(_home_arm_q), joint_ids=_arm_ids)
        robot.set_joint_position_target(_home_arm_q, joint_ids=_arm_ids)
        _reset_link5_camera_mount(env)
        _sync_link5_camera(env, robot)
        _sync_fixed_camera_pose(env)
        # 固定相机 cam_fixed 渲染补丁：执行阶段走 env.step()（内部 sim.step(render=True) 渲染），
        # 不会经过自定义 _step 里的同步。包装 env.step + env.sim.render，保证所有步进路径
        # 渲染前都重同步 cam_fixed（含亚毫米抖动，强制 render product 每帧重渲染）。
        _orig_env_step = env.step
        def _env_step_with_fixed_cam(action):
            _sync_fixed_camera_pose(env)
            return _orig_env_step(action)
        env.step = _env_step_with_fixed_cam
        _orig_sim_render = env.sim.render
        def _sim_render_with_fixed_cam(*args, **kwargs):
            _sync_fixed_camera_pose(env)
            return _orig_sim_render(*args, **kwargs)
        env.sim.render = _sim_render_with_fixed_cam
        base_stabilizer = _install_static_motion_base_stabilizer(env, robot)
        base_stabilizer.apply("post_env_reset")

        settled = _wait_until_static_motion_base_settled(
            env,
            robot,
            base_stabilizer,
            max_wait_time_s=args_cli.settle_timeout_s,
        )
        print(f"[Settle] settled={settled}", flush=True)

        base_stabilizer.apply("post_settle")
        _sync_link5_camera(env, robot)
        if not args_cli.headless:
            env.sim.render()
            _show_camera_windows(env)
        if video_recorder is not None:
            video_recorder.capture(force=True)

        robot_usd_path = _scene_robot_usd_path(env_cfg)
        if robot_usd_path:
            print(f"[CuroboProfile] scene_robot_usd_path={robot_usd_path}", flush=True)
        (
            urdf_path,
            robot_cfg_path,
            base_link,
            tool_frame,
            joint_names,
            default_target_quat_wxyz,
            plan_end_effector_roll_rad,
            pre_target_joint_pos_deg,
        ) = _resolve_curobo_profile(robot_usd_path)
        args_cli.urdf_path = str(urdf_path)
        isaac_base_body = base_link.replace("-", "_")
        if args_cli.disable_auto_curobo_robot_cfg:
            print(f"[CuroboRobotCfg] auto regeneration disabled; using {robot_cfg_path}", flush=True)
        else:
            robot_cfg_path = _regenerate_curobo_robot_cfg_from_urdf(
                urdf_path,
                robot_cfg_path,
                collision_sphere_buffer=args_cli.curobo_collision_sphere_buffer,
                base_link=base_link,
                tool_frame=tool_frame,
                joint_names=joint_names,
            )

        arm_ids = _arm_joint_ids(robot)
        _validate_joint_value_count(args_cli.initial_arm_joint_pos, name="--initial_arm_joint_pos", joint_names=joint_names)
        _validate_joint_value_count(pre_target_joint_pos_deg, name="--pre_target_joint_pos_deg", joint_names=joint_names)
        requested_initial_q = None
        if args_cli.initial_arm_joint_pos is not None:
            requested_initial_q = torch.tensor(
                args_cli.initial_arm_joint_pos,
                device=robot.device,
                dtype=robot.data.joint_pos.dtype,
            )
        q_initial = robot.data.joint_pos[0, arm_ids].detach().clone()
        pre_target_q = _deg_to_rad_joint_values(
            pre_target_joint_pos_deg,
            device=robot.device,
            dtype=robot.data.joint_pos.dtype,
        )
        print(
            f"[PreTargetPose] enabled={args_cli.use_pre_target_joint_pose} "
            f"q_deg={list(pre_target_joint_pos_deg)} q_rad={_format_tensor(pre_target_q)}",
            flush=True,
        )
        arm_base_pos_w, arm_base_quat_w = _body_pose_w(robot, isaac_base_body)
        tool_pos_w, tool_quat_w = _tool_tip_pose_w(robot)
        current_pos_base, current_quat_base = subtract_frame_transforms(
            arm_base_pos_w.reshape(1, 3),
            arm_base_quat_w.reshape(1, 4),
            tool_pos_w.reshape(1, 3),
            tool_quat_w.reshape(1, 4),
        )

        use_builtin_pose_tests = not _is_gripper_joint_profile()
        target_pos_explicit = _option_was_passed("--target_pos")
        world_pose_tests = [] if args_cli.tool_api_server or not use_builtin_pose_tests else list(DEFAULT_WORLD_POSE_TESTS)
        if args_cli.pose_names.strip() and not args_cli.tool_api_server:
            if not use_builtin_pose_tests:
                raise ValueError("--pose_names uses the built-in brush pose list and is not enabled for this task profile.")
            selected_pose_names = [name.strip() for name in args_cli.pose_names.split(",") if name.strip()]
            pose_by_name = {name: (name, pos, quat) for name, pos, quat in world_pose_tests}
            missing_pose_names = [name for name in selected_pose_names if name not in pose_by_name]
            if missing_pose_names:
                raise ValueError(f"Unknown pose_names: {missing_pose_names}")
            world_pose_tests = [pose_by_name[name] for name in selected_pose_names]
            print(f"[PoseSelect] running poses={selected_pose_names}", flush=True)
        elif not args_cli.tool_api_server and use_builtin_pose_tests:
            print("[PoseSelect] running all built-in poses", flush=True)
        elif not args_cli.tool_api_server:
            print("[PoseSelect] built-in brush poses disabled for this task profile; pass --target_pos.", flush=True)
        add_cli_target = (
            args_cli.target_pos is not None
            and not args_cli.tool_api_server
            and (target_pos_explicit or not use_builtin_pose_tests)
        )
        if add_cli_target:
            target_quat = list(args_cli.target_quat_wxyz or default_target_quat_wxyz)
            world_pose_tests.append(("cli_target", list(args_cli.target_pos), target_quat))
            print(
                f"[PoseSelect] added cli_target target_pos={list(args_cli.target_pos)} "
                f"target_quat_wxyz={target_quat} defaulted={args_cli.target_quat_wxyz is None}",
                flush=True,
            )
        if not args_cli.tool_api_server and not world_pose_tests:
            raise ValueError("No pose targets selected. Pass --target_pos X Y Z for this task profile.")

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        scene_mesh_obstacles = _collect_curobo_scene_mesh_obstacles(
            stage,
            isaac_base_body=isaac_base_body,
            env=env,
            base_pose_w=(arm_base_pos_w, arm_base_quat_w),
            target_positions_w=[list(pos_w) for _, pos_w, _ in world_pose_tests],
        )
        # 懒加载：首次工具调用时 PhysX raycast 扫描墙壁
        _wall_cfg = _scene_obstacle_config()
        _ensure_raycast_wall_cuboids_impl(stage, (arm_base_pos_w, arm_base_quat_w), wall_mode=_wall_cfg.wall_obstacle_mode)
        global _CUROBO_SCENE_CUBOID_OBSTACLES
        _CUROBO_SCENE_CUBOID_OBSTACLES = _collect_curobo_scene_cuboid_obstacles(
            stage,
            base_pose_w=(arm_base_pos_w, arm_base_quat_w),
        )
        _export_curobo_scene_collision_obj(
            scene_mesh_obstacles=scene_mesh_obstacles,
            scene_cuboid_obstacles=_CUROBO_SCENE_CUBOID_OBSTACLES,
        )
        if requested_initial_q is not None:
            print(
                f"[InitialArmPose] moving to requested initial_arm_joint_pos={_format_tensor(requested_initial_q)} "
                "via cuRobo cspace",
                flush=True,
            )
            moved_to_initial = _plan_and_execute_cspace_target(
                env=env,
                robot=robot,
                case_name="initial_arm_pose",
                phase_name="requested_initial",
                urdf_path=urdf_path,
                robot_cfg_path=robot_cfg_path,
                base_link=base_link,
                tool_frame=tool_frame,
                joint_names=joint_names,
                scene_mesh_obstacles=scene_mesh_obstacles,
                target_q=requested_initial_q,
                final_settle=True,
            )
            if not moved_to_initial:
                raise RuntimeError("Failed to reach --initial_arm_joint_pos with cuRobo C-space planning.")
            q_initial = robot.data.joint_pos[0, arm_ids].detach().clone()

        print("base_link:", base_link, flush=True)
        print("isaac_base_body:", isaac_base_body, flush=True)
        print("tool_frame:", tool_frame, flush=True)
        print("joint_names:", joint_names, flush=True)
        print("urdf_path:", str(urdf_path), flush=True)
        print("curobo_robot_cfg_path:", str(robot_cfg_path), flush=True)
        print("default_target_quat_wxyz:", default_target_quat_wxyz, flush=True)
        print("plan_end_effector_roll_rad:", plan_end_effector_roll_rad, flush=True)
        print("initial_joint_pos:", _format_tensor(q_initial), flush=True)
        print("arm_base_pos_w:", _format_tensor(arm_base_pos_w), flush=True)
        print("arm_base_quat_wxyz:", _format_tensor(arm_base_quat_w), flush=True)
        print("current_tool_tip_pos_w:", _format_tensor(tool_pos_w), flush=True)
        print("current_tool_tip_quat_wxyz:", _format_tensor(tool_quat_w), flush=True)
        print("current_tool_tip_pos_base_from_isaac:", _format_tensor(current_pos_base), flush=True)

        if args_cli.tool_api_server:
            _run_tool_api_server(
                env=env,
                robot=robot,
                arm_ids=arm_ids,
                urdf_path=urdf_path,
                robot_cfg_path=robot_cfg_path,
                base_link=base_link,
                tool_frame=tool_frame,
                joint_names=joint_names,
                scene_mesh_obstacles=scene_mesh_obstacles,
                q_initial=q_initial,
                pre_target_q=pre_target_q,
                default_target_quat_wxyz=default_target_quat_wxyz,
            )
            return

        if args_cli.multi_pose_polynomial_seed:
            print(
                "[MultiPoseBranch] enter "
                f"pose_count={len(world_pose_tests)} "
                f"pose_names={[name for name, _, _ in world_pose_tests]} "
                f"use_pre_target={bool(args_cli.use_pre_target_joint_pose)} "
                f"seed_mode={args_cli.multi_pose_seed_mode}",
                flush=True,
            )
            if target_pos_explicit:
                raise ValueError("--multi_pose_polynomial_seed currently uses --pose_names/built-in poses; omit --target_pos.")
            if not world_pose_tests:
                raise ValueError("--multi_pose_polynomial_seed requires at least one selected pose.")
            actual_q_before = robot.data.joint_pos[0, arm_ids].detach().clone()
            if args_cli.use_pre_target_joint_pose:
                print(
                    f"[PreTargetPose] case=multi_pose_polynomial_seed moving to q={_format_tensor(pre_target_q)} "
                    "via cuRobo cspace",
                    flush=True,
                )
                moved_to_pre_target = _plan_and_execute_cspace_target(
                    env=env,
                    robot=robot,
                    case_name="multi_pose_polynomial_seed",
                    phase_name="pre_target",
                    urdf_path=urdf_path,
                    robot_cfg_path=robot_cfg_path,
                    base_link=base_link,
                    tool_frame=tool_frame,
                    joint_names=joint_names,
                    scene_mesh_obstacles=scene_mesh_obstacles,
                    target_q=pre_target_q,
                    final_settle=True,
                )
                if not moved_to_pre_target:
                    raise RuntimeError("Failed to reach pre-target pose with cuRobo C-space planning.")
            q_initial = robot.data.joint_pos[0, arm_ids].detach().clone()
            print(
                f"[PlanStart] case=multi_pose_polynomial_seed planner_start_q={_format_tensor(q_initial)} "
                f"sim_q_before_plan={_format_tensor(actual_q_before)} poses={[name for name, _, _ in world_pose_tests]}",
                flush=True,
            )
            original_pose_tests = world_pose_tests
            world_pose_tests = _downsample_ordered_pose_tests(world_pose_tests, max_count=8)
            if len(world_pose_tests) != len(original_pose_tests):
                print(
                    f"[MultiPoseSample] input_count={len(original_pose_tests)} selected_count={len(world_pose_tests)} "
                    f"selected_poses={[name for name, _, _ in world_pose_tests]}",
                    flush=True,
                )
            target_poses = [
                {
                    "name": name,
                    "target_frame": "world",
                    "target_pos": list(pos_w),
                }
                for name, pos_w, quat_w in world_pose_tests
            ]
            result = _plan_multi_pose_polynomial_seed(
                case_name="multi_pose_polynomial_seed",
                env=env,
                robot=robot,
                urdf_path=urdf_path,
                robot_cfg_path=robot_cfg_path,
                base_link=base_link,
                tool_frame=tool_frame,
                joint_names=joint_names,
                q_start=q_initial,
                scene_mesh_obstacles=scene_mesh_obstacles,
                target_poses=target_poses,
                seed_mode=args_cli.multi_pose_seed_mode,
            )
            print(
                f"[MultiPoseTest] success={result['success']} elapsed_s={float(result['elapsed_s']):.3f}",
                flush=True,
            )
            if not result["success"]:
                _plan_and_execute_cspace_target(
                    env=env,
                    robot=robot,
                    case_name="multi_pose_polynomial_seed",
                    phase_name="failed_return_pre_target",
                    urdf_path=urdf_path,
                    robot_cfg_path=robot_cfg_path,
                    base_link=base_link,
                    tool_frame=tool_frame,
                    joint_names=joint_names,
                    scene_mesh_obstacles=scene_mesh_obstacles,
                    target_q=pre_target_q,
                    final_settle=True,
                )
                print(
                    f"[Return] case=multi_pose_polynomial_seed failed returned_to_pre_target "
                    f"actual_q={_format_tensor(robot.data.joint_pos[0, arm_ids].detach())}",
                    flush=True,
                )
                return

            traj = torch.tensor(result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
            if traj.dim() > 2:
                traj = traj.reshape(-1, traj.shape[-1])
            print(
                f"[MultiPoseTest] trajectory_shape={tuple(traj.shape)} "
                f"first={_format_tensor(traj[0])} last={_format_tensor(traj[-1])}",
                flush=True,
            )
            exec_traj = traj.clone()
            if plan_end_effector_roll_rad is not None:
                roll_start_idx, roll_end_idx = _multi_pose_target_waypoint_range(result, int(exec_traj.shape[0]))
                exec_traj = _apply_roll_delta_over_waypoint_range(
                    robot,
                    exec_traj,
                    roll_rad=float(plan_end_effector_roll_rad),
                    start_idx=roll_start_idx,
                    end_idx=roll_end_idx,
                    label="multi_pose_polynomial_seed:end_effector_roll",
                )
                print(
                    f"[EndEffectorRollWholeTrajectory] case=multi_pose_polynomial_seed "
                    f"waypoints={exec_traj.shape[0]} roll_start_waypoint={roll_start_idx} roll_end_waypoint={roll_end_idx} "
                    f"original_final_q={_format_tensor(traj[-1])} rolled_final_q={_format_tensor(exec_traj[-1])} "
                    f"roll_delta={float(plan_end_effector_roll_rad):.6f}",
                    flush=True,
                )
            _execute_trajectory(env, robot, exec_traj, label="multi_pose_polynomial_seed:direct")
            if args_cli.final_hold_steps > 0:
                _hold_joint_target_if_close(
                    env,
                    robot,
                    exec_traj[-1],
                    args_cli.final_hold_steps,
                    label="multi_pose_polynomial_seed:final_hold",
                )
            print(
                f"[Stop] case=multi_pose_polynomial_seed held_final_pose target_q={_format_tensor(exec_traj[-1])} "
                f"actual_q={_format_tensor(robot.data.joint_pos[0, arm_ids].detach())}",
                flush=True,
            )
            return

        total_success = 0
        total_failed = 0
        plan_time_records = []
        for index, (case_name, pos_w, quat_w) in enumerate(world_pose_tests, start=1):
            actual_q_before = robot.data.joint_pos[0, arm_ids].detach().clone()
            if args_cli.use_pre_target_joint_pose:
                print(
                    f"[PreTargetPose] case={case_name} moving to q={_format_tensor(pre_target_q)} "
                    "via cuRobo cspace",
                    flush=True,
                )
                moved_to_pre_target = _plan_and_execute_cspace_target(
                    env=env,
                    robot=robot,
                    case_name=case_name,
                    phase_name="pre_target",
                    urdf_path=urdf_path,
                    robot_cfg_path=robot_cfg_path,
                    base_link=base_link,
                    tool_frame=tool_frame,
                    joint_names=joint_names,
                    scene_mesh_obstacles=scene_mesh_obstacles,
                    target_q=pre_target_q,
                    final_settle=True,
                )
                if not moved_to_pre_target:
                    raise RuntimeError(f"Failed to reach pre-target pose with cuRobo for case={case_name}.")
            q_initial = robot.data.joint_pos[0, arm_ids].detach().clone()
            print(
                f"[PlanStart] case={case_name} planner_start_q={_format_tensor(q_initial)} "
                f"sim_q_before_plan={_format_tensor(actual_q_before)}",
                flush=True,
            )
            if args_cli.use_pre_target_joint_pose:
                print(
                    f"[PreTargetPose] case={case_name} reached q={_format_tensor(q_initial)}",
                    flush=True,
                )
            target_pos_w = torch.tensor(pos_w, device=robot.device, dtype=torch.float32).reshape(1, 3)
            target_quat_w = torch.tensor(quat_w, device=robot.device, dtype=torch.float32).reshape(1, 4)
            target_pos_base, target_quat_base = subtract_frame_transforms(
                arm_base_pos_w.reshape(1, 3),
                arm_base_quat_w.reshape(1, 4),
                target_pos_w,
                target_quat_w,
            )
            print(
                f"[PoseTest {index}/{len(world_pose_tests)}] name={case_name} "
                f"target_pos_base={target_pos_base.reshape(-1).tolist()} "
                f"target_quat_base={target_quat_base.reshape(-1).tolist()}",
                flush=True,
            )
            worker_result = _plan_single_pose(
                case_name=case_name,
                phase_name="direct",
                urdf_path=urdf_path,
                robot_cfg_path=robot_cfg_path,
                base_link=base_link,
                tool_frame=tool_frame,
                joint_names=joint_names,
                q_start=q_initial,
                scene_mesh_obstacles=scene_mesh_obstacles,
                target_pos_base=target_pos_base,
                target_quat_base=target_quat_base,
                target_pos_world_z=float(target_pos_w.reshape(-1)[2].item()),
                env=env,
                robot=robot,
            )
            result = worker_result
            elapsed_s = float(result["elapsed_s"])
            plan_time_records.append((case_name, bool(result["success"]), elapsed_s))
            print(
                f"[PoseTest {index}/{len(world_pose_tests)}] success={result['success']} "
                f"elapsed_s={elapsed_s:.3f}",
                flush=True,
            )
            print(
                f"[PlanTime] case={case_name} success={result['success']} elapsed_s={elapsed_s:.6f}",
                flush=True,
            )
            if result.get("metrics"):
                print(f"[PlannerMetrics] case={case_name} metrics={result['metrics']}", flush=True)
            if not result["success"]:
                total_failed += 1
                _plan_and_execute_cspace_target(
                    env=env,
                    robot=robot,
                    case_name=case_name,
                    phase_name="failed_return_pre_target",
                    urdf_path=urdf_path,
                    robot_cfg_path=robot_cfg_path,
                    base_link=base_link,
                    tool_frame=tool_frame,
                    joint_names=joint_names,
                    scene_mesh_obstacles=scene_mesh_obstacles,
                    target_q=pre_target_q,
                    final_settle=True,
                )
                print(
                    f"[Return] case={case_name} skipped/failed returned_to_pre_target "
                    f"actual_q={_format_tensor(robot.data.joint_pos[0, arm_ids].detach())}",
                    flush=True,
                )
                continue

            traj = torch.tensor(result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
            if traj.dim() > 2:
                traj = traj.reshape(-1, traj.shape[-1])
            print(
                f"[PoseTest {index}/{len(world_pose_tests)}] trajectory_shape={tuple(traj.shape)} "
                f"first={_format_tensor(traj[0])} last={_format_tensor(traj[-1])}",
                flush=True,
            )

            final_waypoint = traj[-1]
            exec_traj = traj.clone()
            roll_target_q = None
            if plan_end_effector_roll_rad is not None:
                requested_roll_target_q = final_waypoint.detach().clone()
                requested_roll_target_q[-1] = requested_roll_target_q[-1] + float(plan_end_effector_roll_rad)
                roll_target_q = _clamp_arm_joint_target_to_soft_limits(
                    robot,
                    requested_roll_target_q,
                    label=f"{case_name}:end_effector_roll",
                )
                print(
                    f"[EndEffectorRollInline] case={case_name} "
                    f"roll_rad={float(plan_end_effector_roll_rad):.6f} "
                    f"planned_final_q={_format_tensor(final_waypoint)} "
                    f"requested_roll_target_q={_format_tensor(requested_roll_target_q)} "
                    f"roll_target_q={_format_tensor(roll_target_q)}",
                    flush=True,
                )
            _execute_trajectory(env, robot, exec_traj, label=f"{case_name}:direct")

            if roll_target_q is not None:
                _execute_direct_joint_roll(
                    env=env,
                    robot=robot,
                    target_q=roll_target_q,
                    label=f"{case_name}:end_effector_roll",
                    final_hold_steps=int(args_cli.final_hold_steps),
                )
            elif args_cli.final_hold_steps > 0:
                _hold_joint_target_if_close(
                    env,
                    robot,
                    exec_traj[-1],
                    args_cli.final_hold_steps,
                    label=f"{case_name}:final_hold",
                )
                actual_q = robot.data.joint_pos[0, arm_ids].detach()
                joint_err_norm, joint_err_max, joint_err = _joint_error(exec_traj[-1], actual_q)
                print(
                    f"[ExecuteHold] case={case_name} final_hold_steps={args_cli.final_hold_steps} "
                    f"joint_err_norm={joint_err_norm.item():.6f} "
                    f"joint_err_max={joint_err_max.item():.6f} "
                    f"joint_err={_format_tensor(joint_err)}"
                    f"{_arm_control_debug_suffix(robot, target_q=exec_traj[-1])}",
                    flush=True,
                )

            total_success += 1
            final_pos, final_quat = _tool_tip_pose_w(robot)
            final_pos_base, final_quat_base = subtract_frame_transforms(
                arm_base_pos_w.reshape(1, 3),
                arm_base_quat_w.reshape(1, 4),
                final_pos.reshape(1, 3),
                final_quat.reshape(1, 4),
            )
            print("final_tool_tip_pos_w:", _format_tensor(final_pos), flush=True)
            print("final_tool_tip_quat_wxyz:", _format_tensor(final_quat), flush=True)
            print("final_tool_tip_pos_base:", _format_tensor(final_pos_base), flush=True)
            print("final_tool_tip_quat_base:", _format_tensor(final_quat_base), flush=True)
            pos_err_w, rot_err_rad_w, rot_err_deg_w = _pose_error(
                target_pos_w,
                target_quat_w,
                final_pos.reshape(1, 3),
                final_quat.reshape(1, 4),
            )
            pos_err_base, rot_err_rad_base, rot_err_deg_base = _pose_error(
                target_pos_base,
                target_quat_base,
                final_pos_base,
                final_quat_base,
            )
            print(
                f"[PoseError] case={case_name} "
                f"world_pos_m={pos_err_w.item():.6f} "
                f"world_rot_rad={rot_err_rad_w.item():.6f} "
                f"world_rot_deg={rot_err_deg_w.item():.3f} "
                f"base_pos_m={pos_err_base.item():.6f} "
                f"base_rot_rad={rot_err_rad_base.item():.6f} "
                f"base_rot_deg={rot_err_deg_base.item():.3f}",
                flush=True,
            )

            if _is_gripper_joint_profile():
                post_target_descend_q = None
                post_target_pre_descend_q = robot.data.joint_pos[0, arm_ids].detach().clone()
                descend_m = _post_target_descend_m()
                if descend_m > 0.0:
                    post_target_descend_q = _execute_post_target_world_descend(
                        env=env,
                        robot=robot,
                        case_name=case_name,
                        urdf_path=urdf_path,
                        robot_cfg_path=robot_cfg_path,
                        base_link=base_link,
                        tool_frame=tool_frame,
                        joint_names=joint_names,
                        scene_mesh_obstacles=scene_mesh_obstacles,
                        isaac_base_body=isaac_base_body,
                        descend_m=descend_m,
                    )
                _execute_gripper_post_target_sequence(
                    env=env,
                    robot=robot,
                    case_name=case_name,
                    urdf_path=urdf_path,
                    robot_cfg_path=robot_cfg_path,
                    base_link=base_link,
                    tool_frame=tool_frame,
                    joint_names=joint_names,
                    scene_mesh_obstacles=scene_mesh_obstacles,
                    isaac_base_body=isaac_base_body,
                    pre_target_q=pre_target_q,
                    post_target_pre_descend_q=post_target_pre_descend_q,
                    post_target_descend_q=post_target_descend_q,
                )
                continue

            if args_cli.pause_after_execute_s > 0:
                print(
                    f"[Pause] case={case_name} sleeping {args_cli.pause_after_execute_s:.2f}s before return",
                    flush=True,
                )
                time.sleep(args_cli.pause_after_execute_s)

            if args_cli.return_via_reverse_trajectory:
                print(f"[Return] case={case_name} replaying reverse trajectory", flush=True)
                _execute_trajectory(env, robot, torch.flip(traj, dims=(0,)), label=f"{case_name}:reverse_return")
            else:
                _plan_and_execute_cspace_target(
                    env=env,
                    robot=robot,
                    case_name=case_name,
                    phase_name="return_pre_target",
                    urdf_path=urdf_path,
                    robot_cfg_path=robot_cfg_path,
                    base_link=base_link,
                    tool_frame=tool_frame,
                    joint_names=joint_names,
                    scene_mesh_obstacles=scene_mesh_obstacles,
                    target_q=pre_target_q,
                    final_settle=True,
                )
            returned_q = robot.data.joint_pos[0, arm_ids].detach()
            print(
                f"[Return] case={case_name} returned_to_pre_target target_q={_format_tensor(pre_target_q)} "
                f"actual_q={_format_tensor(returned_q)}",
                flush=True,
            )

        print(
            f"[Summary] pose_tests={len(world_pose_tests)} success={total_success} failed={total_failed}",
            flush=True,
        )
        if plan_time_records:
            total_plan_time_s = sum(record[2] for record in plan_time_records)
            success_plan_times = [record[2] for record in plan_time_records if record[1]]
            avg_plan_time_s = total_plan_time_s / len(plan_time_records)
            avg_success_plan_time_s = (
                sum(success_plan_times) / len(success_plan_times) if success_plan_times else float("nan")
            )
            print(
                f"[PlanTimeSummary] count={len(plan_time_records)} "
                f"success_count={len(success_plan_times)} "
                f"total_s={total_plan_time_s:.6f} "
                f"avg_s={avg_plan_time_s:.6f} "
                f"avg_success_s={avg_success_plan_time_s:.6f}",
                flush=True,
            )

        if not args_cli.headless:
            print("[GUI] close the viewer to exit.", flush=True)
            while simulation_app.is_running():
                _hold_base_stationary(robot)
                _step(env, render=True)
    finally:
        if video_recorder is not None:
            video_recorder.close()
        if _DEMO_RECORDER is not None:
            # 收尾最后一条未闭合的 demo（完整 SOP 结束时再退出的场景）
            _DEMO_RECORDER.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[Exit] interrupted by KeyboardInterrupt before normal completion.", file=sys.stderr, flush=True)
        raise
    except SystemExit as exc:
        print(f"[Exit] SystemExit before normal completion code={exc.code}", file=sys.stderr, flush=True)
        raise
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr, flush=True)
        raise
