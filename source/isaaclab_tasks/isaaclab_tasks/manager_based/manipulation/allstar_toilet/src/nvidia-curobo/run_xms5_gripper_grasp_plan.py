import argparse

import torch

from curobo._src.types.robot import RobotCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose


URDF_PATH = (
    "/home/stardust/haozhao/pro/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/"
    "manager_based/manipulation/allstar_toilet/data/robots/chassis_arm/"
    "motion_wheel_arm_simple_sphere_urdf/mecanum730_xms5_gripper_joint.urdf"
)

BASE_LINK = "XMS5-R800-W4G3B4C_base"
TOOL_FRAME = "gripper_tcp_link"
EXPECTED_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def make_joint_state(q, joint_names):
    return JointState.from_position(
        torch.tensor([q], device="cuda", dtype=torch.float32),
        joint_names=joint_names,
    )


def bool_tensor_value(value):
    return value is not None and bool(value.any().item())


def flatten_positions(joint_state, dof):
    if joint_state is None or joint_state.position is None:
        return None
    return joint_state.position.reshape(-1, dof)


def flatten_valid_positions(joint_state, dof, last_tstep=None):
    positions = flatten_positions(joint_state, dof)
    if positions is None:
        return None
    if last_tstep is None:
        return positions
    valid_count = int(last_tstep.reshape(-1)[0].item()) + 1
    return positions[:valid_count]


def pose_error(actual_pose, target_position, target_quaternion):
    pos_err = torch.linalg.norm(actual_pose.position - target_position, dim=-1)
    quat_dot = torch.abs(torch.sum(actual_pose.quaternion * target_quaternion, dim=-1))
    quat_dot = torch.clamp(quat_dot, -1.0, 1.0)
    rot_err = 2.0 * torch.acos(quat_dot)
    return pos_err, rot_err


def offset_pose(position, quaternion, axis, offset, in_tool_frame):
    offset_pose_value = Pose.from_list(
        [
            offset * value
            for value in {
                "x": [1.0, 0.0, 0.0],
                "y": [0.0, 1.0, 0.0],
                "z": [0.0, 0.0, 1.0],
            }[axis]
        ]
        + [1.0, 0.0, 0.0, 0.0]
    )
    base_pose = Pose(position=position, quaternion=quaternion)
    if in_tool_frame:
        return base_pose.multiply(offset_pose_value)
    return offset_pose_value.multiply(base_pose)


def make_planner(args):
    robot_cfg = RobotCfg.from_basic(
        urdf_path=args.urdf,
        base_link=args.base_link,
        tool_frames=[args.tool_frame],
    )
    planner_cfg = MotionPlannerCfg.create(
        robot=robot_cfg,
        self_collision_check=False,
        num_ik_seeds=args.num_ik_seeds,
        num_trajopt_seeds=args.num_trajopt_seeds,
        use_cuda_graph=False,
    )
    planner = MotionPlanner(planner_cfg)
    planner.warmup(enable_graph=False, num_warmup_iterations=3)
    return planner


def plan_pose_to_single_goal(planner, tool_frame, target_pose, start_state):
    goal_pose = GoalToolPose.from_poses(
        {
            tool_frame: Pose(
                position=target_pose.position,
                quaternion=target_pose.quaternion,
            )
        },
        ordered_tool_frames=planner.tool_frames,
        num_goalset=1,
    )
    result = planner.plan_pose(goal_pose, start_state)
    if result is None or not bool_tensor_value(result.success):
        return None
    return result.get_interpolated_plan()


def run_grasp_attempt(planner, grasp_poses, start, args, approach_axis, approach_offset,
                      approach_in_tool_frame, lift_axis, lift_offset, lift_in_tool_frame):
    return planner.plan_grasp(
        grasp_poses=grasp_poses,
        current_state=start,
        grasp_approach_axis=approach_axis,
        grasp_approach_offset=approach_offset,
        grasp_approach_in_tool_frame=approach_in_tool_frame,
        grasp_lift_axis=lift_axis,
        grasp_lift_offset=lift_offset,
        grasp_lift_in_tool_frame=lift_in_tool_frame,
        plan_approach_to_grasp=True,
        plan_grasp_to_lift=not args.skip_lift,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test cuRobo grasp planning for the XMS5 gripper TCP."
    )
    parser.add_argument("--urdf", default=URDF_PATH, help="Path to the gripper URDF.")
    parser.add_argument("--base-link", default=BASE_LINK, help="cuRobo kinematic base link.")
    parser.add_argument("--tool-frame", default=TOOL_FRAME, help="Tool frame to plan.")
    parser.add_argument("--num-ik-seeds", type=int, default=64)
    parser.add_argument("--num-trajopt-seeds", type=int, default=8)
    parser.add_argument("--approach-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--approach-offset", type=float, default=0.10)
    parser.add_argument("--lift-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--lift-offset", type=float, default=0.10)
    parser.add_argument(
        "--scan-approaches",
        action="store_true",
        help="Try common approach axes, signs, offsets, and frame modes.",
    )
    parser.add_argument(
        "--scan-offsets",
        type=float,
        nargs="+",
        default=[0.02, 0.03, 0.05, 0.08, 0.10],
        help="Positive offset magnitudes to scan when --scan-approaches is set.",
    )
    parser.add_argument(
        "--fresh-planner-per-scan",
        action="store_true",
        help="Create a new MotionPlanner for every scanned approach candidate.",
    )
    parser.add_argument(
        "--skip-lift",
        action="store_true",
        help="Only test approach and grasp stages.",
    )
    parser.add_argument(
        "--approach-in-world",
        action="store_true",
        help="Apply approach offset in base/world frame instead of tool frame.",
    )
    parser.add_argument(
        "--lift-in-world",
        action="store_true",
        help="Apply lift offset in base/world frame instead of tool frame.",
    )
    parser.add_argument(
        "--start-q",
        type=float,
        nargs=6,
        default=[0.0, -0.6, 0.8, 0.0, 0.4, 0.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Start arm joint values in radians.",
    )
    parser.add_argument(
        "--target-q",
        type=float,
        nargs=6,
        default=[0.25, -0.9, 1.0, 0.2, 0.2, 0.35],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Joint values used only to generate a reachable grasp TCP pose.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cuRobo grasp planning needs a CUDA device.")

    planner = make_planner(args)

    kin_cfg = KinematicsCfg.from_basic_urdf(
        urdf_path=args.urdf,
        base_link=args.base_link,
        tool_frames=[args.tool_frame],
    )
    kin = Kinematics(kin_cfg)

    print("urdf_path:", args.urdf)
    print("base_link:", args.base_link)
    print("tool_frames:", planner.tool_frames)
    print("joint_names:", planner.joint_names)
    print("dof:", kin.get_dof())

    if planner.joint_names != EXPECTED_JOINT_NAMES:
        raise RuntimeError(
            f"Unexpected active joint chain. Expected {EXPECTED_JOINT_NAMES}, got {planner.joint_names}"
        )

    start = make_joint_state(args.start_q, planner.joint_names)
    target_q = make_joint_state(args.target_q, planner.joint_names)
    target_fk = kin.compute_kinematics(target_q)
    grasp_pose = target_fk.tool_poses.get_link_pose(args.tool_frame)

    grasp_poses = GoalToolPose.from_poses(
        {
            args.tool_frame: Pose(
                position=grasp_pose.position,
                quaternion=grasp_pose.quaternion,
            )
        },
        ordered_tool_frames=planner.tool_frames,
        num_goalset=1,
    )

    print("start_q:", args.start_q)
    print("target_q_for_grasp_fk:", args.target_q)
    print("grasp_position_in_base:", grasp_pose.position)
    print("grasp_quaternion_wxyz:", grasp_pose.quaternion)
    print("approach_axis:", args.approach_axis)
    print("approach_offset:", args.approach_offset)
    print("approach_in_tool_frame:", not args.approach_in_world)
    print("lift_axis:", args.lift_axis)
    print("lift_offset:", args.lift_offset)
    print("lift_in_tool_frame:", not args.lift_in_world)

    if args.scan_approaches:
        print("scan_approaches: True")
        print("fresh_planner_per_scan:", args.fresh_planner_per_scan)
        print("skip_lift:", args.skip_lift)
        first_success = None
        for frame_label, approach_in_tool_frame in [
            ("tool", True),
            ("world", False),
        ]:
            for axis in ["x", "y", "z"]:
                for offset_abs in args.scan_offsets:
                    for sign in [1.0, -1.0]:
                        approach_offset = sign * offset_abs
                        attempt_planner = planner
                        attempt_start = start
                        if args.fresh_planner_per_scan:
                            attempt_planner = make_planner(args)
                            attempt_start = make_joint_state(args.start_q, attempt_planner.joint_names)
                        result = run_grasp_attempt(
                            attempt_planner,
                            grasp_poses,
                            attempt_start,
                            args,
                            axis,
                            approach_offset,
                            approach_in_tool_frame,
                            args.lift_axis,
                            args.lift_offset,
                            not args.lift_in_world,
                        )
                        success = bool_tensor_value(result.success)
                        print(
                            "scan_result:",
                            f"frame={frame_label}",
                            f"axis={axis}",
                            f"approach_offset={approach_offset:.3f}",
                            f"success={success}",
                            f"approach={bool_tensor_value(result.approach_success)}",
                            f"grasp={bool_tensor_value(result.grasp_success)}",
                            f"lift={bool_tensor_value(result.lift_success)}",
                            f"status={result.status}",
                        )
                        if success and first_success is None:
                            first_success = (
                                frame_label,
                                axis,
                                approach_offset,
                                attempt_planner,
                                result,
                            )
                        elif args.fresh_planner_per_scan:
                            attempt_planner.destroy()
        if first_success is None:
            raise RuntimeError("No scanned grasp approach succeeded.")
        frame_label, axis, approach_offset, planner, result = first_success
        print(
            "best_scan_success:",
            f"frame={frame_label}",
            f"axis={axis}",
            f"approach_offset={approach_offset:.3f}",
        )
    else:
        result = run_grasp_attempt(
            planner,
            grasp_poses,
            start,
            args,
            args.approach_axis,
            args.approach_offset,
            not args.approach_in_world,
            args.lift_axis,
            args.lift_offset,
            not args.lift_in_world,
        )

    success = bool_tensor_value(result.success)
    print("grasp_planning_success:", success)
    print("status:", result.status)
    print("goalset_index:", result.goalset_index)
    print("approach_success:", bool_tensor_value(result.approach_success))
    print("grasp_success:", bool_tensor_value(result.grasp_success))
    print("lift_success:", bool_tensor_value(result.lift_success))

    dof = kin.get_dof()
    approach_positions = flatten_valid_positions(
        result.approach_interpolated_trajectory,
        dof,
        result.approach_interpolated_last_tstep,
    )
    grasp_positions = flatten_valid_positions(
        result.grasp_interpolated_trajectory,
        dof,
        result.grasp_interpolated_last_tstep,
    )
    lift_positions = flatten_valid_positions(
        result.lift_interpolated_trajectory,
        dof,
        result.lift_interpolated_last_tstep,
    )

    if approach_positions is not None:
        print("approach_trajectory_shape:", tuple(approach_positions.shape))
        print("approach_last_joint_waypoint:", approach_positions[-1])
    if grasp_positions is not None:
        print("grasp_trajectory_shape:", tuple(grasp_positions.shape))
        print("grasp_last_joint_waypoint:", grasp_positions[-1])
    if lift_positions is not None:
        print("lift_trajectory_shape:", tuple(lift_positions.shape))
        print("lift_last_joint_waypoint:", lift_positions[-1])

    if success and args.skip_lift:
        print("OK: grasp planning reached approach and grasp poses for the gripper TCP.")
        return

    if not success or lift_positions is None:
        if approach_positions is not None:
            approach_end = JointState.from_position(
                approach_positions[-1:].contiguous(),
                joint_names=planner.joint_names,
            )
            fallback_grasp_plan = plan_pose_to_single_goal(
                planner,
                args.tool_frame,
                grasp_pose,
                approach_end,
            )
            fallback_grasp_positions = flatten_positions(fallback_grasp_plan, dof)
            print(
                "fallback_approach_to_grasp_plan_pose_success:",
                fallback_grasp_positions is not None,
            )
            if fallback_grasp_positions is not None:
                print(
                    "fallback_approach_to_grasp_trajectory_shape:",
                    tuple(fallback_grasp_positions.shape),
                )
                print(
                    "fallback_approach_to_grasp_last_joint_waypoint:",
                    fallback_grasp_positions[-1],
                )
        raise RuntimeError("Grasp motion planning failed.")

    final_q = lift_positions[-1:].contiguous()
    final_fk = kin.compute_kinematics(
        JointState.from_position(final_q, joint_names=planner.joint_names)
    )
    final_pose = final_fk.tool_poses.get_link_pose(args.tool_frame)
    expected_lift_pose = offset_pose(
        grasp_pose.position,
        grasp_pose.quaternion,
        args.lift_axis,
        args.lift_offset,
        not args.lift_in_world,
    )
    pos_err, rot_err = pose_error(
        final_pose,
        expected_lift_pose.position,
        expected_lift_pose.quaternion,
    )

    print("final_lift_position_in_base:", final_pose.position)
    print("final_lift_quaternion_wxyz:", final_pose.quaternion)
    print("expected_lift_position_in_base:", expected_lift_pose.position)
    print("expected_lift_quaternion_wxyz:", expected_lift_pose.quaternion)
    print("final_lift_position_error_mm:", pos_err * 1000.0)
    print("final_lift_rotation_error_rad:", rot_err)
    print("OK: grasp planning reached approach, grasp, and lift poses for the gripper TCP.")


if __name__ == "__main__":
    main()
