"""Teleoperation + demonstration recording for mobile manipulator tasks.

针对移动机械臂（底盘 + 机械臂）的专用录制脚本。
支持 `keyboard` / `spacemouse` 两种输入方式。

键盘映射
--------
    M           — 切换控制模式（底盘 ↔ 机械臂）

  [底盘模式]
    W / S       — 前进 / 后退   (vx)
    A / D       — 向左平移 / 向右平移  (vy，麦克纳姆横移)
    Z / X       — 左转 / 右转   (wz)

  [机械臂模式]（增量叠加，再次按键继续累积）
    W / S       — joint1 增 / 减（底座旋转）
    A / D       — joint2 增 / 减（肩关节）
    Q / E       — joint3 增 / 减（肘关节）
    Z / X       — joint4 增 / 减（腕关节旋转）
    T / G       — joint5 增 / 减（腕关节俯仰）
    C / V       — joint6 增 / 减（末端滚转）

  [通用]
    R           — 重置环境（清空机械臂积分位置）
    P           — 推进任务阶段: prepare → start → deal → finish

麦轮逆运动学（简化）
--------------------
    wheel_fl = vx - vy - wz
    wheel_fr = vx + vy + wz
    wheel_rl = vx + vy - wz
    wheel_rr = vx - vy + wz

Usage
-----
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \\
    --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Gripper-v0 \\
    --num_demos 10 \\
    --dataset_file /tmp/mecanum730_xms5_gripper_demos.hdf5 \\
    --speed_scale 20.0 \\
    --arm_scale 0.05
"""

import contextlib
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

ALLSTAR_TOILET_ROOT = Path(__file__).resolve().parents[1]
if str(ALLSTAR_TOILET_ROOT) not in sys.path:
    sys.path.insert(0, str(ALLSTAR_TOILET_ROOT))

from mobile_arm_record_utils.record_mobile_arm_args import (
    build_record_mobile_arm_parser,
    finalize_record_mobile_arm_args,
)
from mobile_arm_record_utils.record_mobile_arm_tasks import (
    _GRIPPER_JOINT_PRISMATIC_SPHERE_RAW_USD_LIGHT_TASK_ID,
    _MOTION_BASED_PHYSX_ROOT_VELOCITY_TASKS,
    _MOTION_BASED_TASKS,
    _PHYSX_ROOT_FIXED_Z,
    _PRINT_LINK6_TO_IK_EE_LOCAL_TASKS,
    _PRISMATIC_GRIPPER_KEYBOARD_TASKS,
    _SENSOR_IO_DISABLED_TASKS,
    _SPHERE_RAW_USD_LIGHT_TASK_ID,
    _STRICT_MODE_SPLIT_TASKS,
    _TOOL_TIP_OFFSET_Z_IN_TOOL,
    _USE_TRUE_TOOL_TIP_FOR_YELLOW_FRAME_TASKS,
)

parser = build_record_mobile_arm_parser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
_quiet_runtime_debug = finalize_record_mobile_arm_args(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
_IS_HEADLESS = bool(getattr(app_launcher, "_headless", False))
# ── Isaac Sim 启动完成后才能导入以下模块 ─────────────────────────────────────

import torch
import cv2
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

import isaaclab_tasks.manager_based.manipulation.allstar_toilet  # noqa: F401

from isaaclab_tasks.manager_based.manipulation.allstar_toilet.wrappers.record_wrapper import RecordWrapper
from isaaclab_tasks.manager_based.manipulation.allstar_toilet.utils.interaction import RateLimiter

from mobile_arm_record_utils.record_mobile_arm_actions import _assemble_mobile_arm_action
from mobile_arm_record_utils.record_mobile_arm_arm import (
    _ARM_KEYBOARD_INCREMENTS,
    _absolute_arm_joint_positions_to_action,
    _get_arm_joint_action_term,
    _resync_joint_position_action_offset,
)
from mobile_arm_record_utils.record_mobile_arm_base import (
    _MotionBasedWheelVisualizer,
    _apply_motion_based_base,
    _apply_motion_based_base_velocity,
    _legacy_operator_to_motion_based_twist,
    _make_upright_root_pose,
    _make_wheel_action,
    _physx_root_stabilization_velocity,
    _quat_wxyz_to_yaw_tensor,
)
from mobile_arm_record_utils.record_mobile_arm_debug import (
    _dump_mobile_arm_runtime_debug,
    _print_arm_joint_step_debug,
    print_mobile_arm_control_status,
    print_mobile_arm_pose_line,
    print_mobile_arm_step_pose_line,
)
from mobile_arm_record_utils.record_mobile_arm_input import (
    _create_teleop,
    print_mobile_arm_teleop_help,
)
from mobile_arm_record_utils.record_mobile_arm_math import (
    _quat_wxyz_conjugate,
    _quat_wxyz_to_rpy_deg,
)
from mobile_arm_record_utils.record_mobile_arm_sensors import (
    CameraDisplayRuntime,
    ObservationChunkRecorder,
    acquire_debug_draw_interface,
    draw_viewport_sensor_debug,
)

_MODE_CHASSIS = 0
_MODE_ARM     = 1
_mode = _MODE_CHASSIS


def _toggle_mode():
    global _mode
    _mode = _MODE_ARM if _mode == _MODE_CHASSIS else _MODE_CHASSIS
    label = "机械臂" if _mode == _MODE_ARM else "底盘"
    print(f"\r[模式切换] → {label}控制", flush=True)


def main():
    global _mode

    task_cfg = {
        "record": {
            "dataset_file": args_cli.dataset_file,
            "num_demos":    args_cli.num_demos,
        },
    }

    # ── 解析 env_cfg ──────────────────────────────────────────────────────────
    spec = gym.spec(args_cli.task)
    env_cfg_entry = spec.kwargs["env_cfg_entry_point"]
    module_path, cls_name = env_cfg_entry.rsplit(":", 1)
    import importlib
    env_cfg = getattr(importlib.import_module(module_path), cls_name)()

    if args_cli.episode_length_s > 0:
        env_cfg.episode_length_s = args_cli.episode_length_s
    if _quiet_runtime_debug:
        env_cfg.quiet_runtime_debug = True

    RecordWrapper.crop_env_cfg(env_cfg, task_cfg)

    # ── 创建环境 ──────────────────────────────────────────────────────────────
    base_env = ManagerBasedRLEnv(env_cfg)
    # MotionWheelArmSimpleUsd 任务默认通过 disable_arm_gravity.usda overlay
    # 给机械臂 link 写入 physxRigidBody:disableGravity=1；RawUsd 版本保留原始 USD。
    env      = RecordWrapper(base_env, task_cfg)

    obs_recorder = ObservationChunkRecorder()

    # ── 遥操作设备（默认 keyboard，可切换 spacemouse）──────────────────────────
    # 机械臂关节积分位置（相对默认位置的偏移，单位 rad）
    arm_offsets = torch.zeros(1, 6, device=env.device)
    use_prismatic_gripper_keyboard = args_cli.task in _PRISMATIC_GRIPPER_KEYBOARD_TASKS
    if use_prismatic_gripper_keyboard:
        gripper_joint1_pos = float(args_cli.prismatic_gripper_pos)
        gripper_joint1_pos = max(
            float(args_cli.prismatic_gripper_min),
            min(float(args_cli.prismatic_gripper_max), gripper_joint1_pos),
        )
    else:
        gripper_joint1_pos = float(args_cli.gripper_joint1_pos)
    gripper_action_dim = 0
    for term_name, term_dim in zip(base_env.action_manager.active_terms, base_env.action_manager.action_term_dim):
        if term_name == "gripper_joint_pos":
            gripper_action_dim = int(term_dim)
            break

    def _make_gripper_cmd() -> torch.Tensor | None:
        if gripper_action_dim == 1:
            return torch.tensor([[gripper_joint1_pos]], dtype=torch.float32, device=env.device)
        if gripper_action_dim >= 2:
            return torch.tensor([[gripper_joint1_pos, -gripper_joint1_pos]], dtype=torch.float32, device=env.device)
        return None

    gripper_cmd = _make_gripper_cmd()
    reset_requested = {"pending": False}
    arm_key_prev_held = {key: False for key in _ARM_KEYBOARD_INCREMENTS}

    def _request_env_reset():
        nonlocal gripper_joint1_pos, gripper_cmd
        arm_offsets.zero_()
        if use_prismatic_gripper_keyboard:
            gripper_joint1_pos = max(
                float(args_cli.prismatic_gripper_min),
                min(float(args_cli.prismatic_gripper_max), float(args_cli.prismatic_gripper_pos)),
            )
            gripper_cmd = _make_gripper_cmd()
        reset_requested["pending"] = True
        env.reset_state.next()

    def _sync_arm_key_prev_state():
        if args_cli.teleop_device != "keyboard":
            return
        for _key in arm_key_prev_held:
            arm_key_prev_held[_key] = teleop.is_held(_key)

    teleop = _create_teleop(args_cli.teleop_device, args_cli.keyboard_hold_sec)

    # P 键防抖：evdev 回调在独立线程中触发，连按/长按会在同一个仿真步内多次调用
    # task_state.next()，导致状态机循环多圈。加 1.0s 冷却时间确保每次只前进一格。
    _p_last_time = [0.0]
    _P_DEBOUNCE_SEC = 1.0
    def _advance_task_state():
        now = time.monotonic()
        if now - _p_last_time[0] < _P_DEBOUNCE_SEC:
            print(f"\n[P防抖] 忽略过快的 P 键（距上次 {now - _p_last_time[0]:.2f}s）", flush=True)
            return
        _p_last_time[0] = now
        print(f"\n[P] task_state: {env.task_state.state} → next", flush=True)
        env.task_state.next()

    if args_cli.teleop_device == "keyboard":
        teleop.add_callback("M", _toggle_mode)
        teleop.add_callback("R", _request_env_reset)
        teleop.add_callback("P", _advance_task_state)
    else:
        teleop.add_callback("L", _toggle_mode)
        teleop.add_callback("R", _request_env_reset)
        teleop.add_callback("LR", env.task_state.next)

    rate_limiter = RateLimiter(args_cli.step_hz) if args_cli.step_hz > 0 else None

    env.reset()
    if hasattr(base_env.cfg, "_cam_link5_mount_offset"):
        delattr(base_env.cfg, "_cam_link5_mount_offset")
    _debug_robot = base_env.scene["robot"]
    _debug_arm_joint_ids, _debug_arm_joint_names = _debug_robot.find_joints("joint[1-6]", preserve_order=True)
    _debug_arm_action_term = _get_arm_joint_action_term(base_env)
    if not _quiet_runtime_debug:
        print(
            f"[ActionDebug] active_terms={base_env.action_manager.active_terms} "
            f"dims={base_env.action_manager.action_term_dim} "
            f"total={base_env.action_manager.total_action_dim}",
            flush=True,
        )
        _dump_mobile_arm_runtime_debug(
            "after_reset",
            base_env,
            _debug_robot,
            _debug_arm_joint_ids,
            _debug_arm_joint_names,
            _debug_arm_action_term,
            env_cfg=env_cfg,
            task_id=args_cli.task,
            include_usd=True,
        )

    _draw, _has_debug_draw = acquire_debug_draw_interface(
        is_headless=_IS_HEADLESS,
        quiet_runtime_debug=_quiet_runtime_debug,
    )
    print_mobile_arm_teleop_help(
        args_cli.teleop_device,
        use_prismatic_gripper_keyboard=args_cli.task in _PRISMATIC_GRIPPER_KEYBOARD_TASKS,
    )

    _step = 0
    s = teleop.SENSITIVITY
    use_motion_based_base = args_cli.task in _MOTION_BASED_TASKS
    use_physx_root_velocity_base = args_cli.task in _MOTION_BASED_PHYSX_ROOT_VELOCITY_TASKS
    strict_mode_split = args_cli.task in _STRICT_MODE_SPLIT_TASKS
    use_prismatic_gripper_keyboard = args_cli.task in _PRISMATIC_GRIPPER_KEYBOARD_TASKS
    robot = base_env.scene["robot"]
    cam = base_env.scene.sensors.get("cam_tool", None)
    if cam is None:
        with contextlib.suppress(KeyError):
            cam = base_env.scene["cam_tool"]
    lidar = base_env.scene.sensors.get("lidar", None)
    if cam is None and not _quiet_runtime_debug:
        print("[SensorLoad] cam_tool is disabled.", flush=True)
    if lidar is None and not _quiet_runtime_debug:
        print("[SensorLoad] lidar is disabled.", flush=True)
    sensor_io_enabled = args_cli.task not in _SENSOR_IO_DISABLED_TASKS
    if not sensor_io_enabled and not _quiet_runtime_debug:
        print("[SensorLoad] RGBD/LiDAR readout and display are disabled for this task.", flush=True)
    camera_display = CameraDisplayRuntime(is_headless=_IS_HEADLESS)
    print(
        f"[CameraWindow] sensor_io_enabled={sensor_io_enabled} headless={_IS_HEADLESS} cam_tool_loaded={cam is not None}",
        flush=True,
    )
    arm_joint_ids, arm_joint_names = robot.find_joints("joint[1-6]", preserve_order=True)
    arm_action_term = _get_arm_joint_action_term(base_env)
    last_arm_q = robot.data.joint_pos[:, arm_joint_ids].detach().clone()
    last_arm_q_source_step = -1
    arm_command_q = last_arm_q.clone()
    arm_command_q_source_step = -1
    last_arm_input_debug_t = 0.0

    def _clamp_arm_command(target_q: torch.Tensor) -> torch.Tensor:
        with contextlib.suppress(Exception):
            limits = robot.data.soft_joint_pos_limits[:, arm_joint_ids, :].to(device=target_q.device, dtype=target_q.dtype)
            lower = limits[..., 0]
            upper = limits[..., 1]
            valid = torch.isfinite(lower) & torch.isfinite(upper) & (upper > lower)
            clamped = torch.max(torch.min(target_q, upper), lower)
            return torch.where(valid, clamped, target_q)
        return target_q

    def _write_arm_command_to_action_buffer():
        arm_offsets[:] = _absolute_arm_joint_positions_to_action(
            robot, arm_joint_ids, arm_command_q, arm_action_term
        )

    def _sync_arm_command_to_current_pose(source_step: int) -> None:
        nonlocal arm_command_q, arm_command_q_source_step
        arm_command_q = _clamp_arm_command(robot.data.joint_pos[:, arm_joint_ids].detach().clone())
        arm_command_q_source_step = source_step
        _write_arm_command_to_action_buffer()

    if strict_mode_split:
        _resync_joint_position_action_offset(robot, arm_action_term)
    _sync_arm_command_to_current_pose(-1)
    _sync_arm_key_prev_state()
    motion_based_wheel_visualizer = (
        _MotionBasedWheelVisualizer(robot, write_joint_state=True)
        if use_motion_based_base
        else None
    )
    motion_base_nominal_z = robot.data.root_link_pose_w[:, 2].clone()
    if use_physx_root_velocity_base:
        motion_base_nominal_z = torch.full_like(motion_base_nominal_z, _PHYSX_ROOT_FIXED_Z)
    motion_base_yaw_target = _quat_wxyz_to_yaw_tensor(robot.data.root_link_pose_w[:, 3:7]).clone()
    if use_physx_root_velocity_base:
        # Direct GPU PhysX rejects articulation link setGlobalPose during simulation.
        # Keep this variant velocity-only; avoid pose writes after env.reset().
        robot.write_root_velocity_to_sim(
            torch.zeros((robot.num_instances, 6), device=env.device, dtype=robot.data.joint_pos.dtype)
        )
    last_base_pose_w, _ = _make_upright_root_pose(
        robot.data.root_link_pose_w.clone(), motion_base_nominal_z, motion_base_yaw_target
    )
    last_base_pose_source_step = -1
    zero_root_velocity_w = torch.zeros((robot.num_instances, 6), device=env.device, dtype=robot.data.joint_pos.dtype)
    prev_mode = _mode
    base_hold_debug_t = 0.0

    def _hold_motion_based_base_pose(reason: str = "") -> None:
        """Lock the motion-based chassis root to the cached base pose."""
        nonlocal motion_base_nominal_z, motion_base_yaw_target, base_hold_debug_t
        if not use_motion_based_base:
            return

        current_pose_w = robot.data.root_link_pose_w.detach().clone()
        pos_drift = torch.linalg.norm(current_pose_w[:, :3] - last_base_pose_w[:, :3], dim=-1).max()
        yaw_drift = torch.max(
            torch.abs(_quat_wxyz_to_yaw_tensor(current_pose_w[:, 3:7]) - _quat_wxyz_to_yaw_tensor(last_base_pose_w[:, 3:7]))
        )

        if not use_physx_root_velocity_base:
            motion_base_nominal_z = last_base_pose_w[:, 2].clone()
        motion_base_yaw_target = _quat_wxyz_to_yaw_tensor(last_base_pose_w[:, 3:7]).clone()
        if not use_physx_root_velocity_base:
            robot.write_root_pose_to_sim(last_base_pose_w)
            root_vel_w = zero_root_velocity_w
        else:
            correction_vel_w, _ = _physx_root_stabilization_velocity(current_pose_w, motion_base_nominal_z)
            root_vel_w = zero_root_velocity_w.clone()
            root_vel_w[:, 2] = correction_vel_w[:, 2]
            root_vel_w[:, 3] = correction_vel_w[:, 3]
            root_vel_w[:, 4] = correction_vel_w[:, 4]
        robot.write_root_velocity_to_sim(root_vel_w)

        now = time.monotonic()
        if (
            not _quiet_runtime_debug
            and (float(pos_drift.item()) > 1.0e-4 or float(yaw_drift.item()) > 1.0e-4)
            and now - base_hold_debug_t > 0.5
        ):
            base_hold_debug_t = now
            print(
                f"\n[BaseHold] arm mode locked chassis root"
                f"{f' ({reason})' if reason else ''}: "
                f"pos_drift={float(pos_drift.item()):.6f}m yaw_drift={float(yaw_drift.item()):.6f}rad",
                flush=True,
            )

    def _stabilize_physx_motion_base_pose(reason: str = "") -> None:
        """Damp vertical and roll/pitch root velocity without writing articulation poses."""
        nonlocal motion_base_yaw_target
        if not use_physx_root_velocity_base:
            return

        current_pose_w = robot.data.root_link_pose_w.detach().clone()
        correction_vel_w, yaw = _physx_root_stabilization_velocity(current_pose_w, motion_base_nominal_z)
        root_vel_w = robot.data.root_link_vel_w.detach().clone()
        root_vel_w[:, 2] = correction_vel_w[:, 2]
        root_vel_w[:, 3] = correction_vel_w[:, 3]
        root_vel_w[:, 4] = correction_vel_w[:, 4]

        robot.write_root_velocity_to_sim(root_vel_w)
        motion_base_yaw_target = yaw.detach().clone()

    def _cache_last_step_poses(source_step: int) -> None:
        """Cache the real pose after the just-finished control step."""
        nonlocal last_base_pose_w, last_base_pose_source_step
        nonlocal last_arm_q, last_arm_q_source_step
        nonlocal motion_base_nominal_z, motion_base_yaw_target

        last_arm_q = robot.data.joint_pos[:, arm_joint_ids].detach().clone()
        last_arm_q_source_step = source_step

        if use_motion_based_base:
            current_root_pose_w = robot.data.root_link_pose_w.detach().clone()
            if not use_physx_root_velocity_base:
                motion_base_nominal_z = current_root_pose_w[:, 2].clone()
            motion_base_yaw_target = _quat_wxyz_to_yaw_tensor(current_root_pose_w[:, 3:7]).clone()
            last_base_pose_w, _ = _make_upright_root_pose(
                current_root_pose_w, motion_base_nominal_z, motion_base_yaw_target
            )
            last_base_pose_w = last_base_pose_w.detach().clone()
            last_base_pose_source_step = source_step

    def _ensure_previous_step_pose_cache(current_step: int) -> None:
        """Guarantee the hold cache is from exactly the previous control step."""
        if current_step <= 0:
            return
        expected_step = current_step - 1
        stale = last_arm_q_source_step != expected_step
        stale = stale or (use_motion_based_base and last_base_pose_source_step != expected_step)
        if stale:
            if not _quiet_runtime_debug:
                print(
                    f"\n[HoldPose] cache stale at step={current_step}: "
                    f"base_step={last_base_pose_source_step}, arm_step={last_arm_q_source_step}, "
                    f"expected={expected_step}. Resyncing from current sim state.",
                    flush=True,
                )
            _cache_last_step_poses(expected_step)

    _cache_last_step_poses(-1)
    # dirt_shitstains = None
    # try:
    #     dirt_shitstains = base_env.scene["dirt_shitstains"]
    #     print("[Debug] Found scene entity: dirt_shitstains", flush=True)
    # except KeyError:
    #     print("[Debug] Scene entity missing: dirt_shitstains", flush=True)
    #     dirt_shitstains = None
    _base_idx = robot.data.body_names.index("base_link")
    _link6_idx = robot.data.body_names.index("link6")
    _tool_idx = robot.data.body_names.index("tool_link") if "tool_link" in robot.data.body_names else _link6_idx
    _tool_tip_idx = (
        robot.data.body_names.index("tool_tip_link")
        if "tool_tip_link" in robot.data.body_names
        else _tool_idx
    )
    _cam_mount_uses_link5 = args_cli.task in {
        _SPHERE_RAW_USD_LIGHT_TASK_ID,
        _GRIPPER_JOINT_PRISMATIC_SPHERE_RAW_USD_LIGHT_TASK_ID,
    }
    _link5_idx = None
    _cam_t_in_link5 = None
    _cam_q_in_link5 = None
    if _cam_mount_uses_link5:
        _link5_idx = robot.data.body_names.index("XMS5_R800_W4G3B4C_link5")
    if not _quiet_runtime_debug:
        print(
            f"[WorldPose] 实时输出世界坐标: base='{robot.data.body_names[_base_idx]}', "
            f"link6='{robot.data.body_names[_link6_idx]}', ee='{robot.data.body_names[_tool_tip_idx]}'",
            flush=True,
        )
    if _cam_mount_uses_link5:
        if not _quiet_runtime_debug:
            print(
                f"[CameraMount] cam_tool uses link5 mount body='{robot.data.body_names[_link5_idx]}' "
                "with world pose preserved from the original tool_link camera.",
                flush=True,
            )

    def _sync_cam_tool_pose():
        nonlocal _cam_t_in_link5, _cam_q_in_link5
        if cam is None:
            return None, None

        tool_pos_w_now = robot.data.body_pos_w[:, _tool_idx, :]
        tool_quat_w_now = robot.data.body_quat_w[:, _tool_idx, :]
        cam_offset = torch.tensor(
            [[0.6533, -0.2706, 0.2706, -0.6533]],
            dtype=tool_quat_w_now.dtype,
            device=tool_quat_w_now.device,
        )
        cam_t_in_tool = torch.tensor(
            [getattr(base_env.cfg, "cam_t_in_tool", (0.08, 0.0, 0.01))],
            dtype=tool_quat_w_now.dtype,
            device=tool_quat_w_now.device,
        )
        cam_pos_from_tool_w, cam_quat_from_tool_w = combine_frame_transforms(
            tool_pos_w_now, tool_quat_w_now, cam_t_in_tool, cam_offset
        )
        if _cam_mount_uses_link5:
            link5_pos_w = robot.data.body_pos_w[:, _link5_idx, :]
            link5_quat_w = robot.data.body_quat_w[:, _link5_idx, :]
            if _cam_t_in_link5 is None or _cam_q_in_link5 is None:
                _cam_t_in_link5, _cam_q_in_link5 = subtract_frame_transforms(
                    link5_pos_w, link5_quat_w, cam_pos_from_tool_w, cam_quat_from_tool_w
                )
                if not _quiet_runtime_debug:
                    print(
                        "[CameraMount] computed fixed link5->cam offset "
                        f"t={_cam_t_in_link5[0].detach().cpu().tolist()} "
                        f"q={_cam_q_in_link5[0].detach().cpu().tolist()}",
                        flush=True,
                    )
            cam_pos_w_now, cam_quat_w_now = combine_frame_transforms(
                link5_pos_w, link5_quat_w, _cam_t_in_link5, _cam_q_in_link5
            )
        else:
            cam_pos_w_now = cam_pos_from_tool_w
            cam_quat_w_now = cam_quat_from_tool_w
        cam.set_world_poses(cam_pos_w_now, cam_quat_w_now, convention="ros")
        return cam_pos_w_now, cam_quat_w_now

    with contextlib.suppress(KeyboardInterrupt):
        while simulation_app.is_running():
            with torch.inference_mode():
                _ensure_previous_step_pose_cache(_step)

                if strict_mode_split:
                    _resync_joint_position_action_offset(robot, arm_action_term)
                    _write_arm_command_to_action_buffer()

                if strict_mode_split and _mode != prev_mode:
                    _sync_arm_key_prev_state()
                    if use_motion_based_base and motion_based_wheel_visualizer is not None:
                        motion_based_wheel_visualizer.sync_from_sim()
                    prev_mode = _mode

                delta = torch.zeros(1, 6, device=env.device)
                if _mode == _MODE_CHASSIS:
                    if args_cli.teleop_device == "keyboard":
                        vx = teleop.get_axis("W", "S") * s * args_cli.speed_scale
                        vy = teleop.get_axis("A", "D") * s * args_cli.speed_scale
                        wz = teleop.get_axis("Z", "X") * s * args_cli.speed_scale
                    else:
                        base_x, base_y, base_yaw = teleop.get_base_axes()
                        vx = base_x * s * args_cli.speed_scale
                        vy = base_y * s * args_cli.speed_scale
                        wz = base_yaw * s * args_cli.speed_scale
                    if use_motion_based_base:
                        vx_body, vy_body, wz_body = _legacy_operator_to_motion_based_twist(vx, vy, wz)
                        if use_physx_root_velocity_base:
                            wheel_vel, motion_base_yaw_target, _ = _apply_motion_based_base_velocity(
                                robot,
                                vx_body,
                                vy_body,
                                wz_body,
                                env.device,
                                robot.data.root_link_pose_w.detach().clone(),
                                motion_base_nominal_z,
                                motion_base_yaw_target,
                            )
                        else:
                            wheel_vel, motion_base_yaw_target, _ = _apply_motion_based_base(
                                robot,
                                vx_body,
                                vy_body,
                                wz_body,
                                base_env.physics_dt,
                                env.device,
                                last_base_pose_w,
                                motion_base_nominal_z,
                                motion_base_yaw_target,
                            )
                        if motion_based_wheel_visualizer is not None:
                            motion_based_wheel_visualizer.apply(wheel_vel, base_env.physics_dt)
                    else:
                        wheel_vel = _make_wheel_action(vx, vy, wz, env.device)
                    arm_target_q = arm_command_q
                    arm_cmd = arm_offsets.to(env.device)
                else:
                    if args_cli.teleop_device == "keyboard":
                        for key, (joint_idx, direction) in _ARM_KEYBOARD_INCREMENTS.items():
                            held = teleop.is_held(key)
                            if held:
                                delta[0, joint_idx] += direction * args_cli.arm_scale
                            arm_key_prev_held[key] = held
                    else:
                        delta[0] = torch.from_numpy(teleop.get_arm_axes()).to(env.device) * args_cli.arm_scale
                    wheel_vel = torch.zeros(1, 4, device=env.device)
                    if strict_mode_split:
                        if use_motion_based_base:
                            _hold_motion_based_base_pose("pre_step")
                            if motion_based_wheel_visualizer is not None:
                                motion_based_wheel_visualizer.apply(wheel_vel, base_env.physics_dt)
                    if torch.any(delta != 0.0):
                        arm_command_q = _clamp_arm_command(arm_command_q + delta)
                        arm_command_q_source_step = _step
                    _write_arm_command_to_action_buffer()
                    if use_prismatic_gripper_keyboard and gripper_action_dim > 0 and args_cli.teleop_device == "keyboard":
                        gripper_delta = teleop.get_axis("I", "O") * float(args_cli.prismatic_gripper_step)
                        if gripper_delta != 0.0:
                            gripper_joint1_pos = max(
                                float(args_cli.prismatic_gripper_min),
                                min(float(args_cli.prismatic_gripper_max), gripper_joint1_pos + gripper_delta),
                            )
                            gripper_cmd = _make_gripper_cmd()
                    arm_target_q = arm_command_q
                    arm_cmd = arm_offsets
                    if torch.any(delta != 0.0) and not _quiet_runtime_debug:
                        now = time.monotonic()
                        if now - last_arm_input_debug_t > 0.25:
                            last_arm_input_debug_t = now
                            q_deg = torch.rad2deg(robot.data.joint_pos[:, arm_joint_ids])
                            target_deg = torch.rad2deg(arm_target_q)
                            print(
                                "\n[ArmInput] "
                                f"step={_step} cmd_source_step={arm_command_q_source_step} "
                                f"delta={[round(float(v), 4) for v in delta[0].detach().cpu().tolist()]} "
                                f"target_deg={[round(float(v), 2) for v in target_deg[0].detach().cpu().tolist()]} "
                                f"q_deg={[round(float(v), 2) for v in q_deg[0].detach().cpu().tolist()]} "
                                f"raw_action={[round(float(v), 4) for v in arm_cmd[0].detach().cpu().tolist()]}",
                                flush=True,
                            )

                # ── 强制将相机世界位姿同步到末端相机挂载体（刚性绑定）──────────
                # step 前同步一次，保证本步渲染使用当前 tool_link 对应的相机位姿。
                tool_pos_w  = robot.data.body_pos_w[:, _tool_idx, :]   # [1,3]
                tool_quat_w = robot.data.body_quat_w[:, _tool_idx, :]  # [1,4] wxyz
                tool_tip_pos_w = robot.data.body_pos_w[:, _tool_tip_idx, :]
                tool_tip_quat_w = robot.data.body_quat_w[:, _tool_tip_idx, :]
                cam_pos_w, cam_quat_w = _sync_cam_tool_pose()

                action = _assemble_mobile_arm_action(
                    base_env.action_manager,
                    wheel_vel,
                    arm_cmd,
                    env.device,
                    gripper_cmd=gripper_cmd,
                )
                robot.set_joint_position_target(arm_target_q.to(env.device), joint_ids=arm_joint_ids)
                obs, extra = env.step(action)
                env_step_count = _step + 1
                _stabilize_physx_motion_base_pose("post_step")
                if _mode == _MODE_ARM and use_motion_based_base:
                    _hold_motion_based_base_pose("post_step")
                _cache_last_step_poses(_step)
                if _step == 0 and not _quiet_runtime_debug:
                    _dump_mobile_arm_runtime_debug(
                        "after_first_step",
                        base_env,
                        robot,
                        arm_joint_ids,
                        arm_joint_names,
                        arm_action_term,
                        env_cfg=env_cfg,
                        task_id=args_cli.task,
                        action=action,
                        include_usd=False,
                    )
                if (
                    args_cli.arm_debug_interval > 0
                    and env_step_count % args_cli.arm_debug_interval == 0
                ):
                    _print_arm_joint_step_debug(env_step_count, robot, arm_joint_ids, arm_joint_names, arm_action_term)

                if reset_requested["pending"]:
                    if strict_mode_split:
                        _resync_joint_position_action_offset(robot, arm_action_term)
                    _cam_t_in_link5 = None
                    _cam_q_in_link5 = None
                    if hasattr(base_env.cfg, "_cam_link5_mount_offset"):
                        delattr(base_env.cfg, "_cam_link5_mount_offset")
                    _cache_last_step_poses(_step)
                    _sync_arm_command_to_current_pose(_step)
                    _sync_arm_key_prev_state()
                    if motion_based_wheel_visualizer is not None:
                        motion_based_wheel_visualizer.sync_from_sim()
                    reset_requested["pending"] = False

                base_pos_w = robot.data.body_pos_w[:, _base_idx, :]
                base_quat_w = robot.data.body_quat_w[:, _base_idx, :]
                base_pos = base_pos_w[0].cpu()
                base_quat = base_quat_w[0].cpu()
                base_rpy_deg = _quat_wxyz_to_rpy_deg(base_quat)
                arm_joint_pos = robot.data.joint_pos[0, arm_joint_ids].cpu()
                link6_pos_w = robot.data.body_pos_w[:, _link6_idx, :]
                link6_quat_w = robot.data.body_quat_w[:, _link6_idx, :]
                tool_pos_w = robot.data.body_pos_w[:, _tool_idx, :]
                tool_quat_w = robot.data.body_quat_w[:, _tool_idx, :]
                tool_tip_pos_w = robot.data.body_pos_w[:, _tool_tip_idx, :]
                tool_tip_quat_w = robot.data.body_quat_w[:, _tool_tip_idx, :]
                tool_z_w = torch.stack([
                    2 * (tool_quat_w[:, 1] * tool_quat_w[:, 3] + tool_quat_w[:, 0] * tool_quat_w[:, 2]),
                    2 * (tool_quat_w[:, 2] * tool_quat_w[:, 3] - tool_quat_w[:, 0] * tool_quat_w[:, 1]),
                    1 - 2 * (tool_quat_w[:, 1] ** 2 + tool_quat_w[:, 2] ** 2),
                ], dim=-1)
                if args_cli.task in _USE_TRUE_TOOL_TIP_FOR_YELLOW_FRAME_TASKS:
                    yellow_frame_origin_w = tool_tip_pos_w
                else:
                    yellow_frame_origin_w = tool_pos_w + _TOOL_TIP_OFFSET_Z_IN_TOOL * tool_z_w
                tool_tip_pos = tool_tip_pos_w[0].cpu()
                tool_tip_quat = tool_tip_quat_w[0].cpu()
                tool_tip_rpy_deg = _quat_wxyz_to_rpy_deg(tool_tip_quat)
                yellow_frame_origin = yellow_frame_origin_w[0].cpu()
                link6_to_ik_ee_local_w = quat_apply(
                    _quat_wxyz_conjugate(link6_quat_w),
                    yellow_frame_origin_w - link6_pos_w,
                )
                link6_to_ik_ee_local = link6_to_ik_ee_local_w[0].cpu()
                print_mobile_arm_pose_line(
                    base_pos=base_pos,
                    base_rpy_deg=base_rpy_deg,
                    tool_tip_pos=tool_tip_pos,
                    tool_tip_rpy_deg=tool_tip_rpy_deg,
                    tool_tip_quat=tool_tip_quat,
                    yellow_frame_origin=yellow_frame_origin,
                    link6_to_ik_ee_local=link6_to_ik_ee_local,
                    arm_joint_pos=arm_joint_pos,
                    quiet_runtime_debug=_quiet_runtime_debug,
                    print_link6_to_ik_ee_local=args_cli.task in _PRINT_LINK6_TO_IK_EE_LOCAL_TASKS,
                )

                # ── 同批次观测保存（RGBD + TF 矩阵，每 500 帧自动写盘）────────
                _pol = obs.get("policy", {})
                obs_recorder.maybe_record(
                    save_obs=args_cli.save_obs,
                    sensor_io_enabled=sensor_io_enabled,
                    policy_obs=_pol,
                    cam=cam,
                )

                # ── 相机红色线框（viewport）────────────────────────────────────
                if _step % 90 == 0:
                    print_mobile_arm_step_pose_line(
                        step=_step,
                        base_pos_w=base_pos_w,
                        tool_pos_w=tool_pos_w,
                        yellow_frame_origin_w=yellow_frame_origin_w,
                        tool_tip_pos_w=tool_tip_pos_w,
                        link6_to_ik_ee_local_w=link6_to_ik_ee_local_w,
                        print_link6_to_ik_ee_local=args_cli.task in _PRINT_LINK6_TO_IK_EE_LOCAL_TASKS,
                        show_camera_pose_debug=args_cli.show_camera_pose_debug,
                        sensor_io_enabled=sensor_io_enabled,
                        cam=cam,
                        cam_pos_w=cam_pos_w,
                    )
                if _has_debug_draw:
                    draw_viewport_sensor_debug(
                        _draw,
                        sensor_io_enabled=sensor_io_enabled,
                        lidar=lidar,
                        show_camera_pose_debug=args_cli.show_camera_pose_debug,
                        cam=cam,
                        cam_pos_w=cam_pos_w,
                        cam_quat_w=cam_quat_w,
                    )

                camera_display.update(
                    sensor_io_enabled=sensor_io_enabled,
                    cam=cam,
                    lidar=lidar,
                    obs=obs,
                    step=_step,
                    env_step_count=env_step_count,
                    camera_debug_interval=args_cli.camera_debug_interval,
                    robot=robot,
                    base_pos_w=base_pos_w,
                    tool_pos_w=tool_pos_w,
                    tool_tip_pos_w=tool_tip_pos_w,
                    base_idx=_base_idx,
                    tool_idx=_tool_idx,
                    tool_tip_idx=_tool_tip_idx,
                )

                # 图像读取完成后再同步到 post-step 最新 tool_link 位姿，供下一帧渲染使用。
                cam_pos_w, cam_quat_w = _sync_cam_tool_pose()

            _step += 1
            if _step % 90 == 0:
                print_mobile_arm_control_status(
                    step=_step,
                    is_chassis_mode=_mode == _MODE_CHASSIS,
                    wheel_vel=wheel_vel,
                    delta=delta,
                    arm_offsets=arm_offsets,
                    strict_mode_split=strict_mode_split,
                    use_prismatic_gripper_keyboard=use_prismatic_gripper_keyboard,
                    gripper_joint1_pos=gripper_joint1_pos,
                    teleop_device=args_cli.teleop_device,
                    teleop=teleop,
                )

            if extra.get("task_finished", False):
                print(f"\n任务完成: {extra.get('task_description', '')}")
                break

            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep()

    teleop.close()
    obs_recorder.flush(save_obs=args_cli.save_obs)
    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
