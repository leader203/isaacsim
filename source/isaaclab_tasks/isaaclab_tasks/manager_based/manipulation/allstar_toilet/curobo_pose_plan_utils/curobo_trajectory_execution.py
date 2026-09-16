from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrajectoryExecutionConfig:
    joint_target_tolerance: float
    steps_per_waypoint: int
    max_steps_per_waypoint: int
    trajectory_final_joint_target_tolerance: float | None
    waypoint_log_stride: int
    post_trajectory_settle_steps: int
    post_trajectory_settle_log_stride: int
    trajectory_velocity_feedforward: bool

    @property
    def final_joint_target_tolerance(self) -> float:
        if self.trajectory_final_joint_target_tolerance is not None:
            return float(self.trajectory_final_joint_target_tolerance)
        return float(self.joint_target_tolerance)


def trajectory_execution_config_from_args(args_cli) -> TrajectoryExecutionConfig:
    return TrajectoryExecutionConfig(
        joint_target_tolerance=float(args_cli.joint_target_tolerance),
        steps_per_waypoint=int(args_cli.steps_per_waypoint),
        max_steps_per_waypoint=int(args_cli.max_steps_per_waypoint),
        trajectory_final_joint_target_tolerance=(
            float(args_cli.trajectory_final_joint_target_tolerance)
            if args_cli.trajectory_final_joint_target_tolerance is not None
            else None
        ),
        waypoint_log_stride=int(args_cli.waypoint_log_stride),
        post_trajectory_settle_steps=int(args_cli.post_trajectory_settle_steps),
        post_trajectory_settle_log_stride=int(args_cli.post_trajectory_settle_log_stride),
        trajectory_velocity_feedforward=bool(args_cli.trajectory_velocity_feedforward),
    )


def execute_joint_target(
    env,
    robot,
    arm_target_q: torch.Tensor,
    steps: int,
    *,
    gripper_joint1_pos: float | None,
    build_action_fn,
    sync_link5_camera_fn,
    zero_arm_velocity_target_fn,
    set_arm_effort_target_fn,
    show_camera_windows_fn,
) -> None:
    action = build_action_fn(robot, arm_target_q, gripper_joint1_pos=gripper_joint1_pos)
    for _ in range(max(1, steps)):
        sync_link5_camera_fn(env, robot)
        zero_arm_velocity_target_fn(robot)
        set_arm_effort_target_fn(robot)
        env.step(action)
        show_camera_windows_fn(env)


def execute_joint_target_until_converged(
    env,
    robot,
    arm_target_q: torch.Tensor,
    *,
    max_steps: int,
    min_steps: int,
    label: str,
    gripper_joint1_pos: float | None,
    cfg: TrajectoryExecutionConfig,
    build_action_fn,
    sync_link5_camera_fn,
    zero_arm_velocity_target_fn,
    set_arm_effort_target_fn,
    show_camera_windows_fn,
    arm_joint_ids_fn,
    joint_error_fn,
    format_tensor_fn,
    arm_control_debug_suffix_fn,
) -> tuple[int, float]:
    action = build_action_fn(robot, arm_target_q, gripper_joint1_pos=gripper_joint1_pos)
    actual_q = robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach()
    joint_err_max = torch.tensor(float("inf"), device=robot.device, dtype=robot.data.joint_pos.dtype)
    steps_used = 0
    for step_idx in range(max(1, max_steps)):
        sync_link5_camera_fn(env, robot)
        zero_arm_velocity_target_fn(robot)
        set_arm_effort_target_fn(robot)
        env.step(action)
        show_camera_windows_fn(env)
        steps_used = step_idx + 1
        actual_q = robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach()
        _, joint_err_max, _ = joint_error_fn(arm_target_q, actual_q)
        if steps_used >= max(1, min_steps) and joint_err_max.item() <= cfg.joint_target_tolerance:
            break
    print(
        f"[JointTarget] label={label} target_q={format_tensor_fn(arm_target_q)} "
        f"actual_q={format_tensor_fn(actual_q)} steps_used={steps_used}/{max(1, max_steps)} "
        f"joint_err_max={joint_err_max.item():.6f}"
        f"{arm_control_debug_suffix_fn(robot, target_q=arm_target_q, action=action)}",
        flush=True,
    )
    return steps_used, float(joint_err_max.item())


def hold_joint_target_if_close(
    env,
    robot,
    target_q: torch.Tensor,
    steps: int,
    *,
    label: str,
    gripper_joint1_pos: float | None,
    cfg: TrajectoryExecutionConfig,
    arm_joint_ids_fn,
    joint_error_fn,
    format_tensor_fn,
    arm_control_debug_suffix_fn,
    execute_joint_target_fn,
) -> int:
    """Hold an already reached arm target; do not use this to create a new arm motion."""
    if steps <= 0:
        return 0
    actual_q = robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach()
    joint_err_norm, joint_err_max, joint_err = joint_error_fn(target_q, actual_q)
    if joint_err_max.item() > cfg.joint_target_tolerance:
        print(
            f"[HoldSkipped] label={label} reason=target_not_reached_by_curobo "
            f"target_q={format_tensor_fn(target_q)} actual_q={format_tensor_fn(actual_q)} "
            f"joint_err_norm={joint_err_norm.item():.6f} joint_err_max={joint_err_max.item():.6f} "
            f"joint_err={format_tensor_fn(joint_err)}"
            f"{arm_control_debug_suffix_fn(robot, target_q=target_q)}",
            flush=True,
        )
        if gripper_joint1_pos is not None:
            execute_joint_target_fn(
                env,
                robot,
                actual_q.detach().clone(),
                steps,
                gripper_joint1_pos=gripper_joint1_pos,
            )
            return max(1, steps)
        return 0
    execute_joint_target_fn(env, robot, target_q, steps, gripper_joint1_pos=gripper_joint1_pos)
    return max(1, steps)


def execute_trajectory(
    env,
    robot,
    traj: torch.Tensor,
    *,
    gripper_joint1_pos: float | None,
    label: str,
    cfg: TrajectoryExecutionConfig,
    build_action_fn,
    sync_link5_camera_fn,
    set_arm_velocity_target_fn,
    set_arm_effort_target_fn,
    show_camera_windows_fn,
    arm_joint_ids_fn,
    joint_error_fn,
    trajectory_waypoint_velocity_fn,
    format_tensor_fn,
    format_optional_tensor_fn,
    arm_control_debug_suffix_fn,
) -> torch.Tensor:
    last_actual_q = None
    final_step_budget = max(1, int(cfg.max_steps_per_waypoint))
    final_tolerance = cfg.final_joint_target_tolerance
    for waypoint_idx, waypoint in enumerate(traj):
        action = build_action_fn(robot, waypoint, gripper_joint1_pos=gripper_joint1_pos)
        arm_vel_target = None
        joint_err_norm = None
        joint_err_max = None
        joint_err = None
        steps_used = 0
        step_budget = max(1, cfg.steps_per_waypoint)
        if waypoint_idx < len(traj) - 1:
            arm_vel_target = trajectory_waypoint_velocity_fn(env, robot, traj, waypoint_idx, step_budget)
            for step_idx in range(step_budget):
                sync_link5_camera_fn(env, robot)
                set_arm_velocity_target_fn(robot, arm_vel_target)
                set_arm_effort_target_fn(robot)
                env.step(action)
                show_camera_windows_fn(env)
                steps_used = step_idx + 1
            actual_q = robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach()
            if joint_err_norm is None or joint_err_max is None or joint_err is None:
                joint_err_norm, joint_err_max, joint_err = joint_error_fn(waypoint, actual_q)
        else:
            step_budget = final_step_budget
            arm_vel_target = torch.zeros_like(waypoint)
            min_steps = min(max(1, cfg.steps_per_waypoint), step_budget)
            for step_idx in range(step_budget):
                sync_link5_camera_fn(env, robot)
                set_arm_velocity_target_fn(robot, arm_vel_target)
                set_arm_effort_target_fn(robot)
                env.step(action)
                show_camera_windows_fn(env)
                steps_used = step_idx + 1
                actual_q = robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach()
                joint_err_norm, joint_err_max, joint_err = joint_error_fn(waypoint, actual_q)
                if steps_used >= min_steps and joint_err_max.item() <= final_tolerance:
                    break
        actual_q = robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach()
        last_actual_q = actual_q
        if joint_err_norm is None or joint_err_max is None or joint_err is None:
            joint_err_norm, joint_err_max, joint_err = joint_error_fn(waypoint, actual_q)
        should_log = (
            waypoint_idx == 0
            or waypoint_idx == len(traj) - 1
            or (cfg.waypoint_log_stride > 0 and (waypoint_idx + 1) % cfg.waypoint_log_stride == 0)
        )
        if should_log:
            print(
                f"[Execute] waypoint={waypoint_idx + 1}/{len(traj)}",
                flush=True,
            )
    post_settle_steps = max(0, int(cfg.post_trajectory_settle_steps))
    if post_settle_steps > 0 and len(traj) > 0:
        final_waypoint = traj[-1]
        action = build_action_fn(robot, final_waypoint, gripper_joint1_pos=gripper_joint1_pos)
        log_stride = max(0, int(cfg.post_trajectory_settle_log_stride))
        for settle_idx in range(post_settle_steps):
            sync_link5_camera_fn(env, robot)
            set_arm_velocity_target_fn(robot, torch.zeros_like(final_waypoint))
            set_arm_effort_target_fn(robot)
            env.step(action)
            show_camera_windows_fn(env)
            should_log = (
                settle_idx == 0
                or settle_idx == post_settle_steps - 1
                or (log_stride > 0 and (settle_idx + 1) % log_stride == 0)
            )
            if should_log:
                actual_q = robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach()
                joint_err_norm, joint_err_max, joint_err = joint_error_fn(final_waypoint, actual_q)
                print(
                    f"[PostTrajectorySettle] label={label} step={settle_idx + 1}/{post_settle_steps} "
                    f"target_q={format_tensor_fn(final_waypoint)} actual_q={format_tensor_fn(actual_q)} "
                    f"joint_err_norm={joint_err_norm.item():.6f} joint_err_max={joint_err_max.item():.6f} "
                    f"joint_err={format_tensor_fn(joint_err)} "
                    f"arm_vel_ff_enabled=False arm_vel_ff_target={format_tensor_fn(torch.zeros_like(final_waypoint))}"
                    f"{arm_control_debug_suffix_fn(robot, target_q=final_waypoint, action=action)}",
                    flush=True,
                )
        last_actual_q = robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach()
    return last_actual_q if last_actual_q is not None else robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach()


def execute_trajectory_waypoints_fixed_steps(
    env,
    robot,
    traj: torch.Tensor,
    *,
    label: str,
    cfg: TrajectoryExecutionConfig,
    build_action_fn,
    sync_link5_camera_fn,
    set_arm_velocity_target_fn,
    set_arm_effort_target_fn,
    show_camera_windows_fn,
    arm_joint_ids_fn,
    joint_error_fn,
    trajectory_waypoint_velocity_fn,
    format_tensor_fn,
    format_optional_tensor_fn,
    arm_control_debug_suffix_fn,
) -> int:
    if traj.dim() > 2:
        traj = traj.reshape(-1, traj.shape[-1])
    step_budget = max(1, cfg.steps_per_waypoint)
    step_count = 0
    for waypoint_idx, waypoint in enumerate(traj):
        action = build_action_fn(robot, waypoint)
        arm_vel_target = trajectory_waypoint_velocity_fn(env, robot, traj, waypoint_idx, step_budget)
        for _ in range(step_budget):
            sync_link5_camera_fn(env, robot)
            set_arm_velocity_target_fn(robot, arm_vel_target)
            set_arm_effort_target_fn(robot)
            env.step(action)
            show_camera_windows_fn(env)
            step_count += 1
        if (
            waypoint_idx == 0
            or waypoint_idx == len(traj) - 1
            or (cfg.waypoint_log_stride > 0 and (waypoint_idx + 1) % cfg.waypoint_log_stride == 0)
        ):
            print(
                f"[ExecutePrefix] waypoint={waypoint_idx + 1}/{len(traj)}",
                flush=True,
            )
    return step_count
