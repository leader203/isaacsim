from __future__ import annotations

import torch


def gripper_joint_id(robot, joint_name: str) -> int | None:
    try:
        joint_ids, _ = robot.find_joints(joint_name, preserve_order=True)
    except ValueError as exc:
        if "Not all regular expressions are matched" in str(exc):
            return None
        raise
    if not joint_ids:
        return None
    return int(list(joint_ids)[0])


def gripper_joint_state(robot, joint_name: str) -> tuple[float, float] | None:
    joint_id = gripper_joint_id(robot, joint_name)
    if joint_id is None:
        return None
    pos = float(robot.data.joint_pos[0, joint_id].detach().cpu().item())
    vel = float(robot.data.joint_vel[0, joint_id].detach().cpu().item())
    return pos, vel


def execute_gripper_feedback_grasp(
    env,
    robot,
    arm_target_q: torch.Tensor,
    *,
    label: str,
    target_q: float,
    open_target_q: float,
    pos_eps: float,
    vel_eps: float,
    max_steps: int,
    build_action_fn,
    sync_link5_camera_fn,
    zero_arm_velocity_target_fn,
    show_camera_windows_fn,
    arm_control_debug_suffix_fn,
) -> tuple[float, bool]:
    state = gripper_joint_state(robot, "gripper_joint1")
    if state is None:
        print(f"[GripperGrasp] label={label} skipped: gripper_joint1 not found", flush=True)
        return open_target_q, False

    actual_q, actual_vel = state
    steps_used = 0
    stable_steps = 0
    required_stable_steps = 12
    no_progress_window_steps = 50
    no_progress_min_delta = max(pos_eps * 0.3, 0.0003)
    contact_progress_ratio = 0.5
    q_history = [actual_q]
    early_exit_reason = None
    early_exit_grasped = False
    action = build_action_fn(robot, arm_target_q, gripper_joint1_pos=target_q)
    print(
        f"[GripperGrasp] label={label} start actual_q={actual_q:.6f} actual_vel={actual_vel:.6f} "
        f"target_q={target_q:.6f} target_joint2={-target_q:.6f}",
        flush=True,
    )

    for step_idx in range(max_steps):
        sync_link5_camera_fn(env, robot)
        zero_arm_velocity_target_fn(robot)
        env.step(action)
        show_camera_windows_fn(env)
        steps_used = step_idx + 1

        state = gripper_joint_state(robot, "gripper_joint1")
        if state is None:
            break
        actual_q, actual_vel = state
        pos_err = abs(actual_q - target_q)
        if pos_err <= pos_eps and abs(actual_vel) <= vel_eps:
            stable_steps += 1
        else:
            stable_steps = 0
        if stable_steps >= required_stable_steps:
            break
        q_history.append(actual_q)
        if len(q_history) > no_progress_window_steps:
            window_delta = abs(q_history[-1] - q_history[-no_progress_window_steps])
            command_travel = abs(target_q - open_target_q)
            actual_travel = abs(actual_q - open_target_q)
            progress_ratio = actual_travel / max(command_travel, 1.0e-6)
            if pos_err > pos_eps and window_delta <= no_progress_min_delta:
                if progress_ratio >= contact_progress_ratio:
                    early_exit_reason = "contact_or_limit_no_progress"
                    early_exit_grasped = True
                else:
                    early_exit_reason = "no_progress_before_contact"
                    early_exit_grasped = False
                break
    grasped = stable_steps >= required_stable_steps or early_exit_grasped
    if not grasped:
        print(
            f"[GripperGrasp] label={label} warning=target_not_stable "
            f"actual_q={actual_q:.6f} target_q={target_q:.6f} actual_vel={actual_vel:.6f} "
            f"stable_steps={stable_steps}/{required_stable_steps} steps_used={steps_used}/{max_steps} "
            f"early_exit_reason={early_exit_reason}",
            flush=True,
        )
    print(
        f"[GripperGrasp] label={label} grasped={grasped} "
        f"actual_q={actual_q:.6f} actual_vel={actual_vel:.6f} hold_target_q={target_q:.6f} "
        f"steps_used={steps_used}/{max_steps} early_exit_reason={early_exit_reason}"
        f"{arm_control_debug_suffix_fn(robot, target_q=arm_target_q, action=action)}",
        flush=True,
    )
    return target_q, grasped
