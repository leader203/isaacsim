from __future__ import annotations

import torch


def arm_data_values(robot, arm_joint_ids_fn, attr_name: str) -> torch.Tensor | None:
    value = getattr(robot.data, attr_name, None)
    if value is None:
        return None
    try:
        return value[0, arm_joint_ids_fn(robot)].detach()
    except Exception:
        return None


def arm_private_values(robot, arm_joint_ids_fn, attr_name: str) -> torch.Tensor | None:
    value = getattr(robot, attr_name, None)
    if value is None:
        return None
    try:
        return value[0, arm_joint_ids_fn(robot)].detach()
    except Exception:
        return None


def format_optional_tensor(tensor: torch.Tensor | None, format_tensor_fn) -> str:
    if tensor is None:
        return "None"
    return format_tensor_fn(tensor)


def arm_control_debug_suffix(
    robot,
    *,
    target_q: torch.Tensor | None = None,
    action: torch.Tensor | None = None,
    arm_joint_ids_fn,
    joint_error_fn,
    format_tensor_fn,
) -> str:
    try:
        arm_ids = arm_joint_ids_fn(robot)
        arm_count = len(arm_ids)
        actual_q = robot.data.joint_pos[0, arm_ids].detach()
        actual_vel = robot.data.joint_vel[0, arm_ids].detach()
        data_pos_target = arm_data_values(robot, arm_joint_ids_fn, "joint_pos_target")
        data_vel_target = arm_data_values(robot, arm_joint_ids_fn, "joint_vel_target")
        data_effort_target = arm_data_values(robot, arm_joint_ids_fn, "joint_effort_target")
        sim_pos_target = arm_private_values(robot, arm_joint_ids_fn, "_joint_pos_target_sim")
        sim_vel_target = arm_private_values(robot, arm_joint_ids_fn, "_joint_vel_target_sim")
        sim_effort_target = arm_private_values(robot, arm_joint_ids_fn, "_joint_effort_target_sim")
        computed_torque = arm_data_values(robot, arm_joint_ids_fn, "computed_torque")
        applied_torque = arm_data_values(robot, arm_joint_ids_fn, "applied_torque")
        effort_limit = robot.data.joint_effort_limits[0, arm_ids].detach()
        velocity_limit = arm_data_values(robot, arm_joint_ids_fn, "joint_vel_limits")
        soft_velocity_limit = arm_data_values(robot, arm_joint_ids_fn, "soft_joint_vel_limits")

        def _fmt_optional(tensor: torch.Tensor | None) -> str:
            return format_optional_tensor(tensor, format_tensor_fn)

        parts = [
            f" arm_actual_q={format_tensor_fn(actual_q)}",
            f"arm_actual_vel={format_tensor_fn(actual_vel)}",
            f"arm_data_pos_target={_fmt_optional(data_pos_target)}",
            f"arm_data_vel_target={_fmt_optional(data_vel_target)}",
            f"arm_data_effort_target={_fmt_optional(data_effort_target)}",
            f"arm_sim_pos_target={_fmt_optional(sim_pos_target)}",
            f"arm_sim_vel_target={_fmt_optional(sim_vel_target)}",
            f"arm_sim_effort_target={_fmt_optional(sim_effort_target)}",
            f"arm_computed_torque={_fmt_optional(computed_torque)}",
            f"arm_applied_torque={_fmt_optional(applied_torque)}",
            f"arm_effort_limit={format_tensor_fn(effort_limit)}",
            f"arm_velocity_limit={_fmt_optional(velocity_limit)}",
            f"arm_soft_velocity_limit={_fmt_optional(soft_velocity_limit)}",
        ]
        if target_q is not None:
            target_q = target_q.detach().to(device=actual_q.device, dtype=actual_q.dtype).reshape(-1)
            target_err_norm, target_err_max, target_err = joint_error_fn(target_q, actual_q)
            parts.extend(
                [
                    f"arm_target_err_norm={target_err_norm.item():.6f}",
                    f"arm_target_err_max={target_err_max.item():.6f}",
                    f"arm_target_err={format_tensor_fn(target_err)}",
                ]
            )
            if data_pos_target is not None:
                _, data_target_err_max, data_target_err = joint_error_fn(target_q, data_pos_target)
                parts.extend(
                    [
                        f"arm_data_pos_target_err_max={data_target_err_max.item():.6f}",
                        f"arm_data_pos_target_err={format_tensor_fn(data_target_err)}",
                    ]
                )
            if sim_pos_target is not None:
                _, sim_target_err_max, sim_target_err = joint_error_fn(target_q, sim_pos_target)
                parts.extend(
                    [
                        f"arm_sim_pos_target_err_max={sim_target_err_max.item():.6f}",
                        f"arm_sim_pos_target_err={format_tensor_fn(sim_target_err)}",
                    ]
                )
        if computed_torque is not None and applied_torque is not None:
            torque_clip = computed_torque - applied_torque
            parts.append(f"arm_torque_clip_delta={format_tensor_fn(torque_clip)}")
            if effort_limit is not None:
                denom = torch.clamp(effort_limit.abs(), min=1e-6)
                computed_ratio = computed_torque.abs() / denom
                applied_ratio = applied_torque.abs() / denom
                parts.extend(
                    [
                        f"arm_computed_effort_ratio={format_tensor_fn(computed_ratio)}",
                        f"arm_applied_effort_ratio={format_tensor_fn(applied_ratio)}",
                        f"arm_computed_effort_ratio_max={torch.max(computed_ratio).item():.6f}",
                        f"arm_applied_effort_ratio_max={torch.max(applied_ratio).item():.6f}",
                    ]
                )
        if action is not None:
            action_arm = action.detach()[0, 4 : 4 + arm_count]
            default_arm_q = robot.data.default_joint_pos[0, arm_ids].detach()
            action_abs_target = default_arm_q + action_arm
            parts.extend(
                [
                    f"arm_action_delta={format_tensor_fn(action_arm)}",
                    f"arm_action_abs_target={format_tensor_fn(action_abs_target)}",
                ]
            )
        return " " + " ".join(parts)
    except Exception as exc:
        return f" arm_control_debug_error={exc!r}"
