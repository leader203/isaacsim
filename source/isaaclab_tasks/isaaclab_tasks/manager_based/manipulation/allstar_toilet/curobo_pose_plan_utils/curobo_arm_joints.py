"""Arm joint lookup and low-level target helpers."""

from __future__ import annotations

import torch


def arm_joint_ids(robot, joint_names: list[str]) -> list[int]:
    available_joint_names = list(robot.data.joint_names)
    missing = [joint_name for joint_name in joint_names if joint_name not in available_joint_names]
    if missing:
        raise RuntimeError(f"Robot is missing configured arm joints: {missing}")
    return [available_joint_names.index(joint_name) for joint_name in joint_names]


def arm_joint_names(robot, joint_names: list[str]) -> list[str]:
    available_joint_names = list(robot.data.joint_names)
    return [available_joint_names[int(joint_id)] for joint_id in arm_joint_ids(robot, joint_names)]


def write_arm_target(robot, q_arm: torch.Tensor, joint_names: list[str]) -> None:
    arm_ids = arm_joint_ids(robot, joint_names)
    if q_arm.dim() == 1:
        q_arm = q_arm.unsqueeze(0)
    robot.set_joint_position_target(
        q_arm.to(device=robot.device, dtype=robot.data.joint_pos.dtype),
        joint_ids=arm_ids,
    )


def find_optional_joints(robot, name_key: str) -> tuple[list[int], list[str]]:
    try:
        return robot.find_joints(name_key, preserve_order=True)
    except ValueError as exc:
        if "Not all regular expressions are matched" in str(exc):
            return [], []
        raise


def build_action(
    robot,
    arm_target_q: torch.Tensor,
    joint_names: list[str],
    *,
    default_open_gripper_joint1_pos: float,
    gripper_joint1_pos: float | None = None,
) -> torch.Tensor:
    arm_ids = arm_joint_ids(robot, joint_names)
    action_dim = 4 + len(arm_ids)
    gripper1_ids, _ = find_optional_joints(robot, "gripper_joint1")
    gripper2_ids, _ = find_optional_joints(robot, "gripper_joint2")
    gripper_ids = list(gripper1_ids) + list(gripper2_ids)
    has_gripper_action = bool(gripper_ids)
    if has_gripper_action:
        action_dim += len(gripper_ids)
    action = torch.zeros((1, action_dim), device=robot.device, dtype=robot.data.joint_pos.dtype)
    default_arm_q = robot.data.default_joint_pos[0, arm_ids]
    action[0, 4 : 4 + len(arm_ids)] = arm_target_q - default_arm_q
    if has_gripper_action:
        gripper_target = default_open_gripper_joint1_pos if gripper_joint1_pos is None else gripper_joint1_pos
        gripper_joint_names = list(robot.data.joint_names)
        for offset, gripper_id in enumerate(gripper_ids):
            joint_name = gripper_joint_names[int(gripper_id)]
            target = -float(gripper_target) if joint_name == "gripper_joint2" else float(gripper_target)
            default_gripper_q = robot.data.default_joint_pos[0, int(gripper_id)]
            action[0, 4 + len(arm_ids) + offset] = (
                torch.tensor(target, device=robot.device, dtype=robot.data.joint_pos.dtype) - default_gripper_q
            )
    return action


def zero_arm_velocity_target(robot, joint_names: list[str]) -> None:
    arm_ids = arm_joint_ids(robot, joint_names)
    zero = torch.zeros((1, len(arm_ids)), device=robot.device, dtype=robot.data.joint_pos.dtype)
    robot.set_joint_velocity_target(zero, joint_ids=arm_ids)


def set_arm_effort_target(robot, joint_names: list[str], effort: torch.Tensor | None = None) -> None:
    arm_ids = arm_joint_ids(robot, joint_names)
    if effort is None:
        effort = torch.zeros((1, len(arm_ids)), device=robot.device, dtype=robot.data.joint_pos.dtype)
    else:
        effort = effort.detach().to(device=robot.device, dtype=robot.data.joint_pos.dtype).reshape(1, -1)
    robot.set_joint_effort_target(effort, joint_ids=arm_ids)


def set_arm_velocity_target(robot, joint_names: list[str], arm_target_vel: torch.Tensor | None) -> None:
    if arm_target_vel is None:
        zero_arm_velocity_target(robot, joint_names)
        return
    arm_ids = arm_joint_ids(robot, joint_names)
    target_vel = arm_target_vel.detach().to(device=robot.device, dtype=robot.data.joint_pos.dtype).reshape(1, -1)
    robot.set_joint_velocity_target(target_vel, joint_ids=arm_ids)


def clamp_arm_joint_target_to_soft_limits(
    robot,
    target_q: torch.Tensor,
    joint_names: list[str],
    *,
    label: str,
    format_tensor,
) -> torch.Tensor:
    arm_ids = arm_joint_ids(robot, joint_names)
    clamped = target_q.detach().to(device=robot.device, dtype=robot.data.joint_pos.dtype).reshape(-1).clone()
    limits = getattr(robot.data, "soft_joint_pos_limits", None)
    if limits is None:
        return clamped
    arm_limits = limits[0, arm_ids].detach().to(device=clamped.device, dtype=clamped.dtype)
    if arm_limits.shape[0] != clamped.shape[0] or arm_limits.shape[-1] != 2:
        return clamped
    requested = clamped.clone()
    lower = arm_limits[:, 0]
    upper = arm_limits[:, 1]
    clamped = torch.where(torch.isfinite(lower), torch.maximum(clamped, lower), clamped)
    clamped = torch.where(torch.isfinite(upper), torch.minimum(clamped, upper), clamped)
    clamp_delta = clamped - requested
    if torch.max(torch.abs(clamp_delta)).item() > 1.0e-6:
        print(
            f"[JointLimitClamp] label={label} requested_q={format_tensor(requested)} "
            f"clamped_q={format_tensor(clamped)} soft_joint_pos_limits={format_tensor(arm_limits)} "
            f"clamp_delta={format_tensor(clamp_delta)}",
            flush=True,
        )
    return clamped
