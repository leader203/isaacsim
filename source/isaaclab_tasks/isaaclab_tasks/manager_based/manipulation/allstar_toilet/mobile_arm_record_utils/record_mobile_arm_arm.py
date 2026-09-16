"""Arm action and joint-command helpers."""

import contextlib

import torch


_ARM_KEYBOARD_INCREMENTS = {
    "W": (0, 1.0),
    "S": (0, -1.0),
    "A": (1, 1.0),
    "D": (1, -1.0),
    "Q": (2, 1.0),
    "E": (2, -1.0),
    "Z": (3, 1.0),
    "X": (3, -1.0),
    "T": (4, 1.0),
    "G": (4, -1.0),
    "C": (5, 1.0),
    "V": (5, -1.0),
}


def _get_arm_joint_action_term(base_env):
    """Return the configured arm joint-position action term if present."""
    with contextlib.suppress(Exception):
        return base_env.action_manager.get_term("arm_joint_pos")
    return None


def _resync_joint_position_action_offset(robot, action_term) -> None:
    """Keep JointPositionAction cached offset aligned with robot default joint positions.

    This is needed when startup/reset events modify `robot.data.default_joint_pos`
    after the action term was constructed. Without a resync, sending an absolute
    hold command can turn into a biased target and cause continuous arm drift.
    """
    if action_term is None or not hasattr(action_term, "_joint_ids") or not hasattr(action_term, "_offset"):
        return

    desired_offset = robot.data.default_joint_pos[:, action_term._joint_ids].clone()
    if isinstance(action_term._offset, torch.Tensor):
        desired_offset = desired_offset.to(device=action_term._offset.device, dtype=action_term._offset.dtype)
        if action_term._offset.shape != desired_offset.shape:
            action_term._offset = desired_offset
        else:
            action_term._offset[:] = desired_offset
    else:
        action_term._offset = desired_offset


def _absolute_arm_joint_positions_to_action(
    robot,
    joint_ids,
    absolute_joint_positions: torch.Tensor,
    action_term=None,
) -> torch.Tensor:
    """Convert absolute arm joint targets to raw action values for JointPositionAction."""
    if absolute_joint_positions.dim() == 1:
        absolute_joint_positions = absolute_joint_positions.unsqueeze(0)

    if action_term is not None and hasattr(action_term, "_offset") and hasattr(action_term, "_scale"):
        target = absolute_joint_positions.to(device=robot.device, dtype=robot.data.joint_pos.dtype)
        if isinstance(action_term._offset, torch.Tensor):
            offset = action_term._offset.to(device=target.device, dtype=target.dtype)
        else:
            offset = torch.full_like(target, float(action_term._offset))
        if isinstance(action_term._scale, torch.Tensor):
            scale = action_term._scale.to(device=target.device, dtype=target.dtype)
        else:
            scale = torch.full_like(target, float(action_term._scale))
        return (target - offset) / scale

    default_joint_pos = robot.data.default_joint_pos[:, joint_ids]
    return absolute_joint_positions.to(device=robot.device, dtype=default_joint_pos.dtype) - default_joint_pos

