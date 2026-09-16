"""Action tensor assembly helpers."""

import torch


def _assemble_mobile_arm_action(
    action_manager,
    wheel_vel: torch.Tensor,
    arm_cmd: torch.Tensor,
    device: str,
    gripper_cmd: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assemble action chunks by IsaacLab action-term name, not by a hard-coded order."""
    action_by_term = {
        "wheel_velocity": wheel_vel,
        "arm_joint_pos": arm_cmd,
    }
    if gripper_cmd is not None:
        action_by_term["gripper_joint_pos"] = gripper_cmd
    chunks = []
    for term_name, term_dim in zip(action_manager.active_terms, action_manager.action_term_dim):
        term_action = action_by_term.get(term_name)
        if term_action is None:
            term_action = torch.zeros((wheel_vel.shape[0], term_dim), dtype=wheel_vel.dtype, device=device)
        else:
            term_action = term_action.to(device=device)
            if term_action.shape[-1] != term_dim:
                raise RuntimeError(
                    f"Action term '{term_name}' expects dim={term_dim}, "
                    f"but received dim={term_action.shape[-1]}."
                )
        chunks.append(term_action)
    return torch.cat(chunks, dim=-1)

