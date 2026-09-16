"""Small tensor/math helpers shared by the cuRobo pose-plan script."""

from __future__ import annotations

import torch


def quat_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q.reshape(4)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
        ]
    )


def pose_to_matrix(pos: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    transform = torch.eye(4, device=pos.device, dtype=pos.dtype)
    transform[:3, :3] = quat_wxyz_to_matrix(quat_wxyz.to(dtype=pos.dtype, device=pos.device))
    transform[:3, 3] = pos.reshape(3)
    return transform


def format_tensor(tensor: torch.Tensor) -> str:
    values = tensor.detach().flatten().cpu().tolist()
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def pose_error(
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    actual_pos: torch.Tensor,
    actual_quat: torch.Tensor,
):
    target_pos = target_pos.reshape(-1, 3)
    actual_pos = actual_pos.reshape(-1, 3)
    target_quat = target_quat.reshape(-1, 4)
    actual_quat = actual_quat.reshape(-1, 4)
    pos_error = torch.linalg.norm(actual_pos - target_pos, dim=-1)
    quat_dot = torch.sum(target_quat * actual_quat, dim=-1).abs().clamp(max=1.0)
    rot_error_rad = 2.0 * torch.acos(quat_dot)
    rot_error_deg = rot_error_rad * 180.0 / torch.pi
    return pos_error, rot_error_rad, rot_error_deg


def joint_error(target_q: torch.Tensor, actual_q: torch.Tensor):
    target_q = target_q.reshape(-1)
    actual_q = actual_q.reshape(-1)
    err = actual_q - target_q
    return torch.linalg.norm(err), torch.max(torch.abs(err)), err


def deg_to_rad_joint_values(values: list[float] | tuple[float, ...], *, device, dtype) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype) * (torch.pi / 180.0)


def validate_joint_value_count(values, *, name: str, joint_names: list[str]) -> None:
    if values is None:
        return
    if len(values) != len(joint_names):
        raise ValueError(f"{name} expects {len(joint_names)} values for joint_names={joint_names}, got {len(values)}.")
