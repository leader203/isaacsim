"""Pose and quaternion math helpers."""

import numpy as np
import torch


def _quat_wxyz_to_rpy_deg(quat_wxyz: torch.Tensor) -> tuple[float, float, float]:
    """将 wxyz 四元数转为欧拉角（度）。"""
    w, x, y, z = (float(quat_wxyz[i]) for i in range(4))

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    return float(roll), float(pitch), float(yaw)


def _quat_wxyz_conjugate(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Return inverse rotation for a unit wxyz quaternion."""
    quat_conj = quat_wxyz.clone()
    quat_conj[..., 1:] *= -1.0
    return quat_conj


def _get_scene_entity_pose(entity):
    """Read world pose from either a rigid object or a static prim view style asset."""
    if entity is None:
        return None, None
    if hasattr(entity, "data") and hasattr(entity.data, "root_pos_w"):
        pos = entity.data.root_pos_w[0].cpu()
        quat = entity.data.root_quat_w[0].cpu()
        return pos, quat
    if hasattr(entity, "get_world_poses"):
        pos_w, quat_w = entity.get_world_poses()
        return pos_w[0].cpu(), quat_w[0].cpu()
    return None, None
