from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import reset_joints_by_offset, reset_root_state_uniform  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera import Camera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pos: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    default_pos = torch.as_tensor(default_pos, device=env.device, dtype=asset.data.default_joint_pos.dtype).view(1, -1)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    if asset_cfg.joint_ids == slice(None):
        if default_pos.shape[1] != asset.data.default_joint_pos.shape[1]:
            raise ValueError(
                f"default_pos has {default_pos.shape[1]} entries, but asset '{asset_cfg.name}' has "
                f"{asset.data.default_joint_pos.shape[1]} joints."
            )
        asset.data.default_joint_pos[env_ids] = default_pos.repeat(len(env_ids), 1)
    else:
        if default_pos.shape[1] != len(asset_cfg.joint_ids):
            raise ValueError(
                f"default_pos has {default_pos.shape[1]} entries, but selected joints are "
                f"{len(asset_cfg.joint_ids)} for asset '{asset_cfg.name}'."
            )
        asset.data.default_joint_pos[env_ids[:, None], asset_cfg.joint_ids] = default_pos.repeat(len(env_ids), 1)


def set_default_root_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    default_vel: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    default_root_state = torch.concat(
        [
            torch.as_tensor(default_pose, device=env.device, dtype=asset.data.default_root_state.dtype),
            torch.as_tensor(default_vel, device=env.device, dtype=asset.data.default_root_state.dtype),
        ],
        dim=-1,
    ).view(1, -1)
    if default_root_state.shape[1] != asset.data.default_root_state.shape[1]:
        raise ValueError(
            f"default_root_state has {default_root_state.shape[1]} entries, but asset '{asset_cfg.name}' expects "
            f"{asset.data.default_root_state.shape[1]}."
        )
    asset.data.default_root_state[env_ids] = default_root_state.repeat(len(env_ids), 1)


def reset_camera_pose_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
):
    asset: Camera = env.scene[asset_cfg.name]
    pos = torch.tensor(asset.cfg.offset.pos, dtype=torch.float32, device=asset.device).unsqueeze(0)
    rot = torch.tensor(asset.cfg.offset.rot, dtype=torch.float32, device=asset.device).unsqueeze(0)
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    positions = pos + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(rot, orientations_delta)
    orientations = math_utils.convert_camera_frame_orientation_convention(
        orientations, origin=asset.cfg.offset.convention, target="opengl"
    )
    asset._view.set_local_poses(positions, orientations, env_ids)
