from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.envs.mdp import image, joint_pos_rel, last_action  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ── camera-to-link TF constants (must match run_record_mobile_arm.py) ────────
# Position offset: +8 cm along tool_link X, +1 cm along tool_link Z
_CAM_T_IN_TOOL = (0.08, 0.0, 0.01)
# Rotation offset (wxyz): Ry+45° ⊗ Rz-90°
_CAM_Q_IN_TOOL = (0.6533, -0.2706, 0.2706, -0.6533)


def _cam_t_in_tool(env: ManagerBasedEnv) -> tuple[float, float, float]:
    return tuple(getattr(env.cfg, "cam_t_in_tool", _CAM_T_IN_TOOL))


def _body_index(body_names: list[str], preferred: str, fallback_suffix: str | None = None) -> int:
    """Resolve a body index robustly before SceneEntityCfg body_ids are populated."""
    if preferred in body_names:
        return body_names.index(preferred)
    if fallback_suffix is not None:
        for idx, name in enumerate(body_names):
            if name == fallback_suffix or name.endswith(fallback_suffix):
                return idx
    raise ValueError(f"Body '{preferred}' not found in articulation bodies: {body_names}")


def _get_or_init_cam_link5_mount_offset(
    env: ManagerBasedEnv,
    link5_pos_w: torch.Tensor,
    link5_quat_w: torch.Tensor,
    tool_pos_w: torch.Tensor,
    tool_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize the fixed link5->camera offset from the current tool camera pose."""
    offset = getattr(env.cfg, "_cam_link5_mount_offset", None)
    if offset is not None:
        return offset

    q_tool_cam = torch.tensor([_CAM_Q_IN_TOOL], device=env.device).expand(env.num_envs, -1)
    t_tool_cam = torch.tensor(_cam_t_in_tool(env), device=env.device).expand(env.num_envs, -1)
    init_cam_pos_w, init_cam_quat_w = math_utils.combine_frame_transforms(
        tool_pos_w, tool_quat_w, t_tool_cam, q_tool_cam
    )
    t_link5_cam, q_link5_cam = math_utils.subtract_frame_transforms(
        link5_pos_w, link5_quat_w, init_cam_pos_w, init_cam_quat_w
    )
    env.cfg._cam_link5_mount_offset = (t_link5_cam.detach().clone(), q_link5_cam.detach().clone())
    return t_link5_cam, q_link5_cam


def _q_to_R(q: torch.Tensor) -> torch.Tensor:
    """Batch wxyz quaternion [N,4] → rotation matrix [N,3,3] (v_world = R @ v_local)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)


if str(isaaclab.__version__) <= "0.36.21":
    def euler_xyz_from_quat(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return math_utils.euler_xyz_from_quat(quat)
else:
    def euler_xyz_from_quat(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return math_utils.euler_xyz_from_quat(quat, wrap_to_2pi=True)


if str(isaaclab.__version__) <= "0.36.21":
    def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return math_utils.quat_rotate_inverse(q, v)
else:
    def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return math_utils.quat_apply_inverse(q, v)


def joint_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def body_rotpose_local(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="hand"),
    offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    ee_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w[:, 0, :], ee_quat_w[:, 0, :]
    )
    ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
        ee_pose_b,
        ee_quat_b,
        torch.tensor(offset_pos, device=env.device).repeat(env.num_envs, 1),
        torch.tensor(offset_rot, device=env.device).repeat(env.num_envs, 1),
    )
    ee_r_b, ee_p_b, ee_y_b = euler_xyz_from_quat(ee_quat_b)
    return torch.cat([ee_pose_b, ee_r_b.unsqueeze(-1), ee_p_b.unsqueeze(-1), ee_y_b.unsqueeze(-1)], dim=-1)


def body_pose_world(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="hand"),
    offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    is_relative: bool = True,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    ee_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    ee_pos_w, ee_quat_w = math_utils.combine_frame_transforms(
        ee_pos_w[:, 0, :],
        ee_quat_w[:, 0, :],
        torch.tensor(offset_pos, device=env.device).repeat(env.num_envs, 1),
        torch.tensor(offset_rot, device=env.device).repeat(env.num_envs, 1),
    )
    if is_relative:
        ee_pos_w = ee_pos_w - env.scene.env_origins[:, 0:3]
    return torch.cat([ee_pos_w, ee_quat_w], dim=-1)


def contact_force_local(
    env: ManagerBasedEnv,
    sensor_name: str,
    idx: int | slice = slice(None),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="hand"),
    offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ee_pos_w_no_offset = asset.data.body_pos_w[:, asset_cfg.body_ids]
    ee_quat_w_no_offset = asset.data.body_quat_w[:, asset_cfg.body_ids]
    _, ee_quat_w = math_utils.combine_frame_transforms(
        ee_pos_w_no_offset[:, 0, :],
        ee_quat_w_no_offset[:, 0, :],
        torch.tensor(offset_pos, device=env.device).repeat(env.num_envs, 1),
        torch.tensor(offset_rot, device=env.device).repeat(env.num_envs, 1),
    )
    if type(idx) is slice:
        ee_quat_w = ee_quat_w.unsqueeze(1)
    force_sensor = env.scene[sensor_name]
    assert isinstance(force_sensor, ContactSensor)
    contact_force_world = force_sensor.data.net_forces_w
    assert type(contact_force_world) is torch.Tensor
    contact_force_local_val = quat_apply_inverse(ee_quat_w, contact_force_world[:, idx, :])
    return torch.cat([contact_force_local_val, torch.zeros_like(contact_force_local_val)], dim=-1)


def force_control(env: ManagerBasedEnv) -> torch.Tensor:
    assert hasattr(env.cfg, "last_force_control") and isinstance(env.cfg.last_force_control, torch.Tensor)
    return env.cfg.last_force_control  # type: ignore


def frame_transformer(
    env: ManagerBasedEnv,
    sensor_name: str,
    idx: int | slice = slice(None),
) -> torch.Tensor:
    tf_sensor = env.scene[sensor_name]
    assert isinstance(tf_sensor, FrameTransformer)
    return torch.cat(
        [tf_sensor.data.target_pos_source[:, idx, :], tf_sensor.data.target_quat_source[:, idx, :]], dim=-1
    )


def cam_in_tool_tf(env: ManagerBasedEnv) -> torch.Tensor:
    """固定的相机在 tool_link 系下的位姿矩阵，shape [N, 16]（行主序展开）。

    矩阵语义：T_cam_in_tool
      - T[:3, :3]  = R_cam_in_tool  → 将相机系的列向量旋转到 tool_link 系
      - T[:3,  3]  = t_cam_in_tool  → 相机原点在 tool_link 系下的坐标
      - 用法：p_tool = T @ p_cam_homogeneous

    偏置来源（与 run_record_mobile_arm.py 一致）：
      - 位置：tool_link 原点沿自身 X 轴 +8 cm、Z 轴 +1 cm
      - 旋转：_CAM_Q_IN_TOOL（wxyz），与脚本中 _cam_offset 相同
    """
    q = torch.tensor([_CAM_Q_IN_TOOL], device=env.device)
    t = torch.tensor(_cam_t_in_tool(env), device=env.device)
    T = torch.eye(4, device=env.device)
    T[:3, :3] = _q_to_R(q)[0]
    T[:3, 3]  = t
    return T.flatten().unsqueeze(0).repeat(env.num_envs, 1)


def cam_in_base_tf(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """每步更新的相机在 base_link 系下的位姿矩阵，shape [N, 16]（行主序展开）。

    矩阵语义：T_cam_in_base
      - T[:3, :3]  = R_cam_in_base  → 将相机系的列向量旋转到 base_link 系
      - T[:3,  3]  = t_cam_in_base  → 相机原点在 base_link 系下的坐标
      - 用法：p_base = T @ p_cam_homogeneous

    计算步骤：
      1. 从 tool_link 世界位姿 + 固定偏置推导相机世界位姿（与脚本一致）
      2. 使用真实的底盘 ``base_link`` 刚体世界位姿，而不是 articulation root
      3. T_cam_in_base = inv(T_base_in_world) · T_cam_in_world
    """
    asset: Articulation = env.scene[asset_cfg.name]
    N = env.num_envs

    # Camera mount body. Tool USDs usually expose tool_link; bare/default USDs stop at link6.
    tool_id = _body_index(asset.data.body_names, "tool_link", "link6")
    tool_pos_w  = asset.data.body_pos_w[:, tool_id, :]   # [N, 3]
    tool_quat_w = asset.data.body_quat_w[:, tool_id, :]  # [N, 4] wxyz

    # 相机世界位姿（与 run_record_mobile_arm.py 保持一致）
    R_tool    = _q_to_R(tool_quat_w)                                    # [N, 3, 3]
    t_offset = torch.tensor(_cam_t_in_tool(env), device=env.device)
    cam_pos_w = tool_pos_w + torch.einsum("nij,j->ni", R_tool, t_offset)  # [N, 3]
    q_offset  = torch.tensor([_CAM_Q_IN_TOOL], device=env.device).expand(N, -1)
    cam_quat_w = math_utils.quat_mul(tool_quat_w, q_offset)             # [N, 4] wxyz

    # 真实底盘 base_link 世界位姿。对于拼接 articulation，root_* 不一定等于 base_link。
    base_body_idx = asset.data.body_names.index("base_link")
    base_pos_w = asset.data.body_pos_w[:, base_body_idx, :]     # [N, 3]
    base_quat_w = asset.data.body_quat_w[:, base_body_idx, :]   # [N, 4] wxyz

    # T_cam_in_base = inv(T_base_in_world) · T_cam_in_world
    q_base_inv    = math_utils.quat_inv(base_quat_w)                    # [N, 4]
    q_cam_in_base = math_utils.quat_mul(q_base_inv, cam_quat_w)         # [N, 4]
    R_cam_in_base = _q_to_R(q_cam_in_base)                              # [N, 3, 3]
    delta         = cam_pos_w - base_pos_w                              # [N, 3]
    R_base_inv    = _q_to_R(q_base_inv)                                 # [N, 3, 3]
    t_cam_in_base = torch.bmm(R_base_inv, delta.unsqueeze(-1)).squeeze(-1)  # [N, 3]

    # 组装 T_cam_in_base [N, 4, 4]
    T = torch.eye(4, device=env.device).unsqueeze(0).expand(N, 4, 4).clone()
    T[:, :3, :3] = R_cam_in_base
    T[:, :3, 3]  = t_cam_in_base

    return T.reshape(N, 16)


def cam_link5_mount_in_tool_tf(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["XMS5_R800_W4G3B4C_link5", "tool_link"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Camera transform in tool_link frame for the link5-mounted camera.

    The link5->camera offset is fixed after initialization/reset; it is not
    recomputed from tool_link while the arm moves.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    link5_id = _body_index(asset.data.body_names, "XMS5_R800_W4G3B4C_link5", "link5")
    tool_id = _body_index(asset.data.body_names, "tool_link")
    link5_pos_w = asset.data.body_pos_w[:, link5_id, :]
    link5_quat_w = asset.data.body_quat_w[:, link5_id, :]
    tool_pos_w = asset.data.body_pos_w[:, tool_id, :]
    tool_quat_w = asset.data.body_quat_w[:, tool_id, :]

    t_link5_cam, q_link5_cam = _get_or_init_cam_link5_mount_offset(
        env, link5_pos_w, link5_quat_w, tool_pos_w, tool_quat_w
    )
    cam_pos_w, cam_quat_w = math_utils.combine_frame_transforms(
        link5_pos_w, link5_quat_w, t_link5_cam, q_link5_cam
    )
    t_cam_in_tool, q_cam_in_tool = math_utils.subtract_frame_transforms(
        tool_pos_w, tool_quat_w, cam_pos_w, cam_quat_w
    )
    T = torch.eye(4, device=env.device).unsqueeze(0).expand(env.num_envs, 4, 4).clone()
    T[:, :3, :3] = _q_to_R(q_cam_in_tool)
    T[:, :3, 3] = t_cam_in_tool
    return T.reshape(env.num_envs, 16)


def cam_link5_mount_in_base_tf(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["XMS5_R800_W4G3B4C_link5", "tool_link"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Camera transform in base_link frame for the link5-mounted camera."""
    asset: Articulation = env.scene[asset_cfg.name]
    link5_id = _body_index(asset.data.body_names, "XMS5_R800_W4G3B4C_link5", "link5")
    tool_id = _body_index(asset.data.body_names, "tool_link")
    link5_pos_w = asset.data.body_pos_w[:, link5_id, :]
    link5_quat_w = asset.data.body_quat_w[:, link5_id, :]
    tool_pos_w = asset.data.body_pos_w[:, tool_id, :]
    tool_quat_w = asset.data.body_quat_w[:, tool_id, :]

    t_link5_cam, q_link5_cam = _get_or_init_cam_link5_mount_offset(
        env, link5_pos_w, link5_quat_w, tool_pos_w, tool_quat_w
    )
    cam_pos_w, cam_quat_w = math_utils.combine_frame_transforms(
        link5_pos_w, link5_quat_w, t_link5_cam, q_link5_cam
    )

    base_body_idx = asset.data.body_names.index("base_link")
    base_pos_w = asset.data.body_pos_w[:, base_body_idx, :]
    base_quat_w = asset.data.body_quat_w[:, base_body_idx, :]
    t_cam_in_base, q_cam_in_base = math_utils.subtract_frame_transforms(
        base_pos_w, base_quat_w, cam_pos_w, cam_quat_w
    )

    T = torch.eye(4, device=env.device).unsqueeze(0).expand(env.num_envs, 4, 4).clone()
    T[:, :3, :3] = _q_to_R(q_cam_in_base)
    T[:, :3, 3] = t_cam_in_base
    return T.reshape(env.num_envs, 16)


def lidar_ranges(
    env: ManagerBasedEnv,
    sensor_name: str = "lidar",
) -> torch.Tensor:
    """3D LiDAR 各射线测距，shape [N, num_rays]，单位 m。

    未命中目标（inf/nan）替换为 max_distance，与 RayCaster.max_distance 保持一致。
    """
    from isaaclab.sensors import RayCaster

    sensor: RayCaster = env.scene.sensors[sensor_name]
    hits   = sensor.data.ray_hits_w                          # [N, num_rays, 3]
    origin = sensor.data.pos_w.unsqueeze(1)                  # [N, 1, 3]
    ranges = torch.norm(hits - origin, dim=-1)               # [N, num_rays]
    max_d  = sensor.cfg.max_distance
    ranges = torch.nan_to_num(ranges, nan=max_d, posinf=max_d, neginf=max_d)
    return ranges
