"""Mobile base command and stabilization helpers."""

import numpy as np
import torch

from isaaclab.utils.math import quat_apply, quat_from_angle_axis


_MOTION_BASED_WHEEL_RADIUS = 0.077993
_MOTION_BASED_WHEEL_BASE_SUM = 0.1575 + 0.1725
_MOTION_BASED_MAX_LIN_SPEED_MPS = 0.60
_PHYSX_ROOT_HEIGHT_KP = 8.0
_PHYSX_ROOT_HEIGHT_MAX_VEL = 0.35
_PHYSX_ROOT_UPRIGHT_KP = 10.0
_PHYSX_ROOT_UPRIGHT_MAX_ANG_VEL = 1.5


def _make_wheel_action(vx: float, vy: float, wz: float, device: str) -> torch.Tensor:
    """vx/vy/wz → 麦轮 4 轮速度 [1,4]。"""
    fl = vx - vy - wz
    fr = vx + vy + wz
    rl = vx + vy - wz
    rr = vx - vy + wz
    return torch.tensor([[fl, fr, rl, rr]], dtype=torch.float32, device=device)


def _make_motion_based_wheel_visual_action(vx: float, vy: float, wz: float, device: str) -> torch.Tensor:
    """Body twist → mecanum wheel angular velocity for motion-based visual rolling."""
    fl = (vx - vy - _MOTION_BASED_WHEEL_BASE_SUM * wz) / _MOTION_BASED_WHEEL_RADIUS
    fr = (vx + vy + _MOTION_BASED_WHEEL_BASE_SUM * wz) / _MOTION_BASED_WHEEL_RADIUS
    rl = (vx + vy - _MOTION_BASED_WHEEL_BASE_SUM * wz) / _MOTION_BASED_WHEEL_RADIUS
    rr = (vx - vy + _MOTION_BASED_WHEEL_BASE_SUM * wz) / _MOTION_BASED_WHEEL_RADIUS
    return torch.tensor([[fl, fr, rl, rr]], dtype=torch.float32, device=device)


def _quat_wxyz_to_yaw_tensor(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Extract planar yaw from wxyz quaternion."""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat_tensor(yaw: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Create upright wxyz quaternion from planar yaw."""
    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=device).repeat(yaw.shape[0], 1)
    return quat_from_angle_axis(yaw, axis)


def _legacy_operator_to_motion_based_twist(vx: float, vy: float, wz: float) -> tuple[float, float, float]:
    """Map legacy teleop wheel-space commands to physical body twist for motion-based root driving.

    The original script used `speed_scale` as wheel angular speed scale.
    For motion-based control we keep operator feel compatible by converting:
      vx_legacy, vy_legacy [rad/s-like] -> linear m/s using wheel radius
      wz_legacy [rad/s-like] -> yaw rad/s using r / (Lx + Ly)
    """
    vx_mps = vx * _MOTION_BASED_WHEEL_RADIUS
    vy_mps = vy * _MOTION_BASED_WHEEL_RADIUS
    wz_rps = wz * (_MOTION_BASED_WHEEL_RADIUS / _MOTION_BASED_WHEEL_BASE_SUM)
    planar_norm = float(np.hypot(vx_mps, vy_mps))
    if planar_norm > _MOTION_BASED_MAX_LIN_SPEED_MPS and planar_norm > 1.0e-8:
        scale = _MOTION_BASED_MAX_LIN_SPEED_MPS / planar_norm
        vx_mps *= scale
        vy_mps *= scale
    return vx_mps, vy_mps, wz_rps


def _make_upright_root_pose(
    current_root_pose_w: torch.Tensor,
    nominal_z: torch.Tensor,
    yaw_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp root pose to yaw-only upright orientation and fixed nominal z."""
    yaw = _quat_wxyz_to_yaw_tensor(current_root_pose_w[:, 3:7]) if yaw_override is None else yaw_override
    upright_pose_w = current_root_pose_w.clone()
    upright_pose_w[:, 2] = nominal_z
    upright_pose_w[:, 3:7] = _yaw_to_quat_tensor(yaw, str(current_root_pose_w.device), current_root_pose_w.dtype)
    return upright_pose_w, yaw


def _physx_root_stabilization_velocity(
    current_root_pose_w: torch.Tensor,
    nominal_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Velocity-level height/upright correction for Direct GPU root-velocity control."""
    root_vel_w = torch.zeros(
        (current_root_pose_w.shape[0], 6),
        device=current_root_pose_w.device,
        dtype=current_root_pose_w.dtype,
    )

    z_error = nominal_z.to(device=current_root_pose_w.device, dtype=current_root_pose_w.dtype) - current_root_pose_w[:, 2]
    root_vel_w[:, 2] = torch.clamp(
        _PHYSX_ROOT_HEIGHT_KP * z_error,
        min=-_PHYSX_ROOT_HEIGHT_MAX_VEL,
        max=_PHYSX_ROOT_HEIGHT_MAX_VEL,
    )

    world_up = torch.tensor([[0.0, 0.0, 1.0]], dtype=current_root_pose_w.dtype, device=current_root_pose_w.device)
    world_up = world_up.repeat(current_root_pose_w.shape[0], 1)
    body_up_w = quat_apply(current_root_pose_w[:, 3:7], world_up)
    upright_axis_w = torch.linalg.cross(body_up_w, world_up, dim=-1)
    root_vel_w[:, 3:5] = torch.clamp(
        _PHYSX_ROOT_UPRIGHT_KP * upright_axis_w[:, 0:2],
        min=-_PHYSX_ROOT_UPRIGHT_MAX_ANG_VEL,
        max=_PHYSX_ROOT_UPRIGHT_MAX_ANG_VEL,
    )

    return root_vel_w, _quat_wxyz_to_yaw_tensor(current_root_pose_w[:, 3:7])


def _apply_motion_based_base(
    robot,
    vx: float,
    vy: float,
    wz: float,
    dt: float,
    device: str,
    previous_root_pose_w: torch.Tensor,
    nominal_z: torch.Tensor,
    yaw_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate root motion from the cached previous-step pose."""
    root_pose_w, yaw = _make_upright_root_pose(previous_root_pose_w.clone(), nominal_z, yaw_target)
    root_vel_w = torch.zeros((root_pose_w.shape[0], 6), device=device, dtype=root_pose_w.dtype)

    quat_w = root_pose_w[:, 3:7]
    body_vel_b = torch.tensor([[vx, vy, 0.0]], dtype=root_pose_w.dtype, device=device)
    lin_vel_w = quat_apply(quat_w, body_vel_b)

    root_pose_w[:, 0] += lin_vel_w[:, 0] * dt
    root_pose_w[:, 1] += lin_vel_w[:, 1] * dt

    next_yaw = yaw + wz * dt
    root_pose_w[:, 3:7] = _yaw_to_quat_tensor(next_yaw, device, root_pose_w.dtype)

    root_vel_w[:, 0] = lin_vel_w[:, 0]
    root_vel_w[:, 1] = lin_vel_w[:, 1]
    root_vel_w[:, 5] = wz

    robot.write_root_pose_to_sim(root_pose_w)
    robot.write_root_velocity_to_sim(root_vel_w)
    return _make_motion_based_wheel_visual_action(vx, vy, wz, device), next_yaw, root_pose_w.detach().clone()


def _apply_motion_based_base_velocity(
    robot,
    vx: float,
    vy: float,
    wz: float,
    device: str,
    current_root_pose_w: torch.Tensor,
    nominal_z: torch.Tensor,
    yaw_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Command planar root velocity without writing articulation poses during GPU simulation."""
    yaw = _quat_wxyz_to_yaw_tensor(current_root_pose_w[:, 3:7])
    planar_quat_w = _yaw_to_quat_tensor(yaw, device, current_root_pose_w.dtype)
    root_vel_w, _ = _physx_root_stabilization_velocity(current_root_pose_w, nominal_z)

    body_vel_b = torch.tensor([[vx, vy, 0.0]], dtype=current_root_pose_w.dtype, device=device)
    lin_vel_w = quat_apply(planar_quat_w, body_vel_b)
    root_vel_w[:, 0] = lin_vel_w[:, 0]
    root_vel_w[:, 1] = lin_vel_w[:, 1]
    root_vel_w[:, 5] = wz

    robot.write_root_velocity_to_sim(root_vel_w)
    return _make_motion_based_wheel_visual_action(vx, vy, wz, device), yaw, current_root_pose_w.detach().clone()


class _MotionBasedWheelVisualizer:
    """Drive wheel visuals without letting wheel joints generate chassis traction."""

    def __init__(self, robot, write_joint_state: bool = True):
        self.robot = robot
        self.write_joint_state = write_joint_state
        self.joint_ids, self.joint_names = robot.find_joints("wheel_(fl|fr|rl|rr)_joint", preserve_order=True)
        self.joint_ids = list(self.joint_ids)
        self.wheel_pos = robot.data.joint_pos[:, self.joint_ids].detach().clone()
        self.zero_vel = torch.zeros_like(self.wheel_pos)
        self._resync_threshold_rad = 0.5
        # Motion-based chassis locomotion is handled at the root. Disable wheel drive
        # torque here so the visual wheel spin cannot fight root motion through contact.
        robot.write_joint_stiffness_to_sim(0.0, joint_ids=self.joint_ids)
        robot.write_joint_damping_to_sim(0.0, joint_ids=self.joint_ids)
        robot.write_joint_effort_limit_to_sim(0.0, joint_ids=self.joint_ids)
        robot.set_joint_velocity_target(self.zero_vel, joint_ids=self.joint_ids)
        if self.write_joint_state:
            robot.write_joint_state_to_sim(self.wheel_pos, self.zero_vel, joint_ids=self.joint_ids)
            rolling_mode = "visual-only rolling"
        else:
            rolling_mode = "joint-state writes disabled for Direct GPU root velocity"
        print(
            f"[MotionBasedBase] disabled wheel traction drives ({rolling_mode}): {self.joint_names}",
            flush=True,
        )

    def sync_from_sim(self) -> None:
        self.wheel_pos = self.robot.data.joint_pos[:, self.joint_ids].detach().clone()
        self.zero_vel = torch.zeros_like(self.wheel_pos)

    def apply(self, wheel_omega: torch.Tensor, dt: float) -> None:
        sim_wheel_pos = self.robot.data.joint_pos[:, self.joint_ids].detach()
        if float(torch.max(torch.abs(sim_wheel_pos - self.wheel_pos)).item()) > self._resync_threshold_rad:
            self.wheel_pos = sim_wheel_pos.clone()
        wheel_omega = wheel_omega.to(device=self.robot.device, dtype=self.wheel_pos.dtype)
        self.wheel_pos = torch.remainder(self.wheel_pos + wheel_omega * dt + torch.pi, 2.0 * torch.pi) - torch.pi
        self.robot.set_joint_velocity_target(self.zero_vel, joint_ids=self.joint_ids)
        if self.write_joint_state:
            self.robot.write_joint_state_to_sim(self.wheel_pos, self.zero_vel, joint_ids=self.joint_ids)

