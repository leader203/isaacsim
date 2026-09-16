"""Base hold and static-motion stabilizer helpers."""

from __future__ import annotations

import time

import torch
from isaaclab.utils.math import quat_apply


_STATIC_BASE_ROOT_Z = 0.0
_STATIC_BASE_HEIGHT_KP = 8.0
_STATIC_BASE_HEIGHT_MAX_VEL = 0.35
_STATIC_BASE_UPRIGHT_KP = 10.0
_STATIC_BASE_UPRIGHT_MAX_ANG_VEL = 1.5


def _hold_base_stationary(robot) -> None:
    wheel_ids, _ = robot.find_joints("wheel_.*_joint", preserve_order=True)
    wheel_ids = list(wheel_ids)
    zero = torch.zeros((robot.num_instances, len(wheel_ids)), device=robot.device, dtype=robot.data.joint_pos.dtype)
    robot.set_joint_velocity_target(zero, joint_ids=wheel_ids)
    stabilizer = getattr(robot, "_allstar_static_motion_base_stabilizer", None)
    if stabilizer is not None:
        stabilizer.apply("hold_base_stationary")


def _fixed_root_pose_w(robot) -> torch.Tensor:
    return torch.cat((robot.data.root_pos_w[0], robot.data.root_quat_w[0]), dim=0).unsqueeze(0)


def _root_link_pose_w(robot) -> torch.Tensor:
    root_link_pose_w = getattr(robot.data, "root_link_pose_w", None)
    if root_link_pose_w is not None:
        return root_link_pose_w.detach().clone()
    return torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), dim=-1).detach().clone()


def _default_root_z(robot) -> float:
    default_root_state = getattr(robot.data, "default_root_state", None)
    if default_root_state is not None:
        return float(default_root_state[0, 2].item())
    default_root_pos = getattr(robot.data, "default_root_pos_w", None)
    if default_root_pos is not None:
        return float(default_root_pos[0, 2].item())
    return _STATIC_BASE_ROOT_Z


def _static_base_stabilization_velocity(robot, nominal_z: torch.Tensor | float) -> torch.Tensor:
    current_root_pose_w = _root_link_pose_w(robot)
    root_vel_w = torch.zeros(
        (current_root_pose_w.shape[0], 6),
        device=current_root_pose_w.device,
        dtype=current_root_pose_w.dtype,
    )
    nominal_z_t = torch.as_tensor(nominal_z, device=current_root_pose_w.device, dtype=current_root_pose_w.dtype)
    if nominal_z_t.dim() == 0:
        nominal_z_t = nominal_z_t.expand(current_root_pose_w.shape[0])

    z_error = nominal_z_t - current_root_pose_w[:, 2]
    z_correction = torch.clamp(
        _STATIC_BASE_HEIGHT_KP * z_error,
        min=-_STATIC_BASE_HEIGHT_MAX_VEL,
        max=_STATIC_BASE_HEIGHT_MAX_VEL,
    )
    root_vel_w[:, 2] = z_correction

    world_up = torch.tensor([[0.0, 0.0, 1.0]], dtype=current_root_pose_w.dtype, device=current_root_pose_w.device)
    world_up = world_up.repeat(current_root_pose_w.shape[0], 1)
    body_up_w = quat_apply(current_root_pose_w[:, 3:7], world_up)
    upright_axis_w = torch.linalg.cross(body_up_w, world_up, dim=-1)
    root_vel_w[:, 3:5] = torch.clamp(
        _STATIC_BASE_UPRIGHT_KP * upright_axis_w[:, 0:2],
        min=-_STATIC_BASE_UPRIGHT_MAX_ANG_VEL,
        max=_STATIC_BASE_UPRIGHT_MAX_ANG_VEL,
    )
    return root_vel_w


class _ScriptLocalStaticMotionBaseStabilizer:
    """Keep the mecanum chassis upright through root velocity, matching the recorder task mode."""

    def __init__(self, robot, nominal_z: float | None = None):
        self.robot = robot
        self.nominal_z = _default_root_z(robot) if nominal_z is None else float(nominal_z)
        self.joint_ids, self.joint_names = self._find_wheel_joints()
        self.zero_wheel_vel = torch.zeros(
            (robot.num_instances, len(self.joint_ids)),
            device=robot.device,
            dtype=robot.data.joint_pos.dtype,
        )
        robot.write_joint_stiffness_to_sim(0.0, joint_ids=self.joint_ids)
        robot.write_joint_damping_to_sim(0.0, joint_ids=self.joint_ids)
        robot.write_joint_effort_limit_to_sim(0.0, joint_ids=self.joint_ids)
        robot.set_joint_velocity_target(self.zero_wheel_vel, joint_ids=self.joint_ids)
        self.apply("init")
        root_z = _root_link_pose_w(robot)[:, 2].detach().cpu().tolist()
        print(
            f"[StaticMotionBase] disabled wheel traction drives; nominal_root_z={self.nominal_z:.3f} "
            f"height_max_vel_mps={_STATIC_BASE_HEIGHT_MAX_VEL:.3f} "
            f"root_z={root_z} wheel_joints={self.joint_names}",
            flush=True,
        )

    def _find_wheel_joints(self) -> tuple[list[int], list[str]]:
        try:
            joint_ids, joint_names = self.robot.find_joints("wheel_(fl|fr|rl|rr)_joint", preserve_order=True)
        except ValueError:
            joint_ids, joint_names = self.robot.find_joints("wheel_.*_joint", preserve_order=True)
        return list(joint_ids), list(joint_names)

    def apply(self, reason: str = "") -> None:
        if bool(getattr(self.robot, "_allstar_motion_base_teleop_active", False)):
            return
        self.robot.set_joint_velocity_target(self.zero_wheel_vel, joint_ids=self.joint_ids)
        root_vel_w = _static_base_stabilization_velocity(self.robot, self.nominal_z)
        self.robot.write_root_velocity_to_sim(root_vel_w)


def _install_static_motion_base_stabilizer(
    env,
    robot,
    *,
    patch_env_step: bool = True,
) -> _ScriptLocalStaticMotionBaseStabilizer:
    existing = getattr(robot, "_allstar_static_motion_base_stabilizer", None)
    if existing is not None:
        return existing
    stabilizer = _ScriptLocalStaticMotionBaseStabilizer(robot)
    if not patch_env_step:
        setattr(robot, "_allstar_static_motion_base_stabilizer", stabilizer)
        return stabilizer
    original_step = env.step

    def stabilized_step(*args, **kwargs):
        stabilizer.apply("pre_env_step")
        result = original_step(*args, **kwargs)
        stabilizer.apply("post_env_step")
        return result

    env.step = stabilized_step
    setattr(robot, "_allstar_static_motion_base_stabilizer", stabilizer)
    return stabilizer


def _wait_until_static_motion_base_settled(
    env,
    robot,
    stabilizer: _ScriptLocalStaticMotionBaseStabilizer,
    *,
    step_fn,
    min_settle_time_s: float = 1.5,
    max_wait_time_s: float = 8.0,
    linear_vel_threshold: float = 0.03,
    angular_vel_threshold: float = 0.08,
    z_vel_threshold: float = 0.015,
    poll_interval_steps: int = 1,
) -> bool:
    start_wall = time.monotonic()
    stable_since: float | None = None
    next_report_wall = start_wall
    settle_step = 0

    while True:
        stabilizer.apply("pre_settle_step")
        settle_step += 1
        step_fn(env, render=(settle_step % 4 == 0))
        stabilizer.apply("post_settle_step")

        root_lin_vel = robot.data.root_lin_vel_w[0]
        root_ang_vel = robot.data.root_ang_vel_w[0]
        lin_norm = float(torch.linalg.norm(root_lin_vel[:2]).item())
        ang_norm = float(torch.linalg.norm(root_ang_vel).item())
        z_vel = abs(float(root_lin_vel[2].item()))

        settled_now = (
            lin_norm <= linear_vel_threshold
            and ang_norm <= angular_vel_threshold
            and z_vel <= z_vel_threshold
        )

        now_wall = time.monotonic()
        if now_wall >= next_report_wall:
            root_z = float(_root_link_pose_w(robot)[0, 2].item())
            print(
                "[Settle] "
                f"lin_xy={lin_norm:.4f}  ang={ang_norm:.4f}  z_vel={z_vel:.4f}  root_z={root_z:.4f}  "
                f"stable_for={0.0 if stable_since is None else now_wall - stable_since:.2f}s",
                flush=True,
            )
            next_report_wall = now_wall + 0.5

        if settled_now:
            if stable_since is None:
                stable_since = now_wall
            elif now_wall - stable_since >= min_settle_time_s:
                return True
        else:
            stable_since = None

        if now_wall - start_wall >= max_wait_time_s:
            return False

        for _ in range(max(0, poll_interval_steps - 1)):
            stabilizer.apply("pre_settle_poll_step")
            settle_step += 1
            step_fn(env, render=(settle_step % 4 == 0))
            stabilizer.apply("post_settle_poll_step")
