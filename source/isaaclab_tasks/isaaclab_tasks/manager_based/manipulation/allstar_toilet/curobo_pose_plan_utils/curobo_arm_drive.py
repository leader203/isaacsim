from __future__ import annotations

import torch

from .curobo_arm_control_config import ArmDriveOverrideConfig


def arm_drive_debug_suffix(robot, arm_joint_ids_fn, arm_joint_names_fn, format_tensor_fn) -> str:
    try:
        arm_ids = arm_joint_ids_fn(robot)
        arm_names = arm_joint_names_fn(robot)
        stiffness = robot.data.joint_stiffness[0, arm_ids].detach()
        damping = robot.data.joint_damping[0, arm_ids].detach()
        effort_limits = robot.data.joint_effort_limits[0, arm_ids].detach()
        velocity_limits = robot.data.joint_vel_limits[0, arm_ids].detach()
        soft_velocity_limits = robot.data.soft_joint_vel_limits[0, arm_ids].detach()
        return (
            f" arm_joint_names={arm_names} "
            f"arm_drive_stiffness={format_tensor_fn(stiffness)} "
            f"arm_drive_damping={format_tensor_fn(damping)} "
            f"arm_drive_effort_limit={format_tensor_fn(effort_limits)} "
            f"arm_drive_velocity_limit={format_tensor_fn(velocity_limits)} "
            f"arm_drive_soft_velocity_limit={format_tensor_fn(soft_velocity_limits)}"
        )
    except Exception as exc:
        return f" arm_drive_debug_error={exc!r}"


def apply_arm_drive_overrides(
    robot,
    *,
    label: str,
    cfg: ArmDriveOverrideConfig,
    arm_joint_ids_fn,
    arm_joint_names_fn,
    format_tensor_fn,
) -> None:
    arm_ids = arm_joint_ids_fn(robot)
    arm_names = arm_joint_names_fn(robot)
    num_envs = int(robot.data.joint_pos.shape[0])

    old_stiffness = robot.data.joint_stiffness[0, arm_ids].detach().clone()
    old_damping = robot.data.joint_damping[0, arm_ids].detach().clone()
    old_effort_limits = robot.data.joint_effort_limits[0, arm_ids].detach().clone()
    old_velocity_limits = robot.data.joint_vel_limits[0, arm_ids].detach().clone()
    old_soft_velocity_limits = robot.data.soft_joint_vel_limits[0, arm_ids].detach().clone()
    new_stiffness = old_stiffness.clone()
    new_damping = old_damping.clone()
    new_effort_limits = old_effort_limits.clone()
    new_velocity_limits = old_velocity_limits.clone()

    if cfg.stiffness is not None:
        new_stiffness[:] = cfg.stiffness
    if cfg.damping is not None:
        new_damping[:] = cfg.damping
    if cfg.effort_limit is not None:
        new_effort_limits[:] = cfg.effort_limit
    if cfg.velocity_limit is not None:
        new_velocity_limits[:] = cfg.velocity_limit

    joint2_local_idx = arm_names.index("joint2") if "joint2" in arm_names else None
    if joint2_local_idx is not None:
        if cfg.joint2_stiffness is not None:
            new_stiffness[joint2_local_idx] = cfg.joint2_stiffness
        if cfg.joint2_damping is not None:
            new_damping[joint2_local_idx] = cfg.joint2_damping
        if cfg.joint2_effort_limit is not None:
            new_effort_limits[joint2_local_idx] = cfg.joint2_effort_limit
        if cfg.joint2_velocity_limit is not None:
            new_velocity_limits[joint2_local_idx] = cfg.joint2_velocity_limit
    elif (
        cfg.joint2_stiffness is not None
        or cfg.joint2_damping is not None
        or cfg.joint2_effort_limit is not None
        or cfg.joint2_velocity_limit is not None
    ):
        print(
            f"[ArmDriveOverride] event=joint2_not_found label={label} "
            f"arm_joint_names={arm_names}",
            flush=True,
        )
    joint3_local_idx = arm_names.index("joint3") if "joint3" in arm_names else None
    if joint3_local_idx is not None:
        if cfg.joint3_stiffness is not None:
            new_stiffness[joint3_local_idx] = cfg.joint3_stiffness
        if cfg.joint3_damping is not None:
            new_damping[joint3_local_idx] = cfg.joint3_damping
        if cfg.joint3_effort_limit is not None:
            new_effort_limits[joint3_local_idx] = cfg.joint3_effort_limit
        if cfg.joint3_velocity_limit is not None:
            new_velocity_limits[joint3_local_idx] = cfg.joint3_velocity_limit
    elif (
        cfg.joint3_stiffness is not None
        or cfg.joint3_damping is not None
        or cfg.joint3_effort_limit is not None
        or cfg.joint3_velocity_limit is not None
    ):
        print(
            f"[ArmDriveOverride] event=joint3_not_found label={label} "
            f"arm_joint_names={arm_names}",
            flush=True,
        )

    def _expand(values: torch.Tensor) -> torch.Tensor:
        return values.reshape(1, -1).expand(num_envs, -1).contiguous()

    requested = cfg.requested
    if requested:
        if not torch.allclose(new_stiffness, old_stiffness):
            robot.write_joint_stiffness_to_sim(_expand(new_stiffness), joint_ids=arm_ids)
        if not torch.allclose(new_damping, old_damping):
            robot.write_joint_damping_to_sim(_expand(new_damping), joint_ids=arm_ids)
        if not torch.allclose(new_effort_limits, old_effort_limits):
            robot.write_joint_effort_limit_to_sim(_expand(new_effort_limits), joint_ids=arm_ids)
        if not torch.allclose(new_velocity_limits, old_velocity_limits):
            expanded_velocity_limits = _expand(new_velocity_limits)
            robot.write_joint_velocity_limit_to_sim(expanded_velocity_limits, joint_ids=arm_ids)
            robot.data.soft_joint_vel_limits[:, arm_ids] = expanded_velocity_limits
            arm_actuator = getattr(robot, "actuators", {}).get("arm") if hasattr(robot, "actuators") else None
            if arm_actuator is not None:
                if hasattr(arm_actuator, "velocity_limit_sim"):
                    arm_actuator.velocity_limit_sim[:, :] = expanded_velocity_limits
                if hasattr(arm_actuator, "velocity_limit"):
                    arm_actuator.velocity_limit[:, :] = expanded_velocity_limits

    actual_stiffness = robot.data.joint_stiffness[0, arm_ids].detach()
    actual_damping = robot.data.joint_damping[0, arm_ids].detach()
    actual_effort_limits = robot.data.joint_effort_limits[0, arm_ids].detach()
    actual_velocity_limits = robot.data.joint_vel_limits[0, arm_ids].detach()
    actual_soft_velocity_limits = robot.data.soft_joint_vel_limits[0, arm_ids].detach()
    print(
        f"[ArmDriveOverride] event={'applied' if requested else 'snapshot'} label={label} "
        f"arm_joint_ids={arm_ids} arm_joint_names={arm_names} "
        f"joint2_local_idx={joint2_local_idx} joint3_local_idx={joint3_local_idx} "
        f"old_stiffness={format_tensor_fn(old_stiffness)} new_stiffness={format_tensor_fn(actual_stiffness)} "
        f"old_damping={format_tensor_fn(old_damping)} new_damping={format_tensor_fn(actual_damping)} "
        f"old_effort_limit={format_tensor_fn(old_effort_limits)} "
        f"new_effort_limit={format_tensor_fn(actual_effort_limits)} "
        f"old_velocity_limit={format_tensor_fn(old_velocity_limits)} "
        f"new_velocity_limit={format_tensor_fn(actual_velocity_limits)} "
        f"old_soft_velocity_limit={format_tensor_fn(old_soft_velocity_limits)} "
        f"new_soft_velocity_limit={format_tensor_fn(actual_soft_velocity_limits)}",
        flush=True,
    )


def apply_arm_drive_overrides_to_env_cfg(
    env_cfg,
    *,
    label: str,
    cfg: ArmDriveOverrideConfig,
) -> None:
    if not cfg.requested:
        print(f"[ArmDriveOverrideCfg] event=not_requested label={label}", flush=True)
        return
    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    actuators = getattr(robot_cfg, "actuators", None)
    if not isinstance(actuators, dict) or "arm" not in actuators:
        print(
            f"[ArmDriveOverrideCfg] event=arm_actuator_not_found label={label} "
            f"robot_cfg_type={type(robot_cfg).__name__}",
            flush=True,
        )
        return

    arm_actuator = actuators["arm"]
    old_stiffness = getattr(arm_actuator, "stiffness", None)
    old_damping = getattr(arm_actuator, "damping", None)
    old_effort_limit = getattr(arm_actuator, "effort_limit_sim", None)
    old_velocity_limit = getattr(arm_actuator, "velocity_limit_sim", None)
    old_soft_velocity_limit = getattr(arm_actuator, "velocity_limit", None)

    def _arm_joint_map(
        default_value: float | None,
        joint2_value: float | None,
        joint3_value: float | None,
        old_value,
    ):
        if default_value is None and joint2_value is None and joint3_value is None:
            return old_value
        old_by_joint = old_value if isinstance(old_value, dict) else {}
        value_by_joint = {}
        for joint_idx in range(1, 7):
            joint_name = f"joint{joint_idx}"
            if default_value is not None:
                value_by_joint[joint_name] = float(default_value)
            elif joint_name in old_by_joint:
                value_by_joint[joint_name] = old_by_joint[joint_name]
            else:
                value_by_joint[joint_name] = old_value
        if joint2_value is not None:
            value_by_joint["joint2"] = float(joint2_value)
        if joint3_value is not None:
            value_by_joint["joint3"] = float(joint3_value)
        return value_by_joint

    arm_actuator.stiffness = _arm_joint_map(cfg.stiffness, cfg.joint2_stiffness, cfg.joint3_stiffness, old_stiffness)
    arm_actuator.damping = _arm_joint_map(cfg.damping, cfg.joint2_damping, cfg.joint3_damping, old_damping)
    arm_actuator.effort_limit_sim = _arm_joint_map(
        cfg.effort_limit,
        cfg.joint2_effort_limit,
        cfg.joint3_effort_limit,
        old_effort_limit,
    )
    new_velocity_limit = _arm_joint_map(
        cfg.velocity_limit,
        cfg.joint2_velocity_limit,
        cfg.joint3_velocity_limit,
        old_velocity_limit,
    )
    arm_actuator.velocity_limit_sim = new_velocity_limit
    arm_actuator.velocity_limit = new_velocity_limit
    print(
        f"[ArmDriveOverrideCfg] event=applied label={label} "
        f"old_stiffness={old_stiffness} new_stiffness={arm_actuator.stiffness} "
        f"old_damping={old_damping} new_damping={arm_actuator.damping} "
        f"old_effort_limit={old_effort_limit} new_effort_limit={arm_actuator.effort_limit_sim} "
        f"old_velocity_limit={old_velocity_limit} new_velocity_limit={arm_actuator.velocity_limit_sim} "
        f"old_soft_velocity_limit={old_soft_velocity_limit} new_soft_velocity_limit={arm_actuator.velocity_limit}",
        flush=True,
    )
