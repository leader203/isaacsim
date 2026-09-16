from __future__ import annotations

from dataclasses import dataclass

import torch


def checked_nonnegative_arg(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


@dataclass(frozen=True)
class ArmHoldEffortConfig:
    bias: list[float] | None
    joint2_bias: float | None
    joint3_bias: float | None
    joint5_bias: float | None
    min_abs_joint3: float = 0.0

    @property
    def requested(self) -> bool:
        return (
            self.bias is not None
            or self.joint2_bias is not None
            or self.joint3_bias is not None
            or self.joint5_bias is not None
        )


def arm_hold_effort_config_from_args(args_cli) -> ArmHoldEffortConfig:
    bias = list(args_cli.arm_hold_effort_bias) if args_cli.arm_hold_effort_bias is not None else None
    return ArmHoldEffortConfig(
        bias=bias,
        joint2_bias=args_cli.arm_hold_effort_bias_joint2,
        joint3_bias=args_cli.arm_hold_effort_bias_joint3,
        joint5_bias=args_cli.arm_hold_effort_bias_joint5,
        min_abs_joint3=float(getattr(args_cli, "arm_hold_effort_bias_min_abs_joint3", 0.0)),
    )


def arm_hold_effort_bias_tensor(
    robot,
    arm_ids: list[int],
    cfg: ArmHoldEffortConfig,
    target_q: torch.Tensor | None = None,
) -> torch.Tensor:
    zero = torch.zeros((1, len(arm_ids)), device=robot.device, dtype=robot.data.joint_pos.dtype)
    if target_q is not None and cfg.min_abs_joint3 > 0.0:
        target_flat = target_q.detach().to(device=robot.device, dtype=robot.data.joint_pos.dtype).reshape(-1)
        if len(target_flat) >= 3 and abs(float(target_flat[2].item())) < cfg.min_abs_joint3:
            return zero

    values = list(cfg.bias) if cfg.bias is not None else [0.0] * len(arm_ids)
    if len(values) != len(arm_ids):
        raise ValueError(f"--arm_hold_effort_bias must have {len(arm_ids)} values, got {len(values)}.")
    if cfg.joint2_bias is not None and len(values) >= 2:
        values[1] = float(cfg.joint2_bias)
    if cfg.joint3_bias is not None and len(values) >= 3:
        values[2] = float(cfg.joint3_bias)
    if cfg.joint5_bias is not None and len(values) >= 5:
        values[4] = float(cfg.joint5_bias)
    return torch.tensor(values, device=robot.device, dtype=robot.data.joint_pos.dtype).reshape(1, -1)


@dataclass(frozen=True)
class ArmDriveOverrideConfig:
    stiffness: float | None = None
    damping: float | None = None
    effort_limit: float | None = None
    velocity_limit: float | None = None
    joint2_stiffness: float | None = None
    joint2_damping: float | None = None
    joint2_effort_limit: float | None = None
    joint2_velocity_limit: float | None = None
    joint3_stiffness: float | None = None
    joint3_damping: float | None = None
    joint3_effort_limit: float | None = None
    joint3_velocity_limit: float | None = None

    @property
    def requested(self) -> bool:
        return any(value is not None for value in self.__dict__.values())


def arm_drive_override_config_from_args(args_cli) -> ArmDriveOverrideConfig:
    return ArmDriveOverrideConfig(
        stiffness=checked_nonnegative_arg("--arm_drive_stiffness", args_cli.arm_drive_stiffness),
        damping=checked_nonnegative_arg("--arm_drive_damping", args_cli.arm_drive_damping),
        effort_limit=checked_nonnegative_arg("--arm_drive_effort_limit", args_cli.arm_drive_effort_limit),
        velocity_limit=checked_nonnegative_arg("--arm_drive_velocity_limit", args_cli.arm_drive_velocity_limit),
        joint2_stiffness=checked_nonnegative_arg(
            "--arm_drive_joint2_stiffness",
            args_cli.arm_drive_joint2_stiffness,
        ),
        joint2_damping=checked_nonnegative_arg("--arm_drive_joint2_damping", args_cli.arm_drive_joint2_damping),
        joint2_effort_limit=checked_nonnegative_arg(
            "--arm_drive_joint2_effort_limit",
            args_cli.arm_drive_joint2_effort_limit,
        ),
        joint2_velocity_limit=checked_nonnegative_arg(
            "--arm_drive_joint2_velocity_limit",
            args_cli.arm_drive_joint2_velocity_limit,
        ),
        joint3_stiffness=checked_nonnegative_arg(
            "--arm_drive_joint3_stiffness",
            args_cli.arm_drive_joint3_stiffness,
        ),
        joint3_damping=checked_nonnegative_arg("--arm_drive_joint3_damping", args_cli.arm_drive_joint3_damping),
        joint3_effort_limit=checked_nonnegative_arg(
            "--arm_drive_joint3_effort_limit",
            args_cli.arm_drive_joint3_effort_limit,
        ),
        joint3_velocity_limit=checked_nonnegative_arg(
            "--arm_drive_joint3_velocity_limit",
            args_cli.arm_drive_joint3_velocity_limit,
        ),
    )
