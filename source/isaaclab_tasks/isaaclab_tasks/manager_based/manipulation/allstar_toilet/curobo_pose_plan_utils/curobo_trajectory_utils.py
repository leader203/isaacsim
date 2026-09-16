from __future__ import annotations

import torch


def multi_pose_target_waypoint_range(plan_result: dict, traj_len: int) -> tuple[int, int]:
    metrics = plan_result.get("metrics") if isinstance(plan_result, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    waypoint_errors = metrics.get("waypoint_pose_errors") or []
    target_waypoints: list[int] = []
    for item in waypoint_errors:
        if not isinstance(item, dict) or "nearest_waypoint" not in item:
            continue
        try:
            target_waypoints.append(int(item["nearest_waypoint"]))
        except (TypeError, ValueError):
            continue
    if target_waypoints:
        start_idx = target_waypoints[0]
        end_idx = target_waypoints[-1]
    else:
        segment_lengths = metrics.get("segment_lengths") or []
        if segment_lengths:
            try:
                start_idx = int(segment_lengths[0]) - 1
            except (TypeError, ValueError):
                start_idx = 0
        else:
            start_idx = 0
        end_idx = traj_len - 1
    start_idx = max(0, min(traj_len - 1, start_idx))
    end_idx = max(start_idx, min(traj_len - 1, end_idx))
    return start_idx, end_idx


def apply_roll_delta_over_waypoint_range(
    robot,
    traj: torch.Tensor,
    *,
    roll_rad: float,
    start_idx: int,
    end_idx: int,
    label: str,
    clamp_arm_joint_target_to_soft_limits_fn,
    format_tensor_fn,
) -> torch.Tensor:
    rolled_traj = traj.detach().clone()
    requested_final_q = rolled_traj[end_idx].detach().clone()
    requested_final_q[-1] = requested_final_q[-1] + float(roll_rad)
    clamped_final_q = clamp_arm_joint_target_to_soft_limits_fn(robot, requested_final_q, label=label)
    applied_roll_delta = clamped_final_q[-1] - rolled_traj[end_idx, -1]
    if abs(float(applied_roll_delta.item())) > 1.0e-6:
        blend = torch.linspace(
            0.0,
            1.0,
            end_idx - start_idx + 1,
            device=rolled_traj.device,
            dtype=rolled_traj.dtype,
        )
        rolled_traj[start_idx : end_idx + 1, -1] = rolled_traj[start_idx : end_idx + 1, -1] + blend * applied_roll_delta
        if end_idx + 1 < rolled_traj.shape[0]:
            rolled_traj[end_idx + 1 :, -1] = rolled_traj[end_idx + 1 :, -1] + applied_roll_delta
    print(
        f"[EndEffectorRollOverTrajectory] label={label} start_waypoint={start_idx} end_waypoint={end_idx} "
        f"requested_roll_delta={float(roll_rad):.6f} applied_roll_delta={float(applied_roll_delta.item()):.6f} "
        f"requested_final_q={format_tensor_fn(requested_final_q)} clamped_final_q={format_tensor_fn(clamped_final_q)} "
        f"rolled_final_q={format_tensor_fn(rolled_traj[-1])}",
        flush=True,
    )
    return rolled_traj


def trajectory_waypoint_velocity(
    env,
    robot,
    traj: torch.Tensor,
    waypoint_idx: int,
    step_budget: int,
    *,
    enabled: bool,
    scale: float,
    arm_joint_ids_fn,
) -> torch.Tensor | None:
    if not enabled:
        return None
    if len(traj) <= 1:
        return None
    if waypoint_idx >= len(traj) - 1:
        return torch.zeros_like(traj[waypoint_idx])
    step_dt = float(getattr(env, "step_dt", getattr(env, "physics_dt", 1.0 / 120.0)))
    env_dt = max(1e-6, float(step_budget) * max(1e-6, step_dt))
    vel = (traj[waypoint_idx + 1] - traj[waypoint_idx]) / env_dt * float(scale)
    vel_limits = robot.data.soft_joint_vel_limits[0, arm_joint_ids_fn(robot)].detach()
    finite_limits = torch.where(torch.isfinite(vel_limits), vel_limits, torch.full_like(vel_limits, 1.0e6))
    return torch.clamp(vel, min=-finite_limits, max=finite_limits)
