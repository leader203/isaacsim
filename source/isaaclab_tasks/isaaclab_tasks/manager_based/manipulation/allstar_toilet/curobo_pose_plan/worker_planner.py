"""Worker implementation for isolated cuRobo pose planning.

This script must not import Isaac Sim or Isaac Lab. It only consumes a JSON
payload and returns JSON results so the main test script can keep cuRobo out of
the simulator process.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import traceback
import yaml

import torch


def _log(message: str) -> None:
    print(f"[cuRoboWorker][stage] {message}", file=sys.stderr, flush=True)


def _load_payload(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_local_nvidia_curobo_on_path() -> None:
    curobo_src_root = Path(__file__).resolve().parents[1] / "src" / "nvidia-curobo"
    if not curobo_src_root.is_dir():
        return
    curobo_src_root_str = str(curobo_src_root)
    if curobo_src_root_str not in sys.path:
        sys.path.insert(0, curobo_src_root_str)


def _format_result(name: str, success: bool, elapsed_s: float, trajectory, metrics: dict | None = None):
    return {
        "name": name,
        "success": success,
        "elapsed_s": elapsed_s,
        "trajectory": trajectory,
        "metrics": metrics or {},
    }


def _pose_error(target_pos, target_quat, actual_pos, actual_quat):
    pos_error = torch.linalg.norm(actual_pos.reshape(-1, 3) - target_pos.reshape(-1, 3), dim=-1)
    quat_dot = torch.sum(target_quat.reshape(-1, 4) * actual_quat.reshape(-1, 4), dim=-1).abs().clamp(max=1.0)
    rot_error_rad = 2.0 * torch.acos(quat_dot)
    rot_error_deg = rot_error_rad * 180.0 / torch.pi
    return pos_error, rot_error_rad, rot_error_deg


def _min_float_tensor(value) -> float | None:
    if value is None:
        return None
    return float(value.detach().reshape(-1).min().item())


def _count_true(value) -> int | None:
    if value is None:
        return None
    return int(torch.count_nonzero(value).item())


def _small_tensor_summary(value, *, max_values: int = 16):
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            flat = value.detach().reshape(-1)
            values = flat[:max_values].cpu().tolist()
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(flat.min().item()) if flat.numel() else None,
                "max": float(flat.max().item()) if flat.numel() else None,
                "values": [float(v) if isinstance(v, (int, float)) else v for v in values],
                "truncated": int(flat.numel()) > int(max_values),
            }
        if isinstance(value, (bool, int, float, str)):
            return value
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return str(value)


def _result_debug_summary(result) -> dict:
    if result is None:
        return {"result_is_none": True}
    summary = {"result_type": type(result).__name__}
    for attr_name in (
        "success",
        "feasible",
        "valid_query",
        "optimized_dt",
        "position_error",
        "rotation_error",
        "cspace_error",
        "seed_cost",
        "solve_time",
        "status",
    ):
        if hasattr(result, attr_name):
            summary[attr_name] = _small_tensor_summary(getattr(result, attr_name))
    return summary


def _cspace_endpoint_and_line_diagnostics(
    *,
    kin_model,
    collision_checker,
    JointState,
    joint_names: list[str],
    q0,
    q_goal,
    tool_frame: str,
) -> dict:
    diagnostics = {}
    try:
        delta = (q_goal - q0).reshape(-1)
        diagnostics.update(
            {
                "start_joint_pos": q0.reshape(-1).detach().cpu().tolist(),
                "goal_joint_pos": q_goal.reshape(-1).detach().cpu().tolist(),
                "start_to_goal_joint_delta": delta.detach().cpu().tolist(),
                "start_to_goal_joint_delta_abs_max": float(torch.max(torch.abs(delta)).item()),
                "start_to_goal_joint_delta_norm": float(torch.linalg.norm(delta).item()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        diagnostics["joint_delta_error"] = str(exc)

    try:
        start_fk = kin_model.compute_kinematics(JointState.from_position(q0, joint_names=joint_names))
        goal_fk = kin_model.compute_kinematics(JointState.from_position(q_goal, joint_names=joint_names))
        start_pose = start_fk.tool_poses.get_link_pose(tool_frame)
        goal_pose = goal_fk.tool_poses.get_link_pose(tool_frame)
        diagnostics.update(
            {
                "start_tool_pos": start_pose.position.reshape(-1).detach().cpu().tolist(),
                "start_tool_quat": start_pose.quaternion.reshape(-1).detach().cpu().tolist(),
                "goal_tool_pos": goal_pose.position.reshape(-1).detach().cpu().tolist(),
                "goal_tool_quat": goal_pose.quaternion.reshape(-1).detach().cpu().tolist(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        diagnostics["fk_error"] = str(exc)

    if collision_checker is not None:
        try:
            alphas = torch.linspace(0.0, 1.0, steps=33, dtype=q0.dtype, device=q0.device).reshape(1, -1, 1)
            q_line = q0.reshape(1, 1, -1) + alphas * (q_goal.reshape(1, 1, -1) - q0.reshape(1, 1, -1))
            scene_dist, self_dist = collision_checker.get_scene_self_collision_distance_from_joint_trajectory(q_line)
            scene_flat = scene_dist.detach().reshape(-1)
            self_flat = self_dist.detach().reshape(-1)
            scene_min_idx = int(torch.argmin(scene_flat).item())
            self_min_idx = int(torch.argmin(self_flat).item())
            diagnostics.update(
                {
                    "linear_seed_min_scene_distance": float(scene_flat[scene_min_idx].item()),
                    "linear_seed_min_scene_distance_waypoint": scene_min_idx,
                    "linear_seed_min_self_distance": float(self_flat[self_min_idx].item()),
                    "linear_seed_min_self_distance_waypoint": self_min_idx,
                }
            )
        except Exception as exc:  # noqa: BLE001
            diagnostics["linear_seed_collision_distance_error"] = str(exc)
    return diagnostics


def _top_self_collision_pairs(collision_checker, q_best, *, top_k: int = 5) -> list[dict]:
    """Return the largest positive self-collision sphere pairs for one joint state."""
    try:
        kin_state = collision_checker.get_kinematics(q_best)
        robot_spheres = kin_state.robot_spheres.detach()
        self_cost = getattr(collision_checker, "self_collision_cost", None)
        if self_cost is None:
            return []
        kin_cfg = self_cost.config.self_collision_kin_config
        collision_pairs = kin_cfg.collision_pairs.detach().to(device=robot_spheres.device, dtype=torch.long)
        if collision_pairs.numel() == 0:
            return []
        spheres = robot_spheres.reshape(-1, robot_spheres.shape[-2], robot_spheres.shape[-1])[0]
        padding = kin_cfg.sphere_padding.detach().to(device=robot_spheres.device, dtype=spheres.dtype)
        pair_spheres = spheres[collision_pairs]
        center_distance = torch.linalg.norm(pair_spheres[:, 0, :3] - pair_spheres[:, 1, :3], dim=-1)
        penetration = (
            pair_spheres[:, 0, 3]
            + pair_spheres[:, 1, 3]
            + padding[collision_pairs[:, 0]]
            + padding[collision_pairs[:, 1]]
            - center_distance
        )
        positive = torch.nonzero(penetration > 0.0, as_tuple=False).reshape(-1)
        if positive.numel() == 0:
            return []
        top_count = min(int(top_k), int(positive.numel()))
        top_values, top_order = torch.topk(penetration[positive], k=top_count)
        pair_indices = positive[top_order]

        link_idx_map = collision_checker.kinematics.kinematics_config.link_sphere_idx_map.detach().to(
            device=robot_spheres.device
        )
        name_by_idx = {
            int(link_idx): link_name
            for link_name, link_idx in collision_checker.kinematics.kinematics_config.link_name_to_idx_map.items()
        }
        diagnostics = []
        for value, pair_idx in zip(top_values.detach().cpu().tolist(), pair_indices.detach().cpu().tolist()):
            s0 = int(collision_pairs[pair_idx, 0].item())
            s1 = int(collision_pairs[pair_idx, 1].item())
            link0_idx = int(link_idx_map[s0].item())
            link1_idx = int(link_idx_map[s1].item())
            diagnostics.append(
                {
                    "link_pair": [name_by_idx.get(link0_idx, str(link0_idx)), name_by_idx.get(link1_idx, str(link1_idx))],
                    "sphere_pair": [s0, s1],
                    "penetration_m": float(value),
                    "center_distance_m": float(center_distance[pair_idx].detach().cpu().item()),
                    "radii_plus_padding_m": float(
                        (
                            pair_spheres[pair_idx, 0, 3]
                            + pair_spheres[pair_idx, 1, 3]
                            + padding[collision_pairs[pair_idx, 0]]
                            + padding[collision_pairs[pair_idx, 1]]
                        )
                        .detach()
                        .cpu()
                        .item()
                    ),
                }
            )
        return diagnostics
    except Exception as exc:  # noqa: BLE001
        return [{"error": str(exc)}]


def _iter_payload_pose_targets(payload: dict):
    for target in payload.get("pose_tests", []) or []:
        yield target
    multi_pose_plan = payload.get("multi_pose_plan") or {}
    for target in multi_pose_plan.get("targets", []) or []:
        yield target


def _default_pose_tolerances(payload: dict) -> tuple[float, float]:
    return (
        float(payload.get("position_tolerance_m", 0.005)),
        float(payload.get("orientation_tolerance_rad", 0.05)),
    )


def _high_z_pose_tolerances(payload: dict) -> tuple[float, float]:
    return (
        float(payload.get("high_z_position_tolerance_m", 0.03)),
        float(payload.get("high_z_orientation_tolerance_rad", 0.2617993877991494)),
    )


def _target_uses_high_z_tolerance(target: dict, payload: dict) -> bool:
    high_z_threshold = float(payload.get("high_z_tolerance_threshold_m", 0.5))
    target_world_z = target.get("target_pos_world_z")
    if target_world_z is not None:
        return float(target_world_z) > high_z_threshold
    target_pos = target.get("target_pos")
    return isinstance(target_pos, (list, tuple)) and len(target_pos) >= 3 and float(target_pos[2]) > high_z_threshold


def _target_pose_tolerances(target: dict | None, payload: dict) -> tuple[float, float, bool]:
    if target is not None and _target_uses_high_z_tolerance(target, payload):
        position_tolerance, orientation_tolerance = _high_z_pose_tolerances(payload)
        return position_tolerance, orientation_tolerance, True
    position_tolerance, orientation_tolerance = _default_pose_tolerances(payload)
    return position_tolerance, orientation_tolerance, False


def _diagnose_pose_ik(
    *,
    planner,
    goal_pose,
    start_state,
    return_seeds: int,
    collision_checker=None,
) -> dict:
    metrics = {}
    try:
        ik_result = planner.ik_solver.solve_pose(
            goal_pose,
            return_seeds=return_seeds,
            current_state=start_state,
        )
        metrics.update(
            {
                "ik_success_count": _count_true(ik_result.success),
                "ik_feasible_count": _count_true(getattr(ik_result, "feasible", None)),
                "ik_min_position_error_m": _min_float_tensor(getattr(ik_result, "position_error", None)),
                "ik_min_rotation_error_rad": _min_float_tensor(getattr(ik_result, "rotation_error", None)),
            }
        )
        if collision_checker is not None and getattr(ik_result, "solution", None) is not None:
            position_error = getattr(ik_result, "position_error", None)
            rotation_error = getattr(ik_result, "rotation_error", None)
            solution = ik_result.solution
            if position_error is not None and rotation_error is not None and solution is not None:
                score = (position_error.detach().reshape(-1) + rotation_error.detach().reshape(-1)).to(
                    device=solution.device
                )
                best_idx = int(torch.argmin(score).item())
                q_flat = solution.detach().reshape(-1, solution.shape[-1])
                q_best = q_flat[best_idx].reshape(1, 1, -1).to(device="cuda", dtype=torch.float32)
                scene_dist, self_dist = collision_checker.get_scene_self_collision_distance_from_joint_trajectory(q_best)
                metrics.update(
                    {
                        "ik_best_scene_collision_cost": float(scene_dist.detach().reshape(-1).max().item()),
                        "ik_best_self_collision_cost": float(self_dist.detach().reshape(-1).max().item()),
                        "ik_best_joint_pos": q_best.detach().reshape(-1).cpu().tolist(),
                        "ik_best_self_collision_pairs": _top_self_collision_pairs(collision_checker, q_best),
                    }
                )
    except Exception as exc:  # noqa: BLE001
        metrics["ik_diagnostic_error"] = str(exc)
    return metrics


def _sample_joint_waypoint_curve(
    waypoints: "torch.Tensor",
    horizon: int,
    *,
    mode: str = "cubic",
) -> "torch.Tensor":
    """Sample a smooth joint-space curve through ordered waypoints."""
    if waypoints.ndim != 2:
        raise ValueError(f"waypoints must be 2D [num_waypoints, dof], got {tuple(waypoints.shape)}")
    if waypoints.shape[0] < 2:
        raise ValueError("At least two waypoints are required to sample a trajectory.")
    if horizon < 2:
        raise ValueError(f"horizon must be >= 2, got {horizon}")

    num_segments = waypoints.shape[0] - 1
    device = waypoints.device
    dtype = waypoints.dtype
    scaled_t = torch.linspace(0.0, float(num_segments), horizon, device=device, dtype=dtype)
    segment_idx = torch.floor(scaled_t).to(dtype=torch.long).clamp(max=num_segments - 1)
    u = (scaled_t - segment_idx.to(dtype=dtype)).reshape(-1, 1)
    if mode == "linear":
        blend = u
    elif mode == "quintic":
        blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    elif mode == "cubic":
        blend = 3.0 * u**2 - 2.0 * u**3
    else:
        raise ValueError(f"Unsupported multi-pose seed_mode={mode!r}; expected linear, cubic, or quintic.")

    start = waypoints[segment_idx]
    goal = waypoints[segment_idx + 1]
    sampled = start + blend * (goal - start)
    sampled[0] = waypoints[0]
    sampled[-1] = waypoints[-1]
    return sampled


def _sample_joint_catmull_rom_curve(
    waypoints: "torch.Tensor",
    *,
    samples_per_segment: int,
) -> tuple["torch.Tensor", list[int]]:
    """Sample one C1-continuous joint curve that passes through every waypoint."""
    if waypoints.ndim != 2:
        raise ValueError(f"waypoints must be 2D [num_waypoints, dof], got {tuple(waypoints.shape)}")
    if waypoints.shape[0] < 2:
        raise ValueError("At least two waypoints are required to sample a trajectory.")
    samples_per_segment = max(2, int(samples_per_segment))
    num_points = int(waypoints.shape[0])
    tangents = torch.zeros_like(waypoints)
    tangents[0] = waypoints[1] - waypoints[0]
    tangents[-1] = waypoints[-1] - waypoints[-2]
    if num_points > 2:
        tangents[1:-1] = 0.5 * (waypoints[2:] - waypoints[:-2])

    segments = []
    segment_lengths = []
    for segment_idx in range(num_points - 1):
        p0 = waypoints[segment_idx]
        p1 = waypoints[segment_idx + 1]
        m0 = tangents[segment_idx]
        m1 = tangents[segment_idx + 1]
        u = torch.linspace(
            0.0,
            1.0,
            samples_per_segment,
            device=waypoints.device,
            dtype=waypoints.dtype,
        ).reshape(-1, 1)
        u2 = u * u
        u3 = u2 * u
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2
        segment = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        segment[0] = p0
        segment[-1] = p1
        if segment_idx > 0:
            segment = segment[1:]
        segments.append(segment)
        segment_lengths.append(int(segment.shape[0]))
    return torch.cat(segments, dim=0), segment_lengths


def _make_goal_tool_pose_from_targets(
    targets: list[dict],
    *,
    tool_frame: str,
    GoalToolPose,
    Pose,
):
    if not targets:
        raise ValueError("multi_pose_plan.targets must contain at least one target pose.")
    target_pos = torch.tensor(
        [target["target_pos"] for target in targets],
        dtype=torch.float32,
        device="cuda",
    )
    target_quat = torch.tensor(
        [target["target_quat"] for target in targets],
        dtype=torch.float32,
        device="cuda",
    )
    goal_pose = GoalToolPose.from_poses(
        {
            tool_frame: Pose(
                position=target_pos,
                quaternion=target_quat,
            )
        },
        num_goalset=1,
    )
    return goal_pose, target_pos, target_quat


def _select_first_successful_ik_solution(ik_result, *, targets: list[dict]) -> tuple["torch.Tensor", list[dict]]:
    success = ik_result.success
    if success.ndim != 2:
        raise ValueError(f"Unexpected IK success shape: {tuple(success.shape)}")
    selected = []
    failures = []
    for target_idx, target in enumerate(targets):
        success_indices = torch.nonzero(success[target_idx], as_tuple=False).reshape(-1)
        if success_indices.numel() == 0:
            failures.append(
                {
                    "index": target_idx,
                    "name": target.get("name", f"target_{target_idx + 1}"),
                    "position_error_m": (
                        float(ik_result.position_error[target_idx].min().item())
                        if ik_result.position_error is not None
                        else None
                    ),
                    "rotation_error_rad": (
                        float(ik_result.rotation_error[target_idx].min().item())
                        if ik_result.rotation_error is not None
                        else None
                    ),
                }
            )
            continue
        selected.append(ik_result.solution[target_idx, int(success_indices[0].item())])
    if failures:
        return torch.empty((0, ik_result.solution.shape[-1]), device=ik_result.solution.device), failures
    return torch.stack(selected, dim=0), failures


def _solve_ordered_target_ik(
    *,
    planner,
    planner_selector=None,
    targets: list[dict],
    target_pos: "torch.Tensor",
    target_quat: "torch.Tensor",
    tool_frame: str,
    q_current: list[float],
    joint_names: list[str],
    GoalToolPose,
    JointState,
    Pose,
) -> tuple["torch.Tensor", list[dict]]:
    """Solve IK for ordered targets one-by-one.

    This avoids a cuRobo batch-IK buffer shape issue observed when all targets
    share the same current state, while preserving the later whole-trajectory
    optimization path.

    Uses multiple IK seeds and selects the one closest in joint-space to the
    previous solution (or q_current for the first target), avoiding joint-angle
    wrap-around (e.g. joint4 flipping between +π and -π).
    """
    q_seed = torch.tensor([q_current], dtype=torch.float32, device="cuda")
    selected = []
    failures = []
    ik_return_seeds = 10
    for target_idx, target in enumerate(targets):
        target_planner = planner
        tolerance_info = {}
        if planner_selector is not None:
            target_planner, tolerance_info = planner_selector(target)
        goal_pose = GoalToolPose.from_poses(
            {
                tool_frame: Pose(
                    position=target_pos[target_idx : target_idx + 1],
                    quaternion=target_quat[target_idx : target_idx + 1],
                )
            },
            num_goalset=1,
        )
        current_state = JointState.from_position(q_seed, joint_names=joint_names)
        ik_result = target_planner.ik_solver.solve_pose(
            goal_pose,
            return_seeds=ik_return_seeds,
            current_state=current_state,
        )
        if not bool(ik_result.success.any().item()):
            failures.append(
                {
                    "index": target_idx,
                    "name": target.get("name", f"target_{target_idx + 1}"),
                    "position_error_m": (
                        float(ik_result.position_error.reshape(-1).min().item())
                        if ik_result.position_error is not None
                        else None
                    ),
                    "rotation_error_rad": (
                        float(ik_result.rotation_error.reshape(-1).min().item())
                        if ik_result.rotation_error is not None
                        else None
                    ),
                    **tolerance_info,
                }
            )
            continue

        # Pick the successful seed closest in joint-space to q_seed (current config),
        # rather than blindly trusting the IK solver's internal ranking which may
        # prefer a joint4=-π solution over the joint4=+π one near q_seed.
        solutions = ik_result.solution  # [batch=1, num_seeds, dof]
        success_mask = ik_result.success.reshape(-1)  # [num_seeds]
        if solutions.dim() == 3 and solutions.shape[1] > 1:
            sols = solutions[0]  # [num_seeds, dof]
            valid_indices = [i for i in range(sols.shape[0]) if success_mask[i].item()]
            if valid_indices:
                sols_valid = sols[valid_indices]  # [num_valid, dof]
                # L2 distance in joint space from q_seed (previous solution)
                delta = sols_valid - q_seed.reshape(1, -1)
                dist = (delta * delta).sum(dim=1)
                best_local = int(dist.argmin().item())
                best_idx = valid_indices[best_local]
                q_seed = sols[best_idx].reshape(1, -1).detach()
            else:
                q_seed = ik_result.solution[0, 0].reshape(1, -1).detach()
        else:
            q_seed = ik_result.solution[0, 0].reshape(1, -1).detach()
        selected.append(q_seed.reshape(-1))
    if failures:
        return torch.empty((0, q_seed.shape[-1]), device=q_seed.device), failures
    return torch.stack(selected, dim=0), failures


def _measure_waypoint_pose_errors(
    *,
    kin_model,
    JointState,
    joint_names: list[str],
    tool_frame: str,
    q_traj: "torch.Tensor",
    target_pos: "torch.Tensor",
    target_quat: "torch.Tensor",
    targets: list[dict],
) -> list[dict]:
    fk_state = kin_model.compute_kinematics(
        JointState.from_position(q_traj, joint_names=joint_names)
    )
    traj_pose = fk_state.tool_poses.get_link_pose(tool_frame)
    traj_pos = traj_pose.position.reshape(-1, 3)
    traj_quat = traj_pose.quaternion.reshape(-1, 4)
    errors = []
    for target_idx, target in enumerate(targets):
        pos_delta = torch.linalg.norm(traj_pos - target_pos[target_idx].reshape(1, 3), dim=-1)
        best_idx = int(torch.argmin(pos_delta).item())
        pos_error, rot_error_rad, rot_error_deg = _pose_error(
            target_pos[target_idx],
            target_quat[target_idx],
            traj_pos[best_idx],
            traj_quat[best_idx],
        )
        errors.append(
            {
                "index": target_idx,
                "name": target.get("name", f"target_{target_idx + 1}"),
                "nearest_waypoint": best_idx,
                "pos_error_m": float(pos_error.item()),
                "rot_error_rad": float(rot_error_rad.item()),
                "rot_error_deg": float(rot_error_deg.item()),
            }
        )
    return errors


def _extract_top_interpolated_trajectory(result):
    result = result.get_topk_seeds(1)
    if result.interpolated_trajectory is None:
        raise ValueError("trajopt result does not contain an interpolated trajectory")
    interpolated_pos = result.interpolated_trajectory.position
    if interpolated_pos.dim() == 4:
        traj_tensor = interpolated_pos[0, 0]
    elif interpolated_pos.dim() == 3:
        traj_tensor = interpolated_pos[0]
    else:
        raise ValueError(
            "unexpected interpolated trajectory shape: "
            f"{tuple(interpolated_pos.shape)}"
        )
    if result.interpolated_last_tstep is not None:
        end_idx = int(result.interpolated_last_tstep.reshape(-1)[0].item())
        if end_idx > 0:
            traj_tensor = traj_tensor[:end_idx]
    return traj_tensor.detach().cpu()


def _plan_multi_pose_polynomial_seed(
    *,
    multi_plan: dict,
    planner,
    planner_selector=None,
    kin_model,
    collision_checker,
    JointState,
    GoalToolPose,
    Pose,
    tool_frame: str,
    joint_names: list[str],
    q_current: list[float],
    num_trajopt_seeds: int,
) -> dict:
    name = str(multi_plan.get("name", "multi_pose_polynomial_seed"))
    seed_mode = str(multi_plan.get("seed_mode", "cubic")).lower()
    targets = list(multi_plan.get("targets") or [])
    t0 = time.perf_counter()
    metrics = {
        "seed_mode": seed_mode,
        "target_count": len(targets),
        "num_trajopt_seeds": num_trajopt_seeds,
    }
    try:
        goal_pose, target_pos, target_quat = _make_goal_tool_pose_from_targets(
            targets,
            tool_frame=tool_frame,
            GoalToolPose=GoalToolPose,
            Pose=Pose,
        )
        q0 = torch.tensor([q_current], dtype=torch.float32, device="cuda")
        _log(f"multi-pose ordered IK name={name} targets={len(targets)}")
        q_targets, ik_failures = _solve_ordered_target_ik(
            planner=planner,
            planner_selector=planner_selector,
            targets=targets,
            target_pos=target_pos,
            target_quat=target_quat,
            tool_frame=tool_frame,
            q_current=q_current,
            joint_names=joint_names,
            GoalToolPose=GoalToolPose,
            JointState=JointState,
            Pose=Pose,
        )
        metrics["ik_mode"] = "sequential"
        metrics["ik_success_count"] = int(len(targets) - len(ik_failures))
        metrics["ik_failures"] = ik_failures
        if ik_failures:
            elapsed_s = time.perf_counter() - t0
            return _format_result(name, False, elapsed_s, [], metrics)

        q_waypoints = torch.cat([q0, q_targets], dim=0)
        action_horizon = int(planner.trajopt_solver.action_horizon)
        _log(
            f"multi-pose continuous joint curve name={name} "
            f"segments={q_waypoints.shape[0] - 1} samples_per_segment={action_horizon}"
        )
        traj_tensor, segment_lengths = _sample_joint_catmull_rom_curve(
            q_waypoints,
            samples_per_segment=action_horizon,
        )
        success = True
        if success:
            q_traj_cuda = traj_tensor.to(device="cuda", dtype=torch.float32)
            final_q = q_traj_cuda[-1:].contiguous()
            fk_state = kin_model.compute_kinematics(
                JointState.from_position(final_q, joint_names=joint_names)
            )
            final_pose = fk_state.tool_poses.get_link_pose(tool_frame)
            final_pos_error, final_rot_error_rad, final_rot_error_deg = _pose_error(
                target_pos[-1],
                target_quat[-1],
                final_pose.position,
                final_pose.quaternion,
            )
            metrics.update(
                {
                    "planner_final_pos_error_m": float(final_pos_error.item()),
                    "planner_final_rot_error_rad": float(final_rot_error_rad.item()),
                    "planner_final_rot_error_deg": float(final_rot_error_deg.item()),
                    "planner_mode": "continuous_ik_catmull_rom",
                    "samples_per_segment": action_horizon,
                    "segment_lengths": segment_lengths,
                    "waypoint_pose_errors": _measure_waypoint_pose_errors(
                        kin_model=kin_model,
                        JointState=JointState,
                        joint_names=joint_names,
                        tool_frame=tool_frame,
                        q_traj=q_traj_cuda,
                        target_pos=target_pos,
                        target_quat=target_quat,
                        targets=targets,
                    ),
                    "ik_joint_waypoints": q_targets.detach().cpu().tolist(),
                }
            )
            if collision_checker is not None:
                try:
                    q_traj = q_traj_cuda.reshape(1, -1, q_traj_cuda.shape[-1])
                    scene_dist, self_dist = collision_checker.get_scene_self_collision_distance_from_joint_trajectory(
                        q_traj
                    )
                    scene_dist_flat = scene_dist.detach().reshape(-1)
                    self_dist_flat = self_dist.detach().reshape(-1)
                    scene_min_idx = int(torch.argmin(scene_dist_flat).item())
                    self_min_idx = int(torch.argmin(self_dist_flat).item())
                    metrics.update(
                        {
                            "planner_min_scene_distance": float(scene_dist_flat[scene_min_idx].item()),
                            "planner_min_scene_distance_waypoint": scene_min_idx,
                            "planner_min_self_distance": float(self_dist_flat[self_min_idx].item()),
                            "planner_min_self_distance_waypoint": self_min_idx,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    metrics["planner_collision_distance_error"] = str(exc)
            traj = traj_tensor.tolist()
        else:
            traj = []
    except Exception as exc:  # noqa: BLE001
        success = False
        traj = []
        metrics["error"] = str(exc)
        metrics["traceback"] = traceback.format_exc()
    elapsed_s = time.perf_counter() - t0
    return _format_result(name, success, elapsed_s, traj, metrics)


def _plan_cspace_target(
    *,
    target: dict,
    planner,
    kin_model,
    collision_checker,
    JointState,
    joint_names: list[str],
    q_current: list[float],
    tool_frame: str,
) -> dict:
    name = str(target.get("name", "cspace_target"))
    q_goal_values = target.get("joint_pos")
    if not isinstance(q_goal_values, (list, tuple)) or len(q_goal_values) != len(joint_names):
        raise ValueError(f"cspace target {name!r} requires joint_pos length {len(joint_names)}.")
    t0 = time.perf_counter()
    metrics = {"planner_mode": "cspace", "target_joint_pos": [float(v) for v in q_goal_values]}
    try:
        q0 = torch.tensor([q_current], dtype=torch.float32, device="cuda")
        q_goal = torch.tensor([q_goal_values], dtype=torch.float32, device="cuda")
        current_state = JointState.from_position(q0, joint_names=joint_names)
        goal_state = JointState.from_position(q_goal, joint_names=joint_names)
        _log(f"planning cspace target name={name}")
        result = planner.trajopt_solver.solve_cspace(goal_state, current_state, return_seeds=1)
        success = result is not None and bool(result.success.any().item())
        metrics.update(
            _cspace_endpoint_and_line_diagnostics(
                kin_model=kin_model,
                collision_checker=collision_checker,
                JointState=JointState,
                joint_names=joint_names,
                q0=q0,
                q_goal=q_goal,
                tool_frame=tool_frame,
            )
        )
        metrics["cspace_result_debug"] = _result_debug_summary(result)
        if success:
            traj_tensor = _extract_top_interpolated_trajectory(result)
            q_traj_cuda = traj_tensor.to(device="cuda", dtype=torch.float32)
            final_q = q_traj_cuda[-1:].contiguous()
            joint_delta = torch.abs(final_q - q_goal).reshape(-1)
            metrics.update(
                {
                    "planner_final_joint_err_max": float(torch.max(joint_delta).item()),
                    "planner_final_joint_err_norm": float(torch.linalg.norm(joint_delta).item()),
                }
            )
            fk_state = kin_model.compute_kinematics(
                JointState.from_position(final_q, joint_names=joint_names)
            )
            final_pose = fk_state.tool_poses.get_link_pose(tool_frame)
            metrics.update(
                {
                    "planner_final_tool_pos": final_pose.position.reshape(-1).detach().cpu().tolist(),
                    "planner_final_tool_quat": final_pose.quaternion.reshape(-1).detach().cpu().tolist(),
                }
            )
            if collision_checker is not None:
                try:
                    q_traj = q_traj_cuda.reshape(1, -1, q_traj_cuda.shape[-1])
                    scene_dist, self_dist = collision_checker.get_scene_self_collision_distance_from_joint_trajectory(
                        q_traj
                    )
                    scene_dist_flat = scene_dist.detach().reshape(-1)
                    self_dist_flat = self_dist.detach().reshape(-1)
                    scene_min_idx = int(torch.argmin(scene_dist_flat).item())
                    self_min_idx = int(torch.argmin(self_dist_flat).item())
                    metrics.update(
                        {
                            "planner_min_scene_distance": float(scene_dist_flat[scene_min_idx].item()),
                            "planner_min_scene_distance_waypoint": scene_min_idx,
                            "planner_min_self_distance": float(self_dist_flat[self_min_idx].item()),
                            "planner_min_self_distance_waypoint": self_min_idx,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    metrics["planner_collision_distance_error"] = str(exc)
            traj = traj_tensor.tolist()
        else:
            traj = []
            metrics["planner_failure_stage"] = "cspace_trajopt"
    except Exception as exc:  # noqa: BLE001
        success = False
        traj = []
        metrics["error"] = str(exc)
        metrics["traceback"] = traceback.format_exc()
    elapsed_s = time.perf_counter() - t0
    return _format_result(name, success, elapsed_s, traj, metrics)


def _plan_cspace_to_joint_pos(
    *,
    name: str,
    q_goal_values: list[float],
    planner,
    kin_model,
    collision_checker,
    JointState,
    joint_names: list[str],
    q_current: list[float],
    tool_frame: str,
) -> dict:
    return _plan_cspace_target(
        target={"name": name, "joint_pos": q_goal_values},
        planner=planner,
        kin_model=kin_model,
        collision_checker=collision_checker,
        JointState=JointState,
        joint_names=joint_names,
        q_current=q_current,
        tool_frame=tool_frame,
    )


def _import_curobo_modules():
    _ensure_local_nvidia_curobo_on_path()
    # The distribution is named nvidia-curobo, but the canonical Python package
    # namespace is still curobo. Mixing nvidia_curobo.* and curobo.* creates
    # duplicate class objects and breaks isinstance checks inside cuRobo.
    from curobo._src.types.robot import RobotCfg
    from curobo._src.util import logging as curobo_logging
    from curobo._src.geom.types import Cuboid, Mesh, SceneCfg
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState, Pose

    return (
        RobotCfg,
        curobo_logging,
        Cuboid,
        Mesh,
        SceneCfg,
        RobotCollisionChecker,
        RobotCollisionCheckerCfg,
        Kinematics,
        KinematicsCfg,
        MotionPlanner,
        MotionPlannerCfg,
        GoalToolPose,
        JointState,
        Pose,
    )


# ---- 常驻 serve 模式（--serve）：进程启动一次，warmup 一次，循环处理多个 payload。
# 同一场景（障碍物/参数/机器人配置）下复用已 warmup 的 MotionPlanner，省掉每次
# 冷启动里 MotionPlanner 创建 + warmup（实测约 10s）的开销。
_SERVE_RUNTIME_CACHE: dict = {}


def _runtime_scene_key(payload: dict) -> str:
    import hashlib
    import json as _json

    data = {
        "urdf": payload.get("urdf_path"),
        "robot_cfg": payload.get("robot_cfg_path"),
        "base": payload.get("base_link"),
        "tool": payload.get("tool_frame"),
        "joints": payload.get("joint_names"),
        "mesh": payload.get("scene_mesh_obstacles"),
        "cuboid": payload.get("scene_cuboid_obstacles"),
        "cad": payload.get("collision_activation_distance"),
        "ik": payload.get("num_ik_seeds"),
        "tj": payload.get("num_trajopt_seeds"),
    }
    return hashlib.md5(_json.dumps(data, sort_keys=True).encode()).hexdigest()


def _build_runtime(payload: dict) -> dict:
    """Build once-reusable cuRobo runtime (FK/robot_cfg/scene/planner/warmup).

    The expensive MotionPlanner creation + warmup happens lazily inside
    get_planner() on first use and is then cached, so in serve mode a full
    warmup is done only once per scene.
    """
    _log("importing cuRobo modules")
    (
        RobotCfg,
        curobo_logging,
        Cuboid,
        Mesh,
        SceneCfg,
        RobotCollisionChecker,
        RobotCollisionCheckerCfg,
        Kinematics,
        KinematicsCfg,
        MotionPlanner,
        MotionPlannerCfg,
        GoalToolPose,
        JointState,
        Pose,
    ) = _import_curobo_modules()
    _log("cuRobo modules imported")
    curobo_logging.setup_logger("warning")

    urdf_path = payload["urdf_path"]
    robot_cfg_path = payload.get("robot_cfg_path")
    base_link = payload["base_link"]
    tool_frame = payload["tool_frame"]
    joint_names = payload["joint_names"]

    _log("creating FK kinematics")
    kin_cfg = KinematicsCfg.from_basic_urdf(
        urdf_path=urdf_path,
        base_link=base_link,
        tool_frames=[tool_frame],
    )
    kin_model = Kinematics(kin_cfg)
    _log("FK kinematics created")

    if robot_cfg_path:
        _log(f"loading robot cfg: {robot_cfg_path}")
        with Path(robot_cfg_path).expanduser().open("r", encoding="utf-8") as stream:
            robot_cfg_data = yaml.safe_load(stream)
        robot_cfg = RobotCfg.create(robot_cfg_data)
        robot_kin = robot_cfg_data.get("robot_cfg", robot_cfg_data).get("kinematics", {})
        collision_spheres = robot_kin.get("collision_spheres") or {}
        collision_count = sum(len(spheres) for spheres in collision_spheres.values())
        print(
            json.dumps(
                {
                    "worker_info": {
                        "robot_cfg_path": str(Path(robot_cfg_path).expanduser()),
                        "collision_links": list(collision_spheres.keys()),
                        "collision_sphere_count": collision_count,
                    }
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        _log("robot cfg loaded")
    else:
        _log("creating robot cfg from basic URDF")
        robot_cfg = RobotCfg.from_basic(
            urdf_path=urdf_path,
            base_link=base_link,
            tool_frames=[tool_frame],
        )
        _log("basic robot cfg created")

    scene_mesh_obstacles = payload.get("scene_mesh_obstacles", [])
    scene_cuboid_obstacles = payload.get("scene_cuboid_obstacles", [])
    scene_model = None
    if scene_mesh_obstacles or scene_cuboid_obstacles:
        _log(
            f"creating scene model with {len(scene_mesh_obstacles)} mesh obstacles "
            f"and {len(scene_cuboid_obstacles)} cuboid obstacles"
        )
        scene_model = SceneCfg(
            mesh=[
                Mesh(
                    name=obstacle["name"],
                    file_path=obstacle["file_path"],
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                )
                for obstacle in scene_mesh_obstacles
            ],
            cuboid=[
                Cuboid(
                    name=obstacle["name"],
                    dims=obstacle["dims"],
                    pose=obstacle["pose"],
                )
                for obstacle in scene_cuboid_obstacles
            ],
        )
        _log("scene model created")
    else:
        _log("scene model disabled: no mesh or cuboid obstacles")

    collision_activation_distance = float(payload.get("collision_activation_distance", 0.02))
    num_ik_seeds = int(payload.get("num_ik_seeds", 64))
    num_trajopt_seeds = int(payload.get("num_trajopt_seeds", 8))
    default_position_tolerance, default_orientation_tolerance = _default_pose_tolerances(payload)
    high_z_position_tolerance, high_z_orientation_tolerance = _high_z_pose_tolerances(payload)
    high_z_threshold = float(payload.get("high_z_tolerance_threshold_m", 0.5))
    multi_pose_targets = (payload.get("multi_pose_plan") or {}).get("targets") or []
    max_batch_size = int(
        payload.get(
            "max_batch_size",
            max(1, len(multi_pose_targets), len(payload.get("pose_tests", []))),
        )
    )
    _log(
        "planner safety params "
        f"collision_activation_distance={collision_activation_distance:.6f} "
        f"num_ik_seeds={num_ik_seeds} num_trajopt_seeds={num_trajopt_seeds} "
        f"max_batch_size={max_batch_size} "
        f"default_position_tolerance={default_position_tolerance:.6f} "
        f"default_orientation_tolerance={default_orientation_tolerance:.6f} "
        f"high_z_position_tolerance={high_z_position_tolerance:.6f} "
        f"high_z_orientation_tolerance={high_z_orientation_tolerance:.6f} "
        f"high_z_threshold={high_z_threshold:.6f}"
    )

    planner_cache: dict[str, MotionPlanner] = {}

    def get_planner(*, use_high_z_tolerance: bool) -> MotionPlanner:
        planner_key = "high_z" if use_high_z_tolerance else "default"
        cached = planner_cache.get(planner_key)
        if cached is not None:
            return cached
        position_tolerance = high_z_position_tolerance if use_high_z_tolerance else default_position_tolerance
        orientation_tolerance = high_z_orientation_tolerance if use_high_z_tolerance else default_orientation_tolerance
        _log(
            f"creating MotionPlannerCfg planner={planner_key} "
            f"position_tolerance={position_tolerance:.6f} orientation_tolerance={orientation_tolerance:.6f}"
        )
        planner_cfg = MotionPlannerCfg.create(
            robot=robot_cfg,
            scene_model=scene_model,
            collision_cache={
                "mesh": max(1, len(scene_mesh_obstacles) + 2),
                "primitive": max(1, len(scene_cuboid_obstacles) + 2),
            },
            self_collision_check=True,
            num_ik_seeds=num_ik_seeds,
            num_trajopt_seeds=num_trajopt_seeds,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            optimizer_collision_activation_distance=collision_activation_distance,
            use_cuda_graph=False,
            max_batch_size=max_batch_size,
        )
        _log(f"creating MotionPlanner planner={planner_key}")
        planner = MotionPlanner(planner_cfg)
        _log(f"warming up MotionPlanner planner={planner_key}")
        planner.warmup(enable_graph=False, num_warmup_iterations=1)
        _log(f"MotionPlanner warmup complete planner={planner_key}")
        planner_cache[planner_key] = planner
        return planner

    collision_checker = None
    if scene_model is not None:
        _log("creating collision-distance checker")
        collision_checker_cfg = RobotCollisionCheckerCfg.load_from_config(
            robot_config=robot_cfg,
            scene_model=scene_model,
            n_meshes=max(1, len(scene_mesh_obstacles) + 2),
            n_cuboids=max(1, len(scene_cuboid_obstacles) + 2),
            collision_activation_distance=collision_activation_distance,
            max_collision_distance=max(1.0, collision_activation_distance * 10.0),
        )
        collision_checker = RobotCollisionChecker(collision_checker_cfg)
        _log("collision-distance checker created")

    return {
        "kin_model": kin_model,
        "collision_checker": collision_checker,
        "get_planner": get_planner,
        "JointState": JointState,
        "GoalToolPose": GoalToolPose,
        "Pose": Pose,
        "tool_frame": tool_frame,
        "joint_names": joint_names,
        "num_trajopt_seeds": num_trajopt_seeds,
    }


def _run_payload(payload: dict, rt: dict) -> list:
    """Run one payload using a previously built runtime, return results list."""
    kin_model = rt["kin_model"]
    collision_checker = rt["collision_checker"]
    get_planner = rt["get_planner"]
    JointState = rt["JointState"]
    GoalToolPose = rt["GoalToolPose"]
    Pose = rt["Pose"]
    tool_frame = rt["tool_frame"]
    joint_names = rt["joint_names"]
    num_trajopt_seeds = rt["num_trajopt_seeds"]

    def select_planner_for_target(target: dict):
        use_high_z_tolerance = _target_uses_high_z_tolerance(target, payload)
        position_tolerance, orientation_tolerance, _ = _target_pose_tolerances(target, payload)
        return (
            get_planner(use_high_z_tolerance=use_high_z_tolerance),
            {
                "target_high_z_tolerance_active": bool(use_high_z_tolerance),
                "target_position_tolerance_m": float(position_tolerance),
                "target_orientation_tolerance_rad": float(orientation_tolerance),
            },
        )

    q_current = payload["q_current"]
    start_state = JointState.from_position(
        torch.tensor([q_current], dtype=torch.float32, device="cuda"),
        joint_names=joint_names,
    )

    results = []
    multi_pose_targets = (payload.get("multi_pose_plan") or {}).get("targets") or []
    if payload.get("multi_pose_plan") is not None:
        multi_pose_use_high_z = any(_target_uses_high_z_tolerance(target, payload) for target in multi_pose_targets)
        multi_result = _plan_multi_pose_polynomial_seed(
            multi_plan=payload["multi_pose_plan"],
            planner=get_planner(use_high_z_tolerance=multi_pose_use_high_z),
            planner_selector=select_planner_for_target,
            kin_model=kin_model,
            collision_checker=collision_checker,
            JointState=JointState,
            GoalToolPose=GoalToolPose,
            Pose=Pose,
            tool_frame=tool_frame,
            joint_names=joint_names,
            q_current=q_current,
            num_trajopt_seeds=num_trajopt_seeds,
        )
        results.append(multi_result)

    rolling_q_current = list(q_current)
    for target in payload.get("cspace_targets", []):
        cspace_result = _plan_cspace_target(
            target=target,
            planner=get_planner(use_high_z_tolerance=False),
            kin_model=kin_model,
            collision_checker=collision_checker,
            JointState=JointState,
            joint_names=joint_names,
            q_current=rolling_q_current,
            tool_frame=tool_frame,
        )
        results.append(cspace_result)
        if cspace_result.get("success") and cspace_result.get("trajectory"):
            rolling_q_current = list(cspace_result["trajectory"][-1])
        else:
            break

    for case in payload.get("pose_tests", []):
        _log(f"planning case={case['name']}")
        planner, tolerance_info = select_planner_for_target(case)
        target_pos = torch.tensor([case["target_pos"]], dtype=torch.float32, device="cuda")
        target_quat = torch.tensor([case["target_quat"]], dtype=torch.float32, device="cuda")
        goal_pose = GoalToolPose.from_poses(
            {
                tool_frame: Pose(
                    position=target_pos,
                    quaternion=target_quat,
                )
            },
            num_goalset=1,
        )
        t0 = time.perf_counter()
        try:
            result = planner.plan_pose(goal_pose, start_state)
            success = result is not None and bool(result.success.any().item())
            if success:
                traj_tensor = result.get_interpolated_plan().position.squeeze(0).detach().cpu()
                if traj_tensor.dim() > 2:
                    traj_tensor = traj_tensor.reshape(-1, traj_tensor.shape[-1])
                final_q = traj_tensor[-1].to(device="cuda", dtype=torch.float32).reshape(1, -1)
                fk_state = kin_model.compute_kinematics(
                    JointState.from_position(final_q, joint_names=joint_names)
                )
                final_pose = fk_state.tool_poses.get_link_pose(tool_frame)
                pos_error, rot_error_rad, rot_error_deg = _pose_error(
                    target_pos,
                    target_quat,
                    final_pose.position,
                    final_pose.quaternion,
                )
                metrics = {
                    "planner_final_pos_error_m": float(pos_error.item()),
                    "planner_final_rot_error_rad": float(rot_error_rad.item()),
                    "planner_final_rot_error_deg": float(rot_error_deg.item()),
                    **tolerance_info,
                }
                if collision_checker is not None:
                    try:
                        q_traj = traj_tensor.to(device="cuda", dtype=torch.float32).reshape(
                            1, -1, traj_tensor.shape[-1]
                        )
                        scene_dist, self_dist = collision_checker.get_scene_self_collision_distance_from_joint_trajectory(
                            q_traj
                        )
                        scene_dist_flat = scene_dist.detach().reshape(-1)
                        self_dist_flat = self_dist.detach().reshape(-1)
                        scene_min_idx = int(torch.argmin(scene_dist_flat).item())
                        self_min_idx = int(torch.argmin(self_dist_flat).item())
                        metrics.update(
                            {
                                "planner_min_scene_distance": float(scene_dist_flat[scene_min_idx].item()),
                                "planner_min_scene_distance_waypoint": scene_min_idx,
                                "planner_min_self_distance": float(self_dist_flat[self_min_idx].item()),
                                "planner_min_self_distance_waypoint": self_min_idx,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        metrics["planner_collision_distance_error"] = str(exc)
                traj = traj_tensor.tolist()
            else:
                traj = []
                metrics = _diagnose_pose_ik(
                    planner=planner,
                    goal_pose=goal_pose,
                    start_state=start_state,
                    return_seeds=num_trajopt_seeds,
                    collision_checker=collision_checker,
                )
                metrics.update(tolerance_info)
                if result is None:
                    metrics["planner_failure_stage"] = "ik"
                else:
                    metrics.update(
                        {
                            "planner_failure_stage": "trajopt",
                            "trajopt_success_count": _count_true(getattr(result, "success", None)),
                            "trajopt_feasible_count": _count_true(getattr(result, "feasible", None)),
                            "trajopt_min_position_error_m": _min_float_tensor(
                                getattr(result, "position_error", None)
                            ),
                            "trajopt_min_rotation_error_rad": _min_float_tensor(
                                getattr(result, "rotation_error", None)
                            ),
                            "trajopt_min_seed_cost": _min_float_tensor(getattr(result, "seed_cost", None)),
                        }
                    )
                fallback_joint_pos = metrics.get("ik_best_joint_pos")
                if fallback_joint_pos is not None:
                    _log(f"pose planner failed; trying cspace fallback case={case['name']}")
                    fallback_result = _plan_cspace_to_joint_pos(
                        name=f"{case['name']}:ik_best_cspace_fallback",
                        q_goal_values=[float(value) for value in fallback_joint_pos],
                        planner=planner,
                        kin_model=kin_model,
                        collision_checker=collision_checker,
                        JointState=JointState,
                        joint_names=joint_names,
                        q_current=q_current,
                        tool_frame=tool_frame,
                    )
                    metrics["cspace_fallback"] = {
                        "success": bool(fallback_result.get("success")),
                        "elapsed_s": float(fallback_result.get("elapsed_s", 0.0)),
                        "metrics": fallback_result.get("metrics", {}),
                    }
                    if fallback_result.get("success") and fallback_result.get("trajectory"):
                        success = True
                        traj = fallback_result["trajectory"]
                        metrics["planner_failure_stage"] = "pose_plan_cspace_fallback"
                        metrics["fallback_used"] = True
        except Exception as exc:  # noqa: BLE001
            success = False
            traj = []
            metrics = {"error": str(exc)}
            case = dict(case)
            case["error"] = str(exc)
        elapsed_s = time.perf_counter() - t0
        results.append(_format_result(case["name"], success, elapsed_s, traj, metrics))
        _log(f"planned case={case['name']} success={success} elapsed_s={elapsed_s:.6f}")

    return results


def _serve_main() -> None:
    """Serve mode: read one payload file path per stdin line, reuse warmup runtime."""
    _log("serve mode: waiting for payload paths on stdin")
    runtime_cache: dict[str, dict] = {}
    for line in sys.stdin:
        payload_path = line.strip()
        if not payload_path:
            continue
        try:
            payload = _load_payload(payload_path)
            scene_key = _runtime_scene_key(payload)
            rt = runtime_cache.get(scene_key)
            if rt is None:
                _log(f"building runtime for scene_key={scene_key[:12]}...")
                rt = _build_runtime(payload)
                runtime_cache.clear()
                runtime_cache[scene_key] = rt
                _log("runtime built (warmup done once)")
            else:
                _log(f"reusing runtime scene_key={scene_key[:12]}")
            results = _run_payload(payload, rt)
            print(json.dumps({"results": results}, separators=(",", ":")), flush=True)
        except Exception as exc:  # noqa: BLE001
            _log(f"serve payload error: {exc}")
            traceback.print_exc(file=sys.stderr)
            print(
                json.dumps({"results": [{"name": "error", "success": False, "error": str(exc)}]}),
                flush=True,
            )


def main() -> None:
    if "--serve" in sys.argv:
        _serve_main()
        return
    if len(sys.argv) != 2:
        raise SystemExit("Usage: run_curobo_pose_plan_test_worker.py <payload.json> | --serve")

    payload = _load_payload(sys.argv[1])
    _log("payload loaded")
    rt = _build_runtime(payload)
    results = _run_payload(payload, rt)
    _log("writing final JSON result")
    print(json.dumps({"results": results}, separators=(",", ":")))
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        _log(f"fatal error: {exc}")
        traceback.print_exc(file=sys.stderr)
        raise
