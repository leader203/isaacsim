from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import torch


@dataclass(frozen=True)
class PlannerClientConfig:
    scene_cuboid_obstacles: list[dict]
    collision_activation_distance: float
    position_tolerance_m: float
    orientation_tolerance_rad: float
    num_ik_seeds: int
    num_trajopt_seeds: int
    random_seed: int
    high_z_tolerance_threshold_m: float
    high_z_position_tolerance_m: float
    high_z_orientation_tolerance_rad: float


def planner_client_config_from_args(args_cli, *, scene_cuboid_obstacles: list[dict]) -> PlannerClientConfig:
    return PlannerClientConfig(
        scene_cuboid_obstacles=scene_cuboid_obstacles,
        collision_activation_distance=float(args_cli.curobo_collision_activation_distance),
        position_tolerance_m=float(args_cli.curobo_position_tolerance_m),
        orientation_tolerance_rad=math.radians(args_cli.curobo_orientation_tolerance_deg),
        num_ik_seeds=int(args_cli.curobo_num_ik_seeds),
        num_trajopt_seeds=int(args_cli.curobo_num_trajopt_seeds),
        random_seed=int(getattr(args_cli, "curobo_random_seed", 123)),
        high_z_tolerance_threshold_m=float(args_cli.curobo_high_z_tolerance_threshold_m),
        high_z_position_tolerance_m=float(args_cli.curobo_high_z_position_tolerance_m),
        high_z_orientation_tolerance_rad=math.radians(args_cli.curobo_high_z_orientation_tolerance_deg),
    )


def base_worker_payload(
    *,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    q_start: torch.Tensor,
    scene_mesh_obstacles: list[dict],
    cfg: PlannerClientConfig,
) -> dict:
    return {
        "urdf_path": str(urdf_path),
        "robot_cfg_path": str(robot_cfg_path),
        "base_link": base_link,
        "tool_frame": tool_frame,
        "joint_names": joint_names,
        "q_current": q_start.detach().cpu().tolist(),
        "scene_mesh_obstacles": scene_mesh_obstacles,
        "scene_cuboid_obstacles": cfg.scene_cuboid_obstacles,
        "collision_activation_distance": cfg.collision_activation_distance,
        "position_tolerance_m": cfg.position_tolerance_m,
        "orientation_tolerance_rad": cfg.orientation_tolerance_rad,
        "num_ik_seeds": cfg.num_ik_seeds,
        "num_trajopt_seeds": cfg.num_trajopt_seeds,
        "random_seed": cfg.random_seed,
    }


def plan_single_pose(
    *,
    case_name: str,
    phase_name: str,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    q_start: torch.Tensor,
    scene_mesh_obstacles: list[dict],
    target_pos_base: torch.Tensor,
    target_quat_base: torch.Tensor,
    target_pos_world_z: float | None,
    env,
    robot,
    hold_gripper_joint1_pos: float | None,
    cfg: PlannerClientConfig,
    run_pose_plan_worker_fn,
    write_worker_payload_debug_fn,
) -> dict:
    pose_target = {
        "name": f"{case_name}:{phase_name}",
        "target_pos": target_pos_base.reshape(-1).detach().cpu().tolist(),
        "target_quat": target_quat_base.reshape(-1).detach().cpu().tolist(),
    }
    if target_pos_world_z is not None:
        pose_target["target_pos_world_z"] = float(target_pos_world_z)
    worker_payload = base_worker_payload(
        urdf_path=urdf_path,
        robot_cfg_path=robot_cfg_path,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
        q_start=q_start,
        scene_mesh_obstacles=scene_mesh_obstacles,
        cfg=cfg,
    )
    worker_payload.update(
        {
            "high_z_tolerance_threshold_m": cfg.high_z_tolerance_threshold_m,
            "high_z_position_tolerance_m": cfg.high_z_position_tolerance_m,
            "high_z_orientation_tolerance_rad": cfg.high_z_orientation_tolerance_rad,
            "pose_tests": [pose_target],
        }
    )
    debug_payload_path = write_worker_payload_debug_fn(worker_payload, label=f"{case_name}_{phase_name}")
    print(
        f"[cuRoboWorker] launching isolated pose planner phase={case_name}:{phase_name} "
        f"payload_debug_path={debug_payload_path}",
        flush=True,
    )
    try:
        worker_result = run_pose_plan_worker_fn(
            worker_payload,
            env=env,
            robot=robot,
            hold_q=q_start,
            hold_gripper_joint1_pos=hold_gripper_joint1_pos,
        )
    except BaseException as exc:
        print(
            f"[cuRoboWorker] failed phase={case_name}:{phase_name} "
            f"payload_debug_path={debug_payload_path} error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
    print(f"[cuRoboWorker] completed phase={case_name}:{phase_name}", flush=True)
    result = worker_result["results"][0]
    print(
        f"[PosePlan] case={case_name} phase={phase_name} success={result['success']} "
        f"elapsed_s={float(result['elapsed_s']):.6f}",
        flush=True,
    )
    if result.get("metrics"):
        print(f"[PlannerMetrics] case={case_name} phase={phase_name} metrics={result['metrics']}", flush=True)
    return result


def plan_cspace_target(
    *,
    case_name: str,
    phase_name: str,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    q_start: torch.Tensor,
    scene_mesh_obstacles: list[dict],
    target_q: torch.Tensor,
    env,
    robot,
    hold_gripper_joint1_pos: float | None,
    cfg: PlannerClientConfig,
    run_pose_plan_worker_fn,
    write_worker_payload_debug_fn,
) -> dict:
    cspace_target = {
        "name": f"{case_name}:{phase_name}",
        "joint_pos": target_q.reshape(-1).detach().cpu().tolist(),
    }
    worker_payload = base_worker_payload(
        urdf_path=urdf_path,
        robot_cfg_path=robot_cfg_path,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
        q_start=q_start,
        scene_mesh_obstacles=scene_mesh_obstacles,
        cfg=cfg,
    )
    worker_payload["cspace_targets"] = [cspace_target]
    debug_payload_path = write_worker_payload_debug_fn(worker_payload, label=f"{case_name}_{phase_name}")
    print(
        f"[cuRoboWorker] launching isolated cspace planner phase={case_name}:{phase_name} "
        f"payload_debug_path={debug_payload_path}",
        flush=True,
    )
    worker_result = run_pose_plan_worker_fn(
        worker_payload,
        env=env,
        robot=robot,
        hold_q=q_start,
        hold_gripper_joint1_pos=hold_gripper_joint1_pos,
    )
    print(f"[cuRoboWorker] completed phase={case_name}:{phase_name}", flush=True)
    result = worker_result["results"][0]
    print(
        f"[CspacePlan] case={case_name} phase={phase_name} success={result['success']} "
        f"elapsed_s={float(result['elapsed_s']):.6f}",
        flush=True,
    )
    if result.get("metrics"):
        print(f"[CspaceMetrics] case={case_name} phase={phase_name} metrics={result['metrics']}", flush=True)
    return result


def plan_multi_pose_polynomial_seed(
    *,
    case_name: str,
    env,
    robot,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    q_start: torch.Tensor,
    scene_mesh_obstacles: list[dict],
    target_poses_base: list[dict],
    seed_mode: str,
    cfg: PlannerClientConfig,
    run_pose_plan_worker_fn,
    hold_gripper_joint1_pos: float | None,
) -> dict:
    worker_payload = base_worker_payload(
        urdf_path=urdf_path,
        robot_cfg_path=robot_cfg_path,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
        q_start=q_start,
        scene_mesh_obstacles=scene_mesh_obstacles,
        cfg=cfg,
    )
    worker_payload.update(
        {
            "high_z_tolerance_threshold_m": cfg.high_z_tolerance_threshold_m,
            "high_z_position_tolerance_m": cfg.high_z_position_tolerance_m,
            "high_z_orientation_tolerance_rad": cfg.high_z_orientation_tolerance_rad,
            "num_ik_seeds": cfg.num_ik_seeds,
            "num_trajopt_seeds": cfg.num_trajopt_seeds,
            "max_batch_size": max(1, len(target_poses_base)),
            "pose_tests": [],
            "multi_pose_plan": {
                "name": case_name,
                "seed_mode": seed_mode,
                "targets": target_poses_base,
            },
        }
    )
    print(
        f"[cuRoboWorker] launching isolated multi-pose planner "
        f"case={case_name} targets={len(target_poses_base)} seed_mode={seed_mode} "
        f"num_ik_seeds={worker_payload['num_ik_seeds']} "
        f"num_trajopt_seeds={worker_payload['num_trajopt_seeds']} "
        f"max_batch_size={worker_payload['max_batch_size']}",
        flush=True,
    )
    worker_result = run_pose_plan_worker_fn(
        worker_payload,
        env=env,
        robot=robot,
        hold_q=q_start,
        hold_gripper_joint1_pos=hold_gripper_joint1_pos,
    )
    print(f"[cuRoboWorker] completed multi-pose case={case_name}", flush=True)
    result = worker_result["results"][0]
    print(
        f"[MultiPosePlan] case={case_name} success={result['success']} "
        f"elapsed_s={float(result['elapsed_s']):.6f}",
        flush=True,
    )
    if result.get("metrics"):
        print(f"[MultiPosePlannerMetrics] case={case_name} metrics={result['metrics']}", flush=True)
    return result
