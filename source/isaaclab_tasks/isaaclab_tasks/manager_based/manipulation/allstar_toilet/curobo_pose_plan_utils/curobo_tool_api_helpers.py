from __future__ import annotations

from pathlib import Path

import torch

from .curobo_planner_client import PlannerClientConfig


def build_cspace_tool_payload(
    *,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    q_current: torch.Tensor,
    scene_mesh_obstacles: list[dict],
    cfg: PlannerClientConfig,
    cspace_targets: list[dict],
) -> dict:
    return {
        "urdf_path": str(urdf_path),
        "robot_cfg_path": str(robot_cfg_path),
        "base_link": base_link,
        "tool_frame": tool_frame,
        "joint_names": joint_names,
        "q_current": q_current.detach().cpu().tolist(),
        "scene_mesh_obstacles": scene_mesh_obstacles,
        "scene_cuboid_obstacles": cfg.scene_cuboid_obstacles,
        "collision_activation_distance": cfg.collision_activation_distance,
        "num_ik_seeds": cfg.num_ik_seeds,
        "num_trajopt_seeds": cfg.num_trajopt_seeds,
        "cspace_targets": cspace_targets,
    }


def build_pose_tool_payload(
    parameters: dict,
    *,
    pose_test: dict,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    q_current: torch.Tensor,
    scene_mesh_obstacles: list[dict],
    cfg: PlannerClientConfig,
) -> dict:
    payload = dict(parameters)
    payload["pose_tests"] = [pose_test]
    payload.update(
        {
            "urdf_path": str(urdf_path),
            "robot_cfg_path": str(robot_cfg_path),
            "base_link": base_link,
            "tool_frame": tool_frame,
            "joint_names": joint_names,
            "q_current": q_current.detach().cpu().tolist(),
            "scene_mesh_obstacles": scene_mesh_obstacles,
            "scene_cuboid_obstacles": cfg.scene_cuboid_obstacles,
        }
    )
    payload.setdefault("collision_activation_distance", cfg.collision_activation_distance)
    payload.setdefault("position_tolerance_m", cfg.position_tolerance_m)
    payload.setdefault("orientation_tolerance_rad", cfg.orientation_tolerance_rad)
    payload.setdefault("num_ik_seeds", cfg.num_ik_seeds)
    payload.setdefault("num_trajopt_seeds", cfg.num_trajopt_seeds)
    payload.setdefault("high_z_tolerance_threshold_m", cfg.high_z_tolerance_threshold_m)
    payload.setdefault("high_z_position_tolerance_m", cfg.high_z_position_tolerance_m)
    payload.setdefault("high_z_orientation_tolerance_rad", cfg.high_z_orientation_tolerance_rad)
    return payload


def parse_spot_clean_pose_tests(
    parameters: dict,
    *,
    case_name: str,
    float_list_parameter_fn,
    downsample_ordered_pose_tests_fn,
    max_count: int = 8,
) -> list[tuple[str, list[float], list[float]]]:
    target_positions = parameters.get("target_positions")
    if not isinstance(target_positions, (list, tuple)) or len(target_positions) < 1:
        raise ValueError("brush_execute_spot_clean requires non-empty target_positions.")
    target_quats = parameters.get("target_quats")
    if target_quats is not None:
        if not isinstance(target_quats, (list, tuple)) or len(target_quats) != len(target_positions):
            raise ValueError("target_quats must be omitted or have the same length as target_positions.")

    raw_pose_tests = []
    for target_idx, target_pos_value in enumerate(target_positions):
        item = {"target_pos": target_pos_value}
        target_pos = float_list_parameter_fn(item, "target_pos", 3, required=True)
        target_quat = None
        if target_quats is not None:
            quat_item = {"target_quat": target_quats[target_idx]}
            target_quat = float_list_parameter_fn(quat_item, "target_quat", 4, required=True)
        raw_pose_tests.append(
            (
                f"{case_name}_target_{target_idx + 1:02d}",
                target_pos,
                target_quat or [],
            )
        )
    sampled_pose_tests = downsample_ordered_pose_tests_fn(raw_pose_tests, max_count=max_count)
    if len(sampled_pose_tests) != len(raw_pose_tests):
        print(
            f"[BrushSpotCleanSample] case={case_name} input_count={len(raw_pose_tests)} "
            f"selected_count={len(sampled_pose_tests)} selected_poses={[name for name, _, _ in sampled_pose_tests]}",
            flush=True,
        )
    return sampled_pose_tests


def execute_planning_pose_result_trajectories(
    tool_output: dict,
    *,
    env,
    robot,
    pre_target_q: torch.Tensor,
    steps_per_waypoint: int,
    final_hold_steps: int,
    execute_trajectory_fn,
    hold_joint_target_if_close_fn,
    arm_joint_ids_fn,
    joint_error_fn,
    trajectory_label_suffix: str,
    hold_label: str,
    log_result_prefix: str | None = None,
    no_executable_message: str | None = None,
    include_fallback: bool = False,
    require_nonempty_2d: bool = False,
) -> dict:
    executed_names: list[str] = []
    skipped_names: list[str] = []
    waypoint_count = 0
    env_step_count = 0
    final_err_max = None
    fallback_executed = False
    for result in tool_output.get("results", []):
        case_name = str(result.get("name", "unnamed"))
        if log_result_prefix is not None:
            traj_len = len(result.get("trajectory", [])) if isinstance(result.get("trajectory"), list) else 0
            print(
                f"[{log_result_prefix}] name={case_name} success={bool(result.get('success'))} "
                f"traj_len={traj_len} metrics={result.get('metrics', {})}",
                flush=True,
            )
        if not result.get("success"):
            skipped_names.append(case_name)
            continue
        traj = torch.tensor(result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
        if traj.dim() > 2:
            traj = traj.reshape(-1, traj.shape[-1])
        if require_nonempty_2d and (traj.dim() != 2 or traj.shape[0] < 1):
            skipped_names.append(case_name)
            continue
        execute_trajectory_fn(env, robot, traj, label=f"{case_name}:{trajectory_label_suffix}")
        waypoint_count += int(traj.shape[0])
        env_step_count += int(traj.shape[0]) * max(1, int(steps_per_waypoint))
        executed_names.append(case_name)

    if not executed_names and no_executable_message:
        print(no_executable_message, flush=True)
    elif executed_names:
        actual_q = robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach()
        _, final_err_tensor, _ = joint_error_fn(pre_target_q, actual_q)
        final_err_max = float(final_err_tensor.item())
        if final_hold_steps > 0:
            env_step_count += hold_joint_target_if_close_fn(
                env,
                robot,
                pre_target_q,
                final_hold_steps,
                label=hold_label,
            )

    result = {
        "executed": bool(executed_names) or fallback_executed,
        "executed_results": executed_names,
        "skipped_results": skipped_names,
        "returned_to_planning_pose": bool(executed_names) or fallback_executed,
        "waypoint_count": waypoint_count,
        "env_step_count": env_step_count,
        "final_joint_err_max": final_err_max,
    }
    if include_fallback:
        result["fallback_executed"] = fallback_executed
    return result


def execute_point_clean_results(
    tool_output: dict,
    *,
    env,
    robot,
    pre_target_q: torch.Tensor,
    urdf_path: str,
    robot_cfg_path: Path,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
    scene_mesh_obstacles: list[dict],
    pre_target_executed: bool,
    execute_point_clean_trajectory_fn,
) -> dict:
    executed_names: list[str] = []
    skipped_names: list[str] = []
    waypoint_count = 0
    env_step_count = 0
    joint6_roll_executed = False
    returned_to_pre_target = False
    roll_metrics = {}
    for result in tool_output.get("results", []):
        case_name = str(result.get("name", "unnamed"))
        if not result.get("success"):
            skipped_names.append(case_name)
            continue
        traj = torch.tensor(result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
        if traj.dim() > 2:
            traj = traj.reshape(-1, traj.shape[-1])
        execution_info = execute_point_clean_trajectory_fn(
            env,
            robot,
            traj,
            pre_target_q,
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            base_link=base_link,
            tool_frame=tool_frame,
            joint_names=joint_names,
            scene_mesh_obstacles=scene_mesh_obstacles,
        )
        waypoint_count += int(execution_info["waypoint_count"])
        env_step_count += int(execution_info["env_step_count"])
        joint6_roll_executed = joint6_roll_executed or bool(execution_info["joint6_roll_executed"])
        returned_to_pre_target = returned_to_pre_target or bool(execution_info["returned_to_pre_target"])
        roll_metrics[case_name] = {
            "roll_joint_err_max": execution_info["roll_joint_err_max"],
            "rollback_joint_err_max": execution_info["rollback_joint_err_max"],
        }
        executed_names.append(case_name)
    return {
        "executed": bool(executed_names),
        "executed_results": executed_names,
        "skipped_results": skipped_names,
        "pre_target_executed": bool(pre_target_executed),
        "joint6_roll_executed": joint6_roll_executed,
        "returned_to_pre_target": returned_to_pre_target,
        "waypoint_count": waypoint_count,
        "env_step_count": env_step_count,
        "roll_metrics": roll_metrics,
    }


def execute_spot_clean_results(
    tool_output: dict,
    parameters: dict,
    *,
    env,
    robot,
    urdf_path: str,
    robot_cfg_path: Path,
    tool_frame: str,
    joint_names: list[str],
    scene_mesh_obstacles: list[dict],
    base_link: str,
    pre_target_executed: bool,
    execute_spot_clean_trajectory_fn,
) -> dict:
    executed_names: list[str] = []
    skipped_names: list[str] = []
    waypoint_count = 0
    env_step_count = 0
    final_metrics = {}
    for result in tool_output.get("results", []):
        case_name = str(result.get("name", "unnamed"))
        if not result.get("success"):
            skipped_names.append(case_name)
            continue
        traj = torch.tensor(result["trajectory"], device=robot.device, dtype=robot.data.joint_pos.dtype)
        if traj.dim() > 2:
            traj = traj.reshape(-1, traj.shape[-1])
        final_target_base = tool_output.get("spot_clean_final_target_base")
        if not isinstance(final_target_base, dict):
            raise ValueError("spot clean tool output is missing spot_clean_final_target_base.")
        final_pos_tolerance_m = float(parameters.get("final_pos_tolerance_m", 0.015))
        final_rot_tolerance_rad = parameters.get("final_rot_tolerance_rad")
        if final_rot_tolerance_rad is not None:
            final_rot_tolerance_rad = float(final_rot_tolerance_rad)
        if parameters.get("joint6_roll_rad") is not None:
            joint6_roll_rad = float(parameters["joint6_roll_rad"])
        else:
            joint6_roll_rad = float(parameters.get("joint6_roll_deg", 180.0)) * 3.141592653589793 / 180.0
        execution_info = execute_spot_clean_trajectory_fn(
            env,
            robot,
            traj,
            plan_result=result,
            urdf_path=urdf_path,
            robot_cfg_path=robot_cfg_path,
            tool_frame=tool_frame,
            joint_names=joint_names,
            scene_mesh_obstacles=scene_mesh_obstacles,
            base_link=base_link,
            final_target_base=final_target_base,
            final_pos_tolerance_m=final_pos_tolerance_m,
            final_rot_tolerance_rad=final_rot_tolerance_rad,
            joint6_roll_rad=joint6_roll_rad,
        )
        waypoint_count += int(execution_info["waypoint_count"])
        env_step_count += int(execution_info["env_step_count"])
        final_metrics[case_name] = {
            "final_joint_err_max": execution_info["final_joint_err_max"],
            "final_pose_converged": execution_info["final_pose_converged"],
            "final_pos_err_m": execution_info["final_pos_err_m"],
            "final_rot_err_rad": execution_info["final_rot_err_rad"],
            "final_rot_err_deg": execution_info["final_rot_err_deg"],
            "final_pose_steps": execution_info["final_pose_steps"],
            "joint6_roll_start_waypoint": execution_info.get("joint6_roll_start_waypoint"),
            "joint6_roll_end_waypoint": execution_info.get("joint6_roll_end_waypoint"),
            "joint6_roll_delta_rad": execution_info.get("joint6_roll_delta_rad"),
        }
        executed_names.append(case_name)
    return {
        "executed": bool(executed_names),
        "executed_results": executed_names,
        "skipped_results": skipped_names,
        "pre_target_executed": bool(pre_target_executed),
        "joint6_roll_executed": bool(executed_names),
        "returned_to_pre_target": False,
        "waypoint_count": waypoint_count,
        "env_step_count": env_step_count,
        "final_metrics": final_metrics,
    }
