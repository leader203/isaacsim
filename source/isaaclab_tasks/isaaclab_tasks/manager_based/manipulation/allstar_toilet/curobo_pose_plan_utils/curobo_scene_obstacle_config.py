"""Configuration object for cuRobo scene obstacle export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneObstacleConfig:
    disable_toilet_mesh_collision: bool
    enable_toilet_mesh_collision: bool
    disable_curobo_box_collision: bool
    box_collision_front_depth_m: float
    box_collision_front_half_width_m: float
    box_collision_front_min_x_m: float
    mesh_obstacle_mode: str
    mesh_target_faces: int
    mesh_preserve_lid_seat: bool
    mesh_lid_seat_target_faces: int
    mesh_preserve_outline: bool
    mesh_feature_angle_deg: float
    mesh_log_dir: Path
    collision_log_dir: Path
    wall_obstacle_mode: str = "mesh"  # "mesh" | "cuboid" | "off"


def scene_obstacle_config_from_args(args_cli, *, task_root: str | Path) -> SceneObstacleConfig:
    task_root = Path(task_root)
    return SceneObstacleConfig(
        disable_toilet_mesh_collision=bool(args_cli.disable_toilet_mesh_collision),
        enable_toilet_mesh_collision=bool(args_cli.enable_toilet_mesh_collision),
        disable_curobo_box_collision=bool(args_cli.disable_curobo_box_collision),
        box_collision_front_depth_m=float(args_cli.curobo_box_collision_front_depth_m),
        box_collision_front_half_width_m=float(args_cli.curobo_box_collision_front_half_width_m),
        box_collision_front_min_x_m=float(args_cli.curobo_box_collision_front_min_x_m),
        mesh_obstacle_mode=str(args_cli.curobo_mesh_obstacle_mode),
        mesh_target_faces=max(1, int(args_cli.curobo_mesh_target_faces)),
        mesh_preserve_lid_seat=bool(args_cli.curobo_mesh_preserve_lid_seat),
        mesh_lid_seat_target_faces=max(1, int(args_cli.curobo_mesh_lid_seat_target_faces)),
        mesh_preserve_outline=bool(args_cli.curobo_mesh_preserve_outline),
        mesh_feature_angle_deg=float(args_cli.curobo_mesh_feature_angle_deg),
        mesh_log_dir=task_root / "logs" / "curobo_mesh_obstacles",
        collision_log_dir=task_root / "logs" / "curobo_collision_obstacles",
    )
