"""Scene obstacle collection and export for cuRobo planning."""

from __future__ import annotations

import datetime
import math
from pathlib import Path

from .curobo_scene_mesh_utils import (
    _append_obj_component,
    _combine_mesh_components,
    _cuboid_mesh_from_pose_dims,
    _find_named_ancestor_prim,
    _find_prim_by_name,
    _gf_matrix_from_pose_w,
    _matrix_to_curobo_pose_list,
    _read_obj_mesh,
    _simplify_toilet_mesh_components,
    _toilet_lid_or_seat_name_from_prim,
    _triangulate_face,
    _write_obj_mesh,
)
from .curobo_scene_obstacle_config import SceneObstacleConfig


def _runtime_toilet_lid_seat_pose_maps(env) -> dict[int, dict[str, object]]:
    if env is None:
        return {}

    pose_maps: dict[int, dict[str, object]] = {}
    for root_index, asset_name in ((1, "toilet_a"), (2, "toilet_b")):
        try:
            toilet = env.scene[asset_name]
        except Exception:
            continue
        body_names = list(getattr(toilet.data, "body_names", []))
        body_pose_map: dict[str, object] = {}
        missing_body_names: list[str] = []
        for wanted_name in ("ToiletSeat", "ToiletLid"):
            body_idx = None
            if wanted_name in body_names:
                body_idx = body_names.index(wanted_name)
            else:
                for idx, body_name in enumerate(body_names):
                    if str(body_name).endswith(wanted_name):
                        body_idx = idx
                        break
            if body_idx is None:
                missing_body_names.append(wanted_name)
                continue
            body_pose_map[wanted_name] = _gf_matrix_from_pose_w(
                toilet.data.body_pos_w[0, body_idx],
                toilet.data.body_quat_w[0, body_idx],
            )
        if missing_body_names:
            raise RuntimeError(
                f"Open-seat toilet articulation '{asset_name}' is present, but runtime body poses are missing for "
                f"{missing_body_names}. Available body_names={body_names}. Refusing to export inconsistent cuRobo "
                "toilet collision meshes."
            )
        if body_pose_map:
            pose_maps[root_index] = body_pose_map
    return pose_maps


def _export_toilet_ceramic_mesh_obstacles(
    stage,
    *,
    base_prim_name: str,
    cfg: SceneObstacleConfig,
    base_pose_w: tuple[torch.Tensor, torch.Tensor] | None = None,
    runtime_lid_seat_pose_maps: dict[int, dict[str, object]] | None = None,
    target_positions_w: list[list[float]] | None = None,
) -> list[dict[str, str]]:
    from pxr import Gf, Usd, UsdGeom

    base_prim = _find_prim_by_name(stage, "/World/envs/env_0/robot", base_prim_name)
    if base_prim is None or not base_prim.IsValid():
        print(f"[CuroboSceneMesh] disabled: could not find base prim name={base_prim_name}", flush=True)
        return []

    cache = UsdGeom.XformCache()
    if base_pose_w is not None:
        world_to_base = _gf_matrix_from_pose_w(base_pose_w[0], base_pose_w[1]).GetInverse()
    else:
        world_to_base = cache.GetLocalToWorldTransform(base_prim).GetInverse()
    toilet_roots = [
        "/World/envs/env_0/room/Meshes/Assets/Toilet_YREQ63RVAJ2SUPTULA888888_YlDCP",
        "/World/envs/env_0/room/Meshes/Assets/Toilet_YREQ6BBVAI7FMPTUJQ888888_KIiuW",
    ]
    runtime_lid_seat_pose_maps = runtime_lid_seat_pose_maps or {}
    mesh_dir = cfg.mesh_log_dir
    mesh_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    obstacles: list[dict[str, str]] = []
    candidates: list[dict[str, object]] = []
    front_depth_m = cfg.box_collision_front_depth_m
    front_half_width_m = cfg.box_collision_front_half_width_m
    front_min_x_m = cfg.box_collision_front_min_x_m
    target_points_base_xy: list[tuple[float, float]] = []
    for target_pos_w in target_positions_w or []:
        if target_pos_w is None or len(target_pos_w) < 2:
            continue
        target_point_base = world_to_base.Transform(
            Gf.Vec3d(float(target_pos_w[0]), float(target_pos_w[1]), float(target_pos_w[2]))
        )
        target_points_base_xy.append((float(target_point_base[0]), float(target_point_base[1])))
    if target_points_base_xy:
        print(f"[CuroboSceneMesh] selecting front toilet by target_points_base_xy={target_points_base_xy}", flush=True)

    for root_index, root_path in enumerate(toilet_roots, start=1):
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            print(f"[CuroboSceneMesh] missing toilet root: {root_path}", flush=True)
            continue

        components: list[dict[str, object]] = []
        mesh_count = 0
        skipped_lid_seat_mesh_count = 0
        runtime_lid_seat_mesh_count = 0
        missing_runtime_lid_seat_mesh_count = 0
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            movable_name = _toilet_lid_or_seat_name_from_prim(prim)
            runtime_pose = None
            static_body_to_world = None
            if movable_name is not None:
                runtime_pose = runtime_lid_seat_pose_maps.get(root_index, {}).get(movable_name)
                if root_index in runtime_lid_seat_pose_maps and runtime_pose is None:
                    raise RuntimeError(
                        f"cuRobo toilet mesh export found {movable_name} under {root_path}, but runtime pose for "
                        f"toilet root index {root_index} is missing. Refusing to export a stale closed-pose mesh."
                    )
                if runtime_pose is None:
                    skipped_lid_seat_mesh_count += 1
                    continue
                if runtime_pose is not None:
                    body_prim = _find_named_ancestor_prim(prim, movable_name)
                    if body_prim is None or not body_prim.IsValid():
                        raise RuntimeError(
                            f"cuRobo toilet mesh export found {prim.GetPath()} and runtime pose for {movable_name}, "
                            "but could not find the matching USD body prim. Refusing to export an inconsistent mesh."
                        )
                    static_body_to_world = cache.GetLocalToWorldTransform(body_prim)
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not points or not face_counts or not face_indices:
                continue
            static_mesh_to_world = cache.GetLocalToWorldTransform(prim)
            static_world_to_body = static_body_to_world.GetInverse() if static_body_to_world is not None else None
            if runtime_pose is not None and static_world_to_body is not None:
                runtime_lid_seat_mesh_count += 1
            component_vertices: list[tuple[float, float, float]] = []
            component_faces: list[tuple[int, int, int]] = []
            for point in points:
                point_world = static_mesh_to_world.Transform(point)
                if runtime_pose is not None and static_world_to_body is not None:
                    point_body = static_world_to_body.Transform(point_world)
                    point_world = runtime_pose.Transform(point_body)
                point_base = world_to_base.Transform(point_world)
                component_vertices.append((float(point_base[0]), float(point_base[1]), float(point_base[2])))
            cursor = 0
            for face_count in face_counts:
                polygon = [int(idx) for idx in face_indices[cursor : cursor + face_count]]
                component_faces.extend(_triangulate_face(polygon))
                cursor += face_count
            if not component_vertices or not component_faces:
                continue
            components.append(
                {
                    "name": str(prim.GetPath()),
                    "is_lid_seat": movable_name is not None,
                    "vertices": component_vertices,
                    "faces": component_faces,
                }
            )
            mesh_count += 1

        vertices, faces = _combine_mesh_components(components)
        if not vertices or not faces:
            print(f"[CuroboSceneMesh] no ceramic mesh exported for root: {root_path}", flush=True)
            continue

        x_values = [vertex[0] for vertex in vertices]
        y_values = [vertex[1] for vertex in vertices]
        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)
        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)
        intersects_front_x = x_max >= front_min_x_m and (front_depth_m <= 0.0 or x_min <= front_depth_m)
        intersects_front_y = (
            front_half_width_m <= 0.0 or (y_max >= -front_half_width_m and y_min <= front_half_width_m)
        )
        if not (intersects_front_x and intersects_front_y):
            print(
                f"[CuroboSceneMesh] skipped toilet root outside front window root={root_index} "
                f"x_range=[{x_min:.3f},{x_max:.3f}] y_range=[{y_min:.3f},{y_max:.3f}] path={root_path}",
                flush=True,
            )
            continue

        raw_vertex_count = len(vertices)
        raw_face_count = len(faces)
        simplify_info: dict[str, object] = {"method": "full"}
        output_suffix = ""
        if cfg.mesh_obstacle_mode == "simplified":
            simplified_vertices, simplified_faces, simplify_info = _simplify_toilet_mesh_components(
                components,
                target_faces=cfg.mesh_target_faces,
                preserve_lid_seat=cfg.mesh_preserve_lid_seat,
                lid_seat_target_faces=cfg.mesh_lid_seat_target_faces,
                preserve_outline=cfg.mesh_preserve_outline,
                feature_angle_deg=cfg.mesh_feature_angle_deg,
            )
            method = str(simplify_info.get("method", ""))
            if method.startswith("trimesh_") and method != "trimesh_quadric":
                print(
                    f"[CuroboSceneMesh] simplified fallback_to_full root={root_index} reason={method}",
                    flush=True,
                )
            else:
                vertices, faces = simplified_vertices, simplified_faces
                output_suffix = "_simplified"

        candidates.append(
            {
                "root_index": root_index,
                "root_path": root_path,
                "vertices": vertices,
                "faces": faces,
                "mesh_count": mesh_count,
                "skipped_lid_seat_mesh_count": skipped_lid_seat_mesh_count,
                "runtime_lid_seat_mesh_count": runtime_lid_seat_mesh_count,
                "missing_runtime_lid_seat_mesh_count": missing_runtime_lid_seat_mesh_count,
                "raw_vertex_count": raw_vertex_count,
                "raw_face_count": raw_face_count,
                "simplify_info": simplify_info,
                "output_suffix": output_suffix,
                "x_min_m": x_min,
                "x_max_m": x_max,
                "y_min_m": y_min,
                "y_max_m": y_max,
                "center_x_m": center_x,
                "center_y_m": center_y,
            }
        )

    def _bbox_distance_to_target(candidate: dict[str, object], target_xy: tuple[float, float]) -> float:
        tx, ty = target_xy
        x_min = float(candidate["x_min_m"])
        x_max = float(candidate["x_max_m"])
        y_min = float(candidate["y_min_m"])
        y_max = float(candidate["y_max_m"])
        dx = 0.0 if x_min <= tx <= x_max else min(abs(tx - x_min), abs(tx - x_max))
        dy = 0.0 if y_min <= ty <= y_max else min(abs(ty - y_min), abs(ty - y_max))
        return math.hypot(dx, dy)

    if target_points_base_xy:
        for candidate in candidates:
            candidate["target_xy_distance_m"] = min(
                _bbox_distance_to_target(candidate, target_xy) for target_xy in target_points_base_xy
            )
        candidates.sort(
            key=lambda item: (
                float(item["target_xy_distance_m"]),
                max(0.0, float(item["x_min_m"])),
                abs(float(item["center_y_m"])),
            )
        )
    else:
        candidates.sort(key=lambda item: (max(0.0, float(item["x_min_m"])), abs(float(item["center_y_m"]))))
    if len(candidates) > 1:
        skipped_paths = [str(candidate["root_path"]) for candidate in candidates[1:]]
        print(
            f"[CuroboSceneMesh] selected one front toilet root; skipped_other_front_roots={skipped_paths}",
            flush=True,
        )
    if not candidates:
        print(
            f"[CuroboSceneMesh] no toilet root intersects front window "
            f"front_min_x_m={front_min_x_m:.3f} front_depth_m={front_depth_m:.3f} "
            f"front_half_width_m={front_half_width_m:.3f}",
            flush=True,
        )
    selected_candidates = candidates  # 导出全部马桶，不止 nearest，确保所有马桶盖几何都在碰撞模型里

    for candidate in selected_candidates:
        root_index = int(candidate["root_index"])
        root_path = str(candidate["root_path"])
        vertices = candidate["vertices"]
        faces = candidate["faces"]
        mesh_count = int(candidate["mesh_count"])
        skipped_lid_seat_mesh_count = int(candidate["skipped_lid_seat_mesh_count"])
        runtime_lid_seat_mesh_count = int(candidate["runtime_lid_seat_mesh_count"])
        missing_runtime_lid_seat_mesh_count = int(candidate["missing_runtime_lid_seat_mesh_count"])
        raw_vertex_count = int(candidate["raw_vertex_count"])
        raw_face_count = int(candidate["raw_face_count"])
        simplify_info = candidate["simplify_info"]
        output_suffix = str(candidate["output_suffix"])

        obj_path = mesh_dir / f"toilet_ceramic_{root_index}{output_suffix}_{timestamp}.obj"
        _write_obj_mesh(
            obj_path,
            vertices=vertices,
            faces=faces,
            comments=[
                f"cuRobo obstacle exported from {root_path}",
                "vertices are baked into XMS5 arm base frame",
                f"mesh_obstacle_mode={cfg.mesh_obstacle_mode}",
                f"simplify_info={simplify_info}",
            ],
        )
        name = f"toilet_ceramic_{root_index}"
        obstacles.append({"name": name, "file_path": str(obj_path)})
        print(
            f"[CuroboSceneMesh] exported {name}: meshes={mesh_count} "
            f"skipped_lid_seat_meshes={skipped_lid_seat_mesh_count} "
            f"runtime_lid_seat_meshes={runtime_lid_seat_mesh_count} "
            f"missing_runtime_lid_seat_meshes={missing_runtime_lid_seat_mesh_count} "
            f"mode={cfg.mesh_obstacle_mode} "
            f"x_range=[{float(candidate['x_min_m']):.3f},{float(candidate['x_max_m']):.3f}] "
            f"y_range=[{float(candidate['y_min_m']):.3f},{float(candidate['y_max_m']):.3f}] "
            f"target_xy_distance_m={float(candidate.get('target_xy_distance_m', -1.0)):.3f} "
            f"raw_vertices={raw_vertex_count} raw_faces={raw_face_count} "
            f"vertices={len(vertices)} faces={len(faces)} simplify={simplify_info} path={obj_path}",
            flush=True,
        )

    return obstacles


def _export_room_base_mesh_obstacles(
    stage,
    *,
    base_prim_name: str,
    cfg: SceneObstacleConfig,
    base_pose_w: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> list[dict[str, str]]:
    """Export bathroom room base meshes (walls, partitions) as cuRobo collision obstacles."""
    from pxr import Gf, Usd, UsdGeom

    base_prim = _find_prim_by_name(stage, "/World/envs/env_0/robot", base_prim_name)
    if base_prim is None or not base_prim.IsValid():
        print(f"[CuroboSceneMesh] room_base: could not find base prim name={base_prim_name}", flush=True)
        return []

    cache = UsdGeom.XformCache()
    if base_pose_w is not None:
        world_to_base = _gf_matrix_from_pose_w(base_pose_w[0], base_pose_w[1]).GetInverse()
    else:
        world_to_base = cache.GetLocalToWorldTransform(base_prim).GetInverse()

    room_base_root = "/World/envs/env_0/room/Meshes/Base"
    base_root_prim = stage.GetPrimAtPath(room_base_root)
    if not base_root_prim.IsValid():
        print(f"[CuroboSceneMesh] room_base: missing root {room_base_root}", flush=True)
        return []

    # 只导出明确的墙壁/隔断 prim，避免导出天花板、地板、门等低精度杂物
    wall_prim_names = ["Wall_0000"] + [f"Partition_{i:04d}" for i in range(10)]
    wall_prims: list = []
    for name in wall_prim_names:
        prim = stage.GetPrimAtPath(f"{room_base_root}/{name}")
        if prim.IsValid():
            wall_prims.append(prim)
    if not wall_prims:
        print("[CuroboSceneMesh] room_base: no Wall/Partition prims found", flush=True)
        return []

    mesh_dir = cfg.mesh_log_dir
    mesh_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    components: list[dict[str, object]] = []
    mesh_count = 0
    total_raw_vertices = 0
    total_raw_faces = 0
    for wall_prim in wall_prims:
        for prim in Usd.PrimRange(wall_prim, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not points or not face_counts or not face_indices:
                continue
            mesh_to_world = cache.GetLocalToWorldTransform(prim)
            component_vertices: list[tuple[float, float, float]] = []
            component_faces: list[tuple[int, int, int]] = []
            for point in points:
                point_world = mesh_to_world.Transform(point)
                point_base = world_to_base.Transform(point_world)
                component_vertices.append((float(point_base[0]), float(point_base[1]), float(point_base[2])))
            # 防御：非有限顶点（inf/nan/FLT_MAX）会让 cuRobo 的 SDF 体素化触发
            # Warp illegal memory access 崩溃。跳过整块退化 mesh，避免拖垮整个规划。
            if not all(
                math.isfinite(v) and abs(v) < 1.0e6 for vertex in component_vertices for v in vertex
            ):
                print(
                    f"[CuroboSceneMesh] room_base skip degenerate prim {prim.GetPath()} "
                    f"(non-finite vertices={len(component_vertices)})",
                    flush=True,
                )
                continue
            cursor = 0
            for face_count in face_counts:
                polygon = [int(idx) for idx in face_indices[cursor : cursor + face_count]]
                component_faces.extend(_triangulate_face(polygon))
                cursor += face_count
            if not component_vertices or not component_faces:
                continue
            total_raw_vertices += len(component_vertices)
            total_raw_faces += len(component_faces)
            components.append({
                "name": str(prim.GetPath()),
                "is_lid_seat": False,
                "vertices": component_vertices,
                "faces": component_faces,
            })
            mesh_count += 1

    if not components:
        print("[CuroboSceneMesh] room_base: no meshes found", flush=True)
        return []

    vertices, faces = _combine_mesh_components(components)
    if not vertices or not faces:
        print("[CuroboSceneMesh] room_base: empty combined mesh", flush=True)
        return []

    # Simplify room base mesh
    simplify_info: dict[str, object] = {"method": "full"}
    output_suffix = ""
    if cfg.mesh_obstacle_mode == "simplified":
        simplified_vertices, simplified_faces, simplify_info = _simplify_toilet_mesh_components(
            components,
            target_faces=cfg.mesh_target_faces,
            preserve_lid_seat=cfg.mesh_preserve_lid_seat,
            lid_seat_target_faces=cfg.mesh_lid_seat_target_faces,
            preserve_outline=cfg.mesh_preserve_outline,
            feature_angle_deg=cfg.mesh_feature_angle_deg,
        )
        method = str(simplify_info.get("method", ""))
        if method.startswith("trimesh_") and method != "trimesh_quadric":
            print(f"[CuroboSceneMesh] room_base simplified fallback_to_full reason={method}", flush=True)
        else:
            vertices, faces = simplified_vertices, simplified_faces
            output_suffix = "_simplified"

    obj_path = mesh_dir / f"room_base{output_suffix}_{timestamp}.obj"
    _write_obj_mesh(
        obj_path, vertices=vertices, faces=faces,
        comments=[
            "cuRobo room base obstacle (walls, partitions)",
            "vertices are baked into XMS5 arm base frame",
            f"mesh_obstacle_mode={cfg.mesh_obstacle_mode}",
            f"simplify_info={simplify_info}",
        ],
    )
    name = "room_base"
    print(
        f"[CuroboSceneMesh] exported {name}: meshes={mesh_count} "
        f"raw_vertices={total_raw_vertices} raw_faces={total_raw_faces} "
        f"vertices={len(vertices)} faces={len(faces)} simplify={simplify_info} path={obj_path}",
        flush=True,
    )
    return [{"name": name, "file_path": str(obj_path)}]


def _collect_curobo_scene_mesh_obstacles(
    stage,
    *,
    isaac_base_body: str,
    cfg: SceneObstacleConfig,
    gripper_profile: bool,
    env=None,
    base_pose_w: tuple[torch.Tensor, torch.Tensor] | None = None,
    target_positions_w: list[list[float]] | None = None,
) -> list[dict[str, str]]:
    scene_mesh_obstacles: list[dict[str, str]] = []
    use_toilet_mesh_collision = not cfg.disable_toilet_mesh_collision
    if use_toilet_mesh_collision:
        runtime_lid_seat_pose_maps = _runtime_toilet_lid_seat_pose_maps(env)
        scene_mesh_obstacles.extend(
            _export_toilet_ceramic_mesh_obstacles(
                stage,
                base_prim_name=isaac_base_body,
                cfg=cfg,
                base_pose_w=base_pose_w,
                runtime_lid_seat_pose_maps=runtime_lid_seat_pose_maps,
                target_positions_w=target_positions_w,
            )
        )
    else:
        print(
            "[CuroboSceneMesh] toilet export disabled for this run "
            f"(disable={cfg.disable_toilet_mesh_collision}, force_enable={cfg.enable_toilet_mesh_collision}, "
            f"gripper_profile={gripper_profile})",
            flush=True,
        )
    # Export room base walls (Wall_0000 + Partitions) as cuRobo mesh obstacles
    if cfg.wall_obstacle_mode == "mesh":
        scene_mesh_obstacles.extend(
            _export_room_base_mesh_obstacles(
                stage,
                base_prim_name=isaac_base_body,
                cfg=cfg,
                base_pose_w=base_pose_w,
            )
        )
    elif cfg.wall_obstacle_mode == "cuboid":
        # CuroboOnlyWalls handles cuboid export separately via _ensure_raycast_wall_cuboids
        pass
    # else "off": no wall obstacles
    print(f"[CuroboSceneMesh] enabled_obstacles={len(scene_mesh_obstacles)} wall_mode={cfg.wall_obstacle_mode}", flush=True)
    return scene_mesh_obstacles


# === PhysX raycast 墙壁扫描（懒加载，首次工具调用时执行） ===
_RAYCAST_WALLS_CACHED: list[dict] | None = None  # None = 还没扫过


def _ensure_raycast_wall_cuboids(stage, base_pose_w: tuple, *, wall_mode: str = "mesh") -> list[dict]:
    """懒加载：首次调用时用 PhysX raycast 扫描墙壁，生成 cuRobo-only cuboid。

    Only active when wall_mode == "cuboid".  In mesh/off modes, returns empty."""
    if wall_mode != "cuboid":
        return []
    global _RAYCAST_WALLS_CACHED
    if _RAYCAST_WALLS_CACHED is not None:
        return _RAYCAST_WALLS_CACHED

    _RAYCAST_WALLS_CACHED = []  # 先设为空防重入
    try:
        import omni.physx
        sq = omni.physx.get_physx_scene_query_interface()
    except Exception as e:
        print(f"[RaycastWalls] PhysX scene query unavailable: {e}", flush=True)
        return []

    from pxr import Gf, Sdf, Usd, UsdGeom

    # 确保 CuroboOnlyWalls 容器路径存在
    for pp in ["/World/envs/env_0/room/CollisionFixes/CuroboOnlyWalls"]:
        if not stage.GetPrimAtPath(pp).IsValid():
            stage.DefinePrim(pp, "Xform")

    # 先从高处向下打一条测试射线确认 scene query 可用
    test_hit = sq.raycast_closest((1.8, -1.0, 2.0), (0.0, 0.0, -1.0), 5.0)
    print(f"[RaycastWalls] test down-ray: {test_hit}", flush=True)

    # 扫描 ROI：从任务区外侧向内打
    x_outer_l, x_outer_r = 0.8, 2.6
    y_outer_f, y_outer_b = -1.8, -0.2
    z_list = [0.45, 0.9, 1.4]
    step = 0.05
    max_dist = 3.0

    def _accept_path(s: str) -> bool:
        s = s.lower()
        for k in ["robot", "cube", "trash", "toilet", "gripper", "mecanum", "dirt", "shitstain", "floor", "ceiling"]:
            if k in s: return False
        for k in ["wall", "partition", "door", "/room/meshes/base"]:
            if k in s: return True
        return False

    hits = []
    for z in z_list:
        # Y 扫描：从左右两边向内打
        y = -1.6
        while y <= -0.35:
            for direction, sx in [(1.0, x_outer_l), (-1.0, x_outer_r)]:
                try:
                    h = sq.raycast_closest((sx, y, z), (direction, 0.0, 0.0), max_dist)
                except Exception:
                    continue
                if h and (isinstance(h, dict) and h.get("hit")) or (hasattr(h, "hit") and h.hit):
                    rp = str(h.get("rigidBody", "") if isinstance(h, dict) else getattr(h, "rigidBody", ""))
                    hp = h.get("position", (0, 0, 0)) if isinstance(h, dict) else getattr(h, "position", (0, 0, 0))
                    if _accept_path(rp):
                        hits.append(tuple(float(v) for v in hp))
            y += step

        # X 扫描：从前后两边向内打
        x = 1.0
        while x <= 2.4:
            for direction, sy in [(1.0, y_outer_f), (-1.0, y_outer_b)]:
                try:
                    h = sq.raycast_closest((x, sy, z), (0.0, direction, 0.0), max_dist)
                except Exception:
                    continue
                if h and (isinstance(h, dict) and h.get("hit")) or (hasattr(h, "hit") and h.hit):
                    rp = str(h.get("rigidBody", "") if isinstance(h, dict) else getattr(h, "rigidBody", ""))
                    hp = h.get("position", (0, 0, 0)) if isinstance(h, dict) else getattr(h, "position", (0, 0, 0))
                    if _accept_path(rp):
                        hits.append(tuple(float(v) for v in hp))
            x += step

    print(f"[RaycastWalls] {len(hits)} hits on walls", flush=True)
    if len(hits) < 10:
        return []

    # 聚类：X 常量 → 竖直墙，Y 常量 → 水平墙
    def _cluster(values, binsize=0.04, min_count=8):
        bins = {}
        for v in values:
            k = round(v[0] / binsize) * binsize if v[3] == "x" else round(v[1] / binsize) * binsize
            bins.setdefault(k, []).append(v)
        return [pts for pts in bins.values() if len(pts) >= min_count]

    x_clusters = _cluster([(h[0], h[1], h[2], "x") for h in hits])
    y_clusters = _cluster([(h[0], h[1], h[2], "y") for h in hits])

    cuboids = []
    for c in x_clusters:
        xs, ys = [p[0] for p in c], [p[1] for p in c]
        cuboids.append((sum(xs) / len(xs), (min(ys) + max(ys)) * 0.5, 1.0, 0.06, max(ys) - min(ys) + 0.10, 2.0))
    for c in y_clusters:
        xs, ys = [p[0] for p in c], [p[1] for p in c]
        cuboids.append(((min(xs) + max(xs)) * 0.5, sum(ys) / len(ys), 1.0, max(xs) - min(xs) + 0.10, 0.06, 2.0))

    # 去重
    unique = []
    for c in cuboids:
        if not any(abs(c[0] - u[0]) < 0.1 and abs(c[1] - u[1]) < 0.1 for u in unique):
            unique.append(c)

    # 生成 cuRobo-only cuboid（curobo:obstacle 属性，不加 CollisionAPI）
    for i, (cx, cy, cz, sx, sy, sz) in enumerate(unique):
        cp = f"/World/envs/env_0/room/CollisionFixes/CuroboOnlyWalls/rw_{i:03d}"
        if stage.GetPrimAtPath(cp).IsValid():
            continue
        cube = UsdGeom.Cube.Define(stage, cp)
        cube.CreateSizeAttr(1.0)
        prim = cube.GetPrim()
        prim.CreateAttribute("curobo:obstacle", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("curobo:source", Sdf.ValueTypeNames.String).Set("physx_raycast")
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
        xf.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))

    print(f"[RaycastWalls] {len(unique)} cuRobo-only wall cuboids generated", flush=True)

    # 玻璃门面板（UsdGeom.Cube + curobo:obstacle → 可见 + cuRobo 碰撞）
    _door_base = "/World/envs/env_0/room/CollisionFixes/CuroboOnlyWalls"
    if not stage.GetPrimAtPath(_door_base + "/glass_door_left").IsValid():
        for name, px, py, pz, sx, sy, sz in [
            ("glass_door_left",  1.40, -0.03, 1.27, 0.40, 0.02, 2.46),
            ("glass_door_right", 2.24, -0.03, 1.27, 0.40, 0.02, 2.46),
        ]:
            fp = _door_base + "/" + name
            cube = UsdGeom.Cube.Define(stage, fp)
            cube.CreateSizeAttr(1.0)
            prim = cube.GetPrim()
            prim.CreateAttribute("curobo:obstacle", Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute("curobo:source", Sdf.ValueTypeNames.String).Set("glass_door_panel")
            xf = UsdGeom.Xformable(prim)
            xf.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
            xf.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))
            prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set([(0.02, 0.02, 0.08)])
        print("[RaycastWalls] 2 glass door panels (curobo:obstacle)", flush=True)

    _RAYCAST_WALLS_CACHED = [{"name": f"raycast_wall_{i:03d}", "file_path": ""} for i in range(len(unique))]
    return _RAYCAST_WALLS_CACHED


def _collect_curobo_scene_cuboid_obstacles(
    stage,
    *,
    cfg: SceneObstacleConfig,
    base_pose_w: tuple[torch.Tensor, torch.Tensor],
) -> list[dict]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if cfg.disable_curobo_box_collision:
        print("[CuroboSceneCuboid] disabled by --disable_curobo_box_collision", flush=True)
        return []

    front_depth_m = cfg.box_collision_front_depth_m
    front_half_width_m = cfg.box_collision_front_half_width_m
    front_min_x_m = cfg.box_collision_front_min_x_m
    base_to_world = _gf_matrix_from_pose_w(base_pose_w[0], base_pose_w[1])
    world_to_base = base_to_world.GetInverse()
    cache = UsdGeom.XformCache()
    cuboids: list[dict] = []
    scanned = 0
    skipped_outside_front = 0
    export_dir = cfg.collision_log_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    roots = (
        "/World/envs/env_0/room/CollisionFixes/WallBoxes",
        "/World/envs/env_0/room/CollisionFixes/PartitionBoxes",
    )

    # 也采集 CuroboOnlyWalls（curobo:obstacle 属性，无需 CollisionAPI）
    _curobo_only_root = "/World/envs/env_0/room/CollisionFixes/CuroboOnlyWalls"
    _curobo_only_prim = stage.GetPrimAtPath(_curobo_only_root)
    if _curobo_only_prim.IsValid():
        roots = tuple(list(roots) + [_curobo_only_root])

    # 也采集 paper_waste 下的 cuRobo proxy（curobo:obstacle 属性，无 CollisionAPI）
    _paper_waste_root = "/World/envs/env_0/paper_waste"
    if stage.GetPrimAtPath(_paper_waste_root).IsValid():
        roots = tuple(list(roots) + [_paper_waste_root])

    for root_path in roots:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            continue
        _is_curobo_only = root_path == _curobo_only_root
        _is_paper_waste = root_path == _paper_waste_root
        _is_attr_based = _is_curobo_only or _is_paper_waste
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Cube):
                continue
            if _is_attr_based:
                # 检查 curobo:obstacle 属性（curobo_only / paper_waste 代理）
                cob_attr = prim.GetAttribute("curobo:obstacle")
                cob_val = cob_attr.Get() if (cob_attr and cob_attr.IsValid()) else None
                if cob_val is not True:
                    print(f"[CuroboSceneCuboid] SKIP {'paper_waste' if _is_paper_waste else 'curobo_only'} prim={prim.GetPath()} attr_valid={cob_attr and cob_attr.IsValid()} val={cob_val}", flush=True)
                    continue
                print(f"[CuroboSceneCuboid] ACCEPT {'paper_waste' if _is_paper_waste else 'curobo_only'} prim={prim.GetPath()}", flush=True)
                # 纸团代理：已收集的跳过
                pw_collected_attr = prim.GetAttribute("paper_waste:collected")
                if pw_collected_attr and pw_collected_attr.IsValid() and pw_collected_attr.Get() is True:
                    print(f"[CuroboSceneCuboid] SKIP paper_waste (collected) prim={prim.GetPath()}", flush=True)
                    continue
                # GPT P1: 已抓取/携带的跳过
                if not prim.IsActive():
                    print(f"[CuroboSceneCuboid] SKIP paper_waste (inactive) prim={prim.GetPath()}", flush=True)
                    continue
                pw_carried_attr = prim.GetAttribute("paper_waste:is_carried")
                if pw_carried_attr and pw_carried_attr.IsValid() and pw_carried_attr.Get() is True:
                    print(f"[CuroboSceneCuboid] SKIP paper_waste (carried) prim={prim.GetPath()}", flush=True)
                    continue
                pw_state_attr = prim.GetAttribute("paper_waste:state")
                if pw_state_attr and pw_state_attr.IsValid():
                    state_val = pw_state_attr.Get()
                    if state_val in ("grasped", "carried"):
                        print(f"[CuroboSceneCuboid] SKIP paper_waste (state={state_val}) prim={prim.GetPath()}", flush=True)
                        continue
            elif not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            else:
                attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
                if attr and attr.IsValid() and attr.Get() is False:
                    continue
            scale_attr = prim.GetAttribute("xformOp:scale")
            size_attr = prim.GetAttribute("size")
            scale = scale_attr.Get() if scale_attr and scale_attr.IsValid() else None
            if scale is None:
                continue
            size = size_attr.Get() if size_attr and size_attr.IsValid() and size_attr.Get() is not None else 1.0
            dims = [abs(float(scale[axis]) * float(size)) for axis in range(3)]
            if any(dim <= 0.0 for dim in dims):
                continue
            prim_to_world = cache.GetLocalToWorldTransform(prim)
            scanned += 1
            # Gf.Matrix4d composes in the same order as chained Transform() calls here:
            # world_to_base.Transform(prim_to_world.Transform(p)) == (prim_to_world * world_to_base).Transform(p).
            prim_to_base = prim_to_world * world_to_base
            source_path = str(prim.GetPath())
            cuboid_name = source_path.replace("/World/envs/env_0/room/CollisionFixes/", "scene_box_").replace("/", "_")
            pose = _matrix_to_curobo_pose_list(prim_to_base)
            box_vertices, box_faces = _cuboid_mesh_from_pose_dims(pose=pose, dims=dims)
            x_values = [vertex[0] for vertex in box_vertices]
            y_values = [vertex[1] for vertex in box_vertices]
            x_min = min(x_values)
            x_max = max(x_values)
            y_min = min(y_values)
            y_max = max(y_values)
            center_x = 0.5 * (x_min + x_max)
            center_y = 0.5 * (y_min + y_max)
            intersects_front_x = x_max >= front_min_x_m and (front_depth_m <= 0.0 or x_min <= front_depth_m)
            intersects_front_y = (
                front_half_width_m <= 0.0 or (y_max >= -front_half_width_m and y_min <= front_half_width_m)
            )
            if not (intersects_front_x and intersects_front_y):
                skipped_outside_front += 1
                print(f"[CuroboSceneCuboid] SKIP_FRONT name={cuboid_name} x=[{x_min:.3f},{x_max:.3f}] y=[{y_min:.3f},{y_max:.3f}]", flush=True)
                continue
            obj_path = export_dir / f"{cuboid_name}_{timestamp}.obj"
            _write_obj_mesh(
                obj_path,
                vertices=box_vertices,
                faces=box_faces,
                comments=[
                    f"cuRobo cuboid obstacle exported from {source_path}",
                    "vertices are baked into XMS5 arm base frame",
                    f"dims={dims}",
                    f"pose_wxyz={pose}",
                ],
            )
            cuboids.append(
                {
                    "name": cuboid_name,
                    "dims": dims,
                    "pose": pose,
                    "source_path": source_path,
                    "x_min_m": x_min,
                    "x_max_m": x_max,
                    "y_min_m": y_min,
                    "y_max_m": y_max,
                    "center_x_m": center_x,
                    "center_y_m": center_y,
                    "file_path": str(obj_path),
                }
            )

    cuboids.sort(key=lambda item: (max(0.0, item["x_min_m"]), abs(item["center_y_m"])))
    print(
        f"[CuroboSceneCuboid] enabled_obstacles={len(cuboids)} scanned={scanned} "
        f"skipped_outside_front={skipped_outside_front} front_min_x_m={front_min_x_m:.3f} "
        f"front_depth_m={front_depth_m:.3f} front_half_width_m={front_half_width_m:.3f}",
        flush=True,
    )
    for item in cuboids:
        print(
            f"[CuroboSceneCuboid] obstacle name={item['name']} "
            f"x_range=[{item['x_min_m']:.3f},{item['x_max_m']:.3f}] "
            f"y_range=[{item['y_min_m']:.3f},{item['y_max_m']:.3f}] "
            f"dims={item['dims']} source={item['source_path']} "
            f"path={item['file_path']}",
            flush=True,
        )
    return cuboids


def _export_curobo_scene_collision_obj(
    *,
    cfg: SceneObstacleConfig,
    scene_mesh_obstacles: list[dict],
    scene_cuboid_obstacles: list[dict],
) -> str | None:
    export_dir = cfg.collision_log_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = export_dir / f"curobo_scene_collision_all_{timestamp}.obj"
    vertex_offset = 0
    object_count = 0
    with combined_path.open("w", encoding="utf-8") as stream:
        stream.write("# cuRobo scene collision obstacles considered by this run\n")
        stream.write("# vertices are baked into XMS5 arm base frame\n")
        stream.write("# includes toilet mesh obstacles plus wall/partition cuboid primitives\n")
        stream.write("# floor collision is intentionally excluded\n")
        for obstacle in scene_mesh_obstacles:
            obj_path = Path(str(obstacle["file_path"]))
            vertices, faces = _read_obj_mesh(obj_path)
            if not vertices or not faces:
                continue
            vertex_offset = _append_obj_component(
                stream,
                object_name=str(obstacle["name"]),
                vertices=vertices,
                faces=faces,
                vertex_offset=vertex_offset,
            )
            object_count += 1
        for obstacle in scene_cuboid_obstacles:
            vertices, faces = _cuboid_mesh_from_pose_dims(
                pose=[float(value) for value in obstacle["pose"]],
                dims=[float(value) for value in obstacle["dims"]],
            )
            vertex_offset = _append_obj_component(
                stream,
                object_name=str(obstacle["name"]),
                vertices=vertices,
                faces=faces,
                vertex_offset=vertex_offset,
            )
            object_count += 1
    if object_count == 0:
        combined_path.unlink(missing_ok=True)
        print("[CuroboSceneCollisionObj] no scene collision OBJ exported: no active obstacles", flush=True)
        return None
    print(
        f"[CuroboSceneCollisionObj] exported combined_obj={combined_path} "
        f"objects={object_count} vertices={vertex_offset}",
        flush=True,
    )
    return str(combined_path)
