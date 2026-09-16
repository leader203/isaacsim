"""Pure mesh and USD helpers for cuRobo scene obstacle export."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _find_prim_by_name(stage, root_path: str, prim_name: str):
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetName() == prim_name:
            return prim
    return None


def _triangulate_face(face_indices: list[int]) -> list[tuple[int, int, int]]:
    if len(face_indices) < 3:
        return []
    return [(face_indices[0], face_indices[i], face_indices[i + 1]) for i in range(1, len(face_indices) - 1)]


def _write_obj_mesh(
    obj_path: Path,
    *,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    comments: list[str],
) -> None:
    with obj_path.open("w", encoding="utf-8") as stream:
        for comment in comments:
            stream.write(f"# {comment}\n")
        for vertex in vertices:
            stream.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for face in faces:
            stream.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def _read_obj_mesh(obj_path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with obj_path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f" and len(parts) >= 4:
                polygon: list[int] = []
                for token in parts[1:]:
                    vertex_token = token.split("/", 1)[0]
                    vertex_index = int(vertex_token)
                    if vertex_index < 0:
                        vertex_index = len(vertices) + vertex_index + 1
                    polygon.append(vertex_index - 1)
                faces.extend(_triangulate_face(polygon))
    return vertices, faces


def _cuboid_mesh_from_pose_dims(
    *,
    pose: list[float],
    dims: list[float],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    from pxr import Gf

    if len(pose) != 7 or len(dims) != 3:
        raise ValueError(f"Invalid cuboid pose/dims for OBJ export: pose={pose}, dims={dims}")
    hx, hy, hz = (0.5 * float(dims[0]), 0.5 * float(dims[1]), 0.5 * float(dims[2]))
    local_corners = (
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    )
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Quatd(float(pose[3]), Gf.Vec3d(float(pose[4]), float(pose[5]), float(pose[6]))))
    matrix.SetTranslateOnly(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
    vertices = []
    for corner in local_corners:
        point = matrix.Transform(Gf.Vec3d(*corner))
        vertices.append((float(point[0]), float(point[1]), float(point[2])))
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    ]
    return vertices, faces


def _append_obj_component(
    stream,
    *,
    object_name: str,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    vertex_offset: int,
) -> int:
    stream.write(f"o {object_name}\n")
    for vertex in vertices:
        stream.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
    for face in faces:
        stream.write(
            f"f {face[0] + 1 + vertex_offset} {face[1] + 1 + vertex_offset} {face[2] + 1 + vertex_offset}\n"
        )
    return vertex_offset + len(vertices)


def _cluster_mesh_vertices(
    vertices_np: np.ndarray,
    faces_np: np.ndarray,
    *,
    voxel_size: float,
    protected_vertices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0 or len(faces_np) == 0:
        return vertices_np, faces_np
    mesh_min = vertices_np.min(axis=0)
    keys = np.floor((vertices_np - mesh_min) / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    if protected_vertices is not None and len(protected_vertices) == len(vertices_np) and np.any(protected_vertices):
        centroid = vertices_np.mean(axis=0)
        clustered_vertices = np.zeros((len(counts), 3), dtype=np.float64)
        protected_vertices = np.asarray(protected_vertices, dtype=bool)
        for cluster_idx in range(len(counts)):
            vertex_indices = np.flatnonzero(inverse == cluster_idx)
            protected_indices = vertex_indices[protected_vertices[vertex_indices]]
            if len(protected_indices) > 0:
                offsets = vertices_np[protected_indices] - centroid
                selected = protected_indices[int(np.argmax(np.einsum("ij,ij->i", offsets, offsets)))]
                clustered_vertices[cluster_idx] = vertices_np[selected]
            else:
                clustered_vertices[cluster_idx] = vertices_np[vertex_indices].mean(axis=0)
    else:
        clustered_vertices = np.zeros((len(counts), 3), dtype=np.float64)
        np.add.at(clustered_vertices, inverse, vertices_np)
        clustered_vertices /= counts[:, None]

    clustered_faces = inverse[faces_np]
    valid = (
        (clustered_faces[:, 0] != clustered_faces[:, 1])
        & (clustered_faces[:, 1] != clustered_faces[:, 2])
        & (clustered_faces[:, 0] != clustered_faces[:, 2])
    )
    clustered_faces = clustered_faces[valid]
    if len(clustered_faces) == 0:
        return vertices_np, faces_np

    _, unique_indices = np.unique(np.sort(clustered_faces, axis=1), axis=0, return_index=True)
    clustered_faces = clustered_faces[np.sort(unique_indices)]
    used_vertices, remapped_faces = np.unique(clustered_faces.reshape(-1), return_inverse=True)
    remapped_faces = remapped_faces.reshape(-1, 3)
    return clustered_vertices[used_vertices], remapped_faces.astype(np.int64)


def _outline_protected_vertex_mask(
    vertices_np: np.ndarray,
    faces_np: np.ndarray,
    *,
    feature_angle_deg: float,
) -> np.ndarray:
    protected = np.zeros(len(vertices_np), dtype=bool)
    if len(vertices_np) == 0 or len(faces_np) == 0:
        return protected

    triangles = vertices_np[faces_np]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_norms = np.linalg.norm(normals, axis=1)
    valid_normals = normal_norms > 1e-12
    normals[valid_normals] /= normal_norms[valid_normals, None]

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_idx, face in enumerate(faces_np):
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        for edge in ((a, b), (b, c), (c, a)):
            key = (edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])
            edge_faces.setdefault(key, []).append(face_idx)

    cos_threshold = math.cos(math.radians(max(0.0, min(180.0, float(feature_angle_deg)))))
    for (v0, v1), adjacent_faces in edge_faces.items():
        mark = len(adjacent_faces) == 1
        if len(adjacent_faces) == 2:
            n0 = normals[adjacent_faces[0]]
            n1 = normals[adjacent_faces[1]]
            if float(np.dot(n0, n1)) < cos_threshold:
                mark = True
        if mark:
            protected[v0] = True
            protected[v1] = True

    directions = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
        (0.0, 1.0, 1.0),
        (0.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, -1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ]
    for direction in directions:
        direction_np = np.asarray(direction, dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction_np))
        if direction_norm <= 0.0:
            continue
        direction_np /= direction_norm
        projections = vertices_np @ direction_np
        protected[int(np.argmax(projections))] = True
        protected[int(np.argmin(projections))] = True
    return protected


def _simplify_mesh_with_vertex_clustering(
    *,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    target_faces: int,
    preserve_outline: bool,
    feature_angle_deg: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], str]:
    if target_faces <= 0 or len(faces) <= target_faces:
        return vertices, faces, "not_needed"
    vertices_np = np.asarray(vertices, dtype=np.float64)
    faces_np = np.asarray(faces, dtype=np.int64)
    protected_vertices = (
        _outline_protected_vertex_mask(
            vertices_np,
            faces_np,
            feature_angle_deg=feature_angle_deg,
        )
        if preserve_outline
        else None
    )
    extent = vertices_np.max(axis=0) - vertices_np.min(axis=0)
    max_extent = float(np.max(extent))
    if max_extent <= 0.0:
        return vertices, faces, "vertex_cluster_degenerate_extent"

    upper = max_extent * 0.001
    best_vertices = vertices_np
    best_faces = faces_np
    for _ in range(30):
        candidate_vertices, candidate_faces = _cluster_mesh_vertices(
            vertices_np,
            faces_np,
            voxel_size=upper,
            protected_vertices=protected_vertices,
        )
        if 0 < len(candidate_faces) <= target_faces:
            best_vertices, best_faces = candidate_vertices, candidate_faces
            break
        if upper >= max_extent:
            if len(candidate_faces) > 0 and len(candidate_faces) < len(best_faces):
                best_vertices, best_faces = candidate_vertices, candidate_faces
            break
        upper *= 1.8

    lower = 0.0
    for _ in range(20):
        mid = (lower + upper) * 0.5
        candidate_vertices, candidate_faces = _cluster_mesh_vertices(
            vertices_np,
            faces_np,
            voxel_size=mid,
            protected_vertices=protected_vertices,
        )
        if 0 < len(candidate_faces) <= target_faces:
            best_vertices, best_faces = candidate_vertices, candidate_faces
            upper = mid
        else:
            lower = mid
    simplified_vertices = [tuple(float(x) for x in vertex) for vertex in best_vertices]
    simplified_faces = [tuple(int(x) for x in face) for face in best_faces]
    method = "outline_vertex_cluster" if preserve_outline else "vertex_cluster"
    return simplified_vertices, simplified_faces, method


def _simplify_mesh_with_trimesh(
    *,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    target_faces: int,
    preserve_outline: bool,
    feature_angle_deg: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], str]:
    if target_faces <= 0 or len(faces) <= target_faces:
        return vertices, faces, "not_needed"
    try:
        import trimesh
    except Exception as exc:
        clustered_vertices, clustered_faces, clustered_method = _simplify_mesh_with_vertex_clustering(
            vertices=vertices,
            faces=faces,
            target_faces=target_faces,
            preserve_outline=preserve_outline,
            feature_angle_deg=feature_angle_deg,
        )
        return clustered_vertices, clustered_faces, f"{clustered_method}_after_trimesh_unavailable:{type(exc).__name__}"

    try:
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces), process=False)
        if not hasattr(mesh, "simplify_quadric_decimation"):
            clustered_vertices, clustered_faces, clustered_method = _simplify_mesh_with_vertex_clustering(
                vertices=vertices,
                faces=faces,
                target_faces=target_faces,
                preserve_outline=preserve_outline,
                feature_angle_deg=feature_angle_deg,
            )
            return clustered_vertices, clustered_faces, f"{clustered_method}_after_trimesh_missing_quadric"
        try:
            simplified = mesh.simplify_quadric_decimation(face_count=int(target_faces))
        except TypeError:
            simplified = mesh.simplify_quadric_decimation(int(target_faces))
        if simplified is None or len(simplified.faces) == 0 or len(simplified.vertices) == 0:
            clustered_vertices, clustered_faces, clustered_method = _simplify_mesh_with_vertex_clustering(
                vertices=vertices,
                faces=faces,
                target_faces=target_faces,
                preserve_outline=preserve_outline,
                feature_angle_deg=feature_angle_deg,
            )
            return clustered_vertices, clustered_faces, f"{clustered_method}_after_trimesh_empty"
        simplified_vertices = [tuple(float(x) for x in vertex) for vertex in np.asarray(simplified.vertices)]
        simplified_faces = [tuple(int(x) for x in face) for face in np.asarray(simplified.faces)]
        return simplified_vertices, simplified_faces, "trimesh_quadric"
    except Exception as exc:
        clustered_vertices, clustered_faces, clustered_method = _simplify_mesh_with_vertex_clustering(
            vertices=vertices,
            faces=faces,
            target_faces=target_faces,
            preserve_outline=preserve_outline,
            feature_angle_deg=feature_angle_deg,
        )
        return clustered_vertices, clustered_faces, f"{clustered_method}_after_trimesh_failed:{type(exc).__name__}"


def _combine_mesh_components(
    components: list[dict[str, object]],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for component in components:
        component_vertices = component["vertices"]
        component_faces = component["faces"]
        vertex_offset = len(vertices)
        vertices.extend(component_vertices)
        faces.extend(
            (face[0] + vertex_offset, face[1] + vertex_offset, face[2] + vertex_offset)
            for face in component_faces
        )
    return vertices, faces


def _simplify_toilet_mesh_components(
    components: list[dict[str, object]],
    *,
    target_faces: int,
    preserve_lid_seat: bool,
    lid_seat_target_faces: int,
    preserve_outline: bool,
    feature_angle_deg: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], dict[str, object]]:
    lid_seat_components = [component for component in components if bool(component.get("is_lid_seat"))]
    ceramic_components = [component for component in components if not bool(component.get("is_lid_seat"))]
    raw_lid_vertices, raw_lid_faces = _combine_mesh_components(lid_seat_components)
    raw_ceramic_vertices, raw_ceramic_faces = _combine_mesh_components(ceramic_components)

    lid_method = "none"
    if preserve_lid_seat or not raw_lid_faces:
        lid_vertices, lid_faces = raw_lid_vertices, raw_lid_faces
        lid_method = "preserved" if raw_lid_faces else "empty"
    else:
        lid_vertices, lid_faces, lid_method = _simplify_mesh_with_trimesh(
            vertices=raw_lid_vertices,
            faces=raw_lid_faces,
            target_faces=max(1, min(int(lid_seat_target_faces), int(target_faces))),
            preserve_outline=preserve_outline,
            feature_angle_deg=feature_angle_deg,
        )

    ceramic_target_faces = max(1, int(target_faces) - len(lid_faces))
    ceramic_method = "none"
    if not raw_ceramic_faces:
        ceramic_vertices, ceramic_faces = raw_ceramic_vertices, raw_ceramic_faces
        ceramic_method = "empty"
    else:
        ceramic_vertices, ceramic_faces, ceramic_method = _simplify_mesh_with_trimesh(
            vertices=raw_ceramic_vertices,
            faces=raw_ceramic_faces,
            target_faces=ceramic_target_faces,
            preserve_outline=preserve_outline,
            feature_angle_deg=feature_angle_deg,
        )

    vertices, faces = _combine_mesh_components(
        [
            {"vertices": lid_vertices, "faces": lid_faces},
            {"vertices": ceramic_vertices, "faces": ceramic_faces},
        ]
    )
    return vertices, faces, {
        "method": f"lid_seat={lid_method};ceramic={ceramic_method}",
        "target_faces": int(target_faces),
        "lid_seat_target_faces": int(lid_seat_target_faces),
        "preserve_lid_seat": bool(preserve_lid_seat),
        "preserve_outline": bool(preserve_outline),
        "feature_angle_deg": float(feature_angle_deg),
        "lid_seat_input_faces": len(raw_lid_faces),
        "lid_seat_output_faces": len(lid_faces),
        "ceramic_input_faces": len(raw_ceramic_faces),
        "ceramic_output_faces": len(ceramic_faces),
    }


def _toilet_lid_or_seat_name_from_prim(prim) -> str | None:
    path = str(prim.GetPath())
    for movable_name in ("ToiletSeat", "ToiletLid"):
        if movable_name in path:
            return movable_name
    return None


def _find_named_ancestor_prim(prim, name: str):
    current = prim
    while current.IsValid():
        if current.GetName() == name:
            return current
        current = current.GetParent()
    return None


def _gf_matrix_from_pose_w(pos_w, quat_wxyz):
    from pxr import Gf

    pos = pos_w.detach().cpu().tolist() if hasattr(pos_w, "detach") else pos_w
    quat = quat_wxyz.detach().cpu().tolist() if hasattr(quat_wxyz, "detach") else quat_wxyz
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Quatd(float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3]))))
    matrix.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    return matrix


def _matrix_to_curobo_pose_list(matrix) -> list[float]:
    translation = matrix.ExtractTranslation()
    rotation = matrix.RemoveScaleShear().ExtractRotation()
    quat = rotation.GetQuaternion()
    imaginary = quat.GetImaginary()
    return [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        float(quat.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    ]
