#!/usr/bin/env python3
"""Generate sphere-only collision URDFs from motion_wheel_arm_simple_urdf."""

from __future__ import annotations

import math
import os
from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from trimesh.proximity import signed_distance


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent / "motion_wheel_arm_simple_urdf"
OUTPUT_DIR = SCRIPT_DIR
SPHERE_MESH_DIR = OUTPUT_DIR / "meshes"
SPHERE_MESH_PATH = SPHERE_MESH_DIR / "unit_sphere.stl"
CYLINDER_MESH_PATH = SPHERE_MESH_DIR / "unit_cylinder.stl"
BASE_SURFACE_SPHERE_RADIUS = 0.05
MIN_BASE_SURFACE_SPHERE_RADIUS = 0.005
BASE_SURFACE_SAMPLE_SPACING = 0.065
BASE_SURFACE_INSET = 0.001
GROOVE_SURFACE_CLEARANCE = 0.030
ARM_SWEEP_CLEARANCE = 0.100
SPHERE_CLEARANCE = 0.015
MIN_ADJUSTABLE_SPHERE_RADIUS = 0.00005
ARM_BASE_EXCLUSION_LINKS = {
    "XMS5-R800-W4G3B4C_base",
    "XMS5-R800-W4G3B4C_link1",
    "XMS5-R800-W4G3B4C_link2",
}
ARM_BASE_EXCLUSION_MARGIN = 0.03
MECANUM_BASE_EXCLUSION_MARGIN = 0.04
CHASSIS_CYLINDER_RADIUS = 0.015
CHASSIS_CYLINDER_LENGTH = 0.120
CHASSIS_CYLINDER_COLLISIONS = (
    ("manual_cylinder_mesh_01_front_face", (0.235, 0.000, 0.300), (-1.57079632679, 0.0, 0.0)),
    ("manual_cylinder_mesh_02_rear_face", (-0.235, 0.000, 0.300), (-1.57079632679, 0.0, 0.0)),
    ("manual_cylinder_mesh_03_left_face", (0.000, 0.185, 0.300), (0.0, 1.57079632679, 0.0)),
    ("manual_cylinder_mesh_04_right_face", (0.000, -0.185, 0.300), (0.0, 1.57079632679, 0.0)),
    ("manual_cylinder_mesh_05_top_face", (0.000, 0.000, 0.950), (0.0, 1.57079632679, 0.0)),
    ("manual_cylinder_mesh_06_bottom_face", (0.000, 0.000, 0.230), (0.0, 1.57079632679, 0.0)),
)
GROOVE_LOCAL_BOXES = (
    (
        np.asarray([-0.1463, -0.1478, 0.1766], dtype=float),
        np.asarray([0.0920, 0.0948, 0.4920], dtype=float),
    ),
    (
        np.asarray([-0.0740, -0.0733, -0.1545], dtype=float),
        np.asarray([0.0737, 0.0733, 0.1769], dtype=float),
    ),
)
GROOVE_FACET_INDICES = frozenset(range(6, 14))
EPS = 1e-9


def indent(elem: ET.Element, level: int = 0) -> None:
    newline = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = newline + "    "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = newline
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = newline


def parse_vec(text: str | None, fallback: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(fallback, dtype=float)
    values = np.fromstring(text, sep=" ")
    if values.size != 3:
        return np.asarray(fallback, dtype=float)
    return values.astype(float)


def format_vec(values: np.ndarray | tuple[float, float, float] | list[float]) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def origin_element_transform(origin: ET.Element | None) -> np.ndarray:
    attrib = origin.attrib if origin is not None else {}
    xyz = parse_vec(attrib.get("xyz"))
    rpy = parse_vec(attrib.get("rpy"))
    transform = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    transform[:3, 3] = xyz
    return transform


def origin_transform(collision: ET.Element) -> np.ndarray:
    return origin_element_transform(collision.find("origin"))


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.ones(4, dtype=float)
    homogeneous[:3] = point
    return (transform @ homogeneous)[:3]


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=float)))
    return (transform @ homogeneous.T).T[:, :3]


def sphere_mesh_filename() -> str:
    return os.path.relpath(SPHERE_MESH_PATH, start=OUTPUT_DIR).replace(os.sep, "/")


def cylinder_mesh_filename() -> str:
    return os.path.relpath(CYLINDER_MESH_PATH, start=OUTPUT_DIR).replace(os.sep, "/")


def ensure_sphere_mesh() -> None:
    SPHERE_MESH_DIR.mkdir(parents=True, exist_ok=True)
    if SPHERE_MESH_PATH.exists():
        return
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    sphere.export(SPHERE_MESH_PATH)


def ensure_cylinder_mesh() -> None:
    SPHERE_MESH_DIR.mkdir(parents=True, exist_ok=True)
    if CYLINDER_MESH_PATH.exists():
        return
    cylinder = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=64)
    cylinder.export(CYLINDER_MESH_PATH)


def sphere_mesh_collision(name: str, center: np.ndarray, radius: float) -> ET.Element:
    collision = ET.Element("collision", {"name": name})
    ET.SubElement(collision, "origin", {"xyz": format_vec(center), "rpy": "0.000000 0.000000 0.000000"})
    geometry = ET.SubElement(collision, "geometry")
    scale = f"{radius:.6f} {radius:.6f} {radius:.6f}"
    ET.SubElement(geometry, "mesh", {"filename": sphere_mesh_filename(), "scale": scale})
    return collision


def cylinder_mesh_collision(
    name: str,
    center: np.ndarray,
    rpy: np.ndarray,
    radius: float,
    length: float,
) -> ET.Element:
    collision = ET.Element("collision", {"name": name})
    ET.SubElement(collision, "origin", {"xyz": format_vec(center), "rpy": format_vec(rpy)})
    geometry = ET.SubElement(collision, "geometry")
    scale = f"{radius:.6f} {radius:.6f} {length:.6f}"
    ET.SubElement(geometry, "mesh", {"filename": cylinder_mesh_filename(), "scale": scale})
    return collision


def axis_positions(length: float, radius: float) -> list[float]:
    if length <= 2.0 * radius + EPS:
        return [0.0]
    lower = -0.5 * length + radius
    upper = 0.5 * length - radius
    min_spacing = 2.0 * radius + SPHERE_CLEARANCE
    intervals = int(math.floor((upper - lower) / min_spacing))
    if intervals <= 0:
        return [0.5 * (lower + upper)]
    return np.linspace(lower, upper, intervals + 1).tolist()


def bounded_positions(minimum: float, maximum: float, radius: float) -> list[float]:
    lower = minimum + radius
    upper = maximum - radius
    if upper <= lower + EPS:
        return [0.5 * (minimum + maximum)]
    min_spacing = 2.0 * radius + SPHERE_CLEARANCE
    intervals = int(math.floor((upper - lower) / min_spacing))
    if intervals <= 0:
        return [0.5 * (lower + upper)]
    return np.linspace(lower, upper, intervals + 1).tolist()


def cylinder_to_spheres(link_name: str, collision_index: int, collision: ET.Element, cylinder: ET.Element) -> list[ET.Element]:
    cylinder_radius = float(cylinder.attrib["radius"])
    length = float(cylinder.attrib["length"])
    radius = min(cylinder_radius, 0.5 * length)
    transform = origin_transform(collision)
    spheres = []

    for sphere_index, z_value in enumerate(axis_positions(length, radius), start=1):
        center = transform_point(transform, np.asarray([0.0, 0.0, z_value], dtype=float))
        name = f"{link_name}_collision_{collision_index:02d}_cylinder_sphere_{sphere_index:03d}"
        spheres.append(sphere_mesh_collision(name, center, radius))

    return spheres


def box_to_spheres(link_name: str, collision_index: int, collision: ET.Element, box: ET.Element) -> list[ET.Element]:
    size = parse_vec(box.attrib.get("size"))
    radius = 0.5 * float(np.min(size))
    transform = origin_transform(collision)
    axes = [axis_positions(float(length), radius) for length in size]
    spheres = []

    for sphere_index, coords in enumerate(product(*axes), start=1):
        center = transform_point(transform, np.asarray(coords, dtype=float))
        name = f"{link_name}_collision_{collision_index:02d}_box_sphere_{sphere_index:03d}"
        spheres.append(sphere_mesh_collision(name, center, radius))

    return spheres


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if meshes:
            return trimesh.util.concatenate(meshes)
    raise TypeError(f"unsupported mesh type {type(loaded)!r} for {path}")


def mesh_vertices(mesh: ET.Element, urdf_path: Path) -> np.ndarray:
    filename = mesh.attrib.get("filename")
    if not filename:
        raise ValueError(f"collision mesh in {urdf_path.name} has no filename")
    mesh_path = (urdf_path.parent / filename).resolve()
    scale = parse_vec(mesh.attrib.get("scale"), fallback=(1.0, 1.0, 1.0))
    return load_mesh(mesh_path).vertices * scale


def collision_mesh(mesh: ET.Element, urdf_path: Path) -> trimesh.Trimesh:
    filename = mesh.attrib.get("filename")
    if not filename:
        raise ValueError(f"collision mesh in {urdf_path.name} has no filename")
    mesh_path = (urdf_path.parent / filename).resolve()
    scale = parse_vec(mesh.attrib.get("scale"), fallback=(1.0, 1.0, 1.0))
    loaded = load_mesh(mesh_path)
    vertices = loaded.vertices * scale
    return trimesh.Trimesh(vertices=vertices, faces=loaded.faces, process=True)


def visual_mesh_vertices_in_link(visual: ET.Element, urdf_path: Path) -> np.ndarray | None:
    mesh_elem = visual.find("geometry/mesh")
    if mesh_elem is None:
        return None
    filename = mesh_elem.attrib.get("filename")
    if not filename:
        return None
    mesh_path = (urdf_path.parent / filename).resolve()
    scale = parse_vec(mesh_elem.attrib.get("scale"), fallback=(1.0, 1.0, 1.0))
    mesh = load_mesh(mesh_path)
    vertices = mesh.vertices * scale
    return transform_points(origin_element_transform(visual.find("origin")), vertices)


def link_world_transforms(root: ET.Element) -> dict[str, np.ndarray]:
    link_names = [link.attrib["name"] for link in root.findall("link") if "name" in link.attrib]
    children: dict[str, list[tuple[str, np.ndarray]]] = {}
    parents: set[str] = set()

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.attrib["link"]
        child_name = child.attrib["link"]
        children.setdefault(parent_name, []).append((child_name, origin_element_transform(joint.find("origin"))))
        parents.add(child_name)

    transforms: dict[str, np.ndarray] = {}
    stack: list[str] = []
    for link_name in link_names:
        if link_name not in parents:
            transforms[link_name] = np.eye(4)
            stack.append(link_name)

    while stack:
        parent_name = stack.pop()
        for child_name, joint_transform in children.get(parent_name, []):
            transforms[child_name] = transforms[parent_name] @ joint_transform
            stack.append(child_name)

    return transforms


def exclusion_boxes_in_base_link(root: ET.Element, urdf_path: Path) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
    transforms = link_world_transforms(root)
    if "base_link" not in transforms:
        return []

    base_inverse = np.linalg.inv(transforms["base_link"])
    boxes: list[tuple[np.ndarray, np.ndarray, float, str]] = []
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        margin: float | None = None
        if link_name in ARM_BASE_EXCLUSION_LINKS:
            margin = ARM_BASE_EXCLUSION_MARGIN
        elif link_name.startswith("wheel_"):
            margin = MECANUM_BASE_EXCLUSION_MARGIN

        if margin is None or link_name not in transforms:
            continue

        vertices_by_visual = []
        for visual in link.findall("visual"):
            vertices = visual_mesh_vertices_in_link(visual, urdf_path)
            if vertices is not None and len(vertices):
                vertices_by_visual.append(vertices)
        if not vertices_by_visual:
            continue

        vertices_link = np.concatenate(vertices_by_visual, axis=0)
        vertices_base = transform_points(base_inverse @ transforms[link_name], vertices_link)
        boxes.append((vertices_base.min(axis=0), vertices_base.max(axis=0), margin, link_name))

    return boxes


def child_links_by_parent(root: ET.Element) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        children.setdefault(parent.attrib["link"], []).append(child.attrib["link"])
    return children


def descendant_links(root: ET.Element, root_link: str) -> set[str]:
    children = child_links_by_parent(root)
    descendants: set[str] = set()
    stack = [root_link]
    while stack:
        link_name = stack.pop()
        if link_name in descendants:
            continue
        descendants.add(link_name)
        stack.extend(children.get(link_name, []))
    return descendants


def joint_axis_in_base_link(root: ET.Element, joint_name: str = "joint1") -> tuple[np.ndarray, np.ndarray, str] | None:
    transforms = link_world_transforms(root)
    if "base_link" not in transforms:
        return None
    base_inverse = np.linalg.inv(transforms["base_link"])
    for joint in root.findall("joint"):
        if joint.attrib.get("name") != joint_name:
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        if parent is None or child is None:
            return None
        parent_name = parent.attrib["link"]
        if parent_name not in transforms:
            return None
        joint_transform = base_inverse @ transforms[parent_name] @ origin_element_transform(joint.find("origin"))
        axis_local = parse_vec(axis.attrib.get("xyz") if axis is not None else None, fallback=(0.0, 0.0, 1.0))
        axis_base = joint_transform[:3, :3] @ axis_local
        axis_base = axis_base / np.linalg.norm(axis_base)
        return joint_transform[:3, 3], axis_base, child.attrib["link"]
    return None


def collision_spheres_for_source_collision(collision: ET.Element) -> list[tuple[np.ndarray, float]]:
    geometry = collision.find("geometry")
    if geometry is None or len(geometry) == 0:
        return []
    shape = list(geometry)[0]
    transform = origin_transform(collision)

    if shape.tag == "sphere":
        radius = float(shape.attrib["radius"])
        return [(transform_point(transform, np.zeros(3, dtype=float)), radius)]

    if shape.tag == "cylinder":
        cylinder_radius = float(shape.attrib["radius"])
        length = float(shape.attrib["length"])
        radius = min(cylinder_radius, 0.5 * length)
        return [
            (transform_point(transform, np.asarray([0.0, 0.0, z_value], dtype=float)), radius)
            for z_value in axis_positions(length, radius)
        ]

    if shape.tag == "box":
        size = parse_vec(shape.attrib.get("size"))
        radius = 0.5 * float(np.min(size))
        return [
            (transform_point(transform, np.asarray(coords, dtype=float)), radius)
            for coords in product(*(axis_positions(float(length), radius) for length in size))
        ]

    return []


def arm_sweep_constraints_in_base_link(root: ET.Element) -> list[dict[str, object]]:
    transforms = link_world_transforms(root)
    joint_data = joint_axis_in_base_link(root, "joint1")
    if joint_data is None or "base_link" not in transforms:
        return []
    joint_point, joint_axis, joint_child = joint_data
    swept_links = descendant_links(root, joint_child)
    base_inverse = np.linalg.inv(transforms["base_link"])
    constraints: list[dict[str, object]] = []

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        if link_name == "base_link" or link_name.startswith("wheel_") or link_name not in transforms:
            continue
        if not (link_name.startswith("XMS5") or link_name == "link6" or link_name.startswith("tool")):
            continue
        link_transform = base_inverse @ transforms[link_name]
        for collision in link.findall("collision"):
            for center_link, radius in collision_spheres_for_source_collision(collision):
                center_base = transform_point(link_transform, center_link)
                if link_name in swept_links:
                    relative = center_base - joint_point
                    height = float(np.dot(relative, joint_axis))
                    radial = relative - height * joint_axis
                    constraints.append(
                        {
                            "kind": "sweep",
                            "axis_point": joint_point,
                            "axis": joint_axis,
                            "height": height,
                            "radial": float(np.linalg.norm(radial)),
                            "radius": radius,
                            "link": link_name,
                        }
                    )
                else:
                    constraints.append(
                        {
                            "kind": "static",
                            "center": center_base,
                            "radius": radius,
                            "link": link_name,
                        }
                    )

    return constraints


def has_arm_sweep_clearance(center: np.ndarray, radius: float, constraints: list[dict[str, object]]) -> bool:
    for constraint in constraints:
        arm_radius = float(constraint["radius"])
        if constraint["kind"] == "static":
            arm_center = constraint["center"]
            assert isinstance(arm_center, np.ndarray)
            center_distance = float(np.linalg.norm(center - arm_center))
        else:
            axis_point = constraint["axis_point"]
            axis = constraint["axis"]
            assert isinstance(axis_point, np.ndarray)
            assert isinstance(axis, np.ndarray)
            relative = center - axis_point
            point_height = float(np.dot(relative, axis))
            point_radial = float(np.linalg.norm(relative - point_height * axis))
            center_distance = math.sqrt((point_radial - float(constraint["radial"])) ** 2 + (point_height - float(constraint["height"])) ** 2)

        if center_distance < radius + arm_radius + ARM_SWEEP_CLEARANCE - EPS:
            return False
    return True


def sphere_intersects_expanded_box(center: np.ndarray, radius: float, box_min: np.ndarray, box_max: np.ndarray, margin: float) -> bool:
    closest = np.minimum(np.maximum(center, box_min), box_max)
    return float(np.linalg.norm(center - closest)) <= radius + margin + EPS


def distance_to_box(center: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> float:
    closest = np.minimum(np.maximum(center, box_min), box_max)
    return float(np.linalg.norm(center - closest))


def has_protected_clearance(
    center: np.ndarray,
    radius: float,
    exclusion_boxes: list[tuple[np.ndarray, np.ndarray, float, str]],
) -> bool:
    for box_min, box_max, margin, _box_name in exclusion_boxes:
        if distance_to_box(center, box_min, box_max) < radius + margin - EPS:
            return False
    return True


def point_in_box(point: np.ndarray, box_min: np.ndarray, box_max: np.ndarray, padding: float = 0.0) -> bool:
    return bool(np.all(point >= box_min - padding) and np.all(point <= box_max + padding))


def point_in_groove_region(point_local: np.ndarray) -> bool:
    return any(point_in_box(point_local, box_min, box_max, padding=0.025) for box_min, box_max in GROOVE_LOCAL_BOXES)


def point_on_groove_surface(point_local: np.ndarray, facet_index: int) -> bool:
    return facet_index in GROOVE_FACET_INDICES or point_in_groove_region(point_local)


def triangle_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    helper = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u_axis = np.cross(normal, helper)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    return u_axis, v_axis


def point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray) -> bool:
    a, b, c = triangle
    v0 = c - a
    v1 = b - a
    v2 = point - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denominator = dot00 * dot11 - dot01 * dot01
    if abs(denominator) <= EPS:
        return False
    inv_denominator = 1.0 / denominator
    u_value = (dot11 * dot02 - dot01 * dot12) * inv_denominator
    v_value = (dot00 * dot12 - dot01 * dot02) * inv_denominator
    return u_value >= -EPS and v_value >= -EPS and u_value + v_value <= 1.0 + EPS


def sample_triangle_points(triangle: np.ndarray, normal: np.ndarray, spacing: float) -> list[np.ndarray]:
    origin = triangle[0]
    u_axis, v_axis = triangle_basis(normal)
    triangle_2d = np.column_stack(((triangle - origin) @ u_axis, (triangle - origin) @ v_axis))
    min_corner = triangle_2d.min(axis=0)
    max_corner = triangle_2d.max(axis=0)
    points: list[np.ndarray] = []

    x_values = np.arange(min_corner[0] + 0.5 * spacing, max_corner[0] + EPS, spacing)
    y_values = np.arange(min_corner[1] + 0.5 * spacing, max_corner[1] + EPS, spacing)
    for x_value in x_values:
        for y_value in y_values:
            point_2d = np.asarray([x_value, y_value], dtype=float)
            if point_in_triangle_2d(point_2d, triangle_2d):
                points.append(origin + point_2d[0] * u_axis + point_2d[1] * v_axis)

    if not points:
        points.append(triangle.mean(axis=0))
    return points


def collision_sphere_info(collision: ET.Element) -> tuple[np.ndarray, float] | None:
    mesh = collision.find("geometry/mesh")
    if mesh is None or mesh.attrib.get("filename") != sphere_mesh_filename():
        return None
    scale = parse_vec(mesh.attrib.get("scale"))
    radius = float(scale[0])
    origin = collision.find("origin")
    center = parse_vec(origin.attrib.get("xyz") if origin is not None else None)
    return center, radius


def has_sphere_clearance(center: np.ndarray, radius: float, accepted: list[tuple[np.ndarray, float]]) -> bool:
    for other_center, other_radius in accepted:
        distance = float(np.linalg.norm(center - other_center))
        if distance < radius + other_radius + SPHERE_CLEARANCE - EPS:
            return False
    return True


def max_radius_for_clearance(center: np.ndarray, accepted: list[tuple[np.ndarray, float]]) -> float:
    radius = BASE_SURFACE_SPHERE_RADIUS
    for other_center, other_radius in accepted:
        distance = float(np.linalg.norm(center - other_center))
        radius = min(radius, distance - other_radius - SPHERE_CLEARANCE)
    return radius


def max_radius_for_protected_boxes(
    center: np.ndarray,
    exclusion_boxes: list[tuple[np.ndarray, np.ndarray, float, str]],
) -> float:
    radius = BASE_SURFACE_SPHERE_RADIUS
    for box_min, box_max, margin, _box_name in exclusion_boxes:
        radius = min(radius, distance_to_box(center, box_min, box_max) - margin)
    return radius


def base_radius_candidates(max_radius: float, min_radius: float) -> list[float]:
    ratios = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.15, 0.1)
    candidates = [max(min_radius, max_radius * ratio) for ratio in ratios]
    candidates.append(min_radius)
    deduped: list[float] = []
    for radius in candidates:
        radius = min(max_radius, max(min_radius, radius))
        if all(abs(radius - existing) > 1e-6 for existing in deduped):
            deduped.append(radius)
    return deduped


def acceptable_base_surface_sphere(
    surface_mesh: trimesh.Trimesh,
    transform: np.ndarray,
    surface_point: np.ndarray,
    normal: np.ndarray,
    max_radius: float,
    min_radius: float,
    surface_clearance: float,
    accepted: list[tuple[np.ndarray, float]],
    exclusion_boxes: list[tuple[np.ndarray, np.ndarray, float, str]],
    arm_sweep_constraints: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, float] | None:
    for test_radius in base_radius_candidates(max_radius, min_radius):
        for direction in (-normal, normal):
            local_center = surface_point + direction * (test_radius + surface_clearance + BASE_SURFACE_INSET)
            center = transform_point(transform, local_center)
            distance = float(signed_distance(surface_mesh, np.asarray([local_center], dtype=float))[0])
            if distance < test_radius + surface_clearance + 0.5 * BASE_SURFACE_INSET:
                continue
            if not has_sphere_clearance(local_center, test_radius, accepted):
                continue
            if not has_protected_clearance(center, test_radius, exclusion_boxes):
                continue
            if not has_arm_sweep_clearance(center, test_radius, arm_sweep_constraints):
                continue
            return local_center, center, test_radius
    return None


def mesh_surface_to_spheres(
    link_name: str,
    collision_index: int,
    collision: ET.Element,
    mesh: ET.Element,
    urdf_path: Path,
    exclusion_boxes: list[tuple[np.ndarray, np.ndarray, float, str]],
    arm_sweep_constraints: list[dict[str, object]],
) -> list[ET.Element]:
    surface_mesh = collision_mesh(mesh, urdf_path)
    radius = min(BASE_SURFACE_SPHERE_RADIUS, 0.5 * float(np.min(surface_mesh.extents)))
    min_radius = min(MIN_BASE_SURFACE_SPHERE_RADIUS, radius)
    transform = origin_transform(collision)
    face_to_facet = {
        int(face_index): facet_index
        for facet_index, face_indices in enumerate(surface_mesh.facets)
        for face_index in face_indices
    }
    candidates: list[tuple[np.ndarray, np.ndarray, int]] = []

    for face_index, (triangle, normal) in enumerate(zip(surface_mesh.triangles, surface_mesh.face_normals)):
        facet_index = face_to_facet.get(face_index, -1)
        for surface_point in sample_triangle_points(triangle, normal, BASE_SURFACE_SAMPLE_SPACING):
            candidates.append((surface_point, normal, facet_index))

    spheres: list[ET.Element] = []
    accepted: list[tuple[np.ndarray, float]] = []
    accepted_facets: set[int] = set()
    pending_fallback_facets: set[int] = set(range(len(surface_mesh.facets)))
    for surface_point, normal, facet_index in candidates:
        candidate_radius = radius
        local_center = surface_point - normal * (candidate_radius + BASE_SURFACE_INSET)
        center = transform_point(transform, local_center)
        surface_clearance = (
            GROOVE_SURFACE_CLEARANCE
            if link_name == "base_link" and point_on_groove_surface(surface_point, facet_index)
            else 0.0
        )

        if link_name == "base_link":
            accepted_candidate = acceptable_base_surface_sphere(
                surface_mesh,
                transform,
                surface_point,
                normal,
                radius,
                min_radius,
                surface_clearance,
                accepted,
                exclusion_boxes,
                arm_sweep_constraints,
            )
            if accepted_candidate is None:
                continue
            local_center, center, candidate_radius = accepted_candidate

        elif not has_sphere_clearance(local_center, candidate_radius, accepted):
            continue

        accepted.append((local_center, candidate_radius))
        if facet_index >= 0:
            accepted_facets.add(facet_index)
            pending_fallback_facets.discard(facet_index)
        name = f"{link_name}_collision_{collision_index:02d}_mesh_surface_sphere_{len(spheres) + 1:03d}"
        spheres.append(sphere_mesh_collision(name, center, candidate_radius))

    if link_name == "base_link" and pending_fallback_facets:
        for facet_index in sorted(pending_fallback_facets):
            fallback_candidates: list[tuple[np.ndarray, np.ndarray]] = []
            for face_index in surface_mesh.facets[facet_index]:
                triangle = surface_mesh.triangles[int(face_index)]
                normal = surface_mesh.face_normals[int(face_index)]
                fallback_candidates.extend(
                    (surface_point, normal)
                    for surface_point in sample_triangle_points(triangle, normal, 2.0 * min_radius + SPHERE_CLEARANCE)
                )

            fallback_candidates.sort(key=lambda item: float(np.linalg.norm(item[0] - surface_mesh.centroid)), reverse=True)
            for surface_point, normal in fallback_candidates:
                accepted_candidate = acceptable_base_surface_sphere(
                    surface_mesh,
                    transform,
                    surface_point,
                    normal,
                    radius,
                    min_radius,
                    GROOVE_SURFACE_CLEARANCE if point_on_groove_surface(surface_point, facet_index) else 0.0,
                    accepted,
                    exclusion_boxes,
                    arm_sweep_constraints,
                )
                if accepted_candidate is None:
                    continue
                local_center, center, candidate_radius = accepted_candidate
                accepted.append((local_center, candidate_radius))
                accepted_facets.add(facet_index)
                name = f"{link_name}_collision_{collision_index:02d}_mesh_surface_sphere_{len(spheres) + 1:03d}"
                spheres.append(sphere_mesh_collision(name, center, candidate_radius))
                break

    return spheres


def primitive_sphere_to_mesh(link_name: str, collision_index: int, collision: ET.Element, sphere: ET.Element) -> list[ET.Element]:
    radius = float(sphere.attrib["radius"])
    center = transform_point(origin_transform(collision), np.zeros(3, dtype=float))
    name = collision.attrib.get("name") or f"{link_name}_collision_{collision_index:02d}_original_sphere_mesh"
    return [sphere_mesh_collision(name, center, radius)]


def converted_collisions(
    link_name: str,
    collision_index: int,
    collision: ET.Element,
    urdf_path: Path,
    exclusion_boxes: list[tuple[np.ndarray, np.ndarray, float, str]],
    arm_sweep_constraints: list[dict[str, object]],
) -> list[ET.Element]:
    geometry = collision.find("geometry")
    if geometry is None or len(geometry) == 0:
        return [collision]

    shape = list(geometry)[0]
    if shape.tag == "sphere":
        return primitive_sphere_to_mesh(link_name, collision_index, collision, shape)
    if shape.tag == "cylinder":
        return cylinder_to_spheres(link_name, collision_index, collision, shape)
    if shape.tag == "box":
        return box_to_spheres(link_name, collision_index, collision, shape)
    if shape.tag == "mesh":
        return mesh_surface_to_spheres(
            link_name,
            collision_index,
            collision,
            shape,
            urdf_path,
            exclusion_boxes,
            arm_sweep_constraints,
        )

    raise ValueError(f"unsupported collision geometry {shape.tag!r} in {urdf_path.name}")


def set_sphere_mesh_radius(mesh: ET.Element, radius: float) -> None:
    scale = f"{radius:.6f} {radius:.6f} {radius:.6f}"
    mesh.attrib["scale"] = scale


def collect_sphere_records(root: ET.Element) -> list[dict[str, object]]:
    transforms = link_world_transforms(root)
    records: list[dict[str, object]] = []

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        link_transform = transforms.get(link_name, np.eye(4))
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None or mesh.attrib.get("filename") != sphere_mesh_filename():
                continue
            radius = float(parse_vec(mesh.attrib.get("scale"))[0])
            center = transform_point(link_transform @ origin_transform(collision), np.zeros(3, dtype=float))
            records.append(
                {
                    "link": link_name,
                    "mesh": mesh,
                    "center": center,
                    "radius": radius,
                    "min_radius": MIN_BASE_SURFACE_SPHERE_RADIUS if link_name == "base_link" else MIN_ADJUSTABLE_SPHERE_RADIUS,
                }
            )

    return records


def shrink_record(record: dict[str, object], reduction: float) -> float:
    radius = float(record["radius"])
    min_radius = float(record["min_radius"])
    new_radius = max(min_radius, radius - reduction)
    actual_reduction = radius - new_radius
    record["radius"] = new_radius
    return actual_reduction


def enforce_global_sphere_clearance(root: ET.Element) -> None:
    records = collect_sphere_records(root)

    for _iteration in range(100):
        changed = False
        for index, first in enumerate(records):
            first_center = first["center"]
            assert isinstance(first_center, np.ndarray)
            first_radius = float(first["radius"])
            for second in records[index + 1 :]:
                second_center = second["center"]
                assert isinstance(second_center, np.ndarray)
                second_radius = float(second["radius"])
                distance = float(np.linalg.norm(first_center - second_center))
                allowed_radius_sum = distance - SPHERE_CLEARANCE
                if first_radius + second_radius <= allowed_radius_sum + EPS:
                    continue

                excess = first_radius + second_radius - allowed_radius_sum
                adjustable = [
                    record
                    for record in (first, second)
                    if float(record["radius"]) > float(record["min_radius"]) + EPS
                ]
                if not adjustable:
                    continue

                available = [float(record["radius"]) - float(record["min_radius"]) for record in adjustable]
                available_total = sum(available)
                if available_total <= EPS:
                    continue

                remaining = excess
                for record, record_available in zip(adjustable, available):
                    if remaining <= EPS:
                        break
                    share = min(record_available, excess * record_available / available_total)
                    actual = shrink_record(record, share)
                    remaining -= actual
                    changed = changed or actual > EPS

                first_radius = float(first["radius"])
                second_radius = float(second["radius"])

        if not changed:
            break

    for record in records:
        mesh = record["mesh"]
        assert isinstance(mesh, ET.Element)
        set_sphere_mesh_radius(mesh, float(record["radius"]))


def append_chassis_cylinders(root: ET.Element) -> None:
    link = root.find("./link[@name='base_link']")
    if link is None:
        return

    link_name = link.attrib.get("name", "base_link")
    for suffix, xyz, rpy in CHASSIS_CYLINDER_COLLISIONS:
        name = f"{link_name}_{suffix}"
        link.append(
            cylinder_mesh_collision(
                name,
                np.asarray(xyz, dtype=float),
                np.asarray(rpy, dtype=float),
                CHASSIS_CYLINDER_RADIUS,
                CHASSIS_CYLINDER_LENGTH,
            )
        )


def rewrite_tree(root: ET.Element, urdf_path: Path) -> ET.Element:
    exclusion_boxes = exclusion_boxes_in_base_link(root, urdf_path)
    arm_sweep_constraints = arm_sweep_constraints_in_base_link(root)

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "link")
        new_children = []
        collision_index = 0

        for child in list(link):
            if child.tag != "collision":
                new_children.append(child)
                continue
            collision_index += 1
            new_children.extend(
                converted_collisions(link_name, collision_index, child, urdf_path, exclusion_boxes, arm_sweep_constraints)
            )

        link[:] = new_children

    enforce_global_sphere_clearance(root)
    append_chassis_cylinders(root)
    return root


def main() -> None:
    ensure_sphere_mesh()
    ensure_cylinder_mesh()

    generated = []
    for src_path in sorted(SOURCE_DIR.glob("*.urdf")):
        tree = ET.parse(src_path)
        root = rewrite_tree(tree.getroot(), src_path)
        indent(root)
        dst_path = OUTPUT_DIR / src_path.name
        ET.ElementTree(root).write(dst_path, encoding="utf-8", xml_declaration=True)
        generated.append(dst_path.name)

    print(f"generated {len(generated)} urdf files in {OUTPUT_DIR}")
    for name in generated:
        print(name)


if __name__ == "__main__":
    main()
