#!/usr/bin/env python3
"""Generate a link2-and-after RobotBuilder sphere-collision URDF.

The public entry point is :func:`generate_link2plus_collision_urdf`.
It reads a URDF, automatically selects the arm branch from ``*link2`` onward,
fits collision spheres from visual meshes with cuRobo RobotBuilder, removes
``tool_tip_link`` collision, and writes a new URDF with only those sphere
collisions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re
import sys
from typing import Any
import xml.etree.ElementTree as ET


DEFAULT_TOOL_TIP_NAMES = ("tool_tip_link", "tool_tip", "tcp_link", "tcp")
DEFAULT_WRIST_LINK_NAME = "link6"
DEFAULT_BASE_LINK_NAME = "base_link"
DEFAULT_BODY_DENSITY = 1.0
DEFAULT_END_EFFECTOR_DENSITY = 5.0
DEFAULT_MOVABLE_END_EFFECTOR_DENSITY = 12.0
DEFAULT_SELF_COLLISION_CLEARANCE_M = 0.001
DEFAULT_MIN_SPHERE_RADIUS_M = 0.001
DEFAULT_BASE_ARM_CLEARANCE_RADIUS_M = 0.2
DEFAULT_BASE_ARM_CLEARANCE_BELOW_M = 0.08
DEFAULT_BASE_ARM_CLEARANCE_ABOVE_M = 0.5


def _ensure_local_curobo_on_path() -> None:
    """Allow running via isaaclab.sh without manually exporting PYTHONPATH."""
    try:
        import curobo  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "src/nvidia-curobo"
        if (candidate / "curobo").is_dir():
            sys.path.insert(0, str(candidate))
            return


@dataclass(frozen=True)
class JointInfo:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float
    upper: float
    mimic_joint: str | None
    mimic_multiplier: float
    mimic_offset: float


@dataclass(frozen=True)
class Link2PlusCollisionUrdfResult:
    output_urdf: Path
    selected_links: list[str]
    end_effector_links: list[str]
    collision_sphere_counts: dict[str, int]
    adjusted_sphere_count: int
    removed_sphere_count: int


def _parse_vec3(text: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if text is None:
        return default
    values = tuple(float(value) for value in text.split())
    if len(values) != 3:
        raise RuntimeError(f"Expected 3 values, got {text!r}")
    return values


def _fmt(value: float) -> str:
    text = f"{float(value):.9g}"
    return "0" if text == "-0" else text


def _find_link(root: ET.Element, link_name: str) -> ET.Element:
    link = root.find(f"link[@name='{link_name}']")
    if link is None:
        raise RuntimeError(f"URDF missing link: {link_name}")
    return link


def _has_visual_geometry(link: ET.Element) -> bool:
    return any(visual.find("geometry") is not None for visual in link.findall("visual"))


def _has_visual_mesh(link: ET.Element) -> bool:
    return any(visual.find("geometry/mesh") is not None for visual in link.findall("visual"))


def _joint_infos(root: ET.Element) -> list[JointInfo]:
    joints: list[JointInfo] = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        origin = joint.find("origin")
        axis = joint.find("axis")
        joints.append(
            JointInfo(
                name=joint.attrib.get("name", ""),
                joint_type=joint.attrib.get("type", ""),
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                origin_xyz=_parse_vec3(origin.attrib.get("xyz") if origin is not None else None),
                origin_rpy=_parse_vec3(origin.attrib.get("rpy") if origin is not None else None),
                axis=_parse_vec3(axis.attrib.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
                lower=float(limit.attrib.get("lower", "0")) if limit is not None else 0.0,
                upper=float(limit.attrib.get("upper", "0")) if limit is not None else 0.0,
                mimic_joint=mimic.attrib.get("joint") if mimic is not None else None,
                mimic_multiplier=float(mimic.attrib.get("multiplier", "1")) if mimic is not None else 1.0,
                mimic_offset=float(mimic.attrib.get("offset", "0")) if mimic is not None else 0.0,
            )
        )
    return joints


def _child_edges(joints: list[JointInfo]) -> dict[str, list[JointInfo]]:
    edges: dict[str, list[JointInfo]] = {}
    for joint in joints:
        edges.setdefault(joint.parent, []).append(joint)
    return edges


def _descendants_from(start_link: str, edges: dict[str, list[JointInfo]]) -> list[str]:
    ordered: list[str] = []
    stack = [start_link]
    seen: set[str] = set()
    while stack:
        link_name = stack.pop(0)
        if link_name in seen:
            continue
        seen.add(link_name)
        ordered.append(link_name)
        for joint in edges.get(link_name, []):
            stack.append(joint.child)
    return ordered


def _score_start_link(link_name: str, descendants: list[str]) -> tuple[int, int]:
    lower = " ".join([link_name, *descendants]).lower()
    ee_score = sum(
        token in lower
        for token in (
            "tool",
            "gripper",
            "finger",
            "hand",
            "ee",
            "tip",
            "tcp",
            "brush",
            "mop",
            "suction",
            "suck",
            "nozzle",
            "pick",
            "swab",
            "sponge",
            "roller",
        )
    )
    return ee_score, len(descendants)


def discover_link2plus_links(
    urdf_path: str | Path,
    *,
    start_link: str | None = None,
    tool_tip_names: tuple[str, ...] = DEFAULT_TOOL_TIP_NAMES,
    require_visual_mesh: bool = True,
) -> list[str]:
    """Discover visual links on the arm branch from ``*link2`` onward.

    Args:
        urdf_path: Source URDF.
        start_link: Optional explicit first link. When omitted, the first link
            whose name ends with ``link2`` is selected, preferring candidates
            with tool/gripper descendants.
        tool_tip_names: Links kept as frames only and excluded from collision.
        require_visual_mesh: When true, keep links with visual ``mesh`` entries.
            This matches RobotBuilder visual-mesh fitting.
    """
    root = ET.parse(Path(urdf_path)).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = _joint_infos(root)
    edges = _child_edges(joints)

    if start_link is None:
        candidates = [
            name
            for name in links
            if re.search(r"(^|[_\-/])link2$", name.lower()) or name.lower().endswith("link2")
        ]
        if not candidates:
            raise RuntimeError("Could not auto-discover a start link ending in 'link2'; pass start_link explicitly.")
        start_link = max(candidates, key=lambda name: _score_start_link(name, _descendants_from(name, edges)))
    if start_link not in links:
        raise RuntimeError(f"start_link={start_link!r} is not in the URDF.")

    excluded = {name.lower() for name in tool_tip_names}
    selected: list[str] = []
    for link_name in _descendants_from(start_link, edges):
        if link_name.lower() in excluded:
            continue
        link = links[link_name]
        has_geometry = _has_visual_mesh(link) if require_visual_mesh else _has_visual_geometry(link)
        if has_geometry:
            selected.append(link_name)
    if not selected:
        raise RuntimeError(f"No visual-mesh links found from start_link={start_link!r}.")
    return selected


def _detect_end_effector_links(
    selected_links: list[str],
    joints: list[JointInfo],
    *,
    wrist_link_name: str = DEFAULT_WRIST_LINK_NAME,
) -> list[str]:
    """Detect tool/end-effector links structurally.

    By default, ``link6`` is treated as the wrist/final arm link.  Any selected
    link attached after ``link6`` and all its selected descendants are considered
    end-effector/tool geometry.  This covers brush, mop, suction, pick, gripper,
    and similar tools without relying on link names.
    """
    selected = set(selected_links)
    edges = _child_edges(joints)
    selected_index = {link: index for index, link in enumerate(selected_links)}
    end_effector: set[str] = set()
    queue: list[str] = []

    if wrist_link_name in selected:
        for joint in edges.get(wrist_link_name, []):
            if joint.child in selected:
                end_effector.add(joint.child)
                queue.append(joint.child)
    else:
        # Fallback for robots that use a different wrist link name: once a
        # selected link is reached through a non-arm joint, treat it as tool
        # geometry.
        for joint in joints:
            if joint.parent not in selected or joint.child not in selected:
                continue
            if joint.joint_type not in ("revolute", "continuous"):
                end_effector.add(joint.child)
                queue.append(joint.child)

    while queue:
        link_name = queue.pop(0)
        for joint in edges.get(link_name, []):
            if joint.child in selected and joint.child not in end_effector:
                end_effector.add(joint.child)
                queue.append(joint.child)

    # Fallback for URDFs that encode the tool as a revolute child or use unusual
    # names: treat terminal tool-like links as end-effector geometry.
    tool_name_pattern = re.compile(
        r"(tool|gripper|finger|hand|end[_-]?effector|brush|mop|suction|suck|nozzle|pick|swab|sponge)"
    )
    for link_name in selected_links:
        if tool_name_pattern.search(link_name.lower()):
            end_effector.add(link_name)

    return sorted(end_effector, key=lambda link: selected_index[link])


def _detect_movable_end_effector_links(selected_links: list[str], joints: list[JointInfo]) -> set[str]:
    selected = set(selected_links)
    return {
        joint.child
        for joint in joints
        if joint.parent in selected
        and joint.child in selected
        and joint.joint_type not in ("fixed", "revolute", "continuous")
    }


def _joint_path_to_ancestor(descendant_link: str, ancestor_link: str, joints: list[JointInfo]) -> list[JointInfo]:
    parent_joint_by_child = {joint.child: joint for joint in joints}
    path: list[JointInfo] = []
    link_name = descendant_link
    seen: set[str] = set()
    while link_name != ancestor_link:
        if link_name in seen:
            raise RuntimeError(f"Cycle detected while tracing {descendant_link!r} to {ancestor_link!r}.")
        seen.add(link_name)
        joint = parent_joint_by_child.get(link_name)
        if joint is None:
            raise RuntimeError(f"Could not trace {descendant_link!r} back to ancestor link {ancestor_link!r}.")
        path.append(joint)
        link_name = joint.parent
    return list(reversed(path))


def _resolve_base_arm_mount_origin(
    joints: list[JointInfo],
    *,
    base_link_name: str,
    arm_start_link: str,
    base_arm_root_joint_name: str | None,
) -> tuple[float, float, float]:
    if base_arm_root_joint_name is not None:
        for joint in joints:
            if joint.name == base_arm_root_joint_name:
                if joint.parent != base_link_name:
                    raise RuntimeError(
                        f"base_arm_root_joint={base_arm_root_joint_name!r} parent is {joint.parent!r}, not {base_link_name!r}."
                    )
                return joint.origin_xyz
        raise RuntimeError(f"URDF missing base_arm_root_joint={base_arm_root_joint_name!r}.")

    path = _joint_path_to_ancestor(arm_start_link, base_link_name, joints)
    if not path:
        raise RuntimeError(f"arm_start_link={arm_start_link!r} is already base_link_name={base_link_name!r}.")
    root_joint = path[0]
    if root_joint.parent != base_link_name:
        raise RuntimeError(f"Could not find an arm mount joint parented by {base_link_name!r}.")
    return root_joint.origin_xyz


def _rpy_matrix(rpy: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _mat_vec_mul(matrix: tuple[tuple[float, float, float], ...], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: tuple[float, float, float], value: float) -> tuple[float, float, float]:
    return (a[0] * value, a[1] * value, a[2] * value)


def _normalize(a: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    if norm <= 0.0:
        return (1.0, 0.0, 0.0)
    return (a[0] / norm, a[1] / norm, a[2] / norm)


def _linspace(lower: float, upper: float, samples: int) -> list[float]:
    count = max(samples, 2)
    if count == 1:
        return [lower]
    step = (upper - lower) / float(count - 1)
    return [lower + step * index for index in range(count)]


def _axis_angle_matrix(axis: tuple[float, float, float], angle: float) -> tuple[tuple[float, float, float], ...]:
    x, y, z = _normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )


def _transform_movable_child_sphere_to_parent(
    center: list[float],
    joint: JointInfo,
    joint_position: float,
) -> tuple[float, float, float]:
    rotation = _rpy_matrix(joint.origin_rpy)
    if joint.joint_type == "prismatic":
        local = _add(tuple(center), _scale(joint.axis, joint_position))
    elif joint.joint_type in ("revolute", "continuous"):
        local = _mat_vec_mul(_axis_angle_matrix(joint.axis, joint_position), tuple(center))
    else:
        local = tuple(center)
    return _add(joint.origin_xyz, _mat_vec_mul(rotation, local))


def _joint_sample_values(joint: JointInfo, joint_by_name: dict[str, JointInfo], samples: int) -> list[float]:
    if joint.mimic_joint and joint.mimic_joint in joint_by_name:
        driver = joint_by_name[joint.mimic_joint]
        if driver.joint_type == "continuous":
            lower, upper = -math.pi, math.pi
        else:
            lower, upper = driver.lower, driver.upper
        values = _linspace(lower, upper, samples)
        return [joint.mimic_multiplier * value + joint.mimic_offset for value in values]
    if joint.joint_type == "continuous":
        return _linspace(-math.pi, math.pi, samples)
    lower, upper = joint.lower, joint.upper
    if math.isclose(lower, upper):
        return [lower]
    return _linspace(lower, upper, samples)


def _correlated_joint_samples(
    joint_a: JointInfo,
    joint_b: JointInfo,
    joint_by_name: dict[str, JointInfo],
    samples: int,
) -> list[tuple[float, float]]:
    if joint_b.mimic_joint == joint_a.name:
        values_a = _joint_sample_values(joint_a, joint_by_name, samples)
        return [(value, joint_b.mimic_multiplier * value + joint_b.mimic_offset) for value in values_a]
    if joint_a.mimic_joint == joint_b.name:
        values_b = _joint_sample_values(joint_b, joint_by_name, samples)
        return [(joint_a.mimic_multiplier * value + joint_a.mimic_offset, value) for value in values_b]
    values_a = _joint_sample_values(joint_a, joint_by_name, samples)
    values_b = _joint_sample_values(joint_b, joint_by_name, samples)
    if len(values_a) == len(values_b):
        return list(zip(values_a, values_b))
    return [(values_a[min(i, len(values_a) - 1)], values_b[min(i, len(values_b) - 1)]) for i in range(max(len(values_a), len(values_b)))]


def _shrink_sphere_pair(
    sphere_a: dict[str, Any],
    sphere_b: dict[str, Any],
    allowed_sum_radius: float,
    *,
    clearance: float,
    min_radius: float,
) -> int:
    target_sum = max(0.0, allowed_sum_radius - clearance)
    current_sum = float(sphere_a["radius"]) + float(sphere_b["radius"])
    if current_sum <= target_sum:
        return 0
    if target_sum <= min_radius * 2.0:
        sphere_a["_remove"] = True
        sphere_b["_remove"] = True
        return 2
    scale = target_sum / current_sum
    new_a = max(min_radius, float(sphere_a["radius"]) * scale)
    new_b = max(min_radius, float(sphere_b["radius"]) * scale)
    changed = int(not math.isclose(float(sphere_a["radius"]), new_a)) + int(not math.isclose(float(sphere_b["radius"]), new_b))
    sphere_a["radius"] = new_a
    sphere_b["radius"] = new_b
    return changed


def _make_end_effector_spheres_motion_safe(
    collision_spheres: dict[str, list[dict[str, Any]]],
    selected_links: list[str],
    end_effector_links: list[str],
    joints: list[JointInfo],
    *,
    sample_count: int,
    clearance: float,
    min_radius: float,
) -> tuple[int, int]:
    """Shrink/remove movable end-effector spheres that would block gripper motion."""
    selected = set(selected_links)
    end_effector = set(end_effector_links)
    movable = [
        joint
        for joint in joints
        if joint.joint_type in ("prismatic", "revolute", "continuous")
        and joint.parent in selected
        and joint.child in selected
        and joint.child in end_effector
    ]
    if not movable:
        return 0, 0
    joint_by_name = {joint.name: joint for joint in joints}
    adjusted = 0

    for idx, joint_a in enumerate(movable):
        for joint_b in movable[idx + 1 :]:
            if joint_a.parent != joint_b.parent:
                continue
            for sphere_a in collision_spheres.get(joint_a.child, []):
                for sphere_b in collision_spheres.get(joint_b.child, []):
                    min_distance = math.inf
                    for qa, qb in _correlated_joint_samples(joint_a, joint_b, joint_by_name, sample_count):
                        center_a = _transform_movable_child_sphere_to_parent(sphere_a["center"], joint_a, qa)
                        center_b = _transform_movable_child_sphere_to_parent(sphere_b["center"], joint_b, qb)
                        min_distance = min(min_distance, math.dist(center_a, center_b))
                    adjusted += _shrink_sphere_pair(
                        sphere_a,
                        sphere_b,
                        min_distance,
                        clearance=clearance,
                        min_radius=min_radius,
                    )

    for joint in movable:
        parent_spheres = collision_spheres.get(joint.parent, [])
        child_spheres = collision_spheres.get(joint.child, [])
        for parent_sphere in parent_spheres:
            parent_center = tuple(parent_sphere["center"])
            for child_sphere in child_spheres:
                min_distance = math.inf
                for q in _joint_sample_values(joint, joint_by_name, sample_count):
                    child_center = _transform_movable_child_sphere_to_parent(child_sphere["center"], joint, q)
                    min_distance = min(min_distance, math.dist(parent_center, child_center))
                adjusted += _shrink_sphere_pair(
                    parent_sphere,
                    child_sphere,
                    min_distance,
                    clearance=clearance,
                    min_radius=min_radius,
                )

    removed = 0
    for link_name, spheres in list(collision_spheres.items()):
        kept = []
        for sphere in spheres:
            if sphere.pop("_remove", False) or float(sphere["radius"]) < min_radius:
                removed += 1
                continue
            sphere["radius"] = float(f"{float(sphere['radius']):.9g}")
            sphere["center"] = [float(f"{float(value):.9g}") for value in sphere["center"]]
            kept.append(sphere)
        collision_spheres[link_name] = kept
    return adjusted, removed


def _make_base_link_spheres_arm_mount_safe(
    collision_spheres: dict[str, list[dict[str, Any]]],
    *,
    base_link_name: str,
    mount_origin: tuple[float, float, float],
    radius: float,
    below: float,
    above: float,
    clearance: float,
    min_radius: float,
) -> tuple[int, int]:
    """Clear base-link spheres out of the arm mount volume."""
    base_spheres = collision_spheres.get(base_link_name, [])
    if not base_spheres:
        return 0, 0

    z_min = mount_origin[2] - below
    z_max = mount_origin[2] + above
    adjusted = 0
    removed = 0
    kept: list[dict[str, Any]] = []

    for sphere in base_spheres:
        center = sphere["center"]
        sphere_radius = float(sphere["radius"])
        radial_distance = math.hypot(float(center[0]) - mount_origin[0], float(center[1]) - mount_origin[1])
        radial_gap = max(0.0, radial_distance - radius)
        vertical_gap = max(z_min - float(center[2]), float(center[2]) - z_max, 0.0)
        allowed_radius = max(0.0, math.hypot(radial_gap, vertical_gap) - clearance)

        if sphere_radius <= allowed_radius:
            sphere["radius"] = float(f"{sphere_radius:.9g}")
            sphere["center"] = [float(f"{float(value):.9g}") for value in center]
            kept.append(sphere)
            continue

        adjusted += 1
        if allowed_radius < min_radius:
            removed += 1
            continue

        sphere["radius"] = float(f"{allowed_radius:.9g}")
        sphere["center"] = [float(f"{float(value):.9g}") for value in center]
        kept.append(sphere)

    collision_spheres[base_link_name] = kept
    return adjusted, removed


def _extract_link_collision_spheres_from_urdf(urdf_path: str | Path, link_name: str) -> list[dict[str, Any]]:
    root = ET.parse(Path(urdf_path).expanduser()).getroot()
    link = _find_link(root, link_name)
    spheres: list[dict[str, Any]] = []

    for collision in link.findall("collision"):
        sphere = collision.find("geometry/sphere")
        if sphere is None:
            continue
        radius_text = sphere.attrib.get("radius")
        if radius_text is None:
            raise RuntimeError(f"Sphere collision on link {link_name!r} in {urdf_path!s} is missing radius.")
        origin = collision.find("origin")
        center = _parse_vec3(origin.attrib.get("xyz") if origin is not None else None)
        spheres.append(
            {
                "center": [float(f"{value:.9g}") for value in center],
                "radius": float(f"{float(radius_text):.9g}"),
            }
        )

    if not spheres:
        raise RuntimeError(f"No sphere collisions found on link {link_name!r} in {urdf_path!s}.")
    return spheres


def _write_sphere_collision_urdf(
    source_urdf: Path,
    output_urdf: Path,
    selected_links: list[str],
    collision_spheres: dict[str, list[dict[str, Any]]],
    *,
    tool_tip_names: tuple[str, ...],
) -> None:
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    tool_tip_set = {name.lower() for name in tool_tip_names}
    for link in root.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)

    for link_name in selected_links:
        if link_name.lower() in tool_tip_set:
            continue
        link = _find_link(root, link_name)
        for index, sphere in enumerate(collision_spheres.get(link_name, []), start=1):
            collision = ET.Element("collision", {"name": f"{link_name}_robotbuilder_sphere_{index:03d}"})
            ET.SubElement(
                collision,
                "origin",
                {
                    "xyz": " ".join(_fmt(value) for value in sphere["center"]),
                    "rpy": "0 0 0",
                },
            )
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "sphere", {"radius": _fmt(sphere["radius"])})
            link.append(collision)

    ET.indent(tree, space="    ")
    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)


def generate_link2plus_collision_urdf(
    urdf_path: str | Path,
    output_urdf_path: str | Path | None = None,
    *,
    start_link: str | None = None,
    tool_tip_names: tuple[str, ...] = DEFAULT_TOOL_TIP_NAMES,
    wrist_link_name: str = DEFAULT_WRIST_LINK_NAME,
    include_base_link_collision: bool = False,
    base_link_name: str = DEFAULT_BASE_LINK_NAME,
    base_link_sphere_density: float | None = None,
    base_link_spheres_urdf: str | Path | None = None,
    base_arm_motion_safe: bool | None = None,
    base_arm_root_joint_name: str | None = None,
    base_arm_clearance_radius_m: float = DEFAULT_BASE_ARM_CLEARANCE_RADIUS_M,
    base_arm_clearance_below_m: float = DEFAULT_BASE_ARM_CLEARANCE_BELOW_M,
    base_arm_clearance_above_m: float = DEFAULT_BASE_ARM_CLEARANCE_ABOVE_M,
    body_sphere_density: float = DEFAULT_BODY_DENSITY,
    end_effector_sphere_density: float = DEFAULT_END_EFFECTOR_DENSITY,
    movable_end_effector_sphere_density: float = DEFAULT_MOVABLE_END_EFFECTOR_DENSITY,
    motion_safe_end_effector: bool = True,
    gripper_sample_count: int = 9,
    self_collision_clearance_m: float = DEFAULT_SELF_COLLISION_CLEARANCE_M,
    min_sphere_radius_m: float = DEFAULT_MIN_SPHERE_RADIUS_M,
    random_seed: int = 123,
) -> Link2PlusCollisionUrdfResult:
    """Generate a URDF with RobotBuilder visual-mesh spheres from link2 onward.

    The source URDF is not modified.
    """
    _ensure_local_curobo_on_path()

    import numpy as np
    import torch

    from curobo.logging import setup_logger
    from curobo.robot_builder import RobotBuilder

    urdf = Path(urdf_path).expanduser().resolve()
    output_urdf = (
        Path(output_urdf_path).expanduser().resolve()
        if output_urdf_path is not None
        else urdf.with_name(f"{urdf.stem}_robotbuilder_link2plus_ee_motion_safe_collision.urdf")
    )

    root = ET.parse(urdf).getroot()
    joints = _joint_infos(root)
    selected_links = discover_link2plus_links(urdf, start_link=start_link, tool_tip_names=tool_tip_names)
    arm_start_link = selected_links[0]
    base_arm_mount_origin: tuple[float, float, float] | None = None
    use_base_link_collision = include_base_link_collision or base_link_spheres_urdf is not None
    reuse_base_link_spheres = base_link_spheres_urdf is not None
    effective_base_arm_motion_safe = (
        not reuse_base_link_spheres if base_arm_motion_safe is None else base_arm_motion_safe
    )
    if use_base_link_collision:
        base_link = _find_link(root, base_link_name)
        if not reuse_base_link_spheres and not _has_visual_mesh(base_link):
            raise RuntimeError(f"base_link_name={base_link_name!r} has no visual mesh geometry for RobotBuilder fitting.")
        if effective_base_arm_motion_safe:
            base_arm_mount_origin = _resolve_base_arm_mount_origin(
                joints,
                base_link_name=base_link_name,
                arm_start_link=arm_start_link,
                base_arm_root_joint_name=base_arm_root_joint_name,
            )
        if base_link_name not in selected_links:
            selected_links.insert(0, base_link_name)
    end_effector_links = _detect_end_effector_links(
        selected_links,
        joints,
        wrist_link_name=wrist_link_name,
    )
    movable_end_effector_links = _detect_movable_end_effector_links(selected_links, joints)

    setup_logger("info")
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    urdf_link_names = {link.attrib["name"] for link in root.findall("link")}
    tool_frames = [name for name in tool_tip_names if name in urdf_link_names]
    builder = RobotBuilder(urdf_path=str(urdf), asset_path=str(urdf.parent), tool_frames=tool_frames[:1])
    fit_links = [link for link in selected_links if not (reuse_base_link_spheres and link == base_link_name)]
    missing = [link for link in fit_links if link not in builder._mesh_link_names]
    if missing:
        raise RuntimeError(f"Selected links have no visual mesh geometry for RobotBuilder fitting: {missing}")

    builder._mesh_link_names = list(fit_links)
    builder.fit_collision_spheres(sphere_density=body_sphere_density, use_collision_mesh=False)
    if include_base_link_collision and not reuse_base_link_spheres and base_link_sphere_density is not None:
        builder.refit_link_spheres(
            base_link_name,
            sphere_density=base_link_sphere_density,
            use_collision_mesh=False,
            iterations=300,
            coverage_weight=1500.0,
            protrusion_weight=10.0,
        )

    for link_name in end_effector_links:
        density = (
            movable_end_effector_sphere_density
            if link_name in movable_end_effector_links
            else end_effector_sphere_density
        )
        builder.refit_link_spheres(
            link_name,
            sphere_density=density,
            use_collision_mesh=False,
            iterations=300,
            coverage_weight=1500.0,
            protrusion_weight=10.0,
        )

    collision_spheres = {
        link_name: [dict(center=list(sphere["center"]), radius=float(sphere["radius"])) for sphere in builder.collision_spheres.get(link_name, [])]
        for link_name in selected_links
    }
    if reuse_base_link_spheres:
        collision_spheres[base_link_name] = _extract_link_collision_spheres_from_urdf(base_link_spheres_urdf, base_link_name)
    adjusted, removed = (0, 0)
    if motion_safe_end_effector:
        adjusted, removed = _make_end_effector_spheres_motion_safe(
            collision_spheres,
            selected_links,
            end_effector_links,
            joints,
            sample_count=gripper_sample_count,
            clearance=self_collision_clearance_m,
            min_radius=min_sphere_radius_m,
        )
    if use_base_link_collision and effective_base_arm_motion_safe and base_arm_mount_origin is not None:
        base_adjusted, base_removed = _make_base_link_spheres_arm_mount_safe(
            collision_spheres,
            base_link_name=base_link_name,
            mount_origin=base_arm_mount_origin,
            radius=base_arm_clearance_radius_m,
            below=base_arm_clearance_below_m,
            above=base_arm_clearance_above_m,
            clearance=self_collision_clearance_m,
            min_radius=min_sphere_radius_m,
        )
        adjusted += base_adjusted
        removed += base_removed

    selected_links = [link for link in selected_links if collision_spheres.get(link)]
    _write_sphere_collision_urdf(
        urdf,
        output_urdf,
        selected_links,
        collision_spheres,
        tool_tip_names=tool_tip_names,
    )
    counts = {link_name: len(collision_spheres.get(link_name, [])) for link_name in selected_links}
    return Link2PlusCollisionUrdfResult(
        output_urdf=output_urdf,
        selected_links=selected_links,
        end_effector_links=end_effector_links,
        collision_sphere_counts=counts,
        adjusted_sphere_count=adjusted,
        removed_sphere_count=removed,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True, help="Input URDF. The file is not modified.")
    parser.add_argument("--output", type=Path, default=None, help="Output URDF path.")
    parser.add_argument("--start-link", type=str, default=None, help="Explicit arm start link. Defaults to auto-discovered *link2.")
    parser.add_argument(
        "--wrist-link",
        type=str,
        default=DEFAULT_WRIST_LINK_NAME,
        help="Final arm/wrist link. Selected links attached after this link are treated as end-effector geometry.",
    )
    parser.add_argument(
        "--include-base-link-collision",
        "--include-base-link",
        action="store_true",
        help="Also fit and write RobotBuilder sphere collisions for the chassis base link.",
    )
    parser.add_argument(
        "--base-link-name",
        type=str,
        default=DEFAULT_BASE_LINK_NAME,
        help="Base/chassis link to include when --include-base-link-collision is set.",
    )
    parser.add_argument(
        "--base-link-density",
        type=float,
        default=None,
        help="Optional sphere density for the included base link. Defaults to --body-density.",
    )
    parser.add_argument(
        "--base-link-spheres-urdf",
        "--reuse-base-link-spheres-urdf",
        type=Path,
        default=None,
        help="Optional URDF to copy base-link sphere collisions from instead of fitting base_link with RobotBuilder.",
    )
    parser.add_argument(
        "--disable-base-arm-motion-safe",
        dest="base_arm_motion_safe",
        action="store_false",
        default=None,
        help="Do not shrink/remove base-link spheres around the arm mount.",
    )
    parser.add_argument(
        "--enable-base-arm-motion-safe",
        dest="base_arm_motion_safe",
        action="store_true",
        help="Shrink/remove base-link spheres around the arm mount. Defaults on for fitted base_link spheres and off for reused base_link spheres.",
    )
    parser.add_argument(
        "--base-arm-root-joint",
        type=str,
        default=None,
        help="Optional fixed joint from base link to arm base. Defaults to the first joint on the path to the arm start link.",
    )
    parser.add_argument(
        "--base-arm-clearance-radius-m",
        type=float,
        default=DEFAULT_BASE_ARM_CLEARANCE_RADIUS_M,
        help="XY radius around the arm mount where base-link spheres are shrunk/removed.",
    )
    parser.add_argument(
        "--base-arm-clearance-below-m",
        type=float,
        default=DEFAULT_BASE_ARM_CLEARANCE_BELOW_M,
        help="Clearance volume below the arm mount origin.",
    )
    parser.add_argument(
        "--base-arm-clearance-above-m",
        type=float,
        default=DEFAULT_BASE_ARM_CLEARANCE_ABOVE_M,
        help="Clearance volume above the arm mount origin.",
    )
    parser.add_argument("--body-density", type=float, default=DEFAULT_BODY_DENSITY)
    parser.add_argument(
        "--end-effector-density",
        "--tool-density",
        type=float,
        default=DEFAULT_END_EFFECTOR_DENSITY,
        help="Sphere density for structurally detected fixed tool/end-effector links.",
    )
    parser.add_argument(
        "--movable-end-effector-density",
        "--gripper-density",
        type=float,
        default=DEFAULT_MOVABLE_END_EFFECTOR_DENSITY,
        help="Sphere density for movable end-effector children, such as prismatic fingers.",
    )
    parser.add_argument(
        "--ee-density-multiplier",
        type=float,
        default=1.0,
        help="Multiply both end-effector densities by this factor.",
    )
    parser.add_argument("--tool-tip-name", action="append", default=None, help="Tool-tip frame link to exclude from collision.")
    parser.add_argument("--disable-ee-motion-safe", action="store_true", help="Do not shrink/remove gripper spheres that block prismatic motion.")
    parser.add_argument("--gripper-samples", type=int, default=9)
    parser.add_argument("--self-collision-clearance-m", type=float, default=DEFAULT_SELF_COLLISION_CLEARANCE_M)
    parser.add_argument("--min-sphere-radius-m", type=float, default=DEFAULT_MIN_SPHERE_RADIUS_M)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = generate_link2plus_collision_urdf(
        args.urdf,
        args.output,
        start_link=args.start_link,
        tool_tip_names=tuple(args.tool_tip_name or DEFAULT_TOOL_TIP_NAMES),
        wrist_link_name=args.wrist_link,
        include_base_link_collision=args.include_base_link_collision,
        base_link_name=args.base_link_name,
        base_link_sphere_density=args.base_link_density,
        base_link_spheres_urdf=args.base_link_spheres_urdf,
        base_arm_motion_safe=args.base_arm_motion_safe,
        base_arm_root_joint_name=args.base_arm_root_joint,
        base_arm_clearance_radius_m=args.base_arm_clearance_radius_m,
        base_arm_clearance_below_m=args.base_arm_clearance_below_m,
        base_arm_clearance_above_m=args.base_arm_clearance_above_m,
        body_sphere_density=args.body_density,
        end_effector_sphere_density=args.end_effector_density * args.ee_density_multiplier,
        movable_end_effector_sphere_density=args.movable_end_effector_density * args.ee_density_multiplier,
        motion_safe_end_effector=not args.disable_ee_motion_safe,
        gripper_sample_count=args.gripper_samples,
        self_collision_clearance_m=args.self_collision_clearance_m,
        min_sphere_radius_m=args.min_sphere_radius_m,
    )
    print(f"output_urdf={result.output_urdf}")
    print(f"selected_links={result.selected_links}")
    print(f"end_effector_links={result.end_effector_links}")
    print(f"collision_sphere_count={sum(result.collision_sphere_counts.values())}")
    print(f"per_link={result.collision_sphere_counts}")
    print(f"adjusted_sphere_count={result.adjusted_sphere_count}")
    print(f"removed_sphere_count={result.removed_sphere_count}")


if __name__ == "__main__":
    main()
