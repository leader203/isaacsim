#!/usr/bin/env python3
"""Generate a cuRobo robot config from URDF sphere collisions.

This script treats the URDF as the source of truth for robot collision spheres.
It reads only ``<collision><geometry><sphere radius="...">`` entries from the
selected collision links and writes them into cuRobo's ``robot_cfg`` yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import yaml


REPO_ROOT = Path(__file__).resolve().parents[8]
ALLSTAR_TOILET_DIR = (
    REPO_ROOT
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet"
)
DEFAULT_URDF = (
    ALLSTAR_TOILET_DIR
    / "data/robots/chassis_arm/motion_wheel_arm_simple_sphere_urdf"
    / "mecanum730_xms5_brush_rolling_0512.urdf"
)
DEFAULT_OUTPUT = REPO_ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/configs/curobo/mecanum730_xms5_brush_rolling_0512.yml"

DEFAULT_LINK_NAMES = [
    "XMS5-R800-W4G3B4C_base",
    "XMS5-R800-W4G3B4C_link1",
    "XMS5-R800-W4G3B4C_link2",
    "XMS5-R800-W4G3B4C_link3",
    "XMS5-R800-W4G3B4C_link4",
    "XMS5-R800-W4G3B4C_link5",
    "link6",
    "tool_link",
    "tool_tip_link",
]
DEFAULT_COLLISION_LINKS = [
    "XMS5-R800-W4G3B4C_link2",
    "XMS5-R800-W4G3B4C_link3",
    "XMS5-R800-W4G3B4C_link4",
    "XMS5-R800-W4G3B4C_link5",
    "link6",
    "tool_link",
    "tool_tip_link",
]
DEFAULT_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
DEFAULT_RETRACT_CONFIG = [0.0, 0.561996, -0.570723, -0.561996, 0.55676, 0.0]


def _parse_vec3(text: str | None) -> list[float]:
    if text is None:
        return [0.0, 0.0, 0.0]
    values = [float(value) for value in text.split()]
    if len(values) != 3:
        raise ValueError(f"Expected xyz/rpy with 3 values, got {text!r}")
    return values


def _format_float(value: float) -> float:
    return float(f"{value:.9g}")


def _find_link(root: ET.Element, link_name: str) -> ET.Element:
    for link in root.findall("link"):
        if link.attrib.get("name") == link_name:
            return link
    raise RuntimeError(f"URDF missing link: {link_name}")


def _validate_joints(root: ET.Element, joint_names: list[str]) -> None:
    available = {joint.attrib.get("name") for joint in root.findall("joint")}
    missing = [joint_name for joint_name in joint_names if joint_name not in available]
    if missing:
        raise RuntimeError(f"URDF missing joints: {missing}")


def _joint_edges(root: ET.Element) -> dict[str, list[tuple[str, str, str]]]:
    edges: dict[str, list[tuple[str, str, str]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = parent.attrib.get("link")
        child_link = child.attrib.get("link")
        joint_name = joint.attrib.get("name", "")
        joint_type = joint.attrib.get("type", "")
        if not parent_link or not child_link:
            continue
        edges.setdefault(parent_link, []).append((child_link, joint_name, joint_type))
    return edges


def _find_link_path(root: ET.Element, base_link: str, tool_frame: str) -> list[str]:
    edges = _joint_edges(root)
    stack: list[tuple[str, list[str]]] = [(base_link, [base_link])]
    visited: set[str] = set()
    while stack:
        link_name, path = stack.pop()
        if link_name == tool_frame:
            return path
        if link_name in visited:
            continue
        visited.add(link_name)
        for child_link, _, _ in reversed(edges.get(link_name, [])):
            stack.append((child_link, [*path, child_link]))
    raise RuntimeError(f"Could not find URDF link path from base_link={base_link!r} to tool_frame={tool_frame!r}")


def _link_has_sphere_collisions(root: ET.Element, link_name: str) -> bool:
    try:
        link = _find_link(root, link_name)
    except RuntimeError:
        return False
    return any(collision.find("geometry/sphere") is not None for collision in link.findall("collision"))


def discover_collision_links(
    urdf_path: Path,
    *,
    base_link: str,
    tool_frame: str,
) -> list[str]:
    """Find links on the base->tool path that have sphere collision entries."""
    root = ET.parse(urdf_path).getroot()
    link_path = _find_link_path(root, base_link, tool_frame)
    collision_links = [link_name for link_name in link_path if _link_has_sphere_collisions(root, link_name)]
    if not collision_links:
        raise RuntimeError(
            f"No sphere collision links found on URDF path from {base_link!r} to {tool_frame!r}."
        )
    return collision_links


def extract_collision_spheres(
    urdf_path: Path,
    collision_link_names: list[str],
    *,
    joint_names: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Extract local-frame sphere collisions from a URDF."""
    root = ET.parse(urdf_path).getroot()
    spheres_by_link: dict[str, list[dict[str, Any]]] = {}

    for link_name in collision_link_names:
        link = _find_link(root, link_name)
        spheres: list[dict[str, Any]] = []
        for collision in link.findall("collision"):
            sphere = collision.find("geometry/sphere")
            if sphere is None:
                continue
            origin = collision.find("origin")
            center = _parse_vec3(origin.attrib.get("xyz") if origin is not None else None)
            radius = float(sphere.attrib["radius"])
            if radius <= 0.0:
                raise RuntimeError(f"Invalid radius={radius} for link={link_name}")
            spheres.append(
                {
                    "center": [_format_float(value) for value in center],
                    "radius": _format_float(radius),
                }
            )
        if not spheres:
            raise RuntimeError(
                f"URDF link {link_name!r} has no sphere collision entries. "
                "Regenerate the sphere URDF first or remove this link from collision_link_names."
            )
        spheres_by_link[link_name] = spheres

    _validate_joints(root, joint_names or DEFAULT_JOINT_NAMES)
    return spheres_by_link


def _self_collision_ignore(collision_links: list[str], link_order: list[str] | None = None) -> dict[str, list[str]]:
    ignore = {
        "XMS5-R800-W4G3B4C_link2": ["XMS5-R800-W4G3B4C_link3"],
        "XMS5-R800-W4G3B4C_link3": ["XMS5-R800-W4G3B4C_link2", "XMS5-R800-W4G3B4C_link4"],
        "XMS5-R800-W4G3B4C_link4": ["XMS5-R800-W4G3B4C_link3", "XMS5-R800-W4G3B4C_link5"],
        "XMS5-R800-W4G3B4C_link5": ["XMS5-R800-W4G3B4C_link4", "link6"],
        "link6": ["XMS5-R800-W4G3B4C_link5", "tool_link", "tool_tip_link"],
        "tool_link": ["link6", "tool_tip_link"],
        "tool_tip_link": ["link6", "tool_link"],
    }
    return {
        link: [other for other in others if other in collision_links]
        for link, others in ignore.items()
        if link in collision_links
    }
    if link_order is None:
        return manual_ignore
    ordered_ignore = {link: set(manual_ignore.get(link, [])) for link in collision_links}
    collision_set = set(collision_links)
    ordered_collision_links = [link for link in link_order if link in collision_set]
    for prev_link, next_link in zip(ordered_collision_links, ordered_collision_links[1:]):
        ordered_ignore.setdefault(prev_link, set()).add(next_link)
        ordered_ignore.setdefault(next_link, set()).add(prev_link)
    return {
        link: sorted(other for other in others if other in collision_set)
        for link, others in ordered_ignore.items()
        if link in collision_set and others
    }


def _self_collision_buffer(
    collision_links: list[str],
    collision_sphere_buffer: float | dict[str, float],
) -> dict[str, float]:
    """Keep scene collision padding from tightening robot self-collision checks."""
    if isinstance(collision_sphere_buffer, dict):
        return {
            link_name: -float(collision_sphere_buffer.get(link_name, 0.0))
            for link_name in collision_links
        }
    buffer_value = float(collision_sphere_buffer)
    return {link_name: -buffer_value for link_name in collision_links}


def build_robot_cfg(
    *,
    urdf_path: Path,
    asset_root_path: Path,
    collision_link_names: list[str] | None = None,
    collision_sphere_buffer: float | dict[str, float],
    base_link: str = "XMS5-R800-W4G3B4C_base",
    tool_frame: str = "tool_tip_link",
    joint_names: list[str] | None = None,
    default_joint_position: list[float] | None = None,
) -> dict[str, Any]:
    root = ET.parse(urdf_path).getroot()
    joint_names = list(joint_names or DEFAULT_JOINT_NAMES)
    if default_joint_position is None:
        default_joint_position = list(DEFAULT_RETRACT_CONFIG if len(joint_names) == len(DEFAULT_RETRACT_CONFIG) else [0.0] * len(joint_names))
    else:
        default_joint_position = list(default_joint_position)
    if len(default_joint_position) != len(joint_names):
        raise RuntimeError(
            f"default_joint_position length {len(default_joint_position)} does not match joint_names length {len(joint_names)}"
        )
    link_order = _find_link_path(root, base_link, tool_frame)
    if collision_link_names is None:
        collision_link_names = [
            link_name for link_name in link_order if _link_has_sphere_collisions(root, link_name)
        ]
    collision_spheres = extract_collision_spheres(
        urdf_path,
        collision_link_names,
        joint_names=joint_names,
    )
    return {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": str(urdf_path),
                "asset_root_path": str(asset_root_path),
                "base_link": base_link,
                "tool_frames": [tool_frame],
                "lock_joints": {},
                "extra_links": None,
                "collision_link_names": list(collision_link_names),
                "collision_spheres": collision_spheres,
                "collision_sphere_buffer": collision_sphere_buffer,
                "extra_collision_spheres": {},
                "self_collision_ignore": _self_collision_ignore(collision_link_names, link_order),
                "self_collision_buffer": _self_collision_buffer(
                    collision_link_names,
                    collision_sphere_buffer,
                ),
                "use_global_cumul": True,
                "mesh_link_names": None,
                "cspace": {
                    "joint_names": list(joint_names),
                    "default_joint_position": list(default_joint_position),
                    "null_space_weight": [1.0] * len(joint_names),
                    "cspace_distance_weight": [1.0] * len(joint_names),
                    "max_jerk": 500.0,
                    "max_acceleration": 15.0,
                    "position_limit_clip": 0.01,
                },
            }
        }
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Sphere-collision URDF.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output cuRobo robot yaml.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="cuRobo asset root. Defaults to the URDF parent directory.",
    )
    parser.add_argument(
        "--collision-link",
        action="append",
        dest="collision_links",
        default=None,
        help="Collision link to export. Repeat to override automatic base-to-tool discovery.",
    )
    parser.add_argument("--base-link", type=str, default="XMS5-R800-W4G3B4C_base")
    parser.add_argument("--tool-frame", type=str, default="tool_tip_link")
    parser.add_argument("--joint-name", action="append", dest="joint_names", default=None)
    parser.add_argument("--collision-sphere-buffer", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    urdf_path = args.urdf.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve() if args.asset_root is not None else urdf_path.parent
    collision_links = args.collision_links

    robot_cfg = build_robot_cfg(
        urdf_path=urdf_path,
        asset_root_path=asset_root,
        collision_link_names=collision_links,
        collision_sphere_buffer=args.collision_sphere_buffer,
        base_link=args.base_link,
        tool_frame=args.tool_frame,
        joint_names=args.joint_names or DEFAULT_JOINT_NAMES,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(robot_cfg, stream, sort_keys=False)

    counts = {
        link_name: len(spheres)
        for link_name, spheres in robot_cfg["robot_cfg"]["kinematics"]["collision_spheres"].items()
    }
    print(f"Wrote {output_path}")
    print(f"collision_sphere_count={sum(counts.values())} per_link={counts}")


if __name__ == "__main__":
    main()
