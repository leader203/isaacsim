#!/usr/bin/env python3
"""Add explicit wheel-core collision meshes to chassis_arm URDF assets.

The current 0730 chassis_arm URDFs keep the base collision mesh and all roller
collisions, but omit collisions on the four main wheel-core links. That leaves
the mobile base supported almost entirely by 28 small roller contacts, which is
numerically fragile during yaw and lateral motion.

This script mirrors the more stable 0303-style setup by duplicating each wheel
core visual mesh as an explicit collision mesh on:
  - wheel_fl_link
  - wheel_fr_link
  - wheel_rl_link
  - wheel_rr_link
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent.parent
URDF_DIR = TASK_DIR / "data" / "robots" / "chassis_arm" / "urdf"

WHEEL_LINK_NAMES = (
    "wheel_fl_link",
    "wheel_fr_link",
    "wheel_rl_link",
    "wheel_rr_link",
)


def _find_link(root: ET.Element, name: str) -> ET.Element:
    for link in root.findall("link"):
        if link.attrib.get("name") == name:
            return link
    raise RuntimeError(f"missing link: {name}")


def _ensure_wheel_collision(link: ET.Element) -> bool:
    if link.find("collision") is not None:
        return False

    visual = link.find("visual")
    if visual is None:
        raise RuntimeError(f"wheel link has no visual to clone: {link.attrib.get('name')}")

    collision = copy.deepcopy(visual)
    collision.tag = "collision"

    children = list(link)
    visual_index = children.index(visual)
    link.insert(visual_index + 1, collision)
    return True


def patch_one(path: Path) -> bool:
    tree = ET.parse(path)
    root = tree.getroot()

    changed = False
    for link_name in WHEEL_LINK_NAMES:
        changed |= _ensure_wheel_collision(_find_link(root, link_name))

    if changed:
        ET.indent(tree, space="    ")
        tree.write(path, encoding="utf-8", xml_declaration=True, short_empty_elements=False)
    return changed


def main() -> None:
    urdf_paths = sorted(URDF_DIR.glob("*.urdf"))
    if not urdf_paths:
        raise RuntimeError(f"no URDF files found in {URDF_DIR}")

    changed_count = 0
    for path in urdf_paths:
        if patch_one(path):
            changed_count += 1
            print(f"[patched] {path.name}")
        else:
            print(f"[ok] {path.name}")

    print(f"wheel-core collision stabilization complete: {changed_count}/{len(urdf_paths)} files updated")


if __name__ == "__main__":
    main()
