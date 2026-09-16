#!/usr/bin/env python3
"""Extract a DH-like table from a URDF serial chain.

URDF frames are not uniquely convertible to Denavit-Hartenberg frames. This
script reports the URDF joint data and a conservative DH-like interpretation:

    theta_offset = origin.rpy.z
    d            = origin.xyz.z
    a            = origin.xyz.x
    alpha        = origin.rpy.x

Rows are marked as "dh_ok" only when the joint is close to the common DH shape:
axis ~= z, origin.y ~= 0, origin.rpy.y ~= 0. For other rows, use the raw URDF
columns or define DH frames manually.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ACTIVE_JOINT_TYPES = {"revolute", "continuous", "prismatic"}


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None
    upper: float | None


@dataclass
class Row:
    index: int
    joint: str
    joint_type: str
    parent: str
    child: str
    theta_offset_rad: float
    d_m: float
    a_m: float
    alpha_rad: float
    axis: str
    origin_xyz: str
    origin_rpy: str
    limit: str
    dh_ok: bool
    notes: str


def parse_vec(value: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    parts = value.split()
    if len(parts) != 3:
        raise ValueError(f"Expected 3-vector, got: {value!r}")
    return tuple(float(x) for x in parts)  # type: ignore[return-value]


def parse_urdf(path: Path) -> tuple[set[str], list[Joint]]:
    root = ET.parse(path).getroot()
    links = {link.attrib["name"] for link in root.findall("link")}
    joints: list[Joint] = []

    for elem in root.findall("joint"):
        origin = elem.find("origin")
        parent = elem.find("parent")
        child = elem.find("child")
        axis = elem.find("axis")
        limit = elem.find("limit")
        if parent is None or child is None:
            continue

        lower = float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None
        upper = float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None

        joints.append(
            Joint(
                name=elem.attrib["name"],
                joint_type=elem.attrib.get("type", "fixed"),
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                xyz=parse_vec(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                rpy=parse_vec(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
                axis=parse_vec(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
                lower=lower,
                upper=upper,
            )
        )

    return links, joints


def infer_root(links: set[str], joints: Iterable[Joint]) -> str:
    children = {joint.child for joint in joints}
    roots = sorted(links - children)
    if not roots:
        raise ValueError("Could not infer root link: every link is a joint child.")
    return roots[0]


def build_chain(joints: list[Joint], root: str, tip: str | None) -> list[Joint]:
    by_parent: dict[str, list[Joint]] = {}
    for joint in joints:
        by_parent.setdefault(joint.parent, []).append(joint)

    chain: list[Joint] = []
    link = root
    visited = set()

    while link in by_parent:
        candidates = by_parent[link]
        if len(candidates) > 1 and tip is None:
            names = ", ".join(j.name for j in candidates)
            raise ValueError(
                f"Branch found at link {link!r}: {names}. Provide --tip to select a serial chain."
            )

        if tip is None:
            joint = candidates[0]
        else:
            joint = choose_next_joint_toward_tip(candidates, joints, tip)

        if joint.name in visited:
            raise ValueError(f"Loop detected at joint {joint.name!r}.")
        visited.add(joint.name)
        chain.append(joint)
        link = joint.child
        if tip is not None and link == tip:
            break

    if tip is not None and (not chain or chain[-1].child != tip):
        raise ValueError(f"Could not find a chain from {root!r} to {tip!r}.")
    return chain


def choose_next_joint_toward_tip(candidates: list[Joint], joints: list[Joint], tip: str) -> Joint:
    descendants: dict[str, list[Joint]] = {}
    for joint in joints:
        descendants.setdefault(joint.parent, []).append(joint)

    def reaches_tip(link: str) -> bool:
        if link == tip:
            return True
        return any(reaches_tip(j.child) for j in descendants.get(link, []))

    matches = [joint for joint in candidates if reaches_tip(joint.child)]
    if len(matches) != 1:
        names = ", ".join(j.name for j in candidates)
        raise ValueError(f"Could not choose branch toward tip {tip!r}; candidates: {names}")
    return matches[0]


def is_close(value: float, target: float, tol: float) -> bool:
    return abs(value - target) <= tol


def fmt_vec(values: tuple[float, float, float], precision: int) -> str:
    return " ".join(f"{v:.{precision}g}" for v in values)


def make_rows(chain: list[Joint], include_fixed: bool, joint_regex: str | None, tol: float, precision: int) -> list[Row]:
    pattern = re.compile(joint_regex) if joint_regex else None
    rows: list[Row] = []

    for joint in chain:
        if not include_fixed and joint.joint_type == "fixed":
            continue
        if pattern and not pattern.search(joint.name):
            continue

        x, y, z = joint.xyz
        roll, pitch, yaw = joint.rpy
        ax, ay, az = joint.axis
        notes = []
        axis_is_z = is_close(ax, 0.0, tol) and is_close(ay, 0.0, tol) and is_close(abs(az), 1.0, tol)
        if joint.joint_type in ACTIVE_JOINT_TYPES and not axis_is_z:
            notes.append("axis_not_z")
        if not is_close(y, 0.0, tol):
            notes.append("origin_y_nonzero")
        if not is_close(pitch, 0.0, tol):
            notes.append("pitch_nonzero")
        if joint.joint_type == "fixed":
            notes.append("fixed_joint")

        limit = ""
        if joint.lower is not None or joint.upper is not None:
            lo = "" if joint.lower is None else f"{joint.lower:.{precision}g}"
            hi = "" if joint.upper is None else f"{joint.upper:.{precision}g}"
            limit = f"[{lo}, {hi}]"

        rows.append(
            Row(
                index=len(rows) + 1,
                joint=joint.name,
                joint_type=joint.joint_type,
                parent=joint.parent,
                child=joint.child,
                theta_offset_rad=yaw,
                d_m=z,
                a_m=x,
                alpha_rad=roll,
                axis=fmt_vec(joint.axis, precision),
                origin_xyz=fmt_vec(joint.xyz, precision),
                origin_rpy=fmt_vec(joint.rpy, precision),
                limit=limit,
                dh_ok=not notes or notes == ["fixed_joint"],
                notes=",".join(notes),
            )
        )

    return rows


def print_markdown(rows: list[Row], precision: int) -> None:
    headers = [
        "i",
        "joint",
        "type",
        "theta_offset(rad)",
        "d(m)",
        "a(m)",
        "alpha(rad)",
        "axis",
        "dh_ok",
        "notes",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row.index),
                    row.joint,
                    row.joint_type,
                    f"{row.theta_offset_rad:.{precision}g}",
                    f"{row.d_m:.{precision}g}",
                    f"{row.a_m:.{precision}g}",
                    f"{row.alpha_rad:.{precision}g}",
                    row.axis,
                    str(row.dh_ok),
                    row.notes,
                ]
            )
            + " |"
        )


def print_csv(rows: list[Row]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=list(asdict(rows[0]).keys()) if rows else list(Row.__annotations__))
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))


def print_json(rows: list[Row]) -> None:
    print(json.dumps([asdict(row) for row in rows], indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a DH-like table from a URDF serial chain.")
    parser.add_argument("urdf", type=Path, help="Path to URDF file.")
    parser.add_argument("--root", help="Root link. Defaults to the only link that is not a child.")
    parser.add_argument("--tip", help="Tip link used to select one branch in a branched URDF.")
    parser.add_argument("--include-fixed", action="store_true", help="Include fixed joints in the output.")
    parser.add_argument("--joint-regex", help="Only include joints whose names match this regex.")
    parser.add_argument("--format", choices=["markdown", "csv", "json"], default="markdown")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for DH compatibility checks.")
    parser.add_argument("--precision", type=int, default=8, help="Printed numeric precision.")
    args = parser.parse_args()

    links, joints = parse_urdf(args.urdf)
    root = args.root or infer_root(links, joints)
    chain = build_chain(joints, root, args.tip)
    rows = make_rows(chain, args.include_fixed, args.joint_regex, args.tol, args.precision)

    if args.format == "markdown":
        print_markdown(rows, args.precision)
    elif args.format == "csv":
        print_csv(rows)
    else:
        print_json(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
