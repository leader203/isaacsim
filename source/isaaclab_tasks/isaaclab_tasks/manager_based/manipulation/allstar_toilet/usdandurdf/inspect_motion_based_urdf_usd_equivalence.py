"""Inspect URDF/USD correspondence for chassis_arm assets.

Usage
-----
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/usdandurdf/inspect_motion_based_urdf_usd_equivalence.py
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect chassis_arm URDF/USD correspondence.")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--usd_dir", type=str, default=None)
    parser.add_argument("--sample", type=str, default="mecanum730_xms5_brush_rolling_0512")
    return parser


simulation_app = None
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

    args_cli = _make_parser().parse_args()
except ModuleNotFoundError:
    from isaaclab.app import AppLauncher

    parser = _make_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent.parent
DATA_ROBOTS_DIR = TASK_DIR / "data" / "robots"
DEFAULT_INPUT_DIR = DATA_ROBOTS_DIR / "chassis_arm" / "motion_based_urdf"
DEFAULT_USD_DIR = DATA_ROBOTS_DIR / "chassis_arm" / "motion_based_usd"
SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float] | None
    origin_rpy: tuple[float, float, float] | None
    axis: tuple[float, float, float] | None
    lower_limit: float | None
    upper_limit: float | None


@dataclass(frozen=True)
class UrdfLink:
    name: str
    mass: float | None
    com: tuple[float, float, float] | None
    inertia: tuple[float, float, float, float, float, float] | None
    collision_kinds: tuple[str, ...]


@dataclass(frozen=True)
class UsdJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    axis: str | None
    local_pos0: tuple[float, float, float] | None
    local_pos1: tuple[float, float, float] | None
    local_rot0: Gf.Quatd | None
    local_rot1: Gf.Quatd | None
    lower_limit: float | None
    upper_limit: float | None


@dataclass(frozen=True)
class UsdLink:
    name: str
    mass: float | None
    com: tuple[float, float, float] | None
    inertia: tuple[float, float, float, float, float, float] | None
    collider_kinds: tuple[str, ...]


def sanitize_name(name: str) -> str:
    return SANITIZE_RE.sub("_", name)


def parse_xyz(text: str | None) -> tuple[float, float, float] | None:
    if text is None:
        return None
    values = tuple(float(v) for v in text.split())
    if len(values) != 3:
        raise ValueError(text)
    return values


def parse_rpy(text: str | None) -> tuple[float, float, float] | None:
    return parse_xyz(text)


def parse_inertia(inertia_node: ET.Element | None) -> tuple[float, float, float, float, float, float] | None:
    if inertia_node is None:
        return None
    return (
        float(inertia_node.attrib["ixx"]),
        float(inertia_node.attrib["ixy"]),
        float(inertia_node.attrib["ixz"]),
        float(inertia_node.attrib["iyy"]),
        float(inertia_node.attrib["iyz"]),
        float(inertia_node.attrib["izz"]),
    )


def parse_limit_float(node: ET.Element | None, key: str) -> float | None:
    if node is None or key not in node.attrib:
        return None
    return float(node.attrib[key])


def quat_to_quatd(quat: Gf.Quatf | Gf.Quatd | None) -> Gf.Quatd:
    if quat is None:
        return Gf.Quatd(1.0, 0.0, 0.0, 0.0)
    if isinstance(quat, Gf.Quatd):
        return quat
    imag = quat.GetImaginary()
    return Gf.Quatd(float(quat.GetReal()), Gf.Vec3d(float(imag[0]), float(imag[1]), float(imag[2])))


def quat_to_rotation(quat: Gf.Quatf | Gf.Quatd | None) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    if quat is None:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    q = quat_to_quatd(quat)
    mat = Gf.Matrix3d(q)
    return tuple(tuple(float(mat[i][j]) for j in range(3)) for i in range(3))  # type: ignore[return-value]


def rpy_to_quat(rpy: tuple[float, float, float] | None) -> Gf.Quatd:
    if rpy is None:
        return Gf.Quatd(1.0, 0.0, 0.0, 0.0)
    roll, pitch, yaw = rpy
    rx = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), math.degrees(roll))
    ry = Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), math.degrees(pitch))
    rz = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), math.degrees(yaw))
    # URDF rpy uses fixed-axis roll, then pitch, then yaw.
    return quat_to_quatd((rx * ry * rz).GetQuat())


def normalize_vector(vec: tuple[float, float, float] | None, tol: float = 1e-9) -> tuple[float, float, float] | None:
    if vec is None:
        return None
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= tol:
        return None
    return tuple(v / norm for v in vec)


def rotate_vector(rot: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]], vec: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(rot[i][j] * vec[j] for j in range(3)) for i in range(3))


def axis_token_to_vector(axis: str | None) -> tuple[float, float, float] | None:
    mapping = {
        "X": (1.0, 0.0, 0.0),
        "Y": (0.0, 1.0, 0.0),
        "Z": (0.0, 0.0, 1.0),
    }
    if axis is None:
        return None
    return mapping.get(str(axis).upper())


def compose_transform(
    translation: tuple[float, float, float] | None,
    rotation: Gf.Quatd | Gf.Quatf | None,
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    return (
        translation or (0.0, 0.0, 0.0),
        quat_to_rotation(rotation),
    )


def invert_transform(
    translation: tuple[float, float, float],
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    rot_t = tuple(tuple(rotation[j][i] for j in range(3)) for i in range(3))
    inv_translation = tuple(-sum(rot_t[i][j] * translation[j] for j in range(3)) for i in range(3))
    return inv_translation, rot_t


def multiply_transforms(
    lhs: tuple[tuple[float, float, float], tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    rhs: tuple[tuple[float, float, float], tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    lhs_t, lhs_r = lhs
    rhs_t, rhs_r = rhs
    out_r = tuple(
        tuple(sum(lhs_r[i][k] * rhs_r[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )
    rotated_rhs_t = rotate_vector(lhs_r, rhs_t)
    out_t = tuple(lhs_t[i] + rotated_rhs_t[i] for i in range(3))
    return out_t, out_r


def approx_rotation(
    lhs: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    rhs: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    tol: float = 1e-4,
) -> bool:
    return all(abs(lhs[i][j] - rhs[i][j]) <= tol for i in range(3) for j in range(3))


def reconstruct_inertia(diagonal: Gf.Vec3f | Gf.Vec3d | None, principal_axes: Gf.Quatf | Gf.Quatd | None) -> tuple[float, float, float, float, float, float] | None:
    if diagonal is None:
        return None
    rot = quat_to_rotation(principal_axes)
    d = [
        [float(diagonal[0]), 0.0, 0.0],
        [0.0, float(diagonal[1]), 0.0],
        [0.0, 0.0, float(diagonal[2])],
    ]
    # USD stores principal axes as the rotation from body frame into principal-inertia frame,
    # so the body-frame inertia tensor is reconstructed as R^T * D * R.
    rot_t = [[rot[j][i] for j in range(3)] for i in range(3)]
    rd = [[sum(rot_t[i][k] * d[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    full = [[sum(rd[i][k] * rot_t[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    return (
        full[0][0],
        full[0][1],
        full[0][2],
        full[1][1],
        full[1][2],
        full[2][2],
    )


def approx_tuple(lhs: tuple[float, ...] | None, rhs: tuple[float, ...] | None, tol: float = 1e-5) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    if len(lhs) != len(rhs):
        return False
    return all(abs(a - b) <= tol for a, b in zip(lhs, rhs))


def collider_kind_matches(urdf_kind: str, usd_kind: str) -> bool:
    urdf_lower = urdf_kind.lower()
    usd_lower = usd_kind.lower()
    if urdf_lower == "mesh":
        return usd_lower in {"mesh", "cylinder", "capsule", "sphere", "cube"}
    return urdf_lower == usd_lower


def collider_kind_multiset_matches(urdf_kinds: tuple[str, ...], usd_kinds: tuple[str, ...]) -> bool:
    remaining = list(usd_kinds)
    for urdf_kind in urdf_kinds:
        for index, usd_kind in enumerate(remaining):
            if collider_kind_matches(urdf_kind, usd_kind):
                remaining.pop(index)
                break
        else:
            return False
    return not remaining


def is_effectively_unbounded_limit(lower: float | None, upper: float | None) -> bool:
    lower_ok = lower is None or (math.isinf(lower) and lower < 0.0)
    upper_ok = upper is None or (math.isinf(upper) and upper > 0.0)
    return lower_ok and upper_ok


def approx_optional_float(lhs: float | None, rhs: float | None, tol: float = 1e-5) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    return abs(lhs - rhs) <= tol


def parse_urdf(urdf_path: Path) -> tuple[dict[str, UrdfLink], dict[str, UrdfJoint]]:
    root = ET.parse(urdf_path).getroot()
    links: dict[str, UrdfLink] = {}
    joints: dict[str, UrdfJoint] = {}

    for link in root.findall("link"):
        mass_node = link.find("./inertial/mass")
        origin_node = link.find("./inertial/origin")
        inertia_node = link.find("./inertial/inertia")
        collisions = []
        for collision in link.findall("collision"):
            geom = collision.find("geometry")
            if geom is None or len(geom) == 0:
                collisions.append("unknown")
            else:
                collisions.append(geom[0].tag)
        links[link.attrib["name"]] = UrdfLink(
            name=link.attrib["name"],
            mass=float(mass_node.attrib["value"]) if mass_node is not None else None,
            com=parse_xyz(origin_node.attrib["xyz"]) if origin_node is not None and "xyz" in origin_node.attrib else None,
            inertia=parse_inertia(inertia_node),
            collision_kinds=tuple(collisions),
        )

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        origin = joint.find("origin")
        limit = joint.find("limit")
        joints[joint.attrib["name"]] = UrdfJoint(
            name=joint.attrib["name"],
            joint_type=joint.attrib["type"],
            parent=parent.attrib["link"] if parent is not None else "",
            child=child.attrib["link"] if child is not None else "",
            origin_xyz=parse_xyz(origin.attrib.get("xyz")) if origin is not None else None,
            origin_rpy=parse_rpy(origin.attrib.get("rpy")) if origin is not None else None,
            axis=parse_xyz(axis.attrib["xyz"]) if axis is not None else None,
            lower_limit=parse_limit_float(limit, "lower"),
            upper_limit=parse_limit_float(limit, "upper"),
        )
    return links, joints


def resolve_body_name(rel: Sdf.Path | None) -> str | None:
    if not rel or rel.isEmpty:
        return None
    return rel.name


def gather_colliders(link_prim: Usd.Prim) -> tuple[str, ...]:
    def classify_collider_prim(collider_prim: Usd.Prim) -> str:
        if collider_prim.IsA(UsdGeom.Cylinder):
            return "Cylinder"
        if collider_prim.IsA(UsdGeom.Sphere):
            return "Sphere"
        if collider_prim.IsA(UsdGeom.Mesh):
            return "Mesh"
        if collider_prim.IsA(UsdGeom.Capsule):
            return "Capsule"
        if collider_prim.IsA(UsdGeom.Cube):
            return "Cube"
        for child_prim in Usd.PrimRange(collider_prim):
            if child_prim == collider_prim:
                continue
            if child_prim.IsA(UsdGeom.Cylinder):
                return "Cylinder"
            if child_prim.IsA(UsdGeom.Sphere):
                return "Sphere"
            if child_prim.IsA(UsdGeom.Mesh):
                return "Mesh"
            if child_prim.IsA(UsdGeom.Capsule):
                return "Capsule"
            if child_prim.IsA(UsdGeom.Cube):
                return "Cube"
        return collider_prim.GetTypeName() or collider_prim.GetName()

    colliders: list[str] = []
    for prim in Usd.PrimRange(link_prim):
        if prim == link_prim:
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(classify_collider_prim(prim))
    return tuple(colliders)


def vec3_attr_to_tuple(attr: Usd.Attribute | None) -> tuple[float, float, float] | None:
    if not attr:
        return None
    value = attr.Get()
    if value is None:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def quat_attr_to_quatd(attr: Usd.Attribute | None) -> Gf.Quatd | None:
    if not attr:
        return None
    value = attr.Get()
    if value is None:
        return None
    return quat_to_quatd(value)


def float_attr_value(attr: Usd.Attribute | None) -> float | None:
    if not attr:
        return None
    value = attr.Get()
    if value is None:
        return None
    return float(value)


def parse_usd(usd_path: Path) -> tuple[dict[str, UsdLink], dict[str, UsdJoint], Usd.Stage]:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"failed to open {usd_path}")

    links: dict[str, UsdLink] = {}
    joints: dict[str, UsdJoint] = {}

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass_api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else None
            mass = mass_api.GetMassAttr().Get() if mass_api else None
            com = mass_api.GetCenterOfMassAttr().Get() if mass_api else None
            inertia = reconstruct_inertia(
                mass_api.GetDiagonalInertiaAttr().Get() if mass_api else None,
                mass_api.GetPrincipalAxesAttr().Get() if mass_api else None,
            )
            links[prim.GetName()] = UsdLink(
                name=prim.GetName(),
                mass=float(mass) if mass is not None else None,
                com=(float(com[0]), float(com[1]), float(com[2])) if com is not None else None,
                inertia=inertia,
                collider_kinds=gather_colliders(prim),
            )
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            axis = None
            if prim.IsA(UsdPhysics.RevoluteJoint):
                axis = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                axis = UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get()
            joints[prim.GetName()] = UsdJoint(
                name=prim.GetName(),
                joint_type=prim.GetTypeName() or "Joint",
                parent=resolve_body_name(joint.GetBody0Rel().GetTargets()[0]) if joint.GetBody0Rel().GetTargets() else "",
                child=resolve_body_name(joint.GetBody1Rel().GetTargets()[0]) if joint.GetBody1Rel().GetTargets() else "",
                axis=axis,
                local_pos0=vec3_attr_to_tuple(prim.GetAttribute("physics:localPos0")),
                local_pos1=vec3_attr_to_tuple(prim.GetAttribute("physics:localPos1")),
                local_rot0=quat_attr_to_quatd(prim.GetAttribute("physics:localRot0")),
                local_rot1=quat_attr_to_quatd(prim.GetAttribute("physics:localRot1")),
                lower_limit=float_attr_value(prim.GetAttribute("physics:lowerLimit")),
                upper_limit=float_attr_value(prim.GetAttribute("physics:upperLimit")),
            )
    return links, joints, stage


def normalize_joint_type(type_name: str) -> str:
    lower = type_name.lower()
    if "fixed" in lower:
        return "fixed"
    if "revolute" in lower:
        return "revolute"
    if "prismatic" in lower:
        return "prismatic"
    return lower


def urdf_joint_zero_transform(joint: UrdfJoint):
    return compose_transform(joint.origin_xyz, rpy_to_quat(joint.origin_rpy))


def usd_joint_zero_transform(joint: UsdJoint):
    parent_to_joint = compose_transform(joint.local_pos0, joint.local_rot0)
    child_to_joint = compose_transform(joint.local_pos1, joint.local_rot1)
    return multiply_transforms(parent_to_joint, invert_transform(*child_to_joint))


def urdf_axis_in_parent_frame(joint: UrdfJoint) -> tuple[float, float, float] | None:
    axis = normalize_vector(joint.axis)
    if axis is None:
        return None
    return normalize_vector(rotate_vector(quat_to_rotation(rpy_to_quat(joint.origin_rpy)), axis))


def usd_axis_in_parent_frame(joint: UsdJoint) -> tuple[float, float, float] | None:
    axis = normalize_vector(axis_token_to_vector(joint.axis))
    if axis is None:
        return None
    return normalize_vector(rotate_vector(quat_to_rotation(joint.local_rot0), axis))


def urdf_limit_in_usd_units(joint: UrdfJoint, which: str) -> float | None:
    value = joint.lower_limit if which == "lower" else joint.upper_limit
    if value is None:
        return None
    if joint.joint_type == "revolute":
        return math.degrees(value)
    return value


def compare_all(input_dir: Path, usd_dir: Path) -> list[str]:
    issues: list[str] = []
    urdf_files = sorted(path for path in input_dir.iterdir() if path.suffix == ".urdf")
    usd_files = sorted(path for path in usd_dir.iterdir() if path.suffix == ".usd")
    urdf_stems = {path.stem for path in urdf_files}
    usd_stems = {path.stem for path in usd_files}

    if urdf_stems != usd_stems:
        issues.append(f"file stem mismatch: urdf_only={sorted(urdf_stems - usd_stems)} usd_only={sorted(usd_stems - urdf_stems)}")
        return issues

    for urdf_path in urdf_files:
        usd_path = usd_dir / f"{urdf_path.stem}.usd"
        urdf_links, urdf_joints = parse_urdf(urdf_path)
        usd_links, usd_joints, _ = parse_usd(usd_path)

        if {sanitize_name(name) for name in urdf_links} != set(usd_links):
            issues.append(f"{urdf_path.name}: rigid body name mismatch")
        if set(urdf_joints) != set(usd_joints):
            issues.append(f"{urdf_path.name}: joint name mismatch")

        for name, urdf_link in urdf_links.items():
            usd_link = usd_links.get(sanitize_name(name))
            if usd_link is None:
                continue
            if urdf_link.mass is not None and not approx_optional_float(urdf_link.mass, usd_link.mass, tol=1e-5):
                issues.append(
                    f"{urdf_path.name}: mass mismatch for {name}: urdf={urdf_link.mass} usd={usd_link.mass}"
                )
            if urdf_link.com is not None and not approx_tuple(urdf_link.com, usd_link.com, tol=1e-4):
                issues.append(
                    f"{urdf_path.name}: COM mismatch for {name}: urdf={urdf_link.com} usd={usd_link.com}"
                )
            if urdf_link.inertia is not None and not approx_tuple(urdf_link.inertia, usd_link.inertia, tol=1e-4):
                issues.append(
                    f"{urdf_path.name}: inertia mismatch for {name}: urdf={urdf_link.inertia} usd={usd_link.inertia}"
                )
            if len(urdf_link.collision_kinds) != len(usd_link.collider_kinds):
                issues.append(
                    f"{urdf_path.name}: collision count mismatch for {name}: urdf={urdf_link.collision_kinds} usd={usd_link.collider_kinds}"
                )
            elif not collider_kind_multiset_matches(urdf_link.collision_kinds, usd_link.collider_kinds):
                issues.append(
                    f"{urdf_path.name}: collision kind mismatch for {name}: urdf={urdf_link.collision_kinds} usd={usd_link.collider_kinds}"
                )

        for name, urdf_joint in urdf_joints.items():
            usd_joint = usd_joints.get(name)
            if usd_joint is None:
                continue
            if sanitize_name(urdf_joint.parent) != usd_joint.parent or sanitize_name(urdf_joint.child) != usd_joint.child:
                issues.append(
                    f"{urdf_path.name}: joint parent/child mismatch for {name}: "
                    f"urdf=({urdf_joint.parent}->{urdf_joint.child}) usd=({usd_joint.parent}->{usd_joint.child})"
                )
            usd_type = normalize_joint_type(usd_joint.joint_type)
            if urdf_joint.joint_type == "continuous":
                if usd_type != "revolute":
                    issues.append(f"{urdf_path.name}: continuous joint {name} became {usd_joint.joint_type}")
            elif usd_type != urdf_joint.joint_type:
                issues.append(f"{urdf_path.name}: joint type mismatch for {name}: urdf={urdf_joint.joint_type} usd={usd_joint.joint_type}")
            urdf_transform = urdf_joint_zero_transform(urdf_joint)
            usd_transform = usd_joint_zero_transform(usd_joint)
            if not approx_tuple(urdf_transform[0], usd_transform[0], tol=1e-4) or not approx_rotation(urdf_transform[1], usd_transform[1], tol=1e-4):
                issues.append(
                    f"{urdf_path.name}: joint zero-pose mismatch for {name}: "
                    f"urdf_xyz={urdf_transform[0]} usd_xyz={usd_transform[0]}"
                )
            urdf_axis = urdf_axis_in_parent_frame(urdf_joint)
            usd_axis = usd_axis_in_parent_frame(usd_joint)
            if not approx_tuple(urdf_axis, usd_axis, tol=1e-4):
                issues.append(
                    f"{urdf_path.name}: joint axis mismatch for {name}: urdf={urdf_axis} usd={usd_axis}"
                )
            urdf_lower = urdf_limit_in_usd_units(urdf_joint, "lower")
            urdf_upper = urdf_limit_in_usd_units(urdf_joint, "upper")
            if urdf_joint.joint_type == "continuous":
                if not is_effectively_unbounded_limit(usd_joint.lower_limit, usd_joint.upper_limit):
                    issues.append(
                        f"{urdf_path.name}: continuous joint limit mismatch for {name}: usd=({usd_joint.lower_limit}, {usd_joint.upper_limit})"
                    )
            else:
                if not approx_optional_float(urdf_lower, usd_joint.lower_limit, tol=1e-3):
                    issues.append(
                        f"{urdf_path.name}: lower limit mismatch for {name}: urdf={urdf_lower} usd={usd_joint.lower_limit}"
                    )
                if not approx_optional_float(urdf_upper, usd_joint.upper_limit, tol=1e-3):
                    issues.append(
                        f"{urdf_path.name}: upper limit mismatch for {name}: urdf={urdf_upper} usd={usd_joint.upper_limit}"
                    )
    return issues


def print_sample_details(input_dir: Path, usd_dir: Path, stem: str) -> None:
    urdf_path = input_dir / f"{stem}.urdf"
    usd_path = usd_dir / f"{stem}.usd"
    urdf_links, urdf_joints = parse_urdf(urdf_path)
    usd_links, usd_joints, stage = parse_usd(usd_path)

    print("\nSample asset:", stem)
    print("URDF:", urdf_path)
    print("USD :", usd_path)

    sample_links = [
        "base_footprint",
        "base_link",
        "wheel_fl_link",
        "wheel_fl_roller_1_link",
        "tool_tip_link",
    ]
    for name in sample_links:
        urdf_link = urdf_links[name]
        usd_link = usd_links.get(sanitize_name(name))
        print(f"\nLINK {name}")
        print("  urdf mass:", urdf_link.mass)
        print("  urdf com:", urdf_link.com)
        print("  urdf inertia:", urdf_link.inertia)
        print("  urdf collision kinds:", urdf_link.collision_kinds)
        print("  usd  mass:", usd_link.mass if usd_link else None)
        print("  usd  com:", usd_link.com if usd_link else None)
        print("  usd  inertia:", usd_link.inertia if usd_link else None)
        print("  usd  collider kinds:", usd_link.collider_kinds if usd_link else None)
        if name == "wheel_fl_link":
            prim = next(p for p in stage.Traverse() if p.GetName() == sanitize_name(name))
            for sub_prim in Usd.PrimRange(prim):
                if sub_prim == prim or not sub_prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                print("  usd  collider path:", sub_prim.GetPath(), "type:", sub_prim.GetTypeName())
                if sub_prim.IsA(UsdGeom.Cylinder):
                    cylinder = UsdGeom.Cylinder(sub_prim)
                    print("  usd  cylinder radius:", cylinder.GetRadiusAttr().Get())
                    print("  usd  cylinder height:", cylinder.GetHeightAttr().Get())

    sample_joints = [
        "base_link_joint",
        "wheel_fl_joint",
        "wheel_fl_roller_1_joint",
        "joint1",
        "tool_joint",
    ]
    for name in sample_joints:
        urdf_joint = urdf_joints[name]
        usd_joint = usd_joints[name]
        print(f"\nJOINT {name}")
        print("  urdf type:", urdf_joint.joint_type, "axis:", urdf_joint.axis)
        print("  urdf origin xyz/rpy:", urdf_joint.origin_xyz, urdf_joint.origin_rpy)
        print("  urdf limits:", urdf_joint.lower_limit, urdf_joint.upper_limit)
        print("  usd  type:", usd_joint.joint_type, "axis:", usd_joint.axis)
        print("  usd  local pos0/rot0:", usd_joint.local_pos0, usd_joint.local_rot0)
        print("  usd  local pos1/rot1:", usd_joint.local_pos1, usd_joint.local_rot1)
        print("  usd  limits:", usd_joint.lower_limit, usd_joint.upper_limit)
        print("  urdf zero pose:", urdf_joint_zero_transform(urdf_joint))
        print("  usd  zero pose:", usd_joint_zero_transform(usd_joint))
        print("  urdf axis in parent:", urdf_axis_in_parent_frame(urdf_joint))
        print("  usd  axis in parent:", usd_axis_in_parent_frame(usd_joint))
        print("  urdf parent->child:", urdf_joint.parent, "->", urdf_joint.child)
        print("  usd  parent->child:", usd_joint.parent, "->", usd_joint.child)

    print("\nNotes")
    print("  1. USD importer sanitizes names containing '-' into '_' for prim names.")
    print("  2. URDF continuous joints are represented as USD RevoluteJoint.")
    print("  3. Revolute joint limits are compared after URDF rad -> USD deg conversion.")
    print("  4. Joint zero-pose is compared from URDF origin against USD localPos/localRot composition.")
    print("  5. With collider_type=convex_hull, mesh collisions may appear as authored USD mesh-style colliders.")


def main() -> None:
    input_dir = Path(args_cli.input_dir).resolve() if args_cli.input_dir else DEFAULT_INPUT_DIR.resolve()
    usd_dir = Path(args_cli.usd_dir).resolve() if args_cli.usd_dir else DEFAULT_USD_DIR.resolve()

    issues = compare_all(input_dir, usd_dir)
    if issues:
        print("GLOBAL CHECK FAILED")
        for issue in issues:
            print(" -", issue)
        raise SystemExit(1)

    print("GLOBAL CHECK PASSED")
    print("  file stems: exact match")
    print("  rigid-body sets: exact match after USD name sanitization")
    print("  joint sets: exact match")
    print("  joint parent-child topology: exact match")
    print("  joint type mapping: exact match, with URDF continuous -> USD revolute")
    print("  joint zero-pose transforms: preserved within tolerance")
    print("  joint axis directions: preserved within tolerance in parent frame")
    print("  joint limits: preserved within tolerance, with revolute rad -> deg conversion")
    print("  link mass / center of mass / inertia tensors: preserved within tolerance")
    print("  collision counts: preserved for each rigid link")

    print_sample_details(input_dir, usd_dir, args_cli.sample)


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()
