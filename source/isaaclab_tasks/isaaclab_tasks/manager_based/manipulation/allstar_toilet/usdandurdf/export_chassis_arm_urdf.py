#!/usr/bin/env python3
"""Export chassis_arm USDA assemblies to URDF.

This script:
1. Extracts the 0730 mecanum chassis kinematic tree and rigid-body properties from USD.
2. Reuses the source XMS5 URDF files for arm inertial / collision / tool definitions.
3. Generates one combined URDF per `data/robots/chassis_arm/*.usda`.
4. Exports lightweight chassis OBJ meshes for wheel cores and rollers.

Run with a Python that can import `pxr` and `trimesh`.
"""

from __future__ import annotations

import copy
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import trimesh
from pxr import Gf, Usd, UsdGeom


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent.parent
DATA_ROBOTS_DIR = TASK_DIR / "data" / "robots"

CHASSIS_ARM_DIR = DATA_ROBOTS_DIR / "chassis_arm"
OUT_DIR = DATA_ROBOTS_DIR / "chassis_arm_urdf"
OUT_MESH_DIR = OUT_DIR / "meshes"

CHASSIS_STAGE_PATH = DATA_ROBOTS_DIR / "mecanum" / "Collected_0730Rot_v1.0" / "0730Rot_v1.0.usd"
BASE_LINK_STAGE_PATH = DATA_ROBOTS_DIR / "mecanum" / "Collected_0730Rot_v1.0" / "configuration" / "base_link_730.usd"
WHEEL_LEFT_STAGE_PATH = (
    DATA_ROBOTS_DIR
    / "mecanum"
    / "Collected_0730Rot_v1.0"
    / "configuration"
    / "4.5"
    / "Isaac"
    / "Robots"
    / "O3dyn"
    / "Parts"
    / "wheel_left_core.usd"
)
WHEEL_RIGHT_STAGE_PATH = WHEEL_LEFT_STAGE_PATH.with_name("wheel_right_core.usd")
ROLLER_STAGE_PATH = WHEEL_LEFT_STAGE_PATH.with_name("roller.usd")

ARM_URDF_DIR = DATA_ROBOTS_DIR / "xms5" / "urdf"

BASE_LINK_MESH_BOX = "base_link_box"
LEFT_WHEEL_MESH = "wheel_left_core.obj"
RIGHT_WHEEL_MESH = "wheel_right_core.obj"
ROLLER_MESH = "roller.obj"

COMBO_RE = re.compile(r"prepend references = @(.*/xms5/(?P<arm_usd>xm_urdf_[^@/]+)\.usd)@")
VEC3_RE = re.compile(r"\(([^)]+)\)")


@dataclass
class InertialData:
    mass: float
    com: tuple[float, float, float]
    inertia: tuple[float, float, float, float, float, float]


@dataclass
class Pose:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass
class LinkVisual:
    geometry: tuple[str, object]
    origin: Pose


def fmt(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return text or "0"


def vec_str(vec: tuple[float, float, float]) -> str:
    return " ".join(fmt(v) for v in vec)


def quat_to_matrix(quat: tuple[float, float, float, float]) -> list[list[float]]:
    w, x, y, z = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def matrix_to_rpy(m: list[list[float]]) -> tuple[float, float, float]:
    sy = math.sqrt(m[0][0] * m[0][0] + m[1][0] * m[1][0])
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(m[2][1], m[2][2])
        pitch = math.atan2(-m[2][0], sy)
        yaw = math.atan2(m[1][0], m[0][0])
    else:
        roll = math.atan2(-m[1][2], m[1][1])
        pitch = math.atan2(-m[2][0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def gf_quat_to_tuple(quat: Gf.Quatf | Gf.Quatd) -> tuple[float, float, float, float]:
    imag = quat.GetImaginary()
    return float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])


def gf_matrix_to_pose(matrix: Gf.Matrix4d) -> Pose:
    trans = matrix.ExtractTranslation()
    quat = gf_quat_to_tuple(matrix.ExtractRotation().GetQuaternion())
    return Pose(
        xyz=(float(trans[0]), float(trans[1]), float(trans[2])),
        rpy=matrix_to_rpy(quat_to_matrix(quat)),
    )


def mat_vec_mul(matrix: list[list[float]], vec: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vec[0] + matrix[0][1] * vec[1] + matrix[0][2] * vec[2],
        matrix[1][0] * vec[0] + matrix[1][1] * vec[1] + matrix[1][2] * vec[2],
        matrix[2][0] * vec[0] + matrix[2][1] * vec[1] + matrix[2][2] * vec[2],
    )


def transpose_mul(matrix: list[list[float]], vec: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vec[0] + matrix[1][0] * vec[1] + matrix[2][0] * vec[2],
        matrix[0][1] * vec[0] + matrix[1][1] * vec[1] + matrix[2][1] * vec[2],
        matrix[0][2] * vec[0] + matrix[1][2] * vec[1] + matrix[2][2] * vec[2],
    )


def normalize(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-12:
        return (1.0, 0.0, 0.0)
    return tuple(v / norm for v in vec)


def sanitize_link_name(path: str) -> str:
    parts = [p for p in path.split("/") if p and p != "Root"]
    if parts[-1] == "omniwheel":
        return parts[-2]
    return "_".join(parts)


def add_origin(parent: ET.Element, pose: Pose) -> None:
    ET.SubElement(parent, "origin", xyz=vec_str(pose.xyz), rpy=vec_str(pose.rpy))


def add_inertial(parent: ET.Element, inertial: InertialData) -> None:
    inertial_el = ET.SubElement(parent, "inertial")
    add_origin(inertial_el, Pose(inertial.com, (0.0, 0.0, 0.0)))
    ET.SubElement(inertial_el, "mass", value=fmt(inertial.mass))
    ixx, ixy, ixz, iyy, iyz, izz = inertial.inertia
    ET.SubElement(
        inertial_el,
        "inertia",
        ixx=fmt(ixx),
        ixy=fmt(ixy),
        ixz=fmt(ixz),
        iyy=fmt(iyy),
        iyz=fmt(iyz),
        izz=fmt(izz),
    )


def add_geometry(parent: ET.Element, visual: LinkVisual) -> None:
    tag = ET.SubElement(parent, "geometry")
    geometry_type, param = visual.geometry
    if geometry_type == "mesh":
        ET.SubElement(tag, "mesh", filename=param[0])  # type: ignore[index]
    elif geometry_type == "box":
        ET.SubElement(tag, "box", size=vec_str(param))  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")


def add_visual(parent: ET.Element, visual: LinkVisual) -> None:
    visual_el = ET.SubElement(parent, "visual")
    add_origin(visual_el, visual.origin)
    add_geometry(visual_el, visual)


def add_collision(parent: ET.Element, visual: LinkVisual) -> None:
    collision_el = ET.SubElement(parent, "collision")
    add_origin(collision_el, visual.origin)
    add_geometry(collision_el, visual)


def parse_usda_vec3(line: str) -> tuple[float, float, float]:
    match = VEC3_RE.search(line)
    if not match:
        raise ValueError(f"Unable to parse vec3 from: {line}")
    values = [float(part.strip()) for part in match.group(1).split(",")]
    return values[0], values[1], values[2]


def parse_combo_file(path: Path) -> dict[str, object]:
    arm_usd = None
    arm_translate = None
    root_local_pos0 = None

    for line in path.read_text().splitlines():
        if arm_usd is None:
            match = COMBO_RE.search(line)
            if match:
                arm_usd = match.group("arm_usd")
                continue
        if arm_translate is None and "xformOp:translate" in line:
            arm_translate = parse_usda_vec3(line)
            continue
        if root_local_pos0 is None and "physics:localPos0" in line:
            root_local_pos0 = parse_usda_vec3(line)

    if arm_usd is None or arm_translate is None or root_local_pos0 is None:
        raise RuntimeError(f"Failed to parse combo file: {path}")

    return {
        "arm_urdf": ARM_URDF_DIR / f"{arm_usd}.urdf",
        "arm_translate": arm_translate,
        "root_local_pos0": root_local_pos0,
    }


def mesh_faces(mesh: UsdGeom.Mesh) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    points = [(float(p[0]), float(p[1]), float(p[2])) for p in mesh.GetPointsAttr().Get()]
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces: list[list[int]] = []
    offset = 0
    for count in counts:
        if count == 3:
            faces.append([int(indices[offset]), int(indices[offset + 1]), int(indices[offset + 2])])
        else:
            for i in range(1, count - 1):
                faces.append([int(indices[offset]), int(indices[offset + i]), int(indices[offset + i + 1])])
        offset += count
    return points, faces


def transform_point(matrix: Gf.Matrix4d, point: tuple[float, float, float]) -> tuple[float, float, float]:
    vec = matrix.Transform(Gf.Vec3d(point[0], point[1], point[2]))
    return float(vec[0]), float(vec[1]), float(vec[2])


def export_single_mesh_obj(stage_path: Path, mesh_path: str, out_path: Path) -> trimesh.Trimesh:
    stage = Usd.Stage.Open(str(stage_path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
    points, faces = mesh_faces(mesh)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    world_points = [transform_point(matrix, point) for point in points]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated from USD\n")
        for x, y, z in world_points:
            handle.write(f"v {fmt(x)} {fmt(y)} {fmt(z)}\n")
        for f0, f1, f2 in faces:
            handle.write(f"f {f0 + 1} {f1 + 1} {f2 + 1}\n")

    return trimesh.Trimesh(vertices=world_points, faces=faces, process=False)


def compute_box_from_mesh_extent() -> LinkVisual:
    stage = Usd.Stage.Open(str(BASE_LINK_STAGE_PATH))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/base_link_730/node_/mesh_"))
    extent = mesh.GetExtentAttr().Get()
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())

    corners = []
    for x in (extent[0][0], extent[1][0]):
        for y in (extent[0][1], extent[1][1]):
            for z in (extent[0][2], extent[1][2]):
                corners.append(transform_point(matrix, (float(x), float(y), float(z))))

    min_x = min(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    min_z = min(c[2] for c in corners)
    max_x = max(c[0] for c in corners)
    max_y = max(c[1] for c in corners)
    max_z = max(c[2] for c in corners)

    size = (max_x - min_x, max_y - min_y, max_z - min_z)
    center = ((max_x + min_x) * 0.5, (max_y + min_y) * 0.5, (max_z + min_z) * 0.5)
    return LinkVisual(geometry=("box", size), origin=Pose(center, (0.0, 0.0, 0.0)))


def trimesh_mass_properties(mesh: trimesh.Trimesh, target_mass: float) -> InertialData:
    props = mesh.mass_properties
    raw_mass = float(props["mass"])
    if raw_mass <= 1e-12:
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        x, y, z = float(size[0]), float(size[1]), float(size[2])
        ixx = target_mass * (y * y + z * z) / 12.0
        iyy = target_mass * (x * x + z * z) / 12.0
        izz = target_mass * (x * x + y * y) / 12.0
        center = tuple(float(v) for v in ((bounds[0] + bounds[1]) * 0.5))
        return InertialData(target_mass, center, (ixx, 0.0, 0.0, iyy, 0.0, izz))

    scale = target_mass / raw_mass
    center = tuple(float(v) for v in props["center_mass"])
    inertia = props["inertia"] * scale
    return InertialData(
        target_mass,
        center,  # type: ignore[arg-type]
        (
            float(inertia[0][0]),
            float(inertia[0][1]),
            float(inertia[0][2]),
            float(inertia[1][1]),
            float(inertia[1][2]),
            float(inertia[2][2]),
        ),
    )


def principal_axes_to_inertia(diagonal: tuple[float, float, float], quat: tuple[float, float, float, float]) -> tuple[float, float, float, float, float, float]:
    norm = math.sqrt(sum(q * q for q in quat))
    if norm < 1e-12:
        rot = quat_to_matrix((1.0, 0.0, 0.0, 0.0))
    else:
        rot = quat_to_matrix(tuple(q / norm for q in quat))
    dxx, dyy, dzz = diagonal
    # R * D * R^T
    inertia = [[0.0, 0.0, 0.0] for _ in range(3)]
    diag = [[dxx, 0.0, 0.0], [0.0, dyy, 0.0], [0.0, 0.0, dzz]]
    for i in range(3):
        for j in range(3):
            inertia[i][j] = sum(rot[i][k] * diag[k][k] * rot[j][k] for k in range(3))
    return (
        inertia[0][0],
        inertia[0][1],
        inertia[0][2],
        inertia[1][1],
        inertia[1][2],
        inertia[2][2],
    )


def rewrite_arm_mesh_paths(link_el: ET.Element) -> None:
    for mesh in link_el.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        prefix = "package://pub_srdf_model/meshes/"
        if not filename.startswith(prefix):
            continue
        suffix = filename[len(prefix) :]
        if suffix.startswith("tool/"):
            mesh.attrib["filename"] = f"../xms5/meshes/{suffix}"
        else:
            mesh.attrib["filename"] = f"../xms5/meshes/arm/{suffix}"


def add_arm_subtree(robot_el: ET.Element, arm_urdf_path: Path) -> None:
    source_root = ET.parse(arm_urdf_path).getroot()

    for material_el in source_root.findall("material"):
        robot_el.append(copy.deepcopy(material_el))

    for link_el in source_root.findall("link"):
        name = link_el.attrib["name"]
        if name in {"base_link", "arm_base_link"}:
            continue
        cloned = copy.deepcopy(link_el)
        rewrite_arm_mesh_paths(cloned)
        robot_el.append(cloned)

    for joint_el in source_root.findall("joint"):
        name = joint_el.attrib["name"]
        if name in {"robot_base_fix", "base_link_fix"}:
            continue
        robot_el.append(copy.deepcopy(joint_el))

    kept_joint_names = {joint.attrib["name"] for joint in source_root.findall("joint") if joint.attrib["name"] not in {"robot_base_fix", "base_link_fix"}}
    for transmission_el in source_root.findall("transmission"):
        joint = transmission_el.find("joint")
        if joint is not None and joint.attrib.get("name") in kept_joint_names:
            robot_el.append(copy.deepcopy(transmission_el))


def make_link(name: str, inertial: InertialData | None, visuals: list[LinkVisual]) -> ET.Element:
    link_el = ET.Element("link", name=name)
    if inertial is not None:
        add_inertial(link_el, inertial)
    for visual in visuals:
        add_visual(link_el, visual)
        add_collision(link_el, visual)
    return link_el


def make_joint(
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    pose: Pose,
    axis: tuple[float, float, float] | None = None,
    damping: float | None = None,
) -> ET.Element:
    joint_el = ET.Element("joint", name=name, type=joint_type)
    ET.SubElement(joint_el, "parent", link=parent)
    ET.SubElement(joint_el, "child", link=child)
    add_origin(joint_el, pose)
    if axis is not None:
        ET.SubElement(joint_el, "axis", xyz=vec_str(normalize(axis)))
    if damping is not None:
        ET.SubElement(joint_el, "dynamics", damping=fmt(damping), friction="0")
    return joint_el


def extract_axis_in_child_frame(parent_world: Gf.Matrix4d, child_world: Gf.Matrix4d, joint_prim) -> tuple[float, float, float]:
    joint_rot = joint_prim.GetAttribute("physics:localRot0").Get()
    axis_token = joint_prim.GetAttribute("physics:axis").Get()

    joint_quat = gf_quat_to_tuple(joint_rot)
    joint_rot_mat = quat_to_matrix(joint_quat)
    parent_rot_mat = quat_to_matrix(gf_quat_to_tuple(parent_world.ExtractRotation().GetQuaternion()))
    child_rot_mat = quat_to_matrix(gf_quat_to_tuple(child_world.ExtractRotation().GetQuaternion()))

    axis_joint = {
        "X": (1.0, 0.0, 0.0),
        "Y": (0.0, 1.0, 0.0),
        "Z": (0.0, 0.0, 1.0),
    }[axis_token]
    axis_world = mat_vec_mul(parent_rot_mat, mat_vec_mul(joint_rot_mat, axis_joint))
    return normalize(transpose_mul(child_rot_mat, axis_world))


def extract_chassis_tree() -> tuple[dict[str, dict[str, object]], dict[str, InertialData], LinkVisual]:
    stage = Usd.Stage.Open(str(CHASSIS_STAGE_PATH))
    cache = UsdGeom.XformCache()

    base_visual = compute_box_from_mesh_extent()

    wheel_left_mesh = export_single_mesh_obj(WHEEL_LEFT_STAGE_PATH, "/Root/wheel_left_core", OUT_MESH_DIR / LEFT_WHEEL_MESH)
    wheel_right_mesh = export_single_mesh_obj(WHEEL_RIGHT_STAGE_PATH, "/Root/omniwheel_wheel_core", OUT_MESH_DIR / RIGHT_WHEEL_MESH)
    roller_mesh = export_single_mesh_obj(ROLLER_STAGE_PATH, "/Root/Roller", OUT_MESH_DIR / ROLLER_MESH)

    wheel_mass_props = {
        LEFT_WHEEL_MESH: trimesh_mass_properties(wheel_left_mesh, 12.0),
        RIGHT_WHEEL_MESH: trimesh_mass_properties(wheel_right_mesh, 12.0),
    }

    base_prim = stage.GetPrimAtPath("/Root/base_link")
    base_inertial = InertialData(
        mass=float(base_prim.GetAttribute("physics:mass").Get()),
        com=tuple(float(v) for v in base_prim.GetAttribute("physics:centerOfMass").Get()),
        inertia=principal_axes_to_inertia(
            tuple(float(v) for v in base_prim.GetAttribute("physics:diagonalInertia").Get()),
            gf_quat_to_tuple(base_prim.GetAttribute("physics:principalAxes").Get()),
        ),
    )

    roller_prim = stage.GetPrimAtPath("/Root/wheel_fl_link/Roller_1")
    roller_inertial = InertialData(
        mass=float(roller_prim.GetAttribute("physics:mass").Get()),
        com=tuple(float(v) for v in roller_prim.GetAttribute("physics:centerOfMass").Get()),
        inertia=principal_axes_to_inertia(
            tuple(float(v) for v in roller_prim.GetAttribute("physics:diagonalInertia").Get()),
            gf_quat_to_tuple(roller_prim.GetAttribute("physics:principalAxes").Get()),
        ),
    )

    links: dict[str, dict[str, object]] = {}
    inertials: dict[str, InertialData] = {"base_link": base_inertial}

    base_world = cache.GetLocalToWorldTransform(base_prim)
    joint_scope = stage.GetPrimAtPath("/Root/joints")
    for joint_prim in joint_scope.GetChildren():
        joint_name = joint_prim.GetName()
        child_path = str(joint_prim.GetRelationship("physics:body1").GetTargets()[0])
        child_prim = stage.GetPrimAtPath(child_path)
        child_world = cache.GetLocalToWorldTransform(child_prim)
        relative_pose = gf_matrix_to_pose(base_world.GetInverse() * child_world)
        axis = extract_axis_in_child_frame(base_world, child_world, joint_prim)

        payload = child_prim.GetMetadata("payload")
        payload_name = str(payload.explicitItems[0].assetPath).split("/")[-1] if payload and payload.explicitItems else "wheel_left_core.usd"
        mesh_name = LEFT_WHEEL_MESH if "left" in payload_name else RIGHT_WHEEL_MESH
        link_name = sanitize_link_name(child_path)

        links[link_name] = {
            "parent": "base_link",
            "joint_name": joint_name,
            "joint_pose": relative_pose,
            "joint_axis": axis,
            "joint_damping": float(joint_prim.GetAttribute("drive:angular:physics:damping").Get()),
            "mesh_name": mesh_name,
            "rollers": [],
        }
        inertials[link_name] = wheel_mass_props[mesh_name]

    for wheel_link, data in links.items():
        wheel_prim = stage.GetPrimAtPath(f"/Root/{wheel_link}/omniwheel")
        wheel_world = cache.GetLocalToWorldTransform(wheel_prim)
        for roller_joint_prim in stage.GetPrimAtPath(f"/Root/{wheel_link}").GetChildren():
            if not roller_joint_prim.GetTypeName().startswith("PhysicsRevoluteJoint"):
                continue
            child_path = str(roller_joint_prim.GetRelationship("physics:body1").GetTargets()[0])
            child_prim = stage.GetPrimAtPath(child_path)
            child_world = cache.GetLocalToWorldTransform(child_prim)
            relative_pose = gf_matrix_to_pose(wheel_world.GetInverse() * child_world)
            axis = extract_axis_in_child_frame(wheel_world, child_world, roller_joint_prim)
            roller_link_name = sanitize_link_name(child_path)
            data["rollers"].append(
                {
                    "link_name": roller_link_name,
                    "joint_name": f"{roller_link_name}_joint",
                    "joint_pose": relative_pose,
                    "joint_axis": axis,
                    "joint_damping": float(roller_joint_prim.GetAttribute("drive:angular:physics:damping").Get()),
                }
            )
            inertials[roller_link_name] = roller_inertial

    return links, inertials, base_visual


def build_robot_urdf(stem: str, combo: dict[str, object], chassis_links: dict[str, dict[str, object]], inertials: dict[str, InertialData], base_visual: LinkVisual) -> ET.ElementTree:
    robot_el = ET.Element("robot", name=stem)
    robot_el.append(ET.Comment("Generated from chassis_arm USDA + xms5 source URDF."))
    robot_el.append(
        ET.Comment(
            "USD arm mount pose preserved from /robot/arm xformOp:translate; "
            f"source PhysX root_joint base anchor localPos0={combo['root_local_pos0']}."
        )
    )

    robot_el.append(make_link("base_link", inertials["base_link"], [base_visual]))

    for wheel_link, data in chassis_links.items():
        mesh_path = f"meshes/{data['mesh_name']}"
        wheel_visual = LinkVisual(geometry=("mesh", (mesh_path,)), origin=Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        robot_el.append(make_link(wheel_link, inertials[wheel_link], [wheel_visual]))
        robot_el.append(
            make_joint(
                data["joint_name"],  # type: ignore[arg-type]
                "continuous",
                "base_link",
                wheel_link,
                data["joint_pose"],  # type: ignore[arg-type]
                data["joint_axis"],  # type: ignore[arg-type]
                data["joint_damping"],  # type: ignore[arg-type]
            )
        )

        for roller in data["rollers"]:  # type: ignore[index]
            roller_visual = LinkVisual(geometry=("mesh", (f"meshes/{ROLLER_MESH}",)), origin=Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
            robot_el.append(make_link(roller["link_name"], inertials[roller["link_name"]], [roller_visual]))
            robot_el.append(
                make_joint(
                    roller["joint_name"],
                    "continuous",
                    wheel_link,
                    roller["link_name"],
                    roller["joint_pose"],
                    roller["joint_axis"],
                    roller["joint_damping"],
                )
            )

    add_arm_subtree(robot_el, combo["arm_urdf"])  # type: ignore[arg-type]
    arm_mount_pose = Pose(combo["arm_translate"], (0.0, 0.0, 0.0))  # type: ignore[arg-type]
    robot_el.append(make_joint("root_joint", "fixed", "base_link", "XMS5_R800_W4G3B4C_base", arm_mount_pose))

    return ET.ElementTree(robot_el)


def write_readme(base_visual: LinkVisual) -> None:
    readme = OUT_DIR / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# chassis_arm_urdf",
                "",
                "本目录由 `usdandurdf/export_chassis_arm_urdf.py` 生成。",
                "",
                "说明：",
                "- 为了保持 `chassis_arm/*.usda` 的组合语义，机械臂根部挂载使用了 USDA 中 `/robot/arm` 的实际组合位姿。",
                "- 原始 PhysX `root_joint.physics:localPos0` 作为注释保留在生成的 URDF 头部。",
                "- `base_link_730.usd` 含约 612 万顶点，不适合直接塞进 URDF；因此底盘本体 visual/collision 使用其包围盒近似。",
                f"- 当前 `base_link` 包围盒尺寸: `{vec_str(base_visual.geometry[1])}`，中心: `{vec_str(base_visual.origin.xyz)}`。",
                "- 轮毂与 roller mesh 已导出为 `obj`，质量参数来自 USD；轮毂惯量因 USD 中未显式给出，采用导出 mesh 的体积分布按质量缩放计算。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MESH_DIR.mkdir(parents=True, exist_ok=True)

    chassis_links, inertials, base_visual = extract_chassis_tree()

    combo_paths = sorted(CHASSIS_ARM_DIR.glob("*.usda"))
    for combo_path in combo_paths:
        combo = parse_combo_file(combo_path)
        tree = build_robot_urdf(combo_path.stem, combo, chassis_links, inertials, base_visual)
        ET.indent(tree, space="  ")
        out_path = OUT_DIR / f"{combo_path.stem}.urdf"
        tree.write(out_path, encoding="utf-8", xml_declaration=False)

    write_readme(base_visual)
    print(f"Generated {len(combo_paths)} URDF files into {OUT_DIR}")


if __name__ == "__main__":
    main()
