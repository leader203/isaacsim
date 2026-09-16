"""Pure task-profile helpers for cuRobo pose-plan scripts."""

from __future__ import annotations

import math
from pathlib import Path


DEFAULT_BRUSH_END_EFFECTOR_ROLL_RAD = 3.0 * math.pi / 4.0
DEFAULT_BRUSH_TARGET_QUAT_WXYZ = [0.0, 1.0, 0.0, 0.0]
DEFAULT_GRIPPER_JOINT_TARGET_QUAT_WXYZ = [-0.004, -0.715, -0.699, 0.016]
DEFAULT_BRUSH_PRE_TARGET_JOINT_POS_DEG = [2.84, -0.04, -68.72, -2.86, -123.20, 0.0]
DEFAULT_GRIPPER_JOINT_PRE_TARGET_JOINT_POS_DEG = [5.699887, -11.49924, -45.801613, -0.000241, -8.342674, 0.0]

DEFAULT_WORLD_POSE_TESTS = [
    ("toilet_pose_01", [1.663, -0.980, 0.408], [0.000000, 1.000000, 0.000000, 0.000000]),
    ("toilet_pose_02", [1.648, -1.090, 0.408], [0.000000, 1.000000, 0.000000, 0.000000]),
    ("toilet_pose_03", [1.656, -1.152, 0.408], [0.000000, 1.000000, 0.000000, 0.000000]),
    ("toilet_pose_04", [1.695, -1.176, 0.408], [0.000000, 1.000000, 0.000000, 0.000000]),
    ("toilet_pose_05", [1.730, -1.211, 0.408], [0.189000, 0.925000, -0.325000, 0.049000]),
    ("toilet_pose_06", [1.775, -1.216, 0.408], [0.189000, 0.925000, -0.325000, 0.049000]),
    ("toilet_pose_07", [1.824, -1.217, 0.408], [0.189000, 0.925000, -0.325000, 0.049000]),
    ("toilet_pose_08", [1.854, -1.217, 0.408], [0.189000, 0.925000, -0.325000, 0.049000]),
    ("toilet_pose_09", [1.903, -1.218, 0.408], [0.189000, 0.925000, -0.325000, 0.049000]),
    ("toilet_pose_10", [1.933, -1.199, 0.408], [0.000000, 1.000000, 0.000000, 0.000000]),
    ("toilet_pose_11", [1.948, -1.174, 0.408], [0.000000, 1.000000, 0.000000, 0.000000]),
    ("toilet_pose_12", [1.964, -1.131, 0.408], [0.000000, 1.000000, 0.000000, 0.000000]),
    ("toilet_pose_13", [1.960, -1.082, 0.408], [0.000000, 1.000000, 0.000000, 0.000000]),
    ("toilet_pose_14", [1.963, -1.009, 0.408], [0.000000, 1.000000, 0.000000, 0.000000]),
    ("toilet_pose_15", [1.963, -1.009, 0.408], [0.000000, 1.000000, 0.000000, 0.000000]),
    ("toilet_pose_16", [1.959, -0.984, 0.408], [0.000000, 1.000000, 0.000000, 0.000000]),
]


def parse_joint_names(value: str) -> list[str]:
    joint_names = [item.strip() for item in value.split(",") if item.strip()]
    if not joint_names:
        raise ValueError("--joint_names must contain at least one joint name.")
    return joint_names


def default_curobo_robot_cfg_path(task_root: str | Path, urdf_path: str | Path) -> Path:
    return Path(task_root) / "configs" / "curobo" / f"{Path(urdf_path).stem}.yml"


def _robot_usd_stem_for_urdf(robot_usd_path: str | Path) -> str:
    stem = Path(robot_usd_path).stem
    for suffix in ("_base", "_physics", "_sensor"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def urdf_path_from_robot_usd(robot_usd_path: str | Path, *, task_root: str | Path) -> Path:
    robot_usd_path = Path(robot_usd_path).expanduser()
    task_root = Path(task_root)
    urdf_stem = _robot_usd_stem_for_urdf(robot_usd_path)
    urdf_root = task_root / "data/robots/chassis_arm/motion_wheel_arm_simple_sphere_urdf"

    parts = list(robot_usd_path.parts)
    if "motion_wheel_arm_simple_sphere_usd" in parts:
        usd_dir_idx = parts.index("motion_wheel_arm_simple_sphere_usd")
        inferred_root = Path(*parts[:usd_dir_idx]) / "motion_wheel_arm_simple_sphere_urdf"
        if inferred_root.is_dir():
            urdf_root = inferred_root

    urdf_path = urdf_root / f"{urdf_stem}.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(
            "Could not resolve cuRobo URDF from the scene robot USD. "
            f"robot_usd_path={robot_usd_path} expected_urdf_path={urdf_path}"
        )
    return urdf_path


def is_gripper_joint_profile(task: str, urdf_path: str | Path | None = None) -> bool:
    task_lower = str(task).lower()
    urdf_lower = str(urdf_path or "").lower()
    return (
        "gripperjoint-curoboposeplantest" in task_lower
        or "gripperjointprismatic-curoboposeplantest" in task_lower
        or "gripperjointprismatic-gripperjointprismatic-curoboposeplantest" in task_lower
        or "gripperjointprismaticsanheyifull-gripperjointprismatic-curoboposeplantest" in task_lower
        or "gripper_joint" in urdf_lower
    )


def is_sanheyi_full_prismatic_gripper_joint_profile(task: str, urdf_path: str | Path | None = None) -> bool:
    task_lower = str(task).lower()
    urdf_lower = str(urdf_path or "").lower()
    return "gripperjointprismaticsanheyifull" in task_lower or "gripper_joint_prismatic_sanheyi_full" in urdf_lower


def is_prismatic_gripper_joint_profile(task: str, urdf_path: str | Path | None = None) -> bool:
    task_lower = str(task).lower()
    urdf_lower = str(urdf_path or "").lower()
    return (
        "gripperjointprismatic-curoboposeplantest" in task_lower
        or "gripperjointprismatic-gripperjointprismatic-curoboposeplantest" in task_lower
        or "gripperjointprismaticsanheyifull-gripperjointprismatic-curoboposeplantest" in task_lower
        or "gripper_joint_prismatic" in urdf_lower
    )


def resolve_curobo_profile(
    args_cli,
    *,
    task_root: str | Path,
    isaaclab_root: str | Path,
    robot_usd_path: str | Path | None = None,
) -> tuple[Path, Path, str, str, list[str], list[float], float | None, list[float]]:
    task_root = Path(task_root)

    if args_cli.urdf_path:
        urdf_path = Path(args_cli.urdf_path).expanduser()
    elif robot_usd_path:
        urdf_path = urdf_path_from_robot_usd(robot_usd_path, task_root=task_root)
    else:
        raise ValueError("robot_usd_path is required when --urdf_path is not provided.")

    robot_cfg_path = (
        Path(args_cli.curobo_robot_cfg_path).expanduser()
        if args_cli.curobo_robot_cfg_path
        else default_curobo_robot_cfg_path(task_root, urdf_path)
    )
    joint_names = parse_joint_names(args_cli.joint_names)
    if args_cli.default_target_quat_wxyz is not None:
        default_target_quat = list(args_cli.default_target_quat_wxyz)
    elif is_gripper_joint_profile(args_cli.task, urdf_path):
        default_target_quat = list(DEFAULT_GRIPPER_JOINT_TARGET_QUAT_WXYZ)
    else:
        default_target_quat = list(DEFAULT_BRUSH_TARGET_QUAT_WXYZ)

    if args_cli.plan_end_effector_roll_rad is not None:
        plan_end_effector_roll_rad = args_cli.plan_end_effector_roll_rad
    else:
        plan_end_effector_roll_rad = (
            DEFAULT_BRUSH_END_EFFECTOR_ROLL_RAD
            if not is_gripper_joint_profile(args_cli.task, urdf_path)
            else None
        )

    if args_cli.pre_target_joint_pos_deg is not None:
        pre_target_joint_pos_deg = list(args_cli.pre_target_joint_pos_deg)
    elif is_gripper_joint_profile(args_cli.task, urdf_path):
        pre_target_joint_pos_deg = list(DEFAULT_GRIPPER_JOINT_PRE_TARGET_JOINT_POS_DEG)
    else:
        pre_target_joint_pos_deg = list(DEFAULT_BRUSH_PRE_TARGET_JOINT_POS_DEG)

    return (
        urdf_path,
        robot_cfg_path,
        args_cli.base_link,
        args_cli.tool_frame,
        joint_names,
        default_target_quat,
        plan_end_effector_roll_rad,
        pre_target_joint_pos_deg,
    )
