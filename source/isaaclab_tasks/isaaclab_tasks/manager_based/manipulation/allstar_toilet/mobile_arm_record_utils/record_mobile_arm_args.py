"""Argument parsing helpers for mobile-arm recording."""

import argparse

from .record_mobile_arm_tasks import (
    _CAMERA_DISABLED_TASKS,
    _DEFAULT_GRIPPER_JOINT1_OUTWARD_OPEN_RAD,
    is_quiet_runtime_debug_task,
)


def _parse_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def build_record_mobile_arm_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record mobile-arm demonstrations.")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Allstar-Toilet-Mecanum0730-XMS5Gripper-v0",
        help="Registered gym environment id.",
    )
    parser.add_argument(
        "--speed_scale",
        type=float,
        default=20.0,
        help="Chassis wheel speed scale: sensitivity * scale = wheel rad/s.",
    )
    parser.add_argument(
        "--arm_scale",
        type=float,
        default=0.05,
        help="Arm joint position delta per keypress (rad).",
    )
    parser.add_argument(
        "--step_hz",
        type=int,
        default=30,
        help="Target control loop rate in Hz. 0 disables rate limiting.",
    )
    parser.add_argument(
        "--teleop_device",
        type=str,
        default="keyboard",
        choices=("keyboard", "spacemouse"),
        help="Teleoperation device for this script only.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="./datasets/mobile_arm_dataset.hdf5",
        help="Output HDF5 file path for recorded demonstrations.",
    )
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=0.0,
        help="Episode length in simulation seconds. 0 = use env cfg default (30 s).",
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=0,
        help="Number of demonstrations to record. 0 = unlimited.",
    )
    parser.add_argument(
        "--save_obs",
        type=_parse_bool,
        default=False,
        metavar="True|False",
        help="Save RGBD + TF observations to outputs/<timestamp>/. Default: False.",
    )
    parser.add_argument(
        "--show_camera_pose_debug",
        type=_parse_bool,
        default=False,
        metavar="True|False",
        help="Draw cam_tool position, axes, and frustum in the viewport. Default: False.",
    )
    parser.add_argument(
        "--arm_debug_interval",
        type=int,
        default=0,
        help="Print compact arm joint debug every N completed env.step calls. 0 disables periodic joint debug.",
    )
    parser.add_argument(
        "--camera_debug_interval",
        type=int,
        default=0,
        help="Print RGB/depth/camera-pose diagnostics every N completed env.step calls. 0 disables periodic camera debug.",
    )
    parser.add_argument(
        "--gripper_joint1_pos",
        type=float,
        default=_DEFAULT_GRIPPER_JOINT1_OUTWARD_OPEN_RAD,
        help="Raw gripper_joint1 position target in radians when the task exposes gripper_joint_pos. Default fully opens outward.",
    )
    parser.add_argument(
        "--prismatic_gripper_pos",
        type=float,
        default=0.0,
        help="Initial prismatic gripper close command in meters for the prismatic gripper task.",
    )
    parser.add_argument(
        "--prismatic_gripper_step",
        type=float,
        default=0.001,
        help="Prismatic gripper keyboard increment per control step in meters.",
    )
    parser.add_argument(
        "--prismatic_gripper_min",
        type=float,
        default=0.0,
        help="Minimum prismatic gripper close command in meters.",
    )
    parser.add_argument(
        "--prismatic_gripper_max",
        type=float,
        default=0.025,
        help="Maximum prismatic gripper close command in meters.",
    )
    parser.add_argument(
        "--keyboard_hold_sec",
        type=float,
        default=0.5,
        help="Terminal-keyboard fallback hold window in seconds. Increase this if key presses are missed.",
    )
    return parser


def finalize_record_mobile_arm_args(args_cli) -> bool:
    args_cli.enable_cameras = args_cli.task not in _CAMERA_DISABLED_TASKS
    return is_quiet_runtime_debug_task(args_cli.task)
