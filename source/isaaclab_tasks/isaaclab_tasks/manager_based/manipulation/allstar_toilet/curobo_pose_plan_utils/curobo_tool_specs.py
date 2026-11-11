"""Pure helpers and tool specs for cuRobo pose-plan tool APIs."""

from __future__ import annotations


def float_list_parameter(parameters: dict, key: str, length: int, *, required: bool) -> list[float] | None:
    value = parameters.get(key)
    if value is None:
        if required:
            raise ValueError(f"{key} must be a length-{length} list.")
        return None
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{key} must be a length-{length} list.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must contain only numbers.") from exc


def default_brush_target_quat_wxyz(target_pos: list[float]) -> list[float]:
    if target_pos[1] < -1.200:
        return [0.189, 0.925, -0.325, 0.049]
    return [0.0, 1.0, 0.0, 0.0]


def apply_brush_target_safe_clearance(target_pos: list[float]) -> tuple[list[float], float]:
    return list(target_pos), 0.0


def apply_brush_target_forced_safe_clearance(target_pos: list[float]) -> tuple[list[float], float]:
    return list(target_pos), 0.0


def downsample_ordered_pose_tests(
    pose_tests: list[tuple[str, list[float], list[float]]],
    *,
    max_count: int,
) -> list[tuple[str, list[float], list[float]]]:
    if len(pose_tests) <= max_count:
        return pose_tests
    if max_count < 2:
        raise ValueError(f"max_count must be >= 2, got {max_count}")
    middle_count = max_count - 2
    middle = pose_tests[1:-1]
    if middle_count <= 0:
        return [pose_tests[0], pose_tests[-1]]
    if middle_count >= len(middle):
        return pose_tests
    if middle_count == 1:
        selected_middle = [middle[len(middle) // 2]]
    else:
        selected_middle = []
        used_indices = set()
        for sample_idx in range(middle_count):
            raw_idx = sample_idx * (len(middle) - 1) / float(middle_count - 1)
            middle_idx = int(round(raw_idx))
            while middle_idx in used_indices and middle_idx + 1 < len(middle):
                middle_idx += 1
            while middle_idx in used_indices and middle_idx > 0:
                middle_idx -= 1
            used_indices.add(middle_idx)
            selected_middle.append(middle[middle_idx])
    return [pose_tests[0], *selected_middle, pose_tests[-1]]


def brush_execute_point_clean_spec() -> dict:
    return {
        "name": "brush_execute_point_clean",
        "description": (
            "Plan and execute one tool_tip_link target pose for the brush_rolling_0512 end effector. "
            "The arm first moves to the planning joint pose, then cuRobo plans directly to the target. "
            "At the final waypoint, joint6 rolls +180 degrees and holds that final rolled pose without "
            "rolling back or returning to the planning joint pose."
        ),
        "parameters": {
            "name": {
                "required": False,
                "type": "string",
                "description": "Optional case name shown in job results.",
            },
            "target_frame": {
                "required": False,
                "type": "string",
                "allowed": ["world", "base"],
                "default": "world",
                "description": (
                    "Frame of target_pos and target_quat. world means Isaac world frame; "
                    "base means the XMS5 arm base_link frame."
                ),
            },
            "target_pos": {
                "required": True,
                "type": "array[float]",
                "length": 3,
                "description": (
                    "Target tool_tip_link position [x, y, z] in target_frame. "
                    "The position is used directly without an automatic z offset."
                ),
            },
            "target_quat": {
                "required": False,
                "type": "array[float]",
                "length": 4,
                "convention": "wxyz",
                "default": [0.0, 1.0, 0.0, 0.0],
                "conditional_default": {
                    "when": "target_pos[1] < -1.200 and target_quat is omitted",
                    "value": [0.189, 0.925, -0.325, 0.049],
                },
                "description": (
                    "Optional target tool_tip_link orientation [w, x, y, z] in target_frame. "
                    "When omitted, defaults to a world-down orientation where local +Z points along world -Z; "
                    "if target_pos[1] < -1.200, the omitted default is [0.189, 0.925, -0.325, 0.049]."
                ),
            },
        },
    }


def brush_execute_spot_clean_spec() -> dict:
    return {
        "name": "brush_execute_spot_clean",
        "description": (
            "Plan and execute one continuous brush_rolling_0512 spot-clean path through multiple "
            "tool_tip_link target positions without automatic target offsets. "
            "If more than 8 targets are supplied, the first and last targets are kept and 6 middle "
            "targets are sampled in order. Joint6 starts rolling by the requested angle after the first target "
            "waypoint and finishes by the last target waypoint; the robot holds the final pose without "
            "rolling back or returning. The default roll angle is +180 degrees."
        ),
        "parameters": {
            "name": {
                "required": False,
                "type": "string",
                "description": "Optional case name shown in job results.",
            },
            "target_frame": {
                "required": False,
                "type": "string",
                "allowed": ["world", "base"],
                "default": "world",
                "description": "Frame of all target_positions and optional target_quats.",
            },
            "target_positions": {
                "required": True,
                "type": "array[array[float]]",
                "inner_length": 3,
                "max_planning_count": 8,
                "description": (
                    "Ordered target positions [[x, y, z], ...]. Positions are used directly "
                    "before frame conversion and planning."
                ),
            },
            "target_quats": {
                "required": False,
                "type": "array[array[float]]",
                "inner_length": 4,
                "convention": "wxyz",
                "description": (
                    "Optional orientations matching target_positions. When omitted, each target uses "
                    "the same default orientation logic as point clean."
                ),
            },
            "seed_mode": {
                "required": False,
                "type": "string",
                "allowed": ["linear", "cubic", "quintic"],
                "default": "cubic",
                "description": "Interpolation label passed to the multi-pose planner.",
            },
            "final_pos_tolerance_m": {
                "required": False,
                "type": "float",
                "default": 0.015,
                "description": (
                    "After the joint6 roll, keep stepping at the final command until "
                    "the actual tool_tip_link position in base frame is within this distance of "
                    "the final target."
                ),
            },
            "joint6_roll_deg": {
                "required": False,
                "type": "float",
                "default": 180.0,
                "description": (
                    "Joint6 roll angle in degrees, blended from the first target waypoint to the last. "
                    "Ignored when joint6_roll_rad is provided."
                ),
            },
            "joint6_roll_rad": {
                "required": False,
                "type": "float",
                "default": None,
                "description": (
                    "Joint6 roll angle in radians, blended from the first target waypoint to the last. "
                    "Overrides joint6_roll_deg when provided."
                ),
            },
            "final_rot_tolerance_rad": {
                "required": False,
                "type": "float",
                "default": None,
                "description": (
                    "Optional final orientation tolerance in radians. Omit this for rolling-brush "
                    "paths where joint6 rotation changes the tool quaternion while preserving the "
                    "cleaning contact position."
                ),
            },
        },
    }


def stretch_end_tool_spec() -> dict:
    return {
        "name": "stretch_end_tool",
        "description": (
            "Use cuRobo C-space planning to move the arm to a stretch joint target, then plan back "
            "to the shared planning/pre-target joint pose and hold there."
        ),
        "parameters": {
            "name": {
                "required": False,
                "type": "string",
                "description": "Optional case name shown in job results.",
            },
            "joint_pos_deg": {
                "required": False,
                "type": "array[float]",
                "length": 6,
                "default": [5.7, -14.4, -20.0, 0.0, 8.6, 0.0],
                "description": "Stretch joint target in degrees for joint1..joint6.",
            },
        },
    }


def brush_move_arm_to_joints_spec() -> dict:
    return {
        "name": "brush_move_arm_to_joints",
        "description": (
            "Use cuRobo C-space planning to move the arm from its current joint state back to the "
            "shared planning/pre-target joint pose, then hold still there. If cuRobo fails or returns "
            "no executable trajectory, fall back to a direct joint move to the same planning pose."
        ),
        "parameters": {
            "name": {
                "required": False,
                "type": "string",
                "description": "Optional case name shown in job results.",
            },
        },
    }


def base_keyboard_teleop_spec() -> dict:
    return {
        "name": "base_keyboard_teleop",
        "description": (
            "Shared brush/gripper keyboard teleop tool. First plans and executes a cuRobo C-space "
            "arm move to a fixed pre-teleop posture, then controls the mobile base with motion-based "
            "root pose/root velocity updates while synchronizing wheel visuals. The arm is held at "
            "the pre-teleop posture during base motion."
        ),
        "parameters": {
            "name": {
                "required": False,
                "type": "string",
                "description": "Optional case name shown in job results.",
            },
            "pre_arm_joint_pos_deg": {
                "required": False,
                "type": "array[float]",
                "length": 6,
                "default": [-2.89, -22.96, -28.61, 0.0, -137.52, 0.0],
                "description": "Arm C-space pre-teleop target in degrees for joint1..joint6.",
            },
            "speed_scale": {
                "required": False,
                "type": "float",
                "default": 150.0,
                "description": "Keyboard base speed scale, matching run_record_mobile_arm.py operator feel.",
            },
            "max_duration_s": {
                "required": False,
                "type": "float",
                "default": 300.0,
                "description": "Maximum keyboard-control duration in seconds. Use <=0 to run until stop_key.",
            },
            "stop_key": {
                "required": False,
                "type": "string",
                "default": "Q",
                "description": "Keyboard key that exits base teleop.",
            },
            "keyboard_hold_sec": {
                "required": False,
                "type": "float",
                "default": 0.12,
                "description": "Terminal keyboard fallback hold window in seconds.",
            },
        },
    }


def gripper_prismatic_execute_grab_spec() -> dict:
    return {
        "name": "gripper_prismatic_execute_grab",
        "description": (
            "For prismatic gripper tasks, optionally move to the shared planning/pre-target joint pose, "
            "plan and execute one tool_tip_link target pose, descend in world -Z with the gripper open, "
            "close the prismatic gripper in place, then lift in world +Z while holding the grasp."
        ),
        "parameters": {
            "name": {
                "required": False,
                "type": "string",
                "description": "Optional case name shown in job results.",
            },
            "target_frame": {
                "required": False,
                "type": "string",
                "allowed": ["world", "base"],
                "default": "world",
                "description": (
                    "Frame of target_pos and target_quat. world means Isaac world frame; "
                    "base means the XMS5 arm base_link frame."
                ),
            },
            "target_pos": {
                "required": True,
                "type": "array[float]",
                "length": 3,
                "description": (
                    "Target tool_tip_link position [x, y, z] in target_frame before descend/grasp/lift. "
                    "When paper_cleanup_mode=true, this field is ignored and the paper position is auto-detected."
                ),
            },
            "target_quat": {
                "required": False,
                "type": "array[float]",
                "length": 4,
                "convention": "wxyz",
                "description": (
                    "Optional target tool_tip_link orientation [w, x, y, z] in target_frame. "
                    "When omitted, the prismatic gripper profile default target orientation is used."
                ),
            },
            "move_to_pre_target": {
                "required": False,
                "type": "bool",
                "default": False,
                "description": (
                    "Move to the shared planning/pre-target joint pose before planning the target. "
                    "Default False: plan straight to the target from the current arm pose."
                ),
            },
            "paper_cleanup_mode": {
                "required": False,
                "type": "bool",
                "default": False,
                "description": (
                    "When true, auto-detect the paper-waste position as the grab target and enable "
                    "logical attach/detach. After grasp, the paper follows the gripper. "
                    "The take_out tool will detach and check if the paper entered the collection zone."
                ),
            },
            "paper_prim_path": {
                "required": False,
                "type": "string",
                "default": "/World/envs/env_0/paper_waste/used_wet_paper_clump_01",
                "description": (
                    "USD prim path of the paper-waste object. Only used when paper_cleanup_mode=true."
                ),
            },
        },
    }


def gripper_prismatic_move_arm_to_joints_spec() -> dict:
    return {
        "name": "gripper_prismatic_move_arm_to_joints",
        "description": (
            "For prismatic gripper tasks, use cuRobo C-space planning to move the arm to a joint target "
            "while holding the gripper closed. Defaults to the shared planning/pre-target joint pose."
        ),
        "parameters": {
            "name": {
                "required": False,
                "type": "string",
                "description": "Optional case name shown in job results.",
            },
            "joint_pos_deg": {
                "required": False,
                "type": "array[float]",
                "length": 6,
                "description": (
                    "Optional arm joint target in degrees for joint1..joint6. "
                    "When omitted, the shared planning/pre-target joint pose is used."
                ),
            },
            "gripper_joint1_pos": {
                "required": False,
                "type": "float",
                "default": 0.02,
                "description": "Prismatic gripper_joint1 hold target in meters while moving the arm.",
            },
        },
    }


def gripper_prismatic_take_out_spec() -> dict:
    return {
        "name": "gripper_prismatic_take_out",
        "description": (
            "For prismatic gripper tasks, plan and execute to a drop pose while holding the gripper closed, "
            "open the gripper to release. By default the arm then holds still (no post-drop motion); "
            "pass return_to_pre_target_open=true to additionally return to the shared planning/pre-target pose open."
        ),
        "parameters": {
            "name": {
                "required": False,
                "type": "string",
                "description": "Optional case name shown in job results.",
            },
            "return_to_pre_target_open": {
                "required": False,
                "type": "bool",
                "default": False,
                "description": (
                    "After opening the gripper, also move back to the shared planning/pre-target joint pose. "
                    "Default False: hold still after the drop, so no post-drop motion leaks into videos/demos."
                ),
            },
            "drop_target_pos": {
                "required": False,
                "type": "array[float]",
                "length": 3,
                "default": [2.097, -1.068, 0.727],
                "description": "Drop target tool_tip_link position in Isaac world frame.",
            },
            "drop_target_quat": {
                "required": False,
                "type": "array[float]",
                "length": 4,
                "convention": "wxyz",
                "default": [-0.057, -0.598, -0.798, -0.043],
                "description": "Drop target tool_tip_link orientation in Isaac world frame.",
            },
        },
    }


def gripper_prismatic_local_lift_spec() -> dict:
    return {
        "name": "gripper_prismatic_local_lift",
        "description": (
            "For prismatic gripper tasks, use cuRobo trajopt to lift the tool_tip_link in world +Z "
            "while keeping XY and orientation fixed. Designed to be called after a successful low grasp "
            "to retreat from the cramped grasp pose before attempting cspace reset or long-distance "
            "trajopt moves. The gripper is held closed (grasp position) throughout."
        ),
        "parameters": {
            "name": {
                "required": False,
                "type": "string",
                "description": "Optional case name shown in job results.",
            },
            "lift_height_m": {
                "required": False,
                "type": "float",
                "default": 0.18,
                "description": "Lift distance in world +Z (meters). Default 0.18.",
            },
        },
    }


def execute_spot_capture_spec() -> dict:
    return {
        "name": "execute_spot_capture",
        "description": (
            "Use cuRobo C-space planning to move the arm to a capture joint pose, save one RGB/depth "
            "camera frame plus camera-to-arm-base TF, then return to the shared planning/pre-target pose."
        ),
        "parameters": {
            "name": {
                "required": False,
                "type": "string",
                "description": "Optional case name used in the saved capture filename.",
            },
            "joint_pos_deg": {
                "required": False,
                "type": "array[float]",
                "length": 6,
                "default": [2.8, -2.9, -25.7, 0.0, -94.5, 0.0],
                "description": "Capture joint target in degrees for joint1..joint6.",
            },
        },
    }
