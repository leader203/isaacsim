"""Gymnasium task registrations kept by scripts/scripts_runtime.md."""

import gymnasium as gym


gym.register(
    id="Isaac-Allstar-Toilet-Mecanum0730-XMS5GripperJointPrismatic-GripperJointPrismatic-Dirty-OpenSeat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.mecanum730_xms5_gripper_joint_prismatic_dirty_open_seat_env_cfg:"
            "Mecanum730XMS5GripperJointPrismaticDirtyOpenSeatEnvCfg"
        ),
    },
)

gym.register(
    id=(
        "Isaac-Allstar-Toilet-Mecanum0730-XMS5GripperJointPrismaticSanheyiFull-"
        "GripperJointPrismatic-Dirty-OpenSeat-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.mecanum730_xms5_gripper_joint_prismatic_sanheyi_full_dirty_open_seat_env_cfg:"
            "Mecanum730XMS5GripperJointPrismaticSanheyiFullDirtyOpenSeatEnvCfg"
        ),
    },
)

gym.register(
    id="Isaac-Allstar-Toilet-Mecanum0730-XMS5BrushRolling0512-BrushRolling0512-Dirty-OpenSeat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.mecanum730_xms5_brush_rolling_0512_brush_rolling_0512_dirty_open_seat_env_cfg:"
            "Mecanum730XMS5BrushRolling0512BrushRolling0512DirtyOpenSeatEnvCfg"
        ),
    },
)

_SPHERE_USD_INDEPENDENT_TASKS = [
    (
        "BrushHard",
        "BrushHard",
        "mecanum730_xms5_brush_hard_brush_hard_dirty_open_seat_env_cfg",
        "Mecanum730XMS5BrushHardBrushHardDirtyOpenSeatEnvCfg",
    ),
    (
        "BrushRolling",
        "BrushRolling",
        "mecanum730_xms5_brush_rolling_brush_rolling_dirty_open_seat_env_cfg",
        "Mecanum730XMS5BrushRollingBrushRollingDirtyOpenSeatEnvCfg",
    ),
    (
        "BrushRollingSanheyiLong",
        "BrushRollingSanheyiLong",
        "mecanum730_xms5_brush_rolling_sanheyi_long_brush_rolling_sanheyi_long_dirty_open_seat_env_cfg",
        "Mecanum730XMS5BrushRollingSanheyiLongBrushRollingSanheyiLongDirtyOpenSeatEnvCfg",
    ),
    (
        "BrushRollingSanheyiShort",
        "BrushRollingSanheyiShort",
        "mecanum730_xms5_brush_rolling_sanheyi_short_brush_rolling_sanheyi_short_dirty_open_seat_env_cfg",
        "Mecanum730XMS5BrushRollingSanheyiShortBrushRollingSanheyiShortDirtyOpenSeatEnvCfg",
    ),
    (
        "BrushSimple",
        "BrushSimple",
        "mecanum730_xms5_brush_simple_brush_simple_dirty_open_seat_env_cfg",
        "Mecanum730XMS5BrushSimpleBrushSimpleDirtyOpenSeatEnvCfg",
    ),
    (
        "Gripper",
        "Gripper",
        "mecanum730_xms5_gripper_gripper_dirty_open_seat_env_cfg",
        "Mecanum730XMS5GripperGripperDirtyOpenSeatEnvCfg",
    ),
    (
        "GripperExpanded",
        "GripperExpanded",
        "mecanum730_xms5_gripper_expanded_gripper_expanded_dirty_open_seat_env_cfg",
        "Mecanum730XMS5GripperExpandedGripperExpandedDirtyOpenSeatEnvCfg",
    ),
    (
        "MopSponge",
        "MopSponge",
        "mecanum730_xms5_mop_sponge_mop_sponge_dirty_open_seat_env_cfg",
        "Mecanum730XMS5MopSpongeMopSpongeDirtyOpenSeatEnvCfg",
    ),
    (
        "MopSwab",
        "MopSwab",
        "mecanum730_xms5_mop_swab_mop_swab_dirty_open_seat_env_cfg",
        "Mecanum730XMS5MopSwabMopSwabDirtyOpenSeatEnvCfg",
    ),
    (
        "Pick",
        "Pick",
        "mecanum730_xms5_pick_pick_dirty_open_seat_env_cfg",
        "Mecanum730XMS5PickPickDirtyOpenSeatEnvCfg",
    ),
    (
        "Plus",
        "Plus",
        "mecanum730_xms5_plus_plus_dirty_open_seat_env_cfg",
        "Mecanum730XMS5PlusPlusDirtyOpenSeatEnvCfg",
    ),
    (
        "SuctionIn",
        "SuctionIn",
        "mecanum730_xms5_suction_in_suction_in_dirty_open_seat_env_cfg",
        "Mecanum730XMS5SuctionInSuctionInDirtyOpenSeatEnvCfg",
    ),
    (
        "SuctionIn0512",
        "SuctionIn0512",
        "mecanum730_xms5_suction_in_0512_suction_in_0512_dirty_open_seat_env_cfg",
        "Mecanum730XMS5SuctionIn0512SuctionIn0512DirtyOpenSeatEnvCfg",
    ),
    (
        "SuctionOut",
        "SuctionOut",
        "mecanum730_xms5_suction_out_suction_out_dirty_open_seat_env_cfg",
        "Mecanum730XMS5SuctionOutSuctionOutDirtyOpenSeatEnvCfg",
    ),
    (
        "SuctionOut0512",
        "SuctionOut0512",
        "mecanum730_xms5_suction_out_0512_suction_out_0512_dirty_open_seat_env_cfg",
        "Mecanum730XMS5SuctionOut0512SuctionOut0512DirtyOpenSeatEnvCfg",
    ),
    (
        "SuctionPress",
        "SuctionPress",
        "mecanum730_xms5_suction_press_suction_press_dirty_open_seat_env_cfg",
        "Mecanum730XMS5SuctionPressSuctionPressDirtyOpenSeatEnvCfg",
    ),
]
for _tool_tag, _task_tag, _module_name, _cfg_cls in _SPHERE_USD_INDEPENDENT_TASKS:
    gym.register(
        id=f"Isaac-Allstar-Toilet-Mecanum0730-XMS5{_tool_tag}-{_task_tag}-Dirty-OpenSeat-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_module_name}:{_cfg_cls}",
        },
    )

_SPHERE_USD_INDEPENDENT_LIDAR_TASKS = [
    ("BrushHard", "BrushHard", "Mecanum730XMS5BrushHardBrushHardDirtyOpenSeatLidarEnvCfg"),
    ("BrushRolling", "BrushRolling", "Mecanum730XMS5BrushRollingBrushRollingDirtyOpenSeatLidarEnvCfg"),
    ("BrushRolling0512", "BrushRolling0512", "Mecanum730XMS5BrushRolling0512BrushRolling0512DirtyOpenSeatLidarEnvCfg"),
    ("BrushSimple", "BrushSimple", "Mecanum730XMS5BrushSimpleBrushSimpleDirtyOpenSeatLidarEnvCfg"),
    ("Gripper", "Gripper", "Mecanum730XMS5GripperGripperDirtyOpenSeatLidarEnvCfg"),
    ("GripperExpanded", "GripperExpanded", "Mecanum730XMS5GripperExpandedGripperExpandedDirtyOpenSeatLidarEnvCfg"),
    ("MopSponge", "MopSponge", "Mecanum730XMS5MopSpongeMopSpongeDirtyOpenSeatLidarEnvCfg"),
    ("MopSwab", "MopSwab", "Mecanum730XMS5MopSwabMopSwabDirtyOpenSeatLidarEnvCfg"),
    ("Pick", "Pick", "Mecanum730XMS5PickPickDirtyOpenSeatLidarEnvCfg"),
    ("Plus", "Plus", "Mecanum730XMS5PlusPlusDirtyOpenSeatLidarEnvCfg"),
    ("SuctionIn", "SuctionIn", "Mecanum730XMS5SuctionInSuctionInDirtyOpenSeatLidarEnvCfg"),
    ("SuctionIn0512", "SuctionIn0512", "Mecanum730XMS5SuctionIn0512SuctionIn0512DirtyOpenSeatLidarEnvCfg"),
    ("SuctionOut", "SuctionOut", "Mecanum730XMS5SuctionOutSuctionOutDirtyOpenSeatLidarEnvCfg"),
    ("SuctionOut0512", "SuctionOut0512", "Mecanum730XMS5SuctionOut0512SuctionOut0512DirtyOpenSeatLidarEnvCfg"),
    ("SuctionPress", "SuctionPress", "Mecanum730XMS5SuctionPressSuctionPressDirtyOpenSeatLidarEnvCfg"),
]
for _tool_tag, _task_tag, _cfg_cls in _SPHERE_USD_INDEPENDENT_LIDAR_TASKS:
    gym.register(
        id=f"Isaac-Allstar-Toilet-Mecanum0730-XMS5{_tool_tag}-{_task_tag}-Dirty-OpenSeat-Lidar-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.mecanum730_xms5_independent_lidar_env_cfg:{_cfg_cls}",
        },
    )

gym.register(
    id="Isaac-Allstar-Toilet-Mecanum0730-XMS5GripperJointPrismatic-GripperJointPrismatic-Dirty-OpenSeat-Lidar-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.mecanum730_xms5_independent_lidar_env_cfg:"
            "Mecanum730XMS5GripperJointPrismaticDirtyOpenSeatLidarEnvCfg"
        ),
    },
)

gym.register(
    id="Isaac-Allstar-Toilet-Mecanum0730-XMS5GripperJointPrismatic-CuroboPosePlanTest-Dirty-OpenSeat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.mecanum730_xms5_gripper_joint_prismatic_curobo_pose_plan_test_env_cfg:"
            "Mecanum730XMS5GripperJointPrismaticCuroboPosePlanTestEnvCfg"
        ),
    },
)

gym.register(
    id=(
        "Isaac-Allstar-Toilet-Mecanum0730-XMS5GripperJointPrismaticSanheyiFull-"
        "GripperJointPrismatic-CuroboPosePlanTest-Dirty-OpenSeat-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.mecanum730_xms5_gripper_joint_prismatic_sanheyi_full_curobo_pose_plan_test_env_cfg:"
            "Mecanum730XMS5GripperJointPrismaticSanheyiFullCuroboPosePlanTestEnvCfg"
        ),
    },
)

gym.register(
    id="Isaac-Allstar-Toilet-Mecanum0730-XMS5Common-CuroboPosePlanTest-Dirty-OpenSeat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.mecanum730_xms5_common_curobo_pose_plan_test_env_cfg:"
            "Mecanum730XMS5CommonCuroboPosePlanTestEnvCfg"
        ),
    },
)
