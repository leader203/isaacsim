"""Common cuRobo ToolAPI task with one switchable robot USD."""

import copy

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

from ...assets import ALLSTAR_ASSETS_DATA_DIR
from .mecanum730_xms5_brush_rolling_0512_brush_rolling_0512_dirty_open_seat_env_cfg import (
    BrushRolling0512ActionsCfg,
    BrushRolling0512SceneCfg,
)
from .mecanum730_xms5_gripper_joint_prismatic_dirty_open_seat_env_cfg import (
    TARGET_BASE_POS_W,
    TARGET_BASE_ROT_WXYZ,
    TARGET_ARM_JOINT_POS,
    DEFAULT_PRISMATIC_GRIPPER_OPEN_M,
    GripperJointPrismaticSceneCfg,
    Mecanum730XMS5GripperJointPrismaticDirtyOpenSeatEnvCfg,
)


ROBOT_USD_ROOT = f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_sphere_usd"
COMMON_BASE_ROOT_Z = TARGET_BASE_POS_W[2]
COMMON_TARGET_BASE_POS_W = (TARGET_BASE_POS_W[0], TARGET_BASE_POS_W[1], COMMON_BASE_ROOT_Z)
_BRUSH_ROBOT_TEMPLATE = BrushRolling0512SceneCfg(num_envs=1, env_spacing=2.5).robot
_PRISMATIC_ROBOT_TEMPLATE = GripperJointPrismaticSceneCfg(num_envs=1, env_spacing=2.5).robot
ROBOT_PROFILES = {
    "brush_rolling_0512": {
        "robot_type": "brush",
        "usd_name": "mecanum730_xms5_brush_rolling_0512.usd",
    },
    "brush_rolling_sanheyi_long": {
        "robot_type": "brush",
        "usd_name": "mecanum730_xms5_brush_rolling_sanheyi_long.usd",
    },
    "brush_rolling_sanheyi_short": {
        "robot_type": "brush",
        "usd_name": "mecanum730_xms5_brush_rolling_sanheyi_short.usd",
    },
    "gripper_prismatic": {
        "robot_type": "gripper",
        "usd_name": "mecanum730_xms5_gripper_joint_prismatic.usd",
    },
    "gripper_prismatic_sanheyi_full": {
        "robot_type": "gripper",
        "usd_name": "mecanum730_xms5_gripper_joint_prismatic_sanheyi_full.usd",
    },
}


def _copy_robot_cfg(base_cfg, *, asset_name: str, usd_name: str, pos=COMMON_TARGET_BASE_POS_W):
    cfg = copy.deepcopy(base_cfg).replace(prim_path=f"{{ENV_REGEX_NS}}/{asset_name}")
    cfg.spawn.usd_path = f"{ROBOT_USD_ROOT}/{usd_name}"
    cfg.init_state.pos = tuple(pos)
    cfg.init_state.rot = tuple(TARGET_BASE_ROT_WXYZ)
    for idx, joint_name in enumerate(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")):
        cfg.init_state.joint_pos[joint_name] = TARGET_ARM_JOINT_POS[idx]
    return cfg


def _copy_prismatic_robot_cfg(*, asset_name: str, usd_name: str, pos=COMMON_TARGET_BASE_POS_W):
    cfg = _copy_robot_cfg(_PRISMATIC_ROBOT_TEMPLATE, asset_name=asset_name, usd_name=usd_name, pos=pos)
    cfg.actuators["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=["gripper_joint[12]"],
        effort_limit_sim=20.0,
        velocity_limit_sim=0.2,
        stiffness=800.0,
        damping=80.0,
    )
    cfg.init_state.joint_pos["gripper_joint1"] = DEFAULT_PRISMATIC_GRIPPER_OPEN_M
    cfg.init_state.joint_pos["gripper_joint2"] = -DEFAULT_PRISMATIC_GRIPPER_OPEN_M
    return cfg


def _apply_common_arm_tuning(robot_cfg):
    arm = robot_cfg.actuators["arm"]
    arm.stiffness = {
        "joint1": 2400.0,
        "joint2": 16000.0,
        "joint3": 8000.0,
        "joint4": 2400.0,
        "joint5": 2400.0,
        "joint6": 2400.0,
    }
    arm.damping = {
        "joint1": 960.0,
        "joint2": 3840.0,
        "joint3": 1920.0,
        "joint4": 960.0,
        "joint5": 960.0,
        "joint6": 960.0,
    }
    arm.effort_limit_sim = {
        "joint1": 400.0,
        "joint2": 800.0,
        "joint3": 400.0,
        "joint4": 400.0,
        "joint5": 400.0,
        "joint6": 400.0,
    }
    arm.velocity_limit_sim = {
        "joint1": 3.0,
        "joint2": 3.0,
        "joint3": 3.0,
        "joint4": 3.0,
        "joint5": 3.0,
        "joint6": 3.0,
    }
    arm.velocity_limit = arm.velocity_limit_sim
    return robot_cfg


def make_common_robot_cfg(
    robot_id: str,
    *,
    prim_path: str = "{ENV_REGEX_NS}/robot",
    pos=COMMON_TARGET_BASE_POS_W,
):
    """Build the single active robot cfg for a common robot profile."""
    profile = ROBOT_PROFILES[robot_id]
    usd_name = profile["usd_name"]
    if profile["robot_type"] == "gripper":
        cfg = _copy_prismatic_robot_cfg(asset_name="robot", usd_name=usd_name, pos=pos)
    else:
        cfg = _copy_robot_cfg(_BRUSH_ROBOT_TEMPLATE, asset_name="robot", usd_name=usd_name, pos=pos)
    cfg.prim_path = prim_path
    cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    )
    return _apply_common_arm_tuning(cfg)


@configclass
class Mecanum730XMS5CommonCuroboPosePlanTestSceneCfg(
    GripperJointPrismaticSceneCfg
):
    """Common scene: shared restroom/cubes/trash plus one active robot."""

    def __post_init__(self):
        super().__post_init__()
        self.robot = make_common_robot_cfg("brush_rolling_0512")


@configclass
class Mecanum730XMS5CommonCuroboPosePlanTestEnvCfg(
    Mecanum730XMS5GripperJointPrismaticDirtyOpenSeatEnvCfg
):
    """One API task that can switch among brush and prismatic-gripper USDs."""

    scene: Mecanum730XMS5CommonCuroboPosePlanTestSceneCfg = (
        Mecanum730XMS5CommonCuroboPosePlanTestSceneCfg(num_envs=1, env_spacing=2.5)
    )
    actions: BrushRolling0512ActionsCfg = BrushRolling0512ActionsCfg()
    robot_profiles = copy.deepcopy(ROBOT_PROFILES)
    default_robot_id = "brush_rolling_0512"
    enable_collision_debug_visuals = False

    def __post_init__(self):
        super().__post_init__()
        self.enable_collision_debug_visuals = False
        self.episode_length_s = 300.0
        self.scene.robot = _apply_common_arm_tuning(self.scene.robot)
