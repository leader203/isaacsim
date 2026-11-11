from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.utils import configclass

from ...assets import ALLSTAR_ASSETS_DATA_DIR
from .mecanum730_xms5_gripper_joint_motion_wheel_arm_simple_sphere_raw_usd_light_dirty_open_seat_env_cfg import (
    Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatEnvCfg,
    Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg,
)

DEFAULT_PRISMATIC_GRIPPER_OPEN_M = 0.0
COLLISION_BOX_FIXED_OPEN_SEAT_USD = (
    f"{ALLSTAR_ASSETS_DATA_DIR}/scenes/real/shenxinfu/841837/"
    "start_open_seat.usda"
)


@configclass
class Mecanum730XMS5GripperJointPrismaticMotionWheelArmSimpleSphereRawUsdLightActionsCfg:
    """Wheel, arm, and prismatic gripper control for the gripper_joint_prismatic USD."""

    wheel_velocity = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint"],
        preserve_order=True,
        scale=1.0,
    )

    arm_joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],
        scale=1.0,
        use_default_offset=True,
    )

    gripper_joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_joint1", "gripper_joint2"],
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class Mecanum730XMS5GripperJointPrismaticMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg(
    Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg
):
    """Light dirty open-seat scene with the prismatic gripper USD."""

    def __post_init__(self):
        super().__post_init__()
        self.room.spawn.usd_path = COLLISION_BOX_FIXED_OPEN_SEAT_USD
        self.robot.spawn.usd_path = (
            f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_sphere_usd/"
            "mecanum730_xms5_gripper_joint_prismatic.usd"
        )
        self.robot.actuators["gripper"] = ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint[12]"],
            effort_limit_sim=20.0,
            velocity_limit_sim=0.2,
            stiffness=800.0,
            damping=80.0,
        )
        self.robot.init_state.joint_pos["gripper_joint1"] = DEFAULT_PRISMATIC_GRIPPER_OPEN_M
        self.robot.init_state.joint_pos["gripper_joint2"] = -DEFAULT_PRISMATIC_GRIPPER_OPEN_M


@configclass
class Mecanum730XMS5GripperJointPrismaticMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatEnvCfg(
    Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatEnvCfg
):
    """Same light scene and controls as the gripper-joint task, with the prismatic gripper USD."""

    enable_collision_debug_visuals = False
    # cam_tool 刚性跟随 tool_link（夹爪），joint6 转动时相机与夹爪相对姿态不变。
    cam_follows_tool = True

    scene: Mecanum730XMS5GripperJointPrismaticMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg = (
        Mecanum730XMS5GripperJointPrismaticMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg(
            num_envs=1, env_spacing=2.5
        )
    )
    actions: Mecanum730XMS5GripperJointPrismaticMotionWheelArmSimpleSphereRawUsdLightActionsCfg = (
        Mecanum730XMS5GripperJointPrismaticMotionWheelArmSimpleSphereRawUsdLightActionsCfg()
    )
