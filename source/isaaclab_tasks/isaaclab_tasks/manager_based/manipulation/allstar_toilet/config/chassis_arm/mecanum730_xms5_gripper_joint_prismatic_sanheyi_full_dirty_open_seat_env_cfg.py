"""Independent Sanheyi-full USD variant of the prismatic-gripper recorder task."""

from isaaclab.utils import configclass

from ...assets import ALLSTAR_ASSETS_DATA_DIR
from .mecanum730_xms5_gripper_joint_prismatic_dirty_open_seat_env_cfg import (
    GripperJointPrismaticSceneCfg,
    Mecanum730XMS5GripperJointPrismaticDirtyOpenSeatEnvCfg,
)


@configclass
class GripperJointPrismaticSanheyiFullSceneCfg(GripperJointPrismaticSceneCfg):
    def __post_init__(self):
        self.robot.spawn.usd_path = (
            f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_sphere_usd/"
            "mecanum730_xms5_gripper_joint_prismatic_sanheyi_full.usd"
        )


@configclass
class Mecanum730XMS5GripperJointPrismaticSanheyiFullDirtyOpenSeatEnvCfg(
    Mecanum730XMS5GripperJointPrismaticDirtyOpenSeatEnvCfg
):
    scene: GripperJointPrismaticSanheyiFullSceneCfg = GripperJointPrismaticSanheyiFullSceneCfg(
        num_envs=1, env_spacing=2.5
    )

