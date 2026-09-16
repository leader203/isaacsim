"""Independent Sanheyi-full cuRobo pose-plan test task."""

from isaaclab.utils import configclass

from .mecanum730_xms5_gripper_joint_prismatic_sanheyi_full_dirty_open_seat_env_cfg import (
    GripperJointPrismaticSanheyiFullSceneCfg,
    Mecanum730XMS5GripperJointPrismaticSanheyiFullDirtyOpenSeatEnvCfg,
)


@configclass
class Mecanum730XMS5GripperJointPrismaticSanheyiFullCuroboPosePlanTestSceneCfg(
    GripperJointPrismaticSanheyiFullSceneCfg
):
    """Scene clone for cuRobo pose planning with the Sanheyi-full prismatic gripper USD."""


@configclass
class Mecanum730XMS5GripperJointPrismaticSanheyiFullCuroboPosePlanTestEnvCfg(
    Mecanum730XMS5GripperJointPrismaticSanheyiFullDirtyOpenSeatEnvCfg
):
    """cuRobo pose-planning task that keeps the Sanheyi-full recorder scene isolated."""

    scene: Mecanum730XMS5GripperJointPrismaticSanheyiFullCuroboPosePlanTestSceneCfg = (
        Mecanum730XMS5GripperJointPrismaticSanheyiFullCuroboPosePlanTestSceneCfg(
            num_envs=1, env_spacing=2.5
        )
    )
    enable_collision_debug_visuals = False

    def __post_init__(self):
        super().__post_init__()
        self.enable_collision_debug_visuals = False
        self.episode_length_s = 300.0
        arm = self.scene.robot.actuators["arm"]
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
