from isaaclab.utils import configclass

from ...assets import ALLSTAR_ASSETS_DATA_DIR
from .mecanum730_xms5_brush_rolling_0512_motion_wheel_arm_simple_usd_dirty_open_seat_env_cfg import (
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleUsdDirtyOpenSeatEnvCfg,
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleUsdDirtyOpenSeatSceneCfg,
)

_TARGET_BASE_POS_W = (1.832, -0.229, 0.010)
_TARGET_BASE_ROT_WXYZ = (0.7021530529951624, 0.0, 0.0, -0.7120260459909965)
_TARGET_ARM_JOINT_POS = (
    0.09948376736367678,
    -0.2007128639793479,
    -0.7993607974134029,
    0.0,
    -1.8500490071139892,
    0.0,
)


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleRawUsdDirtyOpenSeatSceneCfg(
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleUsdDirtyOpenSeatSceneCfg
):
    """Motion-wheel-arm-simple direct USD scene using the raw brush-rolling-0512 robot USD."""

    # Keep only the three toilet-side yellow dirt targets inherited as dirt_shitstains_1/2/3.
    dirt_cigarette = None
    dirt_ash = None
    dirt_footprint = None
    dirt_plastic_bag = None
    dirt_roll_paper = None
    dirt_seat_paper = None
    dirt_aunt_scarf = None
    dirt_shitstains = None
    dirt_urine = None
    dirt_water = None

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.usd_path = (
            f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_usd/"
            "mecanum730_xms5_brush_rolling_0512.usd"
        )
        self.robot.spawn.activate_contact_sensors = True
        self.robot.spawn.collision_props = None
        self.robot.spawn.rigid_props.disable_gravity = True
        self.robot.spawn.articulation_props.enabled_self_collisions = True


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleRawUsdDirtyOpenSeatEnvCfg(
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleUsdDirtyOpenSeatEnvCfg
):
    """Dirty open-seat task that only swaps the direct USD robot to the raw USD file."""

    disable_robot_base_collision = True

    scene: Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleRawUsdDirtyOpenSeatSceneCfg = (
        Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleRawUsdDirtyOpenSeatSceneCfg(num_envs=1, env_spacing=2.5)
    )


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdDirtyOpenSeatSceneCfg(
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleRawUsdDirtyOpenSeatSceneCfg
):
    """Motion-wheel-arm-simple direct USD scene using the sphere-collision raw USD file."""

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.usd_path = (
            f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_sphere_usd/"
            "mecanum730_xms5_brush_rolling_0512.usd"
        )
        # This variant tests the full robot as a physical mobile manipulator:
        # chassis and arm gravity are both enabled, and chassis/arm collision
        # remain enabled by the dirty-scene collision policy.
        self.robot.spawn.rigid_props.disable_gravity = False
        self.robot.init_state.pos = _TARGET_BASE_POS_W
        self.robot.init_state.rot = _TARGET_BASE_ROT_WXYZ
        for joint_idx, joint_pos in enumerate(_TARGET_ARM_JOINT_POS, start=1):
            self.robot.init_state.joint_pos[f"joint{joint_idx}"] = joint_pos
        self.robot.spawn.articulation_props.enabled_self_collisions = True


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdDirtyOpenSeatEnvCfg(
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleRawUsdDirtyOpenSeatEnvCfg
):
    """Dirty open-seat task that only swaps the raw USD robot to the sphere-collision USD file."""

    disable_robot_base_collision = False
    enable_robot_all_gravity = True
    enable_robot_arm_gravity = True
    enable_collision_debug_visuals = True

    scene: Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdDirtyOpenSeatSceneCfg = (
        Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdDirtyOpenSeatSceneCfg(
            num_envs=1, env_spacing=2.5
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.events.init_arm_joint_position.params["default_pos"] = list(_TARGET_ARM_JOINT_POS)


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereStlRawUsdDirtyOpenSeatSceneCfg(
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdDirtyOpenSeatSceneCfg
):
    """Motion-wheel-arm-simple direct USD scene using the sphere-STL-collision raw USD file."""

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.usd_path = (
            f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_sphere_stl_usd/"
            "mecanum730_xms5_brush_rolling_0512.usd"
        )


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereStlRawUsdDirtyOpenSeatEnvCfg(
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdDirtyOpenSeatEnvCfg
):
    """Dirty open-seat task that only swaps the sphere raw USD robot to the sphere-STL USD file."""

    enable_collision_debug_visuals = True

    scene: Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereStlRawUsdDirtyOpenSeatSceneCfg = (
        Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereStlRawUsdDirtyOpenSeatSceneCfg(
            num_envs=1, env_spacing=2.5
        )
    )
