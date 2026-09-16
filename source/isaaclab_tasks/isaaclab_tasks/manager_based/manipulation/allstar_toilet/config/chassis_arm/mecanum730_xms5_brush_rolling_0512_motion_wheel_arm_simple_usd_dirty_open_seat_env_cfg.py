from isaaclab.utils import configclass

from ...assets import ALLSTAR_ASSETS_DATA_DIR
from ...assets.robots import MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_WHEEL_ARM_SIMPLE_USD_CFG
from .mecanum730_xms5_brush_rolling_0512_motion_wheel_free_dirty_open_seat_env_cfg import (
    Mecanum730XMS5BrushRolling0512MotionWheelFreeDirtyOpenSeatEnvCfg,
    Mecanum730XMS5BrushRolling0512MotionWheelFreeDirtyOpenSeatSceneCfg,
)


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleUsdDirtyOpenSeatSceneCfg(
    Mecanum730XMS5BrushRolling0512MotionWheelFreeDirtyOpenSeatSceneCfg
):
    """Motion-wheel-arm-simple USD scene that keeps the motion-wheel-free control/task behavior."""

    robot = MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_WHEEL_ARM_SIMPLE_USD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/robot"
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.usd_path = (
            f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_usd/"
            "mecanum730_xms5_brush_rolling_0512_disable_arm_gravity.usda"
        )
        # The direct USD variant has dense convex collision geometry. With self-collision enabled,
        # initial overlap/depenetration impulses can overpower the arm drives even without gravity.
        self.robot.spawn.articulation_props.enabled_self_collisions = False
        self.robot.init_state.pos = (1.840, 0.012, 0.006)
        self.robot.init_state.rot = (0.7437284690450652, 0.0, 0.0, -0.6684818354539511)


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleUsdDirtyOpenSeatEnvCfg(
    Mecanum730XMS5BrushRolling0512MotionWheelFreeDirtyOpenSeatEnvCfg
):
    """Dirty open-seat task that only swaps in the direct motion-wheel-arm-simple USD robot."""

    scene: Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleUsdDirtyOpenSeatSceneCfg = (
        Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleUsdDirtyOpenSeatSceneCfg(num_envs=1, env_spacing=2.5)
    )
