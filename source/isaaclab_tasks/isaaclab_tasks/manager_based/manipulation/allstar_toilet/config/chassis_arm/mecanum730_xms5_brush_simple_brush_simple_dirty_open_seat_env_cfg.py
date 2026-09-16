"""Independent BrushSimple config that only swaps the robot USD."""

from isaaclab.utils import configclass

from ...assets import ALLSTAR_ASSETS_DATA_DIR
from .mecanum730_xms5_brush_rolling_0512_brush_rolling_0512_dirty_open_seat_env_cfg import (
    BrushRolling0512SceneCfg,
    Mecanum730XMS5BrushRolling0512BrushRolling0512DirtyOpenSeatEnvCfg,
)


@configclass
class BrushSimpleSceneCfg(BrushRolling0512SceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.usd_path = (
            f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_sphere_usd/"
            "mecanum730_xms5_brush_simple.usd"
        )


@configclass
class Mecanum730XMS5BrushSimpleBrushSimpleDirtyOpenSeatEnvCfg(
    Mecanum730XMS5BrushRolling0512BrushRolling0512DirtyOpenSeatEnvCfg
):
    scene: BrushSimpleSceneCfg = BrushSimpleSceneCfg(num_envs=1, env_spacing=2.5)
    cam_t_in_tool = (0.11, 0.0, 0.01)
