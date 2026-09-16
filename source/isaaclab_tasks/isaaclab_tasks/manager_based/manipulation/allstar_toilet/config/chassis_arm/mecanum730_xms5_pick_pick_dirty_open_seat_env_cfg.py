"""Independent Pick config that only swaps the robot USD."""

from isaaclab.utils import configclass

from ...assets import ALLSTAR_ASSETS_DATA_DIR
from .mecanum730_xms5_brush_rolling_0512_brush_rolling_0512_dirty_open_seat_env_cfg import (
    BrushRolling0512SceneCfg,
    Mecanum730XMS5BrushRolling0512BrushRolling0512DirtyOpenSeatEnvCfg,
)


@configclass
class PickSceneCfg(BrushRolling0512SceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.usd_path = (
            f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_sphere_usd/"
            "mecanum730_xms5_pick.usd"
        )


@configclass
class Mecanum730XMS5PickPickDirtyOpenSeatEnvCfg(
    Mecanum730XMS5BrushRolling0512BrushRolling0512DirtyOpenSeatEnvCfg
):
    scene: PickSceneCfg = PickSceneCfg(num_envs=1, env_spacing=2.5)

