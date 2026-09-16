from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from ... import mdp
from ...assets import ALLSTAR_ASSETS_DATA_DIR
from ...assets.robots import MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_WHEEL_FREE_CFG
from .mecanum730_xms5_gripper_dirty_env_cfg import _RX, _VISUAL_ONLY_DIRT_SPAWN
from .mecanum730_xms5_gripper_dirty_open_seat_env_cfg import (
    Mecanum730XMS5GripperDirtyOpenSeatEnvCfg,
    Mecanum730XMS5GripperDirtyOpenSeatSceneCfg,
)


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelFreeDirtyOpenSeatSceneCfg(Mecanum730XMS5GripperDirtyOpenSeatSceneCfg):
    """Dirty open-seat scene with the motion-wheel-free brush-rolling-0512 mobile manipulator variant."""

    robot = MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_WHEEL_FREE_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    def __post_init__(self):
        super().__post_init__()
        self.room.spawn = UsdFileCfg(
            usd_path=f"{ALLSTAR_ASSETS_DATA_DIR}/scenes/real/shenxinfu/841837/start_open_seat.usda"
        )
        self.robot.spawn.articulation_props.enabled_self_collisions = True
        self.robot.init_state.pos = (0.515, 0.406, 0.002)
        self.robot.init_state.rot = (0.7169106076504828, 0.0, 0.0, 0.6971651028545646)
        self.robot.init_state.joint_pos["joint1"] = 0.0
        self.robot.init_state.joint_pos["joint2"] = 0.0
        self.robot.init_state.joint_pos["joint3"] = 0.0
        self.robot.init_state.joint_pos["joint4"] = 0.0
        self.robot.init_state.joint_pos["joint5"] = 0.0
        self.robot.init_state.joint_pos["joint6"] = 0.0

    dirt_cigarette = None

    dirt_shitstains_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_shitstains_1",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.663, -0.980, 0.408),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(usd_path=f"{_RX}/Shitstains/Shitstains_proxy.usda", **_VISUAL_ONLY_DIRT_SPAWN),
    )

    dirt_shitstains_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_shitstains_2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.656, -1.152, 0.408),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(usd_path=f"{_RX}/Shitstains/Shitstains_proxy.usda", **_VISUAL_ONLY_DIRT_SPAWN),
    )

    dirt_shitstains_3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_shitstains_3",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.648, -1.090, 0.408),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(usd_path=f"{_RX}/Shitstains/Shitstains_proxy.usda", **_VISUAL_ONLY_DIRT_SPAWN),
    )


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelFreeDirtyOpenSeatEnvCfg(Mecanum730XMS5GripperDirtyOpenSeatEnvCfg):
    """Dirty open-seat task with motion-wheel-free mecanum root control and brush-rolling-0512 asset."""

    @configclass
    class MotionWheelFreeEventsCfg(Mecanum730XMS5GripperDirtyOpenSeatEnvCfg.OpenSeatEventsCfg):
        init_arm_joint_position = EventTerm(
            func=mdp.set_default_joint_pos,
            mode="startup",
            params={
                "default_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
            },
        )

        randomize_arm_joint_state = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
            },
        )

    scene: Mecanum730XMS5BrushRolling0512MotionWheelFreeDirtyOpenSeatSceneCfg = (
        Mecanum730XMS5BrushRolling0512MotionWheelFreeDirtyOpenSeatSceneCfg(num_envs=1, env_spacing=2.5)
    )
    events: MotionWheelFreeEventsCfg = MotionWheelFreeEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["arm"].stiffness = 1600.0
        self.scene.robot.actuators["arm"].damping = 640.0
        self.viewer.eye = (1.660, 0.0, 1.9)
        self.viewer.lookat = (1.663, -0.980, 0.408)
