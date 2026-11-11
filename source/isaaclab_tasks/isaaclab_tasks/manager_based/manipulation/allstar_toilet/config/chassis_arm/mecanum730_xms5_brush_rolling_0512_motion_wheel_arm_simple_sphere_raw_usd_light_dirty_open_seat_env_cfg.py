from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from ...assets.sensors.camera_cfg import OrbbecGemini2
from .mecanum730_xms5_brush_rolling_0512_motion_wheel_arm_simple_raw_usd_dirty_open_seat_env_cfg import (
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdDirtyOpenSeatEnvCfg,
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdDirtyOpenSeatSceneCfg,
)

_TARGET_BASE_POS_W = (1.832, 0.0, 0.010)
_TARGET_BASE_ROT_WXYZ = (0.702153052995162, 0.0, 0.0, -0.712026045990997)
_TARGET_ARM_JOINT_POS = (
    0.099483767364,
    -0.200712863979,
    -0.799360797413,
    0.0,
    1.0,
    0.0,
)


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightActionsCfg:
    """Same wheel and arm control as the SphereRawUsd task."""

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


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightObservationsCfg:
    """Observation manager with joint state and the link5-mounted tool camera."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "wheel_fl_joint",
                        "wheel_fr_joint",
                        "wheel_rl_joint",
                        "wheel_rr_joint",
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
                    preserve_order=True,
                )
            },
        )
        cam_tool_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_tool"), "data_type": "rgb", "normalize": False},
        )
        cam_tool_depth = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_tool"), "data_type": "depth", "normalize": False},
        )
        cam_in_tool_tf = ObsTerm(func=mdp.cam_link5_mount_in_tool_tf)
        cam_in_base_tf = ObsTerm(
            func=mdp.cam_link5_mount_in_base_tf,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["XMS5_R800_W4G3B4C_link5", "tool_link"],
                    preserve_order=True,
                )
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg(
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdDirtyOpenSeatSceneCfg
):
    """SphereRawUsd scene with the tool camera restored and LiDAR disabled."""

    cam_tool = OrbbecGemini2(
        prim_path="{ENV_REGEX_NS}/cam_tool",
        offset=OrbbecGemini2.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )
    lidar = None

    def __post_init__(self):
        super().__post_init__()
        # Keep the light variant physically identical to the full SphereRawUsd task:
        # robot gravity enabled, robot collision preserved from USD, self-collision enabled.
        self.robot.spawn.activate_contact_sensors = True
        self.robot.spawn.collision_props = None
        self.robot.spawn.rigid_props.disable_gravity = False
        self.robot.spawn.articulation_props.enabled_self_collisions = True


@configclass
class Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatEnvCfg(
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdDirtyOpenSeatEnvCfg
):
    """Lightweight SphereRawUsd task without any camera/LiDAR observation terms."""

    disable_robot_base_collision = False
    enable_robot_all_gravity = True
    enable_robot_arm_gravity = True
    enable_collision_debug_visuals = False

    scene: Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg = (
        Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg(
            num_envs=1, env_spacing=2.5
        )
    )
    actions: Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightActionsCfg = (
        Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightActionsCfg()
    )
    observations: Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightObservationsCfg = (
        Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.robot.spawn.collision_props = None
        self.scene.robot.spawn.rigid_props.disable_gravity = False
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.robot.init_state.pos = _TARGET_BASE_POS_W
        self.scene.robot.init_state.rot = _TARGET_BASE_ROT_WXYZ
        for joint_idx, joint_pos in enumerate(_TARGET_ARM_JOINT_POS, start=1):
            self.scene.robot.init_state.joint_pos[f"joint{joint_idx}"] = joint_pos
        self.events.init_arm_joint_position.params["default_pos"] = list(_TARGET_ARM_JOINT_POS)
