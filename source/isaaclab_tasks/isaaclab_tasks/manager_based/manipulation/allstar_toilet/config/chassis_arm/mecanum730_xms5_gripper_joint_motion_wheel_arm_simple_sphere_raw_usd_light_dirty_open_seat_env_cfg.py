import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from ...assets import ALLSTAR_ASSETS_DATA_DIR
from .mecanum730_xms5_brush_rolling_0512_motion_wheel_arm_simple_sphere_raw_usd_light_dirty_open_seat_env_cfg import (
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatEnvCfg,
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg,
)

DEFAULT_GRIPPER_JOINT1_OUTWARD_OPEN_RAD = 0.52


@configclass
class Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightActionsCfg:
    """Wheel, arm, and gripper-joint control for the gripper_joint USD."""

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
class Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightObservationsCfg:
    """Observation manager using the same tool_link-mounted camera TF as the gripper dirty task."""

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
        cam_in_tool_tf = ObsTerm(func=mdp.cam_in_tool_tf)
        cam_in_base_tf = ObsTerm(
            func=mdp.cam_in_base_tf,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["tool_link"])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg(
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg
):
    """Light dirty open-seat scene that only swaps the robot USD to gripper_joint."""

    grasp_test_cube = None

    grasp_test_cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/grasp_test_cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.486, -1.071, 0.021),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.015, 0.05, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_depenetration_velocity=1.0,
                max_linear_velocity=2.0,
                max_angular_velocity=20.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.55, 0.95)),
        ),
    )

    grasp_test_cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/grasp_test_cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.409, -1.157, 0.021),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.015, 0.05, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_depenetration_velocity=1.0,
                max_linear_velocity=2.0,
                max_angular_velocity=20.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.55, 0.95)),
        ),
    )

    grasp_test_cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/grasp_test_cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.355, -1.218, 0.021),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.015, 0.05, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_depenetration_velocity=1.0,
                max_linear_velocity=2.0,
                max_angular_velocity=20.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.55, 0.95)),
        ),
    )

    trash_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/trash_bin",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.097, -1.095, 0.010),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ALLSTAR_ASSETS_DATA_DIR}/props/open_trash_bin.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.2,
                angular_damping=0.2,
                max_depenetration_velocity=2.0,
                max_linear_velocity=2.0,
                max_angular_velocity=20.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.001,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08)),
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.usd_path = (
            f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_sphere_usd/"
            "mecanum730_xms5_gripper_joint.usd"
        )
        self.robot.actuators["gripper"] = ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint[12]"],
            effort_limit_sim=5.0,
            velocity_limit_sim=1.0,
            stiffness=200.0,
            damping=40.0,
        )
        self.robot.init_state.joint_pos["gripper_joint1"] = DEFAULT_GRIPPER_JOINT1_OUTWARD_OPEN_RAD
        self.robot.init_state.joint_pos["gripper_joint2"] = -DEFAULT_GRIPPER_JOINT1_OUTWARD_OPEN_RAD


@configclass
class Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatEnvCfg(
    Mecanum730XMS5BrushRolling0512MotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatEnvCfg
):
    """Same controls and light scene as BrushRolling0512 SphereRawUsdLight, with gripper_joint USD."""

    enable_collision_debug_visuals = False

    scene: Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg = (
        Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg(
            num_envs=1, env_spacing=2.5
        )
    )
    actions: Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightActionsCfg = (
        Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightActionsCfg()
    )
    observations: Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightObservationsCfg = (
        Mecanum730XMS5GripperJointMotionWheelArmSimpleSphereRawUsdLightObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.enable_collision_debug_visuals = False
