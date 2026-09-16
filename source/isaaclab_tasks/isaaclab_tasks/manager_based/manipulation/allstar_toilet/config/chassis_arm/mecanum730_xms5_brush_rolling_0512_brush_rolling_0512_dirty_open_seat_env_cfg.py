"""BrushRolling0512 config for the mecanum730 + XMS5 brush recording task.

This file intentionally avoids inheriting the long dirty/open-seat/brush task
chain. It keeps only the assets, controls, observations, and events used by the
brush-rolling mobile-arm recording command.
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from ... import mdp
from ...assets import ALLSTAR_ASSETS_DATA_DIR
from ...assets.sensors.camera_cfg import OrbbecGemini2


ROOM_USD = (
    f"{ALLSTAR_ASSETS_DATA_DIR}/scenes/real/shenxinfu/841837/"
    "start_open_seat.usda"
)
ROBOT_USD = (
    f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_sphere_usd/"
    "mecanum730_xms5_brush_rolling_0512.usd"
)
SHITSTAINS_PROXY_USD = f"{ALLSTAR_ASSETS_DATA_DIR}/scenes/rx/Shitstains/Shitstains_proxy.usda"

OPEN_SEAT_RAD = 95.0 * torch.pi / 180.0
TARGET_BASE_POS_W = (1.832, -0.264, 0.010)
TARGET_BASE_ROT_WXYZ = (0.702153052995162, 0.0, 0.0, -0.712026045990997)
TARGET_ARM_JOINT_POS = (
    0.099483767364,
    -0.200712863979,
    -0.799360797413,
    0.0,
    -1.850049007114,
    0.0,
)


def _force_open_open_seat_toilets(env, env_ids, asset_names=("toilet_b",)) -> None:
    """Keep the two toilet lid/seat articulations at the open pose."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    target_pos = torch.full((len(env_ids), 2), float(OPEN_SEAT_RAD), device=env.device)
    target_vel = torch.zeros_like(target_pos)

    for asset_name in asset_names:
        toilet = env.scene[asset_name]
        joint_ids, joint_names = toilet.find_joints(".*Toilet(Seat|Lid)$", preserve_order=True)
        if len(joint_ids) != 2:
            raise RuntimeError(f"Expected 2 toilet joints on '{asset_name}', got {joint_names}.")
        toilet.write_joint_state_to_sim(target_pos, target_vel, joint_ids=joint_ids, env_ids=env_ids)
        toilet.set_joint_position_target(target_pos, joint_ids=joint_ids, env_ids=env_ids)
        toilet.set_joint_velocity_target(target_vel, joint_ids=joint_ids, env_ids=env_ids)
        toilet.write_joint_stiffness_to_sim(5000.0, joint_ids=joint_ids, env_ids=env_ids)
        toilet.write_joint_damping_to_sim(500.0, joint_ids=joint_ids, env_ids=env_ids)


def _ensure_open_seat_lid_seat_collisions_enabled(env, _=None) -> None:
    """Enable collision on authored ToiletSeat/ToiletLid prims without debug visualization."""
    from pxr import Usd, UsdPhysics

    stage = env.sim.stage
    room_root = "/World/envs/env_0/room/Meshes/Assets"
    toilet_roots = (
        f"{room_root}/Toilet_YREQ63RVAJ2SUPTULA888888_YlDCP",
        f"{room_root}/Toilet_YREQ6BBVAI7FMPTUJQ888888_KIiuW",
    )

    for toilet_root in toilet_roots:
        toilet_prim = stage.GetPrimAtPath(toilet_root)
        if not toilet_prim.IsValid():
            continue
        for prim in Usd.PrimRange(toilet_prim, Usd.TraverseInstanceProxies()):
            if prim.GetName() in {"ToiletSeat", "ToiletLid"} and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True).Set(True)


def _enable_robot_gravity(env, _=None) -> None:
    """Make the direct USD robot behave as one physical mobile manipulator."""
    from pxr import Sdf, Usd, UsdPhysics

    stage = env.sim.stage
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/robot")
    if not robot_prim.IsValid():
        return

    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        attr = prim.GetAttribute("physxRigidBody:disableGravity")
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, custom=False)
        attr.Set(False)


@configclass
class BrushRolling0512SceneCfg(InteractiveSceneCfg):
    """Minimal scene for the brush-rolling open-seat recording task."""

    room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(usd_path=ROOM_USD),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=TARGET_BASE_POS_W,
            rot=TARGET_BASE_ROT_WXYZ,
            joint_pos={
                "wheel_fl_joint": 0.0,
                "wheel_fr_joint": 0.0,
                "wheel_rl_joint": 0.0,
                "wheel_rr_joint": 0.0,
                "joint1": TARGET_ARM_JOINT_POS[0],
                "joint2": TARGET_ARM_JOINT_POS[1],
                "joint3": TARGET_ARM_JOINT_POS[2],
                "joint4": TARGET_ARM_JOINT_POS[3],
                "joint5": TARGET_ARM_JOINT_POS[4],
                "joint6": TARGET_ARM_JOINT_POS[5],
            },
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_(fl|fr|rl|rr)_joint"],
                stiffness=0.0,
                damping=1000.0,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit_sim=100.0,
                velocity_limit_sim=1.0,
                stiffness=1600.0,
                damping=640.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    cam_tool = OrbbecGemini2(
        prim_path="{ENV_REGEX_NS}/cam_tool",
        offset=OrbbecGemini2.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
        update_latest_camera_pose=True,
    )

    # toilet_a 的 geo.usd 在所有分支中均为空文件（mesh 未上传），注释掉避免启动崩溃
    # toilet_a = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/room/Meshes/Assets/Toilet_YREQ63RVAJ2SUPTULA888888_YlDCP",
    #     spawn=None,
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         joint_pos={".*ToiletSeat$": OPEN_SEAT_RAD, ".*ToiletLid$": OPEN_SEAT_RAD}
    #     ),
    #     actuators={
    #         "seat_lid": ImplicitActuatorCfg(
    #             joint_names_expr=[".*Toilet(Seat|Lid)$"],
    #             effort_limit_sim=1.0e6,
    #             velocity_limit_sim=30.0,
    #             stiffness=5000.0,
    #             damping=500.0,
    #         )
    #     },
    # )

    toilet_b = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/room/Meshes/Assets/Toilet_YREQ6BBVAI7FMPTUJQ888888_KIiuW",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={".*ToiletSeat$": OPEN_SEAT_RAD, ".*ToiletLid$": OPEN_SEAT_RAD}
        ),
        actuators={
            "seat_lid": ImplicitActuatorCfg(
                joint_names_expr=[".*Toilet(Seat|Lid)$"],
                effort_limit_sim=1.0e6,
                velocity_limit_sim=30.0,
                stiffness=5000.0,
                damping=500.0,
            )
        },
    )

    dirt_shitstains_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_shitstains_1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.663, -0.980, 0.408), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=SHITSTAINS_PROXY_USD,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=False),
        ),
    )
    dirt_shitstains_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_shitstains_2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.656, -1.152, 0.408), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=SHITSTAINS_PROXY_USD,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=False),
        ),
    )
    dirt_shitstains_3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_shitstains_3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.648, -1.090, 0.408), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=SHITSTAINS_PROXY_USD,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=False),
        ),
    )


@configclass
class BrushRolling0512ActionsCfg:
    """Wheel and arm commands for manual recording."""

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
class BrushRolling0512ObservationsCfg:
    """Recording observations kept compatible with run_record_mobile_arm.py."""

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
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class BrushRolling0512RewardsCfg:
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class BrushRolling0512TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BrushRolling0512EventsCfg:
    enable_robot_gravity = EventTerm(func=_enable_robot_gravity, mode="prestartup")
    ensure_open_seat_lid_seat_collisions_enabled = EventTerm(
        func=_ensure_open_seat_lid_seat_collisions_enabled,
        mode="prestartup",
    )
    init_arm_joint_position = EventTerm(
        func=mdp.set_default_joint_pos,
        mode="startup",
        params={
            "default_pos": list(TARGET_ARM_JOINT_POS),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
        },
    )
    init_open_seat_toilets = EventTerm(func=_force_open_open_seat_toilets, mode="startup")
    reset_arm_joint_state = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
        },
    )
    reset_open_seat_lid_seat_collisions = EventTerm(
        func=_ensure_open_seat_lid_seat_collisions_enabled,
        mode="reset",
    )
    reset_open_seat_toilets = EventTerm(func=_force_open_open_seat_toilets, mode="reset")


@configclass
class Mecanum730XMS5BrushRolling0512BrushRolling0512DirtyOpenSeatEnvCfg(ManagerBasedRLEnvCfg):
    """BrushRolling0512 mobile-arm task used by manual recording."""

    scene: BrushRolling0512SceneCfg = BrushRolling0512SceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512ObservationsCfg = BrushRolling0512ObservationsCfg()
    actions: BrushRolling0512ActionsCfg = BrushRolling0512ActionsCfg()
    rewards: BrushRolling0512RewardsCfg = BrushRolling0512RewardsCfg()
    terminations: BrushRolling0512TerminationsCfg = BrushRolling0512TerminationsCfg()
    events: BrushRolling0512EventsCfg = BrushRolling0512EventsCfg()

    quiet_runtime_debug = True
    enable_collision_debug_visuals = False

    def __post_init__(self):
        self.decimation = 1
        self.sim.dt = 1 / 120
        self.sim.render_interval = 3
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 256 * 1024 * 1024
        self.sim.physx.gpu_max_soft_body_contacts = 4 * 2**20
        self.sim.physx.gpu_max_particle_contacts = 4 * 2**20
        self.sim.physx.gpu_heap_capacity = 256 * 1024 * 1024
        self.sim.physx.gpu_temp_buffer_capacity = 128 * 1024 * 1024
        self.episode_length_s = 30.0
        self.scene.replicate_physics = False
        self.viewer.eye = (1.660, 0.0, 1.9)
        self.viewer.lookat = (1.663, -0.980, 0.408)
        # 配置录制器：录制关节动作和观测，用于模仿学习数据集
        self.recorders = ActionStateRecorderManagerCfg()
