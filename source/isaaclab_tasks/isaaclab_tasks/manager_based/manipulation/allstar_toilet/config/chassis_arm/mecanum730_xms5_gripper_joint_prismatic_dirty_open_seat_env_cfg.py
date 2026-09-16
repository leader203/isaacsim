"""GripperJointPrismatic config for the mecanum730 + XMS5 prismatic-gripper recording task.

This file intentionally avoids inheriting the long dirty/open-seat/brush/gripper
task chain. It keeps only the assets, controls, observations, and events used by
the prismatic-gripper mobile-arm recording command.
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from ...utils.paper_waste_spawner import spawn_used_wet_paper_clump, spawn_used_flattened_tissue


ROOM_USD = (
    f"{ALLSTAR_ASSETS_DATA_DIR}/scenes/real/shenxinfu/841837/"
    "start_open_seat.usda"
)
ROBOT_USD = (
    f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_sphere_usd/"
    "mecanum730_xms5_gripper_joint_prismatic.usd"
)
SHITSTAINS_PROXY_USD = f"{ALLSTAR_ASSETS_DATA_DIR}/scenes/rx/Shitstains/Shitstains_proxy.usda"
TRASH_BIN_USD = f"{ALLSTAR_ASSETS_DATA_DIR}/props/open_trash_bin.usda"

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
DEFAULT_PRISMATIC_GRIPPER_OPEN_M = 0.0


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


def _add_experiment_wall_obstacle(env, _=None) -> None:
    """PhysX raycast 扫描卫生间墙壁 → 生成 cuRobo-only cuboid（curobo:obstacle 属性，不加 CollisionAPI）。

    扫描 ROI 多个高度平面上的水平射线，命中墙壁/隔断 prim 后聚类生成薄 cuboid。
    """
    from pxr import Gf, Sdf, UsdGeom

    stage = env.sim.stage

    # 确保 cuRobo-only 墙壁容器路径存在
    for prim_path in [
        "/World/envs/env_0/room/CollisionFixes",
        "/World/envs/env_0/room/CollisionFixes/CuroboOnlyWalls",
    ]:
        if not stage.GetPrimAtPath(prim_path).IsValid():
            stage.DefinePrim(prim_path, "Xform")

    # === PhysX raycast 扫描 ===
    try:
        import omni.physx
        sq = omni.physx.get_physx_scene_query_interface()
    except Exception as e:
        print(f"[RaycastWalls] PhysX scene query unavailable: {e}", flush=True)
        return

    # 任务 ROI (world 坐标)
    x_min, x_max = 1.0, 2.4
    y_min, y_max = -1.6, -0.35
    z_scan = [0.2, 0.5, 0.9, 1.4]
    step = 0.05
    max_dist = 4.0

    def accept_hit(hit_path: str) -> bool:
        s = hit_path.lower()
        reject = ["robot", "cube", "trash", "toilet", "gripper", "mecanum", "dirt", "shitstain", "floor", "ceiling"]
        if any(k in s for k in reject):
            return False
        allow = ["wall", "partition", "/room/meshes/base"]
        return any(k in s for k in allow)

    hits = []  # [(x, y, z), ...]

    for z in z_scan:
        # X 方向扫描（从左右两边）
        for y in [y_min + i * step for i in range(int((y_max - y_min) / step) + 1)]:
            for direction, start_x in [(1.0, x_min), (-1.0, x_max)]:
                try:
                    hit = sq.raycast_closest((start_x, y, z), (direction, 0.0, 0.0), max_dist)
                except Exception:
                    continue
                if hit and hit.get("hit"):
                    hp = hit.get("position", (0, 0, 0))
                    rp = hit.get("rigidBody", "")
                    if accept_hit(str(rp)):
                        hits.append(tuple(float(v) for v in hp))

        # Y 方向扫描（从前后两边）
        for x in [x_min + i * step for i in range(int((x_max - x_min) / step) + 1)]:
            for direction, start_y in [(1.0, y_min), (-1.0, y_max)]:
                try:
                    hit = sq.raycast_closest((x, start_y, z), (0.0, direction, 0.0), max_dist)
                except Exception:
                    continue
                if hit and hit.get("hit"):
                    hp = hit.get("position", (0, 0, 0))
                    rp = hit.get("rigidBody", "")
                    if accept_hit(str(rp)):
                        hits.append(tuple(float(v) for v in hp))

    print(f"[RaycastWalls] {len(hits)} raycast hits on walls", flush=True)
    if len(hits) < 10:
        print("[RaycastWalls] too few hits, skipping cuboid generation", flush=True)
        return

    # === 聚类：按 X 常量（竖直墙沿 Y 延伸）或 Y 常量（水平墙沿 X 延伸）===
    def cluster_1d(values, binsize=0.04, min_count=8):
        bins = {}
        for v in values:
            key = round(v[0] / binsize) * binsize if v[2] == "x" else round(v[1] / binsize) * binsize
            bins.setdefault(key, []).append(v)
        return [pts for pts in bins.values() if len(pts) >= min_count]

    # X 方向聚类（hit 点 x 基本固定 → 沿 Y 的墙）
    x_clusters = cluster_1d([(h[0], h[1], h[2], "x") for h in hits])
    # Y 方向聚类（hit 点 y 基本固定 → 沿 X 的墙）
    y_clusters = cluster_1d([(h[0], h[1], h[2], "y") for h in hits])

    cuboids = []
    for cluster in x_clusters:
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        cx = sum(xs) / len(xs)
        cy = (min(ys) + max(ys)) * 0.5
        dy = (max(ys) - min(ys)) + 0.10
        cuboids.append((cx, cy, 1.0, 0.06, dy, 2.0))

    for cluster in y_clusters:
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        cy = sum(ys) / len(ys)
        cx = (min(xs) + max(xs)) * 0.5
        dx = (max(xs) - min(xs)) + 0.10
        cuboids.append((cx, cy, 1.0, dx, 0.06, 2.0))

    # 去重（合并相近 cuboid）
    unique = []
    for c in cuboids:
        is_dup = False
        for u in unique:
            if abs(c[0] - u[0]) < 0.1 and abs(c[1] - u[1]) < 0.1:
                is_dup = True; break
        if not is_dup:
            unique.append(c)
    cuboids = unique

    # === 注入 cuRobo-only cuboid（curobo:obstacle 属性，不加 CollisionAPI）===
    for i, (cx, cy, cz, sx, sy, sz) in enumerate(cuboids):
        cube_path = f"/World/envs/env_0/room/CollisionFixes/CuroboOnlyWalls/raycast_wall_{i:03d}"
        if stage.GetPrimAtPath(cube_path).IsValid():
            continue
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.CreateSizeAttr(1.0)
        prim = cube.GetPrim()
        prim.CreateAttribute("curobo:obstacle", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("curobo:source", Sdf.ValueTypeNames.String).Set("physx_raycast")
        xformable = UsdGeom.Xformable(prim)
        xformable.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
        xformable.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))

    print(f"[RaycastWalls] 生成 {len(cuboids)} 个 cuRobo-only 墙壁 cuboid（无 CollisionAPI）", flush=True)


def _enable_robot_gravity(env, _=None) -> None:
    """Disable robot gravity — 同事建议：机械臂 joint2/3 在低 step 数轨迹执行时 hold 不
    住位置（重力导致 joint 漂移），关掉重力让关节位置跟踪更精准。"""
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
        attr.Set(True)  # 关闭重力，防止 joint2/3 因重力漂移导致末端对不准


def _spawn_paper_waste_objects(env, _=None) -> None:
    """Spawn paper-waste objects in the scene (prestartup EventTerm)."""
    stage = env.sim.stage

    # --- Wet paper clumps (sphere-like) ---
    clumps = [
        ("used_wet_paper_clump_01", (1.75, -0.70, 0.035), 0.04, 0.02),
        ("used_wet_paper_clump_02", (1.409, -1.157, 0.035), 0.04, 0.02),
    ]
    for name, pos, radius, mass in clumps:
        spawn_used_wet_paper_clump(
            stage,
            f"/World/envs/env_0/paper_waste/{name}",
            position=pos,
            radius=radius,
            mass=mass,
            make_dynamic=True,
            curobo_proxy=True,
        )

    # --- Flattened tissues — toilet edge, stuck (grasped via logical attach) ---
    tissues = [
        ("used_flattened_tissue_01", (1.663, -0.980, 0.408)),
        ("used_flattened_tissue_02", (1.656, -1.152, 0.408)),
    ]
    for name, pos in tissues:
        spawn_used_flattened_tissue(
            stage,
            f"/World/envs/env_0/paper_waste/{name}",
            position=pos,
            size=(0.08, 0.06, 0.010),
            mass=0.008,
            initial_state="stuck",         # stay on toilet edge
            make_dynamic=False,            # static; grasped via logical attach
            curobo_proxy=True,
            visual_z_m=0.003,
        )


def _make_grasp_cube_spawn() -> sim_utils.MeshCuboidCfg:
    return sim_utils.MeshCuboidCfg(
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
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.55, 0.95)),
    )


@configclass
class GripperJointPrismaticSceneCfg(InteractiveSceneCfg):
    """Minimal scene for the prismatic-gripper open-seat recording task."""

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
                enabled_self_collisions=False,  # no link self-collision (group0 filter removed)
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
                "gripper_joint1": DEFAULT_PRISMATIC_GRIPPER_OPEN_M,
                "gripper_joint2": -DEFAULT_PRISMATIC_GRIPPER_OPEN_M,
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
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_joint[12]"],
                effort_limit_sim=20.0,
                velocity_limit_sim=0.2,
                stiffness=800.0,
                damping=80.0,
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

    # toilet_a 的 geo.usd 在仓库中是空文件（mesh 未上传），暂时注释掉避免启动崩溃
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

    # 注意：方块绕 Z 转 90°（rot=(0.7071,0,0,0.7071)），使其 50mm 宽的一侧沿世界 X。
    # 夹爪手指沿世界 X 开合（默认抓取姿态 tool-Y∥世界X），原 rot=(1,0,0,0) 时手指
    # 沿方块 15mm 窄边闭合，而夹爪最小闭合间隙 ~19mm > 15mm → 永远夹不住。旋转后
    # 手指沿 50mm 宽边闭合才能夹住。
    grasp_test_cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/grasp_test_cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.486, -1.071, 0.021), rot=(0.7071068, 0.0, 0.0, 0.7071068)),
        spawn=_make_grasp_cube_spawn(),
    )
    grasp_test_cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/grasp_test_cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.409, -1.157, 0.021), rot=(0.7071068, 0.0, 0.0, 0.7071068)),
        spawn=_make_grasp_cube_spawn(),
    )
    grasp_test_cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/grasp_test_cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.355, -1.218, 0.021), rot=(0.7071068, 0.0, 0.0, 0.7071068)),
        spawn=_make_grasp_cube_spawn(),
    )

    trash_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/trash_bin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.097, -1.095, 0.010), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=TRASH_BIN_USD,
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


@configclass
class GripperJointPrismaticActionsCfg:
    """Wheel, arm, and prismatic gripper commands."""

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
class GripperJointPrismaticObservationsCfg:
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
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["tool_link"])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class GripperJointPrismaticRewardsCfg:
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class GripperJointPrismaticTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class GripperJointPrismaticEventsCfg:
    add_experiment_wall_obstacle = EventTerm(func=_add_experiment_wall_obstacle, mode="prestartup")
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
    # init_open_seat_toilets 和 reset_open_seat_toilets 依赖 toilet_a/toilet_b articulation，
    # 两者均已注释掉，这里也跳过
    # init_open_seat_toilets = EventTerm(func=_force_open_open_seat_toilets, mode="startup")
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
    # reset_open_seat_toilets = EventTerm(func=_force_open_open_seat_toilets, mode="reset")
    reset_open_seat_toilets = EventTerm(func=_force_open_open_seat_toilets, mode="reset")


@configclass
class Mecanum730XMS5GripperJointPrismaticDirtyOpenSeatEnvCfg(ManagerBasedRLEnvCfg):
    """GripperJointPrismatic prismatic-gripper mobile-arm task used by manual recording."""

    scene: GripperJointPrismaticSceneCfg = GripperJointPrismaticSceneCfg(num_envs=1, env_spacing=2.5)
    observations: GripperJointPrismaticObservationsCfg = GripperJointPrismaticObservationsCfg()
    actions: GripperJointPrismaticActionsCfg = GripperJointPrismaticActionsCfg()
    rewards: GripperJointPrismaticRewardsCfg = GripperJointPrismaticRewardsCfg()
    terminations: GripperJointPrismaticTerminationsCfg = GripperJointPrismaticTerminationsCfg()
    events: GripperJointPrismaticEventsCfg = GripperJointPrismaticEventsCfg()

    quiet_runtime_debug = True
    enable_collision_debug_visuals = False

    def __post_init__(self):
        self.decimation = 1
        self.sim.enable_scene_query_support = True  # PhysX raycast 扫描墙壁
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
