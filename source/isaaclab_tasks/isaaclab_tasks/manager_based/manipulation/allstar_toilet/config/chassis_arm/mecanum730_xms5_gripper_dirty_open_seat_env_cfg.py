import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from ...assets import ALLSTAR_ASSETS_DATA_DIR
from .mecanum730_xms5_gripper_dirty_env_cfg import (
    DirtyEventsCfg,
    Mecanum730XMS5GripperDirtySceneCfg,
    Mecanum730XMS5GripperDirtyEnvCfg,
)

_OPEN_SEAT_RAD = 95.0 * torch.pi / 180.0


def _debug_print(env, *args, **kwargs) -> None:
    if not bool(getattr(getattr(env, "cfg", None), "quiet_runtime_debug", False)):
        print(*args, **kwargs)


def _force_open_open_seat_toilets(env, env_ids, asset_names=("toilet_b",)):
    """Force all open-seat toilet lid/seat joints to the fully-open pose."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    target_pos = torch.full((len(env_ids), 2), 95.0 * torch.pi / 180.0, device=env.device)
    target_vel = torch.zeros_like(target_pos)

    for asset_name in asset_names:
        toilet = env.scene[asset_name]
        joint_ids, joint_names = toilet.find_joints(".*Toilet(Seat|Lid)$", preserve_order=True)
        if len(joint_ids) != 2:
            raise RuntimeError(
                f"Expected 2 toilet joints (seat/lid) on '{asset_name}', got {joint_names}."
            )
        toilet.write_joint_state_to_sim(target_pos, target_vel, joint_ids=joint_ids, env_ids=env_ids)
        toilet.set_joint_position_target(target_pos, joint_ids=joint_ids, env_ids=env_ids)
        toilet.set_joint_velocity_target(target_vel, joint_ids=joint_ids, env_ids=env_ids)
        toilet.write_joint_stiffness_to_sim(5000.0, joint_ids=joint_ids, env_ids=env_ids)
        toilet.write_joint_damping_to_sim(500.0, joint_ids=joint_ids, env_ids=env_ids)
        _debug_print(
            env,
            f"[OpenSeat] Forced '{asset_name}' joints {joint_names} to 95 deg for env_ids={env_ids.tolist()}",
            flush=True,
        )


def _ensure_open_seat_lid_seat_collisions_enabled(env, _=None) -> None:
    """Keep open-seat lid/seat collision meshes enabled so they follow the joints."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = env.sim.stage
    collision_debug_visuals_enabled = bool(getattr(env.cfg, "enable_collision_debug_visuals", True))
    room_root = "/World/envs/env_0/room/Meshes/Assets"
    toilet_roots = (
        f"{room_root}/Toilet_YREQ63RVAJ2SUPTULA888888_YlDCP",
        f"{room_root}/Toilet_YREQ6BBVAI7FMPTUJQ888888_KIiuW",
    )
    movable_names = {"ToiletSeat", "ToiletLid"}

    stale_debug_root = "/World/collision_mesh_debug/env_0"
    stale_debug_prim = stage.GetPrimAtPath(stale_debug_root)
    if stale_debug_prim.IsValid():
        stale_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(stale_debug_prim)
            if "ToiletSeat" in str(prim.GetPath()) or "ToiletLid" in str(prim.GetPath())
        ]
        for stale_path in stale_paths:
            stage.RemovePrim(stale_path)
            _debug_print(env, f"[OpenSeatCollision] Removed stale lid/seat debug copy: {stale_path}", flush=True)

    for toilet_root in toilet_roots:
        toilet_prim = stage.GetPrimAtPath(toilet_root)
        if not toilet_prim.IsValid():
            continue
        stale_child_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(toilet_prim)
            if prim.GetName() == "collision_debug_red"
        ]
        for stale_path in stale_child_paths:
            stage.RemovePrim(stale_path)
            _debug_print(env, f"[OpenSeatCollision] Removed stale lid/seat child debug mesh: {stale_path}", flush=True)

    def _show_collision_mesh_in_red(prim) -> bool:
        if not collision_debug_visuals_enabled:
            return False
        if not prim.IsA(UsdGeom.Mesh):
            return False

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        if not points or not counts or not indices:
            return False

        debug_path = f"{prim.GetPath()}/collision_debug_red"
        if stage.GetPrimAtPath(debug_path).IsValid():
            stage.RemovePrim(debug_path)
        debug_mesh = UsdGeom.Mesh.Define(stage, debug_path)
        debug_mesh.CreatePointsAttr(points)
        debug_mesh.CreateFaceVertexCountsAttr(counts)
        debug_mesh.CreateFaceVertexIndicesAttr(indices)
        debug_mesh.CreateDoubleSidedAttr(True)
        debug_mesh.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 0.0)])
        debug_mesh.CreateDisplayOpacityAttr([0.78])
        debug_mesh.CreatePurposeAttr(UsdGeom.Tokens.default_)
        debug_mesh.CreateVisibilityAttr(UsdGeom.Tokens.inherited)
        return True

    enabled = 0
    red_debug_created = 0
    missing = []
    for toilet_root in toilet_roots:
        toilet_prim = stage.GetPrimAtPath(toilet_root)
        if not toilet_prim.IsValid():
            missing.append(toilet_root)
            continue
        for prim in Usd.PrimRange(toilet_prim, Usd.TraverseInstanceProxies()):
            if prim.GetName() not in movable_names or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True).Set(True)
            if collision_debug_visuals_enabled and prim.IsA(UsdGeom.Gprim):
                gprim = UsdGeom.Gprim(prim)
                gprim.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 0.0)])
                gprim.CreateDisplayOpacityAttr([0.78])
                gprim.CreatePurposeAttr(UsdGeom.Tokens.default_)
                gprim.CreateVisibilityAttr(UsdGeom.Tokens.inherited)
                if prim.IsA(UsdGeom.Mesh):
                    UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
                    if _show_collision_mesh_in_red(prim):
                        red_debug_created += 1
            enabled += 1
            _debug_print(env, f"[OpenSeatCollision] Kept lid/seat collision enabled: {prim.GetPath()}", flush=True)

    if missing:
        _debug_print(env, f"[OpenSeatCollision] Warning missing toilet roots: {missing}", flush=True)
    if collision_debug_visuals_enabled:
        _debug_print(
            env,
            f"[OpenSeatCollision] Ensured {enabled} open-seat lid/seat collision prims stay enabled; "
            f"created {red_debug_created} red child debug meshes that follow their parent lid/seat prims.",
            flush=True,
        )
    else:
        _debug_print(
            env,
            f"[OpenSeatCollision] Ensured {enabled} open-seat lid/seat collision prims stay enabled; "
            "collision mesh visualization disabled.",
            flush=True,
        )


@configclass
class Mecanum730XMS5GripperDirtyOpenSeatSceneCfg(Mecanum730XMS5GripperDirtySceneCfg):
    """Dirty 841837 restroom scene with both toilet lids and seats opened by default."""

    # toilet_a 的 geo.usd 在所有分支中均为空文件（mesh 未上传），注释掉避免启动崩溃
    # toilet_a = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/room/Meshes/Assets/Toilet_YREQ63RVAJ2SUPTULA888888_YlDCP",
    #     spawn=None,
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         joint_pos={
    #             ".*ToiletSeat$": _OPEN_SEAT_RAD,
    #             ".*ToiletLid$": _OPEN_SEAT_RAD,
    #         }
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
            joint_pos={
                ".*ToiletSeat$": _OPEN_SEAT_RAD,
                ".*ToiletLid$": _OPEN_SEAT_RAD,
            }
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

    def __post_init__(self):
        super().__post_init__()
        self.room.spawn = UsdFileCfg(
            usd_path=f"{ALLSTAR_ASSETS_DATA_DIR}/scenes/real/shenxinfu/841837/start_open_seat.usda"
        )
        self.robot.init_state.joint_pos["joint1"] = -0.0
        self.robot.init_state.joint_pos["joint2"] = 0.561996
        self.robot.init_state.joint_pos["joint3"] = -0.570723
        self.robot.init_state.joint_pos["joint4"] = -0.561996
        self.robot.init_state.joint_pos["joint5"] = 0.55676
        self.robot.init_state.joint_pos["joint6"] = 0.0


@configclass
class Mecanum730XMS5GripperDirtyOpenSeatEnvCfg(Mecanum730XMS5GripperDirtyEnvCfg):
    """Dirty mobile-manipulator task with toilets starting in lid-up + seat-up state."""

    @configclass
    class OpenSeatEventsCfg(DirtyEventsCfg):
        ensure_open_seat_lid_seat_collisions_enabled = EventTerm(
            func=_ensure_open_seat_lid_seat_collisions_enabled,
            mode="prestartup",
        )

        init_open_seat_toilets = EventTerm(
            func=_force_open_open_seat_toilets,
            mode="startup",
        )

        reset_open_seat_lid_seat_collisions = EventTerm(
            func=_ensure_open_seat_lid_seat_collisions_enabled,
            mode="reset",
        )

        reset_open_seat_toilets = EventTerm(
            func=_force_open_open_seat_toilets,
            mode="reset",
        )

    scene: Mecanum730XMS5GripperDirtyOpenSeatSceneCfg = Mecanum730XMS5GripperDirtyOpenSeatSceneCfg(
        num_envs=1, env_spacing=2.5
    )
    events: OpenSeatEventsCfg = OpenSeatEventsCfg()
