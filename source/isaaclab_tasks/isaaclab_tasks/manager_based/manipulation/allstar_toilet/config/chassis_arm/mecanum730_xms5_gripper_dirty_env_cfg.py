# Mecanum 0730 + XMS5 Gripper mobile manipulator — 841837 real restroom scene with 10 dirt props.
#
# 继承关系：
#   Mecanum730XMS5GripperDirtyEnvCfg
#   └── Mecanum730XMS5GripperEnvCfg          (mecanum730_xms5_gripper_env_cfg.py)
#       ├── scene: Mecanum730XMS5GripperDirtySceneCfg
#       │   └── Mecanum730XMS5GripperSceneCfg  ← 基础场景 + 机器人 + 相机
#       │       + 10 个污垢道具（本文件新增）
#       ├── actions: wheel_velocity（4 轮速度） + arm_joint_pos（6 轴位置）
#       └── 其余 observations / rewards / terminations 保持不变
#
# 任务 ID：Isaac-Allstar-Toilet-Mecanum0730-XMS5Gripper-Dirty-v0
#
# 启动命令：
#   ./isaaclab.sh -p .../scripts/run_record_mobile_arm.py \
#       --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Gripper-Dirty-v0 \
#       --num_demos 10 \
#       --dataset_file /tmp/mecanum730_xms5_gripper_dirty_demos.hdf5 \
#       --speed_scale 20.0 --arm_scale 0.05

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from ...assets import ALLSTAR_ASSETS_DATA_DIR
from .mecanum730_xms5_gripper_env_cfg import (
    Mecanum730XMS5GripperSceneCfg,
    Mecanum730XMS5GripperEnvCfg,
)


# ─────────────────────────────────────────────────────────────────────────────
# Startup event: 扫描隔间入口碰撞体，移除低矮障碍物（门槛/底轨）的碰撞
# ─────────────────────────────────────────────────────────────────────────────

_ROBOT_ARM_BODY_NAMES = {
    "XMS5_R800_W4G3B4C_base",
    "XMS5_R800_W4G3B4C_link1",
    "XMS5_R800_W4G3B4C_link2",
    "XMS5_R800_W4G3B4C_link3",
    "XMS5_R800_W4G3B4C_link4",
    "XMS5_R800_W4G3B4C_link5",
    "link6",
    "tool_link",
    "tool_tip_link",
}


def _debug_print(env, *args, **kwargs) -> None:
    if not bool(getattr(getattr(env, "cfg", None), "quiet_runtime_debug", False)):
        print(*args, **kwargs)


def _set_robot_arm_gravity_enabled(env, reason: str) -> None:
    """Enable gravity for robot rigid bodies requested by a task cfg."""
    enable_all_robot_gravity = bool(getattr(env.cfg, "enable_robot_all_gravity", False))
    enable_arm_gravity = bool(getattr(env.cfg, "enable_robot_arm_gravity", False))
    if not (enable_all_robot_gravity or enable_arm_gravity):
        return

    from pxr import Sdf, Usd, UsdPhysics

    stage = env.sim.stage
    robot_root = "/World/envs/env_0/robot"
    robot_prim = stage.GetPrimAtPath(robot_root)
    if not robot_prim.IsValid():
        _debug_print(env, f"[ArmGravity] Robot root not found ({reason}): {robot_root}", flush=True)
        return

    updated = 0
    missing = [] if enable_all_robot_gravity else sorted(_ROBOT_ARM_BODY_NAMES)
    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if not enable_all_robot_gravity and prim.GetName() not in _ROBOT_ARM_BODY_NAMES:
            continue
        attr = prim.GetAttribute("physxRigidBody:disableGravity")
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool, custom=False)
        attr.Set(False)
        updated += 1
        if prim.GetName() in missing:
            missing.remove(prim.GetName())

    if missing:
        _debug_print(env, f"[ArmGravity] Warning missing arm rigid bodies ({reason}): {missing}", flush=True)
    label = "robot" if enable_all_robot_gravity else "arm"
    tag = "RobotGravity" if enable_all_robot_gravity else "ArmGravity"
    _debug_print(
        env,
        f"[{tag}] {reason}: enabled {label} gravity on {updated} rigid bodies "
        "(physxRigidBody:disableGravity=False).",
        flush=True,
    )


def _fix_stall_entrance(env, _=None):
    """Patch unstable collision meshes around the target dirty stall.

    策略：
    0. 先显式打开 robot 与 room 下所有 USD CollisionAPI prim 的 collisionEnabled；
       后续步骤只关闭已确认会造成卡顿/错误阻挡的坏场景 mesh。
    1. 扫描全场（x 全范围）入口前沿（y∈[-0.15, 0.05]）所有含 CollisionAPI 的薄板/竖柱
       (y_extent < 0.30m)，调用 SetActive(False) 让 PhysX 和可视化都彻底忽略它们。
    2. 将剩余零厚度 Wall_0000 mesh 碰撞替换成不可见 box primitive。
    3. 将 Partition_* mesh 碰撞替换成 box primitive，仅合并完全/近似重叠的 box。
    4. 按已知路径直接关闭两扇隔间门（payload USD 里有内嵌碰撞）。
    5. 关闭 3 个黄色 dirty 所在隔间的破碎 floor SDF/入口地板接缝碰撞，
       入口竖向薄片直接停用，并添加一个干净的不可见薄盒地板碰撞替代。
    6. 关闭该隔间上方天花板/灯具/风口碰撞，避免高复杂度 mesh 参与 PhysX。
    7. 当 enable_collision_debug_visuals=True 时，扫描 env_0 下所有非机器人场景碰撞 prim，
       在 /World/collision_mesh_debug/env_0 下生成可见调试几何；每个碰撞 mesh 用 source
       prim path 派生出的稳定颜色区分。调试几何不带 CollisionAPI，不参与物理。
    8. 当 enable_collision_debug_visuals=True 时，扫描 robot 下 USD 自带的 CollisionAPI prim，
       直接让原始碰撞几何可见。
    SetActive 优先级高于任何 sublayer，RemoveAPI 不刷新 PhysX 的问题不存在。
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = env.sim.stage
    env_root = "/World/envs/env_0"
    room_root = "/World/envs/env_0/room"
    robot_root = "/World/envs/env_0/robot"
    debug_root = "/World/collision_mesh_debug/env_0"
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=True,
        ignoreVisibility=True,
    )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    Y_CENTER_LO, Y_CENTER_HI = -0.15, 0.05   # 入口前沿（含门框）
    Y_EXTENT_MAX = 0.30                        # 超过此 Y 跨度的是贯穿侧墙，保留
    WALL_BOX_THICKNESS_M = 0.001
    WALL_BOX_OUTWARD_EXPAND_M = 0.01
    WALL_FACE_MERGE_GAP_M = 0.05
    WALL_FACE_MERGE_OVERLAP_RATIO = 0.85
    PARTITION_BOX_THICKNESS_M = 0.013
    PARTITION_OVERLAP_EPS_M = 0.02

    deactivated = 0
    collision_disabled = 0
    debug_visual_count = 0
    wall_mesh_replaced_collision_paths: set[str] = set()
    wall_box_collision_paths: set[str] = set()
    partition_mesh_replaced_collision_paths: set[str] = set()
    partition_box_collision_paths: set[str] = set()
    collision_debug_visuals_enabled = bool(getattr(env.cfg, "enable_collision_debug_visuals", True))
    if not collision_debug_visuals_enabled:
        _debug_print(env, "[CollisionDebug] Collision mesh visualization disabled by env cfg.", flush=True)

    def _disable_robot_base_collision_requested() -> bool:
        return bool(getattr(env.cfg, "disable_robot_base_collision", False))

    def _safe_debug_name(path_str: str) -> str:
        if path_str.startswith(f"{env_root}/"):
            path_str = path_str.removeprefix(f"{env_root}/")
        return path_str.replace("/", "__").replace(":", "_").replace(".", "_").replace("-", "_")

    def _style_debug_gprim(
        gprim,
        color: tuple[float, float, float],
        opacity: float,
        set_visibility: bool = True,
    ) -> None:
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        gprim.CreateDisplayOpacityAttr([opacity])
        gprim.CreatePurposeAttr(UsdGeom.Tokens.default_)
        if set_visibility:
            gprim.CreateVisibilityAttr(UsdGeom.Tokens.inherited)

    def _color_from_path(path_str: str, salt: str = "") -> tuple[float, float, float]:
        value = 2166136261
        for ch in f"{salt}:{path_str}":
            value ^= ord(ch)
            value = (value * 16777619) & 0xFFFFFFFF
        hue = (value % 360) / 360.0
        sat = 0.68 + (((value >> 9) % 24) / 100.0)
        val = 0.82 + (((value >> 17) % 16) / 100.0)
        sector = int(hue * 6.0)
        frac = hue * 6.0 - sector
        p = val * (1.0 - sat)
        q = val * (1.0 - sat * frac)
        t = val * (1.0 - sat * (1.0 - frac))
        sector %= 6
        if sector == 0:
            return (val, t, p)
        if sector == 1:
            return (q, val, p)
        if sector == 2:
            return (p, val, t)
        if sector == 3:
            return (p, q, val)
        if sector == 4:
            return (t, p, val)
        return (val, p, q)

    def _debug_cube_from_bbox(debug_path: str, bmin, bmax, color: tuple[float, float, float], opacity: float) -> None:
        center = Gf.Vec3d(
            (float(bmin[0]) + float(bmax[0])) * 0.5,
            (float(bmin[1]) + float(bmax[1])) * 0.5,
            (float(bmin[2]) + float(bmax[2])) * 0.5,
        )
        extent = Gf.Vec3f(
            max(float(bmax[0]) - float(bmin[0]), 0.01),
            max(float(bmax[1]) - float(bmin[1]), 0.01),
            max(float(bmax[2]) - float(bmin[2]), 0.01),
        )
        cube = UsdGeom.Cube.Define(stage, debug_path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(center)
        cube.AddScaleOp().Set(extent)
        _style_debug_gprim(cube, color, opacity)

    def _create_collision_debug_visual(
        prim,
        reason: str,
        color: tuple[float, float, float] | None = None,
        opacity: float = 0.35,
    ) -> bool:
        """Create a visible, non-physical copy/proxy of a non-robot scene collision prim."""
        nonlocal debug_visual_count
        if not collision_debug_visuals_enabled:
            return False
        if not prim.IsValid():
            return False

        source_path = str(prim.GetPath())
        if source_path.startswith(robot_root):
            return False
        if color is None:
            color = _color_from_path(source_path)
        debug_path = f"{debug_root}/{_safe_debug_name(source_path)}"
        if stage.GetPrimAtPath(debug_path).IsValid():
            return False

        try:
            bbox = cache.ComputeWorldBound(prim)
            rng = bbox.GetRange()
            if rng.IsEmpty():
                return False
            bmin, bmax = rng.GetMin(), rng.GetMax()
        except Exception:
            return False

        UsdGeom.Xform.Define(stage, "/World/collision_mesh_debug")
        UsdGeom.Xform.Define(stage, debug_root)

        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get() or []
            counts = mesh.GetFaceVertexCountsAttr().Get() or []
            indices = mesh.GetFaceVertexIndicesAttr().Get() or []
            tri_equiv = sum(max(0, int(count) - 2) for count in counts)
            # Copy real mesh for target stall floor/entrance. For very dense fixtures, use bbox
            # proxy to avoid cloning million-triangle light meshes into the debug layer.
            if points and counts and indices and len(points) <= 60000 and tri_equiv <= 120000:
                world_from_local = xform_cache.GetLocalToWorldTransform(prim)
                world_points = [
                    world_from_local.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                    for point in points
                ]
                debug_mesh = UsdGeom.Mesh.Define(stage, debug_path)
                debug_mesh.CreatePointsAttr(world_points)
                debug_mesh.CreateFaceVertexCountsAttr(counts)
                debug_mesh.CreateFaceVertexIndicesAttr(indices)
                debug_mesh.CreateDoubleSidedAttr(True)
                _style_debug_gprim(debug_mesh, color, opacity)
            else:
                _debug_cube_from_bbox(debug_path, bmin, bmax, color, opacity)
        else:
            _debug_cube_from_bbox(debug_path, bmin, bmax, color, opacity)

        debug_visual_count += 1
        _debug_print(env, f"[CollisionDebug] Visualized collision ({reason}): {debug_path} <- {source_path}", flush=True)
        return True

    def _set_collision_enabled(prim, enabled: bool, reason: str) -> bool:
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            return False

        path_str = str(prim.GetPath())
        try:
            ancestor = prim
            while ancestor.IsValid():
                if ancestor.IsInstance():
                    ancestor.SetInstanceable(False)
                    prim = stage.GetPrimAtPath(path_str)
                    break
                ancestor = ancestor.GetParent()

            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                return False

            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(enabled).Set(enabled)
            return True
        except Exception as exc:
            print(
                f"[CollisionEnable] Warning collisionEnabled={enabled} failed "
                f"({reason}): {path_str}: {exc}",
                flush=True,
            )
            return False

    def _is_disabled_robot_base_collision(prim) -> bool:
        return _disable_robot_base_collision_requested() and str(prim.GetPath()) == f"{robot_root}/base_link/collisions"

    def _enforce_robot_collision_policy(reason: str) -> tuple[int, int, int]:
        """Keep chassis collision disabled while all arm collision prims stay enabled."""
        robot_prim = stage.GetPrimAtPath(robot_root)
        if not robot_prim.IsValid():
            _debug_print(env, f"[CollisionPolicy] Robot root not found ({reason}): {robot_root}", flush=True)
            return 0, 0, 0

        total = 0
        base_disabled = 0
        non_base_enabled = 0
        for prim in list(Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies())):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            total += 1
            if _is_disabled_robot_base_collision(prim):
                if _set_collision_enabled(prim, False, reason):
                    base_disabled += 1
            else:
                if _set_collision_enabled(prim, True, reason):
                    non_base_enabled += 1

        _debug_print(
            env,
            f"[CollisionPolicy] {reason}: base_collision_disabled={base_disabled}, "
            f"non_base_robot_collision_enabled={non_base_enabled}, total_robot_collision_prims={total}.",
            flush=True,
        )
        return total, base_disabled, non_base_enabled

    def _enable_collision_subtree(root_path: str, reason: str) -> tuple[int, int]:
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            _debug_print(env, f"[CollisionEnable] Root not found ({reason}): {root_path}", flush=True)
            return 0, 0

        total = 0
        enabled = 0
        for prim in list(Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies())):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            total += 1
            if _set_collision_enabled(prim, True, reason):
                enabled += 1

        _debug_print(
            env,
            f"[CollisionEnable] Explicitly enabled {enabled}/{total} CollisionAPI prims under "
            f"{root_path} ({reason}).",
            flush=True,
        )
        return total, enabled

    def _disable_collision(prim, reason: str) -> bool:
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            return False
        _create_collision_debug_visual(prim, reason)
        if _set_collision_enabled(prim, False, reason):
            _debug_print(env, f"[StallFix] Disabled collision ({reason}): {prim.GetPath()}", flush=True)
            return True
        return False

    def _visualize_collision_subtree(root_prim, reason: str, opacity: float = 0.35) -> int:
        if not root_prim.IsValid():
            return 0
        created = 0
        for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if _create_collision_debug_visual(prim, reason, opacity=opacity):
                    created += 1
        return created

    def _visualize_all_scene_collisions() -> tuple[int, int]:
        scene_collision_color = (0.10, 0.78, 0.34)
        scene_box_collision_color = (1.00, 0.00, 0.00)
        total = 0
        created = 0
        hidden_collision_fix_prims = 0
        if not collision_debug_visuals_enabled and stage.GetPrimAtPath(debug_root).IsValid():
            stage.RemovePrim(debug_root)
            _debug_print(env, f"[CollisionDebug] Removed stale scene debug root: {debug_root}", flush=True)
        for prim in list(stage.Traverse()):
            path_str = str(prim.GetPath())
            if not path_str.startswith(env_root):
                continue
            if path_str.startswith(f"{env_root}/robot"):
                continue
            if path_str in wall_mesh_replaced_collision_paths or path_str in partition_mesh_replaced_collision_paths:
                continue
            if path_str in wall_box_collision_paths or path_str in partition_box_collision_paths:
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            # Articulated toilet lids/seats are visualized in-place by open-seat tasks.
            # A copied debug mesh here would freeze their prestartup closed-pose transform.
            if prim.GetName() in {"ToiletSeat", "ToiletLid"}:
                continue
            color = scene_box_collision_color if prim.IsA(UsdGeom.Cube) else scene_collision_color
            total += 1
            if not collision_debug_visuals_enabled:
                if "/CollisionFixes/" in path_str and prim.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                    hidden_collision_fix_prims += 1
                continue
            if _create_collision_debug_visual(
                prim,
                "scene collision",
                color=color,
                opacity=0.28,
            ):
                created += 1
        if not collision_debug_visuals_enabled:
            _debug_print(
                env,
                f"[CollisionDebug] Scene collision visualization skipped: "
                f"{total} non-robot collision prims found under {env_root}; "
                f"hidden_collision_fix_prims={hidden_collision_fix_prims}.",
                flush=True,
            )
            return total, 0
        _debug_print(
            env,
            f"[CollisionDebug] Scene collision visualization complete: "
            f"{created} newly visualized, {total} non-robot collision prims found under {env_root}.",
            flush=True,
        )
        return total, created

    def _replace_wall_mesh_collisions_with_boxes() -> tuple[int, int, int]:
        """Replace zero-thickness wall mesh collisions with invisible box primitives."""
        wall_geom_root = f"{room_root}/Meshes/Base/Wall_0000/Geom"
        wall_box_root = f"{room_root}/CollisionFixes/WallBoxes"
        wall_root_prim = stage.GetPrimAtPath(wall_geom_root)
        if not wall_root_prim.IsValid():
            _debug_print(env, f"[WallBoxCollision] Wall geom root not found: {wall_geom_root}", flush=True)
            return 0, 0, 0

        UsdGeom.Xform.Define(stage, f"{room_root}/CollisionFixes")
        UsdGeom.Xform.Define(stage, wall_box_root)

        scanned = 0
        created = 0
        disabled = 0
        records: list[dict] = []
        wall_scene_bmin = [float("inf"), float("inf"), float("inf")]
        wall_scene_bmax = [float("-inf"), float("-inf"), float("-inf")]
        for prim in list(Usd.PrimRange(wall_root_prim, Usd.TraverseInstanceProxies())):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            scanned += 1
            source_path = str(prim.GetPath())
            try:
                rng = cache.ComputeWorldBound(prim).GetRange()
                if rng.IsEmpty():
                    continue
                bmin, bmax = rng.GetMin(), rng.GetMax()
            except Exception as exc:
                _debug_print(env, f"[WallBoxCollision] Warning bbox failed: {source_path}: {exc}", flush=True)
                continue

            raw_bmin = [float(bmin[axis]) for axis in range(3)]
            raw_bmax = [float(bmax[axis]) for axis in range(3)]
            wall_scene_bmin = [min(wall_scene_bmin[axis], raw_bmin[axis]) for axis in range(3)]
            wall_scene_bmax = [max(wall_scene_bmax[axis], raw_bmax[axis]) for axis in range(3)]
            raw_extent = [raw_bmax[axis] - raw_bmin[axis] for axis in range(3)]
            thin_axis = min(range(3), key=lambda axis: raw_extent[axis])
            records.append(
                {
                    "source_paths": [source_path],
                    "bmin": raw_bmin,
                    "bmax": raw_bmax,
                    "thin_center": (raw_bmin[thin_axis] + raw_bmax[thin_axis]) * 0.5,
                    "thin_axis": thin_axis,
                    "merged_faces": 1,
                }
            )
            if _set_collision_enabled(prim, False, "wall mesh replaced by box primitive"):
                wall_mesh_replaced_collision_paths.add(source_path)
                disabled += 1

        def _interval_overlap_ratio(a: dict, b: dict, axis: int) -> float:
            overlap = min(a["bmax"][axis], b["bmax"][axis]) - max(a["bmin"][axis], b["bmin"][axis])
            if overlap <= 0.0:
                return 0.0
            a_len = max(a["bmax"][axis] - a["bmin"][axis], 1e-6)
            b_len = max(b["bmax"][axis] - b["bmin"][axis], 1e-6)
            return overlap / min(a_len, b_len)

        def _wall_faces_mergeable(a: dict, b: dict) -> bool:
            if a["thin_axis"] != b["thin_axis"]:
                return False
            thin_axis = a["thin_axis"]
            center_a = (a["bmin"][thin_axis] + a["bmax"][thin_axis]) * 0.5
            center_b = (b["bmin"][thin_axis] + b["bmax"][thin_axis]) * 0.5
            if abs(center_a - center_b) > WALL_FACE_MERGE_GAP_M:
                return False
            for axis in range(3):
                if axis == thin_axis:
                    continue
                if _interval_overlap_ratio(a, b, axis) < WALL_FACE_MERGE_OVERLAP_RATIO:
                    return False
            return True

        merged_records: list[dict] = []
        merged_faces = 0
        for record in records:
            match = None
            for existing in merged_records:
                if _wall_faces_mergeable(existing, record):
                    match = existing
                    break
            if match is None:
                merged_records.append(record)
                continue
            match["source_paths"].extend(record["source_paths"])
            match["bmin"] = [min(match["bmin"][axis], record["bmin"][axis]) for axis in range(3)]
            match["bmax"] = [max(match["bmax"][axis], record["bmax"][axis]) for axis in range(3)]
            match["merged_faces"] += record["merged_faces"]
            merged_faces += 1

        wall_scene_center = [
            (wall_scene_bmin[axis] + wall_scene_bmax[axis]) * 0.5
            if wall_scene_bmin[axis] != float("inf")
            else 0.0
            for axis in range(3)
        ]
        outward_expanded = 0
        for index, record in enumerate(merged_records):
            bmin = list(record["bmin"])
            bmax = list(record["bmax"])
            thin_axis = record["thin_axis"]
            thin_extent = bmax[thin_axis] - bmin[thin_axis]
            if thin_extent < WALL_BOX_OUTWARD_EXPAND_M:
                outward_sign = 1.0 if record["thin_center"] >= wall_scene_center[thin_axis] else -1.0
                if outward_sign >= 0.0:
                    bmax[thin_axis] = bmin[thin_axis] + WALL_BOX_OUTWARD_EXPAND_M
                else:
                    bmin[thin_axis] = bmax[thin_axis] - WALL_BOX_OUTWARD_EXPAND_M
                outward_expanded += 1
            center = Gf.Vec3d(
                (bmin[0] + bmax[0]) * 0.5,
                (bmin[1] + bmax[1]) * 0.5,
                (bmin[2] + bmax[2]) * 0.5,
            )
            extent = Gf.Vec3f(
                max(bmax[0] - bmin[0], WALL_BOX_THICKNESS_M),
                max(bmax[1] - bmin[1], WALL_BOX_THICKNESS_M),
                max(bmax[2] - bmin[2], WALL_BOX_THICKNESS_M),
            )

            box_path = f"{wall_box_root}/wall_box_{index:03d}"
            cube = UsdGeom.Cube.Define(stage, box_path)
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(center)
            cube.AddScaleOp().Set(extent)
            cube.GetVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if collision_debug_visuals_enabled else UsdGeom.Tokens.invisible
            )
            _style_debug_gprim(cube, (1.0, 0.0, 0.0), 0.55, set_visibility=False)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True).Set(True)
            wall_box_collision_paths.add(box_path)
            created += 1

        _debug_print(
            env,
            f"[WallBoxCollision] Replaced wall mesh collisions with box primitives: "
            f"created={created}, disabled_original={disabled}, scanned={scanned}, "
            f"merged_parallel_faces={merged_faces}, outward_expanded={outward_expanded}, "
            f"outward_expand={WALL_BOX_OUTWARD_EXPAND_M:.3f}m, min_thickness={WALL_BOX_THICKNESS_M:.3f}m.",
            flush=True,
        )
        return scanned, created, disabled

    def _style_collision_box(cube) -> None:
        cube.GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if collision_debug_visuals_enabled else UsdGeom.Tokens.invisible
        )
        _style_debug_gprim(cube, (1.0, 0.0, 0.0), 0.55, set_visibility=False)

    def _partition_record_from_bbox(source_path: str, bmin, bmax) -> dict:
        center = [
            (float(bmin[axis]) + float(bmax[axis])) * 0.5
            for axis in range(3)
        ]
        extent = [
            max(float(bmax[axis]) - float(bmin[axis]), PARTITION_BOX_THICKNESS_M)
            for axis in range(3)
        ]
        return {
            "source_paths": [source_path],
            "bmin": [center[axis] - extent[axis] * 0.5 for axis in range(3)],
            "bmax": [center[axis] + extent[axis] * 0.5 for axis in range(3)],
        }

    def _partition_boxes_nearly_overlap(a: dict, b: dict) -> bool:
        return all(
            abs(a["bmin"][axis] - b["bmin"][axis]) <= PARTITION_OVERLAP_EPS_M
            and abs(a["bmax"][axis] - b["bmax"][axis]) <= PARTITION_OVERLAP_EPS_M
            for axis in range(3)
        )

    def _merge_nearly_overlapping_partition_boxes(records: list[dict]) -> tuple[list[dict], int]:
        merged: list[dict] = []
        merge_count = 0
        for record in records:
            matched_record = None
            for existing in merged:
                if _partition_boxes_nearly_overlap(existing, record):
                    matched_record = existing
                    break
            if matched_record is None:
                merged.append(record)
                continue
            matched_record["source_paths"].extend(record["source_paths"])
            matched_record["bmin"] = [
                min(matched_record["bmin"][axis], record["bmin"][axis])
                for axis in range(3)
            ]
            matched_record["bmax"] = [
                max(matched_record["bmax"][axis], record["bmax"][axis])
                for axis in range(3)
            ]
            merge_count += 1
        return merged, merge_count

    def _replace_partition_mesh_collisions_with_boxes() -> tuple[int, int, int, int]:
        """Replace partition mesh collisions with boxes, merging only near-overlapping duplicates."""
        partition_root = f"{room_root}/Meshes/Assets"
        partition_box_root = f"{room_root}/CollisionFixes/PartitionBoxes"
        assets_prim = stage.GetPrimAtPath(partition_root)
        if not assets_prim.IsValid():
            _debug_print(env, f"[PartitionBoxCollision] Assets root not found: {partition_root}", flush=True)
            return 0, 0, 0, 0

        UsdGeom.Xform.Define(stage, f"{room_root}/CollisionFixes")
        UsdGeom.Xform.Define(stage, partition_box_root)

        scanned = 0
        disabled = 0
        records: list[dict] = []
        for prim in list(Usd.PrimRange(assets_prim, Usd.TraverseInstanceProxies())):
            path_str = str(prim.GetPath())
            if "/Partition_" not in path_str or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            scanned += 1
            try:
                rng = cache.ComputeWorldBound(prim).GetRange()
                if rng.IsEmpty():
                    continue
                bmin, bmax = rng.GetMin(), rng.GetMax()
            except Exception as exc:
                _debug_print(env, f"[PartitionBoxCollision] Warning bbox failed: {path_str}: {exc}", flush=True)
                continue

            records.append(_partition_record_from_bbox(path_str, bmin, bmax))
            if _set_collision_enabled(prim, False, "partition mesh replaced by box primitive"):
                partition_mesh_replaced_collision_paths.add(path_str)
                disabled += 1

        merged_records, merged = _merge_nearly_overlapping_partition_boxes(records)
        created = 0
        for index, record in enumerate(merged_records):
            bmin = record["bmin"]
            bmax = record["bmax"]
            center = Gf.Vec3d(
                (bmin[0] + bmax[0]) * 0.5,
                (bmin[1] + bmax[1]) * 0.5,
                (bmin[2] + bmax[2]) * 0.5,
            )
            extent = Gf.Vec3f(
                max(bmax[0] - bmin[0], PARTITION_BOX_THICKNESS_M),
                max(bmax[1] - bmin[1], PARTITION_BOX_THICKNESS_M),
                max(bmax[2] - bmin[2], PARTITION_BOX_THICKNESS_M),
            )
            box_path = f"{partition_box_root}/partition_box_{index:03d}"
            cube = UsdGeom.Cube.Define(stage, box_path)
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(center)
            cube.AddScaleOp().Set(extent)
            _style_collision_box(cube)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True).Set(True)
            partition_box_collision_paths.add(box_path)
            created += 1

        _debug_print(
            env,
            f"[PartitionBoxCollision] Replaced partition mesh collisions with box primitives: "
            f"created={created}, disabled_original={disabled}, scanned={scanned}, "
            f"near_overlap_merged={merged}, min_thickness={PARTITION_BOX_THICKNESS_M:.3f}m.",
            flush=True,
        )
        return scanned, created, disabled, merged

    def _make_robot_collision_source_visible(prim, color: tuple[float, float, float], opacity: float) -> int:
        """Render the original robot collision prims in-place so they follow articulation motion."""
        if _is_disabled_robot_base_collision(prim):
            _set_collision_enabled(prim, False, "disabled chassis collision")
            return 0

        visible_count = 0
        try:
            if prim.IsInstance():
                prim.SetInstanceable(False)
        except Exception as exc:
            _debug_print(env, f"[CollisionDebug] Warning robot collision deinstance: {prim.GetPath()}: {exc}", flush=True)

        _set_collision_enabled(prim, True, "robot collision visualization")

        try:
            ancestor = prim
            while ancestor.IsValid() and str(ancestor.GetPath()).startswith(robot_root):
                if ancestor.IsA(UsdGeom.Imageable):
                    imageable = UsdGeom.Imageable(ancestor)
                    visibility_attr = imageable.GetVisibilityAttr()
                    if visibility_attr and visibility_attr.IsValid() and visibility_attr.Get() == UsdGeom.Tokens.invisible:
                        imageable.CreateVisibilityAttr(UsdGeom.Tokens.inherited)
                ancestor = ancestor.GetParent()
        except Exception as exc:
            _debug_print(env, f"[CollisionDebug] Warning robot ancestor visibility: {prim.GetPath()}: {exc}", flush=True)

        render_prims = []
        if prim.IsA(UsdGeom.Gprim):
            render_prims.append(prim)
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if child != prim and child.IsA(UsdGeom.Gprim):
                render_prims.append(child)

        for render_prim in render_prims:
            try:
                if render_prim.IsInstance():
                    render_prim.SetInstanceable(False)
                gprim = UsdGeom.Gprim(render_prim)
                _style_debug_gprim(gprim, color, opacity)
                if render_prim.IsA(UsdGeom.Mesh):
                    UsdGeom.Mesh(render_prim).CreateDoubleSidedAttr(True)
                visible_count += 1
            except Exception as exc:
                _debug_print(env, f"[CollisionDebug] Warning robot source visual style: {render_prim.GetPath()}: {exc}", flush=True)

        return visible_count

    def _hide_robot_collision_source(prim) -> int:
        """Hide original robot collision render prims without changing collision state."""
        hidden_count = 0
        render_prims = []
        if prim.IsA(UsdGeom.Gprim):
            render_prims.append(prim)
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if child != prim and child.IsA(UsdGeom.Gprim):
                render_prims.append(child)

        for render_prim in render_prims:
            try:
                if render_prim.IsInstance():
                    render_prim.SetInstanceable(False)
                UsdGeom.Imageable(render_prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                hidden_count += 1
            except Exception as exc:
                _debug_print(env, f"[CollisionDebug] Warning robot source hide: {render_prim.GetPath()}: {exc}", flush=True)
        return hidden_count

    def _visualize_robot_collisions() -> tuple[int, int]:
        robot_prim = stage.GetPrimAtPath(robot_root)
        if not robot_prim.IsValid():
            _debug_print(env, f"[CollisionDebug] Robot root not found for collision visualization: {robot_root}", flush=True)
            return 0, 0

        if not collision_debug_visuals_enabled:
            stale_robot_debug_root = f"{debug_root}/robot"
            if stage.GetPrimAtPath(stale_robot_debug_root).IsValid():
                stage.RemovePrim(stale_robot_debug_root)
                _debug_print(env, f"[CollisionDebug] Removed stale robot debug copies: {stale_robot_debug_root}", flush=True)
            stale_attached_debug = [
                str(prim.GetPath())
                for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies())
                if prim.GetName().startswith("collision_debug__")
            ]
            for stale_path in stale_attached_debug:
                stage.RemovePrim(stale_path)
                _debug_print(env, f"[CollisionDebug] Removed stale attached robot debug copy: {stale_path}", flush=True)
            total = 0
            hidden = 0
            for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    total += 1
                    hidden += _hide_robot_collision_source(prim)
            _debug_print(
                env,
                f"[CollisionDebug] Robot collision visualization skipped: "
                f"{total} USD CollisionAPI prims under {robot_root}; hidden_source_prims={hidden}.",
                flush=True,
            )
            return total, 0

        stale_robot_debug_root = f"{debug_root}/robot"
        if stage.GetPrimAtPath(stale_robot_debug_root).IsValid():
            stage.RemovePrim(stale_robot_debug_root)
            _debug_print(env, f"[CollisionDebug] Removed stale robot debug copies: {stale_robot_debug_root}", flush=True)

        stale_attached_debug = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies())
            if prim.GetName().startswith("collision_debug__")
        ]
        for stale_path in stale_attached_debug:
            stage.RemovePrim(stale_path)
            _debug_print(env, f"[CollisionDebug] Removed stale attached robot debug copy: {stale_path}", flush=True)

        robot_collision_color = (0.10, 0.46, 1.00)
        total = 0
        source_visible = 0
        for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                total += 1
                source_visible += _make_robot_collision_source_visible(
                    prim,
                    robot_collision_color,
                    0.62,
                )
        _debug_print(
            env,
            f"[CollisionDebug] Robot collision visualization complete: "
            f"{source_visible} source prims styled from {total} USD CollisionAPI prims under {robot_root}.",
            flush=True,
        )
        return total, source_visible

    def _aabb_overlaps(bmin, bmax, x_range, y_range, z_range=None) -> bool:
        if float(bmax[0]) < x_range[0] or float(bmin[0]) > x_range[1]:
            return False
        if float(bmax[1]) < y_range[0] or float(bmin[1]) > y_range[1]:
            return False
        if z_range is not None and (float(bmax[2]) < z_range[0] or float(bmin[2]) > z_range[1]):
            return False
        return True

    def _define_clean_floor_collider() -> bool:
        collider_path = f"{room_root}/CollisionFixes/dirty_stall_clean_floor"
        if stage.GetPrimAtPath(collider_path).IsValid():
            return False
        UsdGeom.Xform.Define(stage, f"{room_root}/CollisionFixes")
        cube = UsdGeom.Cube.Define(stage, collider_path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(1.4705, 0.0275, -0.03))
        cube.AddScaleOp().Set(Gf.Vec3f(1.9535, 3.145, 0.06))
        cube.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True).Set(True)
        if collision_debug_visuals_enabled:
            _debug_cube_from_bbox(
                f"{debug_root}/dirty_stall_clean_floor_ACTIVE",
                Gf.Vec3d(0.494, -1.545, -0.060),
                Gf.Vec3d(2.447, 1.600, 0.000),
                _color_from_path(f"{room_root}/CollisionFixes/dirty_stall_clean_floor", "active clean floor"),
                0.30,
            )
        _debug_print(
            env,
            "[StallFix] Added clean floor collider: "
            "x=[0.494,2.447] y=[-1.545,1.600] z=[-0.060,0.000]",
            flush=True,
        )
        return True

    # 先打开机器人和房间场景的物理碰撞；后续只关闭已确认异常的局部 scene mesh。
    robot_collision_enable_total, robot_collision_enabled = _enable_collision_subtree(robot_root, "robot")
    room_collision_enable_total, room_collision_enabled = _enable_collision_subtree(room_root, "room")
    _set_robot_arm_gravity_enabled(env, "startup")
    if _disable_robot_base_collision_requested():
        base_collision_prim = stage.GetPrimAtPath(f"{robot_root}/base_link/collisions")
        if _set_collision_enabled(base_collision_prim, False, "disabled chassis collision"):
            print(
                f"[CollisionEnable] Disabled chassis collision while keeping arm collision enabled: "
                f"{base_collision_prim.GetPath()}",
                flush=True,
            )
    _enforce_robot_collision_policy("startup final robot collision policy")

    # ── 1. 遍历全场，停用入口前沿薄板 ────────────────────────────────────────
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if not path_str.startswith(room_root):
            continue
        if "Floor_0000" in path_str:
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        try:
            bbox = cache.ComputeWorldBound(prim)
            rng  = bbox.GetRange()
            if rng.IsEmpty():
                continue
            bmin, bmax = rng.GetMin(), rng.GetMax()
            cy       = (bmin[1] + bmax[1]) / 2
            y_extent = float(bmax[1]) - float(bmin[1])

            if not (Y_CENTER_LO < cy < Y_CENTER_HI):
                continue
            if y_extent >= Y_EXTENT_MAX:
                continue   # 贯穿侧墙，保留

            name = path_str.rsplit("/", 1)[-1]
            h    = float(bmax[2]) - float(bmin[2])
            prim.SetActive(False)
            deactivated += 1
            print(
                f"[StallFix] Deactivated '{name}'  "
                f"y=[{bmin[1]:.3f},{bmax[1]:.3f}]  h={h:.2f}m",
                flush=True,
            )
        except Exception as e:
            _debug_print(env, f"[StallFix] Warning: {path_str.rsplit('/',1)[-1]}: {e}", flush=True)

    # ── 2. 将剩余零厚度墙体 mesh 碰撞替换成有厚度的 box primitive ────────────
    wall_collision_scanned, wall_box_created, wall_mesh_disabled = _replace_wall_mesh_collisions_with_boxes()
    collision_disabled += wall_mesh_disabled

    # ── 3. 将卫生间隔板 mesh 碰撞替换成 box primitive，仅去重重叠 box ────────
    partition_collision_scanned, partition_box_created, partition_mesh_disabled, partition_box_merged = (
        _replace_partition_mesh_collisions_with_boxes()
    )
    collision_disabled += partition_mesh_disabled

    # ── 4. 按已知路径关闭隔间门（含内嵌 payload 碰撞）────────────────────────
    door_paths = [
        f"{room_root}/Meshes/Assets/Door_YS3UEUZVAJ2V4PTUKU888888_mdynP",
        f"{room_root}/Meshes/Assets/Door_YS2HVXZVALGD2PTUJQ888888_7qcYe",
        f"{room_root}/Meshes/Assets/Door_0000",
    ]
    for dp in door_paths:
        prim = stage.GetPrimAtPath(dp)
        if prim.IsValid():
            _visualize_collision_subtree(prim, "deactivated door collision", opacity=0.35)
            prim.SetActive(False)
            deactivated += 1
            _debug_print(env, f"[StallFix] Deactivated door: {dp.rsplit('/',1)[-1]}", flush=True)
        else:
            _debug_print(env, f"[StallFix] Door not found (skipped): {dp.rsplit('/',1)[-1]}", flush=True)

    # ── 5. 修复 3 个黄色 dirty 所在隔间的地板/入口接缝碰撞 ─────────────────────
    # 这些 Floor_0000 SDF 薄片在 dirty 隔间和入口处互相重叠/零厚度，容易造成卡顿。
    dirty_stall_vertical_entrance_names = {
        "A124",  # 入口竖向薄片
        "A125",  # 入口竖向薄片
    }
    for name in sorted(dirty_stall_vertical_entrance_names):
        prim = stage.GetPrimAtPath(f"{room_root}/Meshes/Base/Floor_0000/Geom/{name}")
        if prim.IsValid():
            prim.SetActive(False)
            deactivated += 1
            _debug_print(env, f"[StallFix] Deactivated dirty-stall entrance vertical mesh: {name}", flush=True)

    dirty_stall_floor_names = {
        "A67",   # dirty 隔间主地板薄片
        "A70",   # 入口地板接缝
        "A123",  # 入口地板接缝
        "A150",  # 入口重复薄片
        "A164",  # 入口重复薄片
        "A240",  # 入口重复薄片
        "A262",  # 覆盖 dirty 隔间/通道的大块零厚度地板
    }
    for name in sorted(dirty_stall_floor_names):
        prim = stage.GetPrimAtPath(f"{room_root}/Meshes/Base/Floor_0000/Geom/{name}")
        if _disable_collision(prim, "dirty-stall floor/seam"):
            collision_disabled += 1
    if _define_clean_floor_collider():
        collision_disabled += 1

    # ── 6. 关闭该区域高处天花板/灯具/风口碰撞 ────────────────────────────────
    ceiling_x = (0.45, 2.50)
    ceiling_y = (-1.58, 0.85)
    ceiling_z = (2.20, 3.20)
    for prim in list(stage.Traverse()):
        path_str = str(prim.GetPath())
        if not path_str.startswith(room_root) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if path_str in wall_box_collision_paths or path_str in partition_box_collision_paths:
            continue
        try:
            bbox = cache.ComputeWorldBound(prim)
            rng = bbox.GetRange()
            if rng.IsEmpty():
                continue
            bmin, bmax = rng.GetMin(), rng.GetMax()
            if _aabb_overlaps(bmin, bmax, ceiling_x, ceiling_y, ceiling_z):
                if _disable_collision(prim, "dirty-stall ceiling/fixture"):
                    collision_disabled += 1
        except Exception as e:
            _debug_print(env, f"[StallFix] Warning ceiling scan: {path_str.rsplit('/',1)[-1]}: {e}", flush=True)

    # ── 7. 可选：补齐所有非机器人场景碰撞 mesh 的可视化 ───────────────────────
    scene_collision_total, scene_collision_created = _visualize_all_scene_collisions()

    # ── 8. 可选：补齐机器人自身碰撞 mesh 的可视化 ───────────────────────────
    robot_collision_total, robot_collision_created = _visualize_robot_collisions()
    _enforce_robot_collision_policy("post robot collision setup")

    _debug_print(
        env,
        f"[StallFix] Done — deactivated {deactivated} entrance prims, "
        f"disabled/added {collision_disabled} collision fixes, "
        f"wall_boxes={wall_box_created}/{wall_collision_scanned}, "
        f"partition_boxes={partition_box_created}/{partition_collision_scanned} "
        f"(near_overlap_merged={partition_box_merged}), "
        f"enabled collisions robot={robot_collision_enabled}/{robot_collision_enable_total}, "
        f"room={room_collision_enabled}/{room_collision_enable_total}, "
        f"created {debug_visual_count} collision debug visuals "
        f"({scene_collision_created}/{scene_collision_total} scene, "
        f"{robot_collision_created}/{robot_collision_total} robot).",
        flush=True,
    )

def _visualize_robot_collisions_on_reset(env, _):
    """Maintain reset-time robot collision policy and optionally render collision prims."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = env.sim.stage
    robot_root = "/World/envs/env_0/robot"
    stale_robot_debug_root = "/World/collision_mesh_debug/env_0/robot"
    disable_robot_base_collision = bool(getattr(env.cfg, "disable_robot_base_collision", False))
    collision_debug_visuals_enabled = bool(getattr(env.cfg, "enable_collision_debug_visuals", True))

    def _style_debug_gprim(gprim, color: tuple[float, float, float], opacity: float) -> None:
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        gprim.CreateDisplayOpacityAttr([opacity])
        gprim.CreatePurposeAttr(UsdGeom.Tokens.default_)
        gprim.CreateVisibilityAttr(UsdGeom.Tokens.inherited)

    def _color_from_path(path_str: str) -> tuple[float, float, float]:
        value = 2166136261
        for ch in path_str:
            value ^= ord(ch)
            value = (value * 16777619) & 0xFFFFFFFF
        hue = (value % 360) / 360.0
        sat = 0.74
        val = 0.92
        sector = int(hue * 6.0)
        frac = hue * 6.0 - sector
        p = val * (1.0 - sat)
        q = val * (1.0 - sat * frac)
        t = val * (1.0 - sat * (1.0 - frac))
        sector %= 6
        if sector == 0:
            return (val, t, p)
        if sector == 1:
            return (q, val, p)
        if sector == 2:
            return (p, val, t)
        if sector == 3:
            return (p, q, val)
        if sector == 4:
            return (t, p, val)
        return (val, p, q)

    def _make_source_visible(prim, color: tuple[float, float, float], opacity: float) -> int:
        if disable_robot_base_collision and str(prim.GetPath()) == f"{robot_root}/base_link/collisions":
            try:
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False).Set(False)
                _debug_print(env, f"[CollisionEnable][Reset] Kept chassis collision disabled: {prim.GetPath()}", flush=True)
            except Exception as exc:
                print(
                    f"[CollisionEnable][Reset] Warning chassis collision disable: {prim.GetPath()}: {exc}",
                    flush=True,
                )
            return 0

        count = 0
        try:
            if prim.IsInstance():
                prim.SetInstanceable(False)
        except Exception as exc:
            _debug_print(env, f"[CollisionDebug][Reset] Warning robot collision deinstance: {prim.GetPath()}: {exc}", flush=True)

        try:
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True).Set(True)
        except Exception as exc:
            print(
                f"[CollisionEnable][Reset] Warning robot collision enable: {prim.GetPath()}: {exc}",
                flush=True,
            )

        try:
            ancestor = prim
            while ancestor.IsValid() and str(ancestor.GetPath()).startswith(robot_root):
                if ancestor.IsA(UsdGeom.Imageable):
                    imageable = UsdGeom.Imageable(ancestor)
                    visibility_attr = imageable.GetVisibilityAttr()
                    if visibility_attr and visibility_attr.IsValid() and visibility_attr.Get() == UsdGeom.Tokens.invisible:
                        imageable.CreateVisibilityAttr(UsdGeom.Tokens.inherited)
                ancestor = ancestor.GetParent()
        except Exception as exc:
            _debug_print(env, f"[CollisionDebug][Reset] Warning robot ancestor visibility: {prim.GetPath()}: {exc}", flush=True)

        render_prims = []
        if prim.IsA(UsdGeom.Gprim):
            render_prims.append(prim)
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if child != prim and child.IsA(UsdGeom.Gprim):
                render_prims.append(child)
        for render_prim in render_prims:
            try:
                if render_prim.IsInstance():
                    render_prim.SetInstanceable(False)
                gprim = UsdGeom.Gprim(render_prim)
                _style_debug_gprim(gprim, color, opacity)
                if render_prim.IsA(UsdGeom.Mesh):
                    UsdGeom.Mesh(render_prim).CreateDoubleSidedAttr(True)
                count += 1
            except Exception as exc:
                print(
                    f"[CollisionDebug][Reset] Warning robot source visual style: {render_prim.GetPath()}: {exc}",
                    flush=True,
                )
        return count

    def _hide_source(prim) -> int:
        count = 0
        render_prims = []
        if prim.IsA(UsdGeom.Gprim):
            render_prims.append(prim)
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if child != prim and child.IsA(UsdGeom.Gprim):
                render_prims.append(child)
        for render_prim in render_prims:
            try:
                if render_prim.IsInstance():
                    render_prim.SetInstanceable(False)
                UsdGeom.Imageable(render_prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                count += 1
            except Exception as exc:
                print(
                    f"[CollisionDebug][Reset] Warning robot source hide: {render_prim.GetPath()}: {exc}",
                    flush=True,
                )
        return count

    def _enforce_robot_collision_policy() -> None:
        total = 0
        base_disabled = 0
        non_base_enabled = 0
        for prim in list(Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies())):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            total += 1
            try:
                if disable_robot_base_collision and str(prim.GetPath()) == f"{robot_root}/base_link/collisions":
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False).Set(False)
                    base_disabled += 1
                else:
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(True).Set(True)
                    non_base_enabled += 1
            except Exception as exc:
                _debug_print(env, f"[CollisionPolicy][Reset] Warning robot collision policy: {prim.GetPath()}: {exc}", flush=True)

        _debug_print(
            env,
            f"[CollisionPolicy][Reset] base_collision_disabled={base_disabled}, "
            f"non_base_robot_collision_enabled={non_base_enabled}, total_robot_collision_prims={total}.",
            flush=True,
        )

    robot_prim = stage.GetPrimAtPath(robot_root)
    if not robot_prim.IsValid():
        _debug_print(env, f"[CollisionDebug][Reset] Robot root not found: {robot_root}", flush=True)
        return

    if stage.GetPrimAtPath(stale_robot_debug_root).IsValid():
        stage.RemovePrim(stale_robot_debug_root)
        _debug_print(env, f"[CollisionDebug][Reset] Removed stale robot debug copies: {stale_robot_debug_root}", flush=True)

    stale_attached_debug = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies())
        if prim.GetName().startswith("collision_debug__")
    ]
    for stale_path in stale_attached_debug:
        stage.RemovePrim(stale_path)
        _debug_print(env, f"[CollisionDebug][Reset] Removed stale attached robot debug copy: {stale_path}", flush=True)

    robot_collision_color = (0.10, 0.46, 1.00)
    total = 0
    source_visible = 0
    source_hidden = 0
    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            total += 1
            if collision_debug_visuals_enabled:
                source_visible += _make_source_visible(prim, robot_collision_color, 0.62)
            else:
                source_hidden += _hide_source(prim)

    if collision_debug_visuals_enabled:
        _debug_print(
            env,
            f"[CollisionDebug][Reset] Robot collision visualization complete: "
            f"{source_visible} source prims styled from {total} USD CollisionAPI prims.",
            flush=True,
        )
    else:
        _debug_print(
            env,
            f"[CollisionDebug][Reset] Robot collision visualization skipped: "
            f"{total} USD CollisionAPI prims; hidden_source_prims={source_hidden}.",
            flush=True,
        )
    _enforce_robot_collision_policy()
    _set_robot_arm_gravity_enabled(env, "reset")


@configclass
class DirtyEventsCfg:
    """Startup event: remove low-height collision blockers at stall entrance."""

    fix_stall_entrance = EventTerm(
        func=_fix_stall_entrance,
        mode="prestartup",
    )

    visualize_robot_collisions = EventTerm(
        func=_visualize_robot_collisions_on_reset,
        mode="reset",
    )

# 污垢道具 USD 根目录：data/scenes/rx/
_RX = f"{ALLSTAR_ASSETS_DATA_DIR}/scenes/rx"

_VISUAL_ONLY_DIRT_SPAWN = dict(
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=False),
)


@configclass
class Mecanum730XMS5GripperDirtySceneCfg(Mecanum730XMS5GripperSceneCfg):
    """841837 real restroom with Mecanum 0730 + XMS5 Gripper and 10 scattered dirt/garbage props.

    在 Mecanum730XMS5GripperSceneCfg（机器人 + 相机 + 基础场景）基础上，
    叠加 10 种真实卫生间污垢道具，用于训练清洁任务策略。

    所有道具：
      - prim_path 以 {ENV_REGEX_NS}/ 开头，支持多环境并行实例化
      - pos 的 z=0.01 使道具贴地（避免穿透地面）
      - 道具本身是静态 AssetBaseCfg（无关节、无物理驱动），仅作为视觉/碰撞目标
    """

    # 烟蒂：位于地面右前方
    dirt_cigarette = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_cigarette",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.3, 0.01)),
        spawn=UsdFileCfg(usd_path=f"{_RX}/cigarette/cigarette.usd", **_VISUAL_ONLY_DIRT_SPAWN),
    )

    # 烟灰：位于地面右侧
    dirt_ash = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_ash",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, -0.4, 0.01)),
        spawn=UsdFileCfg(usd_path=f"{_RX}/Cigarette ash/Cigaretteash.usd", **_VISUAL_ONLY_DIRT_SPAWN),
    )

    # 脚印：位于地面左前方
    dirt_footprint = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_footprint",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.3, 0.6, 0.01)),
        spawn=UsdFileCfg(usd_path=f"{_RX}/footprint/footprint.usd", **_VISUAL_ONLY_DIRT_SPAWN),
    )

    # 塑料袋：位于地面远前方
    dirt_plastic_bag = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_plastic_bag",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.2, 0.5, 0.01)),
        spawn=UsdFileCfg(usd_path=f"{_RX}/plastic bag/plastic bag.usd", **_VISUAL_ONLY_DIRT_SPAWN),
    )

    # 卷纸：位于地面左后方
    dirt_roll_paper = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_roll_paper",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.8, -0.3, 0.01)),
        spawn=UsdFileCfg(usd_path=f"{_RX}/roll paper/rollpaper .usd", **_VISUAL_ONLY_DIRT_SPAWN),
    )

    # 座便纸：位于地面右后方
    dirt_seat_paper = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_seat_paper",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, -0.8, 0.01)),
        spawn=UsdFileCfg(usd_path=f"{_RX}/seat paper/seatpaper.usd", **_VISUAL_ONLY_DIRT_SPAWN),
    )

    # 卫生巾：位于地面左侧远处
    dirt_aunt_scarf = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_aunt_scarf",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.1, 0.4, 0.01)),
        spawn=UsdFileCfg(usd_path=f"{_RX}/Aunt scarf/Auntscarf.usd", **_VISUAL_ONLY_DIRT_SPAWN),
    )

    # 粪便污迹：位于地面右侧远处
    dirt_shitstains = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_shitstains",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.5, -0.6, 0.01)),
        spawn=UsdFileCfg(usd_path=f"{_RX}/Shitstains/Shitstains.usd", **_VISUAL_ONLY_DIRT_SPAWN),
    )

    # 尿液污迹：位于地面左后侧
    dirt_urine = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_urine",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5, -1.0, 0.01)),
        spawn=UsdFileCfg(usd_path=f"{_RX}/urinarystains/urinarystains.usd", **_VISUAL_ONLY_DIRT_SPAWN),
    )

    # 水渍：位于地面正前方
    dirt_water = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/dirt_water",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 1.2, 0.01)),
        spawn=UsdFileCfg(usd_path=f"{_RX}/Waterstain/Waterstain.usd", **_VISUAL_ONLY_DIRT_SPAWN),
    )


@configclass
class Mecanum730XMS5GripperDirtyEnvCfg(Mecanum730XMS5GripperEnvCfg):
    """Mecanum 0730 + XMS5 Gripper mobile manipulator in the dirty Shenxinfu 841837 real scene.

    相较于 Mecanum730XMS5GripperEnvCfg：
      - scene 替换为 Mecanum730XMS5GripperDirtySceneCfg（叠加 10 处污垢道具）
      - events 新增 add_stall_ramp（startup）：自动在隔间入口添加斜坡，填平走廊高度差
    """

    # 替换为脏污场景（在基础场景上叠加 10 个污垢 prim）
    scene: Mecanum730XMS5GripperDirtySceneCfg = Mecanum730XMS5GripperDirtySceneCfg(
        num_envs=1, env_spacing=2.5
    )
    enable_collision_debug_visuals = True
    events: DirtyEventsCfg = DirtyEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Dirty scene collision patches run in prestartup, before PhysX parses the stage.
        # IsaacLab requires non-replicated scene physics for USD-level prestartup edits.
        self.scene.replicate_physics = False
