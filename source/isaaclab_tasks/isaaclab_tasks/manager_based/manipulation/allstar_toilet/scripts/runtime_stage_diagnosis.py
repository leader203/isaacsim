#!/usr/bin/env python3
"""Runtime Stage diagnosis — runs inside Isaac Sim environment.
Usage: ./isaaclab.sh -p scripts/runtime_stage_diagnosis.py
Output: docs/runtime-stage-diagnosis.txt
"""
import sys, os

def main():
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf

    stage = omni.usd.get_context().get_stage()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "docs", "runtime-stage-diagnosis.txt"
    )
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        def p(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=f)

        p("=" * 80)
        p("RUNTIME STAGE DIAGNOSIS")
        p("=" * 80)
        p(f"Stage root: {stage.GetRootLayer().identifier}")

        ROOM = "/World/envs/env_0/room"

        # ── 1. Check room exists ──
        room_prim = stage.GetPrimAtPath(ROOM)
        p(f"\nRoom prim valid: {room_prim.IsValid()}")
        if not room_prim.IsValid():
            p("FATAL: room not found — checking /World/envs/env_0")
            env0 = stage.GetPrimAtPath("/World/envs/env_0")
            p(f"  env_0 valid: {env0.IsValid()}")
            if env0.IsValid():
                for child in env0.GetChildren():
                    p(f"  child: {child.GetName()} ({child.GetTypeName()})")
            return

        # ── 2. Check robot ──
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/robot")
        p(f"\nRobot prim valid: {robot_prim.IsValid()}")

        # ── 3. Inspect subtrees ──
        ROOTS = [
            ROOM,
            f"{ROOM}/Meshes",
            f"{ROOM}/Meshes/Base",
            f"{ROOM}/Meshes/Base/Wall_0000",
            f"{ROOM}/Meshes/Assets",
            f"{ROOM}/Meshes/Assets/Wall_Wall",
            f"{ROOM}/CollisionFixes",
            f"{ROOM}/CollisionFixes/CuroboOnlyWalls",
        ]

        for root_path in ROOTS:
            root = stage.GetPrimAtPath(root_path)
            p(f"\n{'─'*60}")
            p(f"PATH: {root_path}")
            p(f"  valid: {root.IsValid()}")

            if not root.IsValid():
                continue

            mesh_count = 0; xform_count = 0; scope_count = 0
            instance_count = 0; collision_count = 0; enabled_collision = 0
            mesh_collision_count = 0; total_descendants = 0

            for prim in Usd.PrimRange(root):
                total_descendants += 1
                if prim.IsA(UsdGeom.Mesh):
                    mesh_count += 1
                    if prim.IsInstanceProxy():
                        instance_count += 1
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        collision_count += 1
                        try:
                            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                            if enabled is not False:
                                enabled_collision += 1
                        except:
                            pass
                    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                        mesh_collision_count += 1
                elif prim.IsA(UsdGeom.Xformable):
                    xform_count += 1
                elif prim.IsA(UsdGeom.Scope):
                    scope_count += 1

            p(f"  total descendants: {total_descendants}")
            p(f"  Mesh: {mesh_count}  Xform: {xform_count}  Scope: {scope_count}")
            p(f"  instance proxy: {instance_count}")
            p(f"  CollisionAPI: {collision_count}  enabled: {enabled_collision}")
            p(f"  MeshCollisionAPI: {mesh_collision_count}")

        # ── 4. Wall_0000 deep inspection ──
        p(f"\n{'='*80}")
        p("WALL_0000 DEEP INSPECTION")
        wall_prim = stage.GetPrimAtPath(f"{ROOM}/Meshes/Base/Wall_0000")
        if wall_prim.IsValid():
            p(f"  type: {wall_prim.GetTypeName()}")
            p(f"  active: {wall_prim.IsActive()}")
            p(f"  loaded: {wall_prim.IsLoaded()}")
            p(f"  children:")
            wall_mesh_count = 0
            for child in wall_prim.GetChildren():
                p(f"    {child.GetName()} ({child.GetTypeName()}) active={child.IsActive()}")
            for prim in Usd.PrimRange(wall_prim):
                if prim.IsA(UsdGeom.Mesh):
                    wall_mesh_count += 1
                    apis = [str(a) for a in prim.GetAppliedSchemas()]
                    p(f"  MESH: {prim.GetPath()}")
                    p(f"    active: {prim.IsActive()}  loaded: {prim.IsLoaded()}")
                    p(f"    instance_proxy: {prim.IsInstanceProxy()}")
                    p(f"    schemas: {apis}")
                    p(f"    CollisionAPI: {prim.HasAPI(UsdPhysics.CollisionAPI)}")
                    p(f"    MeshCollisionAPI: {prim.HasAPI(UsdPhysics.MeshCollisionAPI)}")
            p(f"  total Mesh descendants: {wall_mesh_count}")

        # ── 5. All wall meshes ──
        p(f"\n{'='*80}")
        p("ALL WALL-RELATED MESHES")
        keywords = ("Wall_Wall", "Wall_0000")
        wall_meshes = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = prim.GetPath().pathString
            if not any(k in path for k in keywords):
                continue
            wall_meshes.append(path)
            has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
            enabled = None
            if has_col:
                try:
                    enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                except:
                    pass
            p(f"  {path}")
            p(f"    active={prim.IsActive()} loaded={prim.IsLoaded()}")
            p(f"    instance_proxy={prim.IsInstanceProxy()}")
            p(f"    CollisionAPI={has_col} enabled={enabled}")
            p(f"    schemas={list(prim.GetAppliedSchemas())}")
        p(f"  TOTAL: {len(wall_meshes)}")

        # ── 6. Door prim count ──
        p(f"\n{'='*80}")
        p("DOOR-RELATED PRIMS")
        for door_name in ("Door_YS2HVXZVALGD2PTUJQ888888_7qcYe", "Door_YS3UEUZVAJ2V4PTUKU888888_mdynP"):
            door_count = 0; door_mesh_count = 0
            for prim in stage.Traverse():
                path = prim.GetPath().pathString
                if door_name not in path:
                    continue
                door_count += 1
                if prim.IsA(UsdGeom.Mesh):
                    door_mesh_count += 1
                    p(f"  MESH: {path}")
                    p(f"    active={prim.IsActive()} loaded={prim.IsLoaded()}")
                    p(f"    instance_proxy={prim.IsInstanceProxy()}")
                    p(f"    CollisionAPI={prim.HasAPI(UsdPhysics.CollisionAPI)}")
                    p(f"    schemas={list(prim.GetAppliedSchemas())}")
            p(f"  {door_name}: {door_count} prims, {door_mesh_count} meshes")

        # ── 7. Wall_Wall mesh inspection ──
        p(f"\n{'='*80}")
        p("WALL_WALL ASSETS MESH INSPECTION")
        wall_assets = stage.GetPrimAtPath(f"{ROOM}/Meshes/Assets/Wall_Wall")
        if wall_assets.IsValid():
            for prim in Usd.PrimRange(wall_assets):
                if prim.IsA(UsdGeom.Mesh):
                    has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
                    enabled = None
                    if has_col:
                        try:
                            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                        except:
                            pass
                    p(f"  {prim.GetPath()}")
                    p(f"    CollisionAPI={has_col} enabled={enabled}")
                    p(f"    MeshCollisionAPI={prim.HasAPI(UsdPhysics.MeshCollisionAPI)}")

        # ── 8. Total summary ──
        p(f"\n{'='*80}")
        p("TOTAL STAGE SUMMARY")
        total_mesh = 0; total_collision = 0; total_enabled = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                total_mesh += 1
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    total_collision += 1
                    try:
                        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                            total_enabled += 1
                    except:
                        pass
        p(f"  Total Meshes: {total_mesh}")
        p(f"  Total with CollisionAPI: {total_collision}")
        p(f"  Total collision enabled: {total_enabled}")

    print(f"\nDiagnosis saved to: {out_path}")


if __name__ == "__main__":
    # Wait for stage to be ready
    import time
    time.sleep(5)
    main()
