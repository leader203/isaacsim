#!/usr/bin/env python3
"""Diagnose door/wall collision in the loaded scene — standalone, via isaaclab.sh."""
import time, os, sys

# Wait for the stage to be ready (simulation startup)
print("[Diag] Waiting for stage...", flush=True)
time.sleep(10)

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()
print(f"[Diag] Stage ready: {stage.GetRootLayer().identifier}", flush=True)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "runtime-stage-diagnosis.txt")
OUT = os.path.normpath(OUT)
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(OUT, "w") as f:
    def log(*args):
        line = " ".join(str(a) for a in args)
        print(line, flush=True)
        f.write(line + "\n")

    ROOM = "/World/envs/env_0/room"
    room = stage.GetPrimAtPath(ROOM)
    log(f"Room valid: {room.IsValid()}")
    log(f"Robot valid: {stage.GetPrimAtPath('/World/envs/env_0/robot').IsValid()}")

    # ── Inspect subtrees ──
    for rp in [ROOM, f"{ROOM}/Meshes/Base", f"{ROOM}/Meshes/Base/Wall_0000",
               f"{ROOM}/Meshes/Assets/Wall_Wall", f"{ROOM}/CollisionFixes/CuroboOnlyWalls"]:
        root = stage.GetPrimAtPath(rp)
        log(f"\n{rp} valid={root.IsValid()}")
        if not root.IsValid(): continue
        m = x = c = ec = 0
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                m += 1
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    c += 1
                    try:
                        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                            ec += 1
                    except: pass
            elif prim.IsA(UsdGeom.Xformable): x += 1
        log(f"  Mesh={m} Xform={x} CollisionAPI={c} enabled={ec}")

    # ── Wall_0000 meshes ──
    log("\n=== Wall_0000 descendants ===")
    w = stage.GetPrimAtPath(f"{ROOM}/Meshes/Base/Wall_0000")
    if w.IsValid():
        wm = 0
        for prim in Usd.PrimRange(w):
            if prim.IsA(UsdGeom.Mesh):
                wm += 1
                apis = list(prim.GetAppliedSchemas())
                log(f"  {prim.GetPath()} active={prim.IsActive()} instance_proxy={prim.IsInstanceProxy()} CollisionAPI={prim.HasAPI(UsdPhysics.CollisionAPI)}")
                log(f"    schemas={apis}")
        log(f"  Total Wall_0000 meshes: {wm}")

    # ── All wall meshes ──
    log("\n=== All Wall meshes ===")
    wall_meshes = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and ("Wall_Wall" in prim.GetPath().pathString or "Wall_0000" in prim.GetPath().pathString):
            wall_meshes.append(prim.GetPath().pathString)
            coll = prim.HasAPI(UsdPhysics.CollisionAPI)
            en = None
            if coll:
                try: en = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                except: pass
            log(f"  {prim.GetPath()} CollisionAPI={coll} enabled={en}")
    log(f"  Total: {len(wall_meshes)}")

    # ── Door prims ──
    log("\n=== Door prims ===")
    for dn in ("YS2HVXZVALGD2PTUJQ888888_7qcYe", "YS3UEUZVAJ2V4PTUKU888888_mdynP"):
        dc = dm = 0
        for prim in stage.Traverse():
            if dn in prim.GetPath().pathString:
                dc += 1
                if prim.IsA(UsdGeom.Mesh):
                    dm += 1
                    log(f"  MESH: {prim.GetPath()} active={prim.IsActive()} loaded={prim.IsLoaded()} inst_proxy={prim.IsInstanceProxy()} CollisionAPI={prim.HasAPI(UsdPhysics.CollisionAPI)}")
        log(f"  Door_{dn}: {dc} prims, {dm} meshes")

    # ── Total summary ──
    log("\n=== Total Stage ===")
    tm = tc = tec = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            tm += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                tc += 1
                try:
                    if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                        tec += 1
                except: pass
    log(f"Total Meshes={tm} CollisionAPI={tc} enabled={tec}")

    # ── Floor collision check ──
    log("\n=== Floor collision ===")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and "Floor" in prim.GetPath().pathString:
            coll = prim.HasAPI(UsdPhysics.CollisionAPI)
            en = None
            if coll:
                try: en = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                except: pass
            log(f"  {prim.GetPath()} CollisionAPI={coll} enabled={en}")

log(f"\nDone. Saved to {OUT}")

# Keep running until simulation exits or timeout
time.sleep(30)
