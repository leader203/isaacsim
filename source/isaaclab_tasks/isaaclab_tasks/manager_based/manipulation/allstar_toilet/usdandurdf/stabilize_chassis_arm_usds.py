#!/usr/bin/env python3
"""Add wheel-core collision authoring to chassis_arm mesh_usd assets.

This mirrors the wheel-core collision meshes that were added to the chassis_arm
URDF files, but patches the already-converted USD assets directly using `pxr`.
It avoids depending on the Isaac Sim URDF importer, which may be unavailable on
machines without a usable GPU/display stack.
"""

from __future__ import annotations

from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent.parent
USD_DIR = TASK_DIR / "data" / "robots" / "chassis_arm" / "mesh_usd"

WHEEL_LINK_NAMES = (
    "wheel_fl_link",
    "wheel_fr_link",
    "wheel_rl_link",
    "wheel_rr_link",
)


def _mesh_is_main_wheel_visual(mesh_prim: Usd.Prim, wheel_name: str) -> bool:
    path = str(mesh_prim.GetPath()).lower()
    return wheel_name in path and "roller" not in path and "collision" not in path


def _ensure_mesh_collision(mesh_prim: Usd.Prim) -> bool:
    changed = False
    if not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(mesh_prim)
        changed = True

    if not mesh_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        changed = True

    approx_attr = mesh_prim.GetAttribute("physics:approximation")
    if approx_attr.Get() != "convexHull":
        approx_attr.Set("convexHull")
        changed = True

    enabled_attr = mesh_prim.GetAttribute("physics:collisionEnabled")
    if not enabled_attr:
        enabled_attr = mesh_prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool)
    if enabled_attr.Get() is not True:
        enabled_attr.Set(True)
        changed = True

    return changed


def patch_one(path: Path) -> tuple[bool, int]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"unable to open USD stage: {path}")

    root = stage.GetDefaultPrim().GetPath()
    changed = False
    patched_mesh_count = 0

    for wheel_name in WHEEL_LINK_NAMES:
        wheel_prim = stage.GetPrimAtPath(root.AppendChild(wheel_name))
        if not wheel_prim:
            raise RuntimeError(f"missing wheel prim {wheel_name} in {path}")

        for prim in Usd.PrimRange(wheel_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if not _mesh_is_main_wheel_visual(prim, wheel_name):
                continue
            if _ensure_mesh_collision(prim):
                patched_mesh_count += 1
                changed = True

    if changed:
        stage.GetRootLayer().Save()
    return changed, patched_mesh_count


def main() -> None:
    usd_paths = sorted(path for path in USD_DIR.glob("*.usd") if path.is_file())
    if not usd_paths:
        raise RuntimeError(f"no USD files found in {USD_DIR}")

    changed_count = 0
    total_meshes = 0
    for path in usd_paths:
        changed, mesh_count = patch_one(path)
        total_meshes += mesh_count
        if changed:
            changed_count += 1
            print(f"[patched] {path.name} ({mesh_count} wheel meshes)")
        else:
            print(f"[ok] {path.name}")

    print(f"wheel-core USD stabilization complete: {changed_count}/{len(usd_paths)} files updated, {total_meshes} meshes patched")


if __name__ == "__main__":
    main()
