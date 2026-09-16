"""Inspect collision setup for a chassis_arm mesh_usd asset.

Usage
-----
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/usdandurdf/inspect_mecanum_collision_setup.py \
    --asset source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/data/robots/chassis_arm/mesh_usd/mecanum730_xms5_brush_rolling.usd
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect collision APIs and mesh approximations for a USD asset.")
parser.add_argument("--asset", type=str, required=True, help="Path to the USD/USDa asset.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


def main() -> None:
    asset_path = Path(args_cli.asset).resolve()
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError(f"Unable to open stage: {asset_path}")

    print(f"Asset: {asset_path}")
    print(f"Default prim: {stage.GetDefaultPrim().GetPath()}")
    print()

    for prim in stage.Traverse():
        name = prim.GetName()
        path = str(prim.GetPath())
        if not (
            "wheel_" in name
            or "roller" in name
            or name in {"base_link", "base_footprint"}
            or "wheel_" in path
            or "roller" in path
        ):
            continue

        has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        type_name = prim.GetTypeName()
        approx = prim.GetAttribute("physics:approximation").Get()
        rest_offset = prim.GetAttribute("physxCollision:restOffset").Get()
        contact_offset = prim.GetAttribute("physxCollision:contactOffset").Get()

        mesh_points = None
        if prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            mesh_points = len(points) if points is not None else 0

        print(
            f"{path} | type={type_name or '-'} rigid={has_rigid} collision={has_collision} "
            f"approx={approx} restOffset={rest_offset} contactOffset={contact_offset} points={mesh_points}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
