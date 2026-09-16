"""Disable USD native instancing on imported collision prims.

The URDF importer may author collision containers as native USD instances even
when the exported collision geometry is primitive Sphere. Some downstream checks
and PhysX collision setup paths expect the geometry to appear as direct composed
children under ``<link>/collisions``. This script keeps the referenced Sphere
geometry intact and authors a stronger ``instanceable = false`` opinion on every
collision prim that is still an instance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="De-instance collision prims in generated sphere USD assets.")
    parser.add_argument("--usd_dir", type=str, required=True, help="Directory containing generated robot USD files.")
    parser.add_argument("--usd", type=str, default=None, help="Optional single USD file to process.")
    parser.add_argument("--dry_run", action="store_true", help="Report changes without saving.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = make_parser().parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def _direct_child_count(prim: Usd.Prim) -> int:
    return len(list(prim.GetChildren()))


def _sphere_count(prim: Usd.Prim) -> int:
    from pxr import UsdGeom

    return sum(1 for child in Usd.PrimRange(prim) if child.IsA(UsdGeom.Sphere))


def process_one(usd_path: Path, dry_run: bool) -> tuple[int, int, int]:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {usd_path}")

    collision_prims = [
        prim
        for prim in list(stage.Traverse())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    instance_prims = [prim for prim in collision_prims if prim.IsInstance()]

    fixed = 0
    for prim in instance_prims:
        before_children = _direct_child_count(prim)
        before_spheres = _sphere_count(prim.GetPrototype()) if prim.GetPrototype().IsValid() else 0
        if not dry_run:
            prim.SetInstanceable(False)
        after_children = before_children if dry_run else _direct_child_count(prim)
        after_spheres = before_spheres if dry_run else _sphere_count(prim)
        print(
            f"{usd_path.name}: {prim.GetPath()} "
            f"children {before_children}->{after_children} spheres {before_spheres}->{after_spheres}"
        )
        fixed += 1

    if fixed and not dry_run:
        stage.GetRootLayer().Save()

    remaining_instances = 0
    direct_collision_children = 0
    if not dry_run:
        reopened = Usd.Stage.Open(str(usd_path))
        if reopened is None:
            raise RuntimeError(f"failed to reopen USD after save: {usd_path}")
        for prim in reopened.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if prim.IsInstance():
                    remaining_instances += 1
                direct_collision_children += _direct_child_count(prim)

    return fixed, remaining_instances, direct_collision_children


def main() -> None:
    usd_dir = Path(args_cli.usd_dir).resolve()
    usd_paths = [Path(args_cli.usd).resolve()] if args_cli.usd else sorted(usd_dir.glob("*.usd"))
    if not usd_paths:
        raise RuntimeError(f"No USD files found in: {usd_dir}")

    total_fixed = 0
    total_remaining = 0
    total_direct_children = 0
    for usd_path in usd_paths:
        fixed, remaining, direct_children = process_one(usd_path, args_cli.dry_run)
        total_fixed += fixed
        total_remaining += remaining
        total_direct_children += direct_children

    mode = "DRY RUN" if args_cli.dry_run else "UPDATED"
    print(f"{mode}: files={len(usd_paths)} fixed_collision_instances={total_fixed}")
    if not args_cli.dry_run:
        print(f"remaining_collision_instances={total_remaining}")
        print(f"direct_collision_children={total_direct_children}")
        if total_remaining:
            raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
