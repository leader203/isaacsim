"""Print selected USD collision prim trees for debugging importer output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect USD collision prim tree.")
    parser.add_argument("--usd", type=str, required=True)
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--max_depth", type=int, default=5)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = make_parser().parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def schema_summary(prim: Usd.Prim) -> str:
    schema_names = list(prim.GetAppliedSchemas())
    bits = []
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        bits.append("CollisionAPI")
    if prim.IsA(UsdGeom.Sphere):
        bits.append("Sphere")
    if prim.IsA(UsdGeom.Mesh):
        bits.append("Mesh")
    if prim.IsA(UsdGeom.Cylinder):
        bits.append("Cylinder")
    if prim.IsA(UsdGeom.Capsule):
        bits.append("Capsule")
    if prim.IsA(UsdGeom.Cube):
        bits.append("Cube")
    if schema_names:
        bits.append("schemas=" + ",".join(schema_names))
    return " ".join(bits)


def attr_summary(prim: Usd.Prim) -> list[str]:
    lines: list[str] = []
    for attr in prim.GetAttributes():
        name = attr.GetName()
        lines.append(f"    @{name} = {attr.Get()}")
    return lines


def prim_flags(prim: Usd.Prim) -> str:
    flags = []
    if prim.IsInstance():
        flags.append("instance")
    if prim.IsInstanceProxy():
        flags.append("instanceProxy")
    if prim.IsPrototype():
        flags.append("prototype")
    if prim.HasAuthoredReferences():
        flags.append("authoredReferences")
    if prim.HasAuthoredPayloads():
        flags.append("authoredPayloads")
    prototype = prim.GetPrototype() if prim.IsInstance() else None
    if prototype and prototype.IsValid():
        flags.append(f"prototypePath={prototype.GetPath()}")
    return " ".join(flags)


def print_tree(prim: Usd.Prim, root_path: str, max_depth: int) -> None:
    root_depth = len(root_path.strip("/").split("/"))
    for item in Usd.PrimRange(prim):
        path = str(item.GetPath())
        depth = len(path.strip("/").split("/")) - root_depth
        if depth > max_depth:
            continue
        indent = "  " * depth
        print(f"{indent}{path} type={item.GetTypeName()} {prim_flags(item)} {schema_summary(item)}")
        for line in attr_summary(item):
            print(indent + line)


def print_prototype(prim: Usd.Prim, max_depth: int) -> None:
    if not prim.IsInstance():
        return
    prototype = prim.GetPrototype()
    if not prototype or not prototype.IsValid():
        return
    print("\nPROTOTYPE FOR", prim.GetPath(), "=>", prototype.GetPath())
    print_tree(prototype, str(prototype.GetPath()), max_depth)


def main() -> None:
    usd_path = Path(args_cli.usd).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"failed to open {usd_path}")

    print("USD:", usd_path)
    for path in args_cli.paths:
        prim = stage.GetPrimAtPath(path)
        print("\nPATH:", path, "valid=", prim.IsValid())
        if prim.IsValid():
            print_tree(prim, path, args_cli.max_depth)
            print_prototype(prim, args_cli.max_depth)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
