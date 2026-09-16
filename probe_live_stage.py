"""Headless probe: create Common env, dump live layer stack + prim tree + robot visibility."""
from pathlib import Path
import sys
import argparse

from isaaclab.app import AppLauncher
_parser = argparse.ArgumentParser(description="probe live stage")
_parser.add_argument("--task", type=str,
                     default="Isaac-Allstar-Toilet-Mecanum0730-XMS5Common-CuroboPosePlanTest-Dirty-OpenSeat-v0")
_parser.add_argument("--initial_robot_id", type=str, default="gripper_prismatic")
AppLauncher.add_app_launcher_args(_parser)
_args_cli, _ = _parser.parse_known_args()
_args_cli.headless = True
simulation_app = AppLauncher(_args_cli).app

import gymnasium as gym
import isaaclab_tasks.manager_based.manipulation.allstar_toilet
from isaaclab.envs import ManagerBasedRLEnv
from pxr import Usd, UsdGeom, Sdf

def dump_layer_stack(stage, label):
    print(f"\n=== [{label}] LayerStack ({len(stage.GetLayerStack())} layers) ===")
    for lyr in stage.GetLayerStack():
        print(f"  id={lyr.identifier}")
        print(f"     anon={lyr.anonymous} defaultPrim={lyr.defaultPrim} muted={lyr.muted}")

def walk_prims(stage, root_path, prefix="", depth=0, maxdepth=3):
    prim = stage.GetPrimAtPath(root_path)
    if not prim.IsValid():
        print(f"  {prefix}[MISSING] {root_path}")
        return
    vis = ""
    if prim.HasAttribute("visibility"):
        v = prim.GetAttribute("visibility").Get()
        if v not in (None, "inherited"):
            vis = f" VISIBILITY={v}"
    nchild = len(prim.GetChildren())
    print(f"  {prefix}{prim.GetPath()} type={prim.GetTypeName()} children={nchild}{vis}")
    if depth < maxdepth:
        for ch in list(prim.GetChildren())[:8]:
            walk_prims(stage, str(ch.GetPath()), prefix+"  ", depth+1, maxdepth)

def main():
    spec = gym.spec(_args_cli.task)
    env_cfg_entry = spec.kwargs["env_cfg_entry_point"]
    module_path, cls_name = env_cfg_entry.rsplit(":", 1)
    import importlib
    env_cfg = getattr(importlib.import_module(module_path), cls_name)()
    factory = getattr(importlib.import_module(env_cfg.__class__.__module__), "make_common_robot_cfg", None)
    if callable(factory):
        env_cfg.scene.robot = factory(_args_cli.initial_robot_id, prim_path="{ENV_REGEX_NS}/robot")
    print(f"[Probe] robot usd = {env_cfg.scene.robot.spawn.usd_path}", flush=True)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    stage = env.sim.stage
    dump_layer_stack(stage, "after env create")

    env.reset()
    dump_layer_stack(stage, "after reset")

    print("\n=== /World ===")
    world = stage.GetPrimAtPath("/World")
    for ch in world.GetChildren():
        print(f"  /World/{ch.GetName()} children={len(ch.GetChildren())}")

    print("\n=== /World/envs/env_0 ===")
    walk_prims(stage, "/World/envs/env_0", depth=0, maxdepth=2)

    print("\n=== robot articulation state ===")
    robot = env.scene["robot"]
    print(f"  num_bodies={robot.num_bodies} num_joints={robot.num_joints}")
    import torch
    print(f"  root_pos={robot.data.root_pos_w.detach().cpu().tolist()}")
    print(f"  root_quat={robot.data.root_quat_w.detach().cpu().tolist()}")

    print("\n=== door count in live stage ===")
    door_count = 0
    for p in stage.Traverse():
        if "Door_" in str(p.GetPath()):
            door_count += 1
    print(f"  prims containing 'Door_' = {door_count}")

    simulation_app.close()

if __name__ == "__main__":
    main()
