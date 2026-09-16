import argparse
from pathlib import Path
import numpy as np

from isaaclab.app import AppLauncher
_parser = argparse.ArgumentParser(description="render probe")
_parser.add_argument("--task", type=str,
                     default="Isaac-Allstar-Toilet-Mecanum0730-XMS5Common-CuroboPosePlanTest-Dirty-OpenSeat-v0")
_parser.add_argument("--initial_robot_id", type=str, default="gripper_prismatic")
AppLauncher.add_app_launcher_args(_parser)
_args_cli, _ = _parser.parse_known_args()
_args_cli.headless = True
_args_cli.enable_cameras = True
simulation_app = AppLauncher(_args_cli).app

import gymnasium as gym
import isaaclab_tasks.manager_based.manipulation.allstar_toilet
from isaaclab.envs import ManagerBasedRLEnv

def main():
    import importlib
    spec = gym.spec(_args_cli.task)
    module_path, cls_name = spec.kwargs["env_cfg_entry_point"].rsplit(":", 1)
    env_cfg = getattr(importlib.import_module(module_path), cls_name)()
    factory = getattr(importlib.import_module(env_cfg.__class__.__module__), "make_common_robot_cfg", None)
    if callable(factory):
        env_cfg.scene.robot = factory(_args_cli.initial_robot_id, prim_path="{ENV_REGEX_NS}/robot")

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    for _ in range(15):
        env.step(env.action_manager.sample() if env.action_manager is not None else None)
    frame = env.render(recompute=True)
    if frame is not None:
        arr = np.asarray(frame)
        out = "/tmp/render_probe_observer.png"
        from PIL import Image
        Image.fromarray(arr).save(out)
        print(f"[Render] saved observer frame {arr.shape} -> {out}", flush=True)
        print(f"[Render] pixel stats: min={arr.min()} max={arr.max()} mean={arr.mean():.1f} nonzero={(arr>8).mean()*100:.1f}%", flush=True)
    else:
        print("[Render] frame is None", flush=True)

    # also dump robot prim visibility attr on the live stage
    stage = env.sim.stage
    from pxr import UsdGeom
    rob = stage.GetPrimAtPath("/World/envs/env_0/robot")
    print(f"[Render] robot prim valid={rob.IsValid()} children={len(rob.GetChildren()) if rob.IsValid() else -1}", flush=True)
    if rob.IsValid():
        v = rob.GetAttribute("visibility").Get() if rob.HasAttribute("visibility") else None
        print(f"[Render] robot visibility={v}", flush=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
