"""Minimal headless probe: create the Common env, reset, and dump robot state + stage prim info."""
from pathlib import Path
import sys
import argparse
import importlib

ALLSTAR_TOILET_ROOT = Path(__file__).resolve().parents[1]
if str(ALLSTAR_TOILET_ROOT) not in sys.path:
    sys.path.insert(0, str(ALLSTAR_TOILET_ROOT))

import torch  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

_parser = argparse.ArgumentParser(description="probe robot load")
_parser.add_argument("--task", type=str,
                     default="Isaac-Allstar-Toilet-Mecanum0730-XMS5Common-CuroboPosePlanTest-Dirty-OpenSeat-v0")
_parser.add_argument("--initial_robot_id", type=str, default="gripper_prismatic")
_parser.add_argument("--no_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(_parser)
_args_cli, _ = _parser.parse_known_args()
# mirror run_curobo_pose_plan_test.py:760 — cameras enabled in GUI
_args_cli.enable_cameras = (
    bool(getattr(_args_cli, "enable_cameras", False))
    or not _args_cli.headless
)

app_launcher = AppLauncher(_args_cli)
if _args_cli.no_fabric:
    import carb
    carb.settings.get_settings().set_bool("/physics/fabricEnabled", False)
    print("[Probe] forced /physics/fabricEnabled=False", flush=True)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks.manager_based.manipulation.allstar_toilet  # noqa: E402,F401
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402


def main() -> None:
    spec = gym.spec(_args_cli.task)
    env_cfg_entry = spec.kwargs["env_cfg_entry_point"]
    module_path, cls_name = env_cfg_entry.rsplit(":", 1)
    env_cfg = getattr(importlib.import_module(module_path), cls_name)()

    # replicate _apply_common_initial_robot_profile
    profiles = getattr(env_cfg, "robot_profiles", None) or getattr(env_cfg.scene, "robot_profiles", None)
    print(f"[Probe] profiles={sorted(profiles) if profiles else None}", flush=True)
    robot_id = _args_cli.initial_robot_id
    factory = getattr(importlib.import_module(env_cfg.__class__.__module__), "make_common_robot_cfg", None)
    if callable(factory):
        env_cfg.scene.robot = factory(robot_id, prim_path="{ENV_REGEX_NS}/robot")
    print(f"[Probe] robot spawn usd = {env_cfg.scene.robot.spawn.usd_path}", flush=True)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    print(f"[Probe] articulation num_bodies={robot.num_bodies} num_joints={robot.num_joints}", flush=True)

    # fabric on/off?
    import carb
    print(f"[Probe] /physics/fabricEnabled = {carb.settings.get_settings().get('/physics/fabricEnabled')}", flush=True)

    env.reset()

    stage = SimulationContext.instance().stage
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/robot")
    print(f"[Probe] /World/envs/env_0/robot valid={robot_prim.IsValid()} type={robot_prim.GetTypeName()} "
          f"children={len(robot_prim.GetChildren()) if robot_prim.IsValid() else -1}", flush=True)
    # dump first few children + USD-authored local-to-world transforms (fabric/physx independent)
    if robot_prim.IsValid():
        for i, ch in enumerate(robot_prim.GetChildren()):
            if i >= 14:
                print("  ... (truncated)", flush=True)
                break
            try:
                if ch.IsA(UsdGeom.Xformable):
                    m = UsdGeom.Xformable(ch).ComputeLocalToWorldTransform(0)
                    t = m.ExtractTranslation()
                    xform = f"t=({t[0]:.4f},{t[1]:.4f},{t[2]:.4f})"
                else:
                    xform = "not-xformable"
            except Exception as e:
                xform = f"ERR {e}"
            print(f"  child {ch.GetName()} type={ch.GetTypeName()} {xform}", flush=True)

    for i in range(3):
        try:
            rp = robot.data.root_pos_w.detach().cpu().tolist()
            rq = robot.data.root_quat_w.detach().cpu().tolist()
            lv = robot.data.root_lin_vel_w.detach().cpu().tolist()
            print(f"[Probe] step={i} root_pos={rp} root_quat={rq} lin_vel={lv}", flush=True)
        except Exception as e:
            print(f"[Probe] step={i} root read ERR: {type(e).__name__}: {e}", flush=True)
        try:
            jp = robot.data.joint_pos.detach().cpu().tolist()
            print(f"[Probe] step={i} joint_pos[0:8]={jp[0][:8] if jp else 'empty'}", flush=True)
        except Exception as e:
            print(f"[Probe] step={i} joint read ERR: {type(e).__name__}: {e}", flush=True)
        env.step()

    simulation_app.close()


if __name__ == "__main__":
    main()
