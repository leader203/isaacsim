"""诊断相机 RGB 红色斑点：验证 carb 设置是否生效 + 量化红点。

用法:
  ./isaaclab.sh -p probe_red_spots.py [--spp N] [--no-denoiser]
"""
import argparse

import numpy as np

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser(description="red spot probe")
_parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Allstar-Toilet-Mecanum0730-XMS5Common-CuroboPosePlanTest-Dirty-OpenSeat-v0",
)
_parser.add_argument("--initial_robot_id", type=str, default="gripper_prismatic")
_parser.add_argument("--spp", type=int, default=8, help="direct lighting samples per pixel")
_parser.add_argument("--no-denoiser", action="store_true", help="disable DL denoiser")
AppLauncher.add_app_launcher_args(_parser)
_args_cli, _ = _parser.parse_known_args()
# 用 GUI 模式：与主脚本一致（headless 模式下 omni.hydra.usdrt_delegate 未加载，
# 相机 XformPrimView 初始化会报 AttributeError: usdrt.hierarchy 缺失）
_args_cli.headless = False
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

    env_cfg.sim.render.samples_per_pixel = _args_cli.spp
    env_cfg.sim.render.enable_dl_denoiser = not _args_cli.no_denoiser

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")

    # ── 验证 carb 设置是否真的生效 ─────────────────────────────────────────
    sim = env.sim
    for key in (
        "/rtx/directLighting/sampledLighting/samplesPerPixel",
        "/rtx-transient/dldenoiser/enabled",
        "/rtx/post/aaMode",
        "/rtx/directLighting/sampledLighting/enabled",
        "/rtx/rendermode",
    ):
        try:
            print(f"[Cfg] {key} = {sim.carb_settings.get(key)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[Cfg] {key} read error: {e}", flush=True)

    # ── 渲染若干帧，取 cam_tool RGB ───────────────────────────────────────
    import torch

    for i in range(15):
        act = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        env.step(act)
        env.sim.render()

    cam = env.scene.sensors.get("cam_tool", None)
    if cam is None:
        print("[Cam] cam_tool not found", flush=True)
        simulation_app.close()
        return
    out = cam.data.output
    rgb = out.get("rgb")
    if rgb is None or len(rgb) == 0:
        print("[Cam] no rgb data", flush=True)
        simulation_app.close()
        return
    rgb_np = np.asarray(rgb[0].detach().cpu().numpy(), dtype=np.uint8)
    if rgb_np.ndim == 3 and rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[..., :3]
    print(f"[Cam] rgb shape={rgb_np.shape} min={rgb_np.min()} max={rgb_np.max()} mean={rgb_np.mean():.1f}", flush=True)

    r = rgb_np[..., 0].astype(int)
    g = rgb_np[..., 1].astype(int)
    b = rgb_np[..., 2].astype(int)
    bright = rgb_np.mean(axis=-1) > 60

    # 强红色：r 明显高于 g/b 且够亮
    red = (r - g > 40) & (r - b > 40) & (r > 120) & bright
    # 白热 firefly：三通道都很高（可能被用户看成彩色点）
    firefly = (r > 230) & (g > 230) & (b > 230)

    h, w = red.shape
    total_px = h * w
    red_n = int(red.sum())
    firefly_n = int(firefly.sum())
    print(f"[Red] red_pixels={red_n} frac={100.0 * red_n / total_px:.4f}%", flush=True)
    print(f"[Red] firefly_pixels={firefly_n} frac={100.0 * firefly_n / total_px:.4f}%", flush=True)

    # 简单 4 连通域计数（不用 scipy）
    def label4(mask: np.ndarray) -> tuple[int, int]:
        lab = np.zeros(mask.shape, dtype=np.int32)
        cur = 0
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x]:
                    up = lab[y - 1, x] if y > 0 else 0
                    left = lab[y, x - 1] if x > 0 else 0
                    if up and left:
                        lab[y, x] = min(up, left)
                        if up != left:
                            lab[lab == max(up, left)] = min(up, left)
                    elif up:
                        lab[y, x] = up
                    elif left:
                        lab[y, x] = left
                    else:
                        cur += 1
                        lab[y, x] = cur
        ids, counts = np.unique(lab[lab > 0], return_counts=True)
        sizes = counts if len(ids) else np.zeros(0, dtype=int)
        return int(len(ids)), int(sizes.max()) if len(sizes) else 0

    for name, m in (("red", red), ("firefly", firefly)):
        n, maxblob = label4(m)
        print(f"[Blob] {name}: blobs={n} max_blob={maxblob}px", flush=True)

    # 输出红点像素的 RGB 样本，判断是"真红"还是"白热溢出"
    ys, xs = np.where(red)
    if len(ys):
        print("[Red] 红点样本 RGB（前 10 个）:", flush=True)
        for i in range(min(10, len(ys))):
            yy, xx = ys[i], xs[i]
            print(f"      ({xx},{yy}) rgb={tuple(rgb_np[yy, xx])}", flush=True)

    from PIL import Image

    Image.fromarray(rgb_np).save("/tmp/red_probe_rgb.png")
    Image.fromarray((red.astype(np.uint8) * 255)).save("/tmp/red_probe_mask.png")
    print("[Red] saved /tmp/red_probe_rgb.png /tmp/red_probe_mask.png", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
