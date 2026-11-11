#!/usr/bin/env python
"""hdf5_to_groot.py — 把 run_curobo_pose_plan_test.py 采集的 HDF5 demo 转成 Isaac GR00T (LeRobot v2.1) 数据集。

GR00T 数据契约（已按 nvidia/Isaac-GR00T 的 lerobot_episode_loader + demo_data 核实）：
  root/
  ├── meta/
  │   ├── info.json           # features, data_path, video_path, chunks_size, fps
  │   ├── episodes.jsonl      # {"episode_index":N,"tasks":[...],"length":T}
  │   ├── tasks.jsonl         # {"task_index":0,"task":"..."}
  │   ├── modality.json       # state/action/video/annotation 分组
  │   ├── stats.json          # 归一化统计（loader 强制要求）
  │   └── relative_stats.json # RELATIVE action 必需（见下方 action 说明）
  ├── data/chunk-000/episode_000000.parquet      # 每条 demo 一个 parquet（标量列）
  └── videos/chunk-000/{video_key}/episode_000000.mp4  # 每相机一个 mp4，按 video_path 模式解码

字段映射（HDF5 → parquet）：
  action             7 维关节空间**绝对**量 [6 臂关节位置, 夹爪宽度]，next-step 约定
                     action[t] = joint_pos[t+1]（末帧重复自身）
  observation.state  7 维 [6 臂关节位置, 夹爪宽度] = joint_pos[t]
  timestamp / frame_index / episode_index / index / task_index

⚠️ action 存绝对量、不是增量：GR00T 的 launch_finetune.py 会强制
   use_relative_action=True，加载时由 StateActionProcessor 做 action[t] - state[t]
   得到相对量。转换脚本若再手动差分就是双重相减。
   对应 modality.json 的 action 分组 = single_arm(0:6, RELATIVE) + gripper(6:7, ABSOLUTE)。
   不做末端 RPY 增量：本任务 take_out 阶段 pitch 达 88.7°，万向锁会导致 RPY 差分
   出现最高 59 倍于真实旋转的假增量。

用法：
  python hdf5_to_groot.py --input outputs/cube_grasp_demos.hdf5 --output outputs/cube_grasp_demos_groot
  python hdf5_to_groot.py --input outputs/cube_grasp_demos.hdf5 --output outputs/cube_grasp_demo14_groot --demo demo_14
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

CAMERA_KEYS = ["cam_tool", "cam_fixed"]
DEFAULT_TASK = "grasp cube and place into trash bin"
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


# ── action 构造 ───────────────────────────────────────────────────────────────
def compute_action(joint_pos: np.ndarray) -> np.ndarray:
    """由关节位置序列构造 7 维 action：action[t] = joint_pos[t+1]，末帧重复自身。

    存绝对量、不做差分：GR00T 加载时会用 action[t] - state[t] 自己算相对量
    （见文件头说明），此处再差分会双重相减。state 与 action 同源于 joint_pos，
    保证这个减法得到的就是真实的每步关节增量。
    """
    t = len(joint_pos)
    action = np.empty((t, 7), dtype=np.float32)
    action[: t - 1] = joint_pos[1:, :7]
    action[t - 1] = joint_pos[t - 1, :7]
    return action


# ── 视频编码 ──────────────────────────────────────────────────────────────────
def replace_blank_frames(frames: np.ndarray, std_thresh: float = 3.0) -> np.ndarray:
    """把渲染器产生的空白帧（相机位姿未变时跳过重渲染，读出全白，rgb std≈0.36）
    替换为上一有效帧。位姿未变 → 场景也未变，替换等价无损失；位姿变化的空白帧
    （cam_fixed 亚像素抖动翻转）替换为最近有效帧后误差也 ≤1/30s，肉眼不可察。
    只替换像素不删帧，frame_index/timestamp/action 对齐不受影响。"""
    frames = np.asarray(frames)
    t = int(frames.shape[0])
    if t == 0 or frames.shape[-1] < 3:
        return frames
    stds = frames[..., :3].reshape(t, -1).std(axis=1)
    out = frames.copy()
    last_valid = None
    for i in range(t):
        if stds[i] < std_thresh:
            if last_valid is not None:
                out[i] = last_valid
        else:
            last_valid = frames[i]
    return out


def encode_video(frames: np.ndarray, path: Path, fps: int) -> None:
    """把 (T,H,W,3) uint8 帧序列编码为 mp4。优先 imageio+libx264，回退 cv2 mp4v。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = replace_blank_frames(frames)
    t, h, w, c = frames.shape
    frames = np.ascontiguousarray(frames)
    try:
        import imageio.v2 as imageio

        # macro_block_size=1：禁止 ffmpeg 把视频尺寸放大到 16 的倍数，
        # 保证实际尺寸与 info.json 一致（h264 解码端可处理任意尺寸）。
        with imageio.get_writer(
            str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1
        ) as writer:
            for f in frames:
                writer.append_data(f)
        return
    except Exception:
        pass
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


# ── 统计 ──────────────────────────────────────────────────────────────────────
def numeric_stats(values: np.ndarray) -> dict:
    """对数值特征（可多维，按最后一维逐通道）算 lerobot 风格统计。"""
    arr = np.asarray(values, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(-1, 1)
    out = {}
    out["count"] = int(flat.shape[0])
    for name, fn in [
        ("mean", np.mean),
        ("std", np.std),
        ("min", np.min),
        ("max", np.max),
    ]:
        out[name] = np.round(fn(flat, axis=0).astype(np.float32), 7).tolist()
    qs = np.quantile(flat, [0.25, 0.5, 0.75], axis=0).astype(np.float32)
    out["q1"] = np.round(qs[0], 7).tolist()
    out["median"] = np.round(qs[1], 7).tolist()
    out["q3"] = np.round(qs[2], 7).tolist()
    return out


def image_stats(frames: np.ndarray) -> dict:
    """对视频帧（T,H,W,3）算每通道 [0,1] 归一化统计，shape (3,1,1)，对齐 lerobot 视频统计。"""
    arr = np.asarray(frames, dtype=np.float64) / 255.0
    flat = arr.reshape(-1, arr.shape[-1])  # (T*H*W, 3)
    out = {"count": int(flat.shape[0])}
    for name, fn in [("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]:
        out[name] = np.round(fn(flat, axis=0).astype(np.float32), 7).reshape(3, 1, 1).tolist()
    qs = np.quantile(flat, [0.25, 0.5, 0.75], axis=0).astype(np.float32)
    for nm, q in [("q1", qs[0]), ("median", qs[1]), ("q3", qs[2])]:
        out[nm] = np.round(q, 7).reshape(3, 1, 1).tolist()
    return out


# ── 元数据写入 ────────────────────────────────────────────────────────────────
def write_info(
    meta_dir: Path,
    fps: int,
    robot_type: str,
    n_episodes: int,
    n_frames: int,
    task_str: str,
    video_shapes: dict[str, tuple[int, int]],
    total_tasks: int = 1,
) -> None:
    """video_shapes: {cam: (height, width)}，从实际录制帧读取（各相机分辨率可能不同）。"""
    features = {
        "action": {
            "dtype": "float32",
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
            "shape": [7],
        },
        "observation.state": {
            "dtype": "float32",
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
            "shape": [7],
        },
    }
    for cam in CAMERA_KEYS:
        h, w = video_shapes.get(cam, (460, 460))
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": h,
                "video.width": w,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    features.update({
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    })
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": n_episodes,
        "total_frames": n_frames,
        "total_tasks": total_tasks,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE,
        "features": features,
        "total_chunks": 0,
        "total_videos": n_episodes * len(CAMERA_KEYS),
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4))


def write_tasks(meta_dir: Path, tasks: list[str]) -> None:
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for idx, task in enumerate(tasks):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")


def write_modality(meta_dir: Path) -> None:
    modality = {
        "state": {"single_arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
        "action": {
            "single_arm": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "video": {
            cam: {"original_key": f"observation.images.{cam}"} for cam in CAMERA_KEYS
        },
        # key 必须与 modality config 的 language.modality_keys 去掉 "annotation." 前缀后一致，
        # 否则 loader 会 assert "Key ... not found in language modality"。
        # 这里对齐官方范例 examples/SO100/so100_config.py 的 annotation.human.task_description。
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=4))


def write_stats(meta_dir: Path, action_stats: dict, state_stats: dict, image_stats_map: dict) -> None:
    stats = {"action": action_stats, "observation.state": state_stats}
    for cam in CAMERA_KEYS:
        stats[f"observation.images.{cam}"] = image_stats_map[cam]
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=4))


# ── append 模式辅助 ──────────────────────────────────────────────────────────
def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _episode_data_path(out_root: Path, info: dict | None, ep_idx: int) -> Path:
    template = info.get("data_path", DATA_PATH_TEMPLATE) if info else DATA_PATH_TEMPLATE
    return out_root / template.format(episode_chunk=ep_idx // 1000, episode_index=ep_idx)


def _parquet_stats(out_root: Path, info: dict | None, episodes: list[dict]) -> tuple[dict, dict]:
    """从所有 episode 的 parquet 重新算 action/state 的全局统计（精确，不近似）。"""
    actions = []
    states = []
    for ep in episodes:
        path = _episode_data_path(out_root, info, int(ep["episode_index"]))
        df = pd.read_parquet(path)
        actions.append(np.stack(df["action"].to_numpy()).astype(np.float32))
        states.append(np.stack(df["observation.state"].to_numpy()).astype(np.float32))
    if not actions:
        raise SystemExit("没有 parquet 可统计")
    return numeric_stats(np.concatenate(actions, axis=0)), numeric_stats(np.concatenate(states, axis=0))


def merge_image_stats(old: dict | None, new: dict | None) -> dict | None:
    """合并两组 image stats：mean/std/min/max 精确，q1/median/q3 用加权近似。"""
    if old is None:
        return new
    if new is None:
        return old
    n1 = int(old["count"])
    n2 = int(new["count"])
    n = n1 + n2
    out = {"count": n}
    old_mean = np.asarray(old["mean"], dtype=np.float64)
    new_mean = np.asarray(new["mean"], dtype=np.float64)
    mean = (n1 * old_mean + n2 * new_mean) / n
    for key in ("mean", "std", "min", "max", "q1", "median", "q3"):
        a = np.asarray(old[key], dtype=np.float64)
        b = np.asarray(new[key], dtype=np.float64)
        if key == "mean":
            v = mean
        elif key == "std":
            var = (n1 * (a * a + old_mean * old_mean) + n2 * (b * b + new_mean * new_mean)) / n - mean * mean
            v = np.sqrt(np.maximum(var, 0.0))
        elif key == "min":
            v = np.minimum(a, b)
        elif key == "max":
            v = np.maximum(a, b)
        else:
            # 分位数无法精确合并；用两组分位数按像素数加权近似
            v = (n1 * a + n2 * b) / n
        out[key] = np.round(v.astype(np.float32), 7).reshape(3, 1, 1).tolist()
    return out


def merge_episode_image_stats(stats_list: list[dict]) -> dict | None:
    merged = None
    for s in stats_list:
        merged = merge_image_stats(merged, s)
    return merged


# ── 单条 demo 转换 ────────────────────────────────────────────────────────────
def convert_demo(group: h5py.Group, ep_idx: int, out_root: Path, task_str: str, fps: int,
                 global_frame_offset: int, task_index: int = 0) -> tuple:
    """转换一条 demo_X → 一个 episode_N。返回 (帧数, 统计..., 每相机视频 shape)。"""
    joint_pos = np.asarray(group["joint_pos"], dtype=np.float32)  # (T,8)
    timestamps = np.asarray(group["timestamps"], dtype=np.float32).reshape(-1)  # (T,)
    t = len(joint_pos)
    episode_chunk = ep_idx // 1000

    gripper = joint_pos[:, 6].astype(np.float32)  # gripper_joint1（0=开, 0.02=闭）
    state = np.concatenate([joint_pos[:, 0:6], gripper[:, None]], axis=1).astype(np.float32)  # (T,7)
    action = compute_action(joint_pos)  # (T,7) 绝对量，与 state 同源

    # ── 视频 ──
    video_stats = {}
    video_shapes = {}
    for cam in CAMERA_KEYS:
        frames = np.asarray(group[f"obs/{cam}_rgb"], dtype=np.uint8)  # (T,H,W,3)
        # 空白帧替换后再编码 & 统计（替换幂等），保证视频内容与 stats.json 一致。
        frames = replace_blank_frames(frames)
        video_shapes[cam] = (int(frames.shape[1]), int(frames.shape[2]))  # (H, W) 从实际帧读，各相机可不同分辨率
        video_path = out_root / VIDEO_PATH_TEMPLATE.format(
            episode_chunk=episode_chunk,
            video_key=f"observation.images.{cam}",
            episode_index=ep_idx,
        )
        encode_video(frames, video_path, fps)
        video_stats[cam] = image_stats(frames)

    # ── parquet ──
    ep_df = pd.DataFrame({
        "episode_index": np.full(t, ep_idx, dtype=np.int64),
        "frame_index": np.arange(t, dtype=np.int64),
        "timestamp": timestamps.astype(np.float32),
        "index": np.arange(global_frame_offset, global_frame_offset + t, dtype=np.int64),
        "task_index": np.full(t, int(task_index), dtype=np.int64),
        "action": [a.astype(np.float32) for a in action],
        "observation.state": [s.astype(np.float32) for s in state],
    })
    data_path = out_root / DATA_PATH_TEMPLATE.format(episode_chunk=episode_chunk, episode_index=ep_idx)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    ep_df.to_parquet(data_path, index=False)

    action_stats = numeric_stats(action)
    state_stats = numeric_stats(state)
    return t, action_stats, state_stats, video_stats, video_shapes


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cube-grasp HDF5 demos to Isaac GR00T (LeRobot v2.1) dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to the HDF5 dataset file.")
    parser.add_argument("--output", type=str, required=True, help="Output GR00T dataset directory.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Language task string.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default 24 = stride5 @120Hz). 改录制 stride 时同步为 120/stride.")
    parser.add_argument("--robot_type", type=str, default="mecanum730_xms5_gripper_prismatic", help="robot_type in info.json.")
    parser.add_argument("--no-success-only", action="store_true", help="Also convert demos marked success=False.")
    parser.add_argument("--demo", type=str, default=None, help="Only convert this demo key, e.g. 'demo_14'.")
    parser.add_argument("--append", action="store_true",
                        help="Append new episodes to an existing LeRobot dataset instead of overwriting it.")
    args = parser.parse_args()

    src = Path(args.input)
    out = Path(args.output)

    existing_info = None
    existing_episodes: list[dict] = []
    existing_tasks: list[str] = []
    existing_stats = None
    if out.exists():
        if not args.append:
            shutil.rmtree(out)
        else:
            info_path = out / "meta" / "info.json"
            if not info_path.is_file():
                raise SystemExit(f"--append 指定的目录不是 LeRobot 数据集（缺 meta/info.json）: {out}")
            existing_info = json.loads(info_path.read_text())
            existing_episodes = _read_jsonl(out / "meta" / "episodes.jsonl")
            existing_tasks = [t["task"] for t in _read_jsonl(out / "meta" / "tasks.jsonl")]
            if int(existing_info.get("fps", args.fps)) != int(args.fps):
                raise SystemExit(f"append 失败：已有 fps={existing_info.get('fps')}，本次 --fps={args.fps}")
            if existing_info.get("robot_type") != args.robot_type:
                raise SystemExit(
                    f"append 失败：已有 robot_type={existing_info.get('robot_type')}，"
                    f"本次 {args.robot_type}"
                )
            if existing_info.get("data_path", DATA_PATH_TEMPLATE) != DATA_PATH_TEMPLATE:
                raise SystemExit("append 失败：已有 data_path 模板不兼容")
            if existing_info.get("video_path", VIDEO_PATH_TEMPLATE) != VIDEO_PATH_TEMPLATE:
                raise SystemExit("append 失败：已有 video_path 模板不兼容")
            stats_path = out / "meta" / "stats.json"
            if stats_path.is_file():
                existing_stats = json.loads(stats_path.read_text())

    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(existing_tasks)
    if args.task in tasks:
        task_index = tasks.index(args.task)
    else:
        task_index = len(tasks)
        tasks.append(args.task)

    if existing_episodes:
        ep_idx = max(int(ep["episode_index"]) for ep in existing_episodes) + 1
    elif existing_info:
        ep_idx = int(existing_info.get("total_episodes", 0))
    else:
        ep_idx = 0
    total_frames = int(existing_info.get("total_frames", 0)) if existing_info else 0
    global_frame_offset = total_frames
    if existing_info:
        video_shapes = {
            cam: tuple(existing_info["features"][f"observation.images.{cam}"]["shape"][:2])
            for cam in CAMERA_KEYS
        }
    else:
        video_shapes = {}

    new_episodes: list[dict] = []
    new_image_stats: dict[str, list[dict]] = {cam: [] for cam in CAMERA_KEYS}

    with h5py.File(src, "r") as f:
        demo_keys = sorted(k for k in f.keys() if k.startswith("demo_"))
        if args.demo:
            demo_keys = [k for k in demo_keys if k == args.demo]
            if not demo_keys:
                raise SystemExit(f"指定的 demo 不存在: {args.demo}")
        if not demo_keys:
            raise SystemExit(f"没有找到 demo_* 组: {list(f.keys())}")

        for key in demo_keys:
            g = f[key]
            success = bool(g.attrs.get("success", False))
            if not success and not args.no_success_only:
                print(f"[skip] {key} success=False，跳过（用 --no-success-only 可保留）")
                continue
            length, _, _, video_stats, vshapes = convert_demo(
                g, ep_idx, out, args.task, args.fps, global_frame_offset, task_index=task_index
            )
            for cam, shape in vshapes.items():
                if cam in video_shapes and video_shapes[cam] != tuple(shape):
                    raise SystemExit(
                        f"append 失败：{cam} 分辨率不一致 已有={video_shapes[cam]} 本次={shape}"
                    )
                video_shapes[cam] = tuple(shape)
            new_episodes.append({"episode_index": ep_idx, "tasks": [args.task], "length": length})
            for cam in CAMERA_KEYS:
                new_image_stats[cam].append(video_stats[cam])
            total_frames += length
            global_frame_offset += length
            print(f"[ok] {key} → episode_{ep_idx:06d}  length={length}  success={success}", flush=True)
            ep_idx += 1

    if not new_episodes:
        raise SystemExit("没有可转换的 demo（全部 success=False 被跳过？加 --no-success-only）")

    all_episodes = existing_episodes + new_episodes
    # action/state 统计从所有 parquet 重算，append 后仍然精确
    action_stats, state_stats = _parquet_stats(out, existing_info, all_episodes)

    image_stats_map = {}
    for cam in CAMERA_KEYS:
        merged_new = merge_episode_image_stats(new_image_stats[cam])
        old_img = existing_stats.get(f"observation.images.{cam}") if existing_stats else None
        image_stats_map[cam] = merge_image_stats(old_img, merged_new)

    write_info(
        meta_dir,
        args.fps,
        args.robot_type,
        len(all_episodes),
        total_frames,
        args.task,
        video_shapes,
        total_tasks=len(tasks),
    )
    write_tasks(meta_dir, tasks)
    write_modality(meta_dir)
    write_stats(meta_dir, action_stats, state_stats, image_stats_map)
    _write_jsonl(meta_dir / "episodes.jsonl", all_episodes)

    rel_path = meta_dir / "relative_stats.json"
    if rel_path.is_file():
        rel_path.unlink()
        print("[warn] 已删除旧的 relative_stats.json；追加数据后需要重新跑 GR00T stats。")

    print(f"\n✅ 转换完成: 新增 {len(new_episodes)} 条，总计 {len(all_episodes)} 条 → {out}")
    print(f"   total_frames={total_frames}  fps={args.fps}")
    print("   meta/  data/  videos/ 已生成。")
    print("   ⚠️ 用 GR00T 重算统计（single_arm 是 RELATIVE，缺 relative_stats.json 会在加载时报错）：")
    print("      python -m gr00t.data.stats --dataset-path <out> --embodiment-tag NEW_EMBODIMENT \\")
    print("        --modality-config-path <your_config.py>")


if __name__ == "__main__":
    main()
