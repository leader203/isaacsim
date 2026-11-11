#!/usr/bin/env python
"""extract_grasp_segments.py — 从每条完整 demo 截取"接近→夹紧→起抬"段，追加为新的 episode。

背景：outputs/cube_grasp_demos_groot 里每条 episode 都是完整 SOP（接近 45% → 夹持搬运 45%
→ 回收 10%）。夹取精细操作只占其中一小段。本脚本把这段切出来作为独立 episode，追加到
同一个 LeRobot v2.1 数据集中，用于数据增强。

夹取段定义（默认）：
  锚点 close_start = state[:,6]（夹爪宽度，0=张开、0.02=夹紧）首次进入"长时间闭合"的帧；
  窗口 = [close_start - pre, close_start + post)，默认 pre=60 / post=120（30fps 下 6s）。
  加 --from-start 时窗口 = [0, close_start + post)：包含从初始位姿移到方块附近、下沉到
  抓取高度的完整接近过程（轨迹从固定初始位姿开始，底座不动，前 ~30 帧后才开始运动）。
  取"最长闭合段"作为锚点：部分 episode 在接近阶段或松开后有短暂误闭合，最长段才是真正的
  夹持段。episode 356 夹爪只闭合到 0.0052，用默认阈值 0.0015 仍能检出。

追加规则：
  - 原 episode 0..N-1 不动；
  - 新 episode N..2N-1 = 原 episode 0..N-1 的夹取段（new_episode_index = N + old_index）；
  - 新 parquet 只保留窗口内的行，frame_index/timestamp 从 0 重新编号；
  - 两个相机的 mp4 按帧精确 seek 后重新编码（与原数据集相同的 libx264 quality=8 参数）；
  - meta/info.json、episodes.jsonl、stats.json 在全部 episode 写完后统一重算：
      action / observation.state 精确重算；
      视频 mean/std 用 sum/sumsq 与旧统计精确合并，q1/median/q3 按像素数加权近似。

用法：
  conda activate isaaclab
  python scripts/tools/extract_grasp_segments.py --dry-run          # 只打印切段结果
  python scripts/tools/extract_grasp_segments.py --episodes 0       # 先做一条验证（不更新 meta）
  python scripts/tools/extract_grasp_segments.py                    # 全量追加 1046 条
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CAMERA_KEYS = ["cam_tool", "cam_fixed"]
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _data_path(root: Path, ep_idx: int) -> Path:
    return root / DATA_PATH_TEMPLATE.format(episode_chunk=ep_idx // 1000, episode_index=ep_idx)


def _video_path(root: Path, ep_idx: int, cam: str) -> Path:
    return root / VIDEO_PATH_TEMPLATE.format(
        episode_chunk=ep_idx // 1000, video_key=f"observation.images.{cam}", episode_index=ep_idx
    )


# ── 统计 ──────────────────────────────────────────────────────────────────────
def numeric_stats(values: np.ndarray) -> dict:
    """与 hdf5_to_groot.numeric_stats 完全一致的 lerobot 数值统计。"""
    arr = np.asarray(values, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(-1, 1)
    out = {"count": int(flat.shape[0])}
    for name, fn in [("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]:
        out[name] = np.round(fn(flat, axis=0).astype(np.float32), 7).tolist()
    qs = np.quantile(flat, [0.25, 0.5, 0.75], axis=0).astype(np.float32)
    out["q1"] = np.round(qs[0], 7).tolist()
    out["median"] = np.round(qs[1], 7).tolist()
    out["q3"] = np.round(qs[2], 7).tolist()
    return out


def image_stats_frames(frames: np.ndarray) -> dict:
    """对 (T,H,W,3) uint8 帧算每通道 [0,1] 归一化统计，输出 shape (3,1,1)。

    逐帧累加 sum/sumsq，避免把整段 clip 转成 float64 占用数 GB；分位数在每 4 个像素
    上采样估计（这些分位数只用于 meta 展示，归一化用的是 mean/std）。
    """
    t, h, w, _ = frames.shape
    n = t * h * w
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    lo = np.full(3, 255.0)
    hi = np.zeros(3)
    for f in frames:
        x = f.astype(np.float64) / 255.0
        total += x.sum(axis=(0, 1))
        total_sq += (x * x).sum(axis=(0, 1))
        lo = np.minimum(lo, x.min(axis=(0, 1)))
        hi = np.maximum(hi, x.max(axis=(0, 1)))
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean * mean, 0.0))
    sample = frames[:, :, ::8, :].reshape(-1, 3).astype(np.float64) / 255.0
    qs = np.quantile(sample, [0.25, 0.5, 0.75], axis=0)
    out = {"count": int(n)}
    for name, val in [("mean", mean), ("std", std), ("min", lo), ("max", hi)]:
        out[name] = np.round(val.astype(np.float32), 7).reshape(3, 1, 1).tolist()
    for name, q in zip(["q1", "median", "q3"], qs):
        out[name] = np.round(q.astype(np.float32), 7).reshape(3, 1, 1).tolist()
    return out


def merge_image_stats(old: dict, new: dict) -> dict:
    """按像素数合并两份图像统计。mean/std 精确（用 sum/sumsq 恢复），分位数加权近似。"""
    c_old, c_new = float(old["count"]), float(new["count"])
    c = c_old + c_new
    out = {"count": int(c)}
    for name in ("mean", "min", "max", "q1", "median", "q3"):
        a = np.asarray(old[name], dtype=np.float64)
        b = np.asarray(new[name], dtype=np.float64)
        if name == "mean":
            val = (a * c_old + b * c_new) / c
        elif name == "min":
            val = np.minimum(a, b)
        elif name == "max":
            val = np.maximum(a, b)
        else:  # 分位数：加权近似
            val = (a * c_old + b * c_new) / c
        out[name] = np.round(val.astype(np.float32), 7).tolist()
    a, b = np.asarray(old["mean"], dtype=np.float64), np.asarray(new["mean"], dtype=np.float64)
    var_old = np.asarray(old["std"], dtype=np.float64) ** 2 + a * a
    var_new = np.asarray(new["std"], dtype=np.float64) ** 2 + b * b
    mean = (a * c_old + b * c_new) / c
    var = np.maximum((var_old * c_old + var_new * c_new) / c - mean * mean, 0.0)
    out["std"] = np.round(np.sqrt(var).astype(np.float32), 7).tolist()
    return out


# ── 夹取段检测 ────────────────────────────────────────────────────────────────
def find_grasp_window(gripper: np.ndarray, threshold: float, pre: int, post: int):
    """返回 (close_start, close_end, start, end)；取超过阈值的**最长**连续段作为夹持段。"""
    hi = np.asarray(gripper) > threshold
    d = np.diff(np.r_[0, hi.astype(np.int8), 0])
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if len(starts) == 0:
        raise ValueError("gripper never closes (no run above threshold)")
    i = int(np.argmax(ends - starts))
    close_start, close_end = int(starts[i]), int(ends[i])
    t = len(gripper)
    return close_start, close_end, max(0, close_start - pre), min(t, close_start + post)


# ── 视频读写 ──────────────────────────────────────────────────────────────────
def read_frames(src: Path, start: int, n: int, fps: int) -> np.ndarray:
    """用 ffmpeg 按帧号精确 seek，读出 n 帧 RGB。输入 seek 已验证可精确落到目标帧。"""
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{start / fps:.6f}", "-i", str(src),
        "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    # 从探测结果拿分辨率
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(src)],
        capture_output=True, text=True, check=True,
    )
    w, h = (int(v) for v in probe.stdout.strip().split("x"))
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    if buf.size != n * h * w * 3:
        raise RuntimeError(
            f"{src.name}: 期望 {n} 帧 ({n * h * w * 3} bytes)，实际 {buf.size} bytes"
        )
    return buf.reshape(n, h, w, 3)


def write_video(frames: np.ndarray, dst: Path, fps: int) -> None:
    """与原 hdf5_to_groot.encode_video 相同的编码参数（libx264 quality=8, macro_block_size=1）。"""
    import imageio.v2 as imageio

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.mp4")  # 保留 .mp4 后缀，imageio 靠它选容器
    with imageio.get_writer(
        str(tmp), fps=fps, codec="libx264", quality=8, macro_block_size=1
    ) as writer:
        for f in frames:
            writer.append_data(f)
    tmp.replace(dst)


# ── 单条处理 ──────────────────────────────────────────────────────────────────
def process_episode(root: Path, seg: dict, new_idx: int, index_offset: int, fps: int) -> dict:
    old_idx = seg["episode"]
    start, end, n = seg["start"], seg["end"], seg["end"] - seg["start"]
    parquet_path = _data_path(root, new_idx)
    video_paths = {cam: _video_path(root, new_idx, cam) for cam in CAMERA_KEYS}

    ready = (
        parquet_path.exists()
        and all(p.exists() for p in video_paths.values())
        and len(pd.read_parquet(parquet_path, columns=["frame_index"])) == n
    )

    if ready:
        frames_by_cam = {cam: read_frames(p, 0, n, fps) for cam, p in video_paths.items()}
    else:
        src_parquet = _data_path(root, old_idx)
        df = pd.read_parquet(src_parquet)
        sub = df.iloc[start:end].copy()
        sub["episode_index"] = np.full(n, new_idx, dtype=np.int64)
        sub["frame_index"] = np.arange(n, dtype=np.int64)
        sub["timestamp"] = ((np.arange(n) + 1) / fps).astype(np.float32)
        sub["index"] = np.arange(index_offset, index_offset + n, dtype=np.int64)
        sub["task_index"] = np.full(n, int(seg["task_index"]), dtype=np.int64)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_parquet = parquet_path.with_suffix(".parquet.tmp")
        sub.to_parquet(tmp_parquet, index=False)
        tmp_parquet.replace(parquet_path)

        frames_by_cam = {}
        for cam in CAMERA_KEYS:
            src = root / VIDEO_PATH_TEMPLATE.format(
                episode_chunk=old_idx // 1000, video_key=f"observation.images.{cam}",
                episode_index=old_idx,
            )
            frames = read_frames(src, start, n, fps)
            write_video(frames, video_paths[cam], fps)
            frames_by_cam[cam] = frames

    return {
        "new_episode": new_idx,
        "episode": old_idx,
        "close_start": seg["close_start"],
        "close_end": seg["close_end"],
        "start": start,
        "end": end,
        "length": n,
        "video_stats": {cam: image_stats_frames(f) for cam, f in frames_by_cam.items()},
    }


# ── meta ──────────────────────────────────────────────────────────────────────
def update_meta(root: Path, n_orig: int, records: list[dict], fps: int, dry_run: bool) -> None:
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text())
    if int(info["total_episodes"]) != n_orig:
        raise SystemExit(
            f"info.json total_episodes={info['total_episodes']} != 数据集当前 {n_orig} 条，"
            "可能已经追加过，拒绝重复追加。"
        )
    episodes = _read_jsonl(meta / "episodes.jsonl")
    if len(episodes) != n_orig:
        raise SystemExit(f"episodes.jsonl 有 {len(episodes)} 行，应为 {n_orig}")

    task = episodes[0]["tasks"][0]
    new_episodes = [
        {"episode_index": r["new_episode"], "tasks": [task], "length": r["length"]}
        for r in sorted(records, key=lambda r: r["new_episode"])
    ]
    total_episodes = n_orig * 2
    total_frames = sum(int(e["length"]) for e in episodes) + sum(r["length"] for r in records)

    # action / state 全量精确重算
    actions, states = [], []
    for ep in range(total_episodes):
        p = _data_path(root, ep)
        if not p.exists():
            raise SystemExit(f"缺少 {p}")
        df = pd.read_parquet(p, columns=["action", "observation.state"])
        actions.append(np.stack(df["action"].to_numpy()).astype(np.float32))
        states.append(np.stack(df["observation.state"].to_numpy()).astype(np.float32))
    action_stats = numeric_stats(np.concatenate(actions))
    state_stats = numeric_stats(np.concatenate(states))

    # 视频统计：旧统计 + 新片段统计合并
    old_stats = json.loads((meta / "stats.json").read_text())
    new_stats = {}
    for cam in CAMERA_KEYS:
        agg = None
        for r in records:
            agg = r["video_stats"][cam] if agg is None else merge_image_stats(agg, r["video_stats"][cam])
        new_stats[cam] = agg
    merged_video = {
        cam: merge_image_stats(old_stats[f"observation.images.{cam}"], new_stats[cam])
        for cam in CAMERA_KEYS
    }

    if dry_run:
        _log(f"[dry-run] 将写入 {len(records)} 条新 episode，总帧数 {total_frames}")
        return

    # 备份，再原子写 meta
    backup = root.parent / f"{root.name}_meta_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(meta, backup)
    _log(f"已备份 meta → {backup}")

    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = total_episodes * len(CAMERA_KEYS)
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["total_chunks"] = (total_episodes - 1) // int(info["chunks_size"]) + 1
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    _write_jsonl(meta / "episodes.jsonl", episodes + new_episodes)
    stats = {"action": action_stats, "observation.state": state_stats}
    for cam in CAMERA_KEYS:
        stats[f"observation.images.{cam}"] = merged_video[cam]
    (meta / "stats.json").write_text(json.dumps(stats, indent=4))
    _log(
        f"meta 已更新: episodes {n_orig}→{total_episodes}, frames {total_frames}, "
        f"chunks {info['total_chunks']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="outputs/cube_grasp_demos_groot", help="LeRobot 数据集根目录")
    parser.add_argument("--pre", type=int, default=60, help="夹爪闭合前的帧数（默认 60）")
    parser.add_argument("--post", type=int, default=120, help="夹爪闭合后的帧数（默认 120）")
    parser.add_argument("--from-start", action="store_true",
                        help="夹取段从 episode 第 0 帧开始，包含移到方块附近 + 下沉的完整接近")
    parser.add_argument("--threshold", type=float, default=0.0015, help="夹爪闭合判定阈值（默认 0.0015）")
    parser.add_argument("--workers", type=int, default=6, help="并行处理的 episode 数（默认 6）")
    parser.add_argument("--episodes", type=str, default=None, help="只处理这些 episode，逗号分隔（测试用）")
    parser.add_argument("--no-meta", action="store_true", help="写完 episode 文件但不更新 meta（测试用）")
    parser.add_argument("--dry-run", action="store_true", help="只检测并打印切段，不写任何文件")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    marker = root / ".grasp_segments_added"
    if marker.exists() and not args.dry_run:
        raise SystemExit(f"{marker} 存在：这个数据集已经追加过夹取段，拒绝重复追加。")
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
    n_orig = len(episodes)
    _log(f"数据集 {root}：{n_orig} 条 episode，fps={fps}")

    if "total_episodes" in info and int(info["total_episodes"]) != n_orig:
        raise SystemExit("info.json 与 episodes.jsonl 不一致，先检查数据集完整性")

    # 阶段 1：检测所有夹取段
    segments = []
    for ep in episodes:
        idx = int(ep["episode_index"])
        df = pd.read_parquet(_data_path(root, idx), columns=["observation.state", "task_index"])
        g = np.stack(df["observation.state"].to_numpy())[:, 6]
        close_start, close_end, start, end = find_grasp_window(g, args.threshold, args.pre, args.post)
        if args.from_start:
            start = 0
        segments.append({
            "episode": idx,
            "task_index": int(df["task_index"].iloc[0]),
            "close_start": close_start,
            "close_end": close_end,
            "start": start,
            "end": end,
        })

    if args.episodes:
        keep = {int(x) for x in args.episodes.split(",")}
        segments = [s for s in segments if s["episode"] in keep]
        if not segments:
            raise SystemExit("--episodes 没有匹配到任何 episode")

    lengths = [s["end"] - s["start"] for s in segments]
    _log(
        f"检测到 {len(segments)} 条夹取段: 长度 min/median/max="
        f"{min(lengths)}/{int(np.median(lengths))}/{max(lengths)}，"
        f"闭合起点 min/median/max={min(s['close_start'] for s in segments)}/"
        f"{int(np.median([s['close_start'] for s in segments]))}/"
        f"{max(s['close_start'] for s in segments)}"
    )

    if args.dry_run:
        for s in segments[:20]:
            _log(
                f"ep {s['episode']:4d}: close={s['close_start']}..{s['close_end']} "
                f"hold={s['close_end'] - s['close_start']} → clip [{s['start']}, {s['end']}) "
                f"new_episode={n_orig + s['episode']}"
            )
        if len(segments) > 20:
            _log(f"... 其余 {len(segments) - 20} 条略")
        return

    # 阶段 2：并行写文件。新 episode 的全局 index 偏移按顺序累加。
    rng = [s for s in segments]
    offsets = {}
    cursor = int(info["total_frames"])
    for s in sorted(rng, key=lambda x: n_orig + x["episode"]):
        offsets[n_orig + s["episode"]] = cursor
        cursor += s["end"] - s["start"]

    for s in segments:
        for sub in (f"data/chunk{(n_orig + s['episode']) // 1000:03d}",
                    *[f"videos/chunk{(n_orig + s['episode']) // 1000:03d}/observation.images.{c}" for c in CAMERA_KEYS]):
            (root / sub).mkdir(parents=True, exist_ok=True)

    records = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(process_episode, root, s, n_orig + s["episode"],
                        offsets[n_orig + s["episode"]], fps): s
            for s in segments
        }
        for done, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                records.append(fut.result())
            except Exception as exc:
                _log(f"❌ episode {s['episode']} 失败: {exc}")
                raise
            if done % 25 == 0 or done == len(futs):
                rate = done / (time.time() - t0)
                _log(f"[{done}/{len(futs)}] {rate:.2f} ep/s")

    # 切段映射，便于追溯
    map_path = root.parent / f"{root.name}_grasp_segments.jsonl"
    _write_jsonl(map_path, [
        {k: r[k] for k in ("new_episode", "episode", "close_start", "close_end", "start", "end", "length")}
        for r in sorted(records, key=lambda x: x["new_episode"])
    ])
    _log(f"切段映射 → {map_path}")

    if args.no_meta:
        _log("--no-meta：跳过 meta 更新")
        return
    if len(records) != n_orig:
        raise SystemExit(f"只处理了 {len(records)}/{n_orig} 条，拒绝更新 meta（可重跑续传）")
    update_meta(root, n_orig, records, fps, dry_run=False)
    marker.write_text(json.dumps({
        "pre": args.pre, "post": args.post, "threshold": args.threshold,
        "from_start": bool(args.from_start),
        "added_episodes": len(records), "base_episodes": n_orig,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    _log("完成 ✅")


if __name__ == "__main__":
    main()
