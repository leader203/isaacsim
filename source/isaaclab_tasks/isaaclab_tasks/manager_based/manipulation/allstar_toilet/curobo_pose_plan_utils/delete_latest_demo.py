#!/usr/bin/env python
"""delete_latest_demo.py — 删除 HDF5 数据集里最后 N 条 demo（默认最新 1 条）。

用法（仓库根目录，conda activate isaaclab）：
  python <此脚本>                              # 删最新 1 条，默认 outputs/cube_grasp_demos.hdf5
  python <此脚本> --n 2                        # 删最新 2 条
  python <此脚本> --dry-run                    # 只打印会删哪条，不真删（强烈建议先跑一次）
  python <此脚本> --dataset /path/to/x.hdf5    # 指定数据集

注意：
  - 删除不可逆。录制进行中不要运行（先确认没有仿真进程在写该文件）。
  - HDF5 只在被删 demo 位于文件尾部时才回收磁盘空间；删中间条文件不会变小。
  - 记录器按"第一个空缺索引"命名，删掉尾部后下次录制会复用被删的槽位。
"""

import argparse
import os

import h5py


def _demo_index(key: str) -> int:
    return int(key.rsplit("_", 1)[1])


def _list_demos(f: h5py.File) -> list[str]:
    return sorted((k for k in f.keys() if k.startswith("demo_")), key=_demo_index)


def _next_free_slot(demos: list[str]) -> int:
    """记录器 begin_demo 的槽位规则：从 0 起第一个不存在的 demo_i。"""
    have = set(demos)
    i = 0
    while f"demo_{i}" in have:
        i += 1
    return i


def main() -> None:
    parser = argparse.ArgumentParser(description="删除 HDF5 数据集里最后 N 条 demo。")
    parser.add_argument(
        "--dataset",
        default="outputs/cube_grasp_demos.hdf5",
        help="HDF5 数据集路径（默认 outputs/cube_grasp_demos.hdf5）",
    )
    parser.add_argument("--n", type=int, default=1, help="删除最后 N 条（默认 1）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将删除的 demo，不实际删除")
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        raise SystemExit(f"文件不存在: {args.dataset}")
    if args.n < 1:
        raise SystemExit("--n 必须 >= 1")

    mode = "r" if args.dry_run else "a"
    with h5py.File(args.dataset, mode) as f:
        demos = _list_demos(f)
        if not demos:
            raise SystemExit(f"{args.dataset} 里没有 demo_*")

        targets = demos[-args.n :]
        if len(targets) < args.n:
            print(f"[warn] 只有 {len(demos)} 条 demo，少于请求的 {args.n}，将全部删除", flush=True)

        print(f"数据集: {args.dataset}")
        print(f"现有 {len(demos)} 条: {demos}")
        for k in targets:
            a = dict(f[k].attrs)
            print(
                f"  待删 {k}: success={a.get('success')} grab={a.get('grab_success')} "
                f"collect={a.get('collect_success')} n_frames={a.get('n_frames')}"
            )

        if args.dry_run:
            print("[dry-run] 未删除任何数据")
            return

        for k in targets:
            del f[k]
        f.flush()

        remaining = _list_demos(f)
        print(f"已删除 {len(targets)} 条: {targets}")
        print(f"剩余 {len(remaining)} 条: {remaining}")
        print(f"下次录制将复用槽位: demo_{_next_free_slot(remaining)}")


if __name__ == "__main__":
    main()
