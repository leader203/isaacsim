#!/usr/bin/env python
"""auto_collect.py — 自动采集 cube「抓取 → 抬升 → 投放」demo，直到数据集达到目标条数。

严格按这个顺序，**环境重置只发生在两次 SOP 之间**，绝不会插在 SOP 中间：
    while 未达标:
        [若上一轮已结束] reset_env(随机挪方块)     ← 只在上一轮 SOP 有了明确结论之后
        run_sop_once():  grab → local_lift → take_out
        打印本轮结论（✅ 投放成功 / ❌ 没抓到 / ❌ 没入桶）
        计数

为什么不会中途重置：
  - 每次工具调用都轮询到 state 离开 queued/running 才返回，所以 SOP 一定跑完才有下一步；
  - 环境的回合超时（episode_length_s）已经在 env cfg 里关掉（1e6），不会按时间触发 reset；
  - 本脚本内部没有任何计时/超时逻辑去主动重置（只有 invoke 的“调用级”超时，超时也只是记一次失败）。
  - 结果标志位只从 execution_output 读，不会把"抓到了"误判成"没抓到"。

前提：
  - 仿真服务已在跑，且带 --record_dataset（录制在服务端，本脚本不碰 HDF5）。
  - 服务是**含 reset_env 工具**的版本（改完 run 脚本要重启一次服务）。
  - HDF5 recorder 现在每条 demo 结束后关闭文件，所以本脚本可以在服务运行中
    转换/删除 HDF5；下一条 demo 会自动新建文件（从 demo_0 开始）。

用法：
  conda activate isaaclab

  # 持续采集：每满 10 条成功 → 追加到同一个 LeRobot 数据集 → 删除 HDF5
  # 转换完自动继续采集，不需要重新运行命令；Ctrl+C 停止
  python auto_collect.py --count success \
      --convert-lerobot --lerobot-output outputs/cube_grasp_demos_groot

  # 想采到总量 100 条成功再停（仍然每 10 条转换一次）
  python auto_collect.py --count success --target 100 \
      --convert-lerobot --lerobot-output outputs/cube_grasp_demos_groot

  # 只看采样坐标
  python auto_collect.py --start 0 --target 10 --dry-run

注意：
  - 持续模式下 --convert-lerobot 不填 --target 就是一直采集。
  - 每满 --batch-success 条成功（默认 10）自动 append 到同一个 LeRobot 数据集，
    然后删除 HDF5，继续采下一批。
  - 转换期间会暂停采集，转换完成后自动继续；不需要重新输入命令。
  - 转换失败会停止并保留 HDF5；修复后重跑本脚本，会根据 HDF5 里已有成功数继续。
  - --overwrite-lerobot / --keep-hdf5 与持续采集合用会冲突，已禁用。
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import h5py

# 方块随机范围（world 坐标）
DEFAULT_X = (1.32, 2.36)
DEFAULT_Y = (-0.83, -0.38)
DEFAULT_DATASET = "outputs/cube_grasp_demos.hdf5"
DEFAULT_LEROBOT_OUTPUT = "outputs/cube_grasp_demos_groot"
DEFAULT_CONVERTER = (
    Path(__file__).resolve().parent
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet"
    / "curobo_pose_plan_utils/hdf5_to_groot.py"
)


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── HTTP ─────────────────────────────────────────────────────────────────────
class Api:
    def __init__(self, host: str, port: int, poll_s: float = 2.0, timeout_s: float = 1800.0) -> None:
        self.base = f"http://{host}:{port}"
        self.poll_s = poll_s
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.base + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return json.loads(r.read())

    def invoke(self, tool: str, params: dict) -> dict:
        """调用工具并**轮询到终态**才返回（这是"绝不中途重置"的第一道保证）。"""
        req_id = self._post("/tools/invoke", {"tool_name": tool, "parameters": params})["request_id"]
        deadline = time.monotonic() + self.timeout_s
        while True:
            resp = self._get(f"/tools/jobs/{req_id}")
            state = resp.get("state")
            if state not in ("queued", "running"):
                return resp
            if time.monotonic() > deadline:
                raise TimeoutError(f"{tool} 超时（{self.timeout_s:.0f}s），state={state}")
            time.sleep(self.poll_s)


# ── 结果解析 ──────────────────────────────────────────────────────────────────
def _find_flag(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            got = _find_flag(v, key)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _find_flag(v, key)
            if got is not None:
                return got
    return None


def flag(resp: dict, key: str):
    """只认 execution_output（执行阶段的真实结果），读不到才退回递归。

    递归不可靠：tool_output（规划结果）里有同名但值不同的字段，
    曾把 grabbed=False / paper_grasped=True 这种矛盾组合里的前者读走，
    导致把成功的抓取判成失败。
    """
    ex = resp.get("execution_output")
    if isinstance(ex, dict) and key in ex:
        return ex[key]
    return _find_flag(resp, key)


def truthy(v) -> bool:
    return bool(v) if v is not None else False


# ── SOP 单轮 ─────────────────────────────────────────────────────────────────
def run_sop_once(api: Api, idx: int) -> tuple[bool, str]:
    """跑一次完整 SOP：抓取 → 抬升 → 投放。**不重置环境**。

    返回 (是否成功投放, 结论文本)。日志格式对齐 sop.sh，方便直接看任务进展。
    """
    # ── ② 抓取 ──
    _log("② 抓 grasp_test_cube_1 (gripper_prismatic_execute_grab, paper_cleanup_mode=true)")
    r = api.invoke("gripper_prismatic_execute_grab",
                   {"name": f"auto_grab_{idx}", "paper_cleanup_mode": True})
    if not truthy(flag(r, "executed")):
        return False, "② 抓取未执行"
    if not truthy(flag(r, "paper_grasped")):
        return False, "② 未抓到 cube（paper_grasped=false）"
    _log("② ✅ 抓到 cube")

    # ── ③ 原地抬升 ──
    _log("③ 原地抬升 (gripper_prismatic_local_lift)")
    r = api.invoke("gripper_prismatic_local_lift",
                   {"name": f"auto_lift_{idx}", "lift_height_m": 0.30})
    if truthy(flag(r, "executed")):
        _log("③ ✅ 抬升完成")
    else:
        _log("③ ⚠️ 抬升未执行，仍继续投放")

    # ── ④ 投放 ──
    _log("④ 投放垃圾桶 (gripper_prismatic_take_out)")
    r = api.invoke("gripper_prismatic_take_out", {"name": f"auto_drop_{idx}"})
    if not truthy(flag(r, "executed")):
        return False, "④ 投放未执行"
    if not truthy(flag(r, "paper_collected")):
        return False, "④ cube 未进入垃圾桶（paper_collected=false）"
    _log("④ ✅ cube 已入桶")
    return True, "投放成功"


def reset_env(api: Api, x: float, y: float) -> bool:
    """重置环境并挪方块。**只在两次 SOP 之间调用。**"""
    r = api.invoke("reset_env", {"cube_xy": [x, y]})
    return truthy(flag(r, "reset"))


# ── HDF5 → LeRobot 转换/清理 ─────────────────────────────────────────────────
def _count_success_demos(dataset: Path) -> int | None:
    """统计 HDF5 里 success=True 的 demo 数；文件不存在返回 0，读不了返回 None。"""
    if not dataset.is_file():
        return 0
    try:
        with h5py.File(dataset, "r", locking=False) as f:
            return sum(
                1
                for k in f.keys()
                if k.startswith("demo_") and bool(f[k].attrs.get("success", False))
            )
    except Exception as exc:
        _log(f"读取 HDF5 失败: {dataset} ({exc})")
        return None


def convert_and_cleanup(args) -> int:
    """把 HDF5 转成 LeRobot/GROOT；成功后删除 HDF5。失败时保留 HDF5。"""
    dataset = Path(args.dataset).resolve()
    lerobot_out = Path(args.lerobot_output).resolve()

    if not dataset.is_file():
        _log(f"❌ HDF5 不存在，无法转换: {dataset}")
        return 1

    batch_expected = _count_success_demos(dataset)
    if batch_expected is None:
        _log("❌ 无法读取 HDF5，保留文件不删除")
        return 1
    if batch_expected <= 0:
        _log(f"❌ HDF5 里没有 success=True 的 demo: {dataset}")
        return 1

    # 默认 append：已有 LeRobot 数据集就接着往后写；--overwrite-lerobot 才覆盖重来。
    before_episodes = 0
    if lerobot_out.exists() and not args.overwrite_lerobot:
        episodes_file = lerobot_out / "meta" / "episodes.jsonl"
        info_file = lerobot_out / "meta" / "info.json"
        if not episodes_file.is_file() or not info_file.is_file():
            _log(f"❌ LeRobot 输出目录不是有效数据集（缺 meta 文件）: {lerobot_out}")
            return 1
        try:
            with episodes_file.open("r", encoding="utf-8") as fh:
                before_episodes = sum(1 for _ in fh)
        except Exception as exc:
            _log(f"❌ 读取已有 LeRobot episode 数失败: {exc}")
            return 1
        _log(f"已有 LeRobot: {before_episodes} 条，将追加 {batch_expected} 条")
    else:
        _log(f"新建 LeRobot: {lerobot_out}，写入 {batch_expected} 条")

    cmd = [
        sys.executable,
        str(Path(args.converter).resolve()),
        "--input",
        str(dataset),
        "--output",
        str(lerobot_out),
        "--fps",
        str(args.fps),
    ]
    if not args.overwrite_lerobot:
        cmd.append("--append")
    rc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent)).returncode
    if rc != 0:
        _log(f"❌ 转换失败（exit={rc}），保留 HDF5: {dataset}")
        return rc or 1

    episodes_file = lerobot_out / "meta" / "episodes.jsonl"
    try:
        with episodes_file.open("r", encoding="utf-8") as fh:
            n_episodes = sum(1 for _ in fh)
    except Exception as exc:
        _log(f"❌ 读取转换结果失败: {exc}；保留 HDF5")
        return 1
    expected_total = before_episodes + batch_expected
    if n_episodes != expected_total:
        _log(
            f"❌ 转换后 episode 数={n_episodes}，期望 {expected_total}；"
            f"保留 HDF5 不删除（输出目录: {lerobot_out}）"
        )
        return 1

    _log(
        f"✅ LeRobot 转换完成: 新增 {batch_expected} 条，"
        f"总计 {n_episodes} 条 -> {lerobot_out}"
    )

    if args.keep_hdf5:
        _log(f"保留 HDF5: {dataset}")
        return 0
    try:
        size = dataset.stat().st_size
        dataset.unlink()
        _log(f"✅ 已删除 HDF5: {dataset}（释放 {size / 1e9:.1f} GB）")
    except Exception as exc:
        _log(f"⚠️ LeRobot 已生成，但删除 HDF5 失败: {dataset} ({exc})")
        return 1
    return 0


# ── 主循环 ───────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="自动采集 cube 抓取-投放 demo（重置只在两次 SOP 之间）。")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--start", type=int, default=None,
                   help="数据集里已有的成功条数（本次起点）；不填则自动从 HDF5 读取")
    p.add_argument("--target", type=int, default=None,
                   help="总成功条数达到就停；--convert-lerobot 时不填=持续采集，否则默认 110")
    p.add_argument("--batch-success", type=int, default=10,
                   help="每采满 N 条成功就转换并删除 HDF5（仅 --convert-lerobot，默认 10）")
    p.add_argument("--count", choices=("total", "success"), default="success",
                   help="计入目标的口径：total=每次尝试；success=只算成功（默认）")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"服务端 --record_dataset 的 HDF5 路径（默认 {DEFAULT_DATASET}）")
    p.add_argument("--convert-lerobot", action="store_true",
                   help="每满 --batch-success 条成功就追加到 LeRobot，并删除 HDF5，然后继续采集")
    p.add_argument("--lerobot-output", default=DEFAULT_LEROBOT_OUTPUT,
                   help=f"LeRobot 输出目录（默认 {DEFAULT_LEROBOT_OUTPUT}）")
    p.add_argument("--overwrite-lerobot", action="store_true",
                   help="覆盖已有 LeRobot 数据集（与 --convert-lerobot 冲突，保留给兼容）")
    p.add_argument("--keep-hdf5", action="store_true",
                   help="转换成功后保留 HDF5（与 --convert-lerobot 冲突，保留给兼容）")
    p.add_argument("--fps", type=int, default=30, help="LeRobot 视频 fps（默认 30）")
    p.add_argument("--converter", default=str(DEFAULT_CONVERTER),
                   help="hdf5_to_groot.py 路径")
    p.add_argument("--x-range", type=float, nargs=2, default=list(DEFAULT_X), metavar=("XMIN", "XMAX"))
    p.add_argument("--y-range", type=float, nargs=2, default=list(DEFAULT_Y), metavar=("YMIN", "YMAX"))
    p.add_argument("--seed", type=int, default=None, help="随机种子（复现用）")
    p.add_argument("--dry-run", action="store_true", help="只打印采样坐标，不调工具")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.target is None and not args.convert_lerobot:
        args.target = 110

    if args.convert_lerobot and args.count != "success":
        _log("--convert-lerobot 需要 --count success")
        return 2

    if args.convert_lerobot:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            _log("❌ 当前 Python 缺少 pyarrow，LeRobot 转换需要 conda activate isaaclab 后再运行")
            return 2
        if args.keep_hdf5:
            _log("--keep-hdf5 不能和持续采集合用：HDF5 不删会导致下一批重复转换。")
            return 2
        if args.overwrite_lerobot:
            _log("--overwrite-lerobot 不能和持续追加一起用；想重来请先手动删除 LeRobot 目录。")
            return 2
        if args.batch_success < 1:
            _log("--batch-success 必须 >= 1")
            return 2

    if args.start is None:
        n = _count_success_demos(Path(args.dataset))
        if n is None:
            _log("无法自动读取 HDF5 成功条数，请显式传 --start")
            return 2
        args.start = n
        _log(f"自动检测已有成功 demo: {args.start} 条")

    batch_ok = args.start if args.convert_lerobot else 0
    total_ok = args.start

    if args.convert_lerobot:
        target_desc = "持续采集（Ctrl+C 停止）" if args.target is None else f"总计到 {args.target} 条"
        _log(
            f"{target_desc}，每满 {args.batch_success} 条成功自动转换并删除 HDF5；"
            f"当前 HDF5 已有成功 {batch_ok} 条  cube x∈{args.x_range} y∈{args.y_range}"
        )
    else:
        need = args.target - args.start
        if need <= 0:
            _log(f"已有 {args.start} 条 >= 目标 {args.target}，无需采集。")
            return 0
        _log(
            f"目标 {args.target} 条，已有 {args.start} 条，本次要采 {need} 条（口径={args.count}）"
            f"  cube x∈{args.x_range} y∈{args.y_range}"
        )

    if args.dry_run:
        if args.convert_lerobot:
            n_dry = args.batch_success if args.target is None else max(0, args.target - args.start)
        else:
            n_dry = max(0, args.target - args.start)
        for i in range(n_dry):
            print(f"  #{i+1} cube_xy=({random.uniform(*args.x_range):.5f}, "
                  f"{random.uniform(*args.y_range):.5f})")
        return 0

    api = Api(args.host, args.port)
    try:
        names = api._get("/health").get("tools") or []
    except Exception as exc:
        _log(f"❌ 连不上服务 {args.host}:{args.port} —— {exc}")
        return 2
    if "reset_env" not in names:
        _log("⚠️ 服务里没有 reset_env 工具：说明服务是改动前启动的，先重启一次服务再采集。")
        return 2
    _log(f"服务就绪，工具数 {len(names)}")

    attempts = 0
    ok_cnt = 0
    consecutive_fail = 0
    t_start = time.monotonic()

    # HDF5 里已经攒够一批（例如上次被中断）→ 先转换清空，再继续采
    if args.convert_lerobot and batch_ok >= args.batch_success:
        _log(f"当前 HDF5 已有 {batch_ok} 条成功，先转换/追加并清空 HDF5 ...")
        rc = convert_and_cleanup(args)
        if rc != 0:
            _log("❌ 转换失败，停止；HDF5 保留。")
            return rc
        batch_ok = 0
        _log("✅ 已转换并删除 HDF5，继续采集")

    while True:
        if args.target is not None:
            progress = total_ok if args.count == "success" else args.start + attempts
            if progress >= args.target:
                break

        attempts += 1

        # ── 重置环境（上一轮 SOP 已经结束、有了明确结论，才会走到这里）──
        x = random.uniform(*args.x_range)
        y = random.uniform(*args.y_range)
        if args.convert_lerobot:
            _log(
                f"===== 第 {attempts} 轮（总计 {total_ok}，"
                f"当前批次 {batch_ok}/{args.batch_success}）cube_xy=({x:.5f}, {y:.5f}) ====="
            )
        else:
            _log(f"===== 第 {attempts} 轮 cube_xy=({x:.5f}, {y:.5f}) =====")
        _log("① 重置环境 + 把方块挪到采样点 (reset_env)")
        try:
            if not reset_env(api, x, y):
                _log("  ⚠️ reset_env 返回异常，跳过本轮")
                consecutive_fail += 1
                if consecutive_fail >= 3:
                    _log("❌ 连续 3 轮连不上/失败 —— 仿真服务很可能已经死了。"
                         "请检查 `pgrep -af run_curobo_pose_plan_test.py` 并重启服务后重新运行本脚本。")
                    return 3
                continue
        except Exception as exc:
            _log(f"  ❌ reset_env 失败: {exc}")
            consecutive_fail += 1
            if consecutive_fail >= 3:
                _log("❌ 连续 3 轮连不上服务 —— 仿真很可能挂了（被 OOM 杀 / 段错误）。"
                     "先 `pgrep -af run_curobo_pose_plan_test.py` 确认，再重启服务后重跑。")
                return 3
            continue
        consecutive_fail = 0
        _log(f"① ✅ 已重置，方块 → ({x:.5f}, {y:.5f})")

        # ── 跑一轮 SOP（内部绝不重置）──
        try:
            ok, why = run_sop_once(api, attempts)
        except Exception as exc:
            ok, why = False, f"SOP 异常: {exc}"

        # ── 结论 ──
        if ok:
            ok_cnt += 1
            total_ok += 1
            if args.convert_lerobot:
                batch_ok += 1
            _log(f"  ✅ {why}")
        else:
            _log(f"  ❌ {why}")
        _log(f"  本轮结束（{why}）—— 环境将在下一轮开始时重置")

        # ── 满一批就转换/追加 + 删 HDF5，然后继续采集 ──
        if args.convert_lerobot and batch_ok >= args.batch_success:
            _log(f"===== 当前批次已满 {batch_ok} 条，开始转换/追加 LeRobot =====")
            rc = convert_and_cleanup(args)
            if rc != 0:
                _log("❌ 转换失败，停止采集；HDF5 保留，修复后可重跑本脚本继续。")
                return rc
            batch_ok = 0
            _log("✅ 转换完成并删除 HDF5；继续采集下一批")

        if (args.count == "success" and ok_cnt and ok_cnt % 5 == 0) or (
            args.count == "total" and attempts % 5 == 0
        ):
            el = time.monotonic() - t_start
            if args.convert_lerobot:
                _log(
                    f"--- 进度 总计 {total_ok} 当前批次 {batch_ok}/{args.batch_success} "
                    f"成功 {ok_cnt} 失败 {attempts-ok_cnt} 用时 {el/60:.1f}min ---"
                )
            else:
                _log(
                    f"--- 进度 成功 {ok_cnt} 失败 {attempts-ok_cnt} "
                    f"用时 {el/60:.1f}min 均 {(el/attempts)/60:.1f}min/轮 ---"
                )

    # 到达 --target 时，把不足一批的尾部也转换掉（否则会留在 HDF5 里）
    if args.convert_lerobot and batch_ok > 0:
        _log(f"目标已到，转换最后一批 {batch_ok} 条 ...")
        rc = convert_and_cleanup(args)
        if rc != 0:
            _log("采集结束，但尾部转换失败；HDF5 已保留。")
            return rc

    el = time.monotonic() - t_start
    _log("===== 采集结束 =====")
    _log(f"总尝试 {attempts}，成功 {ok_cnt}，失败 {attempts-ok_cnt}，用时 {el/60:.1f} 分钟")
    if args.convert_lerobot:
        _log(f"LeRobot 数据集: {Path(args.lerobot_output).resolve()}")
        if not args.keep_hdf5 and not Path(args.dataset).exists():
            _log("HDF5 已按批删除。")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[auto_collect] 被 Ctrl+C 中断。", flush=True)
        sys.exit(130)
