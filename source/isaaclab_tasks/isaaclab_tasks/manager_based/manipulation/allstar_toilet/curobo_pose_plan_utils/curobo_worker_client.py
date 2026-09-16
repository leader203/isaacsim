from __future__ import annotations

from dataclasses import dataclass
import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import torch


@dataclass(frozen=True)
class WorkerClientConfig:
    timeout_s: float
    poll_interval_s: float
    heartbeat_s: float
    headless: bool


def worker_client_config_from_args(args_cli) -> WorkerClientConfig:
    return WorkerClientConfig(
        timeout_s=float(args_cli.worker_timeout_s),
        poll_interval_s=float(args_cli.worker_poll_interval_s),
        heartbeat_s=float(args_cli.worker_heartbeat_s),
        headless=bool(args_cli.headless),
    )


def write_worker_payload_debug(payload: dict, *, label: str, task_root: Path) -> Path:
    debug_payload_dir = task_root / "logs" / "curobo_worker_payloads"
    debug_payload_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    debug_payload_path = debug_payload_dir / f"{safe_label}_{timestamp}.json"
    with debug_payload_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return debug_payload_path


def _payload_debug_name(payload: dict) -> str:
    if payload.get("pose_tests"):
        payload_name = str(payload.get("pose_tests", [{}])[0].get("name", "pose_plan"))
    elif payload.get("cspace_targets"):
        payload_name = str(payload.get("cspace_targets", [{}])[0].get("name", "cspace_plan"))
    else:
        payload_name = "pose_plan"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in payload_name)


# 常驻 cuRobo worker（--serve）：进程启动一次，warmup 一次，循环处理多个 payload，
# 省掉每次冷启动里 MotionPlanner 创建 + warmup（实测约 6-10s）的开销。
# 通过环境变量 CUROBO_WORKER_SERVE=1（默认开启）启用；1 关闭时回退到原有单次进程模式。
_serve_worker_process: subprocess.Popen | None = None
_serve_worker_script: Path | None = None
_serve_worker_lock = threading.Lock()


def _serve_enabled() -> bool:
    import os

    return os.environ.get("CUROBO_WORKER_SERVE", "1") != "0"


def shutdown_serve_worker() -> None:
    """Kill the persistent serve worker if any (call on app exit to free GPU)."""
    global _serve_worker_process, _serve_worker_script
    with _serve_worker_lock:
        if _serve_worker_process is not None:
            try:
                _serve_worker_process.kill()
            except Exception:  # noqa: BLE001
                pass
            _serve_worker_process = None
        _serve_worker_script = None
    print("[cuRoboWorker] serve worker shut down", flush=True)


def _ensure_serve_worker(worker_script: Path, cfg: WorkerClientConfig) -> subprocess.Popen:
    """Start (or reuse) a persistent `--serve` worker subprocess."""
    global _serve_worker_process, _serve_worker_script
    with _serve_worker_lock:
        if _serve_worker_process is not None and _serve_worker_process.poll() is None:
            if _serve_worker_script == worker_script:
                return _serve_worker_process
            _serve_worker_process.kill()
        _log_worker("[cuRoboWorker] starting persistent --serve worker")
        _serve_worker_process = subprocess.Popen(
            [sys.executable, str(worker_script), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        _serve_worker_script = worker_script
        return _serve_worker_process


def _log_worker(msg: str) -> None:
    print(msg, flush=True)


def _serve_send_payload(proc: subprocess.Popen, input_path: str) -> None:
    assert proc.stdin is not None
    proc.stdin.write(input_path + "\n")
    proc.stdin.flush()


def _serve_read_result(
    proc: subprocess.Popen,
    *,
    cfg: WorkerClientConfig,
    env=None,
    robot=None,
    hold_q: torch.Tensor | None = None,
    hold_gripper_joint1_pos: float | None = None,
    build_action_fn,
    hold_base_stationary_fn,
    arm_joint_ids_fn,
    set_arm_effort_target_fn,
    step_fn,
    show_camera_windows_fn,
    debug_payload_path,
) -> dict:
    """Read one results JSON line from the serve worker stdout while holding the arm.

    The serve worker writes exactly one `{"results": [...]}` JSON line per payload.
    Returns the parsed payload (the full line dict).
    """
    assert proc.stdout is not None
    assert proc.stderr is not None
    start_time = time.perf_counter()
    last_heartbeat = start_time
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def pump(stream, lines, prefix: str) -> None:
        assert stream is not None
        for line in stream:
            lines.append(line)
            print(f"{prefix}{line}", end="", flush=True)

    import threading as _threading

    stdout_thread = _threading.Thread(target=pump, args=(proc.stdout, stdout_lines, "[cuRoboWorker][stdout] "), daemon=True)
    stderr_thread = _threading.Thread(target=pump, args=(proc.stderr, stderr_lines, "[cuRoboWorker][stderr] "), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    hold_q_fixed = None
    hold_start_actual_q = None
    hold_last_actual_q = None
    hold_step_count = 0
    if env is not None and robot is not None and hold_q is not None:
        hold_q_fixed = hold_q.detach().clone().to(device=robot.device, dtype=robot.data.joint_pos.dtype).reshape(-1)
        hold_base_stationary_fn(robot)
        arm_ids = arm_joint_ids_fn(robot)
        robot.set_joint_position_target(hold_q_fixed.reshape(1, -1), joint_ids=arm_ids)
        robot.set_joint_velocity_target(
            torch.zeros((1, len(arm_ids)), device=robot.device, dtype=robot.data.joint_pos.dtype),
            joint_ids=arm_ids,
        )
        hold_start_actual_q = robot.data.joint_pos[0, arm_ids].detach().clone()
        hold_last_actual_q = hold_start_actual_q

    result_payload: dict | None = None
    while True:
        # 从已读 stdout 行中找含 "results" 的 JSON（serve worker 每 payload 一行）
        for line in stdout_lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "results" in parsed:
                result_payload = parsed
                break
        if result_payload is not None:
            break

        if proc.poll() is not None:
            # serve worker 意外退出
            stdout_thread.join(timeout=2.0)
            stderr_thread.join(timeout=2.0)
            raise RuntimeError(
                "cuRobo serve worker exited unexpectedly:\n"
                f"stdout:\n{''.join(stdout_lines)}\n"
                f"stderr:\n{''.join(stderr_lines)}"
            )
        elapsed_s = time.perf_counter() - start_time
        if elapsed_s > cfg.timeout_s:
            raise RuntimeError(
                f"cuRobo serve worker timed out after {cfg.timeout_s:.1f}s "
                f"payload={debug_payload_path}.\n"
                f"stdout:\n{''.join(stdout_lines)}\n"
                f"stderr:\n{''.join(stderr_lines)}"
            )
        poll_interval_s = max(0.0, cfg.poll_interval_s)
        poll_deadline = time.perf_counter() + poll_interval_s
        stepped_this_poll = False
        max_hold_steps = 1
        if env is not None and robot is not None:
            physics_dt = max(1e-6, float(getattr(env, "physics_dt", 1.0 / 120.0)))
            max_hold_steps = max(1, int(math.ceil(max(poll_interval_s, physics_dt) / physics_dt)))
        for _ in range(max_hold_steps):
            if env is not None and robot is not None:
                hold_base_stationary_fn(robot)
                if hold_q_fixed is not None:
                    set_arm_effort_target_fn(robot)
                    env.step(build_action_fn(robot, hold_q_fixed, gripper_joint1_pos=hold_gripper_joint1_pos))
                    hold_step_count += 1
                    hold_last_actual_q = robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach().clone()
                else:
                    step_fn(env, render=not cfg.headless)
                stepped_this_poll = True
            if time.perf_counter() >= poll_deadline:
                break
        if env is None or robot is None:
            time.sleep(min(0.01, max(0.0, poll_deadline - time.perf_counter())))
            stepped_this_poll = True
        if env is not None and robot is not None:
            show_camera_windows_fn(env)
        now = time.perf_counter()
        if now - last_heartbeat >= max(0.1, cfg.heartbeat_s):
            hold_debug = ""
            if hold_q_fixed is not None and hold_start_actual_q is not None and hold_last_actual_q is not None:
                err_norm, err_max, err = joint_error_fn(hold_q_fixed, hold_last_actual_q)
                hold_debug = (
                    f" hold_steps={hold_step_count} "
                    f"hold_err_norm={err_norm.item():.6f} hold_err_max={err_max.item():.6f}"
                )
            print(
                f"[cuRoboWorker] serve waiting elapsed_s={elapsed_s:.1f} "
                f"stdout_lines={len(stdout_lines)} stderr_lines={len(stderr_lines)} "
                f"payload={debug_payload_path}{hold_debug}",
                flush=True,
            )
            last_heartbeat = now
        if not stepped_this_poll:
            time.sleep(0.001)

    stdout_thread.join(timeout=2.0)
    stderr_thread.join(timeout=2.0)
    return result_payload


def run_pose_plan_worker(
    payload: dict,
    *,
    worker_script: Path,
    task_root: Path,
    cfg: WorkerClientConfig,
    env=None,
    robot=None,
    hold_q: torch.Tensor | None = None,
    hold_gripper_joint1_pos: float | None = None,
    build_action_fn,
    hold_base_stationary_fn,
    arm_joint_ids_fn,
    set_arm_effort_target_fn,
    joint_error_fn,
    format_tensor_fn,
    arm_drive_debug_suffix_fn,
    arm_control_debug_suffix_fn,
    step_fn,
    show_camera_windows_fn,
) -> dict:
    if not worker_script.is_file():
        raise FileNotFoundError(f"Missing worker script: {worker_script}")

    # 常驻 --serve 模式：复用 warmup 后的 worker 进程，省掉每次冷启动。
    if _serve_enabled():
        try:
            proc = _ensure_serve_worker(worker_script, cfg)
            payload_name = _payload_debug_name(payload)
            debug_payload_path = write_worker_payload_debug(payload, label=payload_name, task_root=task_root)
            input_path = str(debug_payload_path)
            _serve_send_payload(proc, input_path)
            return _serve_read_result(
                proc,
                cfg=cfg,
                env=env,
                robot=robot,
                hold_q=hold_q,
                hold_gripper_joint1_pos=hold_gripper_joint1_pos,
                build_action_fn=build_action_fn,
                hold_base_stationary_fn=hold_base_stationary_fn,
                arm_joint_ids_fn=arm_joint_ids_fn,
                set_arm_effort_target_fn=set_arm_effort_target_fn,
                step_fn=step_fn,
                show_camera_windows_fn=show_camera_windows_fn,
                debug_payload_path=debug_payload_path,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[cuRoboWorker] serve mode failed ({exc}); falling back to single-shot", flush=True)
            # 清理坏掉的常驻进程，回退单次
            global _serve_worker_process
            with _serve_worker_lock:
                if _serve_worker_process is not None:
                    try:
                        _serve_worker_process.kill()
                    except Exception:  # noqa: BLE001
                        pass
                    _serve_worker_process = None

    payload_name = _payload_debug_name(payload)
    debug_payload_path = write_worker_payload_debug(payload, label=payload_name, task_root=task_root)
    input_path = str(debug_payload_path)
    print(f"[cuRoboWorker] payload_debug_path={debug_payload_path}", flush=True)

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    return_code = -1
    try:
        hold_q_fixed = None
        hold_start_actual_q = None
        hold_last_actual_q = None
        hold_step_count = 0
        hold_debug_name = payload_name
        if env is not None and robot is not None and hold_q is not None:
            hold_q_fixed = hold_q.detach().clone().to(device=robot.device, dtype=robot.data.joint_pos.dtype).reshape(-1)
            hold_action = build_action_fn(robot, hold_q_fixed, gripper_joint1_pos=hold_gripper_joint1_pos)
            hold_base_stationary_fn(robot)
            arm_ids = arm_joint_ids_fn(robot)
            robot.set_joint_position_target(hold_q_fixed.reshape(1, -1), joint_ids=arm_ids)
            robot.set_joint_velocity_target(
                torch.zeros((1, len(arm_ids)), device=robot.device, dtype=robot.data.joint_pos.dtype),
                joint_ids=arm_ids,
            )
            hold_effort_bias = torch.zeros(
                (1, len(arm_ids)),
                device=robot.device,
                dtype=robot.data.joint_pos.dtype,
            )
            set_arm_effort_target_fn(robot, hold_effort_bias)
            env.scene.write_data_to_sim()
            hold_start_actual_q = robot.data.joint_pos[0, arm_ids].detach().clone()
            hold_last_actual_q = hold_start_actual_q
            err_norm, err_max, err = joint_error_fn(hold_q_fixed, hold_start_actual_q)
            print(
                f"[cuRoboWorkerHold] event=start name={hold_debug_name} "
                f"target_q={format_tensor_fn(hold_q_fixed)} actual_q={format_tensor_fn(hold_start_actual_q)} "
                f"joint_err_norm={err_norm.item():.6f} joint_err_max={err_max.item():.6f} "
                f"joint_err={format_tensor_fn(err)} gripper_joint1={hold_gripper_joint1_pos} "
                f"hold_effort_bias={format_tensor_fn(hold_effort_bias.reshape(-1))}"
                f"{arm_drive_debug_suffix_fn(robot)}"
                f"{arm_control_debug_suffix_fn(robot, target_q=hold_q_fixed, action=hold_action)}",
                flush=True,
            )

        process = subprocess.Popen(
            [sys.executable, str(worker_script), input_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def pump_stream(stream, lines: list[str], prefix: str) -> None:
            assert stream is not None
            for line in stream:
                lines.append(line)
                print(f"{prefix}{line}", end="", flush=True)

        stdout_thread = threading.Thread(
            target=pump_stream,
            args=(process.stdout, stdout_lines, "[cuRoboWorker][stdout] "),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=pump_stream,
            args=(process.stderr, stderr_lines, "[cuRoboWorker][stderr] "),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        start_time = time.perf_counter()
        last_heartbeat = start_time
        while True:
            return_code = process.poll()
            if return_code is not None:
                break
            elapsed_s = time.perf_counter() - start_time
            if elapsed_s > cfg.timeout_s:
                process.kill()
                stdout_thread.join(timeout=2.0)
                stderr_thread.join(timeout=2.0)
                raise RuntimeError(
                    f"cuRobo worker timed out after {cfg.timeout_s:.1f}s "
                    f"payload={debug_payload_path}.\n"
                    f"stdout:\n{''.join(stdout_lines)}\n"
                    f"stderr:\n{''.join(stderr_lines)}"
                )
            poll_interval_s = max(0.0, cfg.poll_interval_s)
            poll_deadline = time.perf_counter() + poll_interval_s
            stepped_this_poll = False
            max_hold_steps = 1
            if env is not None and robot is not None:
                physics_dt = max(1e-6, float(getattr(env, "physics_dt", 1.0 / 120.0)))
                max_hold_steps = max(1, int(math.ceil(max(poll_interval_s, physics_dt) / physics_dt)))
            for _ in range(max_hold_steps):
                if env is not None and robot is not None:
                    hold_base_stationary_fn(robot)
                    if hold_q_fixed is not None:
                        set_arm_effort_target_fn(robot)
                        env.step(build_action_fn(robot, hold_q_fixed, gripper_joint1_pos=hold_gripper_joint1_pos))
                        hold_step_count += 1
                        hold_last_actual_q = robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach().clone()
                    else:
                        step_fn(env, render=not cfg.headless)
                    stepped_this_poll = True
                if process.poll() is not None:
                    return_code = process.poll()
                    break
                if time.perf_counter() >= poll_deadline:
                    break
            if env is None or robot is None:
                time.sleep(min(0.01, max(0.0, poll_deadline - time.perf_counter())))
                stepped_this_poll = True
            if env is not None and robot is not None:
                show_camera_windows_fn(env)
            now = time.perf_counter()
            if now - last_heartbeat >= max(0.1, cfg.heartbeat_s):
                hold_debug = ""
                if hold_q_fixed is not None and hold_start_actual_q is not None and hold_last_actual_q is not None:
                    err_norm, err_max, err = joint_error_fn(hold_q_fixed, hold_last_actual_q)
                    drift_norm, drift_max, drift = joint_error_fn(hold_start_actual_q, hold_last_actual_q)
                    hold_debug = (
                        f" hold_steps={hold_step_count} "
                        f"hold_err_norm={err_norm.item():.6f} hold_err_max={err_max.item():.6f} "
                        f"drift_norm={drift_norm.item():.6f} drift_max={drift_max.item():.6f} "
                        f"actual_q={format_tensor_fn(hold_last_actual_q)} "
                        f"hold_err={format_tensor_fn(err)} drift={format_tensor_fn(drift)}"
                        f"{arm_drive_debug_suffix_fn(robot)}"
                        f"{arm_control_debug_suffix_fn(robot, target_q=hold_q_fixed)}"
                    )
                print(
                    f"[cuRoboWorker] waiting elapsed_s={elapsed_s:.1f} "
                    f"stdout_lines={len(stdout_lines)} stderr_lines={len(stderr_lines)} "
                    f"payload={debug_payload_path}{hold_debug}",
                    flush=True,
                )
                last_heartbeat = now
            if return_code is not None:
                break
            if not stepped_this_poll:
                time.sleep(0.001)

        stdout_thread.join(timeout=2.0)
        stderr_thread.join(timeout=2.0)
        if hold_q_fixed is not None and hold_start_actual_q is not None:
            hold_end_actual_q = robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach().clone()
            err_norm, err_max, err = joint_error_fn(hold_q_fixed, hold_end_actual_q)
            drift_norm, drift_max, drift = joint_error_fn(hold_start_actual_q, hold_end_actual_q)
            print(
                f"[cuRoboWorkerHold] event=end name={hold_debug_name} return_code={return_code} "
                f"elapsed_s={time.perf_counter() - start_time:.3f} hold_steps={hold_step_count} "
                f"target_q={format_tensor_fn(hold_q_fixed)} actual_q={format_tensor_fn(hold_end_actual_q)} "
                f"joint_err_norm={err_norm.item():.6f} joint_err_max={err_max.item():.6f} "
                f"drift_norm={drift_norm.item():.6f} drift_max={drift_max.item():.6f} "
                f"joint_err={format_tensor_fn(err)} drift={format_tensor_fn(drift)} "
                f"gripper_joint1={hold_gripper_joint1_pos}"
                f"{arm_drive_debug_suffix_fn(robot)}"
                f"{arm_control_debug_suffix_fn(robot, target_q=hold_q_fixed)}",
                flush=True,
            )
    finally:
        # input_path now uses debug_payload_path (persistent log dir), no cleanup needed
        pass

    stdout_text = "".join(stdout_lines)
    stderr_text = "".join(stderr_lines)

    if return_code != 0:
        raise RuntimeError(
            "cuRobo worker failed:\n"
            f"stdout:\n{stdout_text}\n"
            f"stderr:\n{stderr_text}"
        )
    parsed_stdout_lines = [line for line in stdout_text.splitlines() if line.strip()]
    if not parsed_stdout_lines:
        raise RuntimeError(f"cuRobo worker produced no stdout. stderr:\n{stderr_text}")
    result_payload = None
    for line in parsed_stdout_lines:
        try:
            parsed_payload = json.loads(line)
        except json.JSONDecodeError:
            print(f"[cuRoboWorker][stdout] {line}", flush=True)
            continue
        if "worker_info" in parsed_payload:
            print(f"[cuRoboWorker][info] {parsed_payload['worker_info']}", flush=True)
        if "results" in parsed_payload:
            result_payload = parsed_payload
    if result_payload is None:
        raise RuntimeError(
            "cuRobo worker did not produce a results payload.\n"
            f"stdout:\n{stdout_text}\n"
            f"stderr:\n{stderr_text}"
        )
    return result_payload
