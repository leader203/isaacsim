"""Runtime setup helpers for the cuRobo pose-plan scripts."""

from __future__ import annotations

import datetime
import importlib.util
import os
from pathlib import Path
import sys
import threading


def setup_terminal_log(task_root: str | Path) -> int:
    log_dir = Path(task_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_curobo_pose_plan_test_{timestamp}.log"
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)

    def tee_fd(fd: int) -> None:
        original_fd = os.dup(fd)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, fd)
        os.close(write_fd)

        def pump() -> None:
            while True:
                try:
                    data = os.read(read_fd, 8192)
                except OSError:
                    break
                if not data:
                    break
                os.write(original_fd, data)
                os.write(log_fd, data)

        threading.Thread(target=pump, daemon=True).start()

    tee_fd(1)
    tee_fd(2)
    print(f"[Log] terminal output tee: {log_path}", flush=True)
    return log_fd


def require_curobo_preflight(task_root: str | Path) -> None:
    curobo_src_root = Path(task_root) / "src" / "nvidia-curobo"
    if curobo_src_root.is_dir():
        curobo_src_root_str = str(curobo_src_root)
        if curobo_src_root_str not in sys.path:
            sys.path.insert(0, curobo_src_root_str)
    if importlib.util.find_spec("curobo") is not None:
        return
    raise ModuleNotFoundError(
        "No module named 'curobo'. Install the nvidia-curobo distribution into the current "
        "'isaaclab' conda environment first, or keep the local allstar_toilet/src/nvidia-curobo checkout present."
    )


def regenerate_curobo_robot_cfg_from_urdf(
    urdf_path: str | Path,
    output_path: str | Path,
    *,
    script_dir: str | Path,
    collision_sphere_buffer: float,
    base_link: str,
    tool_frame: str,
    joint_names: list[str],
) -> Path:
    """Refresh cuRobo robot cfg from URDF sphere collisions."""
    script_dir = Path(script_dir)
    allstar_toilet_root = script_dir.parent
    usdandurdf_dir = allstar_toilet_root / "usdandurdf"
    if not usdandurdf_dir.is_dir():
        usdandurdf_dir = Path(__file__).resolve().parents[1] / "usdandurdf"
    usdandurdf_dir_str = str(usdandurdf_dir)
    if usdandurdf_dir_str not in sys.path:
        sys.path.insert(0, usdandurdf_dir_str)

    from generate_curobo_robot_cfg_from_urdf import build_robot_cfg

    urdf_path = Path(urdf_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    robot_cfg = build_robot_cfg(
        urdf_path=urdf_path,
        asset_root_path=urdf_path.parent,
        collision_link_names=None,
        collision_sphere_buffer=collision_sphere_buffer,
        base_link=base_link,
        tool_frame=tool_frame,
        joint_names=joint_names,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(robot_cfg, stream, sort_keys=False)

    collision_spheres = robot_cfg["robot_cfg"]["kinematics"]["collision_spheres"]
    counts = {link_name: len(spheres) for link_name, spheres in collision_spheres.items()}
    print(
        f"[CuroboRobotCfg] regenerated_from_urdf={urdf_path} output={output_path} "
        f"collision_sphere_count={sum(counts.values())} "
        f"collision_sphere_buffer={collision_sphere_buffer:.6f} per_link={counts}",
        flush=True,
    )
    return output_path
