"""Thin entrypoint for the isolated cuRobo worker."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_worker_module_on_path() -> None:
    module_root = Path(__file__).resolve().parents[1] / "curobo_pose_plan"
    module_root_str = str(module_root)
    if module_root_str not in sys.path:
        sys.path.insert(0, module_root_str)


_ensure_worker_module_on_path()

from worker_planner import main  # noqa: E402


if __name__ == "__main__":
    main()
