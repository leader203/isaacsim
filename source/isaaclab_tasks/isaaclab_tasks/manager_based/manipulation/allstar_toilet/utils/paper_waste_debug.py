"""Debug logging helpers for paper-waste simulation."""

from __future__ import annotations

from .paper_waste_state import PaperWasteObject


def log_paper_state(obj: PaperWasteObject, label: str = "") -> None:
    """Print one-line paper-waste state summary."""
    pos = obj.current_pos_w
    msg = (
        f"[PaperWaste:{label}] "
        f"prim={obj.prim_path} "
        f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
        f"state={obj.state} "
        f"collected={obj.is_collected} "
        f"stuck={obj.is_stuck} "
        f"grasped={obj.grasped_by_gripper} "
        f"score={obj.cleaned_score}"
    )
    print(msg, flush=True)


def log_grasp_event(
    paper_obj: PaperWasteObject,
    tool_pos_w: tuple[float, float, float],
    gripper_center_w: tuple[float, float, float],
    dist: float,
    threshold: float,
    label: str = "",
) -> None:
    """Print grasp-decision details."""
    print(
        f"[PaperWaste:Grasp:{label}] "
        f"paper=({paper_obj.current_pos_w[0]:.3f},{paper_obj.current_pos_w[1]:.3f},{paper_obj.current_pos_w[2]:.3f}) "
        f"tool=({tool_pos_w[0]:.3f},{tool_pos_w[1]:.3f},{tool_pos_w[2]:.3f}) "
        f"gripper_center=({gripper_center_w[0]:.3f},{gripper_center_w[1]:.3f},{gripper_center_w[2]:.3f}) "
        f"dist={dist:.4f} threshold={threshold:.4f} "
        f"grasped={paper_obj.grasped_by_gripper}",
        flush=True,
    )


def log_curobo_proxy(proxy_path: str, exported: bool, reason: str = "") -> None:
    """Print cuRobo proxy export decision."""
    status = "EXPORTED" if exported else "SKIP"
    print(
        f"[PaperWaste:CuroboProxy] {status} {proxy_path}"
        + (f" reason={reason}" if reason else ""),
        flush=True,
    )
