"""Metrics and collection-zone checks for paper-waste cleanup tasks."""

from __future__ import annotations

import json
import math
from typing import Optional

from .paper_waste_state import PaperWasteObject, inside_aabb

# ---------------------------------------------------------------------------
# Collection zone — trash bin AABB in world frame
# ---------------------------------------------------------------------------
# Initial estimate around the trash bin position.
# **To be refined after the user confirms the bin location in the GUI.**
TRASH_BIN_COLLECTION_AABB: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (2.097 - 0.12, -1.095 - 0.12, -0.05),  # min [x, y, z]
    (2.097 + 0.12, -1.095 + 0.12, 0.01 + 0.40),   # max [x, y, z]
)


def set_trash_bin_aabb(
    center_xy: tuple[float, float],
    half_extents: tuple[float, float, float] = (0.12, 0.12, 0.40),
) -> None:
    """Update the global trash-bin collection AABB from a center point.

    Parameters
    ----------
    center_xy:
        (x, y) center of the trash bin opening in world frame.
    half_extents:
        Half-extents (dx, dy, dz) in meters.  Default ~12 cm radial + 40 cm tall.
    """
    global TRASH_BIN_COLLECTION_AABB  # noqa: PLW0603
    cx, cy = float(center_xy[0]), float(center_xy[1])
    dx, dy, dz = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])
    # Bin floor is near z=0.01; collection zone starts slightly below and goes up.
    TRASH_BIN_COLLECTION_AABB = (
        (cx - dx, cy - dy, -0.05),
        (cx + dx, cy + dy, 0.01 + dz),
    )


def check_paper_collected(
    paper_pos_w: tuple[float, float, float],
    aabb: Optional[tuple[tuple[float, float, float], tuple[float, float, float]]] = None,
) -> bool:
    """Return True if *paper_pos_w* is inside the collection AABB."""
    _aabb = aabb if aabb is not None else TRASH_BIN_COLLECTION_AABB
    return inside_aabb(paper_pos_w, _aabb[0], _aabb[1])


# ---------------------------------------------------------------------------
# Episode metrics
# ---------------------------------------------------------------------------

def compute_paper_cleanup_metrics(
    paper_obj: PaperWasteObject,
    tool_path_length_m: float = 0.0,
    num_curobo_failures: int = 0,
) -> dict:
    """Produce a JSON-serializable metrics dict for one cleanup episode."""
    initial = paper_obj.initial_pos_w
    final = paper_obj.current_pos_w
    displacement_m = math.sqrt(
        sum((float(final[i]) - float(initial[i])) ** 2 for i in range(3))
    )

    return {
        "paper_kind": paper_obj.kind,
        "initial_pos_w": list(initial),
        "final_pos_w": list(final),
        "state_final": paper_obj.state,
        "success": paper_obj.is_collected,
        "cleaned_score": paper_obj.cleaned_score,
        "contact_steps": paper_obj.contact_steps,
        "grasped": paper_obj.grasped_by_gripper,
        "collected": paper_obj.is_collected,
        "displacement_m": round(displacement_m, 4),
        "tool_path_length_m": round(tool_path_length_m, 4),
        "num_curobo_failures": num_curobo_failures,
    }


def metrics_to_json(metrics: dict) -> str:
    """Serialize metrics dict to a compact JSON string."""
    return json.dumps(metrics, ensure_ascii=False)
