"""Paper waste state machine for toilet cleaning simulation.

Paper waste objects (wet clumps, flattened tissues) transition through states:
  STUCK → LOOSE → GRASPED/PUSHED → COLLECTED/REMOVED

State transitions are driven by tool proximity, gripper state, and collection-zone checks.
Real soft-body physics is NOT used — the state machine emulates wet-paper adhesion/tearing.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Phase / state constants
# ---------------------------------------------------------------------------

class PaperWastePhase(str, enum.Enum):
    """Discrete phases a paper-waste object can occupy."""

    STUCK = "stuck"        # stuck to floor (emulated adhesion)
    LOOSE = "loose"        # free to be pushed / dragged
    GRASPED = "grasped"    # held by gripper
    PUSHED = "pushed"      # being pushed by brush / tool
    COLLECTED = "collected" # inside collection zone
    REMOVED = "removed"    # pushed out of the cleaning area


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PaperWasteObject:
    """Runtime state for one paper-waste entity in the simulation."""

    prim_path: str
    kind: str = "used_wet_clump"  # "used_wet_clump" | "used_flattened_tissue"

    state: str = PaperWastePhase.LOOSE.value

    initial_pos_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    current_pos_w: tuple[float, float, float] = (0.0, 0.0, 0.0)

    is_collected: bool = False
    is_stuck: bool = False

    contact_steps: int = 0
    last_contact_tool: Optional[str] = None
    cleaned_score: float = 0.0

    # cuRobo integration
    curobo_proxy_path: str = ""

    # grasp tracking
    grasped_by_gripper: bool = False
    attach_offset: tuple[float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0)
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def inside_aabb(
    pos: tuple[float, float, float],
    aabb_min: tuple[float, float, float],
    aabb_max: tuple[float, float, float],
) -> bool:
    """Return True if *pos* is inside the axis-aligned bounding box."""
    return all(
        float(aabb_min[i]) <= float(pos[i]) <= float(aabb_max[i])
        for i in range(3)
    )


# ---------------------------------------------------------------------------
# State-transition logic
# ---------------------------------------------------------------------------

def update_paper_state(
    obj: PaperWasteObject,
    tool_pos_w: tuple[float, float, float],
    gripper_closed: bool,
    collection_aabb: tuple[tuple[float, float, float], tuple[float, float, float]],
    contact_radius: float = 0.08,
    stuck_contact_threshold: int = 20,
    grasp_distance_threshold: float = 0.06,
) -> PaperWasteObject:
    """Advance the paper-waste state machine by one step.

    Parameters
    ----------
    obj:
        The paper-waste object to update (mutated in-place).
    tool_pos_w:
        Current tool_tip_link position in world frame.
    gripper_closed:
        Whether the prismatic gripper is in the closed (grasp) position.
    collection_aabb:
        (aabb_min, aabb_max) in world frame for the collection zone.
    contact_radius:
        Distance below which the tool is considered "in contact" with the paper.
    stuck_contact_threshold:
        Number of consecutive contact steps needed to transition STUCK → LOOSE.
    grasp_distance_threshold:
        Distance below which a closed gripper "captures" the paper.

    Returns
    -------
    The same *obj* (mutated) for chaining convenience.
    """
    aabb_min, aabb_max = collection_aabb

    # --- already collected → nothing to do ---
    if obj.state == PaperWastePhase.COLLECTED.value:
        obj.cleaned_score = 1.0
        obj.is_collected = True
        return obj

    # --- check collection zone first ---
    if inside_aabb(obj.current_pos_w, aabb_min, aabb_max):
        obj.state = PaperWastePhase.COLLECTED.value
        obj.is_collected = True
        obj.cleaned_score = 1.0
        return obj

    # --- distance to tool ---
    import math
    dist = math.sqrt(
        sum(
            (float(obj.current_pos_w[i]) - float(tool_pos_w[i])) ** 2
            for i in range(3)
        )
    )

    if obj.state == PaperWastePhase.STUCK.value:
        if dist < contact_radius:
            obj.contact_steps += 1
            obj.last_contact_tool = "brush"  # or gripper — generic for now
        else:
            obj.contact_steps = max(0, obj.contact_steps - 1)

        if obj.contact_steps >= stuck_contact_threshold:
            obj.state = PaperWastePhase.LOOSE.value
            obj.is_stuck = False

    elif obj.state == PaperWastePhase.LOOSE.value:
        if gripper_closed and dist < grasp_distance_threshold:
            obj.state = PaperWastePhase.GRASPED.value
            obj.grasped_by_gripper = True

    elif obj.state == PaperWastePhase.GRASPED.value:
        if not gripper_closed:
            # gripper opened — check if we're in collection zone
            if inside_aabb(obj.current_pos_w, aabb_min, aabb_max):
                obj.state = PaperWastePhase.COLLECTED.value
                obj.is_collected = True
                obj.cleaned_score = 1.0
            else:
                obj.state = PaperWastePhase.LOOSE.value
                obj.grasped_by_gripper = False
                obj.attach_offset = (0.0, 0.0, 0.0)

    elif obj.state == PaperWastePhase.PUSHED.value:
        if inside_aabb(obj.current_pos_w, aabb_min, aabb_max):
            obj.state = PaperWastePhase.COLLECTED.value
            obj.is_collected = True
            obj.cleaned_score = 1.0

    return obj


def mark_collected(obj: PaperWasteObject) -> None:
    """Force the object into COLLECTED state."""
    obj.state = PaperWastePhase.COLLECTED.value
    obj.is_collected = True
    obj.cleaned_score = 1.0
