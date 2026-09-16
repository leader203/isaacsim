"""Visual compatibility fixes for scene assets.

Applies material overrides for assets whose original MDL materials
do not resolve in Isaac Sim 4.5 / Isaac Lab 0.54.3.

Usage:
    from utils.scene_visual_fixes import apply_glass_door_visual_fix

    result = apply_glass_door_visual_fix(stage, "/World/envs/env_0")
"""

from __future__ import annotations

from typing import Any


DOOR_KEYWORDS = (
    "Door_YS2HVXZVALGD2PTUJQ888888_7qcYe",
    "Door_YS3UEUZVAJ2V4PTUKU888888_mdynP",
)

_GLASS_COLOR = (0.45, 0.70, 0.85)
_GLASS_ROUGHNESS = 0.20
_GLASS_METALLIC = 0.0

_FRAME_COLOR = (0.08, 0.08, 0.10)
_FRAME_ROUGHNESS = 0.35
_FRAME_METALLIC = 0.15


def _create_preview_surface_material(
    stage,
    material_path: str,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float,
):
    """Create or reuse a UsdPreviewSurface material at *material_path*."""
    from pxr import Sdf, Gf, UsdShade

    existing_prim = stage.GetPrimAtPath(material_path)
    if existing_prim.IsValid() and existing_prim.IsA(UsdShade.Material):
        return UsdShade.Material(existing_prim)

    material = UsdShade.Material.Define(stage, material_path)
    shader_path = material_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def apply_glass_door_visual_fix(
    stage,
    env_root: str = "/World/envs/env_0",
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """Bind opaque PreviewSurface materials to glass door meshes.

    Original MDL material bindings in the door look files do not resolve
    to the composed Stage prims in Isaac Sim 4.5 / Isaac Lab 0.54.3.
    This function applies compatible opaque fallback materials so the
    doors are visible while preserving collision geometry.

    Parameters
    ----------
    stage:
        The USD stage.
    env_root:
        Environment root path (e.g. ``"/World/envs/env_0"``).
    enabled:
        If False, return immediately with zero bound meshes.

    Returns
    -------
    dict with keys:
        found_meshes (int), bound_meshes (int),
        door_meshes (list[str]), frame_meshes (list[str]),
        warnings (list[str])
    """
    from pxr import UsdGeom, UsdShade

    result: dict[str, Any] = {
        "found_meshes": 0,
        "bound_meshes": 0,
        "door_meshes": [],
        "frame_meshes": [],
        "warnings": [],
    }

    if not enabled:
        return result

    if stage is None:
        result["warnings"].append("stage is None")
        return result

    env_prim = stage.GetPrimAtPath(env_root)
    if not env_prim.IsValid():
        result["warnings"].append(f"env_root not valid: {env_root}")
        return result

    # Create materials (idempotent — reuses existing if already defined)
    glass_mat_path = f"{env_root}/Looks/GlassDoorVisualFix/Glass"
    frame_mat_path = f"{env_root}/Looks/GlassDoorVisualFix/Frame"

    glass_mat = _create_preview_surface_material(
        stage, glass_mat_path,
        color=_GLASS_COLOR, roughness=_GLASS_ROUGHNESS, metallic=_GLASS_METALLIC,
    )
    frame_mat = _create_preview_surface_material(
        stage, frame_mat_path,
        color=_FRAME_COLOR, roughness=_FRAME_ROUGHNESS, metallic=_FRAME_METALLIC,
    )

    # Find and bind door meshes under env_root only
    door_mesh_prims: list = []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith(env_root):
            continue
        if not any(kw in path for kw in DOOR_KEYWORDS):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        door_mesh_prims.append(prim)

    result["found_meshes"] = len(door_mesh_prims)

    if not door_mesh_prims:
        result["warnings"].append(
            f"No door meshes found under {env_root} matching {DOOR_KEYWORDS}"
        )
        return result

    for prim in door_mesh_prims:
        name = prim.GetName().lower()
        is_frame = "frame" in name
        material = frame_mat if is_frame else glass_mat

        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            UsdShade.Tokens.strongerThanDescendants,
        )

        # Verify binding
        bound_mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if bound_mat is None:
            result["warnings"].append(
                f"ComputeBoundMaterial returned None after bind: {prim.GetPath()}"
            )
        else:
            result["bound_meshes"] += 1

        if is_frame:
            result["frame_meshes"].append(prim.GetPath().pathString)
        else:
            result["door_meshes"].append(prim.GetPath().pathString)

    print(
        f"[GlassDoorFix] {env_root}: found={result['found_meshes']} "
        f"bound={result['bound_meshes']} warnings={len(result['warnings'])}",
        flush=True,
    )

    return result
