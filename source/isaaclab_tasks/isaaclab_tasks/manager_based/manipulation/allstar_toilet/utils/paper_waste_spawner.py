"""Runtime spawner for paper-waste USD prims (used_wet_paper_clump, etc.).

Creates a visual mesh + simplified collision proxy + rigid-body physics + cuRobo
obstacle proxy, all via the USD API.  Designed to be called from an env_cfg
prestartup EventTerm before simulation begins.
"""

from __future__ import annotations

import math
import random
from typing import Optional


def spawn_used_wet_paper_clump(
    stage,
    prim_path: str,
    position: tuple[float, float, float],
    radius: float = 0.04,
    mass: float = 0.02,
    *,
    make_dynamic: bool = True,
    curobo_proxy: bool = True,
    collision_sphere_count: int = 10,
    seed: int = 42,
) -> str:
    """Create a used-wet-paper-clump rigid body in the USD stage.

    Parameters
    ----------
    stage:
        The USD stage (``omni.usd.get_context().get_stage()``).
    prim_path:
        Full USD path for the paper-waste root Xform, e.g.
        ``"/World/envs/env_0/paper_waste/used_wet_paper_clump_01"``.
    position:
        Initial world position ``(x, y, z)``.
    radius:
        Approximate visual radius in metres (default 0.04 → ~8 cm diameter).
    mass:
        Rigid-body mass in kg (default 0.02).
    make_dynamic:
        If True, apply RigidBodyAPI + MassAPI so the object falls under gravity.
    curobo_proxy:
        If True, also create a simplified cuRobo obstacle proxy Cube as a child
        prim (with ``curobo:obstacle=True``, no CollisionAPI).
    collision_sphere_count:
        Number of child sphere colliders (default 10).
    seed:
        Random seed for reproducible visual shape.

    Returns
    -------
    The *prim_path* of the created root prim.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 1. Root Xform
    # ------------------------------------------------------------------
    root_xform = UsdGeom.Xform.Define(stage, prim_path)
    root_prim = root_xform.GetPrim()
    xf_root = UsdGeom.Xformable(root_prim)
    xf_root.AddTranslateOp().Set(Gf.Vec3d(*position))

    # ------------------------------------------------------------------
    # 2. Visual mesh — wrinkled sphere
    # ------------------------------------------------------------------
    _create_wrinkled_sphere_visual(
        stage, prim_path + "/visual_mesh", radius, rng
    )

    # PreviewSurface material
    _apply_visual_material(
        stage, root_prim,
        diffuse_color=(0.72, 0.67, 0.58),  # warm grey / light brown
        roughness=0.88,
    )

    # ------------------------------------------------------------------
    # 3. Collision spheres (child prims with CollisionAPI)
    # ------------------------------------------------------------------
    for i in range(collision_sphere_count):
        # random offset within the visual radius, biased outward
        theta = rng.uniform(0.0, 2.0 * math.pi)
        phi = math.acos(2.0 * rng.random() - 1.0)
        r = radius * rng.uniform(0.5, 0.95)
        cx = r * math.sin(phi) * math.cos(theta)
        cy = r * math.sin(phi) * math.sin(theta)
        cz = r * math.cos(phi)

        sphere_path = f"{prim_path}/collision_sphere_{i:02d}"
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.CreateRadiusAttr().Set(radius * rng.uniform(0.25, 0.40))
        sp = sphere.GetPrim()
        xf = UsdGeom.Xformable(sp)
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))

        UsdPhysics.CollisionAPI.Apply(sp)

    # ------------------------------------------------------------------
    # 4. Physics
    # ------------------------------------------------------------------
    if make_dynamic:
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        mass_api.CreateMassAttr().Set(mass)

        # PhysX-specific properties
        physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        physx_api.CreateLinearDampingAttr().Set(3.0)
        physx_api.CreateAngularDampingAttr().Set(3.0)
        physx_api.CreateMaxDepenetrationVelocityAttr().Set(1.0)
        physx_api.CreateMaxLinearVelocityAttr().Set(2.0)
        physx_api.CreateMaxAngularVelocityAttr().Set(20.0)
        physx_api.CreateSolverPositionIterationCountAttr().Set(16)
        physx_api.CreateSolverVelocityIterationCountAttr().Set(4)

        # Physics material (high friction)
        mat_api = UsdPhysics.MaterialAPI.Apply(root_prim)
        mat_api.CreateStaticFrictionAttr().Set(1.5)
        mat_api.CreateDynamicFrictionAttr().Set(1.2)
        mat_api.CreateRestitutionAttr().Set(0.0)

    # ------------------------------------------------------------------
    # 5. Custom attributes (semantic labels)
    # ------------------------------------------------------------------
    _set_attr(root_prim, "paper_waste:kind", Sdf.ValueTypeNames.String, "used_wet_clump")
    _set_attr(root_prim, "paper_waste:state", Sdf.ValueTypeNames.String, "loose")
    _set_attr(root_prim, "paper_waste:is_collected", Sdf.ValueTypeNames.Bool, False)
    _set_attr(root_prim, "paper_waste:contact_steps", Sdf.ValueTypeNames.Int, 0)

    # ------------------------------------------------------------------
    # 6. cuRobo proxy (child prim, curobo:obstacle, NO CollisionAPI, invisible)
    # ------------------------------------------------------------------
    proxy_path = ""
    if curobo_proxy:
        proxy_path = f"{prim_path}/curobo_proxy"
        _create_curobo_proxy_cube(stage, proxy_path, radius)

    print(
        f"[PaperWasteSpawner] created {prim_path} "
        f"radius={radius:.3f}m mass={mass:.3f}kg "
        f"proxy={proxy_path or 'none'}",
        flush=True,
    )

    return prim_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_wrinkled_sphere_visual(
    stage,
    mesh_path: str,
    radius: float,
    rng: random.Random,
    subdivisions: int = 2,
) -> None:
    """Generate a low-poly wrinkled sphere mesh via random vertex perturbation."""
    from pxr import Gf, UsdGeom, Vt

    # Start with a regular icosahedron (12 vertices, 20 faces)
    verts, faces = _icosahedron()
    # Subdivide for more detail
    for _ in range(subdivisions):
        verts, faces = _subdivide(verts, faces)

    # Perturb vertices radially to create a "wrinkled" look
    wrinkled = []
    for v in verts:
        r = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if r < 1e-8:
            wrinkled.append(v)
            continue
        n = (v[0] / r, v[1] / r, v[2] / r)
        scale = radius * (0.75 + 0.25 * rng.random())
        wrinkled.append((n[0] * scale, n[1] * scale, n[2] * scale))

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr().Set([Gf.Vec3f(*v) for v in wrinkled])
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr().Set(
        [i for f in faces for i in f]
    )
    mesh.CreateExtentAttr().Set(
        UsdGeom.PointBased(mesh).ComputeExtent(mesh.GetPointsAttr().Get())
    )


def _apply_visual_material(
    stage,
    prim,
    diffuse_color: tuple[float, float, float] = (0.72, 0.67, 0.58),
    roughness: float = 0.88,
) -> None:
    """Bind a simple PreviewSurface material to *prim* and its descendants."""
    from pxr import Gf, Sdf, UsdShade

    mat_path = prim.GetPath().AppendChild("paper_waste_material")
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("preview_surface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*diffuse_color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _create_curobo_proxy_cube(
    stage,
    proxy_path: str,
    radius: float,
) -> None:
    """Create a cuRobo obstacle proxy Cube (child prim, no CollisionAPI, invisible)."""
    from pxr import Gf, Sdf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, proxy_path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()

    # cuRobo attributes
    _set_attr(prim, "curobo:obstacle", Sdf.ValueTypeNames.Bool, True)
    _set_attr(prim, "curobo:kind", Sdf.ValueTypeNames.String, "paper_waste")
    _set_attr(prim, "curobo:source", Sdf.ValueTypeNames.String, "paper_waste_proxy")
    _set_attr(prim, "paper_waste:collected", Sdf.ValueTypeNames.Bool, False)

    # Size: slightly larger than the visual radius
    dim = radius * 2.5  # e.g. 0.10m for 0.04m radius
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, dim * 0.3))
    xf.AddScaleOp().Set(Gf.Vec3d(dim, dim, dim * 0.6))

    # Make invisible — proxy is for cuRobo only, not for rendering
    UsdGeom.Imageable(prim).MakeInvisible()


def _set_attr(prim, name: str, type_name, value):
    """Set or create a custom attribute on *prim*."""
    from pxr import Sdf
    attr = prim.GetAttribute(name)
    if not attr or not attr.IsValid():
        attr = prim.CreateAttribute(name, type_name)
    attr.Set(value)


# ---------------------------------------------------------------------------
# Mesh generators (icosahedron base for wrinkled sphere)
# ---------------------------------------------------------------------------

def _icosahedron():
    """Return (vertices, faces) for a regular icosahedron of radius 1."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    a = 1.0 / math.sqrt(1.0 + phi * phi)
    b = phi * a
    verts = [
        (-a,  b,  0), ( a,  b,  0), (-a, -b,  0), ( a, -b,  0),
        ( 0, -a,  b), ( 0,  a,  b), ( 0, -a, -b), ( 0,  a, -b),
        ( b,  0, -a), ( b,  0,  a), (-b,  0, -a), (-b,  0,  a),
    ]
    faces = [
        (0, 11, 5), (0, 5, 1),  (0, 1, 7),  (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    return verts, faces


def _subdivide(verts, faces):
    """One iteration of Loop-like subdivision: mid-point of each edge → new vertex.

    Returns (new_verts, new_faces).
    """
    edge_map: dict[tuple[int, int], int] = {}
    new_verts = list(verts)

    def _mid(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key not in edge_map:
            v0, v1 = new_verts[i], new_verts[j]
            mx = ((v0[0] + v1[0]) * 0.5, (v0[1] + v1[1]) * 0.5, (v0[2] + v1[2]) * 0.5)
            # Normalise to sphere
            r = math.sqrt(mx[0] ** 2 + mx[1] ** 2 + mx[2] ** 2)
            if r > 1e-8:
                mx = (mx[0] / r, mx[1] / r, mx[2] / r)
            edge_map[key] = len(new_verts)
            new_verts.append(mx)
        return edge_map[key]

    new_faces = []
    for a, b, c in faces:
        ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
        new_faces.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])

    return new_verts, new_faces


# ---------------------------------------------------------------------------
# Flattened tissue spawner
# ---------------------------------------------------------------------------


def _make_wrinkled_plane_visual(
    stage,
    mesh_path: str,
    size: tuple[float, float, float],
    rng: random.Random,
    grid_x: int = 6,
    grid_y: int = 4,
) -> None:
    """Generate a low-poly wrinkled plane (thin sheet with Z perturbations)."""
    from pxr import Gf, UsdGeom, Vt

    sx, sy, sz = size
    half_x, half_y = sx * 0.5, sy * 0.5
    verts = []
    for iy in range(grid_y + 1):
        for ix in range(grid_x + 1):
            vx = -half_x + (sx * ix / grid_x)
            vy = -half_y + (sy * iy / grid_y)
            vz = rng.uniform(-sz * 0.5, sz * 0.5)
            verts.append(Gf.Vec3f(vx, vy, vz))

    faces = []
    stride = grid_x + 1
    for iy in range(grid_y):
        for ix in range(grid_x):
            a = iy * stride + ix
            b = a + 1
            c = a + stride
            d = c + 1
            faces.extend([(a, b, d), (a, d, c)])

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr().Set(verts)
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr().Set([i for f in faces for i in f])
    mesh.CreateExtentAttr().Set(
        UsdGeom.PointBased(mesh).ComputeExtent(mesh.GetPointsAttr().Get())
    )


def spawn_used_flattened_tissue(
    stage,
    prim_path: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float] = (0.14, 0.10, 0.015),
    mass: float = 0.008,
    *,
    initial_state: str = "loose",  # "loose" | "stuck"
    make_dynamic: bool = True,
    curobo_proxy: bool = True,
    visual_z_m: float = 0.003,
    seed: int = 0,
) -> str:
    """Create a used-flattened-tissue (thin wet paper sheet) rigid body.

    Visual / collision / cuRobo-proxy are separated:
    - Visual:       wrinkled plane, ~3mm thick (looks like damp tissue)
    - Collision:    single box, ~15mm thick (stable in PhysX GPU)
    - cuRobo proxy: box, ~50mm thick (generous margin for planner)

    Parameters
    ----------
    stage:
        The USD stage.
    prim_path:
        Full USD path.
    position:
        Initial world position ``(x, y, z)``.
    size:
        XY size ``(sx, sy, sz_collision)`` in metres.  Default 0.14×0.10×0.015.
    mass:
        Rigid-body mass in kg (default 0.008).
    initial_state:
        ``"loose"`` (dynamic) or ``"stuck"`` (static / position-enforced).
    make_dynamic:
        If True, apply RigidBodyAPI + MassAPI.  Ignored when *initial_state* is
        ``"stuck"``.
    curobo_proxy:
        If True, create a cuRobo obstacle proxy Cube child prim.
    visual_z_m:
        Visual wrinkle thickness in metres (default 3mm).
    seed:
        Random seed for reproducible visual shape.

    Returns
    -------
    The *prim_path* of the created root prim.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema, UsdShade

    rng = random.Random(seed if seed != 0 else hash(prim_path) % (2**31))

    is_stuck = initial_state == "stuck"
    is_dynamic = make_dynamic and not is_stuck
    sx, sy, sz_coll = size

    # ------------------------------------------------------------------
    # 1. Root Xform
    # ------------------------------------------------------------------
    root_xform = UsdGeom.Xform.Define(stage, prim_path)
    root_prim = root_xform.GetPrim()
    xf_root = UsdGeom.Xformable(root_prim)
    xf_root.AddTranslateOp().Set(Gf.Vec3d(*position))
    rot_op = xf_root.AddRotateXYZOp()
    rot_op.Set(Gf.Vec3f(0, 0, rng.uniform(0, 360)))

    # ------------------------------------------------------------------
    # 2. Visual — thin wrinkled plane (3mm thick)
    # ------------------------------------------------------------------
    visual_path = prim_path + "/visual_mesh"
    _make_wrinkled_plane_visual(stage, visual_path, (sx, sy, visual_z_m), rng)

    mat_path = prim_path + "/tissue_material"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/preview_surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.65, 0.63, 0.58)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.88)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(root_prim).Bind(mat)

    # ------------------------------------------------------------------
    # 3. Collision — thick box for PhysX stability (12~15mm)
    # ------------------------------------------------------------------
    coll_path = prim_path + "/collision_box"
    coll = UsdGeom.Cube.Define(stage, coll_path)
    coll.CreateSizeAttr(1.0)
    cp = coll.GetPrim()
    cx = UsdGeom.Xformable(cp)
    cx.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz_coll))
    UsdPhysics.CollisionAPI.Apply(cp)

    # ------------------------------------------------------------------
    # 4. Physics
    # ------------------------------------------------------------------
    if is_dynamic:
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        mass_api.CreateMassAttr().Set(mass)

        physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        physx_api.CreateLinearDampingAttr().Set(4.0)
        physx_api.CreateAngularDampingAttr().Set(6.0)
        physx_api.CreateMaxDepenetrationVelocityAttr().Set(0.3)
        physx_api.CreateMaxLinearVelocityAttr().Set(1.0)
        physx_api.CreateSolverPositionIterationCountAttr().Set(16)
        physx_api.CreateSolverVelocityIterationCountAttr().Set(4)
    # else: stuck/static — no RigidBodyAPI, collision only (GPT: avoid GPU PhysX kinematic issues)

    # Physics material (shared)
    phys_mat_api = UsdPhysics.MaterialAPI.Apply(root_prim)
    phys_mat_api.CreateStaticFrictionAttr().Set(2.0)
    phys_mat_api.CreateDynamicFrictionAttr().Set(1.8)
    phys_mat_api.CreateRestitutionAttr().Set(0.0)

    # ------------------------------------------------------------------
    # 5. Custom attributes
    # ------------------------------------------------------------------
    _set_attr(root_prim, "paper_waste:kind", Sdf.ValueTypeNames.String, "used_flattened_tissue")
    _set_attr(root_prim, "paper_waste:state", Sdf.ValueTypeNames.String, initial_state)
    _set_attr(root_prim, "paper_waste:is_collected", Sdf.ValueTypeNames.Bool, False)
    _set_attr(root_prim, "paper_waste:contact_steps", Sdf.ValueTypeNames.Int, 0)
    _set_attr(root_prim, "paper_waste:is_stuck", Sdf.ValueTypeNames.Bool, is_stuck)

    # ------------------------------------------------------------------
    # 6. cuRobo proxy — thicker Z for planner safety (50mm)
    # ------------------------------------------------------------------
    proxy_path = ""
    if curobo_proxy:
        proxy_sz = (sx * 1.15, sy * 1.15, max(0.05, sz_coll * 2.5))
        proxy_path = f"{prim_path}/curobo_proxy"
        proxy_cube = UsdGeom.Cube.Define(stage, proxy_path)
        proxy_cube.CreateSizeAttr(1.0)
        pp = proxy_cube.GetPrim()
        _set_attr(pp, "curobo:obstacle", Sdf.ValueTypeNames.Bool, True)
        _set_attr(pp, "curobo:kind", Sdf.ValueTypeNames.String, "paper_waste")
        _set_attr(pp, "curobo:source", Sdf.ValueTypeNames.String, "paper_waste_proxy")
        _set_attr(pp, "paper_waste:collected", Sdf.ValueTypeNames.Bool, False)
        px = UsdGeom.Xformable(pp)
        px.AddScaleOp().Set(Gf.Vec3f(*proxy_sz))
        UsdGeom.Imageable(pp).MakeInvisible()

    # ------------------------------------------------------------------
    # 7. Debug AABB
    # ------------------------------------------------------------------
    import omni.usd
    _stage_for_aabb = stage or omni.usd.get_context().get_stage()
    _vis_prim = _stage_for_aabb.GetPrimAtPath(visual_path)
    _coll_prim = _stage_for_aabb.GetPrimAtPath(coll_path)
    _vis_aabb = UsdGeom.Imageable(_vis_prim).ComputeWorldBound(0.0, "default") if _vis_prim and _vis_prim.IsValid() else "N/A"
    _coll_aabb = UsdGeom.Imageable(_coll_prim).ComputeWorldBound(0.0, "default") if _coll_prim and _coll_prim.IsValid() else "N/A"
    print(
        f"[PaperWasteSpawner] created {prim_path} "
        f"size=({sx:.3f},{sy:.3f},{sz_coll:.3f}) mass={mass:.3f}kg "
        f"state={initial_state} dynamic={is_dynamic} proxy_z={proxy_sz[2] if curobo_proxy else 0:.4f} "
        f"visual_aabb={_vis_aabb} coll_aabb={_coll_aabb}",
        flush=True,
    )

    return prim_path


# ---------------------------------------------------------------------------
# Scanned towel model spawner (mujoco_scanned_objects)
# ---------------------------------------------------------------------------

import os as _os

_SCANNED_ASSETS_DIR = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "..", "assets", "scanned_objects",
)

_SCANNED_MODEL_META = {
    "dish_towel": {
        "dir": "dish_towel",
        "obj_file": "model.obj",
        "texture_file": "texture.png",
        "native_size": (0.1539, 0.3575, 0.0323),  # metres (X, Y, Z)
        "label": "Now Designs Dish Towel Mojave 18x28",
    },
    "bath_towel": {
        "dir": "bath_towel",
        "obj_file": "model.obj",
        "texture_file": "texture.png",
        "native_size": (0.2630, 0.3562, 0.0882),
        "label": "Provence Bath Towel Royal Blue",
    },
}


def _parse_obj_faces(obj_path: str) -> dict:
    """Parse an OBJ file and return vertices, UVs, face indices.

    Returns
    -------
    dict with keys:
        vertices: list[tuple[float, float, float]]
        uvs: list[tuple[float, float]]
        face_vert_idx: list[tuple[int, int, int]]  0-based vertex indices
        face_uv_idx: list[tuple[int, int, int]]    0-based UV indices (may be None)
    """
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    face_vert_idx: list[tuple[int, int, int]] = []
    face_uv_idx: list[tuple[int, int, int]] = []
    has_uv = False

    with open(obj_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("vt "):
                parts = line.split()
                uvs.append((float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0))
            elif line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                vi = []
                ui = []
                for p in parts:
                    segs = p.split("/")
                    vi.append(int(segs[0]) - 1)  # OBJ is 1-based
                    if len(segs) >= 2 and segs[1]:
                        ui.append(int(segs[1]) - 1)
                        has_uv = True
                    else:
                        ui.append(0)
                # Triangulate: if 3 verts, single tri; if 4, fan triangulation
                if len(vi) == 3:
                    face_vert_idx.append((vi[0], vi[1], vi[2]))
                    face_uv_idx.append((ui[0], ui[1], ui[2]))
                elif len(vi) == 4:
                    face_vert_idx.append((vi[0], vi[1], vi[2]))
                    face_vert_idx.append((vi[0], vi[2], vi[3]))
                    face_uv_idx.append((ui[0], ui[1], ui[2]))
                    face_uv_idx.append((ui[0], ui[2], ui[3]))
                else:
                    # Fan triangulation for N-gon
                    for k in range(1, len(vi) - 1):
                        face_vert_idx.append((vi[0], vi[k], vi[k + 1]))
                        face_uv_idx.append((ui[0], ui[k], ui[k + 1]))

    return {
        "vertices": vertices,
        "uvs": uvs,
        "face_vert_idx": face_vert_idx,
        "face_uv_idx": face_uv_idx if has_uv else None,
    }


def _create_scanned_mesh_visual(
    stage,
    mesh_path: str,
    obj_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Create a USD Mesh prim from a scanned OBJ file.

    The mesh includes UV ``st`` primvar for texture mapping.
    """
    from pxr import Gf, Sdf, UsdGeom, Vt

    data = _parse_obj_faces(obj_path)
    verts = data["vertices"]
    faces = data["face_vert_idx"]
    uvs = data["uvs"]
    face_uvs = data["face_uv_idx"]

    sx, sy, sz = scale

    # Scale vertices
    pts = [Gf.Vec3f(v[0] * sx, v[1] * sy, v[2] * sz) for v in verts]

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr().Set(pts)
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr().Set([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr().Set("none")  # poly mesh, no subdivision
    mesh.CreateExtentAttr().Set(
        UsdGeom.PointBased(mesh).ComputeExtent(mesh.GetPointsAttr().Get())
    )

    # UV primvar (st) — faceVarying, same indexing as face vertices
    if uvs and face_uvs:
        uv_values = []
        uv_indices = []
        for f_uv in face_uvs:
            for uvi in f_uv:
                u, v = uvs[uvi]
                uv_values.append(Gf.Vec2f(u, 1.0 - v))  # flip V for USD convention
                uv_indices.append(len(uv_values) - 1)

        primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st_primvar = primvars_api.CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        st_primvar.Set(uv_values)
        st_primvar.SetIndices(Vt.IntArray(uv_indices))


def _apply_scanned_material(
    stage,
    prim,
    texture_path: str,
    *,
    color_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    roughness: float = 0.85,
) -> None:
    """Bind a solid-colour UsdPreviewSurface material to *prim*.

    Texture connection is deliberately omitted — Isaac Sim 4.5 Hydra
    (Storm) renders black when UsdUVTexture cannot resolve the asset
    path at runtime, and the connection overrides the fallback value.

    Once texture-path resolution is confirmed working we can re-enable
    the UsdUVTexture→PreviewSurface connection.
    """
    from pxr import Gf, Sdf, UsdShade

    mat_path = prim.GetPath().AppendChild("scanned_towel_material")
    mat = UsdShade.Material.Define(stage, mat_path)

    surf_path = mat_path.AppendChild("preview_surface")
    surf_shader = UsdShade.Shader.Define(stage, surf_path)
    surf_shader.CreateIdAttr("UsdPreviewSurface")
    surf_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color_scale)
    )
    surf_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    surf_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    surf_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(surf_shader.ConnectableAPI(), "surface")

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    print(
        f"[PaperWasteSpawner] solid material bound: "
        f"colour=({color_scale[0]:.2f},{color_scale[1]:.2f},{color_scale[2]:.2f}) "
        f"roughness={roughness} → {mat_path}",
        flush=True,
    )


def spawn_scanned_towel(
    stage,
    prim_path: str,
    position: tuple[float, float, float],
    *,
    model_type: str = "dish_towel",  # "dish_towel" | "bath_towel"
    target_size: tuple[float, float] = (0.08, 0.06),  # XY in metres
    thickness_z: float = 0.006,  # visual Z thickness after scaling
    mass: float = 0.008,
    initial_state: str = "loose",  # "loose" | "stuck"
    curobo_proxy: bool = True,
    seed: int = 0,
) -> str:
    """Spawn a scanned towel model as a rigid body in the USD stage.

    Uses the mujoco_scanned_objects mesh as visual, with separate
    collision box and cuRobo proxy (same 3-layer architecture as
    ``spawn_used_flattened_tissue``).

    Parameters
    ----------
    stage:
        The USD stage.
    prim_path:
        Full USD path for the root Xform.
    position:
        Initial world position ``(x, y, z)``.
    model_type:
        ``"dish_towel"`` (Now Designs, 15×36cm native) or
        ``"bath_towel"`` (Provence Bath, 26×36cm native).
    target_size:
        Desired XY size ``(sx, sy)`` in metres.  The native mesh is
        uniformly scaled to fit.  Default 8×6 cm.
    thickness_z:
        Desired visual Z thickness in metres after scaling (default 6mm).
    mass:
        Rigid-body mass in kg.
    initial_state:
        ``"loose"`` or ``"stuck"``.
    curobo_proxy:
        If True, create a cuRobo obstacle proxy Cube child prim.
    seed:
        Random seed for Z-rotation variation.

    Returns
    -------
    The *prim_path* of the created root prim.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
    import random as _random_mod

    rng = _random_mod.Random(seed if seed != 0 else hash(prim_path) % (2**31))

    meta = _SCANNED_MODEL_META[model_type]
    obj_path = _os.path.join(_SCANNED_ASSETS_DIR, meta["dir"], meta["obj_file"])
    texture_path = _os.path.join(_SCANNED_ASSETS_DIR, meta["dir"], meta["texture_file"])

    if not _os.path.isfile(obj_path):
        raise FileNotFoundError(f"Scanned model OBJ not found: {obj_path}")
    if not _os.path.isfile(texture_path):
        texture_path = ""  # spawn without texture if missing

    native_sx, native_sy, native_sz = meta["native_size"]
    target_sx, target_sy = target_size

    # Uniform scale to fit target XY (preserve aspect ratio)
    scale_xy = min(target_sx / native_sx, target_sy / native_sy)
    scale_z = thickness_z / native_sz if native_sz > 0 else scale_xy

    is_stuck = initial_state == "stuck"
    is_dynamic = (not is_stuck)

    # Collision box size (thicker than visual for PhysX stability)
    coll_sz = (target_sx, target_sy, max(0.012, thickness_z * 2.5))

    print(
        f"[PaperWasteSpawner] scanned_towel model={model_type} "
        f"native=({native_sx:.3f},{native_sy:.3f},{native_sz:.3f}) "
        f"scale_xy={scale_xy:.4f} scale_z={scale_z:.4f} "
        f"target_xy=({target_sx:.3f},{target_sy:.3f}) z_thick={thickness_z:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 1. Root Xform
    # ------------------------------------------------------------------
    root_xform = UsdGeom.Xform.Define(stage, prim_path)
    root_prim = root_xform.GetPrim()
    xf_root = UsdGeom.Xformable(root_prim)
    xf_root.AddTranslateOp().Set(Gf.Vec3d(*position))
    rot_op = xf_root.AddRotateXYZOp()
    rot_op.Set(Gf.Vec3f(0, 0, rng.uniform(0, 360)))

    # ------------------------------------------------------------------
    # 2. Visual — scanned mesh with texture
    # ------------------------------------------------------------------
    visual_path = prim_path + "/visual_mesh"
    _create_scanned_mesh_visual(
        stage, visual_path, obj_path,
        scale=(scale_xy, scale_xy, scale_z),
    )

    # Debug: verify mesh geometry
    _vis_prim = stage.GetPrimAtPath(visual_path)
    if _vis_prim and _vis_prim.IsValid():
        _vis_mesh = UsdGeom.Mesh(_vis_prim)
        _pts = _vis_mesh.GetPointsAttr().Get()
        _counts = _vis_mesh.GetFaceVertexCountsAttr().Get()
        print(
            f"[PaperWasteSpawner] visual_mesh debug: "
            f"num_points={len(_pts) if _pts else 0}, "
            f"num_faces={len(_counts) if _counts else 0}",
            flush=True,
        )

    # Apply solid-colour material directly to visual mesh prim
    visual_prim = stage.GetPrimAtPath(visual_path)
    # Distinct colours per model type for visual debugging
    _model_colors = {
        "dish_towel": (0.78, 0.72, 0.62),   # warm beige (used dish towel)
        "bath_towel": (0.55, 0.58, 0.65),   # muted blue-grey (bath towel)
    }
    _color = _model_colors.get(model_type, (0.70, 0.68, 0.60))
    _apply_scanned_material(
        stage, visual_prim, texture_path,
        color_scale=_color,
        roughness=0.88,
    )

    # ------------------------------------------------------------------
    # 3. Collision — box proxy
    # ------------------------------------------------------------------
    coll_path = prim_path + "/collision_box"
    coll = UsdGeom.Cube.Define(stage, coll_path)
    coll.CreateSizeAttr(1.0)
    cp = coll.GetPrim()
    cx = UsdGeom.Xformable(cp)
    cx.AddScaleOp().Set(Gf.Vec3f(*coll_sz))
    UsdPhysics.CollisionAPI.Apply(cp)
    # Hide collision box from rendering (purpose="guide")
    UsdGeom.Imageable(cp).CreatePurposeAttr().Set("guide")

    # ------------------------------------------------------------------
    # 4. Physics
    # ------------------------------------------------------------------
    if is_dynamic:
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        mass_api.CreateMassAttr().Set(mass)

        physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        physx_api.CreateLinearDampingAttr().Set(4.0)
        physx_api.CreateAngularDampingAttr().Set(6.0)
        physx_api.CreateMaxDepenetrationVelocityAttr().Set(0.3)
        physx_api.CreateMaxLinearVelocityAttr().Set(1.0)
        physx_api.CreateSolverPositionIterationCountAttr().Set(16)
        physx_api.CreateSolverVelocityIterationCountAttr().Set(4)

    # Physics material
    phys_mat_api = UsdPhysics.MaterialAPI.Apply(root_prim)
    phys_mat_api.CreateStaticFrictionAttr().Set(2.0)
    phys_mat_api.CreateDynamicFrictionAttr().Set(1.8)
    phys_mat_api.CreateRestitutionAttr().Set(0.0)

    # ------------------------------------------------------------------
    # 5. Custom attributes
    # ------------------------------------------------------------------
    _set_attr(root_prim, "paper_waste:kind", Sdf.ValueTypeNames.String, f"scanned_{model_type}")
    _set_attr(root_prim, "paper_waste:state", Sdf.ValueTypeNames.String, initial_state)
    _set_attr(root_prim, "paper_waste:is_collected", Sdf.ValueTypeNames.Bool, False)
    _set_attr(root_prim, "paper_waste:contact_steps", Sdf.ValueTypeNames.Int, 0)
    _set_attr(root_prim, "paper_waste:is_stuck", Sdf.ValueTypeNames.Bool, is_stuck)
    _set_attr(root_prim, "paper_waste:model_type", Sdf.ValueTypeNames.String, model_type)

    # ------------------------------------------------------------------
    # 6. cuRobo proxy
    # ------------------------------------------------------------------
    proxy_path = ""
    if curobo_proxy:
        proxy_sz = (target_sx * 1.2, target_sy * 1.2, max(0.05, coll_sz[2] * 2.5))
        proxy_path = f"{prim_path}/curobo_proxy"
        proxy_cube = UsdGeom.Cube.Define(stage, proxy_path)
        proxy_cube.CreateSizeAttr(1.0)
        pp = proxy_cube.GetPrim()
        _set_attr(pp, "curobo:obstacle", Sdf.ValueTypeNames.Bool, True)
        _set_attr(pp, "curobo:kind", Sdf.ValueTypeNames.String, "paper_waste")
        _set_attr(pp, "curobo:source", Sdf.ValueTypeNames.String, "paper_waste_proxy")
        _set_attr(pp, "paper_waste:collected", Sdf.ValueTypeNames.Bool, False)
        px = UsdGeom.Xformable(pp)
        px.AddScaleOp().Set(Gf.Vec3f(*proxy_sz))
        UsdGeom.Imageable(pp).MakeInvisible()

    # ------------------------------------------------------------------
    # 7. Debug
    # ------------------------------------------------------------------
    print(
        f"[PaperWasteSpawner] created scanned_towel {prim_path} "
        f"model={model_type} target_xy=({target_sx:.3f},{target_sy:.3f}) "
        f"coll=({coll_sz[0]:.3f},{coll_sz[1]:.3f},{coll_sz[2]:.3f}) "
        f"mass={mass:.3f}kg state={initial_state} dynamic={is_dynamic} "
        f"texture={'yes' if texture_path else 'no'}",
        flush=True,
    )

    return prim_path
