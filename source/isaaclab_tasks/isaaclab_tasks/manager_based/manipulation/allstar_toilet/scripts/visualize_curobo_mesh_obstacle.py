#!/usr/bin/env python3
"""Visualize a cuRobo-exported mesh obstacle OBJ.

Examples
--------
python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/visualize_curobo_mesh_obstacle.py

python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/visualize_curobo_mesh_obstacle.py \
    --save /tmp/toilet_ceramic_2.png --no-show
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
DEFAULT_OBJ = (
    TASK_DIR
    / "logs"
    / "curobo_collision_obstacles"
    / "curobo_scene_collision_all_20260604_110916.obj"
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize a cuRobo obstacle OBJ mesh.")
    parser.add_argument(
        "--obj",
        type=Path,
        default=DEFAULT_OBJ,
        help=f"OBJ mesh path. Defaults to {DEFAULT_OBJ}",
    )
    parser.add_argument(
        "--viewer",
        choices=("matplotlib", "trimesh"),
        default="matplotlib",
        help="Viewer backend. matplotlib also supports --save.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional PNG output path for the matplotlib view.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window. Useful with --save.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=60000,
        help="Maximum faces drawn by matplotlib. Use <=0 to draw all faces.",
    )
    parser.add_argument(
        "--wireframe",
        action="store_true",
        help="Draw mesh face edges in the matplotlib viewer.",
    )
    parser.add_argument("--azim", type=float, default=-55.0, help="Matplotlib camera azimuth.")
    parser.add_argument("--elev", type=float, default=25.0, help="Matplotlib camera elevation.")
    return parser


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported trimesh load result: {type(loaded)!r}")

    if mesh.vertices.size == 0:
        raise ValueError(f"Mesh has no vertices: {path}")
    return mesh


def print_mesh_summary(path: Path, mesh: trimesh.Trimesh) -> None:
    bounds = mesh.bounds
    extents = mesh.extents
    center = mesh.centroid

    print(f"OBJ: {path}")
    print(f"vertices: {len(mesh.vertices)}")
    print(f"faces: {len(mesh.faces)}")
    print(f"bounds min: {bounds[0]}")
    print(f"bounds max: {bounds[1]}")
    print(f"extents: {extents}")
    print(f"centroid: {center}")


def choose_faces(mesh: trimesh.Trimesh, max_faces: int) -> np.ndarray:
    faces = mesh.faces
    if max_faces <= 0 or len(faces) <= max_faces:
        return faces

    indices = np.linspace(0, len(faces) - 1, num=max_faces, dtype=np.int64)
    print(f"matplotlib: drawing {len(indices)} / {len(faces)} faces")
    return faces[indices]


def set_equal_axes(ax, vertices: np.ndarray) -> None:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float((maxs - mins).max() * 0.5)
    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def visualize_matplotlib(args: argparse.Namespace, mesh: trimesh.Trimesh) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = np.asarray(mesh.vertices)
    faces = choose_faces(mesh, args.max_faces)
    triangles = vertices[faces]

    z = triangles.mean(axis=1)[:, 2]
    z_span = np.ptp(z)
    colors = plt.cm.viridis((z - z.min()) / z_span) if z_span > 0.0 else plt.cm.viridis(np.zeros_like(z))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(
        triangles,
        facecolors=colors,
        edgecolors=(0.08, 0.08, 0.08, 0.35) if args.wireframe else "none",
        linewidths=0.15 if args.wireframe else 0.0,
        alpha=0.95,
    )
    ax.add_collection3d(collection)

    set_equal_axes(ax, vertices)
    ax.set_xlabel("X (arm base frame)")
    ax.set_ylabel("Y (arm base frame)")
    ax.set_zlabel("Z (arm base frame)")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_title(args.obj.name)
    fig.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=180)
        print(f"saved: {args.save}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = make_parser().parse_args()
    obj_path = args.obj.expanduser().resolve()
    if not obj_path.is_file():
        raise FileNotFoundError(f"OBJ not found: {obj_path}")

    mesh = load_mesh(obj_path)
    print_mesh_summary(obj_path, mesh)

    if args.viewer == "trimesh":
        mesh.show()
    else:
        args.obj = obj_path
        visualize_matplotlib(args, mesh)


if __name__ == "__main__":
    main()
