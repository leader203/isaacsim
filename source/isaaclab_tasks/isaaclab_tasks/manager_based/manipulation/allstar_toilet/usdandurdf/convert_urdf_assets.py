"""Convert pub_srdf_model URDF files to USD format for IsaacLab.

Usage
-----
# From the IsaacLab root, run via the isaaclab.sh launcher:
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/usdandurdf/convert_urdf_assets.py

# Override output directory:
./isaaclab.sh -p .../convert_urdf_assets.py --output_dir /tmp/usd_out

# Convert a single custom URDF:
./isaaclab.sh -p .../convert_urdf_assets.py --urdf /path/to/robot.urdf --output_dir /tmp/usd_out
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batch-convert pub_srdf_model URDFs to USD.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help=(
        "Directory to write USD files. "
        "Defaults to <this_script_dir>/../data/robots/xms5/"
    ),
)
parser.add_argument(
    "--urdf",
    type=str,
    default=None,
    help="Path to a single URDF to convert. If omitted, all default URDFs are converted.",
)
parser.add_argument(
    "--fix_base",
    action="store_true",
    default=True,
    help="Fix the robot base at the import origin (default: True for fixed-base arms).",
)
parser.add_argument(
    "--merge_joints",
    action="store_true",
    default=False,
    help="Merge links connected by fixed joints.",
)
parser.add_argument(
    "--joint_stiffness",
    type=float,
    default=80.0,
    help="Joint drive stiffness (default: 80.0).",
)
parser.add_argument(
    "--joint_damping",
    type=float,
    default=4.0,
    help="Joint drive damping (default: 4.0).",
)
parser.add_argument(
    "--joint_target_type",
    type=str,
    default="position",
    choices=["position", "velocity", "none"],
    help="Joint drive target type (default: position).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # no GUI needed for batch conversion

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Everything below runs after Isaac Sim is up ───────────────────────────────

import os  # noqa: E402 (re-import after sim launch is fine)

import omni.kit.app  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

# Enable the URDF importer extension (not loaded by default in headless mode).
_ext_manager = omni.kit.app.get_app().get_extension_manager()
if not _ext_manager.is_extension_enabled("isaacsim.asset.importer.urdf"):
    _ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

# Patch missing API for older importer (v2.3.10 lacks set_merge_fixed_ignore_inertia).
try:
    from isaacsim.asset.importer.urdf._urdf import ImportConfig as _ImportConfig
    if not hasattr(_ImportConfig, "set_merge_fixed_ignore_inertia"):
        _ImportConfig.set_merge_fixed_ignore_inertia = lambda self, v: None
except Exception:
    pass

# ── Locate default paths ──────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../../../../../../.."))
_PUB_SRDF_ROOT = os.path.join(_REPO_ROOT, "pub_srdf_model")
_PUB_SRDF_ROOT_PARENT = _REPO_ROOT  # package://pub_srdf_model/ → _REPO_ROOT/pub_srdf_model/

# 项目内自洽的 mesh/urdf 存储路径（已通过 data/ 目录归档）
_DATA_XMS5 = os.path.join(_SCRIPT_DIR, "..", "..", "data", "robots", "xms5")
_URDF_DIR = os.path.join(_DATA_XMS5, "urdf")

def _pending_urdfs() -> list[str]:
    """返回 data/robots/xms5/urdf/ 下尚未转换为 USD 的 xm_urdf_*.urdf 文件。

    已存在同名 .usd 的文件自动跳过，避免重复转换。
    """
    if not os.path.isdir(_URDF_DIR):
        return []
    existing_usd = {
        os.path.splitext(f)[0]
        for f in os.listdir(_DATA_XMS5)
        if f.endswith(".usd")
    }
    return sorted(
        os.path.join(_URDF_DIR, f)
        for f in os.listdir(_URDF_DIR)
        if f.startswith("xm_urdf_") and f.endswith(".urdf")
        and os.path.splitext(f)[0] not in existing_usd
    )

_DEFAULT_URDFS = _pending_urdfs() or [
    os.path.join(_PUB_SRDF_ROOT, "urdf", "xm_urdf_default.urdf"),
    os.path.join(_PUB_SRDF_ROOT, "urdf", "xm_urdf_plus.urdf"),
]

_DEFAULT_OUTPUT_DIR = os.path.join(_DATA_XMS5)


def _sanitize_prim_name(name: str) -> str:
    """Replace characters invalid in USD prim names with underscores."""
    import re
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _resolve_package_uris(urdf_path: str, pkg_root_map: dict) -> str:
    """Return path to a temp URDF with package:// URIs replaced by absolute paths.

    pkg_root_map: {package_name: absolute_dir_that_contains_that_package}
    Also strips <transmission> blocks and sanitizes STL filenames so they produce
    valid USD prim names (Isaac Sim rejects names containing '-').
    """
    import re
    import shutil
    import tempfile

    with open(urdf_path, "r") as f:
        content = f.read()

    # Pre-build a filename→absolute-path lookup for each package root.
    _file_cache: dict[str, str] = {}
    for pkg, root in pkg_root_map.items():
        pkg_dir = os.path.join(root, pkg)
        if os.path.isdir(pkg_dir):
            for dirpath, _, filenames in os.walk(pkg_dir):
                for fname in filenames:
                    _file_cache[fname] = os.path.join(dirpath, fname)

    # Temp dir to hold copies of mesh files with sanitized names.
    mesh_tmp_dir = tempfile.mkdtemp(prefix="urdf_meshes_")

    def replace_uri(m):
        pkg = m.group(1)
        rest = m.group(2)
        if pkg not in pkg_root_map:
            return m.group(0)
        abs_path = os.path.join(pkg_root_map[pkg], pkg, rest)
        if not os.path.exists(abs_path):
            fname = os.path.basename(rest)
            if fname in _file_cache:
                abs_path = _file_cache[fname]

        # Sanitize: USD prim names must not contain '-' or other special chars.
        stem, ext = os.path.splitext(os.path.basename(abs_path))
        sanitized_stem = _sanitize_prim_name(stem)
        if sanitized_stem != stem:
            sanitized_path = os.path.join(mesh_tmp_dir, sanitized_stem + ext)
            if not os.path.exists(sanitized_path):
                shutil.copy2(abs_path, sanitized_path)
            return sanitized_path
        return abs_path

    content = re.sub(r"package://([^/]+)/(.+?)(?=[\"' ])", replace_uri, content)

    # Strip <transmission> blocks — ROS-only, not needed for USD and can
    # reference non-existent joint names that break the Isaac Sim importer.
    content = re.sub(r"<transmission\b[^>]*>.*?</transmission>", "", content, flags=re.DOTALL)

    # Strip ROS-convention wrapper links/joints that cause branching at the root
    # and trigger IndexError: map::at in Isaac Sim's C++ URDF importer when
    # root_link_name is used with a branching empty-link structure.
    content = re.sub(r"<link\s+name=[\"']base_link[\"'][^/]*/?>.*?</link>", "", content, flags=re.DOTALL)
    content = re.sub(r"<link\s+name=[\"']base_link[\"']\s*/>", "", content)
    content = re.sub(r"<link\s+name=[\"']arm_base_link[\"'][^/]*/>", "", content)
    content = re.sub(r"<joint\s+name=[\"']robot_base_fix[\"'][^>]*>.*?</joint>", "", content, flags=re.DOTALL)
    content = re.sub(r"<joint\s+name=[\"']base_link_fix[\"'][^>]*>.*?</joint>", "", content, flags=re.DOTALL)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".urdf", delete=False, mode="w",
        prefix=os.path.splitext(os.path.basename(urdf_path))[0] + "_resolved_"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


def convert_one(urdf_path: str, output_dir: str, cfg_kwargs: dict) -> str:
    """Convert a single URDF to USD and return the generated USD path."""
    urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Replace package:// URIs so Isaac Sim's importer can find the meshes.
    resolved_urdf = _resolve_package_uris(urdf_path, {"pub_srdf_model": _PUB_SRDF_ROOT_PARENT})
    print(f"  [resolved URDF] {resolved_urdf}")

    os.makedirs(output_dir, exist_ok=True)
    usd_name = os.path.splitext(os.path.basename(urdf_path))[0] + ".usd"

    urdf_cfg = UrdfConverterCfg(
        asset_path=resolved_urdf,
        usd_dir=os.path.abspath(output_dir),
        usd_file_name=usd_name,
        force_usd_conversion=True,
        fix_base=cfg_kwargs["fix_base"],
        merge_fixed_joints=cfg_kwargs["merge_joints"],
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=cfg_kwargs["joint_stiffness"],
                damping=cfg_kwargs["joint_damping"],
            ),
            target_type=cfg_kwargs["joint_target_type"],
        ),
    )

    print("\n" + "=" * 72)
    print(f"Converting: {urdf_path}")
    print(f"Output dir: {os.path.abspath(output_dir)}")
    print("Config:")
    print_dict(urdf_cfg.to_dict(), nesting=2)
    print("=" * 72)

    converter = UrdfConverter(urdf_cfg)
    print(f"  -> USD written to: {converter.usd_path}")
    return converter.usd_path


def main():
    output_dir = os.path.abspath(args_cli.output_dir or _DEFAULT_OUTPUT_DIR)

    cfg_kwargs = {
        "fix_base": args_cli.fix_base,
        "merge_joints": args_cli.merge_joints,
        "joint_stiffness": args_cli.joint_stiffness,
        "joint_damping": args_cli.joint_damping,
        "joint_target_type": args_cli.joint_target_type,
    }

    urdf_list = [args_cli.urdf] if args_cli.urdf else _DEFAULT_URDFS

    results = []
    for urdf in urdf_list:
        usd_path = convert_one(urdf, output_dir, cfg_kwargs)
        results.append(usd_path)

    print("\n" + "=" * 72)
    print("Conversion complete. USD files:")
    for p in results:
        print(f"  {p}")
    print("=" * 72)


if __name__ == "__main__":
    main()
    simulation_app.close()
