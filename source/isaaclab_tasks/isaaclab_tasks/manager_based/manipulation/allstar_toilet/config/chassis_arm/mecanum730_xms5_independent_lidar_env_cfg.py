"""LiDAR variants for the independent mobile-arm recording configs.

These configs intentionally leave the existing camera-only recording configs
unchanged. Each variant inherits the corresponding independent config and only
adds the standard base-mounted 3D LiDAR sensor.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

from ... import mdp
from .mecanum730_xms5_brush_hard_brush_hard_dirty_open_seat_env_cfg import (
    BrushHardSceneCfg,
    Mecanum730XMS5BrushHardBrushHardDirtyOpenSeatEnvCfg,
)
from .mecanum730_xms5_brush_rolling_brush_rolling_dirty_open_seat_env_cfg import (
    BrushRollingSceneCfg,
    Mecanum730XMS5BrushRollingBrushRollingDirtyOpenSeatEnvCfg,
)
from .mecanum730_xms5_brush_rolling_0512_brush_rolling_0512_dirty_open_seat_env_cfg import (
    BrushRolling0512ObservationsCfg,
    BrushRolling0512SceneCfg,
    Mecanum730XMS5BrushRolling0512BrushRolling0512DirtyOpenSeatEnvCfg,
)
from .mecanum730_xms5_brush_simple_brush_simple_dirty_open_seat_env_cfg import (
    BrushSimpleSceneCfg,
    Mecanum730XMS5BrushSimpleBrushSimpleDirtyOpenSeatEnvCfg,
)
from .mecanum730_xms5_gripper_gripper_dirty_open_seat_env_cfg import (
    GripperSceneCfg,
    Mecanum730XMS5GripperGripperDirtyOpenSeatEnvCfg,
)
from .mecanum730_xms5_gripper_expanded_gripper_expanded_dirty_open_seat_env_cfg import (
    GripperExpandedSceneCfg,
    Mecanum730XMS5GripperExpandedGripperExpandedDirtyOpenSeatEnvCfg,
)
from .mecanum730_xms5_gripper_joint_prismatic_dirty_open_seat_env_cfg import (
    GripperJointPrismaticObservationsCfg,
    GripperJointPrismaticSceneCfg,
    Mecanum730XMS5GripperJointPrismaticDirtyOpenSeatEnvCfg,
)
from .mecanum730_xms5_mop_sponge_mop_sponge_dirty_open_seat_env_cfg import (
    Mecanum730XMS5MopSpongeMopSpongeDirtyOpenSeatEnvCfg,
    MopSpongeSceneCfg,
)
from .mecanum730_xms5_mop_swab_mop_swab_dirty_open_seat_env_cfg import (
    Mecanum730XMS5MopSwabMopSwabDirtyOpenSeatEnvCfg,
    MopSwabSceneCfg,
)
from .mecanum730_xms5_pick_pick_dirty_open_seat_env_cfg import (
    Mecanum730XMS5PickPickDirtyOpenSeatEnvCfg,
    PickSceneCfg,
)
from .mecanum730_xms5_plus_plus_dirty_open_seat_env_cfg import (
    Mecanum730XMS5PlusPlusDirtyOpenSeatEnvCfg,
    PlusSceneCfg,
)
from .mecanum730_xms5_suction_in_suction_in_dirty_open_seat_env_cfg import (
    Mecanum730XMS5SuctionInSuctionInDirtyOpenSeatEnvCfg,
    SuctionInSceneCfg,
)
from .mecanum730_xms5_suction_in_0512_suction_in_0512_dirty_open_seat_env_cfg import (
    Mecanum730XMS5SuctionIn0512SuctionIn0512DirtyOpenSeatEnvCfg,
    SuctionIn0512SceneCfg,
)
from .mecanum730_xms5_suction_out_suction_out_dirty_open_seat_env_cfg import (
    Mecanum730XMS5SuctionOutSuctionOutDirtyOpenSeatEnvCfg,
    SuctionOutSceneCfg,
)
from .mecanum730_xms5_suction_out_0512_suction_out_0512_dirty_open_seat_env_cfg import (
    Mecanum730XMS5SuctionOut0512SuctionOut0512DirtyOpenSeatEnvCfg,
    SuctionOut0512SceneCfg,
)
from .mecanum730_xms5_suction_press_suction_press_dirty_open_seat_env_cfg import (
    Mecanum730XMS5SuctionPressSuctionPressDirtyOpenSeatEnvCfg,
    SuctionPressSceneCfg,
)


def _build_open_seat_base_lidar_cfg() -> MultiMeshRayCasterCfg:
    """Build the base-mounted LiDAR for the open-seat scene used by independent configs."""
    mesh_targets = [
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="/World/envs/env_0/room",
            is_shared=True,
            track_mesh_transforms=False,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="/World/envs/env_.*/room/Meshes/Base/Wall_.*",
            is_shared=False,
            track_mesh_transforms=False,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="/World/envs/env_.*/room/Meshes/Assets/Toilet_.*",
            is_shared=False,
            track_mesh_transforms=False,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="/World/envs/env_.*/room/Meshes/Assets/Edestalurinal_.*",
            is_shared=False,
            track_mesh_transforms=False,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="/World/envs/env_.*/room/Meshes/Assets/Partition_.*",
            is_shared=False,
            track_mesh_transforms=False,
        ),
    ]
    return MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/robot/base_link",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.303, 0.12, 0.995), rot=(0.0, 0.0, 0.0, 1.0)),
        mesh_prim_paths=mesh_targets,
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        max_distance=10.0,
    )


@configclass
class BrushRolling0512LidarObservationsCfg(BrushRolling0512ObservationsCfg):
    @configclass
    class PolicyCfg(BrushRolling0512ObservationsCfg.PolicyCfg):
        lidar_ranges = ObsTerm(func=mdp.lidar_ranges, params={"sensor_name": "lidar"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class GripperJointPrismaticLidarObservationsCfg(GripperJointPrismaticObservationsCfg):
    @configclass
    class PolicyCfg(GripperJointPrismaticObservationsCfg.PolicyCfg):
        lidar_ranges = ObsTerm(func=mdp.lidar_ranges, params={"sensor_name": "lidar"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class BrushRolling0512LidarSceneCfg(BrushRolling0512SceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5BrushRolling0512BrushRolling0512DirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5BrushRolling0512BrushRolling0512DirtyOpenSeatEnvCfg
):
    scene: BrushRolling0512LidarSceneCfg = BrushRolling0512LidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class BrushHardLidarSceneCfg(BrushHardSceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5BrushHardBrushHardDirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5BrushHardBrushHardDirtyOpenSeatEnvCfg
):
    scene: BrushHardLidarSceneCfg = BrushHardLidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class BrushRollingLidarSceneCfg(BrushRollingSceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5BrushRollingBrushRollingDirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5BrushRollingBrushRollingDirtyOpenSeatEnvCfg
):
    scene: BrushRollingLidarSceneCfg = BrushRollingLidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class BrushSimpleLidarSceneCfg(BrushSimpleSceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5BrushSimpleBrushSimpleDirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5BrushSimpleBrushSimpleDirtyOpenSeatEnvCfg
):
    scene: BrushSimpleLidarSceneCfg = BrushSimpleLidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class GripperLidarSceneCfg(GripperSceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5GripperGripperDirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5GripperGripperDirtyOpenSeatEnvCfg
):
    scene: GripperLidarSceneCfg = GripperLidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class GripperExpandedLidarSceneCfg(GripperExpandedSceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5GripperExpandedGripperExpandedDirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5GripperExpandedGripperExpandedDirtyOpenSeatEnvCfg
):
    scene: GripperExpandedLidarSceneCfg = GripperExpandedLidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class MopSpongeLidarSceneCfg(MopSpongeSceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5MopSpongeMopSpongeDirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5MopSpongeMopSpongeDirtyOpenSeatEnvCfg
):
    scene: MopSpongeLidarSceneCfg = MopSpongeLidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class MopSwabLidarSceneCfg(MopSwabSceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5MopSwabMopSwabDirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5MopSwabMopSwabDirtyOpenSeatEnvCfg
):
    scene: MopSwabLidarSceneCfg = MopSwabLidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class PickLidarSceneCfg(PickSceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5PickPickDirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5PickPickDirtyOpenSeatEnvCfg
):
    scene: PickLidarSceneCfg = PickLidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class PlusLidarSceneCfg(PlusSceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5PlusPlusDirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5PlusPlusDirtyOpenSeatEnvCfg
):
    scene: PlusLidarSceneCfg = PlusLidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class SuctionInLidarSceneCfg(SuctionInSceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5SuctionInSuctionInDirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5SuctionInSuctionInDirtyOpenSeatEnvCfg
):
    scene: SuctionInLidarSceneCfg = SuctionInLidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class SuctionIn0512LidarSceneCfg(SuctionIn0512SceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5SuctionIn0512SuctionIn0512DirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5SuctionIn0512SuctionIn0512DirtyOpenSeatEnvCfg
):
    scene: SuctionIn0512LidarSceneCfg = SuctionIn0512LidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class SuctionOutLidarSceneCfg(SuctionOutSceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5SuctionOutSuctionOutDirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5SuctionOutSuctionOutDirtyOpenSeatEnvCfg
):
    scene: SuctionOutLidarSceneCfg = SuctionOutLidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class SuctionOut0512LidarSceneCfg(SuctionOut0512SceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5SuctionOut0512SuctionOut0512DirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5SuctionOut0512SuctionOut0512DirtyOpenSeatEnvCfg
):
    scene: SuctionOut0512LidarSceneCfg = SuctionOut0512LidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class SuctionPressLidarSceneCfg(SuctionPressSceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5SuctionPressSuctionPressDirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5SuctionPressSuctionPressDirtyOpenSeatEnvCfg
):
    scene: SuctionPressLidarSceneCfg = SuctionPressLidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: BrushRolling0512LidarObservationsCfg = BrushRolling0512LidarObservationsCfg()


@configclass
class GripperJointPrismaticLidarSceneCfg(GripperJointPrismaticSceneCfg):
    lidar = _build_open_seat_base_lidar_cfg()


@configclass
class Mecanum730XMS5GripperJointPrismaticDirtyOpenSeatLidarEnvCfg(
    Mecanum730XMS5GripperJointPrismaticDirtyOpenSeatEnvCfg
):
    scene: GripperJointPrismaticLidarSceneCfg = GripperJointPrismaticLidarSceneCfg(num_envs=1, env_spacing=2.5)
    observations: GripperJointPrismaticLidarObservationsCfg = GripperJointPrismaticLidarObservationsCfg()
