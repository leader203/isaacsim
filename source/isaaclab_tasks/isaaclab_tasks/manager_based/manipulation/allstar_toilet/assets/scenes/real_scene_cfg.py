# Shenxinfu real scene configurations.
#
# Two variants:
#   ShenxinfuRealSceneCfg              — 841837 room, static (no interactive joints)
#   ShenxinfuRealInsideSceneCfg        — 841838 room with toilet + door joints

from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from .. import ALLSTAR_ASSETS_DATA_DIR


@configclass
class ShenxinfuRealSceneCfg(InteractiveSceneCfg):
    """841837 room — full static scene (room + fixtures, no interactive joints)."""

    room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=f"{ALLSTAR_ASSETS_DATA_DIR}/scenes/real/shenxinfu/841837/start.usda",
        ),
    )


@configclass
class ShenxinfuRealInsideSceneCfg(InteractiveSceneCfg):
    """841838 inside cubicle — room loaded as AssetBaseCfg.

    841838/start.usda 内部通过 prepend payload 已嵌入 toilet（马桶盖板）和 door（隔间门）
    的完整物理关节，PhysX 会自动激活它们的关节动力学。
    因此此处仅以 AssetBaseCfg 加载整个场景 USD，无需单独定义 ArticulationCfg：
      - 避免与 USD 内部 payload 形成重复物理对象（DOF backend failure）
      - 马桶盖板和门依然响应机器人的物理接触（涨力可开合盖板/门），满足遥操作演示录制需求
      - 若后续需要 IsaacLab 关节状态访问（obs/reward），可在此添加 spawn=None 的 ArticulationCfg
    """

    room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=f"{ALLSTAR_ASSETS_DATA_DIR}/scenes/real/shenxinfu/841838/start.usda",
        ),
    )

