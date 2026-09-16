# Camera sensor configurations matching physical Berxel P100 and Orbbec Gemini2 specs.

from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass


@configclass
class BerxelP100(CameraCfg):
    """Berxel P100 depth camera — 640×450, RGB + depth + semantic segmentation."""

    @configclass
    class OffsetCfg(CameraCfg.OffsetCfg):
        convention = "world"

    spawn = PinholeCameraCfg(
        clipping_range=(0.01, 10.0),
        focal_length=24.0,
        horizontal_aperture=34.874,
    )
    data_types: list[str] = ["rgb", "depth", "semantic_segmentation"]
    width: int = 640
    height: int = 450


@configclass
class OrbbecGemini2(CameraCfg):
    """Orbbec Gemini2 — 1920×1080, RGB + depth + semantic segmentation.

    内参（实测）：fx=1035.422, fy=1035.302, cx=961.676, cy=546.355
    换算：focal_length = fx × horizontal_aperture / width
                       = 1035.422 × 44.760724135 / 1920 ≈ 24.139 mm
    cx/cy 偏心量（<7px）在 IsaacSim 中忽略（主点固定在图像中心）。
    """

    @configclass
    class OffsetCfg(CameraCfg.OffsetCfg):
        convention = "world"

    spawn = PinholeCameraCfg(
        clipping_range=(0.01, 10.0),
        focal_length=24.139,
        horizontal_aperture=44.760724135,
    )
    data_types: list[str] = ["rgb", "depth", "semantic_segmentation"]
    width: int = 460
    height: int = 460
