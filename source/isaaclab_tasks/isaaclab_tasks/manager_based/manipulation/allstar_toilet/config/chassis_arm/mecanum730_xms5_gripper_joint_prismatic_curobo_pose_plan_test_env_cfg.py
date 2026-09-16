from isaaclab.utils import configclass

from ...assets.sensors.camera_cfg import OrbbecGemini2
from .mecanum730_xms5_brush_rolling_0512_motion_wheel_free_dirty_open_seat_env_cfg import (
    Mecanum730XMS5BrushRolling0512MotionWheelFreeDirtyOpenSeatEnvCfg,
)
from .mecanum730_xms5_gripper_joint_prismatic_motion_wheel_arm_simple_sphere_raw_usd_light_dirty_open_seat_env_cfg import (
    Mecanum730XMS5GripperJointPrismaticMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatEnvCfg,
    Mecanum730XMS5GripperJointPrismaticMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg,
)


def _sample_cube_xy() -> tuple[float, float]:
    """在房间矩形区域内随机采样 cube 的 XY，排除机器人/马桶/垃圾桶覆盖区。

    矩形 4 角: (2.4005,-1.545) (1.23167,-1.53227) (1.24472,-0.00691) (2.39119,-0.005327)
    排除区（world 坐标）：(cx, cy, half_w, half_h)
      机器人底盘 ~0.9x0.3m 朝 -Y    马桶占位 ~1.0x0.9m    垃圾桶占位 ~0.7x0.7m
    """
    import random

    exclude_zones = (
        (1.832, -0.264, 0.35, 0.50),
        (1.810, -1.192, 0.50, 0.45),
        (2.097, -1.095, 0.35, 0.35),
    )
    while True:
        x = random.uniform(1.23167, 2.4005)
        y = random.uniform(-1.545, -0.005327)
        for cx, cy, hw, hh in exclude_zones:
            if abs(x - cx) < hw and abs(y - cy) < hh:
                break  # 落在排除区 → 重新采样
        else:
            return x, y


@configclass
class Mecanum730XMS5GripperJointPrismaticCuroboPosePlanTestSceneCfg(
    Mecanum730XMS5GripperJointPrismaticMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg
):
    """Scene clone for isolated cuRobo pose planning with the prismatic gripper robot."""

    # 固定相机：挂在 env_0 下（与能持续出图的 cam_tool 同一层级），offset 就是世界固定位姿，
    # 不随机器人动。位置/姿态由用户指定，world 约定（forward=+X, up=+Z），俯视房间。
    # ⚠️ 不能挂全局 /World/cam_fixed：2026-09-16 实测全局路径从第 1 帧就是空白。
    #    挂在 env_0 下 + run 脚本 _sync_fixed_camera_pose 每步 set_world_poses（只写一次，
    #    不叠加直写 USD xformOp）可出图。
    # data_types 去掉 semantic_segmentation：Isaac Sim 4.5 对 instance/semantic 有已知 bug
    # （camera.py 里有专门 workaround），先移除验证。
    cam_fixed = OrbbecGemini2(
        prim_path="{ENV_REGEX_NS}/cam_fixed",
        offset=OrbbecGemini2.OffsetCfg(
            pos=(1.66468, -1.00862, 1.96617),
            rot=(0.53479, -0.46169, 0.45678, 0.54054),
            convention="world",
        ),
        data_types=["rgb", "depth"],
        update_latest_camera_pose=True,
    )


@configclass
class CuroboPosePlanTestEventsCfg(
    Mecanum730XMS5BrushRolling0512MotionWheelFreeDirtyOpenSeatEnvCfg.MotionWheelFreeEventsCfg
):
    """Pose-plan-test events (base events; grasp_test_cube needs no spawning)."""


@configclass
class Mecanum730XMS5GripperJointPrismaticCuroboPosePlanTestEnvCfg(
    Mecanum730XMS5GripperJointPrismaticMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatEnvCfg
):
    """Isolated test task for cuRobo tool-tip pose planning on the prismatic gripper robot."""

    scene: Mecanum730XMS5GripperJointPrismaticCuroboPosePlanTestSceneCfg = (
        Mecanum730XMS5GripperJointPrismaticCuroboPosePlanTestSceneCfg(num_envs=1, env_spacing=2.5)
    )
    events: CuroboPosePlanTestEventsCfg = CuroboPosePlanTestEventsCfg()
    enable_collision_debug_visuals = False

    def __post_init__(self):
        super().__post_init__()
        # 只保留 grasp_test_cube_1；每次场景加载随机位置（避开机器人/马桶/垃圾桶）
        x, y = _sample_cube_xy()
        self.scene.grasp_test_cube_1.init_state.pos = (x, y, 0.021)
        self.scene.grasp_test_cube_2 = None
        self.scene.grasp_test_cube_3 = None
        self.enable_collision_debug_visuals = False
        self.episode_length_s = 300.0
