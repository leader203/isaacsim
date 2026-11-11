from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass

from ...assets.sensors.camera_cfg import OrbbecGemini2
from .mecanum730_xms5_brush_rolling_0512_motion_wheel_free_dirty_open_seat_env_cfg import (
    Mecanum730XMS5BrushRolling0512MotionWheelFreeDirtyOpenSeatEnvCfg,
)
from .mecanum730_xms5_gripper_joint_prismatic_motion_wheel_arm_simple_sphere_raw_usd_light_dirty_open_seat_env_cfg import (
    Mecanum730XMS5GripperJointPrismaticMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatEnvCfg,
    Mecanum730XMS5GripperJointPrismaticMotionWheelArmSimpleSphereRawUsdLightDirtyOpenSeatSceneCfg,
)


# ✏️ 方块位置手动指定（world 坐标，X=左右，Y=前后，地面 Z=0）。想换位置改下面这一行即可。
# 需避开覆盖区：机器人底盘 (~1.832,-0.264)、马桶 (~1.810,-1.192)、垃圾桶 (2.097,-1.095)。
CUBE_POS_XY: tuple[float, float] = (1.45, -0.6776)

# ✏️ 起始机械臂关节角（弧度，joint1..6）。想换起始位姿改这一行即可。
# 为什么写在这里而不是 _TARGET_ARM_JOINT_POS：本任务的 events 用的是
# CuroboPosePlanTestEventsCfg，继承自 MotionWheelFreeEventsCfg（另一条分支），基类
# __post_init__ 里对 init_arm_joint_position.default_pos 的覆写落不到实际生效的 events 上，
# 结果机械臂停在全 0 位（实测 initial_joint_pos≈[0,0,0,0,0,0]）。所以这里两处都显式写。
# joint5 决定 cam_tool 俯仰：往负方向 = 低头。cam_tool 安装已改为"光轴沿 tool_link +Z
# （夹爪正下方）"，起始画面要在新安装下同时框住方块和垃圾桶 → joint5 取 -1.95（FK 校核
# cube 0.23 / bin 0.22；原值 -1.85 也满足，cube 0.31 / bin 0.20）。
# ⚠️ 不要同时把 joint6 转 180°：相机会被甩到另一侧并朝上，方块跑到相机背后（FK 实测）。
ARM_HOME_JOINT_POS: tuple[float, float, float, float, float, float] = (
    0.099483767364,
    -0.200712863979,
    -0.799360797413,
    0.0,
    -1.95,
    0.0,
)


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
            pos=(1.81735, -1.24027, 2.19445),
            rot=(0.5621, -0.4323, 0.4429, 0.5485),
            convention="world",
        ),
        data_types=["rgb", "depth"],
        update_latest_camera_pose=True,
        # 长焦 + 高分辨率：focal 24.139→40.0（水平 FOV 85.7°→58.5°，方块放大 ~1.7×），
        # 分辨率 460→640（像素更多）。cam_tool 仍是 OrbbecGemini2 默认 460（广角末端视角）。
        width=640,
        height=640,
        spawn=PinholeCameraCfg(
            clipping_range=(0.01, 10.0),
            focal_length=40.0,
            horizontal_aperture=44.760724135,
        ),
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
    # cam_tool 刚性跟随 tool_link（夹爪/工具），joint6 转动时相机与夹爪相对姿态不变。
    cam_follows_tool = True

    def __post_init__(self):
        super().__post_init__()
        # 起始机械臂关节角：init_state（生成态）和 startup 事件的 default_pos（实际保持的位姿）
        # 两处都要显式写，原因见文件顶部 ARM_HOME_JOINT_POS 注释。
        for joint_idx, joint_pos in enumerate(ARM_HOME_JOINT_POS, start=1):
            self.scene.robot.init_state.joint_pos[f"joint{joint_idx}"] = joint_pos
        self.events.init_arm_joint_position.params["default_pos"] = list(ARM_HOME_JOINT_POS)
        # 只保留 grasp_test_cube_1；位置由 CUBE_POS_XY 手动指定（不再随机）
        self.scene.grasp_test_cube_1.init_state.pos = (CUBE_POS_XY[0], CUBE_POS_XY[1], 0.021)
        self.scene.grasp_test_cube_2 = None
        self.scene.grasp_test_cube_3 = None
        self.enable_collision_debug_visuals = False
        # 关掉回合超时：工具服务是长驻的，超时会在 SOP 中途自动 reset
        # （改前是 300s，仿真的时间推进比墙钟快很多，一条 SOP 就会超）
        self.episode_length_s = 1.0e6
