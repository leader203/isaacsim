# Mecanum 0730 + XMS5 Gripper 移动操作臂 — 841837 真实卫生间场景。
#
# 机器人：Mecanum 0730 底盘（4 麦轮）+ XMS5-R800-W4G3B4C 六轴机械臂（夹爪末端）
#   USD：data/robots/mecanum730_xms5_gripper.usda
#   CFG：MECANUM730_XMS5_GRIPPER_CFG（assets/robots/mecanum_xms5_cfg.py）
#
# 动作空间（10 维）：
#   wheel_velocity  — 4 轮关节速度（rad/s），关节名：wheel_fl/fr/rl/rr_joint
#   arm_joint_pos   — 6 臂关节位置（rad），关节名正则：joint[1-6]
#
# 观测空间：
#   joint_pos — 所有 10 个关节的当前位置，不拼接（concatenate_terms=False）
#
# 相机：
#   cam_tool — OrbbecGemini2，prim 置于机器人层级外（自由浮动）
#              每帧由脚本调用 set_world_poses 对齐到 tool_link 位置
#              内参：focal_length=24, horizontal_aperture=44.76，分辨率 640×360
#
# 任务 ID：Isaac-Allstar-Toilet-Mecanum0730-XMS5Gripper-v0
#
# 启动命令：
#   ./isaaclab.sh -p .../scripts/run_record_mobile_arm.py \
#       --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Gripper-v0 \
#       --num_demos 10 \
#       --dataset_file /tmp/mecanum730_xms5_gripper_demos.hdf5 \
#       --speed_scale 20.0 --arm_scale 0.05

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

from ...assets.robots import MECANUM730_XMS5_GRIPPER_CFG
from ...assets.scenes import ShenxinfuRealSceneCfg
from ...assets.sensors.camera_cfg import OrbbecGemini2
from ... import mdp


def build_mecanum730_base_lidar_cfg(
    extra_targets: list[MultiMeshRayCasterCfg.RaycastTargetCfg] | None = None,
) -> MultiMeshRayCasterCfg:
    """Build the standard restroom LiDAR config, with optional per-env dynamic targets."""
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
            prim_expr="/World/envs/env_.*/room/Meshes/Assets/Door_.*",
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
    ]
    if extra_targets:
        mesh_targets.extend(extra_targets)

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
class Mecanum730XMS5GripperSceneCfg(ShenxinfuRealSceneCfg):
    """841837 real restroom with Mecanum 0730 + XMS5 Gripper mobile manipulator.

    继承 ShenxinfuRealSceneCfg（申新福 841837 真实卫生间场景），
    在其基础上添加移动操作臂机器人和工具相机。
    """

    # ── 机器人 ────────────────────────────────────────────────────────────────
    # MECANUM730_XMS5_GRIPPER_CFG：Mecanum 0730 底盘 + XMS5 夹爪臂组合体
    #   底盘关节（速度控制）：wheel_fl/fr/rl/rr_joint，stiffness=0，damping=1000
    #   臂关节（位置控制）：joint1-6，stiffness=400，damping=160
    robot = MECANUM730_XMS5_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    # ── 工具相机 ──────────────────────────────────────────────────────────────
    # prim_path 刻意置于 robot 层级外（{ENV_REGEX_NS}/cam_tool 而非 robot/arm/...）
    # 原因：非物理 prim 若位于关节链内，USD 只会给它静态父变换，不会随物理仿真更新。
    #       置于外部后，脚本每帧用 set_world_poses 直接写入世界位姿，实现刚性跟随。
    # offset 初始值无所谓，脚本第一帧就会覆盖。
    cam_tool = OrbbecGemini2(
        prim_path="{ENV_REGEX_NS}/cam_tool",
        offset=OrbbecGemini2.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            # 初始朝向无所谓，脚本每帧用 set_world_poses 覆盖
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )

    # ── 底盘激光雷达 ───────────────────────────────────────────────────────────
    # 挂载在 base_link，使用 16 线 3D LiDAR：
    #   - 水平 360° 扫描
    #   - 垂直视场 ±15°
    #   - horizontal_res=1.0°，用于覆盖整个隔间
    # 射线投影范围：
    #   - 整体 room
    #   - 墙体、隔板、门、马桶、小便池等对路径规划/避障重要的独立目标
    lidar = build_mecanum730_base_lidar_cfg()

    def __post_init__(self):

        self.robot.init_state.pos = (1.80, -0.15, 0.008)
        self.robot.init_state.rot = (0.782608, 0, 0, -0.622515)
        # disable_gravity=None：不全局覆盖重力设置
        #   底盘刚体：沿用 USD 默认（有重力），使轮子自然落地
        #   机械臂各 link：沿用 USDA 里逐 link 写入的 disableGravity=1（无重力，由 PD 维持姿态）
        self.robot.spawn.rigid_props.disable_gravity = None


@configclass
class ActionsCfg:
    """移动操作臂动作空间（共 10 维）。"""

    # 底盘：4 轮速度控制（rad/s）
    #   scale=1.0：脚本输出直接作为目标速度，不做额外缩放
    #   关节顺序由 IsaacLab 按 joint_names 正则匹配后字母序排列
    wheel_velocity = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint"],
        preserve_order=True,
        scale=1.0,
    )

    # 机械臂：6 轴关节位置控制（rad）
    #   scale=1.0：脚本输出的增量直接叠加到当前位置
    #   use_default_offset=True：以 USDA 里 init_state.joint_pos 为零点偏移，
    #     使动作 0 对应初始姿态（而非绝对 0 rad）
    arm_joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """观测空间：关节位置 + 相机 RGB/Depth + 相机 TF 矩阵。"""

    @configclass
    class PolicyCfg(ObsGroup):
        # 所有 10 个关节（4 轮 + 6 臂）的当前位置，shape=[num_envs, 10]
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "wheel_fl_joint",
                        "wheel_fr_joint",
                        "wheel_rl_joint",
                        "wheel_rr_joint",
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                    ],
                    preserve_order=True,
                )
            },
        )

        # 工具相机 RGB 图像，shape=[num_envs, H, W, 3]，uint8 [0,255]
        cam_tool_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_tool"), "data_type": "rgb", "normalize": False},
        )

        # 工具相机深度图，shape=[num_envs, H, W, 1]，单位 m
        cam_tool_depth = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_tool"), "data_type": "depth", "normalize": False},
        )

        # 相机在 tool_link 系下的位姿矩阵（固定），shape=[num_envs, 16]（行主序 4×4）
        # T_cam_in_tool：p_tool = T @ p_cam_homogeneous
        cam_in_tool_tf = ObsTerm(func=mdp.cam_in_tool_tf)

        # 相机在 base_link 系下的位姿矩阵（每步更新），shape=[num_envs, 16]（行主序 4×4）
        # T_cam_in_base：p_base = T @ p_cam_homogeneous
        cam_in_base_tf = ObsTerm(
            func=mdp.cam_in_base_tf,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["tool_link"])},
        )

        # 底盘 3D LiDAR 测距，shape=[num_envs, num_rays]，单位 m（未命中填 max_distance=10.0）
        lidar_ranges = ObsTerm(
            func=mdp.lidar_ranges,
            params={"sensor_name": "lidar"},
        )

        def __post_init__(self):
            self.enable_corruption = False   # 不加观测噪声
            self.concatenate_terms = False   # 各观测项保持独立，不拼成单一向量

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """奖励函数：仅含动作平滑惩罚，防止轮速/臂位指令剧烈抖动。"""

    # 相邻两步动作差的 L2 范数，权重 -1e-4（轻微惩罚，不主导训练）
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """终止条件：仅超时（由 episode_length_s 控制）。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class Mecanum730XMS5GripperEnvCfg(ManagerBasedRLEnvCfg):
    """Mecanum 0730 + XMS5 Gripper mobile manipulator in Shenxinfu 841837 real scene.

    完整环境配置，汇总场景、动作、观测、奖励、终止五个子配置。
    """

    scene: Mecanum730XMS5GripperSceneCfg = Mecanum730XMS5GripperSceneCfg(
        num_envs=1, env_spacing=2.5  # 单环境推理/录制，环境间距 2.5m（多环境时防碰撞）
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # decimation=1：每个物理步都执行一次策略（无跳帧）
        self.decimation = 1
        # 物理步长 1/120s（120Hz），与真实机器人控制频率接近
        self.sim.dt = 1 / 120
        # 每 3 个物理步渲染一帧（渲染频率 40Hz），降低 GPU 渲染开销
        self.sim.render_interval = 3
        # PhysX GPU 内存设置：防止复杂场景（卫生间障碍物多）触发内存溢出
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 256 * 1024 * 1024  # 256 MB
        # 软体接触缓存：当前场景会触发
        #   "Deformable volume contact buffer overflow detected"
        # 日志要求至少 1048576，这里显式提升到 4x，避免运行中反复溢出。
        self.sim.physx.gpu_max_soft_body_contacts = 4 * 2**20
        self.sim.physx.gpu_max_particle_contacts = 4 * 2**20
        # 一并增大相关 GPU 临时/堆缓存，给复杂接触求解留出余量。
        self.sim.physx.gpu_heap_capacity = 256 * 1024 * 1024
        self.sim.physx.gpu_temp_buffer_capacity = 128 * 1024 * 1024
        # 单集最长 30 秒，超时触发 TerminationsCfg.time_out
        self.episode_length_s = 30.0
        # 默认观察者视角：站在机器人右侧斜上方俯视
        self.viewer.eye = (1.5, 0.0, 1.8)
        self.viewer.lookat = (0.0, 0.0, 0.5)
