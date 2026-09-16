# Mecanum 0730 底盘 + XMS5-R800-W4G3B4C 夹爪臂 — 移动操作臂组合配置。
#
# USD 来源：
#   data/robots/mecanum730_xms5_gripper.usda
#   该文件通过 scripts/compose_mecanum_xms5.py 生成，将底盘和机械臂
#   合并为单一关节链（底盘为 ArticulationRoot，臂通过 root_joint 连接）。
#
# 关节布局（共 10 个关节）：
#   底盘（速度控制）：wheel_fl_joint, wheel_fr_joint, wheel_rl_joint, wheel_rr_joint
#   机械臂（位置控制）：joint1, joint2, joint3, joint4, joint5, joint6
#
# USD prim 路径正则（IsaacLab actuator 匹配规则）：
#   "wheel_(fl|fr|rl|rr)_joint"  → 4 个麦轮主轮关节（显式排除 roller 关节）
#   "joint[1-6]"      → 6 个机械臂关节
#
# 控制模式对比：
#   底盘轮组：速度控制（stiffness=0, damping=1000），直接设定目标角速度
#   机械臂：位置控制（stiffness=400, damping=160），PD 控制器跟踪目标关节角
#
# 重力处理：
#   disable_gravity=None：不全局覆盖，依赖各 prim 的单独设置
#     底盘刚体（base_link, wheel_* 等）：USD 默认有重力，使底盘落地
#     机械臂各 link：USDA 中逐 link 设置 physxRigidBody:disableGravity=1，
#                   由 PD 控制器维持姿态，不依赖重力补偿
#
# 调用关系：
#   本文件 → mecanum730_xms5_gripper_env_cfg.py（Mecanum730XMS5GripperSceneCfg）
#         → mecanum730_xms5_gripper_dirty_env_cfg.py（Mecanum730XMS5GripperDirtySceneCfg）

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .. import ALLSTAR_ASSETS_DATA_DIR


MECANUM730_XMS5_GRIPPER_CFG = ArticulationCfg(
    # ── USD 文件加载配置 ────────────────────────────────────────────────────
    spawn=sim_utils.UsdFileCfg(
        # 组合后的移动操作臂 USD：Mecanum 0730 底盘 + XMS5 夹爪臂
        # ALLSTAR_ASSETS_DATA_DIR 解析为 data/ 目录的绝对路径
        usd_path=f"{ALLSTAR_ASSETS_DATA_DIR}/robots/mecanum730_xms5_gripper.usda",

        # 启用接触传感器：底盘轮组与地面的接触力会被记录，
        # 供 EventManager 中的接触事件或 ObservationManager 读取
        activate_contact_sensors=True,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # None：不全局覆盖重力设置，各 prim 按其 USD 属性独立处理：
            #   底盘 prim（base_link 等）：遵循 USD 默认（有重力）→ 轮子能压地
            #   臂 link（link1-6, tool_link 等）：USDA 中已逐个写入
            #     physxRigidBody:disableGravity=1 → 无重力，由 PD 维持姿态
            disable_gravity=None,

            # 当两刚体重叠时（仿真开始前的穿透），允许的最大去穿透速度（m/s）
            # 较大值可快速分离穿透体，但可能导致瞬间速度过大；5.0 为常用稳定值
            max_depenetration_velocity=5.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 关闭自碰撞：机械臂各相邻 link 之间不检测碰撞，避免误触发和性能开销
            # 移动操作臂自身链路间距离近，默认开启会导致大量无效碰撞计算
            enabled_self_collisions=False,

            # PhysX 位置求解器迭代次数：更多迭代 → 约束（关节、接触）更精确，但更慢
            # 16 次适合包含大量关节和接触点的复杂机器人（比默认 8 更稳定）
            solver_position_iteration_count=16,

            # PhysX 速度求解器迭代次数：主要影响碰撞后的速度修正
            # 8 次在速度精度和性能之间取得平衡
            solver_velocity_iteration_count=8,
        ),
    ),

    # ── 初始状态 ────────────────────────────────────────────────────────────
    init_state=ArticulationCfg.InitialStateCfg(
        # 机器人在世界坐标系中的初始位置（x, y, z），单位米
        # scene cfg 的 __post_init__ 会覆盖此值（通常设为场景原点）
        pos=(0.0, 0.0, 0.0),

        # 初始朝向（四元数 wxyz），(1,0,0,0) 表示无旋转
        rot=(1.0, 0.0, 0.0, 0.0),

        # 各关节初始角度（弧度）
        # 轮子从静止开始（0 rad），机械臂各关节从零位开始
        joint_pos={
            "wheel_fl_joint": 0.0,  # 前左轮
            "wheel_fr_joint": 0.0,  # 前右轮
            "wheel_rl_joint": 0.0,  # 后左轮
            "wheel_rr_joint": 0.0,  # 后右轮
            "joint1": 0.0,          # 臂第 1 轴（底座旋转，绕 Z 轴）
            "joint2": 0.0,          # 臂第 2 轴（大臂俯仰）
            "joint3": 0.0,          # 臂第 3 轴（小臂俯仰）
            "joint4": 0.0,          # 臂第 4 轴（腕部旋转）
            "joint5": 0.0,          # 臂第 5 轴（腕部俯仰）
            "joint6": 0.0,          # 臂第 6 轴（末端旋转）
        },
    ),

    # ── 执行器配置 ──────────────────────────────────────────────────────────
    actuators={
        # ── 底盘麦轮（速度控制）──────────────────────────────────────────────
        # ImplicitActuatorCfg：使用 PhysX 内置 PD 驱动器（隐式积分）
        # 速度控制模式：stiffness=0（不追踪位置），damping=1000（黏性阻力追踪速度）
        #   驱动力 τ = stiffness×(θ_target - θ) + damping×(ω_target - ω)
        #           = 0 + 1000×(ω_target - ω)
        # damping=1000 的含义：速度误差 1 rad/s 产生 1000 N·m 的修正力矩，
        # 响应快速，接近理想速度控制，适合地面轮式移动底盘
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_(fl|fr|rl|rr)_joint"],  # 只匹配 4 个麦轮主轮关节
            stiffness=0.0,       # 位置刚度为 0：纯速度控制，不追踪位置
            damping=1000.0,      # 速度阻尼：决定轮速跟踪的响应速度
        ),

        # ── 机械臂（位置控制，中等 PD）──────────────────────────────────────
        # 位置控制模式：stiffness=400（追踪目标关节角），damping=160（抑制震荡）
        #   阻尼比 ζ = damping / (2 × sqrt(stiffness × inertia))
        #   stiffness=400, damping=160 在典型臂关节惯量（~0.1 kg·m²）下接近临界阻尼
        #
        # 与 xms5_cfg.py 中高 PD（stiffness=1600, damping=640）的区别：
        #   本配置（中等 PD）：响应稍慢，更平滑，适合遥操作录制演示
        #   高 PD 配置：响应更快，刚性更强，适合轨迹回放和 Gr00t 推理
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],  # 匹配 6 个臂关节
            effort_limit_sim=100.0,  # 仿真中关节最大力矩（N·m），防止数值爆炸
            velocity_limit_sim=1.0,  # 仿真中关节最大速度（rad/s），限制运动速度
            stiffness=400.0,         # 位置刚度：目标角度误差 1 rad 产生 400 N·m
            damping=160.0,           # 速度阻尼：角速度误差 1 rad/s 产生 160 N·m
        ),
    },

    # 关节位置软限位系数：1.0 表示完全遵守 USD 中定义的关节角度硬限位，
    # 不额外缩小限位范围（< 1.0 会在硬限位内留出安全余量）
    soft_joint_pos_limit_factor=1.0,
)


MECANUM730_XMS5_BRUSH_ROLLING_CFG = MECANUM730_XMS5_GRIPPER_CFG.copy()  # type: ignore
MECANUM730_XMS5_BRUSH_ROLLING_CFG.spawn.usd_path = (
    f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/mecanum730_xms5_brush_rolling.usda"
)

MECANUM730_XMS5_BRUSH_ROLLING_0512_CFG = MECANUM730_XMS5_GRIPPER_CFG.copy()  # type: ignore
MECANUM730_XMS5_BRUSH_ROLLING_0512_CFG.spawn.usd_path = (
    f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/mesh_usd/mecanum730_xms5_brush_rolling_0512.usd"
)
MECANUM730_XMS5_BRUSH_ROLLING_0512_CFG.spawn.articulation_props.enabled_self_collisions = True

MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_BASED_CFG = MECANUM730_XMS5_GRIPPER_CFG.copy()  # type: ignore
MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_BASED_CFG.spawn.usd_path = (
    f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_based_usd/mecanum730_xms5_brush_rolling_0512.usd"
)
MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_BASED_CFG.spawn.articulation_props.enabled_self_collisions = True

MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_WHEEL_FREE_CFG = MECANUM730_XMS5_GRIPPER_CFG.copy()  # type: ignore
MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_WHEEL_FREE_CFG.spawn.usd_path = (
    f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_free_usd/mecanum730_xms5_brush_rolling_0512.usd"
)
MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_WHEEL_FREE_CFG.spawn.articulation_props.enabled_self_collisions = True

MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_WHEEL_ARM_SIMPLE_USD_CFG = MECANUM730_XMS5_GRIPPER_CFG.copy()  # type: ignore
MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_WHEEL_ARM_SIMPLE_USD_CFG.spawn.usd_path = (
    f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_usd/mecanum730_xms5_brush_rolling_0512.usd"
)
MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_WHEEL_ARM_SIMPLE_USD_CFG.spawn.articulation_props.enabled_self_collisions = True

MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_WHEEL_ARM_SIMPLE_CFG = MECANUM730_XMS5_GRIPPER_CFG.copy()  # type: ignore
MECANUM730_XMS5_BRUSH_ROLLING_0512_MOTION_WHEEL_ARM_SIMPLE_CFG.spawn = sim_utils.UrdfFileCfg(
    asset_path=(
        f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/"
        "motion_wheel_arm_simple_urdf/mecanum730_xms5_brush_rolling_0512.urdf"
    ),
    usd_dir=f"{ALLSTAR_ASSETS_DATA_DIR}/robots/chassis_arm/motion_wheel_arm_simple_usd",
    usd_file_name="mecanum730_xms5_brush_rolling_0512.usd",
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=None,
        max_depenetration_velocity=5.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=8,
    ),
    force_usd_conversion=False,
    make_instanceable=True,
    fix_base=False,
    link_density=0.0,
    merge_fixed_joints=False,
    convert_mimic_joints_to_normal_joints=False,
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="none",
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=80.0,
            damping=4.0,
        ),
    ),
    collision_from_visuals=False,
    collider_type="convex_hull",
    self_collision=False,
    replace_cylinders_with_capsules=False,
)
