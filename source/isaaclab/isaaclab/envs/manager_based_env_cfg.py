# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.

模块说明
--------
本文件定义两个配置类：

1. DefaultEventManagerCfg
   默认事件管理器配置，仅包含一个 reset 事件（reset_scene_to_default），
   负责在每集开始时将场景恢复到初始状态。

2. ManagerBasedEnvCfg
   基础环境的顶层配置，汇聚所有子系统配置：
     sim        — PhysX 物理仿真参数（时间步、求解器、GPU 设置）
     scene      — 场景资产定义（机器人、障碍物、传感器）
     observations — 观测空间定义
     actions    — 动作空间定义
     events     — 随机化/重置事件定义
     recorders  — 数据录制配置
     viewer     — 调试视口相机配置
   以及若干运行时控制参数（decimation、seed、render 行为等）。

使用方式
--------
用户通过继承 ManagerBasedEnvCfg 并填写 MISSING 字段来定义具体任务：

    @configclass
    class MyEnvCfg(ManagerBasedEnvCfg):
        scene = MySceneCfg(num_envs=4096, env_spacing=2.5)
        observations = MyObsCfg()
        actions = MyActionsCfg()
        decimation = 4

        def __post_init__(self):
            self.sim.dt = 1 / 120
            self.episode_length_s = 20.0
"""

from dataclasses import MISSING, field

import isaaclab.envs.mdp as mdp
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RecorderManagerBaseCfg as DefaultEmptyRecorderManagerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .common import ViewerCfg
from .ui import BaseEnvWindow


@configclass
class DefaultEventManagerCfg:
    """Configuration of the default event manager.

    This manager is used to reset the scene to a default state. The default state is specified
    by the scene configuration.

    说明
    ----
    该配置只包含一个事件项：reset_scene_to_default。
    当 ManagerBasedEnv._reset_idx() 调用 EventManager.apply(mode="reset") 时，
    该函数会把所有资产（关节位置/速度、刚体位姿/速度）恢复为
    ArticulationCfg.InitialStateCfg / RigidObjectCfg.InitialStateCfg 中定义的默认值。

    如需自定义 reset 行为（如随机化初始位姿），可在子类中覆盖或添加 EventTerm：

        @configclass
        class MyEventCfg(DefaultEventManagerCfg):
            randomize_robot_joints = EventTerm(
                func=mdp.randomize_joint_by_gaussian_offset,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.1},
            )
    """

    # mode="reset"：每次调用 _reset_idx() 时触发，将场景恢复到配置文件中定义的初始状态
    reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class ManagerBasedEnvCfg:
    """Base configuration of the environment.

    说明
    ----
    所有基于 Manager 工作流的环境（ManagerBasedEnv 及其子类 ManagerBasedRLEnv）
    均使用本配置类作为顶层配置。

    标注为 MISSING 的字段是**必填项**，子类必须在 @configclass 或 __post_init__ 中赋值，
    否则环境初始化时会抛出 MissingMandatoryParam 错误。
    """

    # ══════════════════════════════════════════════════════════════════════════
    # 仿真与渲染设置
    # ══════════════════════════════════════════════════════════════════════════

    viewer: ViewerCfg = ViewerCfg()
    """Viewer configuration. Default is ViewerCfg().

    控制调试视口（Isaac Sim GUI 中的摄像机）的初始位置、朝向和分辨率。
    常用字段：
      eye    — 相机世界坐标（x, y, z），单位米
      lookat — 相机注视点
      resolution — 视口分辨率（宽, 高），影响 render(mode="rgb_array") 的输出尺寸
    示例：
      self.viewer.eye = (1.5, 0.0, 1.8)
      self.viewer.lookat = (0.0, 0.0, 0.5)
    """

    sim: SimulationCfg = SimulationCfg()
    """Physics simulation configuration. Default is SimulationCfg().

    控制 PhysX 物理引擎和 Isaac Sim 渲染管线的核心参数。
    常用字段：
      dt              — 物理步长（秒），决定仿真精度，常设为 1/60 或 1/120
      render_interval — 每隔多少个物理步渲染一帧（降低渲染频率以节省 GPU）
      physx           — PhysX 求解器参数（GPU 内存、迭代次数等）
      device          — 计算设备（"cuda:0" 或 "cpu"）
    示例：
      self.sim.dt = 1 / 120
      self.sim.render_interval = 3   # 渲染频率 = 120/3 = 40Hz
      self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
    """

    # ══════════════════════════════════════════════════════════════════════════
    # UI 设置
    # ══════════════════════════════════════════════════════════════════════════

    ui_window_class_type: type | None = BaseEnvWindow
    """The class type of the UI window. Default is None.

    If None, then no UI window is created.

    说明：
      有 GUI 时（sim.has_gui() 返回 True），环境会创建一个 IsaacLab 调试窗口，
      显示各 Manager 的实时状态（关节位置、奖励曲线等）。
      BaseEnvWindow 是默认实现，用户可继承后覆盖此字段以定制 UI 面板。

    Note:
        If you want to make your own UI window, you can create a class that inherits from
        from :class:`isaaclab.envs.ui.base_env_window.BaseEnvWindow`. Then, you can set
        this attribute to your class type.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # 通用运行时设置
    # ══════════════════════════════════════════════════════════════════════════

    seed: int | None = None
    """The seed for the random number generator. Defaults to None, in which case the seed is not set.

    说明：
      种子在 ManagerBasedEnv.__init__ 的最早阶段设置，影响：
        - Omniverse Replicator（合成数据随机化）
        - PyTorch / NumPy / Python random 等所有随机库
      设置种子可使实验在不同机器、不同运行次数间产生完全相同的场景初始化。
      None 时发出警告但不影响运行，适合快速调试。

    Note:
      The seed is set at the beginning of the environment initialization. This ensures that the environment
      creation is deterministic and behaves similarly across different runs.
    """

    decimation: int = MISSING
    """Number of control action updates @ sim dt per policy dt.

    说明：
      decimation 决定策略频率与物理频率的比值：
        step_dt（策略步长） = sim.dt × decimation
        physics_dt（物理步长）= sim.dt

      例：sim.dt=1/120, decimation=4
        → 物理频率 120Hz，策略频率 30Hz
        → step() 每次执行 4 次物理仿真，对应真实时间 1/30 秒

      较大的 decimation 可提高数值稳定性（物理步更小），但增加每 env step 的计算量。
      对于轮式移动机器人，decimation=1（与 sim.dt=1/60）常用；
      对于高频关节控制，decimation=4（sim.dt=1/120）更稳定。

    For instance, if the simulation dt is 0.01s and the policy dt is 0.1s, then the decimation is 10.
    This means that the control action is updated every 10 simulation steps.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # 环境子系统配置（Manager 配置）
    # ══════════════════════════════════════════════════════════════════════════

    scene: InteractiveSceneCfg = MISSING
    """Scene settings. [必填]

    定义仿真场景中的所有实体：机器人、静态障碍物、动态物体、传感器（相机、激光雷达）等。
    每个实体对应 InteractiveSceneCfg 的一个字段（类型为 ArticulationCfg / RigidObjectCfg 等）。

    必须指定：
      num_envs   — 并行子环境数量（训练时通常 1024~8192，推理时 1）
      env_spacing — 相邻子环境的间距（米），防止碰撞检测跨环境干扰

    Please refer to the :class:`isaaclab.scene.InteractiveSceneCfg` class for more details.
    """

    recorders: object = DefaultEmptyRecorderManagerCfg()
    """Recorder settings. Defaults to recording nothing.

    说明：
      RecorderManager 在 pre/post reset 和 pre/post step 阶段收集数据，
      并通过 DatasetFileHandler（如 HDF5Handler）写入磁盘。

      默认值 DefaultEmptyRecorderManagerCfg 表示不录制任何数据。
      如需录制演示数据（用于模仿学习），可替换为自定义 RecorderManagerCfg：

        self.recorders = HDF5RecorderManagerCfg(
            dataset_file="demos.hdf5",
            num_demos=100,
        )

      通常不在 env cfg 中直接设置，而由 RecordWrapper 在包装环境时动态注入。

    Please refer to the :class:`isaaclab.managers.RecorderManager` class for more details.
    """

    observations: object = MISSING
    """Observation space settings. [必填]

    定义策略网络的输入：从仿真状态中提取哪些量、如何处理（归一化、裁剪、加噪声）。
    每个观测组（ObsGroup）对应一个策略头，组内包含若干 ObsTerm（具体的观测量）。

    示例结构：
      class PolicyCfg(ObsGroup):          # 策略观测组
          joint_pos = ObsTerm(...)         # 关节位置
          joint_vel = ObsTerm(...)         # 关节速度
          body_pos  = ObsTerm(...)         # 末端位姿

      class ObservationsCfg:
          policy: PolicyCfg = PolicyCfg()  # 可有多个组（如 critic 组）

    Please refer to the :class:`isaaclab.managers.ObservationManager` class for more details.
    """

    actions: object = MISSING
    """Action space settings. [必填]

    定义策略网络的输出如何映射到仿真底层控制指令。
    每个 ActionTerm 对应一类动作（如关节位置、关节速度、末端位姿增量）。

    示例（移动操作臂）：
      wheel_velocity = JointVelocityActionCfg(
          asset_name="robot", joint_names=["wheel_.*_joint"], scale=1.0
      )
      arm_joint_pos = JointPositionActionCfg(
          asset_name="robot", joint_names=["joint[1-6]"], scale=1.0
      )

    多个 ActionTerm 的动作维度会被拼接成一个 flat 向量传入策略网络。

    Please refer to the :class:`isaaclab.managers.ActionManager` class for more details.
    """

    events: object = DefaultEventManagerCfg()
    """Event settings. Defaults to the basic configuration that resets the scene to its default state.

    说明：
      EventManager 根据触发模式（mode）在不同时机执行随机化或其他操作：
        "prestartup" — 物理引擎启动前（如随机化网格缩放）
        "startup"    — 所有 Manager 就绪后执行一次（如生成程序化地形）
        "reset"      — 每集开始时（如随机化初始关节角、摩擦系数）
        "interval"   — 每隔固定步数（如定期施加随机外力扰动）

      默认值 DefaultEventManagerCfg 只包含 reset_scene_to_default，
      即每集开始时将场景恢复到 InitialStateCfg 定义的默认值，不做随机化。

      如需 Domain Randomization，在子类中覆盖：
        self.events.randomize_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass, mode="reset", ...
        )

    Please refer to the :class:`isaaclab.managers.EventManager` class for more details.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # 渲染行为控制
    # ══════════════════════════════════════════════════════════════════════════

    rerender_on_reset: bool = False
    """Whether a render step is performed again after at least one environment has been reset.
    Defaults to False, which means no render step will be performed after reset.

    ⚠ 已废弃（deprecated since 2.3.1），请使用 num_rerenders_on_reset 替代。

    说明：
      reset 后相机等 RTX 传感器的数据不会自动更新，仍是重置前的旧数据。
      设为 True 等价于 num_rerenders_on_reset=1，会在 reset 后额外渲染一帧，
      使传感器数据反映重置后的新状态，但会增加每次 reset 的耗时。

    * When this is False, data collected from sensors after performing reset will be stale and will not reflect the
      latest states in simulation caused by the reset.
    * When this is True, an extra render step will be performed to update the sensor data
      to reflect the latest states from the reset. This comes at a cost of performance as an additional render
      step will be performed after each time an environment is reset.

    .. deprecated:: 2.3.1
        This attribute is deprecated and will be removed in the future. Please use
        :attr:`num_rerenders_on_reset` instead.

        To get the same behaviour as setting this parameter to ``True`` or ``False``, set
        :attr:`num_rerenders_on_reset` to 1 or 0, respectively.
    """

    num_rerenders_on_reset: int = 0
    """Number of render steps to perform after reset. Defaults to 0, which means no render step will be
    performed after reset.

    说明：
      0（默认）：reset 后不额外渲染，传感器数据可能滞后一帧。
                适合不依赖相机观测的任务（如纯关节状态观测），性能最优。
      1        ：reset 后渲染一帧，传感器数据与新状态同步。
                适合依赖相机图像的任务（如视觉导航、工具相机拍摄）。
      >1       ：多次渲染，适合需要光线追踪预热（RTX denoiser warm-up）的场景。
                每次额外渲染约增加 10~30ms 耗时（取决于场景复杂度）。

    * When this is 0, no render step will be performed after reset. Data collected from sensors after performing
      reset will be stale and will not reflect the latest states in simulation caused by the reset.
    * When this is greater than 0, the specified number of extra render steps will be performed to update the
      sensor data to reflect the latest states from the reset. This comes at a cost of performance as additional
      render steps will be performed after each time an environment is reset.
    """

    wait_for_textures: bool = True
    """True to wait for assets to be loaded completely, False otherwise. Defaults to True.

    说明：
      Isaac Sim 的贴图和材质是异步加载的，首次 reset 后可能尚未就绪。
      True（默认）：reset 后轮询 SimulationManager.assets_loading()，
                   直到所有贴图加载完成再返回观测，确保视觉数据完整。
                   仅在有 RTX 传感器时生效，无 RTX 传感器时此参数无效。
      False      ：不等待，立即返回（可能包含黑色/未完成贴图的图像）。
                   适合纯物理仿真任务或贴图不重要的快速调试场景。
    """

    # ══════════════════════════════════════════════════════════════════════════
    # 扩展功能设置
    # ══════════════════════════════════════════════════════════════════════════

    xr: XrCfg | None = None
    """Configuration for viewing and interacting with the environment through an XR device.

    说明：
      XR（扩展现实）配置，用于将仿真画面投影到 VR/AR 头显设备，
      并支持通过手柄等 XR 输入设备与仿真交互（如遥操作机器人）。
      None（默认）表示不使用 XR 设备。
    """

    teleop_devices: DevicesCfg = field(default_factory=DevicesCfg)
    """Configuration for teleoperation devices.

    说明：
      定义用于遥操作的输入设备（如键盘、SpaceMouse、手柄、触觉手套等）。
      DevicesCfg 默认为空（不配置任何设备）。
      遥操作脚本（如 run_record_mobile_arm.py）通常直接在脚本中创建设备实例，
      而不通过此字段配置，此字段主要供 InteractionWrapper 等框架层使用。
    """

    export_io_descriptors: bool = False
    """Whether to export the IO descriptors for the environment. Defaults to False.

    说明：
      True 时，环境初始化完成后会将观测空间、动作空间、关节信息、场景结构
      以 YAML 格式导出到 log_dir/io_descriptors/IO_descriptors.yaml。
      适用于：自动文档生成、模型部署前的接口校验、调试动作/观测维度。
      False（默认）：不导出，节省初始化时间。
    """

    log_dir: str | None = None
    """Directory for logging experiment artifacts. Defaults to None, in which case no specific log directory is set.

    说明：
      指定实验日志根目录，供以下功能使用：
        - export_io_descriptors=True 时，IO 描述符写入 log_dir/io_descriptors/
        - RecorderManager 的 DatasetFileHandler 可读取此路径确定输出位置
      None（默认）：不设置日志目录，export_io_descriptors=True 时必须提供
                   output_dir 参数，否则抛出 ValueError。
    """
