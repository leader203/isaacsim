# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ============================================================
# ManagerBasedEnv — IsaacLab 基础环境类
#
# 职责：
#   封装 Isaac Sim 仿真场景与若干"Manager"，提供统一的
#   reset / step 接口，供上层 RL 框架（如 gym）调用。
#
# 不包含：
#   奖励函数、终止条件（这些属于 MDP 定义，由子类
#   ManagerBasedRLEnv 扩展）。
#
# 关键时序（standalone 模式）：
#   __init__
#     ├── cfg.validate()
#     ├── SimulationContext 创建/复用
#     ├── InteractiveScene 构建（USD stage）
#     ├── EventManager 创建（支持 prestartup 事件）
#     ├── sim.reset() → 激活 PhysX TensorAPI
#     ├── scene.update()  → 预填充数据缓冲区
#     └── load_managers() → ActionManager / ObservationManager / RecorderManager
#
# 主循环：
#   reset()  →  _reset_idx()  →  scene.write / sim.forward  →  obs_buf
#   step()   →  action_manager.process  →  decimation × physics  →  obs_buf
# ============================================================

import builtins
import logging
import warnings
from collections.abc import Sequence
from typing import Any

import torch

import omni.physx
from isaacsim.core.simulation_manager import SimulationManager

from isaaclab.managers import ActionManager, EventManager, ObservationManager, RecorderManager
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import attach_stage_to_usd_context, use_stage
from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.timer import Timer
from isaaclab.utils.version import get_isaac_sim_version

from .common import VecEnvObs
from .manager_based_env_cfg import ManagerBasedEnvCfg
from .ui import ViewportCameraController
from .utils.io_descriptors import export_articulations_data, export_scene_data

# 模块级 logger，日志名为模块全路径（isaaclab.envs.manager_based_env）
logger = logging.getLogger(__name__)


class ManagerBasedEnv:
    """The base environment encapsulates the simulation scene and the environment managers for
    the manager-based workflow.

    While a simulation scene or world comprises of different components such as the robots, objects,
    and sensors (cameras, lidars, etc.), the environment is a higher level abstraction
    that provides an interface for interacting with the simulation. The environment is comprised of
    the following components:

    * **Scene**: The scene manager that creates and manages the virtual world in which the robot operates.
      This includes defining the robot, static and dynamic objects, sensors, etc.
    * **Observation Manager**: The observation manager that generates observations from the current simulation
      state and the data gathered from the sensors. These observations may include privileged information
      that is not available to the robot in the real world. Additionally, user-defined terms can be added
      to process the observations and generate custom observations. For example, using a network to embed
      high-dimensional observations into a lower-dimensional space.
    * **Action Manager**: The action manager that processes the raw actions sent to the environment and
      converts them to low-level commands that are sent to the simulation. It can be configured to accept
      raw actions at different levels of abstraction. For example, in case of a robotic arm, the raw actions
      can be joint torques, joint positions, or end-effector poses. Similarly for a mobile base, it can be
      the joint torques, or the desired velocity of the floating base.
    * **Event Manager**: The event manager orchestrates operations triggered based on simulation events.
      This includes resetting the scene to a default state, applying random pushes to the robot at different intervals
      of time, or randomizing properties such as mass and friction coefficients. This is useful for training
      and evaluating the robot in a variety of scenarios.
    * **Recorder Manager**: The recorder manager that handles recording data produced during different steps
      in the simulation. This includes recording in the beginning and end of a reset and a step. The recorded data
      is distinguished per episode, per environment and can be exported through a dataset file handler to a file.

    The environment provides a unified interface for interacting with the simulation. However, it does not
    include task-specific quantities such as the reward function, or the termination conditions. These
    quantities are often specific to defining Markov Decision Processes (MDPs) while the base environment
    is agnostic to the MDP definition.

    The environment steps forward in time at a fixed time-step. The physics simulation is decimated at a
    lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
    independently using the :attr:`ManagerBasedEnvCfg.decimation` (number of simulation steps per environment step)
    and the :attr:`ManagerBasedEnvCfg.sim.dt` (physics time-step) parameters. Based on these parameters, the
    environment time-step is computed as the product of the two. The two time-steps can be obtained by
    querying the :attr:`physics_dt` and the :attr:`step_dt` properties respectively.
    """

    def __init__(self, cfg: ManagerBasedEnvCfg):
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment.

        Raises:
            RuntimeError: If a simulation context already exists. The environment must always create one
                since it configures the simulation context and controls the simulation.
        """
        # ── 配置校验 ─────────────────────────────────────────────────────────
        # 检查 cfg 中各字段的合法性（如 decimation > 0，sim.dt > 0 等）
        cfg.validate()
        # 保存配置引用，后续所有子系统均通过 self.cfg 访问参数
        self.cfg = cfg
        # 标记环境是否已关闭，防止 close() 被重复调用
        self._is_closed = False

        # ── 随机种子 ─────────────────────────────────────────────────────────
        # 固定种子保证实验可复现；未设置时发出警告（不影响运行）
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            logger.warning("Seed not set for the environment. The environment creation may not be deterministic.")

        # ── 仿真上下文（SimulationContext）────────────────────────────────────
        # SimulationContext 封装了 Isaac Sim 的 Stage、PhysX、渲染管线。
        # standalone 模式（从终端启动）：此时没有已有 context，直接创建。
        # extension 模式（在 Isaac Sim GUI 内加载扩展）：context 已存在，复用。
        if SimulationContext.instance() is None:
            # 类型注解防止与 Isaac Sim 自带 SimulationContext 混淆
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            # extension 模式：外部已创建 context，只能复用
            if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
                raise RuntimeError("Simulation context already exists. Cannot create a new one.")
            self.sim: SimulationContext = SimulationContext.instance()

        # ── CUDA 设备对齐 ────────────────────────────────────────────────────
        # 确保 PyTorch 默认 CUDA 设备与仿真设备一致，避免跨设备张量传输
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        # ── 启动信息打印 ──────────────────────────────────────────────────────
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        # render_interval < decimation 时，每个 env step 会触发多次渲染，通常不是预期行为
        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) is smaller than the decimation "
                f"({self.cfg.decimation}). Multiple render calls will happen for each environment step. "
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)

        # ── 仿真步计数器 ──────────────────────────────────────────────────────
        # 记录物理步总数，用于判断渲染时机（_sim_step_counter % render_interval == 0）
        self._sim_step_counter = 0

        # ── 额外信息字典 ──────────────────────────────────────────────────────
        # extras 随 reset/step 返回给调用方，内含各 Manager 的日志数据
        self.extras = {}

        # ── 场景构建 ──────────────────────────────────────────────────────────
        # InteractiveScene 负责在 USD Stage 上实例化机器人、障碍物、传感器等所有 prim。
        # use_stage 确保场景构建期间操作的是正确的 Stage（多 Stage 时防止混淆）。
        # attach_stage_to_usd_context 将 Stage 挂载到 USD context，使渲染管线可见。
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            with use_stage(self.sim.get_initial_stage()):
                self.scene = InteractiveScene(self.cfg.scene)
                attach_stage_to_usd_context()
        print("[INFO]: Scene manager: ", self.scene)

        # ── 视口相机控制器 ────────────────────────────────────────────────────
        # 只在有渲染（GUI 或 RTX 离屏渲染）时创建，纯 headless 模式不需要
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        else:
            self.viewport_camera_controller = None

        # ── EventManager（提前创建）──────────────────────────────────────────
        # EventManager 管理随机化事件（domain randomization）、场景重置等。
        # 必须在 sim.reset() 之前创建，因为 "prestartup" 模式的事件（如随机化网格缩放）
        # 需要在物理引擎启动前作用于 USD Stage。
        self.event_manager = EventManager(self.cfg.events, self)

        # 执行 prestartup 事件（如有）：在物理仿真启动前修改 USD
        if "prestartup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="prestartup")

        # ── 启动仿真（激活 PhysX TensorAPI）─────────────────────────────────
        # sim.reset() 会触发 PhysX 编译物理场景，之后才能访问
        # robot.data.joint_pos 等 TensorAPI 句柄。
        # Extension 模式下不在此处 reset，由外部调用 sim.reset_async()。
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                with use_stage(self.sim.get_initial_stage()):
                    self.sim.reset()
                # 预填充场景数据缓冲区（joint_pos、body_pos_w 等），
                # 使 ObservationManager 初始化时能获得有效张量形状。
                # 后续用户调用 reset() 时会重新填充，此处数据不影响训练。
                self.scene.update(dt=self.physics_dt)
            # 物理引擎就绪后再创建各 Manager（它们需要访问 TensorAPI 句柄）
            self.load_managers()

        # ── UI 窗口 ───────────────────────────────────────────────────────────
        # 有 GUI 时创建 IsaacLab 调试窗口（包含各 Manager 的实时可视化面板）
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            self.setup_manager_visualizers()
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            self._window = None

        # ── 观测缓冲区 ────────────────────────────────────────────────────────
        # obs_buf 在每次 reset/step 后被 ObservationManager 填充，
        # 格式为 {group_name: tensor}，与 gym.Env 的 observation_space 对应
        self.obs_buf = {}

        # ── IO 描述符导出（可选）────────────────────────────────────────────
        # 将动作空间、观测空间、关节信息等以 YAML 格式导出，便于调试和文档生成
        if self.cfg.export_io_descriptors:
            self.export_IO_descriptors()

        # ── 废弃字段兼容处理 ──────────────────────────────────────────────────
        # rerender_on_reset 已被 num_rerenders_on_reset 替代，保留兼容逻辑并发出警告
        if self.cfg.rerender_on_reset:
            msg = (
                "\033[93m\033[1m[DEPRECATION WARNING] ManagerBasedEnvCfg.rerender_on_reset is deprecated. Use"
                " ManagerBasedEnvCfg.num_rerenders_on_reset instead.\033[0m"
            )
            warnings.warn(
                msg,
                FutureWarning,
                stacklevel=2,
            )
            if self.cfg.num_rerenders_on_reset == 0:
                self.cfg.num_rerenders_on_reset = 1

    def __del__(self):
        """Cleanup for the environment."""
        # Python 对象被垃圾回收时自动调用，确保仿真资源被释放
        self.close()

    """
    Properties.
    """

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        # 从 InteractiveScene 获取，与 cfg.scene.num_envs 一致
        return self.scene.num_envs

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which the simulation is happening.
        """
        # 等于 cfg.sim.dt，即 PhysX 每步的积分时间（例：1/120 ≈ 8.33ms）
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        # step_dt = physics_dt × decimation
        # 例：physics_dt=1/120, decimation=4 → step_dt=1/30 ≈ 33.3ms（30Hz 策略）
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def device(self):
        """The device on which the environment is running."""
        # 从 SimulationContext 获取，通常为 "cuda:0" 或 "cpu"
        return self.sim.device

    @property
    def get_IO_descriptors(self):
        """Get the IO descriptors for the environment.

        Returns:
            A dictionary with keys as the group names and values as the IO descriptors.
        """
        # 汇总观测、动作、关节、场景的描述信息，供 export_IO_descriptors 使用
        return {
            "observations": self.observation_manager.get_IO_descriptors,
            "actions": self.action_manager.get_IO_descriptors,
            "articulations": export_articulations_data(self),
            "scene": export_scene_data(self),
        }

    def export_IO_descriptors(self, output_dir: str | None = None):
        """Export the IO descriptors for the environment.

        Args:
            output_dir: The directory to export the IO descriptors to.
        """
        import os
        import yaml

        IO_descriptors = self.get_IO_descriptors

        # output_dir 未显式指定时，使用 cfg.log_dir/io_descriptors
        if output_dir is None:
            if self.cfg.log_dir is not None:
                output_dir = os.path.join(self.cfg.log_dir, "io_descriptors")
            else:
                raise ValueError(
                    "Output directory is not set. Please set the log directory using the `log_dir`"
                    " configuration or provide an explicit output_dir parameter."
                )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "IO_descriptors.yaml"), "w") as f:
            print(f"[INFO]: Exporting IO descriptors to {os.path.join(output_dir, 'IO_descriptors.yaml')}")
            yaml.safe_dump(IO_descriptors, f)

    """
    Operations - Setup.
    """

    def load_managers(self):
        """Load the managers for the environment.

        This function is responsible for creating the various managers (action, observation,
        events, etc.) for the environment. Since the managers require access to physics handles,
        they can only be created after the simulator is reset (i.e. played for the first time).

        .. note::
            In case of standalone application (when running simulator from Python), the function is called
            automatically when the class is initialized.

            However, in case of extension mode, the user must call this function manually after the simulator
            is reset. This is because the simulator is only reset when the user calls
            :meth:`SimulationContext.reset_async` and it isn't possible to call async functions in the constructor.

        """
        # EventManager 已在 __init__ 提前创建，此处仅打印（保持日志顺序一致）
        print("[INFO] Event Manager: ", self.event_manager)

        # RecorderManager：负责录制 pre/post reset 和 pre/post step 阶段的数据，
        # 输出格式由 cfg.recorders 中的 DatasetFileHandler 配置决定（如 HDF5）
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)

        # ActionManager：将策略输出的原始动作转换为关节速度/位置/力矩指令，
        # 并通过 scene.write_data_to_sim() 写入 PhysX
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)

        # ObservationManager：从 scene 各资产（机器人、传感器）读取数据，
        # 按 cfg.observations 中定义的 ObsTerm 函数组合成观测张量
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)

        # 执行 startup 事件（如有）：用于整个仿真生命周期只执行一次的随机化
        # 注意：子类（ManagerBasedRLEnv）可能有额外 Manager，需等子类初始化完
        # 再执行 startup，因此这里仅在当前类为 ManagerBasedEnv 时执行
        if self.__class__ == ManagerBasedEnv and "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""
        # 在 IsaacLab GUI 窗口中显示各 Manager 的实时状态（动作值、观测值等）
        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
        }

    """
    Operations - MDP.
    """

    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VecEnvObs, dict]:
        """Resets the specified environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset the specified environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        # env_ids 为 None 时重置所有环境
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # 录制：记录 reset 前的状态（如上一集末尾帧）
        self.recorder_manager.record_pre_reset(env_ids)

        if seed is not None:
            self.seed(seed)

        # 核心重置逻辑：恢复场景、执行随机化事件、重置各 Manager 缓冲区
        self._reset_idx(env_ids)

        # 将重置后的关节位置/速度写入 PhysX，并推进一个物理步使变换生效
        self.scene.write_data_to_sim()
        self.sim.forward()

        # 若有 RTX 传感器（相机），重置后需要重新渲染以获得最新图像
        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        # 录制：记录 reset 后的初始状态
        self.recorder_manager.record_post_reset(env_ids)

        # 计算并返回观测（update_history=True 更新历史帧缓冲区）
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # 等待贴图加载完成（含 RTX 传感器时，首帧可能需要等待 GPU 资源就绪）
        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()

        return self.obs_buf, self.extras

    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None,
        seed: int | None = None,
        is_relative: bool = False,
    ):
        """Resets specified environments to provided states.

        This function resets the environments to the provided states. The state is a dictionary
        containing the state of the scene entities. Please refer to :meth:`InteractiveScene.get_state`
        for the format.

        The function is different from the :meth:`reset` function as it resets the environments to specific states,
        instead of using the randomization events for resetting the environments.

        Args:
            state: The state to reset the specified environments to. Please refer to
                :meth:`InteractiveScene.get_state` for the format.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.
        """
        # reset_to 与 reset 的区别：不走随机化事件，而是直接恢复到指定状态
        # 常用于：从数据集回放轨迹、模仿学习、状态估计对照实验等

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        self.recorder_manager.record_pre_reset(env_ids)

        if seed is not None:
            self.seed(seed)

        # 先执行通用重置（清空缓冲区、执行 reset 模式事件）
        self._reset_idx(env_ids)

        # 再将场景状态覆盖为指定值（覆盖 _reset_idx 设置的随机初始状态）
        # is_relative=True 时 state 中的位置是相对于环境原点的偏移
        self.scene.reset_to(state, env_ids, is_relative=is_relative)

        # 推进物理步使状态生效
        self.sim.forward()

        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        self.recorder_manager.record_post_reset(env_ids)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.extras

    def step(self, action: torch.Tensor) -> tuple[VecEnvObs, dict]:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is
        decimated at a lower time-step. This is to ensure that the simulation is stable. These two
        time-steps can be configured independently using the :attr:`ManagerBasedEnvCfg.decimation` (number of
        simulation steps per environment step) and the :attr:`ManagerBasedEnvCfg.sim.dt` (physics time-step).
        Based on these parameters, the environment time-step is computed as the product of the two.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations and extras.
        """
        # ── 动作预处理 ────────────────────────────────────────────────────────
        # process_action：将原始动作（如归一化值）转换为物理量（关节角度/速度/力矩），
        # 存入 ActionManager 的内部缓冲区，供 apply_action 使用
        self.action_manager.process_action(action.to(self.device))

        # 录制：记录动作执行前的状态
        self.recorder_manager.record_pre_step()

        # 提前判断是否需要渲染（避免在循环内重复查询）
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # ── 物理仿真 decimation 循环 ──────────────────────────────────────────
        # 每个 env step 执行 decimation 次物理步，提高数值稳定性。
        # 例：decimation=4, physics_dt=1/120 → env_step_dt=1/30（策略 30Hz，物理 120Hz）
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            # 将当前步的动作写入 PhysX 关节驱动器（PD 控制器目标值）
            self.action_manager.apply_action()

            # 将 Isaac Lab 内部缓冲区（关节目标、刚体速度等）同步到 PhysX
            self.scene.write_data_to_sim()

            # 推进物理仿真一步（render=False：物理和渲染分离，由下方逻辑控制渲染时机）
            self.sim.step(render=False)

            # 按 render_interval 触发渲染（降低渲染频率以节省 GPU 开销）
            # 注意：render_interval 是物理步计数，不是 env step 计数
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()

            # 从 PhysX 回读最新数据（joint_pos, body_pos_w 等），更新 scene 缓冲区
            self.scene.update(dt=self.physics_dt)

        # ── 步间事件（interval mode）─────────────────────────────────────────
        # 例：每 N 步施加一次随机扰动（push_robot），dt 用于计算事件触发频率
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # ── 计算观测 ──────────────────────────────────────────────────────────
        # 从 scene 中提取最新传感器数据，按 ObservationsCfg 组合成 obs_buf
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # 录制：记录动作执行后的状态（含观测、关节数据等）
        self.recorder_manager.record_post_step()

        return self.obs_buf, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # 为 Omniverse Replicator（合成数据生成）设置全局种子
        try:
            import omni.replicator.core as rep
            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # 为 PyTorch / NumPy / Python random 等所有库设置种子
        return configure_seed(seed)

    def close(self):
        """Cleanup for the environment."""
        if not self._is_closed:
            # ── 析构顺序敏感 ──────────────────────────────────────────────────
            # Manager 依赖 scene，scene 依赖 sim，必须按依赖反序销毁
            del self.viewport_camera_controller
            del self.action_manager
            del self.observation_manager
            del self.event_manager
            del self.recorder_manager
            del self.scene

            # Isaac Sim 5.x 内存 Stage 模式：需手动 detach PhysX stage 再清理
            if get_isaac_sim_version().major >= 5:
                if self.cfg.sim.create_stage_in_memory:
                    omni.physx.get_physx_simulation_interface().detach_stage()
                    self.sim.stop()
                    self.sim.clear()

            # 清除所有时间线回调，防止 dangling callback 在下次仿真时触发
            self.sim.clear_all_callbacks()
            # 清除 SimulationContext 单例，允许后续重新创建
            self.sim.clear_instance()

            if self._window is not None:
                self._window = None

            self._is_closed = True

    """
    Helper functions.
    """

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # ── 1. 重置场景资产（机器人、物体、传感器）的内部缓冲区 ────────────────
        # scene.reset 会将 env_ids 对应的关节位置/速度恢复为 InitialStateCfg 中的值，
        # 并清空各资产的历史数据缓存
        self.scene.reset(env_ids)

        # ── 2. 执行 "reset" 模式事件（Domain Randomization）──────────────────
        # 例：随机化摩擦系数、质量、初始位置扰动等
        # global_env_step_count 用于判断周期性事件是否应在本次 reset 触发
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # ── 3. 重置各 Manager 的内部状态，收集日志信息 ───────────────────────
        # extras["log"] 汇总各 Manager 在 reset 时产生的统计数据（如随机化参数范围），
        # 供 TensorBoard / WandB 等训练框架记录
        self.extras["log"] = dict()

        # 顺序敏感：observation 依赖 action（历史动作），需先重置 observation
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)

        # recorder 最后重置，确保能记录到本集完整的终态数据
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)
