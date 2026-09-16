# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ============================================================
# ManagerBasedRLEnv — IsaacLab RL 环境类（继承自 ManagerBasedEnv）
#
# 在 ManagerBasedEnv（基础仿真封装）之上增加了 MDP 的完整定义：
#   TerminationManager — 判断 episode 是否结束（terminated / truncated）
#   RewardManager      — 计算标量奖励信号
#   CurriculumManager  — 动态调整任务难度（课程学习）
#   CommandManager     — 生成目标命令（如导航目标、末端目标位姿）
#
# 同时实现了 gym.Env 接口，使其可直接被 stable-baselines3 / RSL-RL /
# skrl 等 RL 框架调用。
#
# 与 ManagerBasedEnv 的核心差异：
#   step() 返回值多了 reward、terminated、truncated（符合 Gymnasium API）
#   step() 内部自动检测并 reset 终止的子环境（无需外部手动 reset）
#   load_managers() 扩展了基类，额外创建 4 个 RL 专属 Manager
#
# Manager 创建顺序（load_managers 内）：
#   1. CommandManager        ← ObservationManager 可能依赖命令
#   2. super().load_managers()
#      ├── RecorderManager
#      ├── ActionManager
#      └── ObservationManager（依赖上述两者）
#   3. TerminationManager
#   4. RewardManager         ← 可能依赖 TerminationManager
#   5. CurriculumManager
#   6. _configure_gym_env_spaces()
#   7. event_manager.apply("startup")
#
# step() 内部执行顺序：
#   process_action → decimation 物理循环 → episode_length_buf++ →
#   termination_manager.compute → reward_manager.compute →
#   reset 终止环境 → command_manager.compute → obs_buf
# ============================================================

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch

from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.ui.widgets import ManagerLiveVisualizer

from .common import VecEnvStepReturn
from .manager_based_env import ManagerBasedEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


class ManagerBasedRLEnv(ManagerBasedEnv, gym.Env):
    """The superclass for the manager-based workflow reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """

    # 告知 gym 包装器（如 VecEnvWrapper）这是向量化环境，批次维度为 num_envs
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""

    # render_modes 列表供 gym 检查；render_fps 在 __init__ 中根据 step_dt 动态设置
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
    }
    """Metadata for the environment."""

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # ── 课程学习公共步数计数器 ────────────────────────────────────────────
        # common_step_counter 在所有子环境间共享，用于 CurriculumManager 判断
        # 当前训练进度（如逐步增大奖励权重、缩小随机化范围等）
        self.common_step_counter = 0

        # ── 各子环境当前集内步数缓冲区 ────────────────────────────────────────
        # episode_length_buf[i] 记录第 i 个子环境本集已走的步数。
        # 必须在 super().__init__() 之前初始化，因为父类 load_managers() 中
        # ObservationManager / TerminationManager 可能在初始化时就访问该缓冲区
        # （如 mdp.time_out 用它与 max_episode_length 比较判断超时）
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)

        # ── 调用父类初始化（ManagerBasedEnv.__init__）────────────────────────
        # 父类负责：创建 SimulationContext、InteractiveScene、EventManager、
        # 启动仿真、调用 load_managers()（本类重写了该方法）
        super().__init__(cfg=cfg)

        # gym.Env 标准字段，控制 render() 行为
        self.render_mode = render_mode

        # 设置视频录制器的帧率，使回放速度与仿真实时速度一致
        # render_fps = 1 / step_dt（例：step_dt=1/30 → render_fps=30）
        self.metadata["render_fps"] = 1 / self.step_dt

        print("[INFO]: Completed setting up the environment...")

    """
    Properties.
    """

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        # 直接来自 cfg.episode_length_s，例如 30.0 秒
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        # 向上取整：确保至少能走满 episode_length_s 秒
        # 例：episode_length_s=30, step_dt=1/30 → max_episode_length=900
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """

    def load_managers(self):
        """加载所有 Manager（重写父类，扩展 RL 专用 Manager）。

        创建顺序有严格依赖关系：
          1. CommandManager  — ObservationManager 的某些 ObsTerm 可能引用命令目标
          2. super().load_managers() — RecorderManager / ActionManager / ObservationManager
          3. TerminationManager — 独立，无依赖
          4. RewardManager  — 某些 RewardTerm 可能读取 TerminationManager 的终止标志
          5. CurriculumManager — 需要知道奖励和终止情况才能调整课程
          6. _configure_gym_env_spaces() — 所有 Manager 就绪后才能确定空间维度
          7. startup 事件 — 在所有 Manager 就绪后执行一次性初始化随机化
        """
        # ── 1. CommandManager ───────────────────────────────────────────────
        # 生成每步的任务命令（如目标速度、目标关节角、目标末端位姿）。
        # ObservationManager 中的 ObsTerm 可能会读取命令（如 generated_commands），
        # 因此必须先于 ObservationManager 创建。
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # ── 2. 父类 Manager（RecorderManager / ActionManager / ObservationManager）
        super().load_managers()

        # ── 3. TerminationManager ───────────────────────────────────────────
        # 根据 cfg.terminations 中定义的条件（如超时、跌倒、越界）判断每个
        # 子环境是否 terminated（违反约束）或 truncated（超时）
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)

        # ── 4. RewardManager ────────────────────────────────────────────────
        # 根据 cfg.rewards 中定义的 RewardTerm 函数计算标量奖励。
        # 某些 RewardTerm（如 is_terminated 惩罚项）依赖 TerminationManager 的结果，
        # 因此在 TerminationManager 之后创建。
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)

        # ── 5. CurriculumManager ────────────────────────────────────────────
        # 根据训练进度动态调整任务参数（如奖励权重、随机化幅度、命令范围等）。
        # 依赖 common_step_counter 和各集 episode 统计信息。
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # ── 6. 配置 Gym 动作 / 观测空间 ────────────────────────────────────
        # 所有 Manager 就绪后才能确定各 ObsTerm 的张量维度，进而构造
        # gym.spaces.Box / gym.spaces.Dict 并批量化为 num_envs 份
        self._configure_gym_env_spaces()

        # ── 7. startup 事件 ──────────────────────────────────────────────────
        # 在所有 Manager 就绪后执行一次性初始化（如生成程序化地形、初始随机化）。
        # 父类中此逻辑被跳过（"if self.__class__ == ManagerBasedEnv"），在此补充执行。
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""
        # 扩展父类，增加 RL 专属的 4 个 Manager 实时面板
        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }

    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # ── 1. 动作预处理 ─────────────────────────────────────────────────────
        # 将原始策略输出（如归一化值）转换为关节目标量，存入 ActionManager 缓冲区
        self.action_manager.process_action(action.to(self.device))

        # 录制：记录执行前的状态快照（动作、观测等）
        self.recorder_manager.record_pre_step()

        # 提前判断渲染需求，避免在循环内多次查询
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # ── 2. 物理仿真 decimation 循环 ───────────────────────────────────────
        # 策略频率 = 1/step_dt，物理频率 = 1/physics_dt = decimation/step_dt
        # 例：step_dt=1/30, physics_dt=1/120 → 每 env step 执行 4 次物理步
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            # 将 ActionManager 缓冲区中的目标值写入 PhysX 关节驱动器
            self.action_manager.apply_action()
            # 将 Isaac Lab 内部所有数据缓冲区同步到 PhysX（包括关节目标、刚体速度等）
            self.scene.write_data_to_sim()
            # 推进物理仿真一步（物理与渲染分离）
            self.sim.step(render=False)
            # 录制每次 decimation 后的物理状态（用于高频数据采集场景）
            self.recorder_manager.record_post_physics_decimation_step()
            # 按 render_interval 触发渲染（节省 GPU）
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # 从 PhysX 回读最新关节/刚体数据，更新 scene 缓冲区
            self.scene.update(dt=self.physics_dt)

        # ── 3. 计步器更新 ─────────────────────────────────────────────────────
        # episode_length_buf：各子环境本集内步数，用于 time_out 判断
        self.episode_length_buf += 1
        # common_step_counter：全局步数（所有环境共享），用于课程学习进度参考
        self.common_step_counter += 1

        # ── 4. 终止条件检测 ───────────────────────────────────────────────────
        # reset_buf[i]=True 表示第 i 个环境本步结束（terminated 或 truncated）
        self.reset_buf = self.termination_manager.compute()
        # terminated：因违反约束（如跌倒、碰撞）而终止，对应 MDP 的非自然终止
        self.reset_terminated = self.termination_manager.terminated
        # time_outs：因超时（episode_length_buf >= max_episode_length）而截断，
        # 对应 Gymnasium API 中的 truncated，许多 RL 算法对两者处理不同
        self.reset_time_outs = self.termination_manager.time_outs

        # ── 5. 奖励计算 ───────────────────────────────────────────────────────
        # reward_buf[i] = 第 i 个环境本步的标量奖励
        # dt=step_dt 用于将"每秒奖励率"类型的 RewardTerm 缩放到当前步长
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # ── 6. 录制后处理（如有）────────────────────────────────────────────
        # 只有存在活跃录制 Term 时才提前计算观测（避免无谓开销）
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # ── 7. 自动 reset 已终止的子环境 ────────────────────────────────────
        # RL 框架通常期望环境在终止后自动 reset 并返回新集初始观测，
        # 而不需要外部显式调用 reset()。
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # 录制：记录 reset 前的终态（上一集最后一步）
            self.recorder_manager.record_pre_reset(reset_env_ids)

            # 重置场景状态、执行随机化事件、清空各 Manager 缓冲区
            self._reset_idx(reset_env_ids)

            # RTX 传感器需要重新渲染以获得重置后的图像
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # 录制：记录 reset 后的初始状态
            self.recorder_manager.record_post_reset(reset_env_ids)

        # ── 8. 命令更新 ───────────────────────────────────────────────────────
        # 在 reset 之后更新命令，确保刚被 reset 的环境也能获得新命令
        # （如导航目标点、末端目标位姿等）
        self.command_manager.compute(dt=self.step_dt)

        # ── 9. 步间事件 ───────────────────────────────────────────────────────
        # 例：每隔固定步数施加一次随机外力扰动（push_robot）
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # ── 10. 计算观测 ──────────────────────────────────────────────────────
        # 在 reset 之后计算，使已重置环境的观测反映新集初始状态（而非终止帧）
        # update_history=True：更新帧历史缓冲区（如需要多帧堆叠的观测）
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # 返回 Gymnasium 标准五元组：obs, reward, terminated, truncated, info
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return a numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # 无 RTX 传感器时需手动触发渲染以更新画面；有 RTX 传感器时 step() 内已渲染
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()

        if self.render_mode == "human" or self.render_mode is None:
            # human 模式：直接显示到 GUI 窗口，无需返回数组
            return None
        elif self.render_mode == "rgb_array":
            # rgb_array 模式：返回 numpy 图像，供 gymnasium.wrappers.RecordVideo 等使用
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # 懒初始化 Replicator 渲染产品和 RGB 注释器（首次调用时创建，之后复用）
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # render_product 绑定到 cfg.viewer.cam_prim_path 视口相机
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # AnnotatorRegistry 提供从渲染产品读取 RGB 数据的接口
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])

            rgb_data = self._rgb_annotator.get_data()
            # 将 bytes buffer 转为 H×W×4 uint8 数组（RGBA），然后截取 RGB 三通道
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # 渲染器预热期间可能返回空数据，此时返回全黑帧避免崩溃
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        if not self._is_closed:
            # ── 析构顺序：RL Manager 先于父类 Manager 销毁 ───────────────────
            # command / reward / termination / curriculum 依赖 scene 和 sim，
            # 必须在父类 close()（会销毁 scene 和 sim）之前先销毁
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager
            # 父类负责销毁 ActionManager / ObservationManager / scene / sim 等
            super().close()

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment.

        根据各 Manager 的实际张量维度，构造 gym.spaces 对象，供 RL 框架进行
        形状检查和空间采样（如 action_space.sample() 用于随机策略测试）。
        """
        # ── 观测空间（无界，RL 算法通常自行归一化）────────────────────────────
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]

            if has_concatenated_obs:
                # concatenate_terms=True：整组观测拼成一个 flat 向量
                # shape=group_dim（如 (64,) 表示 64 维状态向量）
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                # concatenate_terms=False：每个 ObsTerm 单独一个 Box，打包成 Dict
                # 各 Box 的 low/high 由 ObsTerm.clip 参数决定（None 则无界）
                group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
                term_dict = {}
                for term_name, term_dim, term_cfg in zip(group_term_names, group_dim, group_term_cfgs):
                    low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
                    high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
                    term_dict[term_name] = gym.spaces.Box(low=low, high=high, shape=term_dim)
                self.single_observation_space[group_name] = gym.spaces.Dict(term_dict)

        # ── 动作空间（无界）──────────────────────────────────────────────────
        # action_dim = 所有 ActionTerm 维度之和（例：4 轮速 + 6 臂位 = 10）
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))

        # ── 向量化批量扩展 ────────────────────────────────────────────────────
        # batch_space 将 single 空间在第 0 维重复 num_envs 次，
        # 使 observation_space / action_space 反映整个批次的形状
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # ── 1. 课程学习更新 ───────────────────────────────────────────────────
        # 在 reset 时调用，记录本集统计（成功率、平均奖励等），
        # 并根据进度调整下一集的参数（如命令范围、随机化幅度）
        self.curriculum_manager.compute(env_ids=env_ids)

        # ── 2. 场景资产内部缓冲区重置 ──────────────────────────────────────────
        # 将 env_ids 对应的关节位置/速度恢复为 InitialStateCfg 默认值
        self.scene.reset(env_ids)

        # ── 3. "reset" 模式随机化事件 ─────────────────────────────────────────
        # 例：随机化初始姿态、摩擦系数、质量等（Domain Randomization）
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # ── 4. 各 Manager 缓冲区重置（顺序敏感）──────────────────────────────
        # extras["log"] 汇总本集统计信息（平均奖励、成功率等），
        # 由 RL 框架写入 TensorBoard / WandB
        self.extras["log"] = dict()

        # observation：清空历史帧缓冲区（如多帧堆叠）
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)

        # action：清空上一步动作缓存（action_rate 奖励依赖此缓存）
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reward：清空累积奖励统计，记录本集 episode_reward 到 extras["log"]
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)

        # curriculum：记录课程进度指标
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)

        # command：为重置的环境重新采样命令目标（如新的导航目标点）
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)

        # event：清空事件计数器（如 push_robot 的间隔计数）
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)

        # termination：清空终止标志缓冲区（terminated / time_outs）
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)

        # recorder：最后重置，确保已完整记录本集数据后再开始新集
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # ── 5. 重置集内步数计数器 ────────────────────────────────────────────
        # 清零已终止环境的步数，开始计数新集
        self.episode_length_buf[env_ids] = 0
