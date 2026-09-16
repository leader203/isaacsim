# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ============================================================
# ManagerBasedRLEnvCfg — RL 环境顶层配置类
#
# 继承关系：
#   ManagerBasedRLEnvCfg
#   └── ManagerBasedEnvCfg          (manager_based_env_cfg.py)
#
# 在父类基础上新增的字段：
#   episode_length_s  — 每集时长（秒）[必填]
#   rewards           — 奖励函数配置  [必填]
#   terminations      — 终止条件配置  [必填]
#   curriculum        — 课程学习配置  [可选，默认 None]
#   commands          — 目标命令配置  [可选，默认 None]
#   is_finite_horizon — 有限/无限视野标志（影响 bootstrapping）
#
# 使用方式：
#   @configclass
#   class MyRLEnvCfg(ManagerBasedRLEnvCfg):
#       scene        = MySceneCfg(num_envs=4096, env_spacing=2.5)
#       observations = MyObsCfg()
#       actions      = MyActionsCfg()
#       rewards      = MyRewardsCfg()
#       terminations = MyTerminationsCfg()
#       decimation   = 4
#
#       def __post_init__(self):
#           self.sim.dt = 1 / 120
#           self.episode_length_s = 20.0
# ============================================================

from dataclasses import MISSING

from isaaclab.utils import configclass

from .manager_based_env_cfg import ManagerBasedEnvCfg
from .ui import ManagerBasedRLEnvWindow


@configclass
class ManagerBasedRLEnvCfg(ManagerBasedEnvCfg):
    """Configuration for a reinforcement learning environment with the manager-based workflow.

    说明
    ----
    本类是所有 RL 任务环境配置的基类，在 ManagerBasedEnvCfg（基础仿真配置）
    之上增加了完整 MDP 定义所需的额外字段：奖励、终止、课程学习、目标命令。

    父类已提供的字段（无需重复定义）：
      sim, scene, observations, actions, events, recorders,
      viewer, seed, decimation, num_rerenders_on_reset 等。

    子类必须填写的 MISSING 字段（共 5 个，含父类继承的）：
      decimation       (来自父类)
      scene            (来自父类)
      observations     (来自父类)
      actions          (来自父类)
      episode_length_s (本类新增)
      rewards          (本类新增)
      terminations     (本类新增)
    """

    # ══════════════════════════════════════════════════════════════════════════
    # UI 设置
    # ══════════════════════════════════════════════════════════════════════════

    # 覆盖父类默认值，使用包含 RL 专属面板（奖励曲线、终止统计、课程进度）的窗口
    ui_window_class_type: type | None = ManagerBasedRLEnvWindow

    # ══════════════════════════════════════════════════════════════════════════
    # 通用 RL 设置
    # ══════════════════════════════════════════════════════════════════════════

    is_finite_horizon: bool = False
    """Whether the learning task is treated as a finite or infinite horizon problem for the agent.
    Defaults to False, which means the task is treated as an infinite horizon problem.

    说明：
      该标志决定环境在超时（time_out）时向 RL 智能体发送哪种 done 信号，
      影响值函数的 bootstrapping 策略：

      False（默认，无限视野）：
        超时时发送 truncated=True（time-limit done），智能体收到此信号后
        会对超时状态的值函数进行 bootstrapping（V(s_T) ≠ 0），
        适合周期性任务或理论上没有自然终点的任务（如平衡控制、导航）。

      True（有限视野）：
        超时时不发送 truncated 信号（或等价地视为 terminated），
        智能体将超时状态的值视为 0，无需 bootstrapping，
        适合有明确终点的任务（如抓取成功即结束、限定步数的游戏）。

    实现细节：
      ManagerBasedRLEnv 本身不直接使用此标志，而是由上层 RL 库封装层
      （如 RSL-RL Wrapper、skrl Wrapper）读取并据此决定是否传递 truncated 信号。

    This flag handles the subtleties of finite and infinite horizon tasks:

    * **Finite horizon**: no penalty or bootstrapping value is required by the the agent for
      running out of time. However, the environment still needs to terminate the episode after the
      time limit is reached.
    * **Infinite horizon**: the agent needs to bootstrap the value of the state at the end of the episode.
      This is done by sending a time-limit (or truncated) done signal to the agent, which triggers this
      bootstrapping calculation.

    If True, then the environment is treated as a finite horizon problem and no time-out (or truncated) done signal
    is sent to the agent. If False, then the environment is treated as an infinite horizon problem and a time-out
    (or truncated) done signal is sent to the agent.

    Note:
        The base :class:`ManagerBasedRLEnv` class does not use this flag directly. It is used by the environment
        wrappers to determine what type of done signal to send to the corresponding learning agent.
    """

    episode_length_s: float = MISSING
    """Duration of an episode (in seconds). [必填]

    说明：
      每集的最大时长（秒），超过此时长触发 TerminationManager 中的 time_out 条件。

      实际步数上限由以下公式计算（在 ManagerBasedRLEnv.max_episode_length 中实现）：
        max_episode_length = ceil(episode_length_s / step_dt)
                           = ceil(episode_length_s / (decimation × sim.dt))

      示例：
        episode_length_s=30, decimation=4, sim.dt=1/120
        → step_dt = 4/120 = 1/30
        → max_episode_length = ceil(30 / (1/30)) = 900 步

      设置原则：
        过短 → 智能体来不及完成任务就超时，难以获得正奖励
        过长 → 无效探索时间多，训练效率低
        通常根据任务物理时间需求设定（如机械臂抓取 10~20s，导航任务 30~60s）

    Based on the decimation rate and physics time step, the episode length is calculated as:

    .. code-block:: python

        episode_length_steps = ceil(episode_length_s / (decimation_rate * physics_time_step))

    For example, if the decimation rate is 10, the physics time step is 0.01, and the episode length is 10 seconds,
    then the episode length in steps is 100.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # MDP 子系统配置（RL 专属）
    # ══════════════════════════════════════════════════════════════════════════

    rewards: object = MISSING
    """Reward settings. [必填]

    说明：
      定义奖励函数，由 RewardManager 在每个 env step 计算标量奖励。
      每个 RewardTerm 由一个函数 + 权重（weight）组成，最终奖励是所有项的加权和：
        reward = Σ weight_i × term_i(env, dt)

      示例结构：
        @configclass
        class RewardsCfg:
            # 正奖励：机械臂末端接近目标
            reaching_object = RewTerm(
                func=mdp.object_ee_distance, weight=1.0,
                params={"std": 0.1}
            )
            # 负奖励：惩罚过大的动作幅度（防止震荡）
            action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
            # 负奖励：终止时额外惩罚（可选）
            undesired_contacts = RewTerm(
                func=mdp.undesired_contacts, weight=-1.0, ...
            )

      权重设计原则：
        - 稀疏奖励（完成任务才得分）适合简单任务
        - 密集奖励（接近目标即得分）适合复杂任务的早期训练
        - 惩罚项权重通常远小于主奖励项（防止智能体只规避惩罚）

    Please refer to the :class:`isaaclab.managers.RewardManager` class for more details.
    """

    terminations: object = MISSING
    """Termination settings. [必填]

    说明：
      定义何时终止当前集，由 TerminationManager 在每步检查。
      每个 TerminationTerm 返回 bool 张量（shape=[num_envs]），表示各环境是否触发该条件。

      终止类型：
        terminated（非自然终止，time_out=False）：
          机器人违反约束（如跌倒、碰撞、越出工作空间），
          RL 框架将此状态的值视为 0（episode 自然结束）。
        truncated（超时截断，time_out=True）：
          达到 max_episode_length 但任务未完成，
          RL 框架需对此状态进行 bootstrapping（值不为 0）。

      必须包含的条件：
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

      常见附加条件：
        robot_fell     = DoneTerm(func=mdp.base_height_below_threshold, ...)
        bad_contact    = DoneTerm(func=mdp.illegal_contact, ...)
        out_of_bounds  = DoneTerm(func=mdp.root_out_of_bounds, ...)

    Please refer to the :class:`isaaclab.managers.TerminationManager` class for more details.
    """

    curriculum: object | None = None
    """Curriculum settings. Defaults to None, in which case no curriculum is applied.

    说明：
      课程学习（Curriculum Learning）在训练过程中动态调整任务难度，
      使智能体从简单场景逐步过渡到困难场景，提升训练效率和最终性能。

      None（默认）：不使用课程学习，任务参数固定。

      有课程时，CurriculumManager 在每次 _reset_idx() 时调用 compute()，
      根据当前训练进度（common_step_counter）或集内统计（成功率、平均奖励）
      更新任务参数（如命令速度范围、随机化幅度、地形难度）。

      示例结构：
        @configclass
        class CurriculumCfg:
            terrain_levels = CurrTerm(
                func=mdp.terrain_levels_vel,
                params={"asset_cfg": SceneEntityCfg("robot")}
            )

    Please refer to the :class:`isaaclab.managers.CurriculumManager` class for more details.
    """

    commands: object | None = None
    """Command settings. Defaults to None, in which case no commands are generated.

    说明：
      命令生成器（CommandManager）在每个 env step 计算目标命令，
      并将其注入观测空间（ObservationManager 可通过 generated_commands ObsTerm 读取），
      引导智能体朝特定目标运动。

      None（默认）：不生成命令，适合无目标命令的任务（如平衡控制、固定目标抓取）。

      有命令时，CommandManager 在每步 compute(dt) 中：
        1. 根据配置采样新命令（如随机速度目标 [vx, vy, wz]）
        2. 在集结束/reset 时重新采样命令
        3. 提供命令可视化（箭头、坐标系等，在 GUI 中显示）

      示例结构（速度跟踪任务）：
        @configclass
        class CommandsCfg:
            base_velocity = UniformVelocityCommandCfg(
                asset_name="robot",
                resampling_time_range=(10.0, 10.0),  # 每 10s 重采样一次
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-1.0, 1.0),
                    lin_vel_y=(-0.5, 0.5),
                    ang_vel_z=(-1.0, 1.0),
                ),
            )

    Please refer to the :class:`isaaclab.managers.CommandManager` class for more details.
    """
