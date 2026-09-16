# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ============================================================
# InteractiveScene — 交互式仿真场景运行时对象
#
# 职责：
#   解析 InteractiveSceneCfg，在 USD stage 上实例化并管理
#   所有场景实体（地形、关节体、传感器、灯光、额外道具等），
#   并通过 Cloner API 将单个原型环境（env_0）复制为 num_envs 份。
#
# 实体分类（按字典存储）：
#   _terrain              — 地形（TerrainImporter），决定 env_origins
#   _articulations        — 关节体（机器人）
#   _deformable_objects   — 可变形体（布料、软体）
#   _rigid_objects        — 刚体（物体、障碍物）
#   _rigid_object_collections — 刚体集合
#   _surface_grippers     — 表面夹爪
#   _sensors              — 传感器（相机、力传感器等）
#   _extras               — 杂项（灯光、静态道具等），存为 XformPrimView
#
# 初始化顺序（顺序敏感）：
#   1. 创建 GridCloner，定义原型环境路径 /World/envs/env_0
#   2. replicate_physics=False 时先预克隆 Xform（允许各环境独立 USD prim）
#   3. 调用 _add_entities_from_cfg() 在 env_0 内实例化所有实体
#   4. replicate_physics=True 时调用 clone_environments() 统一克隆
#   5. 调用 cloner.replicate_physics() 优化 PhysX 解析
#   6. 若需要碰撞过滤，调用 filter_collisions()
#
# 每步仿真循环（由 ManagerBasedEnv.step 驱动）：
#   scene.reset(env_ids)        → 重置指定环境的所有实体内部缓冲
#   scene.write_data_to_sim()   → 将缓冲数据写入 PhysX（关节目标位置/速度等）
#   sim.step()                  → PhysX 推进一步物理
#   scene.update(dt)            → 从 PhysX 读取最新状态，更新传感器
#
# 访问方式：
#   scene["robot"]              → 按名称查找任意实体
#   scene.articulations["robot"]→ 按类别字典访问
# ============================================================

import logging
from collections.abc import Sequence
from typing import Any

import torch

import carb
from isaacsim.core.cloner import GridCloner
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObject,
    DeformableObjectCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
    SurfaceGripper,
    SurfaceGripperCfg,
)
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, SensorBase, SensorBaseCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage, get_current_stage_id
from isaaclab.sim.views import XformPrimView
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils.version import get_isaac_sim_version

# Note: This is a temporary import for the VisuoTactileSensorCfg class.
# It will be removed once the VisuoTactileSensor class is added to the core Isaac Lab framework.
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

from .interactive_scene_cfg import InteractiveSceneCfg

# import logger
logger = logging.getLogger(__name__)


class InteractiveScene:
    """A scene that contains entities added to the simulation.

    The interactive scene parses the :class:`InteractiveSceneCfg` class to create the scene.
    Based on the specified number of environments, it clones the entities and groups them into different
    categories (e.g., articulations, sensors, etc.).

    Cloning can be performed in two ways:

    * For tasks where all environments contain the same assets, a more performant cloning paradigm
      can be used to allow for faster environment creation. This is specified by the ``replicate_physics`` flag.

      .. code-block:: python

          scene = InteractiveScene(cfg=InteractiveSceneCfg(replicate_physics=True))

    * For tasks that require having separate assets in the environments, ``replicate_physics`` would have to
      be set to False, which will add some costs to the overall startup time.

      .. code-block:: python

          scene = InteractiveScene(cfg=InteractiveSceneCfg(replicate_physics=False))

    Each entity is registered to scene based on its name in the configuration class. For example, if the user
    specifies a robot in the configuration class as follows:

    .. code-block:: python

        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass

        from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


        @configclass
        class MySceneCfg(InteractiveSceneCfg):
            # ANYmal-C robot spawned in each environment
            robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    Then the robot can be accessed from the scene as follows:

    .. code-block:: python

        from isaaclab.scene import InteractiveScene

        # create 128 environments
        scene = InteractiveScene(cfg=MySceneCfg(num_envs=128))

        # access the robot from the scene
        robot = scene["robot"]
        # access the robot based on its type
        robot = scene.articulations["robot"]

    If the :class:`InteractiveSceneCfg` class does not include asset entities, the cloning process
    can still be triggered if assets were added to the stage outside of the :class:`InteractiveScene` class:

    .. code-block:: python

        scene = InteractiveScene(cfg=InteractiveSceneCfg(num_envs=128, replicate_physics=True))
        scene.clone_environments()

    .. note::
        It is important to note that the scene only performs common operations on the entities. For example,
        resetting the internal buffers, writing the buffers to the simulation and updating the buffers from the
        simulation. The scene does not perform any task specific to the entity. For example, it does not apply
        actions to the robot or compute observations from the robot. These tasks are handled by different
        modules called "managers" in the framework. Please refer to the :mod:`isaaclab.managers` sub-package
        for more details.
    """

    def __init__(self, cfg: InteractiveSceneCfg):
        """Initializes the scene.

        Args:
            cfg: The configuration class for the scene.

        初始化流程详解：
          1. 验证配置（cfg.validate），确保无 MISSING 字段
          2. 初始化各实体字典（_articulations, _sensors, ...)
          3. 获取 SimulationContext / USD stage 句柄
          4. 创建 GridCloner，设定原型环境路径与间距
          5. 根据 replicate_physics 选择克隆时机：
               False → 先预克隆 env Xform，让 Spawner 在各环境下独立实例化
               True  → 先在 env_0 实例化，再统一 clone + replicate_physics
          6. 调用 _add_entities_from_cfg 在 env_0 内按 cfg 字段顺序实例化
          7. 碰撞过滤（filter_collisions）
        """
        # 验证配置对象中无 MISSING 字段（缺值直接报错，避免运行时神秘崩溃）
        cfg.validate()
        # 保存配置引用
        self.cfg = cfg

        # ── 实体容器初始化 ─────────────────────────────────────────────────────
        # 每类实体用独立字典存储，key=cfg 字段名，value=实体运行时对象
        self._terrain = None                          # TerrainImporter（只有一份，不用字典）
        self._articulations = dict()                  # 关节体机器人
        self._deformable_objects = dict()             # 可变形体（布料/软体）
        self._rigid_objects = dict()                  # 刚体
        self._rigid_object_collections = dict()       # 刚体集合
        self._sensors = dict()                        # 传感器
        self._surface_grippers = dict()               # 表面夹爪
        self._extras = dict()                         # 杂项（灯光/静态道具等）

        # ── 仿真上下文 ────────────────────────────────────────────────────────
        # SimulationContext 是单例，在 sim.app.run 之前已由框架创建
        self.sim = SimulationContext.instance()
        # 当前 USD stage（用于直接操作 USD prim）
        self.stage = get_current_stage()
        # stage ID（用于多 stage 场景的区分，通常不需要直接使用）
        self.stage_id = get_current_stage_id()

        # 物理场景路径（如 /World/physicsScene），延迟查找（首次访问时才遍历 stage）
        self._physics_scene_path = None

        # ── GridCloner 初始化 ─────────────────────────────────────────────────
        # GridCloner 负责将 /World/envs/env_0 复制为 num_envs 份，
        # 在世界空间中按网格排列，相邻环境间距 env_spacing 米。
        self.cloner = GridCloner(spacing=self.cfg.env_spacing, stage=self.stage)
        # 定义所有环境所在的父命名空间（/World/envs）
        self.cloner.define_base_env(self.env_ns)
        # 生成各环境的 prim 路径：["/World/envs/env_0", "/World/envs/env_1", ...]
        self.env_prim_paths = self.cloner.generate_paths(f"{self.env_ns}/env", self.cfg.num_envs)
        # 在 stage 上创建原型环境 /World/envs/env_0（空的 Xform prim）
        self.stage.DefinePrim(self.env_prim_paths[0], "Xform")

        # 全局环境索引张量，shape=[num_envs]，用于批量索引操作
        self._ALL_INDICES = torch.arange(self.cfg.num_envs, dtype=torch.long, device=self.device)

        # ── 异构环境预克隆（replicate_physics=False）────────────────────────────
        # replicate_physics=False 表示各环境可能有不同的 USD prim 结构（异构场景）。
        # 此时需要先克隆 env Xform，让 Spawner 在各环境独立的 prim 路径下实例化资产；
        # 之后还会在 _add_entities_from_cfg 里对每个环境分别 spawn。
        if not self.cfg.replicate_physics:
            # 检查 Isaac Sim 版本，IS5 新增了 clone_in_fabric 参数
            if get_isaac_sim_version().major < 5:
                # 克隆环境 Xform（仅复制变换，不复制物理属性）
                env_origins = self.cloner.clone(
                    source_prim_path=self.env_prim_paths[0],
                    prim_paths=self.env_prim_paths,
                    replicate_physics=False,
                    copy_from_source=True,
                    enable_env_ids=(
                        self.cfg.filter_collisions if self.device != "cpu" else False
                    ),  # replicate_physics=False 时 enable_env_ids 无效，仅占位
                )
            else:
                # IS5+ 版本，额外支持 clone_in_fabric 参数
                env_origins = self.cloner.clone(
                    source_prim_path=self.env_prim_paths[0],
                    prim_paths=self.env_prim_paths,
                    replicate_physics=False,
                    copy_from_source=True,
                    enable_env_ids=(
                        self.cfg.filter_collisions if self.device != "cpu" else False
                    ),  # replicate_physics=False 时 enable_env_ids 无效，仅占位
                    clone_in_fabric=self.cfg.clone_in_fabric,
                )
            # 将 Cloner 返回的环境原点坐标（ndarray）转为 torch Tensor，存储到设备上
            self._default_env_origins = torch.tensor(env_origins, device=self.device, dtype=torch.float32)
        else:
            # replicate_physics=True（同构环境）：环境原点在 clone_environments() 之后才确定
            self._default_env_origins = None

        # ── 从配置实例化实体并克隆 ────────────────────────────────────────────
        # _global_prim_paths：需要与所有环境保持碰撞的全局 prim 路径（如地面）
        self._global_prim_paths = list()

        if self._is_scene_setup_from_cfg():
            # 在 env_0 内按 cfg 字段顺序实例化所有实体
            self._add_entities_from_cfg()

            # 同构场景：在 env_0 填充完毕后统一克隆到所有环境
            if self.cfg.replicate_physics:
                self.clone_environments(copy_from_source=False)

            # 同构场景 + 多环境：调用 replicate_physics 优化 PhysX batch 解析
            # 这让 PhysX 把所有环境的同类资产视为一个批次，大幅加速物理初始化
            if self.cfg.replicate_physics and self.cfg.num_envs > 1:
                if get_isaac_sim_version().major < 5:
                    self.cloner.replicate_physics(
                        source_prim_path=self.env_prim_paths[0],
                        prim_paths=self.env_prim_paths,
                        base_env_path=self.env_ns,
                        root_path=self.env_regex_ns.replace(".*", ""),
                        enable_env_ids=self.cfg.filter_collisions if self.device != "cpu" else False,
                    )
                else:
                    self.cloner.replicate_physics(
                        source_prim_path=self.env_prim_paths[0],
                        prim_paths=self.env_prim_paths,
                        base_env_path=self.env_ns,
                        root_path=self.env_regex_ns.replace(".*", ""),
                        enable_env_ids=self.cfg.filter_collisions if self.device != "cpu" else False,
                        clone_in_fabric=self.cfg.clone_in_fabric,
                    )

            # 异构场景（replicate_physics=False）或 CPU 仿真时，
            # enable_env_ids 无效，必须手动调用 filter_collisions()
            # GPT fix: num_envs=1 时跳过，单环境无需环境间过滤
            if self.cfg.num_envs > 1 and (
                (not self.cfg.replicate_physics and self.cfg.filter_collisions)
                or self.device == "cpu"
            ):
                self.filter_collisions(self._global_prim_paths)

    def clone_environments(self, copy_from_source: bool = False):
        """Creates clones of the environment ``/World/envs/env_0``.

        Args:
            copy_from_source: (bool): If set to False, clones inherit from /World/envs/env_0 and mirror its changes.
            If True, clones are independent copies of the source prim and won't reflect its changes (start-up time
            may increase). Defaults to False.

        说明：
          copy_from_source=False（默认）：
            克隆体继承 env_0（类 USD 引用），env_0 的后续更改会同步到所有克隆。
            启动速度更快，推荐用于同构场景。

          copy_from_source=True：
            完全独立拷贝，env_0 更改不影响克隆体。
            启动更慢，适用于克隆后还需对各环境单独修改的场景。

          多资产警告：
            若用户在不同环境生成了不同资产（multi_assets=True），
            但 replicate_physics=True，会有潜在物理解析问题，此处发出警告。
        """
        # 检查 carb 设置中是否有多资产标记（Spawner 在生成不同资产时会设置此标记）
        carb_settings_iface = carb.settings.get_settings()
        has_multi_assets = carb_settings_iface.get("/isaaclab/spawn/multi_assets")
        if has_multi_assets and self.cfg.replicate_physics:
            logger.warning(
                "Varying assets might have been spawned under different environments."
                " However, the replicate physics flag is enabled in the 'InteractiveScene' configuration."
                " This may adversely affect PhysX parsing. We recommend disabling this property."
            )

        # 执行实际克隆
        if get_isaac_sim_version().major < 5:
            env_origins = self.cloner.clone(
                source_prim_path=self.env_prim_paths[0],
                prim_paths=self.env_prim_paths,
                replicate_physics=self.cfg.replicate_physics,
                copy_from_source=copy_from_source,
                enable_env_ids=(
                    self.cfg.filter_collisions if self.device != "cpu" else False
                ),  # True 时自动为跨环境碰撞对添加过滤规则
            )
        else:
            env_origins = self.cloner.clone(
                source_prim_path=self.env_prim_paths[0],
                prim_paths=self.env_prim_paths,
                replicate_physics=self.cfg.replicate_physics,
                copy_from_source=copy_from_source,
                enable_env_ids=(
                    self.cfg.filter_collisions if self.device != "cpu" else False
                ),  # True 时自动为跨环境碰撞对添加过滤规则
                clone_in_fabric=self.cfg.clone_in_fabric,
            )

        # 提醒用户：异构场景或 CPU 仿真时需手动调用 filter_collisions
        if (not self.cfg.replicate_physics and self.cfg.filter_collisions) or self.device == "cpu":
            if not self._is_scene_setup_from_cfg():
                logger.warning(
                    "Collision filtering can only be automatically enabled when replicate_physics=True and using GPU"
                    " simulation. Please call scene.filter_collisions(global_prim_paths) to filter collisions across"
                    " environments."
                )

        # 同构场景中，env_origins 此前为 None，在此完成赋值
        # 异构场景已在 __init__ 预克隆时设置过，此处跳过
        if self._default_env_origins is None:
            self._default_env_origins = torch.tensor(env_origins, device=self.device, dtype=torch.float32)

    def filter_collisions(self, global_prim_paths: list[str] | None = None):
        """Filter environments collisions.

        Disables collisions between the environments in ``/World/envs/env_.*`` and enables collisions with the prims
        in global prim paths (e.g. ground plane).

        Args:
            global_prim_paths: A list of global prim paths to enable collisions with.
                Defaults to None, in which case no global prim paths are considered.

        说明：
          碰撞过滤规则存储在 /World/collisions 路径下。
          若该路径已存在（说明已过滤过），直接返回，避免重复过滤。

          global_prim_paths：全局 prim（如地面平面 /World/ground）需要与
          所有环境保持碰撞，因此不加入过滤列表。
          反之，/World/envs/env_i 内的所有 prim 之间（跨环境）的碰撞会被禁用。
        """
        # 处理 None 输入
        if global_prim_paths is None:
            global_prim_paths = []
        else:
            # 去重（同一路径出现多次时只保留一份）
            global_prim_paths = list(set(global_prim_paths))

        # 幂等保护：若 /World/collisions 已存在，说明已经做过过滤，直接返回
        if self.stage.GetPrimAtPath("/World/collisions"):
            return

        # 补充全局路径
        if len(self._global_prim_paths) < 1:
            self._global_prim_paths += global_prim_paths

        # 调用 Cloner 的碰撞过滤接口：
        #   - physics_scene_path：物理场景 prim 路径（如 /World/physicsScene）
        #   - "/World/collisions"：过滤规则存储路径
        #   - env_prim_paths：所有环境路径（过滤环境间碰撞）
        #   - global_paths：全局路径（这些路径与所有环境保持碰撞）
        self.cloner.filter_collisions(
            self.physics_scene_path,
            "/World/collisions",
            self.env_prim_paths,
            global_paths=self._global_prim_paths,
        )

    def __str__(self) -> str:
        """Returns a string representation of the scene."""
        msg = f"<class {self.__class__.__name__}>\n"
        msg += f"\tNumber of environments: {self.cfg.num_envs}\n"
        msg += f"\tEnvironment spacing   : {self.cfg.env_spacing}\n"
        msg += f"\tSource prim name      : {self.env_prim_paths[0]}\n"
        msg += f"\tGlobal prim paths     : {self._global_prim_paths}\n"
        msg += f"\tReplicate physics     : {self.cfg.replicate_physics}"
        return msg

    """
    Properties.
    """

    @property
    def physics_scene_path(self) -> str:
        """The path to the USD Physics Scene.

        USD stage 上 PhysxSceneAPI 所在的 prim 路径（如 /World/physicsScene）。
        首次访问时懒查找（遍历 stage 所有 prim），之后缓存结果。
        若 stage 中没有物理场景 prim 则抛 RuntimeError。
        """
        if self._physics_scene_path is None:
            # 遍历整个 USD stage，找到第一个携带 PhysxSceneAPI 的 prim
            for prim in self.stage.Traverse():
                if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                    self._physics_scene_path = prim.GetPrimPath().pathString
                    logger.info(f"Physics scene prim path: {self._physics_scene_path}")
                    break
            if self._physics_scene_path is None:
                raise RuntimeError("No physics scene found! Please make sure one exists.")
        return self._physics_scene_path

    @property
    def physics_dt(self) -> float:
        """The physics timestep of the scene.

        单位：秒。等于 SimulationContext 中配置的 sim.dt（如 1/120）。
        """
        return sim_utils.SimulationContext.instance().get_physics_dt()  # pyright: ignore [reportOptionalMemberAccess]

    @property
    def device(self) -> str:
        """The device on which the scene is created.

        通常为 "cuda:0"（GPU 仿真）或 "cpu"（CPU 仿真）。
        由 SimulationContext 在启动时确定，场景生命周期内不变。
        """
        return sim_utils.SimulationContext.instance().device  # pyright: ignore [reportOptionalMemberAccess]

    @property
    def env_ns(self) -> str:
        """The namespace ``/World/envs`` in which all environments created.

        The environments are present w.r.t. this namespace under "env_{N}" prim,
        where N is a natural number.

        所有并行环境的父命名空间，固定为 "/World/envs"。
        各子环境路径形如：/World/envs/env_0, /World/envs/env_1, ...
        """
        return "/World/envs"

    @property
    def env_regex_ns(self) -> str:
        """The namespace ``/World/envs/env_.*`` in which all environments created.

        正则命名空间，用于 IsaacLab asset/sensor 的 prim_path 通配符展开。
        cfg 中的 prim_path="{ENV_REGEX_NS}/robot" 会被替换为
        "/World/envs/env_.*/robot"，然后 PhysX 按正则匹配所有环境的对应 prim。
        """
        return f"{self.env_ns}/env_.*"

    @property
    def num_envs(self) -> int:
        """The number of environments handled by the scene."""
        return self.cfg.num_envs

    @property
    def env_origins(self) -> torch.Tensor:
        """The origins of the environments in the scene. Shape is (num_envs, 3).

        各环境世界坐标系下的原点位置，单位：米。
        优先使用地形（TerrainImporter）提供的 env_origins（地形可定义非均匀间距）；
        若无地形，则使用 GridCloner 按 env_spacing 均匀排列的原点坐标。
        """
        if self._terrain is not None:
            return self._terrain.env_origins
        else:
            return self._default_env_origins

    @property
    def terrain(self) -> TerrainImporter | None:
        """The terrain in the scene. If None, then the scene has no terrain.

        Note:
            We treat terrain separate from :attr:`extras` since terrains define environment origins and are
            handled differently from other miscellaneous entities.

        地形单独处理的原因：
          TerrainImporter 不仅是视觉/碰撞资产，还负责定义每个环境的世界原点
          (env_origins)，这影响机器人初始位置的计算，因此不归入 extras。
        """
        return self._terrain

    @property
    def articulations(self) -> dict[str, Articulation]:
        """A dictionary of articulations in the scene.

        key=cfg 字段名（如 "robot"），value=Articulation 运行时对象。
        Articulation 封装了 PhysX ArticulationView，提供关节状态读写接口。
        """
        return self._articulations

    @property
    def deformable_objects(self) -> dict[str, DeformableObject]:
        """A dictionary of deformable objects in the scene.

        可变形体（布料、软体），底层使用 PhysX FEM（有限元方法）。
        注意：deformable objects 不支持 replicate_physics 优化，需设置为 False。
        """
        return self._deformable_objects

    @property
    def rigid_objects(self) -> dict[str, RigidObject]:
        """A dictionary of rigid objects in the scene.

        刚体（无关节，只有根 link），提供 root_pose / root_vel 读写接口。
        """
        return self._rigid_objects

    @property
    def rigid_object_collections(self) -> dict[str, RigidObjectCollection]:
        """A dictionary of rigid object collections in the scene.

        刚体集合：将多个同类刚体合并为一个视图，支持批量操作。
        适用于抓取目标堆（一批独立的苹果/方块等）。
        """
        return self._rigid_object_collections

    @property
    def sensors(self) -> dict[str, SensorBase]:
        """A dictionary of the sensors in the scene, such as cameras and contact reporters.

        key=cfg 字段名，value=SensorBase 子类实例（如 Camera, RayCaster, ContactSensor）。
        传感器数据通过 sensor.data 属性访问（lazy 模式下首次访问时触发计算）。
        """
        return self._sensors

    @property
    def surface_grippers(self) -> dict[str, SurfaceGripper]:
        """A dictionary of the surface grippers in the scene.

        表面夹爪：通过真空/磁力模拟抓取的效应器，不属于关节也不属于普通传感器。
        """
        return self._surface_grippers

    @property
    def extras(self) -> dict[str, XformPrimView]:
        """A dictionary of miscellaneous simulation objects that neither inherit from assets nor sensors.

        The keys are the names of the miscellaneous objects, and the values are the
        :class:`~isaaclab.sim.views.XformPrimView` instances of the corresponding prims.

        As an example, lights or other props in the scene that do not have any attributes or properties that you
        want to alter at runtime can be added to this dictionary.

        Note:
            These are not reset or updated by the scene. They are mainly other prims that are not necessarily
            handled by the interactive scene, but are useful to be accessed by the user.

        杂项说明：
          AssetBaseCfg 类型的字段（灯光、静态背景道具等）会被加入 extras。
          它们以 XformPrimView 形式存储（支持批量变换操作），但场景本身
          不调用它们的 reset/update，逻辑由用户在需要时手动操作。
        """
        return self._extras

    @property
    def state(self) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """A dictionary of the state of the scene entities in the simulation world frame.

        Please refer to :meth:`get_state` for the format.

        返回世界坐标系下的绝对状态（非相对于环境原点）。
        等价于 scene.get_state(is_relative=False)。
        """
        return self.get_state(is_relative=False)

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the scene entities.

        Args:
            env_ids: The indices of the environments to reset.
                Defaults to None (all instances).

        说明：
          reset 只重置各实体的「内部缓冲」（data 属性中的 Tensor），
          并不直接写入 PhysX。实际物理状态的更新需要调用 write_data_to_sim()。

          调用顺序：articulations → deformable → rigid → surface_grippers
                   → rigid_collections → sensors
          传感器在最后重置，确保物理资产状态已更新后传感器获取一致状态。
        """
        # 重置所有关节体的内部缓冲（关节位置/速度目标、根位姿缓冲等）
        for articulation in self._articulations.values():
            articulation.reset(env_ids)
        # 重置可变形体
        for deformable_object in self._deformable_objects.values():
            deformable_object.reset(env_ids)
        # 重置刚体
        for rigid_object in self._rigid_objects.values():
            rigid_object.reset(env_ids)
        # 重置表面夹爪
        for surface_gripper in self._surface_grippers.values():
            surface_gripper.reset(env_ids)
        # 重置刚体集合
        for rigid_object_collection in self._rigid_object_collections.values():
            rigid_object_collection.reset(env_ids)
        # 重置传感器（清空历史数据缓冲，让下一次 update 重新从 PhysX 读取）
        for sensor in self._sensors.values():
            sensor.reset(env_ids)

    def write_data_to_sim(self):
        """Writes the data of the scene entities to the simulation.

        说明：
          将各实体内部缓冲（如 joint position target、root pose）写入 PhysX。
          必须在 sim.step() 之前调用，确保 PhysX 使用最新的控制目标。

          传感器不需要写入（传感器只读出数据，不写入），因此此处不遍历 _sensors。
        """
        # 将关节体目标（位置/速度/力矩）写入 PhysX
        for articulation in self._articulations.values():
            articulation.write_data_to_sim()
        # 将可变形体节点状态写入 PhysX
        for deformable_object in self._deformable_objects.values():
            deformable_object.write_data_to_sim()
        # 将刚体根状态写入 PhysX
        for rigid_object in self._rigid_objects.values():
            rigid_object.write_data_to_sim()
        # 将表面夹爪命令写入 PhysX
        for surface_gripper in self._surface_grippers.values():
            surface_gripper.write_data_to_sim()
        # 将刚体集合状态写入 PhysX
        for rigid_object_collection in self._rigid_object_collections.values():
            rigid_object_collection.write_data_to_sim()

    def update(self, dt: float) -> None:
        """Update the scene entities.

        Args:
            dt: The amount of time passed from last :meth:`update` call.
                单位：秒（通常 = physics_dt，即 sim.dt）。

        说明：
          在 sim.step() 之后调用，从 PhysX 读取最新状态并更新各实体的 data 缓冲。

          传感器更新带 force_recompute 参数：
            - lazy_sensor_update=True（默认）：force_recompute=False，
              传感器只在 sensor.data 被访问时才触发计算（懒更新）。
            - lazy_sensor_update=False：force_recompute=True，
              每次 update() 强制重算所有传感器（即时更新）。

          调用顺序：articulations → deformable → rigid → rigid_collections → grippers → sensors
          传感器在最后更新，确保读取的物理状态是最新的（本帧 PhysX 推进后的结果）。
        """
        # 从 PhysX 读取关节体最新状态（关节位置/速度/力矩、根位姿/速度）
        for articulation in self._articulations.values():
            articulation.update(dt)
        # 从 PhysX 读取可变形体节点最新位置/速度
        for deformable_object in self._deformable_objects.values():
            deformable_object.update(dt)
        # 从 PhysX 读取刚体最新根状态
        for rigid_object in self._rigid_objects.values():
            rigid_object.update(dt)
        # 从 PhysX 读取刚体集合最新状态
        for rigid_object_collection in self._rigid_object_collections.values():
            rigid_object_collection.update(dt)
        # 更新表面夹爪状态
        for surface_gripper in self._surface_grippers.values():
            surface_gripper.update(dt)
        # 更新传感器：force_recompute=True 时强制重算，否则懒更新
        for sensor in self._sensors.values():
            sensor.update(dt, force_recompute=not self.cfg.lazy_sensor_update)

    """
    Operations: Scene State.
    """

    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None = None,
        is_relative: bool = False,
    ):
        """Resets the entities in the scene to the provided state.

        Args:
            state: The state to reset the scene entities to. Please refer to :meth:`get_state` for the format.
            env_ids: The indices of the environments to reset. Defaults to None, in which case
                all environment instances are reset.
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.

        说明：
          用于从录制数据或仿真回放恢复场景到指定状态，配合 get_state 使用。

          is_relative=True：state 中的位置是相对于各环境原点的坐标，
          读取时需加上 env_origins[env_ids] 转换为世界坐标。
          is_relative=False：state 中直接是世界坐标。

          关节体重置时同时设置关节位置和速度的控制目标（PD 控制假设），
          使机器人在接管控制前保持目标姿态，不会因重力或惯性漂移。

          最后调用 write_data_to_sim() 将所有状态立即写入 PhysX，
          确保下一次 sim.step() 从正确状态开始。
        """
        # 解析 env_ids（None → 全部环境）
        if env_ids is None:
            env_ids = self._ALL_INDICES

        # ── 关节体 ─────────────────────────────────────────────────────────────
        for asset_name, articulation in self._articulations.items():
            asset_state = state["articulation"][asset_name]

            # 根位姿（位置 + 四元数朝向）
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self.env_origins[env_ids]  # 转换为世界坐标
            root_velocity = asset_state["root_velocity"].clone()
            articulation.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            articulation.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)

            # 关节状态（位置 + 速度）
            joint_position = asset_state["joint_position"].clone()
            joint_velocity = asset_state["joint_velocity"].clone()
            articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)

            # FIXME: 这里假设所有关节都是 PD 控制（位置/速度目标），
            #   对于力矩控制的关节（effort control）此处设置无意义。
            articulation.set_joint_position_target(joint_position, env_ids=env_ids)
            articulation.set_joint_velocity_target(joint_velocity, env_ids=env_ids)

        # ── 可变形体 ────────────────────────────────────────────────────────────
        for asset_name, deformable_object in self._deformable_objects.items():
            asset_state = state["deformable_object"][asset_name]
            nodal_position = asset_state["nodal_position"].clone()
            if is_relative:
                nodal_position[:, :3] += self.env_origins[env_ids]
            nodal_velocity = asset_state["nodal_velocity"].clone()
            deformable_object.write_nodal_pos_to_sim(nodal_position, env_ids=env_ids)
            deformable_object.write_nodal_velocity_to_sim(nodal_velocity, env_ids=env_ids)

        # ── 刚体 ───────────────────────────────────────────────────────────────
        for asset_name, rigid_object in self._rigid_objects.items():
            asset_state = state["rigid_object"][asset_name]
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone()
            rigid_object.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)

        # ── 表面夹爪 ───────────────────────────────────────────────────────────
        for asset_name, surface_gripper in self._surface_grippers.items():
            asset_state = state["gripper"][asset_name]
            surface_gripper.set_grippers_command(asset_state)

        # 将所有修改立即写入 PhysX，确保下一步物理推进从正确初态出发
        self.write_data_to_sim()

    def get_state(self, is_relative: bool = False) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """Returns the state of the scene entities.

        Based on the type of the entity, the state comprises of different components.

        * For an articulation, the state comprises of the root pose, root velocity, and joint position and velocity.
        * For a deformable object, the state comprises of the nodal position and velocity.
        * For a rigid object, the state comprises of the root pose and root velocity.

        The returned state is a dictionary with the following format:

        .. code-block:: python

            {
                "articulation": {
                    "entity_1_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                        "joint_position": torch.Tensor,
                        "joint_velocity": torch.Tensor,
                    },
                    "entity_2_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                        "joint_position": torch.Tensor,
                        "joint_velocity": torch.Tensor,
                    },
                },
                "deformable_object": {
                    "entity_3_name": {
                        "nodal_position": torch.Tensor,
                        "nodal_velocity": torch.Tensor,
                    }
                },
                "rigid_object": {
                    "entity_4_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                    }
                },
            }

        where ``entity_N_name`` is the name of the entity registered in the scene.

        Args:
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.

        Returns:
            A dictionary of the state of the scene entities.

        说明：
          is_relative=True：返回相对于各环境原点的坐标（用于跨场景迁移或标准化数据）。
          is_relative=False：返回世界坐标系下的绝对坐标。

          所有 Tensor 均通过 .clone() 返回拷贝，避免外部修改污染内部缓冲。
        """
        state = dict()

        # ── 关节体状态 ─────────────────────────────────────────────────────────
        state["articulation"] = dict()
        for asset_name, articulation in self._articulations.items():
            asset_state = dict()
            asset_state["root_pose"] = articulation.data.root_pose_w.clone()  # 世界系位姿，shape=[num_envs, 7]
            if is_relative:
                asset_state["root_pose"][:, :3] -= self.env_origins  # 减去环境原点 → 相对坐标
            asset_state["root_velocity"] = articulation.data.root_vel_w.clone()  # 世界系速度，shape=[num_envs, 6]
            asset_state["joint_position"] = articulation.data.joint_pos.clone()  # 关节位置，shape=[num_envs, n_joints]
            asset_state["joint_velocity"] = articulation.data.joint_vel.clone()  # 关节速度，shape=[num_envs, n_joints]
            state["articulation"][asset_name] = asset_state

        # ── 可变形体状态 ────────────────────────────────────────────────────────
        state["deformable_object"] = dict()
        for asset_name, deformable_object in self._deformable_objects.items():
            asset_state = dict()
            asset_state["nodal_position"] = deformable_object.data.nodal_pos_w.clone()  # 节点位置，shape=[num_envs, n_nodes, 3]
            if is_relative:
                asset_state["nodal_position"][:, :3] -= self.env_origins
            asset_state["nodal_velocity"] = deformable_object.data.nodal_vel_w.clone()  # 节点速度
            state["deformable_object"][asset_name] = asset_state

        # ── 刚体状态 ───────────────────────────────────────────────────────────
        state["rigid_object"] = dict()
        for asset_name, rigid_object in self._rigid_objects.items():
            asset_state = dict()
            asset_state["root_pose"] = rigid_object.data.root_pose_w.clone()  # 世界系位姿，shape=[num_envs, 7]
            if is_relative:
                asset_state["root_pose"][:, :3] -= self.env_origins
            asset_state["root_velocity"] = rigid_object.data.root_vel_w.clone()  # 世界系速度，shape=[num_envs, 6]
            state["rigid_object"][asset_name] = asset_state

        # ── 表面夹爪状态 ───────────────────────────────────────────────────────
        state["gripper"] = dict()
        for asset_name, gripper in self._surface_grippers.items():
            # 夹爪状态（吸附/释放命令），shape 视具体实现而定
            state["gripper"][asset_name] = gripper.state.clone()

        return state

    """
    Operations: Iteration.
    """

    def keys(self) -> list[str]:
        """Returns the keys of the scene entities.

        Returns:
            The keys of the scene entities.

        返回所有已注册实体的名称列表，包括 "terrain" 以及所有
        articulations / deformable_objects / rigid_objects / rigid_object_collections /
        sensors / surface_grippers / extras 中的 key。
        """
        all_keys = ["terrain"]
        for asset_family in [
            self._articulations,
            self._deformable_objects,
            self._rigid_objects,
            self._rigid_object_collections,
            self._sensors,
            self._surface_grippers,
            self._extras,
        ]:
            all_keys += list(asset_family.keys())
        return all_keys

    def __getitem__(self, key: str) -> Any:
        """Returns the scene entity with the given key.

        Args:
            key: The key of the scene entity.

        Returns:
            The scene entity.

        支持 scene["robot"] 语法，按名称查找实体。
        查找顺序：terrain → articulations → deformable → rigid → rigid_collections
                 → sensors → surface_grippers → extras
        若未找到则抛 KeyError 并列出所有可用 key，方便调试。
        """
        # 特殊处理：地形不在任何字典中，单独判断
        if key == "terrain":
            return self._terrain

        all_keys = ["terrain"]
        # 依次在各实体字典中查找
        for asset_family in [
            self._articulations,
            self._deformable_objects,
            self._rigid_objects,
            self._rigid_object_collections,
            self._sensors,
            self._surface_grippers,
            self._extras,
        ]:
            out = asset_family.get(key)
            # 找到则立即返回，避免不必要的遍历
            if out is not None:
                return out
            all_keys += list(asset_family.keys())

        # 所有字典都未找到 → 报错并列出可用 key，方便用户调试
        raise KeyError(f"Scene entity with key '{key}' not found. Available Entities: '{all_keys}'")

    """
    Internal methods.
    """

    def _is_scene_setup_from_cfg(self) -> bool:
        """Check if scene entities are setup from the config or not.

        Returns:
            True if scene entities are setup from the config, False otherwise.

        实现原理：
          遍历 cfg.__dict__ 的所有字段，跳过 InteractiveSceneCfg 基类自身定义的字段
          （num_envs, env_spacing, lazy_sensor_update, replicate_physics, filter_collisions, clone_in_fabric）
          以及值为 None 的字段。
          若有任何「用户自定义字段」（即子类新增的实体配置字段），则返回 True。

          只要子类 SceneCfg 中有任何实体（terrain/robot/sensor/...），此函数返回 True，
          触发 _add_entities_from_cfg。
        """
        return any(
            not (asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None)
            for asset_name, asset_cfg in self.cfg.__dict__.items()
        )

    def _add_entities_from_cfg(self):
        """Add scene entities from the config.

        核心初始化方法：遍历 cfg 中所有用户自定义字段，
        按字段类型将实体实例化并分类存储到对应字典。

        处理流程（按 cfg 字段声明顺序）：
          1. 跳过 InteractiveSceneCfg 基类字段和 None 值字段
          2. 将 prim_path 中的 {ENV_REGEX_NS} 替换为实际正则命名空间
          3. 按 cfg 类型分发到对应处理分支：
               TerrainImporterCfg      → _terrain
               ArticulationCfg         → _articulations
               DeformableObjectCfg     → _deformable_objects
               RigidObjectCfg          → _rigid_objects
               RigidObjectCollectionCfg→ _rigid_object_collections
               SurfaceGripperCfg       → _surface_grippers
               SensorBaseCfg           → _sensors（含 FrameTransformer / ContactSensor 特殊处理）
               AssetBaseCfg            → _extras（spawn 后存为 XformPrimView）
          4. 遇到 collision_group=-1 的实体，将其路径加入 _global_prim_paths
             （表示该实体要与所有环境保持碰撞，如地面平面）

        注意：cfg 字段的声明顺序即为实体实例化顺序，
        InteractiveSceneCfg docstring 推荐顺序：terrain → 物理资产 → 传感器 → 灯光。
        """
        # 重置全局碰撞路径列表
        self._global_prim_paths = list()

        # 遍历 cfg 的所有字段（configclass 保证顺序与声明顺序一致）
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            # 跳过基类字段（num_envs, env_spacing 等非实体字段）和空字段
            if asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
                continue

            # 将 prim_path 中的 {ENV_REGEX_NS} 替换为 /World/envs/env_.*
            # 这样 PhysX/SensorView 才能通过正则匹配到所有环境中的对应 prim
            if hasattr(asset_cfg, "prim_path"):
                asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)

            # ── 地形 ──────────────────────────────────────────────────────────
            if isinstance(asset_cfg, TerrainImporterCfg):
                # 地形需要知道环境数量和间距（用于生成多地形块）
                asset_cfg.num_envs = self.cfg.num_envs
                asset_cfg.env_spacing = self.cfg.env_spacing
                # 实例化 TerrainImporter（或其子类，由 asset_cfg.class_type 决定）
                self._terrain = asset_cfg.class_type(asset_cfg)

            # ── 关节体（机器人） ───────────────────────────────────────────────
            elif isinstance(asset_cfg, ArticulationCfg):
                self._articulations[asset_name] = asset_cfg.class_type(asset_cfg)

            # ── 可变形体 ────────────────────────────────────────────────────────
            elif isinstance(asset_cfg, DeformableObjectCfg):
                self._deformable_objects[asset_name] = asset_cfg.class_type(asset_cfg)

            # ── 刚体 ──────────────────────────────────────────────────────────
            elif isinstance(asset_cfg, RigidObjectCfg):
                self._rigid_objects[asset_name] = asset_cfg.class_type(asset_cfg)

            # ── 刚体集合 ────────────────────────────────────────────────────────
            elif isinstance(asset_cfg, RigidObjectCollectionCfg):
                # 集合内每个子刚体也需要展开 {ENV_REGEX_NS}
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    rigid_object_cfg.prim_path = rigid_object_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                self._rigid_object_collections[asset_name] = asset_cfg.class_type(asset_cfg)
                # collision_group=-1 的子刚体路径加入全局碰撞列表
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    if hasattr(rigid_object_cfg, "collision_group") and rigid_object_cfg.collision_group == -1:
                        asset_paths = sim_utils.find_matching_prim_paths(rigid_object_cfg.prim_path)
                        self._global_prim_paths += asset_paths

            # ── 表面夹爪 ────────────────────────────────────────────────────────
            elif isinstance(asset_cfg, SurfaceGripperCfg):
                self._surface_grippers[asset_name] = asset_cfg.class_type(asset_cfg)

            # ── 传感器 ─────────────────────────────────────────────────────────
            elif isinstance(asset_cfg, SensorBaseCfg):
                # FrameTransformer：其 target_frames 中的 prim_path 也需要展开正则
                if isinstance(asset_cfg, FrameTransformerCfg):
                    updated_target_frames = []
                    for target_frame in asset_cfg.target_frames:
                        target_frame.prim_path = target_frame.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                        updated_target_frames.append(target_frame)
                    asset_cfg.target_frames = updated_target_frames
                # ContactSensor：过滤路径（filter_prim_paths_expr）也需展开
                elif isinstance(asset_cfg, ContactSensorCfg):
                    updated_filter_prim_paths_expr = []
                    for filter_prim_path in asset_cfg.filter_prim_paths_expr:
                        updated_filter_prim_paths_expr.append(filter_prim_path.format(ENV_REGEX_NS=self.env_regex_ns))
                    asset_cfg.filter_prim_paths_expr = updated_filter_prim_paths_expr
                # VisuoTactileSensor（触觉传感器，临时贡献实现）：
                # 其内嵌相机和接触对象路径也需展开
                elif isinstance(asset_cfg, VisuoTactileSensorCfg):
                    if hasattr(asset_cfg, "camera_cfg") and asset_cfg.camera_cfg is not None:
                        asset_cfg.camera_cfg.prim_path = asset_cfg.camera_cfg.prim_path.format(
                            ENV_REGEX_NS=self.env_regex_ns
                        )
                    if (
                        hasattr(asset_cfg, "contact_object_prim_path_expr")
                        and asset_cfg.contact_object_prim_path_expr is not None
                    ):
                        asset_cfg.contact_object_prim_path_expr = asset_cfg.contact_object_prim_path_expr.format(
                            ENV_REGEX_NS=self.env_regex_ns
                        )

                # 实例化传感器（Camera / RayCaster / ContactSensor / FrameTransformer 等）
                self._sensors[asset_name] = asset_cfg.class_type(asset_cfg)

            # ── 杂项资产（灯光、静态道具等） ────────────────────────────────────
            elif isinstance(asset_cfg, AssetBaseCfg):
                # spawn 不为 None 时，在 prim_path 处实例化对应 USD prim
                # （如 DistantLightCfg.func 会创建灯光，UsdFileCfg.func 会加载 USD 文件）
                if asset_cfg.spawn is not None:
                    asset_cfg.spawn.func(
                        asset_cfg.prim_path,
                        asset_cfg.spawn,
                        translation=asset_cfg.init_state.pos,    # 初始位置
                        orientation=asset_cfg.init_state.rot,    # 初始朝向（四元数 wxyz）
                    )
                # 将对应 prim 包装为 XformPrimView，支持批量变换操作（可选）
                self._extras[asset_name] = XformPrimView(asset_cfg.prim_path, device=self.device, stage=self.stage)

            else:
                # 未知 cfg 类型 → 报错，防止新增资产类型未被处理时静默跳过
                raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")

            # ── 全局碰撞路径收集 ───────────────────────────────────────────────
            # collision_group=-1 表示该实体需要与所有环境交叉碰撞（如全局地面）
            # 将其所有匹配 prim 路径加入列表，后续 filter_collisions 时作为白名单
            if hasattr(asset_cfg, "collision_group") and asset_cfg.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
                self._global_prim_paths += asset_paths
