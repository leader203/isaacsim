# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ============================================================
# InteractiveSceneCfg — 交互式仿真场景配置基类
#
# 职责：
#   定义 InteractiveScene 的顶层配置，用户通过继承此类并添加
#   字段来声明场景中的所有实体（地形、机器人、传感器、灯光等）。
#   InteractiveScene 在初始化时遍历子类的所有字段，按字段类型
#   自动实例化对应的 USD prim 和物理/传感器组件。
#
# 场景实体添加顺序（顺序敏感）：
#   推荐按以下顺序声明字段，确保依赖关系正确（如传感器依赖地形存在）：
#     1. terrain    — 地形（TerrainImporterCfg）
#     2. articulations / rigid_bodies — 物理资产（机器人、物体）
#     3. sensors    — 传感器（相机、激光雷达、力传感器等）
#     4. lights / extras — 非物理资产（灯光、静态背景等）
#
# 多环境并行机制（Cloner）：
#   IsaacLab 使用 Omniverse Cloner API 将单个原型环境复制为
#   num_envs 份，每份占据独立的世界空间区域（由 env_spacing 决定）。
#   prim_path 中的 {ENV_REGEX_NS} 占位符会被展开为各环境的命名空间
#   （如 /World/envs/env_0, /World/envs/env_1, ...）。
#
# 与 env_cfg 的关系：
#   ManagerBasedEnvCfg.scene 字段类型为 InteractiveSceneCfg，
#   用户定义的 SceneCfg 子类赋值给 scene 后，由 InteractiveScene
#   读取并在 __init__ 中构建整个 USD stage。
#
# 使用示例见类 docstring。
# ============================================================

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass


@configclass
class InteractiveSceneCfg:
    """Configuration for the interactive scene.

    The users can inherit from this class to add entities to their scene. This is then parsed by the
    :class:`InteractiveScene` class to create the scene.

    .. note::
        The adding of entities to the scene is sensitive to the order of the attributes in the configuration.
        Please make sure to add the entities in the order you want them to be added to the scene.
        The recommended order of specification is terrain, physics-related assets (articulations and rigid bodies),
        sensors and non-physics-related assets (lights).

    For example, to add a robot to the scene, the user can create a configuration class as follows:

    .. code-block:: python

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors.ray_caster import GridPatternCfg, RayCasterCfg
        from isaaclab.utils import configclass

        from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


        @configclass
        class MySceneCfg(InteractiveSceneCfg):
            # terrain - flat terrain plane
            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
            )

            # articulation - robot 1
            robot_1 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
            # articulation - robot 2
            robot_2 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
            robot_2.init_state.pos = (0.0, 1.0, 0.6)

            # sensor - ray caster attached to the base of robot 1 that scans the ground
            height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot_1/base",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
                debug_vis=True,
                mesh_prim_paths=["/World/ground"],
            )

            # extras - light
            light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
            )

    """

    num_envs: int = MISSING
    """Number of environment instances handled by the scene. [必填]

    说明：
      并行运行的子环境数量，决定了场景中克隆出多少份原型环境。

      典型取值：
        训练阶段：1024 ~ 8192（越多越能充分利用 GPU 并行度，加速数据采集）
        推理/录制：1（单环境，便于人工遥操作和观察）
        调试阶段：4 ~ 16（较少环境，便于观察每个环境状态）

      注意：num_envs 越大，GPU 显存占用越高。
      超过显存上限时需降低 num_envs 或减小场景复杂度。
    """

    env_spacing: float = MISSING
    """Spacing between environments (in meters). [必填]

    说明：
      相邻子环境原点之间的距离（米），用于在世界空间中排列克隆出的环境，
      防止不同环境的几何体和物理碰撞体相互重叠。

      此值仅在 num_envs > 1 时生效；num_envs=1 时场景只有一份，无需间距。

      典型取值：
        小型场景（机械臂工作台）：1.5 ~ 2.5 m
        中型场景（卫生间/房间）：2.5 ~ 5.0 m
        大型场景（户外地形）：10.0 ~ 20.0 m

      设置原则：env_spacing 应大于场景中最大实体的外接圆直径，
      确保不同环境的碰撞体不会跨环境干扰（配合 filter_collisions=True）。

    This is the default distance between environment origins in the scene. Used only when the
    number of environments is greater than one.
    """

    lazy_sensor_update: bool = True
    """Whether to update sensors only when they are accessed. Default is True.

    说明：
      控制传感器数据的更新策略：

      True（默认，懒更新）：
        传感器数据只在外部代码访问 sensor.data 属性时才触发更新计算。
        优点：避免每步都更新所有传感器，节省未使用传感器的 GPU 计算开销。
        适合：大多数场景，传感器并非每步都被读取（如相机只在特定条件下拍摄）。

      False（即时更新）：
        每次调用 scene.update() 时所有传感器立即更新，无论是否被访问。
        优点：数据始终是最新的，不存在访问延迟。
        适合：传感器每步都被使用，或需要精确控制更新时机的场景。

    If true, the sensor data is only updated when their attribute ``data`` is accessed. Otherwise, the sensor
    data is updated every time sensors are updated.
    """

    replicate_physics: bool = True
    """Enable/disable replication of physics schemas when using the Cloner APIs. Default is True.

    说明：
      控制 Cloner API 克隆场景时是否复用物理 schema（物理属性定义）：

      True（默认，推荐）：
        所有克隆环境共享同一套物理 schema，PhysX 能够高效批量处理，
        大幅加快场景初始化和物理解析速度。
        适合：绝大多数标准资产（ArticulationCfg / RigidObjectCfg）。

      False：
        每个环境拥有独立的物理 schema 实例，允许各环境使用不同的物理属性
        （如不同质量、不同摩擦系数的独立实例）。
        代价：场景初始化和解析变慢，GPU 内存占用更高。
        适合：需要每环境独立物理属性、或使用变形体（deformable objects）等
              PhysX 尚不支持优化解析的特殊资产类型。

    If True, the simulation will have the same asset instances (USD prims) in all the cloned environments.
    Internally, this ensures optimization in setting up the scene and parsing it via the physics stage parser.

    If False, the simulation allows having separate asset instances (USD prims) in each environment.
    This flexibility comes at a cost of slowdowns in setting up and parsing the scene.

    .. note::
        Optimized parsing of certain prim types (such as deformable objects) is not currently supported
        by the physics engine. In these cases, this flag needs to be set to False.
    """

    filter_collisions: bool = True
    """Enable/disable collision filtering between cloned environments. Default is True.

    说明：
      控制不同子环境之间是否过滤碰撞（即不同环境的物体不会相互碰撞）：

      True（默认，推荐）：
        Cloner 自动为不同环境之间添加碰撞过滤规则，确保环境 i 的机器人
        不会与环境 j 的障碍物发生碰撞，各环境物理上完全隔离。
        适合：所有标准并行 RL 训练场景。

      False：
        不过滤跨环境碰撞，不同环境的实体可能相互碰撞。
        通常不需要此设置，仅用于特殊的跨环境交互仿真。

      注意事项：
        自动过滤仅在 replicate_physics=True 的 direct workflow 下生效。
        若 replicate_physics=False 且需要过滤碰撞，需手动调用
        scene.filter_collisions()。

    If True, collisions will not occur between cloned environments.

    If False, the simulation will generate collisions between environments.

    .. note::
        Collisions can only be filtered automatically in direct workflows when physics replication is enabled.
        If :attr:`replicated_physics` is ``False`` and collision filtering is desired, make sure to call
        ``scene.filter_collisions()``.
    """

    clone_in_fabric: bool = False
    """Enable/disable cloning in fabric. Default is False.

    说明：
      控制是否使用 Omniverse Fabric（USDRT）进行场景克隆：

      False（默认）：
        使用标准 USD API 进行克隆，兼容性最好，可通过常规 USD stage API 访问所有 prim。
        适合：大多数场景，尤其是需要在克隆后通过 USD API 修改 prim 属性的工作流。

      True（Fabric 模式）：
        使用 Omniverse Fabric 进行克隆，速度更快（尤其在环境数量极多时），
        但克隆后的 stage 只能通过 USDRT API 访问，不能使用标准 USD Python API。
        代价：灵活性降低，调试难度增加。
        适合：num_envs 极大（> 4096）、对场景构建速度有极高要求的场景。

      依赖关系：
        clone_in_fabric=True 要求 replicate_physics=True，
        若 replicate_physics=False，则 clone_in_fabric 会被自动强制设为 False。

    Omniverse Fabric is a more optimized method for performing cloning in scene creation. This reduces the time
    taken to create the scene. However, it limits flexibility in accessing the stage through USD APIs and instead,
    the stage must be accessed through USDRT.

    .. note::
        Cloning in fabric can only be enabled if :attr:`replicated_physics` is also enabled.
        If :attr:`replicated_physics` is ``False``, cloning in Fabric will automatically
        default to ``False``.

    """
