# scenes — 场景资产配置模块

本目录定义 `allstar_toilet` 任务中所有**静态场景**（地面、房间、固定装置、可开合铰链体）
的 `InteractiveSceneCfg` 基类。上层的 env_cfg 通过继承这些基类，在其上叠加机械臂、摄像头、
接触传感器等动态资产，构成完整的仿真环境。

---

## 目录结构

```
assets/scenes/
├── __init__.py           # 统一导出 5 个公开配置名
├── toilet_scene_cfg.py   # 卫生间/隔间场景（3 个变体，按迭代顺序）
├── real_scene_cfg.py     # 申鑫福真实场景扫描（2 个变体）
└── scenes.md             # 本文档
```

USD 文件存放于同级 `../data/scenes/` 目录下：

```
data/scenes/
├── toilet_v0311/
│   └── 20250311B.usd          # 完整卫生间房间（已弃用）
├── pallet/
│   └── 20250411/
│       ├── sen0324.usd         # 单体隔间场景（主力版本）
│       └── sen0401.usd         # 备用版本（当前未使用）
└── real/
    └── shenxinfu/
        ├── 841837/start.usda   # 静态完整房间
        ├── 841838/start.usda   # 隔间内部（不含铰链体）
        ├── Toilet_YREQ63RVAJ2SUPTULA888888_YlDCP/
        │   └── *.usda          # 独立马桶（含 RevoluteJoint 盖板）
        └── Door_YS2HVXZVALGD2PTUJQ888888_7qcYe/
            └── *.usda          # 独立门扇（含 RevoluteJoint）
```

---

## 文件详解

### `__init__.py` — 统一出口

```python
from .toilet_scene_cfg import PalletInstantiableSceneCfg, PalletSceneCfg, ToiletSceneV0311Cfg
from .real_scene_cfg   import ShenxinfuRealSceneCfg, ShenxinfuRealInsideSceneCfg
```

上层代码通过 `from ...assets.scenes import <配置名>` 引用，无需关心内部文件拆分。

---

### `toilet_scene_cfg.py` — 仿真卫生间场景（3 个变体）

按迭代时间由旧到新排列。

#### 1. `ToiletSceneV0311Cfg` — 已弃用的完整卫生间

```python
room = AssetBaseCfg(usd_path=".../toilet_v0311/20250311B.usd",
                    pos=(0.5, -0.1, 0.0), rot=(0.707, 0, 0, 0.707))
light = AssetBaseCfg(DomeLightCfg(intensity=3000.0))
```

- **用途**：早期完整卫生间房间（2025-03-11 版本），包含四面墙、地板、马桶等全套装置。
- **状态**：已弃用（Deprecated）。房间坐标系与当前任务对齐方式不同，需旋转 90°。
- **特点**：带穹顶光源，其他场景配置均不含光源（默认使用 Isaac Sim 全局光）。

#### 2. `PalletSceneCfg` — 隔间场景（非实例化）

```python
room  = AssetBaseCfg(usd_path=".../pallet/20250411/sen0324.usd", pos=(0,0,0))
plane = AssetBaseCfg(GroundPlaneCfg(...), pos=(0,0,-0.20))
```

- **用途**：单个卫生间隔间场景，带地面平面（`GroundPlaneCfg`）。
- **限制**：非 PhysX 实例化 USD，**仅支持 `num_envs=1`**；多环境并行会导致 prim 路径冲突。
- **地面**：在 z=-0.20 处放置 Grid 环境地面，补偿场景 USD 内部坐标偏移。
- **`__post_init__`**：空实现，留给子类扩展（`AllstarToiletSceneCfg` 在此追加机器人）。

#### 3. `PalletInstantiableSceneCfg` — 隔间场景（支持多环境）

```python
room = AssetBaseCfg(usd_path=".../pallet/20250411/sen0324.usd", pos=(0,0,0))
# 无 plane
```

- **用途**：与 `PalletSceneCfg` 使用相同 USD，但不含地面，USD 本身已标记为
  PhysX-instanceable，支持 `num_envs > 1` 的并行仿真。
- **与 PalletSceneCfg 的唯一区别**：去掉了 `plane`。地面由各 env 的 USD 内部处理，
  或由子类按需添加。

---

### `real_scene_cfg.py` — 申鑫福真实场景（2 个变体）

基于对真实卫生间的 3D 扫描重建，用于 sim-to-real 迁移验证。

#### 1. `ShenxinfuRealSceneCfg` — 841837 静态全房间

```python
room = AssetBaseCfg(usd_path=".../841837/start.usda")
```

- **用途**：完整静态房间（841837 号隔间），包含所有固定装置，无交互式铰链体。
- **资产类型**：`AssetBaseCfg`（纯静态，不参与物理求解）。
- **适用场景**：机械臂在真实场景几何中运动、测试路径规划，无需开门/开盖操作。

#### 2. `ShenxinfuRealInsideSceneCfg` — 841838 隔间内部（含铰链体）

```python
room    = RigidObjectCfg(usd_path=".../841838/start.usda")     # 刚体
toilet  = ArticulationCfg(usd_path="...Toilet_*.usda")          # 含 RevoluteJoint_.*
door    = ArticulationCfg(usd_path="...Door_*.usda")            # 含 RevoluteJoint
```

- **用途**：841838 号隔间内视角，房间本身作为刚体，马桶和门作为独立关节体。
- **资产类型对比**：

  | 资产 | 类型 | 说明 |
  |------|------|------|
  | `room` | `RigidObjectCfg` | 刚体，有碰撞，contact_offset=0.005 |
  | `toilet` | `ArticulationCfg` | 含 `RevoluteJoint_.*`（盖板/座圈等多关节） |
  | `door` | `ArticulationCfg` | 含单个 `RevoluteJoint`（门扇转动） |

- **驱动参数**（马桶 & 门均相同）：
  - `stiffness=None`（无主动刚度，完全被动弹簧）
  - `damping=4.0`（阻尼保证关节不自由振荡）
  - `effort_limit_sim=300 N·m`，`velocity_limit_sim=30 rad/s`
  - 求解器：`position_iteration=3`（马桶）/ `position_iteration=2`（门）

- **初始位姿**（硬编码，与真实扫描坐标系对齐）：
  - 马桶：`pos=(1.103, -0.019, 0.468)`, `rot=(0.707, 0, 0, -0.707)`
  - 门：`pos=(-0.011, -0.025, 1.264)`, `rot=(0.707, 0, 0, -0.707)`

---

## 各变体横向对比

| 配置名 | 场景来源 | 多环境 | 交互铰链体 | 当前使用 |
|--------|----------|--------|------------|----------|
| `ToiletSceneV0311Cfg` | 手工建模全房间 | 否 | 无 | 已弃用 |
| `PalletSceneCfg` | 扫描/建模隔间 | 否（`num_envs=1`）| 无 | 主任务基类 |
| `PalletInstantiableSceneCfg` | 同上 | 是 | 无 | 多环境并行备用 |
| `ShenxinfuRealSceneCfg` | 真实扫描 841837 | 否 | 无 | 真实场景验证 |
| `ShenxinfuRealInsideSceneCfg` | 真实扫描 841838 | 否 | 马桶 + 门 | 开盖/开门任务 |

---

## 调用关系

```
allstar_toilet_env_cfg.py
    └── from .assets.scenes import PalletSceneCfg
            └── class AllstarToiletSceneCfg(PalletSceneCfg):
                    # 追加: robot(MISSING), cam_wrist, cam_table, contact_forces, cam_left, cam_right
                    └── scene: AllstarToiletSceneCfg = AllstarToiletSceneCfg(num_envs=1, env_spacing=2.5)

config/chassis_arm/joint_pos_env_cfg.py
    └── 继承 allstar_toilet_env_cfg.py 的 AllstarToiletEnvCfg
            └── robot = ARM_RML63_6F_BRUSH_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/arm")

config/chassis_arm/rml63_real_shenxinfu_joint_pos_env_cfg.py
    └── from ...assets.scenes import ShenxinfuRealSceneCfg
            └── class RML63RealShenxinfuSceneCfg(ShenxinfuRealSceneCfg):
                    # 追加: robot, contact_forces=None（真实场景无接触传感器）
                    └── scene: RML63RealShenxinfuSceneCfg = RML63RealShenxinfuSceneCfg(num_envs=1)

config/chassis_arm/mecanum_real_shenxinfu_velocity_env_cfg.py
    └── from ...assets.scenes import ShenxinfuRealSceneCfg
            └── class MecanumRealShenxinfuSceneCfg(ShenxinfuRealSceneCfg):
                    # 追加: robot=MECANUM_CHASSIS_CFG
```

**继承规律**：场景基类只定义**无机器人**的环境几何；机器人、摄像头、传感器在子类中按任务需求注入。
这使得同一场景可与不同机器人型号组合，无需重复定义场景资产。

---

## 扩展新场景的步骤

1. 将 USD/USDA 文件放入 `data/scenes/<场景名>/`
2. 在 `toilet_scene_cfg.py` 或 `real_scene_cfg.py` 中新增 `@configclass` 继承
   `InteractiveSceneCfg`，声明 `room`（以及需要独立物理的铰链体）
3. 在 `__init__.py` 中导出新配置名
4. 在 env_cfg 中继承该场景基类并注入机器人资产
