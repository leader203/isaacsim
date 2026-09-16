# 场景资产说明 — allstar_toilet

## 目录结构

```
data/scenes/
├── pallet/20250411/
│   ├── sen0324.usd              # 合成单隔间（训练用）
│   ├── sen0401.usd              # 合成单隔间·可实例化（多环境并行）
│   ├── instances/               # 马桶/小便器/卫生纸/垃圾桶子场景 USD
│   └── SubUSDs/                 # 同上（用于 USD 引用层级）
├── real/shenxinfu/
│   ├── 841837/start.usda        # 真实公共卫生间（整体场景）
│   ├── 841838/start.usda        # 真实单隔间（含可交互马桶 + 门）
│   ├── 841843/start.usda        # 真实单隔间备用版本
│   └── Toilet_YREQ.../          # 申新福马桶单品资产
├── toilet_v0311/20250311B.usd  # 废弃整体场景（旧版）
└── rx/                          # 脏污 / 垃圾资产库
```

---

## 场景详情

### pallet/20250411/sen0324.usd — 合成单隔间（标准训练场景）

| 属性 | 说明 |
|------|------|
| 范围 | 单个马桶隔间 |
| 关节 | 马桶盖/圈：RevoluteJoint（可开合） |
| IsaacLab 配置 | `PalletSceneCfg`（非实例化，单环境） |
| 脏污支持 | 无内置脏污；需通过 `rx/` 资产手动添加 |
| 用途 | 键盘遥操作录制、单环境调试 |

---

### pallet/20250411/sen0401.usd — 合成单隔间·可实例化

| 属性 | 说明 |
|------|------|
| 范围 | 单个马桶隔间 |
| 关节 | 马桶盖/圈：RevoluteJoint（同 sen0324） |
| IsaacLab 配置 | `PalletInstantiableSceneCfg`（PhysX Instantiable，支持多环境并行） |
| 脏污支持 | 无内置脏污；需通过 `rx/` 资产手动添加 |
| 用途 | 多环境并行训练（推荐） |

---

### real/shenxinfu/841837/start.usda — 真实公共卫生间（整体场景）

| 属性 | 说明 |
|------|------|
| 范围 | 完整公共卫生间（多个隔间 + 洗手台区域） |
| 关节 | 全部静态，无可交互关节 |
| IsaacLab 配置 | `ShenxinfuRealSceneCfg` |
| 脏污支持 | 无内置脏污 |
| 用途 | 真实场景外观参考、麦轮底盘导航（Mecanum） |

---

### real/shenxinfu/841838/start.usda — 真实单隔间（含可交互马桶 + 门）

| 属性 | 说明 |
|------|------|
| 范围 | 单个隔间（含隔墙、马桶、隔板门） |
| 关节 | 马桶座圈：RevoluteJoint（0°–95°）<br>马桶盖：RevoluteJoint（0°–95°）<br>冲水旋钮：PrismaticJoint<br>隔间门：RevoluteJoint |
| IsaacLab 配置 | `ShenxinfuRealInsideSceneCfg` |
| 脏污支持 | 无内置脏污；可叠加 `rx/` 资产 |
| 用途 | 真实场景单隔间操作任务（盖子开合、冲水等） |

---

### real/shenxinfu/841843/start.usda — 真实单隔间（备用版本）

| 属性 | 说明 |
|------|------|
| 范围 | 单个隔间 |
| 关节 | 待确认（与 841838 结构类似） |
| IsaacLab 配置 | 暂无对应 CFG |
| 脏污支持 | 无内置脏污 |
| 用途 | 备用场景，尚未接入训练流程 |

---

### toilet_v0311/20250311B.usd — 废弃整体场景

| 属性 | 说明 |
|------|------|
| 范围 | 完整卫生间 |
| 关节 | 不详 |
| IsaacLab 配置 | 已废弃 |
| 用途 | **请勿使用**，保留仅作历史参考 |

---

## 脏污 / 垃圾资产库 (`rx/`)

所有资产均为独立 USD 文件，可按需叠加到上方任意场景中。每类资产存放在同名子目录下。

| 类别 | 路径 | 描述 |
|------|------|------|
| 烟蒂 | `rx/cigarette/cigarette.usd` | 卷烟残留物 |
| 烟灰 | `rx/Cigarette ash/Cigaretteash.usd` | 烟灰散落 |
| 脚印 | `rx/footprint/footprint.usd` | 地面脚印污迹 |
| 塑料袋 | `rx/plastic bag/plastic bag.usd` | 废弃塑料袋 |
| 卷筒卫生纸 | `rx/roll paper/rollpaper .usd` | 散落卷筒纸 |
| 座便纸 | `rx/seat paper/seatpaper.usd` | 马桶座便纸残留 |
| 卫生巾 | `rx/Aunt scarf/Auntscarf.usd` | 废弃卫生巾 |
| 粪便污迹 | `rx/Shitstains/Shitstains.usd` | 马桶/地面粪便污迹 |
| 团状纸巾 | `rx/Tuanshaped/Tuanshaped01~06.usd` | 6 种揉团卫生纸变体 |
| 尿液污迹 | `rx/urinarystains/urinarystains.usd` | 尿液痕迹 |
| 水渍 | `rx/Waterstain/Waterstain.usd` | 水迹/水渍 |

> 使用方式：在场景 CFG 的 `extras` 或 `rigid_objects` 中以 `RigidObjectCfg` / `AssetBaseCfg` 引入对应 USD 文件，并指定初始位姿。

---

## 快速选型指南

| 需求 | 推荐场景 |
|------|----------|
| 单环境调试 / 遥操作录制 | `pallet/20250411/sen0324.usd` |
| 多环境并行训练 | `pallet/20250411/sen0401.usd` |
| 麦轮底盘导航 | `real/shenxinfu/841837/start.usda` |
| 真实单隔间操作（含门/盖）| `real/shenxinfu/841838/start.usda` |
| 添加脏污任务 | 上述任意场景 + `rx/` 资产 |
