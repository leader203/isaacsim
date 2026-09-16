# 玻璃门与墙壁碰撞诊断报告

> 诊断日期：2026-07-13
> 执行环境：Native Linux (no Docker), Isaac Lab 0.54.3, Isaac Sim 4.5.0.0
> 场景路径：/home/stardust/czs/roboos/source/.../data/scenes/real/shenxinfu/

## 1. 环境信息

| 项目 | 值 |
|------|-----|
| Isaac Sim | 4.5.0.0 (pip install) |
| Isaac Lab | 0.54.3 |
| Python | 3.10.20 |
| OpenUSD | 不可在外部导入 (需 Isaac Sim runtime) |
| OS | Ubuntu 22.04, native (no Docker) |
| 场景绝对路径 | .../data/scenes/real/shenxinfu/841837/start_open_seat.usda |

## 2. 玻璃门检查结果

### 门 #1: Door_YS2HVXZVALGD2PTUJQ888888_7qcYe

```
引用路径:       ../Door_YS2HVXZVALGD2PTUJQ888888_7qcYe/Door_YS2HVXZVALGD2PTUJQ888888_7qcYe/Door_YS2HVXZVALGD2PTUJQ888888_7qcYe.usda
解析结果:       文件存在 ✅ (shenxinfu/ 外部, 1608 bytes)
入口层:         打开正常 ✅
defaultPrim:    "Door_YS2HVXZVALGD2PTUJQ888888_7qcYe" ✅
payload:        Door_YS2HVXZVALGD2PTUJQ888888_7qcYe_payload.usda ✅
geo:            Door_YS2HVXZVALGD2PTUJQ888888_7qcYe_geo.usd (USDC crate 0.8.0, 30KB) ✅
look:           Door_YS2HVXZVALGD2PTUJQ888888_7qcYe_look.usda ✅
材质:           2 MDL (glass + frame) ✅
纹理:           3 JPG (color, specular, normal) ✅
CollisionAPI:   载荷层/外观层均无 ❌
extentsHint:    [(-0.446, -0.0355, -1.251), (0.446, 0.0355, 1.251)]
                宽89cm × 厚7cm × 高2.5m
scale:          (-1, -1, 1) — 镜像反射
841837内部副本: 无 ❌ (仅存在于shenxinfu/外部)
```

### 门 #2: Door_YS3UEUZVAJ2V4PTUKU888888_mdynP

```
引用路径:       ../Door_YS3UEUZVAJ2V4PTUKU888888_mdynP/Door_YS3UEUZVAJ2V4PTUKU888888_mdynP/Door_YS3UEUZVAJ2V4PTUKU888888_mdynP.usda
解析结果:       文件存在 ✅ (shenxinfu/ 外部, 344 bytes)
入口层:         打开正常 ✅
defaultPrim:    "Door_YS3UEUZVAJ2V4PTUKU888888_mdynP" ✅
payload:        Door_YS3UEUZVAJ2V4PTUKU888888_mdynP_payload.usda ✅
geo:            Door_YS3UEUZVAJ2V4PTUKU888888_mdynP_geo.usd (USDC crate 0.8.0, 13KB) ✅
look:           Door_YS3UEUZVAJ2V4PTUKU888888_mdynP_look.usda ✅
材质:           2 MDL (glass + frame) ✅
纹理:           1 JPG (normal) ✅
CollisionAPI:   载荷层/外观层均无 ❌
extentsHint:    [(-0.5905, -0.0115, -1.251), (0.5905, 0.0115, 1.251)]
                宽1.18m × 厚2.3cm × 高2.5m
841837内部副本: 有 ✅ (841837/meshes/Assets/ 内副本与外部完全一致)
```

**玻璃门初步结论**：
- 两个门的资产文件完整无缺失 ✅
- 引用链正确（entry → payload → subLayers(geo+look)） ✅
- 外观层中无 CollisionAPI（载荷层也是空的Xform） ❌
- CollisionAPI 可能存在于 USDC 二进制 geo 文件中，但当前无法验证（需 Isaac Sim runtime 中的 pxr 库）
- **Door_YS2H 在 841837 内部无副本**，仅存在于 shenxinfu/ 外部 ⚠️

## 3. 墙体碰撞检查结果

```
Wall_Wall 引用链:    Wall_Wall.usda → payload → subLayers(geo.usd + look.usda) ✅
载荷层:             空 Xform（几何体和碰撞在 USDC 二进制中）
外观层:             多个 Geom (A63-A75等) 下的 mesh_0000，仅有 MaterialBindingAPI，无 CollisionAPI ❌
extentsHint:        [(-3.49, -1.75, -1.4), (3.49, 1.75, 1.4)] — 7m×3.5m×2.8m
geo格式:            USDC crate 0.9.0 (16KB, 二进制)

场景中Wall_0000:    在 Meshes/Base 下，引用 @./meshes/Assets/Wall_Wall/Wall_Wall.usda@
场景中碰撞可见:     仅在 Floor_0000 子prim上有 physics:collisionEnabled=1 (115处)
                  Wall_0000 场景层无 collisionEnabled
                  CollisionAPI 在 USDC 二进制层中
```

## 4. 房间加载配置

```python
# 当前配置 — 已使用 AssetBaseCfg ✅ (不是 RigidObjectCfg)
room = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/room",
    spawn=sim_utils.UsdFileCfg(usd_path=ROOM_USD),
)
```

房间已正确配置为静态资产，不是刚体。

## 5. cuRobo 障碍物扫描

```
扫描路径:
  /World/envs/env_0/room/Meshes/Base/          ← Wall_0000 + Partition_*
  /World/envs/env_0/room/CollisionFixes/CuroboOnlyWalls/ ← 手动Raycast代理墙
  /World/envs/env_0/paper_waste/               ← 纸废弃代理

实际墙体Mesh位置:
  /World/envs/env_0/room/Meshes/Assets/Wall_Wall/Geom/... ← cuRobo 未扫描

问题: cuRobo 扫描 Meshes/Base/Wall_0000，但 Wall_0000 只是 reference 包装 Xform。
      实际几何在 Meshes/Assets/Wall_Wall/Geom/A63/mesh_0000 等路径，
      且 geo 文件为 USDC 二进制 → cuRobo mesh exporter 返回 0 顶点。
```

## 6. 关键限制说明

由于 pxr 模块仅在 Isaac Sim 完整 runtime 中可用，当前无法在外部执行以下操作：
- 打开组合后的 Stage 并遍历叶子 Mesh
- 检查 USDC 二进制 geo 文件中的 CollisionAPI
- 检查 PhysX cooking 日志
- 运行碰撞可视化
- 计算世界包围盒
- 执行物理接触测试

**需在 Isaac Sim 运行时中继续执行的操作**: 第三阶段(Stage检查)、第五阶段(CollisionAPI统计)、第六阶段(PhysX碰撞可视化)、第八阶段(cuRobo障碍物采集验证)

## 7. 已确认的发现

| 发现 | 状态 |
|------|------|
| 门资产文件完整 | ✅ 确认 |
| 门引用链可解析 | ✅ 确认 (entry→payload→subLayers) |
| 门文本层无 CollisionAPI | ❌ 确认 |
| 墙文本层无 CollisionAPI | ❌ 确认 |
| 房间配置正确 (AssetBaseCfg) | ✅ 确认 |
| cuRobo扫描路径与实际Mesh路径不匹配 | ⚠️ 确认 (Meshes/Base vs Assets) |
| USDC二进制geo文件无法外部检查 | ⚠️ 需Isaac Sim runtime |

## 8. 备份状态

```
start_open_seat.original_backup.usda ✅ (72KB)
start.original_backup.usda ✅ (72KB)
```
