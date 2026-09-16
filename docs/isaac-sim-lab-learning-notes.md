# Isaac Sim + Isaac Lab 官方文档学习笔记

> 学习日期：2026-07-11
> 涵盖：Isaac Sim 6.0.1, Isaac Lab (latest), cuRobo, PhysX GPU Pipeline, USD API

---

## 1. 仿真架构与数据流

### 1.1 三层数据通路

Isaac Sim 使用三条数据路径：

| 通路 | 用途 | 时机 |
|------|------|------|
| **USD** | 场景持久化/创建 | 仿真前后 |
| **Fabric** | GPU 加速运行时数据 | 仿真中（渲染同步） |
| **Physics Tensors** | 批量 GPU 读写 | RL/控制循环 |

### 1.2 仿真步骤流程（核心）

```python
# 正确顺序（Isaac Lab 官方）
# ① 设置目标
robot.set_joint_position_target(target_pos)   # PD 控制器目标
robot.set_joint_effort_target(target_effort)  # 力矩控制

# ② 写入物理引擎
scene.write_data_to_sim()

# ③ 步进物理仿真
sim.step(render=False)    # PhysX 计算动力学

# ④ 读取更新状态
scene.update(dt)          # robot.data.joint_pos/vel 更新
```

### 1.3 ManagerBasedRLEnv 的 env.step() 内部流程

```
env.step(action)
  → action manager 处理 → 设为关节目标
  → scene.write_data_to_sim()
  → sim.step() × decimation 次 (如 decimation=4 → 50Hz策略/200Hz物理)
  → scene.update(dt) 每子步
  → 计算观测/奖励/终止
```

---

## 2. 关节控制（Actuators）

### 2.1 两类执行器

| 类型 | 行为 | PD 控制器位置 |
|------|------|---------------|
| **Implicit** (`ImplicitActuatorCfg`) | PhysX 内部 PD | 物理引擎内 |
| **Explicit** (`IdealPDActuatorCfg` / `DCMotorCfg`) | 用户计算力矩 | Python/torch |

### 2.2 关键参数

| 参数 | 含义 | 效果 |
|------|------|------|
| `stiffness` | P-gain (Nm/rad) | 越大→更快逼近目标，过高→振荡 |
| `damping` | D-gain (Nm/(rad/s)) | 越大→越平滑，过高→响应慢 |
| `effort_limit` | 最大力矩/力 | 限幅保护 |
| `velocity_limit` | 最大速度 | 限制关节速度 |
| `armature` | 数值阻尼 | 提高稳定性 |

### 2.3 控制模式切换

```python
# 位置控制：高 stiffness + 适中 damping
stiffness=40.0, damping=10.0

# 速度控制：stiffness=0
stiffness=0.0, damping=10.0

# 力矩控制：两者均为0
stiffness=0.0, damping=0.0
```

### 2.4 关键陷阱：reset 时必须同时设 target 和 current state

```python
# ❌ 错误：只写 current state
robot.write_joint_state_to_sim(joint_pos, joint_vel)

# ✅ 正确：先设 target，再写 state
robot.set_joint_position_target(joint_pos)
robot.set_joint_velocity_target(joint_vel)
robot.write_joint_state_to_sim(joint_pos, joint_vel)
```

**原理**：如果只设 current state 不设 target，PD 控制器看到 target 为 0，会在下一帧把机器人拽回原点。

---

## 3. GPU PhysX 限制（Direct GPU API）

### 3.1 不可用的 API

当 `PxSceneFlag::eENABLE_DIRECT_GPU_API` 开启时：

| 禁止操作 | 错误信息 |
|----------|---------|
| `setGlobalPose()` | illegal if eENABLE_DIRECT_GPU_API is enabled |
| `setLinearVelocity()` / `setAngularVelocity()` | 同上 |
| `updateKinematic()` | 同上（无法切换 kinematic↔dynamic） |
| `addForce()` / `addTorque()` | 同上 |
| 动态生成 articulation | 不支持 |

### 3.2 根因

- GPU pipeline 将物理状态锁定在 GPU 显存中
- CPU 侧直接修改会导致 GPU-CPU 内存不一致
- 这是不可变的场景设置

### 3.3 解决途径

1. **场景搭建在仿真前完成** — 不在运行时 spawn/delete 物理体
2. **使用 CPU physics pipeline** — `use_gpu_pipeline=False`（牺牲性能）
3. **使用 PxDirectGPUAPI** — GPU kernel 操作（需要 CUDA 开发）
4. **只改 visual prim** — 修改视觉层不影响物理层

---

## 4. USD API 运行时 Prim 操作

### 4.1 创建物理体

```python
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf

# 创建 Xform + 几何体
xform = UsdGeom.Xform.Define(stage, prim_path)
xf = UsdGeom.Xformable(xform.GetPrim())
xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))

# 碰撞几何
coll = UsdGeom.Cube.Define(stage, coll_path)
coll.CreateSizeAttr(1.0)
UsdPhysics.CollisionAPI.Apply(coll.GetPrim())

# 刚体
UsdPhysics.RigidBodyAPI.Apply(root_prim)
mass_api = UsdPhysics.MassAPI.Apply(root_prim)
mass_api.CreateMassAttr().Set(mass)

# PhysX 特定属性
physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
physx_api.CreateLinearDampingAttr().Set(3.0)
physx_api.CreateAngularDampingAttr().Set(3.0)
```

### 4.2 材质绑定

```python
mat = UsdShade.Material.Define(stage, mat_path)
shader = UsdShade.Shader.Define(stage, mat_path + "/preview_surface")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.88)
mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
```

### 4.3 碰撞几何体可见性

```python
# 隐藏碰撞体（仅物理，不渲染）
img = UsdGeom.Imageable(collision_prim)
img.CreatePurposeAttr().Set("guide")  # guide = 不渲染

# 完全隐藏 prim
img.MakeInvisible()
```

### 4.4 关键陷阱

1. **`CreateSizeAttr(1.0)` 必调** — 否则 scale 翻倍（cube 默认 size=2）
2. **碰撞体用简单几何** — sphere/cube/convex，不用复杂 mesh
3. **材质绑定到 visual mesh**，不要绑到 root Xform 上子 prim 不可靠
4. **纹理路径** — `UsdUVTexture` 用绝对路径时，部分渲染器可能解析失败

---

## 5. cuRobo 运动规划

### 5.1 规划管线

```
IK (并行) → Trajectory Seeding → TrajOpt (并行优化) → Graph Planner (回退)
```

### 5.2 核心配置参数

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `num_ik_seeds` | 32 | 64 | IK 并行种子数 |
| `num_trajopt_seeds` | 4 | 32 | TrajOpt 并行种子数 |
| `collision_activation_distance` | 0.01 | **0.02** | 碰撞膨胀距离（m） |
| `position_tolerance` | 0.005 | 0.005 | 位置容差（m） |
| `orientation_tolerance` | 0.262 | 0.262 | 姿态容差（rad, 15°） |

### 5.3 `collision_activation_distance` 详解

- **含义**：障碍物膨胀半径。碰撞成本在此距离外为 0，距离内逐渐增加
- **关键影响**：设太大（如 0.10m）会在狭窄空间封死路径，trajopt 返回 success=False
- **推荐值**：狭窄环境 0.02m，开放环境 0.05-0.10m

### 5.4 规划速度

- 中位规划时间：30-100ms（桌面 GPU）
- 实际观测：cspace 0.5s，trajopt 5-15s（含重试）
- 慢的原因：seed 数多 + 障碍物多 + 狭窄空间

### 5.5 Planner Error vs Execution Error

```
planner_final_pos_error_m  → cuRobo 规划结果离目标多远
actual_final_pos_error_m   → Isaac 执行轨迹后实际末端离目标多远
```

两个误差可能差几个数量级。必须先区分再定位问题。

---

## 6. 仿真性能优化

### 6.1 关键调优参数

| 参数 | 影响 |
|------|------|
| `solver_position_iteration_count` | 关节约束精度（越大越准但越慢） |
| `solver_velocity_iteration_count` | 速度约束精度 |
| `gpu_max_rigid_contact_count` | GPU 接触对上限（超出回退 CPU） |
| `decimation` | 策略频率 = 物理频率 / decimation |
| `steps_per_waypoint` | 轨迹每点停留步数（越大越慢） |

### 6.2 提速检查清单

1. 头显模式（headless）用于训练
2. 简化碰撞几何（convex hull > mesh）
3. 降低 `solver iteration count`
4. 增大 `steps_per_waypoint` 但降低 waypoint 数量
5. CPU governor 设为 `performance`

---

## 7. 与项目相关的关键教训

### 7.1 为什么不能运行时修改动态体
- GPU PhysX 模式下 `setGlobalPose()` 被禁用
- 不能切换 kinematic ↔ dynamic
- 不能动态 deactive rigid body（`SetActive(False)` 崩溃）

### 7.2 为什么视觉跟随要放在 runtime_visuals/
- 在 robot articulation 下挂 prim → PhysX 崩溃
- `/World/envs/env_0/runtime_visuals/` 是安全的非物理子树

### 7.3 为什么抓取偏差要区分 planner vs execution
- cuRobo planner 精度通常 < 1mm（planner_final_pos_error_m）
- 执行偏差来自关节扭矩饱和、PD 收敛不足、重力漂移
- 5.7cm 偏差几乎肯定是执行问题，不是规划问题

### 7.4 为什么机械臂慢
- cuRobo 核心代码未修改 → 慢是原始管线特性
- 影响因素：seed 数、障碍物数量、狭窄空间、关节扭矩限制
- 关节扭矩饱和 → PD 无法收敛 → hold 步数耗尽 → 位置不准

### 7.5 为什么 `write_joint_state_to_sim` 必须配合 target
- 只设 current state 不设 target → PD 拽回原点 → 机械臂抽搐
- 这是 Isaac Lab GitHub issue #2663 的已知 bug

---

## 8. 参考资源

- Isaac Sim 官方文档：https://docs.isaacsim.omniverse.nvidia.com/
- Isaac Lab 官方文档：https://isaac-sim.github.io/IsaacLab/
- cuRobo 官方文档：https://curobo.org/
- PhysX API 参考：https://nvidia-omniverse.github.io/PhysX/
- Isaac Lab GitHub：https://github.com/isaac-sim/IsaacLab
