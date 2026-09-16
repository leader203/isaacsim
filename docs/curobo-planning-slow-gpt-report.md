# cuRobo 机械臂规划慢 — GPT 诊断报告

> 项目：Isaac Lab 4.5 + cuRobo GPU 运动规划，清洁机器人仿真
> 场景：狭窄卫生间 (841837)，7 个纸废弃物 + 墙壁 + 马桶
> 机器人：6-DOF 机械臂 + prismatic gripper

---

## 一、核心问题

机械臂一次纸团抓取全流程 ~176 秒（仅 cuRobo 规划等待，不含执行），导师反馈"速度实在太慢"。

## 二、规划耗时分解

```
阶段                   规划耗时    类型
pre_target_open         0.5s      cspace (关节空间)
target (trajopt)       15.3s      trajopt (笛卡尔优化)
post_target_descend    15.3s      trajopt (下降重规划)
gripper_lift           14.7s      trajopt (抬升重规划)
worker 等待总计        176.2s     (含多次内部重试)
```

**问题**：Descend 和 Lift 各自走了 trajopt 全局重规划（各 15s），而非简单直线运动。

## 三、关节扭矩严重饱和

```
computed/applied ratio: max 3.6x, mean 2.3x
29 个采样点中 23 个超过 2x (79%)
```

关节 effort_limit 配置：
```
joint1: 400, joint2: 800, joint3: 400
joint4: 400, joint5: 400, joint6: 400
```

PD 控制器请求的力矩被频繁截断 → 轨迹跟踪偏差 → 永远无法收敛。

## 四、PD 收敛统计

```
hold: 250/250 steps (100%) err_max=0.040 rad ⚠️ 耗尽
hold: 250/250 steps (100%) err_max=0.034 rad ⚠️ 耗尽
```

250 步全部用完后关节仍在漂移，从未真正收敛到目标位姿。

## 五、当前配置

```
IK seeds:        64
trajopt seeds:   32
collision_activation_distance: 0.02m
steps_per_waypoint: 10
post_trajectory_settle_steps: 0

关节 stiffness: [2400, 16000, 8000, 2400, 2400, 2400]
关节 damping:    [960,  3840,  1920,  960,  960,  960]
障碍物: 13 cuboids + 3 mesh = 16 个
```

> ⚠️ 所有 cuRobo 核心文件 (curobo_arm_joints.py, curobo_planner_client.py, curobo_trajectory_execution.py, curobo_arm_drive.py, curobo_profiles.py, curobo_runtime_setup.py) 和上述参数均与原始仓库一致，未被我们修改。

## 六、墙壁碰撞分析

- **PhysX 碰撞**：墙壁有完整的 collision mesh（physics:collisionEnabled=1，115 prims），机器人碰墙会被挡住
- **cuRobo 无法读取**：墙壁几何体是 USDC 二进制格式 (USD crate 0.9.0)，cuRobo mesh exporter 返回 0 顶点；且路径不匹配（场景是 `meshes/Assets/`，代码查找 `Meshes/Base/`）
- **当前方案**：手动添加 8 个薄长方体代理墙壁 (0.06×2m)，O(1) 碰撞检测

## 七、已知的 Isaac Lab 关节控制 bug

- **reset 必须同时设 target + state**：只设 `write_joint_state_to_sim` 不设 `set_joint_position_target` → PD 拽回原点 → 机械臂抽搐（GitHub issue #2663）
- **GPU PhysX 限制**：`setGlobalPose` / `updateKinematic` / `addForce` 在 eENABLE_DIRECT_GPU_API 下被禁用
- **`steps_per_waypoint=10`** 是轨迹执行的唯一速度参数

## 八、核心问题（请 GPT 分析）

1. **关节扭矩持续 2-3x 饱和的根本原因**：是 stiffness 太高？effort_limit 太低？还是机械臂自重 + 末端负载（夹爪）导致关节 2/3 的静载远超电机能力？

2. **Descend/Lift 走 trajopt 全局重规划是否合理**：这 2 段轨迹本质上只是 Z 方向直线运动，能否改成笛卡尔直线 servo（warp 或简单 joint interpolation），从而砍掉各 15s 的规划耗时？

3. **PD 收敛失败**：`steps_per_waypoint=10` + `post_trajectory_settle_steps=0` + 扭矩饱和 → 轨迹跟踪偏差放大。在不改电机参数的前提下，哪些参数可以调？

4. **种子数优化**：当前 64/32 seeds。在狭窄空间降低 seeds 会影响成功率，但实际观测 planner 精度极高（0.0002mm），execution 精度才是瓶颈（2-4cm）。降低 seeds 对实际抓取成功率的影响有多大？

5. **CuroboOnlyWalls 障碍物方案**：8 个简单长方体对 cuRobo 碰撞检测影响极小（O(1)），是否建议保留？还是应该尝试用 USD API 读取原始墙壁 USDC 格式 mesh？

6. **整体优化路线建议**：砍 descned/lift trajopt → 降低 seeds → 调 PD 参数 → 增加 settle steps 的顺序是否合理？
