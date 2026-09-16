# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 知识库

**`docs/isaac-sim-lab-learning-notes.md`** — Isaac Sim + Isaac Lab 官方文档学习笔记，涵盖：仿真架构与数据流、关节控制（actuators/stiffness/damping）、GPU PhysX 限制（Direct GPU API 禁止的操作）、USD API 运行时 prim 操作、cuRobo 运动规划配置、仿真性能优化。**修改物理参数、USD API 调用、cuRobo 配置前必须查阅此文件。**

## 备份

**`/home/stardust/czs/roboos-backup-20260711_142901/`** — 9 个核心文件的完整备份（2026-07-11）。修改效果不理想时从此恢复。

## 铁律

1. **遇到困难立即求助 ChatGPT**：疑惑、反复失败、涉及 PhysX/USD/cuRobo 底层、方案不完善时，提供完整上下文（代码、日志、已尝试方案）问 GPT。
2. **严格按 SOP 顺序调用工具**：每次规划前后必须回到规划起始位姿。不遵守必失败。
3. **每个大改动前经 GPT 审查**：新增功能、修改物理参数、USD API 调用、cuRobo 配置等大改动，先整理方案+代码发给 GPT 审核通过后再执行。
4. **pycache 必须清除**：修改 Python 代码后运行 `find /home/stardust/czs/roboos/source -name "__pycache__" -type d -exec rm -rf {} +`，否则可能执行旧代码。
5. **P1 visual carry 禁用手动测试**：任何试图在 articulation 下挂 prim 或 teleport dynamic body 的操作都会导致 PhysX 崩溃。

## 环境

- conda: `isaaclab` (Python 3.10, Isaac Sim 4.5.0.0, torch 2.5.1, CUDA 12.4)
- GPU: RTX 4070 SUPER 12GB, i7-14700K, Ubuntu 22.04
- 必须 `conda activate isaaclab` + `OMNI_KIT_ACCEPT_EULA=YES`
- 所有脚本通过 `./isaaclab.sh -p <script>` 运行，不能直接 `python`
- 首次启动编译 shader 3-5 分钟，GUI 显示"无响应"正常
- 训练环境: `RoboTwin`, `SimpleVLA` conda 已装
- 内网镜像 `mirrors.sangfor.org` (IP `200.200.1.241`)，开 WiFi 后需 `sudo ip route add 200.200.0.0/16 via 10.49.17.254 dev enp4s0`

## 仓库定位

清洁机器人仿真仓库，基于 NVIDIA Isaac Lab (Isaac Sim 4.5.0.0) + cuRobo GPU 运动规划 + FastAPI HTTP 接口。
核心代码：`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/`

```text
上游感知模块 → HTTP POST /tools/invoke → FastAPI 服务 → cuRobo 规划器 → 关节轨迹 → PhysX 仿真
```

## 代码架构

| 目录/文件 | 职责 |
|---|---|
| `scripts/run_curobo_pose_plan_test.py` | 主入口，启动仿真+FastAPI，约 5200 行 |
| `config/chassis_arm/` | 场景布局、机器人初始位姿、物体定义 |
| `configs/curobo/*.yml` | 每种机器人的碰撞球体（cuRobo 使用） |
| `curobo_pose_plan_utils/` | cuRobo 规划工具层（profiles, tool_specs, scene_obstacles 等） |
| `curobo_pose_plan/worker_planner.py` | 独立进程规划器 |
| `data/robots/` | 机器人 USD 模型和 URDF |
| `data/scenes/real/shenxinfu/` | 卫生间场景 USD (Git LFS) |
| `src/nvidia-curobo/` | cuRobo 源码（editable 安装） |
| `mdp/` | Isaac Lab 观测量和事件定义 |

### 任务配置继承
```
gripper_joint_prismatic_env_cfg (基础场景：cube、垃圾桶、马桶)
  └── common_curobo_pose_plan_test_env_cfg (可切换 5 种机器人 → gym task)
```

### 5 种机器人

| robot_id | robot_type | 专属工具 |
|---|---|---|
| `brush_rolling_0512` | brush | point_clean, spot_clean, move_arm_to_joints |
| `brush_rolling_sanheyi_long/short` | brush | 同上 |
| `gripper_prismatic` | gripper | execute_grab, take_out, move_arm_to_joints |
| `gripper_prismatic_sanheyi_full` | gripper | 同上 |

common: `list_robot_profiles`, `get_active_robot`, `switch_robot`, `stretch_end_tool`, `execute_spot_capture`, `base_keyboard_teleop`

## 启动服务

```bash
cd <roboos_root> && conda activate isaaclab
OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./isaaclab.sh -p source/.../scripts/run_curobo_pose_plan_test.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Common-CuroboPosePlanTest-Dirty-OpenSeat-v0 \
  --initial_robot_id <gripper_prismatic|brush_rolling_0512> \
  --curobo_collision_activation_distance 0.02 \
  --curobo_box_collision_front_min_x_m -0.5 \
  --tool_api_server --tool_api_host 127.0.0.1 --tool_api_port 8000
```

> **⚠️ `collision_activation_distance` 不要设 0.10！** 在狭窄卫生间里 0.10m 会把低位目标路径封死（trajopt 返回 success=False）。0.02 是安全值。详见下方"已知问题"。

GPU 显存：Isaac Sim 占用 ~7.5GB，剩余 ~4GB 给 cuRobo。32/16 seeds 可跑，8/4 seeds 保险。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少碎片。

## FastAPI SOP（已验证，来源 scripts_runtime.md）

```bash
wait_tool_job() {
  local req="$1" resp state
  while true; do
    resp=$(curl -sS "http://127.0.0.1:8000/tools/jobs/$req")
    state=$(printf '%s' "$resp" | python -c 'import sys,json; print(json.load(sys.stdin)["state"])')
    if [ "$state" != "queued" ] && [ "$state" != "running" ]; then
      printf '%s' "$resp" | python -m json.tool; break; fi; sleep 1
  done
}
```

### Gripper 抓取 SOP

**铁律：每次规划前 → 回规划起始位姿，每次规划完成后 → 也回规划起始位姿。**
`gripper_prismatic_execute_grab` 的 `move_to_pre_target` 默认 `true`，工具自动回起始位姿。但完成后必须手动调 `gripper_prismatic_move_arm_to_joints`。

流程：`[起始] → grab → [结束] → move_arm_to_joints → [起始] → take_out → [结束] → move_arm_to_joints → [起始]`

### Paper Cleanup SOP（湿纸团抓取→投放→收集）✅ 已验证通过

**基线配置（2026-07-10 确认稳定）：**

```bash
cd <roboos_root> && conda activate isaaclab
OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./isaaclab.sh -p source/.../scripts/run_curobo_pose_plan_test.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Common-CuroboPosePlanTest-Dirty-OpenSeat-v0 \
  --initial_robot_id gripper_prismatic \
  --curobo_collision_activation_distance 0.02 \
  --curobo_num_ik_seeds 64 \
  --curobo_num_trajopt_seeds 32 \
  --curobo_box_collision_front_min_x_m -0.5 \
  --tool_api_server --tool_api_host 127.0.0.1 --tool_api_port 8000
```

**场景物体**：6 个纸废弃对象
- 3 个湿纸团（clump）：`(1.75,-0.70,0.035)`, `(1.409,-1.157,0.035)`, `(1.355,-1.218,0.035)`
- 3 个纸巾片（tissue）：`(1.663,-0.980,0.408)`, `(1.656,-1.152,0.408)`, `(1.730,-1.211,0.408)` — stuck/static 状态

```bash
wait_tool_job() {
  local req="$1" resp state
  while true; do
    resp=$(curl -sS "http://127.0.0.1:8000/tools/jobs/$req")
    state=$(printf '%s' "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["state"])')
    if [ "$state" != "queued" ] && [ "$state" != "running" ]; then
      printf '%s' "$resp" | python3 -m json.tool; break; fi; sleep 2
  done
}

# ① 纸团抓取（paper_cleanup_mode 自动检测纸团位置 + Z 方向 3 档 retry）
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"gripper_prismatic_execute_grab","parameters":{"name":"paper_grab","paper_cleanup_mode":true}}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"
# 返回: {"executed":true, "paper_grasped":true, ...}

# ② 回位（跳过 lift，直接用 reverse grab trajectory）
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"gripper_prismatic_move_arm_to_joints","parameters":{"name":"paper_return"}}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"

# ③ 投放垃圾桶 + teleport + collection check
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"gripper_prismatic_take_out","parameters":{"name":"paper_drop"}}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"
# 返回: {"executed":true, "paper_collected":true, ...}

# ④ 回位
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"gripper_prismatic_move_arm_to_joints","parameters":{"name":"paper_return_open"}}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"
```

**⚠️ 流程铁律**：
1. 每次 grab 前机械臂必须在初始位姿（`move_arm_to_joints`）
2. grab 后**不要调 lift**（lift 导致姿态漂移，reverse return 失效）
3. 连续多次 grab 测试需在每次之间重置（`move_arm_to_joints`）
4. 纸团位置 `(1.75,-0.70)` 经测试是最稳定抓取点（精度 9mm）

**关键参数解释**：
| 参数 | 值 | 说明 |
|------|-----|------|
| `ad=0.02` | 2cm | 碰撞膨胀，0.01 太窄（低位封路），0.10 太宽 |
| `seeds=64/32` | IK 64, trajopt 32 | 提高规划成功率，默认 32/16 不够 |
| retry Z offsets | 0, -1cm, -2cm | 目标高度 0.115→0.105→0.095 三档尝试 |
| reverse return | 自动启用 | 抓取成功后 return 用反向轨迹，不重新规划 |

### 已修复的关键 Bug（2026-07-10）

1. **`_build_action` gripper 冲突**：direct mode 下 `_build_action_impl` 总是传 `gripper_joint1_pos=None`，导致 OPEN action 覆盖直接 API 的 CLOSE 指令，机械臂抽搐。修复：传参 `gripper_joint1_pos=gripper_joint1_pos`。

2. **`_sync_paper_waste_proxies` NameError**：`arm_base_pos_w` 在 `_run_tool_api_server` 中未定义。修复：用 `_body_pose_w(robot, base_link)` 计算。

3. **cuRobo worker temp file**：`/tmp/tmp*.json` 被清理导致 FileNotFoundError。修复：改用日志目录持久存储。

4. **Retry 函数 NameError**：闭包内未传入 `parameters`、`paper_cleanup_mode`、`target_pos` 等外部变量。修复：显式传参或从 metadata 读取。

5. **动态 descend NameError**：执行阶段引用了 plan 阶段的局部变量。修复：从 `tool_output` metadata 读取。

### 已知待解决问题

1. **视觉跟随（P1）**：纸团不跟手。GPT 推荐方案：`/World/envs/env_0/runtime_visuals/` 下 visual-only，每 step 手动同步 world pose，不放 articulation 下。
2. **GPU PhysX teleport 警告**：`setGlobalPose() illegal in GPU mode`。仅 take_out 时触发一次，不致命，接受为 MVP hack。
3. **Grab 成功率**：约 50%，retry 可缓解但需更多测试。
4. **Tissue stuck→loose 状态切换**：未实现。

```bash
# ① 抓取 cube（自动回预位姿+规划+执行+下降夹取+抬升）
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"gripper_prismatic_execute_grab","parameters":{"name":"grab","target_frame":"world","target_pos":[1.486,-1.071,0.05]}}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"

# ② 夹持回位
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"gripper_prismatic_move_arm_to_joints","parameters":{"name":"hold"}}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"

# ③ 投放垃圾桶
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"gripper_prismatic_take_out","parameters":{"name":"drop","drop_target_pos":[2.097,-1.068,0.727]}}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"

# ④ 回位
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"gripper_prismatic_move_arm_to_joints","parameters":{"name":"return"}}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"
```

### Brush SOP

```bash
# 多点区域清洁 (初始机器人 brush_rolling_0512)
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"brush_execute_spot_clean","parameters":{"name":"clean","target_frame":"world","target_positions":[[1.663,-0.980,0.428],[1.656,-1.152,0.428],[1.730,-1.211,0.428]],"seed_mode":"cubic"}}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"

# 完成后回位
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"brush_move_arm_to_joints","parameters":{"name":"return"}}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"

# 单点清洁
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"brush_execute_point_clean","parameters":{"name":"point","target_frame":"world","target_pos":[1.834,-1.234,0.428]}}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"
```

## 录制模式 (run_record_mobile_arm.py)

### 启动
```bash
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p .../run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5BrushRolling0512-BrushRolling0512-Dirty-OpenSeat-v0 \
  --num_demos 1 --dataset_file /tmp/demo.hdf5 --speed_scale 150.0 --episode_length_s 60
```

### 键盘控制
```
M       切换 底盘 ↔ 机械臂模式
P       推进录制状态（第 1 次开始录制，第 2 次保存）

[底盘模式]  W/S 前进/后退  A/D 左右平移  Q/E(或Z/X) 旋转
[机械臂模式] W/S = joint1  A/D = joint2  Q/E = joint3  Z/X = joint4
             T/G = joint5  C/V = joint6  I/O = 夹爪闭合/张开（仅 prismatic gripper）
[brush 模式] M 键 = Z轴恒力下压开关（刷马桶）
```

**录制前提**：env_cfg 的 `__post_init__` 必须加：
```python
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
self.recorders = ActionStateRecorderManagerCfg()
```
否则 HDF5 不会生成。

**P 键防抖**：已在 `run_record_mobile_arm.py` 加 1 秒冷却（evdev 回调在独立线程，连按会循环多圈）。
**evdev 权限**：`sudo usermod -aG input stardust` 后需**重启电脑**。

### HDF5 数据结构
```
demo_X/actions          (T, 10) float32  ← 4 轮 + 6 关节动作
demo_X/processed_actions(T, 10)          ← 加偏移后的动作
demo_X/obs/
  ├── cam_tool_rgb      (T, 460, 460, 3) uint8
  ├── cam_tool_depth    (T, 460, 460, 1) float32
  ├── joint_pos         (T, 10)
  ├── cam_in_base_tf    (T, 16)  ← 相机→基座变换矩阵
  └── cam_in_tool_tf    (T, 16)  ← 相机→工具变换矩阵
```
用途：模仿学习训练数据（RGB→动作映射），对接 RoboTwin/SimpleVLA。

## 场景物体修改

文件：`config/.../mecanum730_xms5_gripper_joint_prismatic_dirty_open_seat_env_cfg.py`

### 已有物体改位置
```python
grasp_test_cube_1 = RigidObjectCfg(
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.68, -0.80, 0.021), ...), ...)
trash_bin = RigidObjectCfg(
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.55, -0.70, 0.010), ...), ...)
```

### 添加新物体
```python
# MeshCuboidCfg 方式（cube）
my_obj = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/my_obj",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(x,y,z), rot=(w,x,y,z)),
    spawn=sim_utils.MeshCuboidCfg(size=(sx,sy,sz), rigid_props=..., mass_props=...),
)
# UsdFileCfg 方式（USD 文件）
my_obj = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/my_obj",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(x,y,z)),
    spawn=sim_utils.UsdFileCfg(usd_path="...usda", rigid_props=...),
)
```

### 如果在 common cfg 里不生效
在 `common_curobo_pose_plan_test_env_cfg.py` 的 `SceneCfg.__post_init__` 里强制覆写（绕开 @configclass 缓存）：
```python
def __post_init__(self):
    super().__post_init__()
    self.robot = make_common_robot_cfg("brush_rolling_0512")
    self.grasp_test_cube_1.init_state.pos = (1.68, -0.80, 0.021)
    self.grasp_test_cube_2.init_state.pos = (1.66, -0.96, 0.408)
    self.grasp_test_cube_3.init_state.pos = (1.75, -0.60, 0.021)
    self.trash_bin.init_state.pos = (1.55, -0.70, 0.010)
```
改后需清除 `__pycache__` 重启。

**世界坐标系**：X=左右, Y=前后（机器人朝向-Y）, Z=高度(地面=0)。机器人初始：`(1.832, -0.264, 0.010)`。

## cuRobo 障碍体机制

碰撞障碍体**每次工具调用时动态导出**，非启动时静态加载。采集逻辑在 `curobo_scene_obstacles.py`。

### 采集路径
- mesh: 马桶陶瓷（全部马桶 + 运行时开启姿态）
- cuboid: `/World/.../WallBoxes`, `/World/.../PartitionBoxes`（`Cube+CollisionAPI`）
- cuboid: `/World/.../CuroboOnlyWalls`（`Cube+curobo:obstacle` 属性，**不加 CollisionAPI**，防 PhysX 爆炸）
- **玻璃门面板**：2 个 `UsdGeom.Cube`（各 0.40×0.02×2.46m），位于 CuroboOnlyWalls 下，`curobo:obstacle=True`，首次工具调用懒加载

过滤窗口（机器人基坐标系）：X[box_collision_front_min_x_m, 2m] 前方, Y[±1m] 横向。AABB 相交判断。
**注意**：`box_collision_front_min_x_m` 默认 0.0，玻璃门面板在机器人坐标中 X≈0.03 可能被截断，需设为 `-0.5`。

### 墙壁检测方案演进

| 方案 | 结果 |
|---|---|
| 导出 Wall mesh 顶点 | ❌ USDC 二进制 geo.usd 返回 0 顶点 |
| BBoxCache 自动提取 | ❌ Wall_0000 extentsHint 是整个房间 7m×3.5m 合集，注入后与机器人重叠 PhysX 爆炸 |
| 手工 cuboid 坐标 | ❌ 猜不准，挡路或穿墙 |
| **PhysX raycast 懒加载** | ✅ 首次工具调用时从 ROI 外侧 raycast，294 hits → 聚类 → 5 面墙 cuboid（curobo:obstacle 标记，无 CollisionAPI） |

### 已执行的对准精度优化

同事反馈：stiffness/settle steps 已调过，IK 精度没问题。env.step 步数很少 → joint2/3 没 hold 住。

| 优化 | 结果 |
|---|---|
| 关重力 (`disableGravity=True`) | 同事建议，防止 joint2/3 重力漂移 |
| `collision_activation_distance=0.02`（⚠️ 不要 0.10） | 碰撞边距。0.10 在狭窄空间封死低位路径 |

### 当前已知问题

1. **低位 trajopt 失败（已定位 → ad=0.02）**：`collision_activation_distance=0.10` 在狭窄卫生间里把低位路径封死。降到 **0.02** 后 Z=0.05 抓取成功（`executed: True`，pos_err 5.4cm）。原理：ad 等效于障碍物膨胀半径，0.10m 过大导致 trajopt 无可行路径。
   - ✅ ad=0.02：cube2 Z=0.05 → executed: True
   - ❌ ad=0.10：cube2 Z=0.05 → executed: False
2. **抓取偏差已定位 → descend 重规划是根因**：主轨迹精度 2.8cm ✅。descend 阶段用 cuRobo 重规划把 tool_tip 从 (1.432,-1.148,0.037) 压到 (1.420,-1.099,**0.000**) 贴地，引入 ~7cm 额外偏差。**结论：descend/lift 不应走 cuRobo 全局重规划，应改成直线笛卡尔 servo 或直接禁用（`descend_m=0`）。** 抓取精度指标也应从 lift 后 final 改成夹爪闭合前位置。
   - 逐阶段数据见上方
3. **玻璃门材质不可见**：841837 Door_YS3UEUZVAJ2V4PTUKU888888_mdynP (OmniGlass) 在 Isaac Sim 4.5.0.0 中不渲染。改用 raycast 函数中生成 UsdGeom.Cube 代理面板（0.40×0.02×2.46m，curobo:obstacle）。

## 关键陷阱

1. **场景文件**：`start_open_seat_collision_box_fixed.usda` 是空文件（75行），必须用 `start_open_seat.usda` (71KB)。
2. **toilet_a geo.usd**：所有分支均为空（492字节），ArticulationCfg 必须保持注释状态。
3. **录制 recorders**：默认 `DefaultEmptyRecorderManagerCfg` 有 0 active terms，必须加 `ActionStateRecorderManagerCfg()`。
4. **base_keyboard_teleop**：容忍值 0.005→0.15rad (`run_curobo_pose_plan_test.py:3202`)。
5. **cuboid**：`CreateSizeAttr(1.0)` 必调，否则 scale 尺寸翻倍。
6. **`move_to_pre_target`** 默认 `true`，连续抓取时第二帧起设 `false`。
7. **target_pos 格式**：world 坐标系 `[x, y, z]`。

## 卫生纸废弃物仿真（Paper Waste Simulation）

**核心路线**：视觉软体 + 刚体/多刚体代理 + 状态机。不使用 particle cloth / deformable 作为主线。

### 工程对象三层

| 层 | 职责 |
|---|---|
| 视觉层 | 看起来像湿的、揉皱的、脏污卫生纸 |
| 物理层 | 用刚体/多刚体代理产生可交互效果 |
| 任务层 | 状态机记录 stuck / loose / grasped / collected / cleaned |

### 三类卫生纸废弃物

| 类型 | 优先级 | 建模方式 | cuRobo 代理 |
|------|--------|----------|-------------|
| A: 湿纸团 `used_wet_paper_clump` | P0 | 8~12 sphere/capsule compound rigid body, mass 0.02kg, diam 4-8cm | sphere radius 0.05m 或 cuboid 0.10×0.10×0.06 |
| B: 贴地湿纸片 `used_flattened_tissue` | P1 | 8~20 thin box segment, 碰撞厚度 3-8mm, mass 0.001-0.005kg/片 | thin cuboid 0.14×0.10×0.02 |
| C: 卫生纸卷/垂纸 `toilet_paper_roll` | P2 | cylinder visual + thin cuboid collider | roll cuboid/sphere + sheet cuboid |

### 新增文件结构

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/
├── data/props/paper_waste/
│   ├── used_wet_paper_clump_01.usda
│   ├── used_flattened_tissue_01.usda
│   └── materials/
│       ├── tissue_clean.mdl
│       ├── tissue_used_wet.mdl
│       └── tissue_dirty_stain.mdl
├── utils/
│   ├── paper_waste_state.py    # 状态机
│   ├── paper_waste_spawner.py  # 生成器
│   ├── paper_waste_metrics.py  # 收集判定+指标
│   └── paper_waste_debug.py    # 调试日志
```

### 状态机 (paper_waste_state.py)

```python
class PaperWastePhase:
    STUCK = "stuck"        # 粘在地上
    LOOSE = "loose"        # 松动，可被推动
    GRASPED = "grasped"    # 被夹爪抓住
    PUSHED = "pushed"      # 被刷子推动中
    COLLECTED = "collected" # 进入收集区
    REMOVED = "removed"    # 被推出清洁区域
```

状态转换：
- STUCK → LOOSE：刷子/推板接触累计 contact_steps >= N
- LOOSE → GRASPED：夹爪闭合且 paper center 距 gripper center < 阈值
- LOOSE → COLLECTED：进入垃圾桶 AABB
- GRASPED → COLLECTED：夹爪张开且进入垃圾桶
- GRASPED → LOOSE：掉落地面

### 阶段化实施顺序

**阶段 0（MVP）**：湿纸团夹取闭环
1. Task 1: `utils/paper_waste_state.py` — PaperWasteObject dataclass, 状态枚举, update/mark_collected/inside_aabb
2. Task 2: `utils/paper_waste_spawner.py` — `spawn_used_wet_paper_clump(stage, prim_path, position, radius, mass, curobo_proxy=True)`，创建 visual + collision proxy + RigidBodyAPI/CollisionAPI/MassAPI + paper_waste:* 和 curobo:* 自定义属性
3. Task 3: env_cfg prestartup/startup EventTerm 中生成 1 个纸团，pos=(1.409, -1.157, 0.035)
4. Task 4: 修改 `curobo_scene_obstacles.py`，识别 `curobo:obstacle=True` + `curobo:kind="paper_waste"` 的 proxy，未 collected 导出 cuboid，collected 跳过
5. Task 5: `utils/paper_waste_metrics.py` — 垃圾桶 collection AABB + `check_collected(paper_pos_w)`
6. Task 6: gripper 流程添加 paper cleanup demo（接近→闭合→逻辑attach→lift→垃圾桶→detach→collection zone 判定）
7. Task 7: 刷子推纸团流程（阶段 1）
8. Task 8: 随机化（位置/尺寸/材质/stuck/mass）

**阶段 1**：贴地湿纸片（先 thin cuboid，再 D6 joint 多片段链）
**阶段 2**：stuck/unstuck 状态（状态机模拟湿纸粘地，不靠真实粘性）

### 关键参数

**湿纸团刚体**：
```
mass = 0.02 kg
linear_damping = 2.0~5.0
angular_damping = 2.0~5.0
dynamic_friction = 1.2, static_friction = 1.5
restitution = 0.0
```

**调参**：乱飞 → mass↑ damping↑；推不动 → friction↓ mass↓

**cuRobo proxy**：
- 湿纸团：cuboid [0.10, 0.10, 0.06]
- 贴地纸片：cuboid [0.14, 0.10, 0.02]
- 被夹住后：不作为 environment obstacle 导出

### 避免踩坑

1. **visual mesh 可复杂，collision proxy 必须简单**（sphere/capsule/convex，不做 custom geometry 碰撞）
2. **不要一开始做真实软体**（不碰 particle cloth / deformable surface / 撕裂）
3. **被夹住的纸团不作为 cuRobo 环境障碍**（否则夹爪和纸团碰撞导致投放规划失败）
4. **抓取评价用闭合前位置，不用 lift 后 final pose**
5. **第一版用逻辑 attach**（每步把 paper root pose 跟随 gripper center），不追求 PhysX 真实夹取
6. **D6 joint 多片段链只做增强版本**，不作为第一阶段主线（TGS 下行为可能异常，可考虑 PGS）

### MVP 验收标准

1. Stage 中出现 used_wet_paper_clump_01，外观像湿纸团
2. 机器人/刷子/夹爪碰到它时不穿透
3. cuRobo obstacle export 能看到 paper_waste_proxy
4. 夹爪闭合后逻辑 attach，lift 后纸团跟随
5. 到垃圾桶后 detach，进入 collection zone 后 collected=True
6. 输出 success=True

### 调试日志格式

每次失败时收集：
```
paper prim path / position / state
gripper tool pos
paper 是否 exported as cuRobo obstacle
cuRobo plan result：IK success / trajopt success / failure stage
paper 是否 attach / lift 后是否跟随
drop 后 paper 是否进入 collection AABB
collected 判断结果
```

## 已修改代码清单（2026-07-10 更新）

### 新增文件
| 文件 | 职责 |
|------|------|
| `utils/paper_waste_state.py` | 状态机：PaperWastePhase 枚举 + PaperWasteObject dataclass |
| `utils/paper_waste_spawner.py` | 生成器：`spawn_used_wet_paper_clump()` + `spawn_used_flattened_tissue()` + 皱褶 visual 生成 |
| `utils/paper_waste_metrics.py` | 垃圾桶 AABB + `check_paper_collected()` |
| `utils/paper_waste_debug.py` | 调试日志 |

### 修改文件
| 文件 | 修改 |
|------|------|
| `env_cfg.py` | `spawn_paper_waste` EventTerm（3 clump + 3 tissue），删 cube2/cube3 |
| `run_curobo_pose_plan_test.py` | 大量修改（见下） |
| `curobo_scene_obstacles.py` | 采集 paper_waste proxy，跳过 collected/grasped/inactive |
| `curobo_tool_specs.py` | `paper_cleanup_mode` + `paper_prim_path` 参数 |
| `curobo_worker_client.py` | temp file → log 目录持久存储 |
| `curobo_arm_joints.py` | IK 多 seed 选择 |

### `run_curobo_pose_plan_test.py` 关键修改
| 功能 | 说明 |
|------|------|
| `_build_action` gripper fix | direct mode 传 `gripper_joint1_pos` 而非 None |
| `_check_paper_grasp()` | 距离检测（默认 0.12m） |
| `_try_plan_with_z_offsets()` | GPT P0: Z 方向 3 档 retry（0, -1cm, -2cm） |
| `_sync_paper_waste_proxies()` | 每次工具调用前同步 proxy |
| `_logical_attach_paper_safe()` | P1 v2: visual-only 附着（暂未启用） |
| `_update_carried_paper_visuals()` | 每步同步 carried visual（暂未启用） |
| Reverse return | `execute_gripper_prismatic_move_arm_to_joints` 优先反向轨迹 |
| Dynamic descend | paper 模式下降高度动态计算（target_z - safe_close_z） |
| Teleport collection | take_out 时瞬移纸团到垃圾桶（MVP hack） |
| Pre-target graceful fallback | paper 模式 pre-target 失败不崩溃 |
