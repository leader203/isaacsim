# 抓取→复位→投放 SOP（复现记录）

## 环境

```bash
cd /home/stardust/czs/roboos
conda activate isaaclab
```

## 启动命令

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./isaaclab.sh -p \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_curobo_pose_plan_test.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Common-CuroboPosePlanTest-Dirty-OpenSeat-v0 \
  --initial_robot_id gripper_prismatic \
  --curobo_collision_activation_distance 0.02 \
  --curobo_box_collision_front_min_x_m -0.5 \
  --post_target_descend_m 0 \
  --tool_api_server --tool_api_host 127.0.0.1 --tool_api_port 8000
```

> **参数说明：**
> - `collision_activation_distance=0.02` — **必须 0.02，0.10 会封死低位路径**
> - `box_collision_front_min_x_m=-0.5` — 玻璃门面板在机器人坐标 X≈0.03 需负值才能通过过滤
> - `post_target_descend_m=0` — 禁用 cuRobo descend 重规划（重规划会把末端压到地面）

## 轮询函数

```bash
wait_tool_job() {
  local req="$1" resp state
  while true; do
    resp=$(curl -sS "http://127.0.0.1:8000/tools/jobs/$req")
    state=$(printf '%s' "$resp" | python -c 'import sys,json; print(json.load(sys.stdin)["state"])')
    if [ "$state" != "queued" ] && [ "$state" != "running" ]; then
      printf '%s' "$resp" | python -m json.tool; break
    fi
    sleep 1
  done
}
```

## 完整 SOP 流程

> **核心改进**：抓取成功后先 `local_lift` 把机械臂从贴地低位抬到安全高度（Z+0.18），
> 再执行 cspace 复位和 trajopt 投放。避免墙壁 cuboid 封死低位→高位的路径。

### ① 抓取 cube2

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"gripper_prismatic_execute_grab","parameters":{"name":"grab_cube2","target_frame":"world","target_pos":[1.409,-1.157,0.05]}}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"
```

### ② 局部抬升（撤离低位，Z+0.18）

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"gripper_prismatic_local_lift","parameters":{"name":"local_lift","lift_height_m":0.18}}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"
```

### ③ 夹持回位

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"gripper_prismatic_move_arm_to_joints","parameters":{"name":"return_pre_target_grasp_hold"}}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"
```

### ④ 投放垃圾桶（默认投放位置）

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"gripper_prismatic_take_out","parameters":{"name":"take_out"}}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"
```

### ⑤ 回位（夹爪打开）

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"gripper_prismatic_move_arm_to_joints","parameters":{"name":"return_pre_target_open"}}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')
wait_tool_job "$REQ"
```

## 物体原始坐标

| 物体 | 位置 |
|------|------|
| cube1 | (1.486, -1.071, 0.021) |
| cube2 | (1.409, -1.157, 0.021) |
| cube3 | (1.355, -1.218, 0.021) |
| trash_bin | (2.097, -1.095, 0.010) |

## 前置代码修改

| 文件 | 修改 | 说明 |
|------|------|------|
| `curobo_scene_obstacles.py:504` | `_ensure_raycast_wall_cuboids` | PhysX raycast 墙壁扫描懒加载 |
| `curobo_scene_obstacles.py:619-637` | 玻璃门面板 Cube | 2个 curobo:obstacle 面板（0.40×0.02×2.46m） |
| `curobo_scene_obstacles.py:536` | `_accept_path` 加 `"door"` | 让 raycast 接受 door 路径 |
| `curobo_scene_obstacles.py:654-670` | 支持 `curobo:obstacle` 属性采集 | |
| `curobo_scene_obstacles.py:290` | `candidates[:1]` → `candidates` | 导出全部马桶 |
| `curobo_scene_obstacles.py:26-63` | `_runtime_toilet_lid_seat_pose_maps` | 马桶盖开启姿态 |
| `env_cfg:579` | `sim.enable_scene_query_support=True` | PhysX raycast |
| `common_env_cfg.py:95-101` | arm effort_limit | joint1=400 joint2=800 joint3=400 joint4=400 joint5=400 joint6=400 |
| `run_curobo_pose_plan_test.py:1150` | **Bug 修复**：`action_robot.data.joint_pos` → `arm_target_q` | gripper grasp 时 arm 漂移 |
| `run_curobo_pose_plan_test.py:2632-2683` | **新增** `_make_phase_planner_cfg` 阶段化墙壁过滤 | lift/return/takeout 阶段排除玻璃门面板+远墙 |
| `run_curobo_pose_plan_test.py:4318-4414` | **新增** `gripper_prismatic_local_lift` 工具 | 抓取后局部抬升 Z+0.18（trajopt） |
| `curobo_tool_specs.py:419-437` | **新增** `gripper_prismatic_local_lift_spec` | 工具 API spec |
| `run_curobo_pose_plan_test.py:3202` | base_keyboard_teleop 容忍值 0.15rad | |
| `run_curobo_pose_plan_test.py:4697` | 触发 `_ensure_raycast_wall_cuboids` | |

## 已知问题

1. **抓取后 cspace 复位失败**：raycast 墙壁 cuboid 遮挡低位→高位 cspace 路径
   - **缓解方案**：抓取后先 `local_lift` 抬升到 Z≥0.18，再 cspace 复位
2. **投放路径撞墙**：从抓取位置到垃圾桶的 trajopt 被墙 cuboid 阻挡，cube 掉落
   - **缓解方案**：抬升到安全高度后再投放，避免从低位直接规划长距离 trajopt
3. **低位 trajopt 必须 ad=0.02**（0.10 会封死路径）
4. **descend 重规划 bug**：cuRobo descend 把末端压到地面（用 `--post_target_descend_m 0` 绕过）
5. **gripper grasp arm 漂移**：`_build_action` direct_tool_control 分支用了当前 joint_pos 代替 arm_target_q → 已修复（第 1150 行）
