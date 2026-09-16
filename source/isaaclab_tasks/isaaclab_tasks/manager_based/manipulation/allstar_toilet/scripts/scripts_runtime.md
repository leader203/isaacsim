# scripts runtime

以下命令默认从 IsaacLab 仓库根目录执行：

```bash
cd <IsaacLab repo root>
```

## generate_robotbuilder_link2plus_collision_urdf.py

用于给 URDF 中从机械臂 `link2` 开始的 link 自动生成 RobotBuilder collision spheres，也可以同时给底盘 `base_link` 生成或复用 collision spheres。

### 通用命令

```bash
TERM=xterm ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/usdandurdf/generate_robotbuilder_link2plus_collision_urdf.py \
  --urdf <input.urdf> \
  --output <output.urdf> \
  --start-link <link_name> \
  --wrist-link <link_name> \
  --include-base-link-collision \
  --base-link-name <link_name> \
  --base-link-density <float> \
  --reuse-base-link-spheres-urdf <source.urdf> \
  --disable-base-arm-motion-safe \
  --enable-base-arm-motion-safe \
  --base-arm-root-joint <joint_name> \
  --base-arm-clearance-radius-m <float> \
  --base-arm-clearance-below-m <float> \
  --base-arm-clearance-above-m <float> \
  --body-density <float> \
  --end-effector-density <float> \
  --movable-end-effector-density <float> \
  --ee-density-multiplier <float> \
  --tool-tip-name <link_name> \
  --disable-ee-motion-safe \
  --gripper-samples <int> \
  --self-collision-clearance-m <float> \
  --min-sphere-radius-m <float>
```

### 参数说明

| 参数 | 作用 |
|---|---|
| `--urdf` | 输入 URDF，不会修改原文件。 |
| `--output` | 输出生成后的 URDF。 |
| `--start-link` | 指定从哪个 link 开始生成机械臂 collision spheres；默认自动找 `*link2`。 |
| `--wrist-link` | 指定腕部/final arm link；其后的 link 会被当作末端工具处理。 |
| `--include-base-link-collision` | 让脚本也给底盘 `base_link` 写 collision spheres。别名：`--include-base-link`。 |
| `--base-link-name` | 底盘 link 名称，默认 `base_link`。 |
| `--base-link-density` | 重新拟合 `base_link` 时的 sphere 密度；复用 URDF spheres 时无效。默认跟随 `--body-density`。 |
| `--reuse-base-link-spheres-urdf` | 从另一个 URDF 复制 `base_link` 的 sphere collisions，不重新拟合。别名：`--base-link-spheres-urdf`。 |
| `--disable-base-arm-motion-safe` | 禁用底盘和机械臂安装点附近的安全裁剪。 |
| `--enable-base-arm-motion-safe` | 启用底盘和机械臂安装点附近的安全裁剪。拟合 `base_link` spheres 时默认开启，复用 spheres 时默认关闭。 |
| `--base-arm-root-joint` | 指定底盘到机械臂基座的 joint，比如 `root_joint`；不指定时默认使用到机械臂起始 link 路径上的第一个 joint。 |
| `--base-arm-clearance-radius-m` | 安装点周围 XY 方向保护半径，单位米。 |
| `--base-arm-clearance-below-m` | 安装点下方保护高度，单位米。 |
| `--base-arm-clearance-above-m` | 安装点上方保护高度，单位米。 |
| `--body-density` | 普通机械臂 link 的 sphere 拟合密度。 |
| `--end-effector-density` | 固定末端工具 link 的 sphere 拟合密度。别名：`--tool-density`。 |
| `--movable-end-effector-density` | 可动末端 link 的 sphere 拟合密度，比如 gripper finger。别名：`--gripper-density`。 |
| `--ee-density-multiplier` | 同时放大 `--end-effector-density` 和 `--movable-end-effector-density`。 |
| `--tool-tip-name` | 指定不生成 collision 的工具 tip frame，可重复传多个。 |
| `--disable-ee-motion-safe` | 禁用 gripper/末端运动安全处理。 |
| `--gripper-samples` | 检查 gripper prismatic 运动时采样数量。 |
| `--self-collision-clearance-m` | sphere 间自碰撞安全间隙，单位米。 |
| `--min-sphere-radius-m` | 小于该半径的 sphere 会被删除，单位米。 |

### 示例

#### mecanum730_xms5_gripper_joint_prismatic

```bash
TERM=xterm ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/usdandurdf/generate_robotbuilder_link2plus_collision_urdf.py \
  --urdf source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/data/robots/chassis_arm/motion_wheel_arm_simple_sphere_urdf/mecanum730_xms5_brush_rolling_sanheyi_short.urdf \
  --output source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/data/robots/chassis_arm/motion_wheel_arm_simple_sphere_urdf/mecanum730_xms5_brush_rolling_sanheyi_short_robotbuilder_link2plus_ee_motion_safe_collision.urdf \
  --wrist-link link6 \
  --ee-density-multiplier 15 \
  --include-base-link-collision \
  --base-link-name base_link \
  --base-arm-root-joint root_joint \
  --enable-base-arm-motion-safe \
  --base-link-density 10.0 \
  --base-arm-clearance-radius-m 0.10 \
  --base-arm-clearance-below-m 0.20 \
  --base-arm-clearance-above-m 0.5
```

#### visualize_curobo_mesh_obstacle.py

用于查看 `run_curobo_pose_plan_test.py` 导出的 cuRobo collision obstacle OBJ 网格，默认读取：

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/logs/curobo_collision_obstacles/curobo_scene_collision_all_20260531_221358.obj
```

默认用 `matplotlib` 打开交互窗口，并在终端打印 OBJ 路径、顶点数、面数、bounds、extents 和 centroid：

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/visualize_curobo_mesh_obstacle.py
```

指定其他 OBJ：

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/visualize_curobo_mesh_obstacle.py \
  --obj source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/logs/curobo_collision_obstacles/<mesh.obj>
```

保存 PNG 且不打开窗口，适合 headless 环境：

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/visualize_curobo_mesh_obstacle.py \
  --save /tmp/toilet_curobo_obstacle.png \
  --no-show
```

使用 `trimesh` viewer：

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/visualize_curobo_mesh_obstacle.py \
  --viewer trimesh
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--obj` | 指定要查看的 cuRobo obstacle OBJ 文件。 |
| `--viewer {matplotlib,trimesh}` | 选择显示后端；默认 `matplotlib`。 |
| `--save <png>` | 用 `matplotlib` 后端保存截图。 |
| `--no-show` | 不打开交互窗口，通常和 `--save` 一起使用。 |
| `--max-faces <int>` | `matplotlib` 最多绘制的面数；`<=0` 表示绘制全部面。 |
| `--wireframe` | 在 `matplotlib` 视图中显示网格边线。 |
| `--azim <float>` | `matplotlib` 相机方位角。 |
| `--elev <float>` | `matplotlib` 相机俯仰角。 |





## cuRobo Planner

`run_curobo_pose_plan_test.py` 当前保留 FASTAPI 工具服务入口：

| 模式 | 用途 |
|---|---|
| FASTAPI | 启动工具 API 服务后，通过 `/tools/invoke` 远程调用工具。 |

### FASTAPI

启动公共 API 服务。`--initial_robot_id` 决定初始 active robot，后续可用 `switch_robot` 切换：

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_curobo_pose_plan_test.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Common-CuroboPosePlanTest-Dirty-OpenSeat-v0 \
  --initial_robot_id gripper_prismatic \
  --tool_api_server \
  --tool_api_host 127.0.0.1 \
  --tool_api_port 8000
```

`/tools/invoke` 是异步提交接口。第一次返回 `queued` 是正常的，工具由仿真主循环执行，完成后用 `request_id` 查询。下面的 `wait_tool_job` 会轮询到 `succeeded` 或 `failed` 后打印完整结果：

```bash
wait_tool_job() {
  local req="$1"
  local resp state
  while true; do
    resp=$(curl -sS "http://127.0.0.1:8000/tools/jobs/$req")
    state=$(printf '%s' "$resp" | python -c 'import sys,json; print(json.load(sys.stdin)["state"])')
    if [ "$state" != "queued" ] && [ "$state" != "running" ]; then
      printf '%s' "$resp" | python -m json.tool
      break
    fi
    sleep 1
  done
}
```

#### common tools

公共 task 只保留一个 active `/robot`。`switch_robot` 会按 profile 重建 `/robot`，并按 active USD 自动解析对应 URDF 和 cuRobo robot cfg。active 是 brush 时允许 brush 工具，active 是 gripper 时允许 gripper 工具；类型不匹配的工具会直接返回错误。

```text
switch-ready joint1..joint6 deg = [-2.89, -22.96, -28.61, 0.0, -137.52, 0.0]
```

可用 profile：

| robot_id | 类型 | 可用工具 |
|---|---|---|
| `brush_rolling_0512` | brush | common + brush |
| `brush_rolling_sanheyi_long` | brush | common + brush |
| `brush_rolling_sanheyi_short` | brush | common + brush |
| `gripper_prismatic` | gripper | common + gripper |
| `gripper_prismatic_sanheyi_full` | gripper | common + gripper |

1. `list_robot_profiles`

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "list_robot_profiles",
    "parameters": {}
  }' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

wait_tool_job "$REQ"
```

2. `get_active_robot`

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "get_active_robot",
    "parameters": {}
  }' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

wait_tool_job "$REQ"
```

3. `switch_robot`

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "switch_robot",
    "parameters": {
      "name": "switch_to_gripper_brush",
      "robot_id": "brush_rolling_sanheyi_long"
    }
  }' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

wait_tool_job "$REQ"
```

4. `stretch_end_tool`

C-space 到伸展姿态，再回 shared planning/pre-target 姿态。可选 `joint_pos_deg` 覆盖 6 个 arm joints，单位为 degree。

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "stretch_end_tool",
    "parameters": {
      "name": "stretch_end_01"
    }
  }' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

wait_tool_job "$REQ"
```

覆盖 joint 示例：

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "stretch_end_tool",
    "parameters": {
      "name": "stretch_end_custom",
      "joint_pos_deg": [5.7, -14.4, -20.0, 0.0, 8.6, 0.0]
    }
  }' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

wait_tool_job "$REQ"
```

5. `execute_spot_capture`

C-space 到拍摄关节姿态，保存 RGB/depth 和 camera-to-arm-base TF，再返回 planning pose。可选 `joint_pos_deg` 覆盖 6 个 arm joints，单位为 degree。

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "execute_spot_capture",
    "parameters": {
      "name": "spot_capture_custom",
      "joint_pos_deg": [2.8, -2.9, -25.7, 0.0, -94.5, 0.0]
    }
  }' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

wait_tool_job "$REQ"
```

6. `base_keyboard_teleop`

brush/gripper 共用的底盘键盘控制工具。调用后先用 cuRobo C-space 到机械臂预备姿态：

```text
joint1..joint6 deg = [-2.89, -22.96, -28.61, 0.0, -137.52, 0.0]
```

随后进入 motion-based base 键盘控制：更新底盘 root pose/root velocity，并同步轮子视觉滚动；机械臂保持预备姿态。按键：

```text
W/S 前进/后退
A/D 左/右平移
Z/X 左/右旋转
Q 退出
```

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "base_keyboard_teleop",
    "parameters": {
      "name": "base_keyboard_teleop_01",
      "speed_scale": 150.0,
      "max_duration_s": 300.0,
      "stop_key": "Q"
    }
  }' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

wait_tool_job "$REQ"
```

可选 `pre_arm_joint_pos_deg` 覆盖预备姿态；`max_duration_s <= 0` 时不会按时间自动退出，只能用 `stop_key` 退出。

#### brush tools

调用前 active robot 必须是 brush profile。

7. `brush_execute_point_clean`

单点刷洗。`target_frame` 支持 `world` 或 `base`。默认只需传入 `target_pos`，工具会使用 brush profile 的默认姿态逻辑。

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "brush_execute_point_clean",
    "parameters": {
      "name": "remote_pose_01",
      "target_frame": "world",
      "target_pos": [1.834, -1.234, 0.428]
    }
  }' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

wait_tool_job "$REQ"
```

如果需要指定工具姿态，可额外传入 `target_quat`。格式为 `[w, x, y, z]`，坐标系与 `target_frame` 一致：

```json
"target_quat": [-0.004, -0.715, -0.699, 0.016]
```

8. `brush_execute_spot_clean`

多点区域刷洗。最多保留 8 个规划点；输入更多点时保留首尾并采样中间点。默认只需传入 `target_positions`，每个点会使用 brush 默认姿态逻辑。

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "brush_execute_spot_clean",
    "parameters": {
      "name": "remote_spot_clean_01",
      "target_frame": "world",
      "target_positions": [
        [1.663, -0.980, 0.428],
        [1.656, -1.152, 0.428],
        [1.730, -1.211, 0.428]
      ],
      "seed_mode": "cubic"
    }
  }' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

wait_tool_job "$REQ"
```

如果需要为每个点指定工具姿态，可额外传入 `target_quats`。格式为 `[[w, x, y, z], ...]`，数量必须与 `target_positions` 一致，坐标系与 `target_frame` 一致：

```json
"target_quats": [
  [-0.004, -0.715, -0.699, 0.016],
  [-0.004, -0.715, -0.699, 0.016]
]
```

9. `brush_move_arm_to_joints`

brush profile 下回到 planning/pre-target 姿态。可选 `joint_pos_deg` 覆盖 6 个 arm joints，单位为 degree。

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "brush_move_arm_to_joints",
    "parameters": {
      "name": "return_to_planning_pose"
    }
  }' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

wait_tool_job "$REQ"
```

#### gripper tools

调用前 active robot 必须是 gripper profile。

10. `gripper_prismatic_move_arm_to_joints`

执行 C-space 规划移动到关节目标，默认目标为 shared planning/pre-target joint pose，并保持夹爪闭合。可选 `joint_pos_deg` 覆盖 6 个 arm joints；可选 `gripper_joint1_pos` 覆盖夹爪保持位置，默认约 `0.020` m。

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "gripper_prismatic_move_arm_to_joints",
    "parameters": {
      "name": "return_pre_target_grasp_hold"
    }
  }' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

wait_tool_job "$REQ"
```

11. `gripper_prismatic_execute_grab`

执行：

```text
可选 pre-target
-> pose plan 到 target
-> 执行 target trajectory
-> hold 并测量 target pose error
-> pose plan 世界 -Z 下探，夹爪打开
-> 在下探点闭合 prismatic gripper
-> pose plan 世界 +Z 抬升，夹爪闭合
```

`target_frame` 支持 `world` 或 `base`。默认只需传入 `target_pos`，工具会使用当前 gripper profile 的默认姿态。`move_to_pre_target` 默认 `true`，连续调用时可设为 `false`。

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "gripper_prismatic_execute_grab",
    "parameters": {
      "name": "grab_01",
      "target_frame": "world",
      "target_pos": [1.486, -1.101, 0.05]
    }
  }' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

wait_tool_job "$REQ"
```

如果需要指定抓取目标姿态，可额外传入 `target_quat`。格式为 `[w, x, y, z]`，坐标系与 `target_frame` 一致：

```json
"target_quat": [-0.004, -0.715, -0.699, 0.016]
```

12. `gripper_prismatic_take_out`

执行：

```text
pose plan 到 drop pose，夹爪闭合
-> 打开夹爪释放
-> C-space 返回 pre-target，夹爪打开
```

默认使用下面的 drop pose：

```text
drop_target_pos_w = [2.097, -1.068, 0.727]
drop_target_quat_wxyz = [-0.057, -0.598, -0.798, -0.043]
```

```bash
REQ=$(curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "tool_name": "gripper_prismatic_take_out",
    "parameters": {
      "name": "take_out_01"
    }
  }' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["request_id"])')

wait_tool_job "$REQ"
```

如果需要覆盖投放位姿，可额外传入 `drop_target_pos` 和 `drop_target_quat`。`drop_target_quat` 格式为 world frame `[w, x, y, z]`：

```json
"drop_target_pos": [2.097, -1.068, 0.727],
"drop_target_quat": [-0.057, -0.598, -0.798, -0.043]
```

## Recorder

### 参数说明

| 参数 | 作用 |
|---|---|
| `--task` | 指定要录制的 gym environment id。 |
| `--num_demos` | 要录制的 demonstration 数量；`0` 表示不限数量。 |
| `--dataset_file` | 输出 HDF5 数据集路径。 |
| `--speed_scale` | 底盘轮速控制缩放，影响移动灵敏度。 |
| `--arm_scale` | 每次按键对应的机械臂关节位置增量，单位 rad。 |
| `--step_hz` | 控制循环目标频率，单位 Hz；`0` 表示不做限频，默认 `30`。 |
| `--teleop_device` | 遥操作输入设备；可选 `keyboard` 或 `spacemouse`，默认 `keyboard`。 |
| `--episode_length_s` | 每个 episode 的仿真时长；`0` 表示使用环境默认值。 |
| `--arm_debug_interval` | 每隔 N 次 `env.step` 打印一次机械臂关节调试信息；`0` 表示关闭。 |
| `--camera_debug_interval` | 每隔 N 次 `env.step` 打印一次 RGB/depth/相机位姿诊断；`0` 表示关闭。 |
| `--show_camera_pose_debug` | 是否在 viewport 中绘制相机位置、三轴和视锥；需要非 headless，默认 `False`。 |
| `--gripper_joint1_pos` | 旋转关节夹爪任务使用的 gripper_joint1 目标位置，单位 rad；prismatic gripper 任务通常不用。 |
| `--prismatic_gripper_pos` | prismatic gripper 初始闭合指令，单位 m。 |
| `--prismatic_gripper_step` | prismatic gripper 每次控制步进增量，单位 m。 |
| `--prismatic_gripper_min` | prismatic gripper 最小闭合指令，单位 m。 |
| `--prismatic_gripper_max` | prismatic gripper 最大闭合指令，单位 m。 |
| `--keyboard_hold_sec` | 终端键盘 fallback 模式下按键保持时间窗口，单位 s；按键容易丢失时可适当增大。 |
| `--save_obs` | 是否保存 RGBD 和 TF observations 到 `outputs/<timestamp>/`。 |


`run_record_mobile_arm.py` 用于遥操作录制 mobile-arm demonstrations。

### prismatic gripper sanheyi full 录制

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5GripperJointPrismaticSanheyiFull-GripperJointPrismatic-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --dataset_file /tmp/mecanum730_xms5_gripper_joint_prismatic_sanheyi_full_dirty_open_seat_motion_wheel_arm_simple_sphere_raw_usd_light_demos.hdf5 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 5 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --prismatic_gripper_pos 0.0 \
  --prismatic_gripper_step 0.001 \
  --prismatic_gripper_min 0.0 \
  --prismatic_gripper_max 0.025 \
  --save_obs False
```

### prismatic gripper 录制

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5GripperJointPrismatic-GripperJointPrismatic-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --dataset_file /tmp/mecanum730_xms5_gripper_joint_prismatic_dirty_open_seat_motion_wheel_arm_simple_sphere_raw_usd_light_demos.hdf5 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 5 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --prismatic_gripper_pos 0.0 \
  --prismatic_gripper_step 0.001 \
  --prismatic_gripper_min 0.0 \
  --prismatic_gripper_max 0.025 \
  --save_obs False
```

### brush rolling 0512 录制

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5BrushRolling0512-BrushRolling0512-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --dataset_file /tmp/mecanum730_xms5_brush_rolling_0512_dirty_open_seat_motion_wheel_arm_simple_sphere_raw_usd_light_demos.hdf5 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --save_obs False
```

#### brush rolling sanheyi long

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5BrushRollingSanheyiLong-BrushRollingSanheyiLong-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --dataset_file /tmp/mecanum730_xms5_brush_rolling_sanheyi_long_dirty_open_seat_motion_wheel_arm_simple_sphere_raw_usd_light_demos.hdf5 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --save_obs False
```

#### brush rolling sanheyi short

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5BrushRollingSanheyiShort-BrushRollingSanheyiShort-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --dataset_file /tmp/mecanum730_xms5_brush_rolling_sanheyi_short_dirty_open_seat_motion_wheel_arm_simple_sphere_raw_usd_light_demos.hdf5 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --save_obs False
```


### sphere usd 独立配置录制

以下 task 使用同一套 scene、camera、action、observation 和 event 配置，只区分robot USD。


#### brush hard

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5BrushHard-BrushHard-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### brush rolling

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5BrushRolling-BrushRolling-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### brush simple

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5BrushSimple-BrushSimple-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### gripper

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Gripper-Gripper-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### gripper expanded

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5GripperExpanded-GripperExpanded-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### mop sponge

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5MopSponge-MopSponge-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### mop swab

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5MopSwab-MopSwab-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### pick

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Pick-Pick-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### plus

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Plus-Plus-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### suction in

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5SuctionIn-SuctionIn-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### suction in 0512

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5SuctionIn0512-SuctionIn0512-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### suction out

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5SuctionOut-SuctionOut-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### suction out 0512

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5SuctionOut0512-SuctionOut0512-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### suction press

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5SuctionPress-SuctionPress-Dirty-OpenSeat-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

### camera + 3D LiDAR 独立配置录制

以下 task 继承对应的 Recorder 独立配置，相机配置保持不变，仅额外启用 base_link 上的 3D LiDAR；原有非 LiDAR task 不受影响。

#### prismatic gripper + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5GripperJointPrismatic-GripperJointPrismatic-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --dataset_file /tmp/mecanum730_xms5_gripper_joint_prismatic_dirty_open_seat_lidar_demos.hdf5 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 5 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --prismatic_gripper_pos 0.0 \
  --prismatic_gripper_step 0.001 \
  --prismatic_gripper_min 0.0 \
  --prismatic_gripper_max 0.025 \
  --save_obs False
```

#### brush rolling 0512 + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5BrushRolling0512-BrushRolling0512-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --dataset_file /tmp/mecanum730_xms5_brush_rolling_0512_dirty_open_seat_lidar_demos.hdf5 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --save_obs False
```

#### brush hard + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5BrushHard-BrushHard-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### brush rolling + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5BrushRolling-BrushRolling-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### brush simple + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5BrushSimple-BrushSimple-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### gripper + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Gripper-Gripper-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### gripper expanded + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5GripperExpanded-GripperExpanded-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### mop sponge + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5MopSponge-MopSponge-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### mop swab + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5MopSwab-MopSwab-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### pick + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Pick-Pick-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### plus + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Plus-Plus-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### suction in + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5SuctionIn-SuctionIn-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### suction in 0512 + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5SuctionIn0512-SuctionIn0512-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### suction out + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5SuctionOut-SuctionOut-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### suction out 0512 + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5SuctionOut0512-SuctionOut0512-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```

#### suction press + lidar

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5SuctionPress-SuctionPress-Dirty-OpenSeat-Lidar-v0 \
  --num_demos 10 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 300 \
  --arm_debug_interval 1 \
  --camera_debug_interval 60 \
  --show_camera_pose_debug False \
  --keyboard_hold_sec 0.5 \
  --save_obs False
```
