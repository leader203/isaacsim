#!/usr/bin/env bash
# =============================================================================
# collect_fast.sh — 加速版抓取数据采集服务启动脚本
#
# 默认（max 档，规划压到最低、执行压到关节速度上限）：
#   --curobo_num_ik_seeds 1             IK 种子 8 → 1
#   --curobo_num_trajopt_seeds 1        TrajOpt 种子 4 → 1
#   --steps_per_waypoint 3              每个轨迹点 6 个 env.step → 3（约 2× 执行速度）
#   --trajectory_velocity_feedforward   用相邻轨迹点算关节速度前馈
#
# 为什么不能再降到 1/2 步：仿真 soft_joint_vel_limit = 1.0 rad/s，cuRobo 的
# interpolation_dt = 0.025s（120Hz 下正好 3 个 env.step）。steps_per_waypoint=1 会让
# 要求速度达到规划的 ~3 倍，被 1 rad/s 截断，实测跟踪误差 0.4 rad / 14cm，抓取直接 abort。
# 3 步/点 = 按规划速度执行，是这套关节速度上限下的物理最快值。
#
# safe 档：只把种子加回来（4/2），执行速度相同，成功率更稳。
#
# 用法:
#   ./collect_fast.sh                     # max 档，前台启动服务（不录制）
#   ./collect_fast.sh --record            # 同时 --record_dataset outputs/cube_grasp_demos.hdf5
#   ./collect_fast.sh --profile safe      # 稳妥档
#   ./collect_fast.sh --profile max --record
#
# 服务起来后，另开终端按原流程采集：
#   conda activate isaaclab && python auto_collect.py --count success \
#       --convert-lerobot --lerobot-output outputs/cube_grasp_demos_groot
# =============================================================================
set -euo pipefail

ROBOOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REL="source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_curobo_pose_plan_test.py"

PROFILE="max"
RECORD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:?--profile 需要 max 或 safe}"; shift 2 ;;
    --record) RECORD=1; shift ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  max)
    FAST_ARGS=(
      --curobo_num_ik_seeds 1
      --curobo_num_trajopt_seeds 1
      --steps_per_waypoint 3
      --trajectory_velocity_feedforward
      --trajectory_velocity_feedforward_scale 1.0
    )
    ;;
  safe)
    FAST_ARGS=(
      --curobo_num_ik_seeds 4
      --curobo_num_trajopt_seeds 2
      --steps_per_waypoint 3
      --trajectory_velocity_feedforward
      --trajectory_velocity_feedforward_scale 1.0
    )
    ;;
  *) echo "--profile 只能是 max 或 safe，收到: $PROFILE" >&2; exit 2 ;;
esac

# 固定等待压缩（两档共用）。这些是固定 env.step 次数的上限/固定保持：
#   --final_hold_steps 80              默认 240
#   --post_target_descend_hold_steps 40  下降后固定保持 40 步（原 240）
#   --gripper_grasp_max_steps 80       夹爪反馈闭合上限（原 240，接触时因速度阈值跑满）
#   --gripper_hold_steps 60            grasp/lift/drop 保持上限（原 240）
FAST_ARGS+=(
  --final_hold_steps 80
  --post_target_descend_hold_steps 40
  --gripper_grasp_max_steps 80
  --gripper_hold_steps 60
)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate isaaclab
cd "$ROBOOS_ROOT"

CMD=(
  ./isaaclab.sh -p "$SCRIPT_REL"
  --curobo_collision_activation_distance 0.02
  "${FAST_ARGS[@]}"
  --tool_api_server --tool_api_host 127.0.0.1 --tool_api_port 8000
)
if [[ "$RECORD" == "1" ]]; then
  CMD+=(--record_dataset outputs/cube_grasp_demos.hdf5)
fi

echo "[collect_fast] profile=$PROFILE record=$RECORD"
echo "[collect_fast] ${CMD[*]}"
OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True exec "${CMD[@]}"
