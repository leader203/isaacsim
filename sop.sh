#!/usr/bin/env bash
# =============================================================================
# sop.sh — Grasp Test Cube SOP 一键执行脚本（抓 grasp_test_cube → 原地抬升 → 投放垃圾桶 → 回位）
#
# 用法:
#   ./sop.sh                     # FastAPI 仿真服务已在运行（另开终端启动过）
#   ./sop.sh --start-server      # 服务未运行：自动后台启动仿真并等待就绪
#   ./sop.sh --server            # 同 --start-server
#
# 环境变量（可选）:
#   API_HOST   默认 127.0.0.1
#   API_PORT   默认 8000
#
# 对应 CLAUDE.md / grab_drop_sop.md 中的抓取 SOP（最早版本，场景 = 3 个 grasp_test_cube）。
# 注：此 SOP 不含"挪桶/桶复位"，桶固定在初始位置，drop 点写死。
# 注：gripper_prismatic_execute_grab 的 paper_cleanup_mode=true 已改为默认抓取
#     /World/envs/env_0/grasp_test_cube_1（见 run_curobo_pose_plan_test.py），
#     返回字段仍为 paper_grasped / paper_collected，语义现在是 cube 是否抓到/入桶。
# =============================================================================
set -u

ROBOOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
API_BASE="http://${API_HOST}:${API_PORT}"
SERVER_LOG="${ROBOOS_ROOT}/sop_server.log"

# 仿真服务启动命令（与 CLAUDE.md "启动服务 + 抓取 SOP 基线配置" 一致）
SCRIPT_REL="source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_curobo_pose_plan_test.py"
SERVER_CMD=(
  ./isaaclab.sh -p "$SCRIPT_REL"
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Common-CuroboPosePlanTest-Dirty-OpenSeat-v0
  --initial_robot_id gripper_prismatic
  --curobo_collision_activation_distance 0.02
  --curobo_num_ik_seeds 16
  --curobo_num_trajopt_seeds 8
  --curobo_box_collision_front_min_x_m -0.5
  --tool_api_server --tool_api_host "$API_HOST" --tool_api_port "$API_PORT"
)

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
log() { printf '[sop] %s\n' "$*"; }
fail() { printf '[sop] ERROR: %s\n' "$*"; exit 1; }

# 服务是否在监听（能拿到 HTTP 响应即视为就绪）
api_ready() { curl -sS -m 2 "${API_BASE}/tools/jobs/nonexistent" >/dev/null 2>&1; }

# 后台启动仿真服务并等待 /tools API 就绪
start_server() {
  log "后台启动仿真服务..."
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
    || fail "找不到 conda，请先安装/激活 conda"
  conda activate isaaclab || fail "conda 环境 isaaclab 不存在"
  cd "$ROBOOS_ROOT" || fail "无法进入仓库目录 $ROBOOS_ROOT"
  nohup "${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
  log "服务已后台启动 PID=$! （首次编译 shader 约 3-5 分钟，GUI 显示'无响应'正常）"
  log "日志文件: $SERVER_LOG  用 Ctrl+C 中断本脚本不影响服务"

  log "等待服务就绪（最多 15 分钟）..."
  local waited=0
  while [ "$waited" -lt 900 ]; do
    if api_ready; then log "服务已就绪: ${API_BASE}"; return 0; fi
    sleep 1
    waited=$((waited + 1))
    if [ $((waited % 30)) -eq 0 ]; then
      log "已等待 ${waited}s ... 可另开终端 tail -f $SERVER_LOG 查看进度"
    fi
  done
  fail "等待服务就绪超时，请查看 $SERVER_LOG"
}

# 调用一个工具：POST 拿 request_id → 轮询直到终态 → 存结果 JSON 并打印
LAST_RESP=""
invoke_tool() {
  local body="$1"
  local req
  req=$(curl -sS -X POST "${API_BASE}/tools/invoke" -H 'Content-Type: application/json' -d "$body" \
        | python3 -c 'import sys,json; print(json.load(sys.stdin)["request_id"])') \
    || fail "POST /tools/invoke 失败（body=$body）"
  log "request_id=$req"
  local resp state
  while true; do
    resp=$(curl -sS "${API_BASE}/tools/jobs/$req") || fail "GET /tools/jobs/$req 失败"
    state=$(printf '%s' "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["state"])')
    if [ "$state" != "queued" ] && [ "$state" != "running" ]; then break; fi
    sleep 2
  done
  printf '%s' "$resp" > "$ROBOOS_ROOT/.sop_last_resp.json"
  LAST_RESP="$ROBOOS_ROOT/.sop_last_resp.json"
  printf '%s' "$resp" | python3 -m json.tool
}

# 检查结果 JSON 中某字段是否为 true（对嵌套结构也生效）
json_has_true() { grep -q "\"$1\"[[:space:]]*:[[:space:]]*true" "$LAST_RESP"; }

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
MODE="${1:-}"

if [ "$MODE" = "--start-server" ] || [ "$MODE" = "--server" ]; then
  if api_ready; then
    log "检测到服务已在运行，直接开始 SOP"
  else
    start_server
  fi
elif [ -n "$MODE" ] && [ "$MODE" != "--help" ] && [ "$MODE" != "-h" ]; then
  fail "未知参数: $MODE （支持: 无参数 | --start-server | --server）"
fi

log "===== Grasp Test Cube SOP 开始 ====="

# ── ① 抓 grasp_test_cube_1 ───────────────────────────────────────────────────
log "① 抓 grasp_test_cube_1 (gripper_prismatic_execute_grab, paper_cleanup_mode=true)"
invoke_tool '{"tool_name":"gripper_prismatic_execute_grab","parameters":{"name":"grab_cube","paper_cleanup_mode":true}}'
if ! json_has_true executed; then
  fail "① 抓取失败，终止 SOP（抓取未执行）"
fi
if ! json_has_true paper_grasped; then
  fail "① 未抓到 cube（paper_grasped=false），终止 SOP"
fi
log "① ✅ 抓到 cube"

# ── ② 原地抬升 ──────────────────────────────────────────────────────────────
# 抓完直接投放时，规划起点是贴地的抓取姿态（很"憋屈"），IK 容易解不出来 / 翻 joint4。
# local_lift 保持 XY 和姿态不变、world +Z 抬 0.18m，把机械臂从低位拔出来再投放。
# （之前那步"回起始位姿"删掉后就需要它来替代——但只抬升，不改姿态，动作不突兀。）
log "② 原地抬升 (gripper_prismatic_local_lift)"
invoke_tool '{"tool_name":"gripper_prismatic_local_lift","parameters":{"name":"pre_drop_lift","lift_height_m":0.30}}'
if ! json_has_true executed; then
  fail "② 抬升失败，终止 SOP"
fi
log "② ✅ 抬升完成"

# ── ③ 投放垃圾桶 ────────────────────────────────────────────────────────────
log "③ 投放垃圾桶 (gripper_prismatic_take_out)"
invoke_tool '{"tool_name":"gripper_prismatic_take_out","parameters":{"name":"cube_drop"}}'
if ! json_has_true executed; then
  fail "③ 投放未执行，终止 SOP"
fi
if ! json_has_true paper_collected; then
  fail "③ cube 未进入垃圾桶（paper_collected=false）——请检查 GUI/日志"
fi
log "③ ✅ cube 已入桶"

# 投放成功即结束：不再做"张开回位"。
# 数据集也在 take_out 执行完就收尾（见 run_curobo_pose_plan_test.py 的 _with_demo_recording），
# 所以投放之后的回位/空转帧不会进 demo。

log "===== Grasp Test Cube SOP 完成 ✅ ====="
log "本 SOP 不含挪桶/桶复位；完整版见 garbage_pickup_sop 方案。"
exit 0
