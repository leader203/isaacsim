#!/usr/bin/env bash
# =============================================================================
# paper_grab.sh — 湿纸团抓取 SOP 一键脚本（抓纸团 → 夹持回位 → 投放垃圾桶 → 张开回位）
#
# 前置条件: 仿真服务已在运行（另开终端启动，命令见下方注释），本脚本只做 HTTP 调用。
# 用法:
#   ./paper_grab.sh
# 环境变量（可选）:
#   API_HOST   默认 127.0.0.1
#   API_PORT   默认 8000
#
# 对应 CLAUDE.md "Paper Cleanup SOP（湿纸团抓取→投放→收集）✅ 已验证通过"。
# 服务启动命令（--task 缺省 = 纸团场景，3 个湿纸团）:
#   cd <roboos_root> && conda activate isaaclab
#   OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_curobo_pose_plan_test.py \
#     --curobo_collision_activation_distance 0.02 \
#     --tool_api_server --tool_api_host 127.0.0.1 --tool_api_port 8000
#   # --task 缺省 = Isaac-...-GripperJointPrismatic-CuroboPosePlanTest-Dirty-OpenSeat-v0
#   # paper_cleanup_mode=true 自动检测 used_wet_paper_clump_01(1.75,-0.70,0.035)，
#   # 目标高度 max(z+0.03, 0.05) = 0.065。
# =============================================================================
set -u

ROBOOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
API_BASE="http://${API_HOST}:${API_PORT}"

log() { printf '[paper_grab] %s\n' "$*"; }
fail() { printf '[paper_grab] ERROR: %s\n' "$*"; exit 1; }

# 服务是否在监听（能拿到 HTTP 响应即视为就绪）
api_ready() { curl -sS -m 2 "${API_BASE}/tools/jobs/nonexistent" >/dev/null 2>&1; }

# 等一个工具任务到终态：轮询 /tools/jobs/{request_id}，直到 state 离开 queued/running。
# 终态结果 JSON 写入 LAST_RESP 并打印（含 tool_output / execution_output 两层）。
LAST_RESP=""
wait_tool_job() {
  local req="$1" resp state
  while true; do
    resp=$(curl -sS "${API_BASE}/tools/jobs/$req") || fail "GET /tools/jobs/$req 失败"
    state=$(printf '%s' "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["state"])')
    if [ "$state" != "queued" ] && [ "$state" != "running" ]; then break; fi
    sleep 2
  done
  printf '%s' "$resp" > "$ROBOOS_ROOT/.paper_last_resp.json"
  LAST_RESP="$ROBOOS_ROOT/.paper_last_resp.json"
  printf '%s' "$resp" | python3 -m json.tool
}

# 递归检查结果 JSON 中某字段是否为 true（对 tool_output / execution_output 嵌套都生效）
json_has_true() {
  python3 -c '
import json, sys
def has_true(o, k):
    if isinstance(o, dict):
        if o.get(k) is True:
            return True
        return any(has_true(v, k) for v in o.values())
    if isinstance(o, list):
        return any(has_true(v, k) for v in o)
    return False
data = json.load(open(sys.argv[1]))
sys.exit(0 if has_true(data, sys.argv[2]) else 1)
' "$LAST_RESP" "$1"
}

# 提交一个工具调用：POST /tools/invoke 拿 request_id → wait_tool_job 等到终态
invoke_tool() {
  local body="$1" req
  req=$(curl -sS -X POST "${API_BASE}/tools/invoke" -H 'Content-Type: application/json' -d "$body" \
        | python3 -c 'import sys,json; print(json.load(sys.stdin)["request_id"])') \
    || fail "POST /tools/invoke 失败（body=$body）"
  log "request_id=$req"
  wait_tool_job "$req"
}

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
if ! api_ready; then
  fail "未检测到仿真服务在 ${API_BASE} 监听。请先另开终端启动场景（见脚本头部注释）。"
fi

log "===== 湿纸团抓取 SOP 开始 ====="

# ── ① 抓纸团（paper_cleanup_mode 自动检测 used_wet_paper_clump_01 + Z 档 retry）─
log "① 抓纸团 (gripper_prismatic_execute_grab, paper_cleanup_mode=true)"
invoke_tool '{"tool_name":"gripper_prismatic_execute_grab","parameters":{"name":"paper_grab","paper_cleanup_mode":true}}'
if ! json_has_true executed; then
  fail "① 抓取未执行（executed=false，trajopt 全部失败？），终止 SOP"
fi
if ! json_has_true paper_grasped; then
  fail "① 未抓到纸团（paper_grasped=false），终止 SOP"
fi
log "① ✅ 抓到纸团"

# ── ② 夹持回位（跳过 lift，直接用反向 grab 轨迹）─────────────────────────────
log "② 夹持回位 (gripper_prismatic_move_arm_to_joints)"
invoke_tool '{"tool_name":"gripper_prismatic_move_arm_to_joints","parameters":{"name":"paper_return"}}'
if ! json_has_true executed; then
  fail "② 回位失败，终止 SOP"
fi
log "② ✅ 回位成功"

# ── ③ 投放垃圾桶（+ teleport 回收 + collection check）────────────────────────
log "③ 投放垃圾桶 (gripper_prismatic_take_out)"
invoke_tool '{"tool_name":"gripper_prismatic_take_out","parameters":{"name":"paper_drop"}}'
if ! json_has_true executed; then
  fail "③ 投放未执行，终止 SOP"
fi
if ! json_has_true paper_collected; then
  fail "③ 纸团未进入垃圾桶（paper_collected=false）——请检查 GUI/日志"
fi
log "③ ✅ 纸团已入桶"

# ── ④ 张开回位 ──────────────────────────────────────────────────────────────
log "④ 张开回位 (gripper_prismatic_move_arm_to_joints)"
invoke_tool '{"tool_name":"gripper_prismatic_move_arm_to_joints","parameters":{"name":"paper_return_open"}}'
if ! json_has_true executed; then
  fail "④ 回位失败"
fi
log "④ ✅ 回位成功"

log "===== 湿纸团抓取 SOP 完成 ✅ ====="
exit 0
