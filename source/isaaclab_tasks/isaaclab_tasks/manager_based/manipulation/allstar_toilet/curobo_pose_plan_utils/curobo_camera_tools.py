from __future__ import annotations

import datetime
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

# cam_tool 相对 tool_link 的固定安装位姿。cam_follows_tool=True 时相机刚性跟随
# tool_link，这组值就是训练数据里恒定的相机-夹爪相对位姿。
CAM_T_IN_TOOL: tuple[float, float, float] = (0.12, 0.0, 0.30)
CAM_Q_IN_TOOL: tuple[float, float, float, float] = (0.704883, 0.056034, -0.056034, -0.704883)

_LINK5_CAMERA_STATE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
# 空闲保活状态：相机位姿未变化时叠加亚像素抖动，强制 render product 每帧重渲染
# （位姿不变 → 不重渲染 → 输出空白帧 → 视频闪烁）。
_LINK5_LAST_POSE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_LINK5_IDLE_TICK: dict[int, int] = {}


def reset_link5_camera_mount(env) -> None:
    _LINK5_CAMERA_STATE.pop(id(env), None)
    _LINK5_LAST_POSE.pop(id(env), None)
    _LINK5_IDLE_TICK.pop(id(env), None)
    if hasattr(env.cfg, "_cam_link5_mount_offset"):
        delattr(env.cfg, "_cam_link5_mount_offset")


def sync_link5_camera(env, robot) -> tuple[torch.Tensor, torch.Tensor] | None:
    cam = env.scene.sensors.get("cam_tool", None)
    if cam is None:
        return None
    try:
        tool_idx = robot.data.body_names.index("tool_link")
    except ValueError:
        return None

    tool_pos_w = robot.data.body_pos_w[:, tool_idx, :]
    tool_quat_w = robot.data.body_quat_w[:, tool_idx, :]
    # 首帧可能在 env reset 完成前读到 NaN 的 body pose（实测 root_z=[nan]）。此时不能
    # 缓存 link5->cam offset，否则相机偏移会永久变成 NaN、整路 cam_tool 空白。
    if not (torch.isfinite(tool_pos_w).all() and torch.isfinite(tool_quat_w).all()):
        return None
    # 安装点沿 tool_link +Z 前移到 z=0.30、横向 +X 12cm（原 0.08,0,0.01）：原位置到
    # 手指约 44cm，视线被 tool_base_link 本体端部法兰遮挡，夹爪闭合时手指可见仅 ~5%。
    # 前移后光轴朝 tool z≈1.05 看（相对 +Z 内倾约 9°），并翻转 180° 让画面正立。
    # 演示轨迹中方块 100% 完整在画面内（最小边距 ~24px），手指可见率闭合 ~50%、
    # 张开 ~68%，开合间距 ~25↔70px。
    cam_t_in_tool = torch.tensor([CAM_T_IN_TOOL], dtype=tool_quat_w.dtype, device=tool_quat_w.device)
    cam_q_in_tool = torch.tensor([CAM_Q_IN_TOOL], dtype=tool_quat_w.dtype, device=tool_quat_w.device)

    # cam_follows_tool=True：相机与夹爪一起跟随 joint6 旋转，相对姿态恒定，避免训练
    # 数据里夹爪在相机画面中转动。未设置的旧任务保持原来的 link5 固定安装行为。
    if getattr(env.cfg, "cam_follows_tool", False):
        cam_pos_w, cam_quat_w = combine_frame_transforms(
            tool_pos_w, tool_quat_w, cam_t_in_tool, cam_q_in_tool
        )
    else:
        try:
            link5_idx = robot.data.body_names.index("XMS5_R800_W4G3B4C_link5")
        except ValueError:
            return None
        link5_pos_w = robot.data.body_pos_w[:, link5_idx, :]
        link5_quat_w = robot.data.body_quat_w[:, link5_idx, :]
        if not (torch.isfinite(link5_pos_w).all() and torch.isfinite(link5_quat_w).all()):
            return None

        state_key = id(env)
        offset = _LINK5_CAMERA_STATE.get(state_key)
        if offset is None:
            init_cam_pos_w, init_cam_quat_w = combine_frame_transforms(
                tool_pos_w, tool_quat_w, cam_t_in_tool, cam_q_in_tool
            )
            cam_t_in_link5, cam_q_in_link5 = subtract_frame_transforms(
                link5_pos_w, link5_quat_w, init_cam_pos_w, init_cam_quat_w
            )
            _LINK5_CAMERA_STATE[state_key] = (cam_t_in_link5.detach().clone(), cam_q_in_link5.detach().clone())
            env.cfg._cam_link5_mount_offset = _LINK5_CAMERA_STATE[state_key]
            print(
                "[CameraMount] curobo_pose_plan_test fixed link5->cam offset "
                f"t={cam_t_in_link5[0].detach().cpu().tolist()} "
                f"q={cam_q_in_link5[0].detach().cpu().tolist()}",
                flush=True,
            )
        else:
            cam_t_in_link5, cam_q_in_link5 = offset

        cam_pos_w, cam_quat_w = combine_frame_transforms(
            link5_pos_w, link5_quat_w, cam_t_in_link5, cam_q_in_link5
        )
    # 空闲保活：机械臂静止时相机位姿不变 → render product 不重渲染 → 输出空白帧。
    # 位姿与上次写入完全相同时，叠加亚像素单调抖动（步进 0.15mm、包络 ±0.15mm，
    # 相邻值必不同），强制每帧重渲染；位姿有变化（机械臂在动）时不扰动。
    _key = id(env)
    _last = _LINK5_LAST_POSE.get(_key)
    if _last is not None:
        _pos_delta = float((cam_pos_w[0] - _last[0]).abs().max().item())
        _quat_delta = float((cam_quat_w[0] - _last[1]).abs().max().item())
        if _pos_delta < 1.0e-6 and _quat_delta < 1.0e-6:
            _tick = _LINK5_IDLE_TICK.get(_key, 0) + 1
            _LINK5_IDLE_TICK[_key] = _tick
            cam_pos_w = cam_pos_w.clone()
            cam_pos_w[0, 0] = cam_pos_w[0, 0] + 0.00005 * ((3 * _tick) % 7 - 3)
    cam.set_world_poses(cam_pos_w, cam_quat_w, convention="ros")
    _LINK5_LAST_POSE[_key] = (cam_pos_w[0].detach().clone(), cam_quat_w[0].detach().clone())
    return cam_pos_w[0].detach().clone(), cam_quat_w[0].detach().clone()


_BLANK_SKIP_LAST_LOG: float = 0.0
_CAM_WINDOW_CREATED: dict[str, bool] = {}
_DIAG_LAST_LOG: float = 0.0
_CAM_FIXED_FRAME_COUNT: int = 0


def show_camera_windows(env, *, headless: bool) -> None:
    if headless:
        return
    # 显示所有存在并启用的工具相机（cam_tool 机器人相机 + cam_fixed 固定相机等）
    for cam_name in ("cam_tool", "cam_fixed"):
        _show_one_camera_window(env, cam_name)


def _show_one_camera_window(env, cam_name: str) -> None:
    cam = env.scene.sensors.get(cam_name, None)
    if cam is None:
        return
    try:
        rgb_raw = cam.data.output["rgb"][0].detach().cpu().numpy()
        depth_raw = cam.data.output["depth"][0].detach().cpu().numpy()[..., 0]
    except Exception as exc:
        print(f"[CameraDisplay:{cam_name}] unavailable: {exc}", flush=True)
        return

    # 空白帧保护：只有 "RGB 无纹理 且 depth 无有效几何" 才算真空白，保留上一帧不更新。
    # 之前只拿 depth 判定会把"RGB 正常但 depth 异常"的相机（如固定相机）整个冻结。
    # 注意：窗口尚未创建时（首帧）即使空白也照常 imshow，确保每个相机都有窗口。
    rgb = rgb_raw[..., :3].astype(np.uint8)
    valid_depth = int(np.sum(np.isfinite(depth_raw) & (depth_raw > 0.0)))
    rgb_std = float(np.std(rgb.astype(np.float32)))
    # 诊断：cam_fixed 前 60 帧逐帧打印（不受限流），精确定位数据在哪一帧、以何种方式变空白
    if cam_name == "cam_fixed":
        global _CAM_FIXED_FRAME_COUNT
        _CAM_FIXED_FRAME_COUNT += 1
        _rbg_np = rgb_raw[..., :3].astype(np.float32)
        _dep_np = depth_raw.astype(np.float32)
        _finite = int(np.sum(np.isfinite(_dep_np)))
        _sem_extra = ""
        try:
            _sem_out = cam.data.output.get("semantic_segmentation", None)
            if _sem_out is not None:
                _sem_np = _sem_out[0].detach().cpu().numpy()
                _sem_extra = f" sem={_sem_np.shape} mean={_sem_np.reshape(-1).mean():.2f} std={_sem_np.reshape(-1).std():.2f}"
        except Exception as _e_sem:
            _sem_extra = f" sem=ERR:{_e_sem}"
        if _CAM_FIXED_FRAME_COUNT <= 60:
            print(
                f"[CamFrame:{cam_name}#{_CAM_FIXED_FRAME_COUNT}] rgb mean={_rbg_np.mean():.1f} "
                f"std={rgb_std:.2f} min={_rbg_np.min():.0f} max={_rbg_np.max():.0f} "
                f"| depth finite={_finite} valid={valid_depth} dmin={np.nanmin(_dep_np):.3f}{_sem_extra}",
                flush=True,
            )
    # 诊断：cam_fixed 数据统计（每 10s 一次），确认数据是否真的空白
    if cam_name == "cam_fixed":
        global _DIAG_LAST_LOG
        _now_d = time.time()
        if _now_d - _DIAG_LAST_LOG >= 10.0:
            _DIAG_LAST_LOG = _now_d
            _rbg_np = rgb_raw[..., :3].astype(np.float32)
            _dep_np = depth_raw.astype(np.float32)
            _finite = int(np.sum(np.isfinite(_dep_np)))
            print(
                f"[CamDiag:{cam_name}] rgb shape={rgb_raw.shape} dtype={rgb_raw.dtype} "
                f"mean={_rbg_np.mean():.1f} std={rgb_std:.1f} min={_rbg_np.min():.0f} max={_rbg_np.max():.0f} "
                f"| depth finite={_finite} sum_valid={valid_depth} "
                f"dmin={np.nanmin(_dep_np):.3f} dmax={np.nanmax(_dep_np):.3f}",
                flush=True,
            )
    if rgb_std < 3.0 and valid_depth < 100 and _CAM_WINDOW_CREATED.get(cam_name, False):
        global _BLANK_SKIP_LAST_LOG
        _now = time.time()
        if _now - _BLANK_SKIP_LAST_LOG >= 5.0:
            print(
                f"[CameraDisplay:{cam_name}] blank frame skipped (rgb_std={rgb_std:.1f} valid_depth={valid_depth}), keep last",
                flush=True,
            )
            _BLANK_SKIP_LAST_LOG = _now
        return

    depth_vis = np.nan_to_num(depth_raw, nan=0.0, posinf=0.0, neginf=0.0)
    depth_vis = np.clip(depth_vis / 3.0, 0.0, 1.0)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imshow(f"{cam_name} RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow(f"{cam_name} Depth", depth_vis)
    cv2.waitKey(1)
    _CAM_WINDOW_CREATED[cam_name] = True


def save_spot_capture_frame(
    env,
    robot,
    *,
    case_name: str,
    arm_base_body: str,
    target_q: torch.Tensor,
    task_root: Path,
    sync_link5_camera_fn,
    show_camera_windows_fn,
    body_pose_w_fn,
    pose_to_matrix_fn,
    arm_joint_ids_fn,
) -> dict:
    cam = env.scene.sensors.get("cam_tool", None)
    if cam is None:
        raise RuntimeError("cam_tool is not available; cannot execute spot capture.")

    cam_pose = sync_link5_camera_fn(env, robot)
    env.sim.render()
    show_camera_windows_fn(env)
    if cam_pose is None:
        raise RuntimeError("Failed to synchronize link5-mounted cam_tool.")
    cam_pos_w, cam_quat_w = cam_pose

    rgb = cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
    depth = cam.data.output["depth"][0].detach().cpu().numpy()[..., 0]

    arm_base_pos_w, arm_base_quat_w = body_pose_w_fn(robot, arm_base_body)
    cam_pos_base, cam_quat_base = subtract_frame_transforms(
        arm_base_pos_w.reshape(1, 3),
        arm_base_quat_w.reshape(1, 4),
        cam_pos_w.reshape(1, 3),
        cam_quat_w.reshape(1, 4),
    )
    cam_in_arm_base_tf = pose_to_matrix_fn(cam_pos_base[0], cam_quat_base[0]).detach().cpu().numpy()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in case_name)
    capture_dir = task_root / "outputs" / "curobo" / "spot_capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    capture_path = capture_dir / f"{safe_name}_{timestamp}.npz"
    actual_q = robot.data.joint_pos[0, arm_joint_ids_fn(robot)].detach().cpu().numpy()
    np.savez_compressed(
        capture_path,
        rgb=rgb,
        depth=depth,
        cam_in_arm_base_tf=cam_in_arm_base_tf,
        cam_pos_w=cam_pos_w.detach().cpu().numpy(),
        cam_quat_wxyz=cam_quat_w.detach().cpu().numpy(),
        arm_base_pos_w=arm_base_pos_w.detach().cpu().numpy(),
        arm_base_quat_wxyz=arm_base_quat_w.detach().cpu().numpy(),
        joint_pos=actual_q,
        target_joint_pos=target_q.detach().cpu().numpy(),
    )
    print(f"[SpotCapture] saved {capture_path}", flush=True)
    return {
        "capture_path": str(capture_path),
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "cam_in_arm_base_tf": cam_in_arm_base_tf.tolist(),
    }
