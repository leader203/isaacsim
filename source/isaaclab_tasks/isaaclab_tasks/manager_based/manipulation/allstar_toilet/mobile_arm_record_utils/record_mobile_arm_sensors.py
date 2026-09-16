"""Sensor visualization and recording helpers for mobile-arm recording."""

import datetime
import importlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch


_ALLSTAR_TOILET_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUTS_ROOT = str(_ALLSTAR_TOILET_ROOT / "outputs")


class ObservationChunkRecorder:
    def __init__(self, outputs_root: str = _DEFAULT_OUTPUTS_ROOT, chunk_size: int = 0):
        self.outputs_root = outputs_root
        self.chunk_size = chunk_size
        self.obs_buf = []
        self.chunk_idx = 0
        self.session_dir = None

    def maybe_record(self, *, save_obs: bool, sensor_io_enabled: bool, policy_obs: dict, cam) -> None:
        if not (save_obs and sensor_io_enabled and policy_obs and cam is not None):
            return
        if self.session_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(self.outputs_root, timestamp)
            os.makedirs(self.session_dir, exist_ok=True)
            print(f"\n[观测保存] 会话目录: {self.session_dir}")
        self.obs_buf.append({
            "rgb": policy_obs["cam_tool_rgb"][0].cpu().numpy(),
            "depth": policy_obs["cam_tool_depth"][0].cpu().numpy(),
            "cam_in_tool_tf": policy_obs["cam_in_tool_tf"][0].cpu().numpy(),
            "cam_in_base_tf": policy_obs["cam_in_base_tf"][0].cpu().numpy(),
        })
        if len(self.obs_buf) >= self.chunk_size:
            self.flush(save_obs=True)

    def flush(self, *, save_obs: bool) -> None:
        if not (save_obs and self.obs_buf and self.session_dir):
            return
        output_path = os.path.join(self.session_dir, f"chunk_{self.chunk_idx:04d}_obs.npz")
        np.savez_compressed(
            output_path,
            rgb=np.stack([frame["rgb"] for frame in self.obs_buf]),
            depth=np.stack([frame["depth"] for frame in self.obs_buf]),
            cam_in_tool_tf=np.stack([frame["cam_in_tool_tf"] for frame in self.obs_buf]),
            cam_in_base_tf=np.stack([frame["cam_in_base_tf"] for frame in self.obs_buf]),
        )
        print(f"\n[观测保存] {output_path}  ({len(self.obs_buf)} 帧)")
        self.obs_buf.clear()
        self.chunk_idx += 1


def acquire_debug_draw_interface(*, is_headless: bool, quiet_runtime_debug: bool):
    if is_headless:
        if not quiet_runtime_debug:
            print("[DebugDraw] headless 模式下跳过 viewport debug draw", flush=True)
        return None, False
    for module_name in ("omni.isaac.debug_draw", "isaacsim.util.debug_draw"):
        try:
            module = importlib.import_module(module_name)
            draw = module._debug_draw.acquire_debug_draw_interface()
            if not quiet_runtime_debug:
                print(f"[DebugDraw] 使用模块: {module_name}")
            return draw, True
        except Exception as exc:
            if not quiet_runtime_debug:
                print(f"[DebugDraw] {module_name} 不可用: {exc}")
    return None, False


def _quat_wxyz_axes(quat_wxyz) -> tuple[list[float], list[float], list[float]]:
    w, x, y, z = (float(quat_wxyz[i]) for i in range(4))
    x_axis = [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ]
    y_axis = [
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    ]
    z_axis = [
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
    return x_axis, y_axis, z_axis


def _draw_lidar_sensor_debug(draw, lidar_sensor) -> None:
    base_pos = lidar_sensor.data.pos_w[0].cpu().tolist()
    x_axis, y_axis, z_axis = _quat_wxyz_axes(lidar_sensor.data.quat_w[0].cpu())

    offset_x, offset_y, offset_z = lidar_sensor.cfg.offset.pos
    origin = [
        base_pos[i] + offset_x * x_axis[i] + offset_y * y_axis[i] + offset_z * z_axis[i]
        for i in range(3)
    ]

    half_extent = 0.025

    def corner(sign_x, sign_y, sign_z):
        return [
            origin[i]
            + sign_x * half_extent * x_axis[i]
            + sign_y * half_extent * y_axis[i]
            + sign_z * half_extent * z_axis[i]
            for i in range(3)
        ]

    corners = {
        (sx, sy, sz): corner(sx, sy, sz)
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    }
    edges = [
        ((-1, -1, -1), (+1, -1, -1)), ((-1, -1, +1), (+1, -1, +1)),
        ((-1, +1, -1), (+1, +1, -1)), ((-1, +1, +1), (+1, +1, +1)),
        ((-1, -1, -1), (-1, +1, -1)), ((-1, -1, +1), (-1, +1, +1)),
        ((+1, -1, -1), (+1, +1, -1)), ((+1, -1, +1), (+1, +1, +1)),
        ((-1, -1, -1), (-1, -1, +1)), ((-1, +1, -1), (-1, +1, +1)),
        ((+1, -1, -1), (+1, -1, +1)), ((+1, +1, -1), (+1, +1, +1)),
    ]
    draw.draw_lines(
        [corners[start] for start, _ in edges],
        [corners[end] for _, end in edges],
        [(1.0, 0.0, 0.0, 1.0)] * len(edges),
        [2.0] * len(edges),
    )

    tip = [origin[i] + x_axis[i] * 0.20 for i in range(3)]
    draw.draw_lines([origin], [tip], [(1.0, 0.5, 0.0, 1.0)], [3.0])


def _draw_camera_pose_debug(draw, cam_pos_w, cam_quat_w) -> None:
    cx, cy, cz = (float(cam_pos_w[0, i].item()) for i in range(3))
    x_axis, y_axis, z_axis = _quat_wxyz_axes(cam_quat_w[0])
    origin = [cx, cy, cz]

    axis_len = 0.18
    draw.draw_lines(
        [origin, origin, origin],
        [
            [cx + x_axis[0] * axis_len, cy + x_axis[1] * axis_len, cz + x_axis[2] * axis_len],
            [cx + y_axis[0] * axis_len, cy + y_axis[1] * axis_len, cz + y_axis[2] * axis_len],
            [cx + z_axis[0] * axis_len, cy + z_axis[1] * axis_len, cz + z_axis[2] * axis_len],
        ],
        [(1.0, 0.0, 0.0, 1.0), (0.0, 0.75, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0)],
        [5.0, 5.0, 6.0],
    )

    cross_half = 0.035
    draw.draw_lines(
        [[cx - cross_half, cy, cz], [cx, cy - cross_half, cz], [cx, cy, cz - cross_half]],
        [[cx + cross_half, cy, cz], [cx, cy + cross_half, cz], [cx, cy, cz + cross_half]],
        [(1.0, 1.0, 1.0, 1.0)] * 3,
        [5.0] * 3,
    )

    frustum_len = 0.24
    frustum_half_w = 0.085
    frustum_half_h = 0.055
    frustum_center = [
        cx + z_axis[0] * frustum_len,
        cy + z_axis[1] * frustum_len,
        cz + z_axis[2] * frustum_len,
    ]
    frustum_corners = []
    for sx, sy in ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)):
        frustum_corners.append([
            frustum_center[i] + sx * x_axis[i] * frustum_half_w + sy * y_axis[i] * frustum_half_h
            for i in range(3)
        ])
    draw.draw_lines(
        [origin] * 4 + frustum_corners,
        frustum_corners + frustum_corners[1:] + frustum_corners[:1],
        [(0.0, 1.0, 0.0, 1.0)] * 8,
        [3.0] * 8,
    )

    marker_half = 0.045

    def marker_corner(sx, sy, sz):
        return [
            origin[i]
            + sx * x_axis[i] * marker_half
            + sy * y_axis[i] * marker_half
            + sz * z_axis[i] * marker_half
            for i in range(3)
        ]

    marker_corners = {
        (sx, sy, sz): marker_corner(sx, sy, sz)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    }
    marker_edges = [
        ((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0)),
        ((-1.0, 1.0, -1.0), (1.0, 1.0, -1.0)),
        ((-1.0, -1.0, 1.0), (1.0, -1.0, 1.0)),
        ((-1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        ((-1.0, -1.0, -1.0), (-1.0, 1.0, -1.0)),
        ((1.0, -1.0, -1.0), (1.0, 1.0, -1.0)),
        ((-1.0, -1.0, 1.0), (-1.0, 1.0, 1.0)),
        ((1.0, -1.0, 1.0), (1.0, 1.0, 1.0)),
        ((-1.0, -1.0, -1.0), (-1.0, -1.0, 1.0)),
        ((1.0, -1.0, -1.0), (1.0, -1.0, 1.0)),
        ((-1.0, 1.0, -1.0), (-1.0, 1.0, 1.0)),
        ((1.0, 1.0, -1.0), (1.0, 1.0, 1.0)),
    ]
    draw.draw_lines(
        [marker_corners[start] for start, _ in marker_edges],
        [marker_corners[end] for _, end in marker_edges],
        [(1.0, 0.9, 0.0, 1.0)] * len(marker_edges),
        [4.0] * len(marker_edges),
    )


def draw_viewport_sensor_debug(
    draw,
    *,
    sensor_io_enabled: bool,
    lidar,
    show_camera_pose_debug: bool,
    cam,
    cam_pos_w,
    cam_quat_w,
) -> None:
    if draw is None:
        return
    draw.clear_lines()
    if sensor_io_enabled and lidar is not None:
        _draw_lidar_sensor_debug(draw, lidar)
    if show_camera_pose_debug and sensor_io_enabled and cam is not None and cam_pos_w is not None:
        _draw_camera_pose_debug(draw, cam_pos_w, cam_quat_w)


def _camera_forward_from_opengl_quat(quat_wxyz) -> tuple[float, float, float]:
    w, x, y, z = (float(quat_wxyz[i]) for i in range(4))
    return (
        -(2 * (x * z + w * y)),
        -(2 * (y * z - w * x)),
        -(1 - 2 * (x * x + y * y)),
    )


class CameraDisplayRuntime:
    def __init__(self, *, is_headless: bool):
        self.is_headless = is_headless
        self.camera_windows_created = False

    def update(
        self,
        *,
        sensor_io_enabled: bool,
        cam,
        lidar,
        obs: dict,
        step: int,
        env_step_count: int,
        camera_debug_interval: int,
        robot,
        base_pos_w,
        tool_pos_w,
        tool_tip_pos_w,
        base_idx: int,
        tool_idx: int,
        tool_tip_idx: int,
    ) -> None:
        if sensor_io_enabled and cam is not None:
            if not self.is_headless and not self.camera_windows_created:
                cv2.namedWindow("cam_tool RGB", cv2.WINDOW_NORMAL)
                cv2.namedWindow("cam_tool Depth", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("cam_tool RGB", 320, 320)
                cv2.resizeWindow("cam_tool Depth", 320, 320)
                cv2.moveWindow("cam_tool RGB", 40, 80)
                cv2.moveWindow("cam_tool Depth", 380, 80)
                self.camera_windows_created = True

            camera_source = "cam.data"
            policy_obs = obs.get("policy", {})
            if "cam_tool_rgb" in policy_obs and "cam_tool_depth" in policy_obs:
                rgb_raw = policy_obs["cam_tool_rgb"][0].detach().cpu().numpy()
                depth_raw = policy_obs["cam_tool_depth"][0].detach().cpu().numpy()[..., 0]
                camera_source = "obs.policy"
            else:
                rgb_raw = cam.data.output["rgb"][0].cpu().numpy()
                depth_raw = cam.data.output["depth"][0].cpu().numpy()[..., 0]

            rgb_min = float(np.nanmin(rgb_raw))
            rgb_max = float(np.nanmax(rgb_raw))
            rgb_mean = float(np.nanmean(rgb_raw[..., :3]))
            rgb_std = float(np.nanstd(rgb_raw[..., :3]))
            depth_finite = depth_raw[np.isfinite(depth_raw)]
            depth_min = float(np.nanmin(depth_finite)) if depth_finite.size else float("nan")
            depth_max = float(np.nanmax(depth_finite)) if depth_finite.size else float("nan")
            depth_mean = float(depth_finite.mean()) if depth_finite.size else float("nan")

            if step == 1:
                self._print_first_camera_diag(
                    rgb_raw=rgb_raw,
                    depth_raw=depth_raw,
                    rgb_min=rgb_min,
                    rgb_max=rgb_max,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth_mean=depth_mean,
                    cam=cam,
                    robot=robot,
                    base_pos_w=base_pos_w,
                    tool_pos_w=tool_pos_w,
                    tool_tip_pos_w=tool_tip_pos_w,
                    base_idx=base_idx,
                    tool_idx=tool_idx,
                    tool_tip_idx=tool_tip_idx,
                )
            elif camera_debug_interval > 0 and env_step_count % camera_debug_interval == 0:
                cam_pos = cam.data.pos_w[0].cpu().tolist()
                cam_quat = cam.data.quat_w_world[0].cpu()
                cam_forward = _camera_forward_from_opengl_quat(cam_quat)
                print(
                    "\n[CameraDebug] "
                    f"step={env_step_count} "
                    f"source={camera_source} "
                    f"rgb_min={rgb_min:.1f} rgb_max={rgb_max:.1f} "
                    f"rgb_mean={rgb_mean:.1f} rgb_std={rgb_std:.2f} "
                    f"depth_min={depth_min:.3f} depth_max={depth_max:.3f} depth_mean={depth_mean:.3f} "
                    f"cam_pos={[round(v, 4) for v in cam_pos]} "
                    f"cam_forward={[round(v, 4) for v in cam_forward]}",
                    flush=True,
                )
            elif step % 120 == 0 and (
                rgb_min == rgb_max
                or not np.isfinite(rgb_min + rgb_max + rgb_mean + rgb_std)
                or (rgb_mean > 248.0 and rgb_std < 3.0)
            ):
                cam_pos = cam.data.pos_w[0].cpu().tolist()
                cam_quat = cam.data.quat_w_world[0].cpu().tolist()
                print(
                    "\n[CameraWarn] abnormal rgb buffer "
                    f"step={step} source={camera_source} min={rgb_min:.1f} max={rgb_max:.1f} "
                    f"mean={rgb_mean:.1f} std={rgb_std:.2f} "
                    f"cam_pos={[round(v, 4) for v in cam_pos]} "
                    f"cam_quat_opengl={[round(v, 4) for v in cam_quat]}",
                    flush=True,
                )

            if not self.is_headless:
                rgb = rgb_raw[..., :3].astype(np.uint8)
                depth_vis = np.clip(depth_raw / 3.0, 0.0, 1.0)
                depth_vis = (depth_vis * 255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow("cam_tool RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imshow("cam_tool Depth", depth_vis)

        if sensor_io_enabled and not self.is_headless and lidar is not None:
            cv2.imshow("LiDAR (top view)", _make_lidar_vis(lidar))

        if sensor_io_enabled and not self.is_headless and (cam is not None or lidar is not None):
            cv2.waitKey(1)

    @staticmethod
    def _print_first_camera_diag(
        *,
        rgb_raw,
        depth_raw,
        rgb_min: float,
        rgb_max: float,
        depth_min: float,
        depth_max: float,
        depth_mean: float,
        cam,
        robot,
        base_pos_w,
        tool_pos_w,
        tool_tip_pos_w,
        base_idx: int,
        tool_idx: int,
        tool_tip_idx: int,
    ) -> None:
        print(f"\n[Camera] rgb shape={rgb_raw.shape} dtype={rgb_raw.dtype} min={rgb_min:.1f} max={rgb_max:.1f}")
        print(f"[Camera] depth shape={depth_raw.shape} min={depth_min:.3f} max={depth_max:.3f} mean={depth_mean:.3f}")
        cam_pos = cam.data.pos_w[0].cpu().tolist()
        base_pos = base_pos_w[0].cpu().tolist()
        tool_pos = tool_pos_w[0].cpu().tolist()
        tool_tip_pos = tool_tip_pos_w[0].cpu().tolist()
        print(f"[Diag] body_names = {robot.data.body_names}")
        print(f"[Diag] _base_idx = {base_idx}  ->  '{robot.data.body_names[base_idx]}'")
        print(f"[Diag] _tool_idx  = {tool_idx}  ->  '{robot.data.body_names[tool_idx]}'")
        print(f"[Diag] _tool_tip_idx = {tool_tip_idx}  ->  '{robot.data.body_names[tool_tip_idx]}'")
        print(f"[Diag] base_link world pos : [{base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f}]")
        print(f"[Diag] tool_link world pos : [{tool_pos[0]:.4f}, {tool_pos[1]:.4f}, {tool_pos[2]:.4f}]")
        print(f"[Diag] tool_tip world pos  : [{tool_tip_pos[0]:.4f}, {tool_tip_pos[1]:.4f}, {tool_tip_pos[2]:.4f}]")
        print(f"[Diag] cam.data.pos_w      : [{cam_pos[0]:.4f}, {cam_pos[1]:.4f}, {cam_pos[2]:.4f}]")
        cam_quat = cam.data.quat_w_world[0].cpu()
        cam_forward = _camera_forward_from_opengl_quat(cam_quat)
        print(
            f"[Diag] cam forward (world) : [{cam_forward[0]:.3f}, {cam_forward[1]:.3f}, {cam_forward[2]:.3f}]"
            f"  (x=front/back, y=left/right, z=up/down)"
        )
        print(f"[Diag] cam quat (opengl)   : {[round(v, 4) for v in cam_quat.tolist()]}")


def _make_lidar_vis(lidar_sensor, canvas_size: int = 450, max_range: float = 10.0) -> np.ndarray:
    """单线激光雷达俯视可视化（传感器局部坐标系，正前方朝上）。

    颜色编码：近处=绿 → 远处=红；机器人位于画面中心，箭头指向前方。
    返回 BGR uint8 图像，shape=(canvas_size, canvas_size, 3)。
    """
    hits_w = lidar_sensor.data.ray_hits_w[0].cpu().float()   # [N_rays, 3]
    pos_w  = lidar_sensor.data.pos_w[0].cpu().float()         # [3] base_link 位置
    quat_w = lidar_sensor.data.quat_w[0].cpu().float()        # [4] wxyz

    # ── 修正：data.pos_w 不含 cfg.offset，需手动叠加 ──────────────────────
    w_, x_, y_, z_ = float(quat_w[0]), float(quat_w[1]), float(quat_w[2]), float(quat_w[3])
    R = np.array([
        [1 - 2*(y_*y_ + z_*z_),  2*(x_*y_ - w_*z_),   2*(x_*z_ + w_*y_)],
        [2*(x_*y_ + w_*z_),      1 - 2*(x_*x_ + z_*z_), 2*(y_*z_ - w_*x_)],
        [2*(x_*z_ - w_*y_),      2*(y_*z_ + w_*x_),    1 - 2*(x_*x_ + y_*y_)],
    ], dtype=np.float32)
    off = np.array(list(lidar_sensor.cfg.offset.pos), dtype=np.float32)
    pos_w = pos_w + torch.from_numpy(R @ off)

    # ── 世界坐标系下的相对向量，旋转到传感器局部坐标系 ─────────────────────
    delta_w = (hits_w - pos_w.unsqueeze(0)).numpy()           # [N_rays, 3]
    ranges  = np.linalg.norm(delta_w, axis=-1)                # [N_rays]
    delta_local = (R.T @ delta_w.T).T                        # [N_rays, 3]
    dx = delta_local[:, 0]   # 前向分量（+X = 前方）
    dy = delta_local[:, 1]   # 侧向分量（+Y = 左侧，ROS 惯例）

    # ── 画面初始化 ─────────────────────────────────────────────────────────
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    cx = cy = canvas_size // 2
    scale = (canvas_size * 0.44) / max_range    # px/m

    # 十字参考线
    cv2.line(img, (cx, 0), (cx, canvas_size), (25, 25, 25), 1)
    cv2.line(img, (0, cy), (canvas_size, cy), (25, 25, 25), 1)

    # 距离环（2.5 / 5 / 7.5 / 10 m）
    for r_m in (2.5, 5.0, 7.5, 10.0):
        r_px = int(r_m * scale)
        cv2.circle(img, (cx, cy), r_px, (45, 45, 45), 1)
        cv2.putText(img, f"{r_m:.0f}m", (cx + r_px + 3, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70, 70, 70), 1)

    # 方向标注
    cv2.putText(img, "F", (cx - 5, 14),                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(img, "L", (6, cy + 5),                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(img, "R", (canvas_size - 16, cy + 5),             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # ── 激光点云（按距离着色） ─────────────────────────────────────────────
    for i in range(len(dx)):
        r = ranges[i]
        if not np.isfinite(r) or not np.isfinite(dx[i]) or not np.isfinite(dy[i]):
            continue
        if r >= max_range * 0.99:
            continue              # 未命中（抵达 max_distance），跳过
        ratio = float(r) / max_range
        # 近=绿(0,255,0) → 中=黄(0,255,255) → 远=红(0,0,255)   BGR
        green = int(255 * (1.0 - min(ratio * 2, 1.0)))
        red   = int(255 * min(ratio * 2, 1.0))
        bgr   = (0, green, red)
        # 屏幕映射：前(+dx)→上，左(+dy)→左
        px = int(cx - dy[i] * scale)
        py = int(cy - dx[i] * scale)
        if 0 <= px < canvas_size and 0 <= py < canvas_size:
            cv2.circle(img, (px, py), 2, bgr, -1)

    # ── 机器人标记 ────────────────────────────────────────────────────────
    robot_color = (0, 200, 255)
    cv2.circle(img, (cx, cy), 6, robot_color, -1)
    cv2.arrowedLine(img, (cx, cy), (cx, cy - 22), robot_color, 2, tipLength=0.35)

    return img
