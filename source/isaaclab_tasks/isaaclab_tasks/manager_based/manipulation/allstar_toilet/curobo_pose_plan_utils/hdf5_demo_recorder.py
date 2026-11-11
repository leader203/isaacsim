"""HDF5 demonstration recorder for the cuRobo cube-grasp scene.

为什么不用 Isaac Lab 的 RecorderManager：
cuRobo 场景在 tool server 模式下把 env.step 替换成了自定义 _step()
（绕过了 ManagerBasedEnv.step()），RecorderManager 不会逐帧驱动。
所以这里写一个独立记录器，挂在 _step() 末尾 + 工具 executor 包装上逐帧采样。

一条 demo = 完整 SOP：grab → move_arm_to_joints → take_out → move_arm_to_joints。
每次抓取从 grab 开始到下一次 grab（或进程退出）结束为一条 demo_X，带 success 标签。

设计约束：
- 模块可脱离 Isaac Sim 独立导入（torch/isaaclab 延迟到方法内 import），便于离线验证。
- 所有读写都 try/except 包裹，录制绝不让仿真崩溃。
"""

import os

import h5py
import numpy as np


def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz 四元数 → 3x3 旋转矩阵（行主序，列 = 各轴在父系下的方向）。"""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _quat_conjugate(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.array([quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]], dtype=np.float32)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = (float(v) for v in q1)
    w2, x2, y2, z2 = (float(v) for v in q2)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _subtract_frames(
    a_pos: np.ndarray, a_quat: np.ndarray, b_pos: np.ndarray, b_quat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """b 位姿在 a 系下的表达：(p_b_in_a, q_b_in_a)。"""
    r_a = _quat_to_matrix(a_quat)
    p_b_in_a = r_a.T @ (np.asarray(b_pos, dtype=np.float32) - np.asarray(a_pos, dtype=np.float32))
    q_b_in_a = _quat_mul(_quat_conjugate(a_quat), b_quat)
    return p_b_in_a.astype(np.float32), q_b_in_a


def _compose_frames(
    a_pos: np.ndarray, a_quat: np.ndarray, b_pos_in_a: np.ndarray, b_quat_in_a: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """b 位姿（在 a 系下给出）→ a 系中的世界位姿：(a * b)。"""
    r_a = _quat_to_matrix(a_quat)
    p_w = np.asarray(a_pos, dtype=np.float32) + r_a @ np.asarray(b_pos_in_a, dtype=np.float32)
    q_w = _quat_mul(a_quat, b_quat_in_a)
    return p_w.astype(np.float32), q_w.astype(np.float32)


def _tf4x16(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """位姿 → 4x4 齐次矩阵，行主序展开为 16。T[:3,:3]=R, T[:3,3]=t。"""
    t = np.eye(4, dtype=np.float32)
    t[:3, :3] = _quat_to_matrix(quat_wxyz)
    t[:3, 3] = np.asarray(pos, dtype=np.float32).reshape(3)
    return t.reshape(16)


def _extract_target_pos(parameters: dict | None) -> np.ndarray | None:
    """从 grab 工具参数里提取世界目标位置（若有）。"""
    if not isinstance(parameters, dict):
        return None
    pos = parameters.get("target_pos")
    if isinstance(pos, (list, tuple)) and len(pos) == 3:
        return np.asarray(pos, dtype=np.float32)
    return None


# 阶段标签常量
PHASE_GRAB = "grab"
PHASE_RETURN_AFTER_GRAB = "return_after_grab"
PHASE_TAKE_OUT = "take_out"
PHASE_RETURN_AFTER_DROP = "return_after_drop"


class HDF5DemoRecorder:
    """增量写 HDF5 的演示记录器（resizable datasets，每 flush_every 帧落盘一次）。"""

    def __init__(
        self,
        dataset_file: str,
        *,
        enable_cameras: bool = True,
        flush_every: int = 64,
        max_frames_per_demo: int = 5000,
        stride: int = 20,
        cam_follows_tool: bool = False,
        cam_t_in_tool: tuple[float, float, float] | None = None,
        cam_q_in_tool: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.dataset_file = str(dataset_file)
        self.enable_cameras = bool(enable_cameras)
        # cam_tool 的相机位姿：cam.data.pos_w 在本项目中实测为 0，不能用于 TF。
        # cam_follows_tool=True 时用 tool_link 位姿 + 固定 offset 计算；否则回退到
        # link5 缓存 offset（旧安装方式），再不行才退回 cam.data.pos_w。
        self.cam_follows_tool = bool(cam_follows_tool)
        self.cam_t_in_tool = None if cam_t_in_tool is None else np.asarray(cam_t_in_tool, dtype=np.float32)
        self.cam_q_in_tool = None if cam_q_in_tool is None else np.asarray(cam_q_in_tool, dtype=np.float32)
        self.flush_every = max(1, int(flush_every))
        self.max_frames_per_demo = max(1, int(max_frames_per_demo))
        # 每 stride 个物理步采一帧：完整 SOP 全量采会有 1-2 万步（30-60GB），
        # 默认每 20 步采一帧 → 每个 demo 约 1.5-2GB（~700 帧，460×460 双相机）。
        self.stride = max(1, int(stride))

        # 文件延迟到第一条 demo 的 begin_demo() 才打开，且每条 demo finalize 后关闭：
        # 这样服务空闲时 HDF5 不被锁，外部脚本可以转换/删除；删掉后下一条 demo 会自动
        # 新建文件并从 demo_0 开始。
        self._file = None
        self._demo_idx = 0

        self._group = None
        self._buf: dict[str, list[np.ndarray]] = {}
        self._frame = 0
        self._phase: str | None = None
        self._last_segment: str | None = None
        self._grab_success: bool | None = None
        self._collect_success: bool | None = None
        self._target_pos_w: np.ndarray | None = None
        self._max_frames_warned = False
        self._steps_seen = 0
        # id(robot) → (arm_ids, gripper_ids)
        self._joint_cache: dict[int, tuple] = {}
        # 空白帧保护：相机位姿未变时渲染器跳过重渲染，读出全白帧 → 用上一有效帧替换。
        self._last_valid_rgb: dict[str, np.ndarray | None] = {}
        self._last_valid_depth: dict[str, np.ndarray | None] = {}

    # ── 生命周期 ──────────────────────────────────────────────────────────────
    def begin_demo(self, parameters: dict | None = None) -> None:
        """开始一条新 demo；若上一条未收尾先 finalize。"""
        try:
            self.finalize_demo()
            self._open_file()
            # 每次重新找第一个空闲槽位。数据集里一旦有空洞（删过失败的 demo、或 repack
            # 重写过），_demo_idx 递增后会落到已存在的编号上，create_group 撞名 →
            # begin_demo 每次失败 → 之后一条都存不下来。
            while f"demo_{self._demo_idx}" in self._file:
                self._demo_idx += 1
            self._group = self._file.create_group(f"demo_{self._demo_idx}")
            self._buf = {}
            self._frame = 0
            self._steps_seen = 0
            self._phase = PHASE_GRAB
            self._last_segment = PHASE_GRAB
            self._grab_success = None
            self._collect_success = None
            self._target_pos_w = _extract_target_pos(parameters)
            self._max_frames_warned = False
            self._last_valid_rgb = {}
            self._last_valid_depth = {}
            print(
                f"[HDF5DemoRecorder] begin demo_{self._demo_idx} "
                f"target_pos_w={None if self._target_pos_w is None else self._target_pos_w.tolist()}",
                flush=True,
            )
        except Exception as exc:
            print(f"[HDF5DemoRecorder] begin_demo failed: {exc}", flush=True)
            self._group = None

    def set_phase(self, phase: str) -> None:
        """切换当前阶段标签。'return' 按上一段自动消歧。"""
        if self._group is None:
            return
        if phase == "return":
            phase = PHASE_RETURN_AFTER_DROP if self._last_segment == PHASE_TAKE_OUT else PHASE_RETURN_AFTER_GRAB
        self._phase = phase
        self._last_segment = phase

    def note_result(self, kind: str, result) -> None:
        """记录 grab / take_out 工具执行的成功与否，并立即把标签写入 attrs。

        立即写是为了进程被强杀（SIGKILL/OOM）时 demo 也能带上已确认的成功标签。
        """
        if self._group is None or not isinstance(result, dict):
            return
        if kind == "grab":
            self._grab_success = bool(result.get("executed", False)) or bool(result.get("paper_grasped", False))
        elif kind == "take_out":
            self._collect_success = bool(result.get("executed", False)) or bool(result.get("paper_collected", False))
        self._write_success_attrs()

    def _write_success_attrs(self) -> None:
        if self._group is None:
            return
        try:
            grab_ok = bool(self._grab_success) if self._grab_success is not None else False
            collect_ok = bool(self._collect_success) if self._collect_success is not None else False
            self._group.attrs["grab_success"] = grab_ok
            self._group.attrs["collect_success"] = collect_ok
            self._group.attrs["success"] = bool(self._grab_success is True and self._collect_success is True)
            self._file.flush()
        except Exception as exc:
            print(f"[HDF5DemoRecorder] write success attrs failed: {exc}", flush=True)

    def is_recording(self) -> bool:
        return self._group is not None

    def record_frame(self, env, robot, cube_key: str = "grasp_test_cube_1") -> None:
        """从 env.step 录制包装调用：每 stride 个物理步采一帧追加到当前 demo。"""
        if self._group is None:
            return
        self._steps_seen += 1
        if self._frame >= self.max_frames_per_demo:
            if not self._max_frames_warned:
                self._max_frames_warned = True
                print(
                    f"[HDF5DemoRecorder] demo_{self._demo_idx} reached max_frames_per_demo="
                    f"{self.max_frames_per_demo}; truncating (will save partial demo)",
                    flush=True,
                )
            return
        if self._steps_seen % self.stride != 0:
            return  # 降采样：跳过本步
        if self.enable_cameras:
            try:
                # 普通模式 env.step 内部不一定 sim.render()，强制渲染一次保证相机出图
                env.sim.render()
            except Exception:
                pass
        frame = self._collect_frame(env, robot, cube_key)
        if frame is None:
            return
        self._frame += 1
        for name, arr in frame.items():
            self._buf.setdefault(name, []).append(arr)
        self._buf.setdefault("phase", []).append(self._phase)
        if self._frame % self.flush_every == 0:
            self._flush_buf()

    def finalize_demo(self) -> None:
        """收尾当前 demo：落盘剩余帧 + 写 attrs。"""
        if self._group is None:
            return
        try:
            self._flush_buf()
            n = int(self._group["joint_pos"].shape[0]) if "joint_pos" in self._group else 0
            grab_ok = bool(self._grab_success) if self._grab_success is not None else False
            collect_ok = bool(self._collect_success) if self._collect_success is not None else False
            self._group.attrs["n_frames"] = n
            self._group.attrs["grab_success"] = grab_ok
            self._group.attrs["collect_success"] = collect_ok
            self._group.attrs["success"] = bool(self._grab_success is True and self._collect_success is True)
            if self._target_pos_w is not None:
                self._group.attrs["target_pos_w"] = self._target_pos_w
            print(
                f"[HDF5DemoRecorder] saved demo_{self._demo_idx}: n_frames={n} "
                f"grab_success={grab_ok} collect_success={collect_ok} "
                f"success={bool(grab_ok and collect_ok)}",
                flush=True,
            )
        except Exception as exc:
            print(f"[HDF5DemoRecorder] finalize_demo failed: {exc}", flush=True)
        finally:
            self._demo_idx += 1
            self._group = None
            self._buf = {}
            self._frame = 0
            # 每条 demo 收尾后关闭文件：保证落盘，同时释放 HDF5 锁，让外部转换/删除
            # 脚本可以操作；下一条 demo 会在 begin_demo() 中重新打开（或新建）。
            self._close_file()

    def close(self) -> None:
        """进程退出时调用：收尾未闭合 demo 并关闭文件。"""
        try:
            self.finalize_demo()
        finally:
            self._close_file()
            print(f"[HDF5DemoRecorder] dataset closed: {self.dataset_file}", flush=True)

    def _open_file(self) -> None:
        """按需打开 HDF5；文件被删除后重新创建，并从第一个空槽位继续。"""
        if self._file is not None:
            return
        out_dir = os.path.dirname(os.path.abspath(self.dataset_file))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self._file = h5py.File(self.dataset_file, "a")
        self._demo_idx = 0
        while f"demo_{self._demo_idx}" in self._file:
            self._demo_idx += 1

    def _close_file(self) -> None:
        if self._file is None:
            return
        try:
            self._file.close()
        except Exception as exc:
            print(f"[HDF5DemoRecorder] close file failed: {exc}", flush=True)
        finally:
            self._file = None

    # ── 内部 ──────────────────────────────────────────────────────────────────
    def _flush_buf(self) -> None:
        if not self._buf or self._group is None:
            return
        for name, arrs in self._buf.items():
            data = np.stack(arrs, axis=0) if name != "phase" else np.asarray(arrs, dtype=h5py.string_dtype("utf-8", 24))
            if name not in self._group:
                maxshape = (None,) + data.shape[1:] if data.ndim > 1 else (None,)
                self._group.create_dataset(name, data=data, maxshape=maxshape, chunks=True)
            else:
                ds = self._group[name]
                cur = int(ds.shape[0])
                ds.resize(cur + data.shape[0], axis=0)
                ds[cur:] = data
        # 立即把元数据/数据落盘，进程被强杀（SIGKILL/OOM）时文件也保持可读，避免坏文件。
        try:
            if "joint_pos" in self._group:
                self._group.attrs["n_frames"] = int(self._group["joint_pos"].shape[0])
            self._file.flush()
        except Exception as exc:
            print(f"[HDF5DemoRecorder] file flush failed: {exc}", flush=True)
        self._buf = {}

    def _resolve_joint_ids(self, robot) -> tuple[np.ndarray, np.ndarray]:
        key = id(robot)
        cached = self._joint_cache.get(key)
        if cached is not None:
            return cached
        arm_ids = np.asarray(robot.find_joints("joint[1-6]", preserve_order=True)[0], dtype=np.int64)
        gripper_ids = np.asarray(robot.find_joints("gripper_joint[12]", preserve_order=True)[0], dtype=np.int64)
        if gripper_ids.size == 0:
            gripper_ids = np.asarray(robot.find_joints("gripper_joint*", preserve_order=True)[0], dtype=np.int64)
        self._joint_cache[key] = (arm_ids, gripper_ids)
        return arm_ids, gripper_ids

    def _tool_tip_pose(self, robot) -> tuple[np.ndarray, np.ndarray]:
        names = robot.data.body_names
        for target in ("tool_tip_link", "tool_link", "link6"):
            if target in names:
                idx = names.index(target)
                pos = robot.data.body_pos_w[0, idx].detach().cpu().numpy()
                quat = robot.data.body_quat_w[0, idx].detach().cpu().numpy()
                return pos.astype(np.float32), quat.astype(np.float32)
        pos = robot.data.body_pos_w[0, 0].detach().cpu().numpy()
        quat = robot.data.body_quat_w[0, 0].detach().cpu().numpy()
        return pos.astype(np.float32), quat.astype(np.float32)

    def _body_pose(self, robot, name: str) -> tuple[np.ndarray, np.ndarray]:
        idx = robot.data.body_names.index(name)
        pos = robot.data.body_pos_w[0, idx].detach().cpu().numpy()
        quat = robot.data.body_quat_w[0, idx].detach().cpu().numpy()
        return pos.astype(np.float32), quat.astype(np.float32)

    def _tool_link_pose(self, robot) -> tuple[np.ndarray, np.ndarray]:
        if "tool_link" in robot.data.body_names:
            return self._body_pose(robot, "tool_link")
        # 没有 tool_link 的旧 USD 退回 tool_tip/link6
        return self._tool_tip_pose(robot)

    def _link5_pose(self, robot) -> tuple[np.ndarray, np.ndarray] | None:
        for name in ("XMS5_R800_W4G3B4C_link5", "link5"):
            if name in robot.data.body_names:
                return self._body_pose(robot, name)
        return None

    def _base_pose(self, robot) -> tuple[np.ndarray, np.ndarray]:
        names = robot.data.body_names
        if "base_link" in names:
            idx = names.index("base_link")
            pos = robot.data.body_pos_w[0, idx].detach().cpu().numpy()
            quat = robot.data.body_quat_w[0, idx].detach().cpu().numpy()
            return pos.astype(np.float32), quat.astype(np.float32)
        pose = robot.data.root_link_pose_w[0].detach().cpu().numpy()
        return pose[:3].astype(np.float32), pose[3:7].astype(np.float32)

    def _cube_pos(self, env, cube_key: str) -> np.ndarray:
        try:
            cube = env.scene[cube_key]
            return cube.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    def _camera_data(self, env, cam_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """返回 (rgb HWC uint8, depth HWC float32米)，相机缺失/未渲染则 None。"""
        try:
            cam = env.scene.sensors.get(cam_name, None)
            if cam is None:
                return None
            output = cam.data.output
            rgb = output.get("rgb")
            depth = output.get("depth")
            if rgb is None or depth is None or len(rgb) == 0:
                return None
            rgb_np = np.asarray(rgb[0].detach().cpu().numpy(), dtype=np.uint8)
            depth_np = np.asarray(depth[0].detach().cpu().numpy(), dtype=np.float32)
            if rgb_np.ndim == 3 and rgb_np.shape[-1] == 4:
                rgb_np = rgb_np[..., :3]
            return rgb_np, depth_np
        except Exception:
            return None

    def _collect_frame(self, env, robot, cube_key: str) -> dict | None:
        """采集单帧数据（不触碰相机渲染；相机数据需调用方在 _step 内 render 过）。"""
        try:
            arm_ids, gripper_ids = self._resolve_joint_ids(robot)
            jp = robot.data.joint_pos[0].detach().cpu().numpy()
            jv = robot.data.joint_vel[0].detach().cpu().numpy()

            out: dict = {}
            out["joint_pos"] = np.concatenate([jp[arm_ids], jp[gripper_ids]]).astype(np.float32)
            out["joint_vel"] = np.concatenate([jv[arm_ids], jv[gripper_ids]]).astype(np.float32)

            try:
                jt = robot.data.joint_pos_target
                if jt is not None and jt.numel() > 0:
                    jt_np = jt[0].detach().cpu().numpy()
                    out["joint_target"] = np.concatenate([jt_np[arm_ids], jt_np[gripper_ids]]).astype(np.float32)
                else:
                    out["joint_target"] = out["joint_pos"]
            except Exception:
                out["joint_target"] = out["joint_pos"]

            try:
                wheel_ids = np.asarray(
                    robot.find_joints(
                        ["wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint"],
                        preserve_order=True,
                    )[0],
                    dtype=np.int64,
                )
                out["wheel_vel"] = jv[wheel_ids].astype(np.float32)
            except Exception:
                out["wheel_vel"] = np.zeros(4, dtype=np.float32)

            tool_pos, tool_quat = self._tool_tip_pose(robot)
            tool_link_pos, tool_link_quat = self._tool_link_pose(robot)
            base_pos, base_quat = self._base_pose(robot)
            out["tool_tip_pos_w"] = tool_pos
            out["tool_tip_quat_w"] = tool_quat
            out["base_pos_w"] = base_pos
            out["base_quat_w"] = base_quat
            out["cube_pos_w"] = self._cube_pos(env, cube_key)
            out["timestamps"] = np.array([self._steps_seen * float(env.physics_dt)], dtype=np.float32)

            if self.enable_cameras:
                for cam_name in ("cam_tool", "cam_fixed"):
                    cam_data = self._camera_data(env, cam_name)
                    if cam_data is None:
                        continue
                    rgb_np, depth_np = cam_data
                    # 空白帧保护：相机位姿未变时渲染器跳过重渲染，读出全白/空数据
                    # （rgb 近似均匀）。位姿未变场景也未变，用上一有效帧替换等价无损失，
                    # 避免转 video 后每 3 帧/空闲段出现闪烁。仅替换像素，不删帧，时序不变。
                    is_blank = float(rgb_np[..., :3].std()) < 3.0
                    if is_blank and self._last_valid_rgb.get(cam_name) is not None:
                        rgb_np = self._last_valid_rgb[cam_name]
                        depth_np = self._last_valid_depth.get(cam_name, depth_np)
                    else:
                        self._last_valid_rgb[cam_name] = rgb_np
                        self._last_valid_depth[cam_name] = depth_np
                    out[f"obs/{cam_name}_rgb"] = rgb_np
                    out[f"obs/{cam_name}_depth"] = depth_np
                    # 相机位姿在 tool_tip_link / base 系下的 4x4（行主序 16），与
                    # tool_tip_pos_w 使用同一 tool 帧。
                    # 注意：cam_tool 的 cam.data.pos_w 在本项目中实测为 0（相机由
                    # set_world_poses 手动同步），不能直接拿来算 TF。cam_follows_tool=True
                    # 时用 tool_link 位姿 + 固定 offset 计算相机世界位姿，结果恒定且正确。
                    cam = env.scene.sensors.get(cam_name)
                    cam_pos = None
                    cam_quat = None
                    if (
                        cam_name == "cam_tool"
                        and self.cam_follows_tool
                        and self.cam_t_in_tool is not None
                        and self.cam_q_in_tool is not None
                    ):
                        cam_pos, cam_quat = _compose_frames(
                            tool_link_pos, tool_link_quat, self.cam_t_in_tool, self.cam_q_in_tool
                        )
                    elif cam_name == "cam_tool":
                        # 旧 link5 安装：优先用 sync_link5_camera 缓存的 link5->cam offset
                        offset = getattr(env.cfg, "_cam_link5_mount_offset", None)
                        link5_pose = self._link5_pose(robot)
                        if offset is not None and link5_pose is not None:
                            try:
                                t_l5 = offset[0][0].detach().cpu().numpy().astype(np.float32)
                                q_l5 = offset[1][0].detach().cpu().numpy().astype(np.float32)
                                cam_pos, cam_quat = _compose_frames(
                                    link5_pose[0], link5_pose[1], t_l5, q_l5
                                )
                            except Exception:
                                cam_pos = None
                    if cam_pos is None or cam_quat is None:
                        try:
                            cam_pos = cam.data.pos_w[0].detach().cpu().numpy().astype(np.float32)
                            cam_quat = cam.data.quat_w_world[0].detach().cpu().numpy().astype(np.float32)
                        except Exception:
                            cam_pos = None
                    if (
                        cam_pos is None
                        or cam_quat is None
                        or not np.all(np.isfinite(cam_pos))
                        or float(np.linalg.norm(cam_pos)) < 1e-6
                    ):
                        # 没有可用相机位姿：宁可不写 TF，也不要写 cam.data.pos_w=0 的垃圾数据
                        continue
                    p_cam_in_tool, q_cam_in_tool = _subtract_frames(
                        tool_pos, tool_quat, cam_pos, cam_quat
                    )
                    p_cam_in_base, q_cam_in_base = _subtract_frames(base_pos, base_quat, cam_pos, cam_quat)
                    out[f"obs/{cam_name}_in_tool_tf"] = _tf4x16(p_cam_in_tool, q_cam_in_tool)
                    out[f"obs/{cam_name}_in_base_tf"] = _tf4x16(p_cam_in_base, q_cam_in_base)
            return out
        except Exception as exc:
            print(f"[HDF5DemoRecorder] record_frame failed: {exc}", flush=True)
            return None
