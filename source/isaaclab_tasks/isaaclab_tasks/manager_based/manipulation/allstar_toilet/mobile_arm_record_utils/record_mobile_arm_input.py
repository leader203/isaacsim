"""Keyboard and SpaceMouse teleoperation devices."""

import select
import sys
import termios
import threading
import time
import tty

import numpy as np


class EvdevKeyboard:
    """Linux evdev 全局键盘读取器。

    直接读取 /dev/input/event*，不依赖任何窗口焦点。
    需要当前用户在 input 组：sudo usermod -a -G input $USER  (重新登录后生效)。
    """

    EV_KEY    = 1
    KEY_PRESS = 1
    KEY_REPEAT = 2

    # Linux kernel 键码 → 大写字母
    KEY_CODES: dict[int, str] = {
        16: 'Q', 17: 'W', 18: 'E', 19: 'R', 20: 'T',
        24: 'O', 23: 'I',
        30: 'A', 31: 'S', 32: 'D', 34: 'G',
        44: 'Z', 45: 'X', 46: 'C', 47: 'V',
        25: 'P', 50: 'M',
    }

    SENSITIVITY: float = 0.05

    def __init__(self):
        self._pressed: set[str] = set()
        self._callbacks: dict[str, callable] = {}
        self._lock = threading.Lock()
        self._running = True
        self._fds = self._open_keyboards()
        if not self._fds:
            raise RuntimeError(
                "没有可访问的键盘设备。请先运行：\n"
                "  sudo usermod -a -G input $USER\n"
                "然后重新登录。"
            )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    @staticmethod
    def _open_keyboards() -> list:
        import glob, fcntl, array
        EVIOCGBIT_0 = 0x80044520   # EVIOCGBIT(0, 4)
        EVIOCGNAME  = 0x81004506   # EVIOCGNAME(256)
        EV_KEY_FLAG = 2            # bit 1
        fds = []
        all_paths = sorted(glob.glob('/dev/input/event*'))
        print(f"[evdev] 扫描 {len(all_paths)} 个输入设备:", flush=True)
        for path in all_paths:
            try:
                fd = open(path, 'rb', buffering=0)
                ev_bits = array.array('B', [0] * 4)
                fcntl.ioctl(fd, EVIOCGBIT_0, ev_bits)
                has_key = bool(ev_bits[0] & EV_KEY_FLAG)
                # 真正键盘还支持 EV_REP（按键重复, bit 20）；LED 控制器/鼠标没有
                has_rep = bool(ev_bits[2] & (1 << 4))
                is_keyboard = has_key and has_rep
                try:
                    name_buf = array.array('B', [0] * 256)
                    fcntl.ioctl(fd, EVIOCGNAME, name_buf)
                    name = bytes(name_buf).decode('utf-8', errors='replace').rstrip('\x00')
                except Exception:
                    name = '<unknown>'
                if is_keyboard:
                    tag = "✅ 键盘"
                elif has_key:
                    tag = "  (EV_KEY 但无 EV_REP，跳过)"
                else:
                    tag = "  (跳过)"
                print(f"[evdev]   {tag}  {path}  —  {name}", flush=True)
                if is_keyboard:
                    fds.append(fd)
                else:
                    fd.close()
            except Exception as e:
                print(f"[evdev]   ❌ 无法访问 {path}: {e}", flush=True)
        return fds

    def _reader(self) -> None:
        import struct
        fmt = 'llHHi'   # timeval(8+8) + type(2) + code(2) + value(4) = 24 B
        sz  = struct.calcsize(fmt)
        _debug_until = time.monotonic() + 30.0  # 前 30 s 打印所有按键事件
        while self._running:
            try:
                ready, _, _ = select.select(self._fds, [], [], 0.05)
                for fd in ready:
                    data = fd.read(sz)
                    if len(data) < sz:
                        continue
                    _, _, ev_type, code, value = struct.unpack(fmt, data)
                    if ev_type == self.EV_KEY and time.monotonic() < _debug_until:
                        print(f"\n[evdev KEY] code={code} value={value}"
                              f" → {self.KEY_CODES.get(code, '未映射')}", flush=True)
                    # P 键（code=25）永久打印，方便排查录制流程
                    if ev_type == self.EV_KEY and code == 25 and value == self.KEY_PRESS:
                        print(f"\n[evdev P] 收到 P 键，触发 task_state.next()", flush=True)
                    if ev_type != self.EV_KEY:
                        continue
                    key = self.KEY_CODES.get(code)
                    if key is None:
                        continue
                    with self._lock:
                        if value in (self.KEY_PRESS, self.KEY_REPEAT):
                            self._pressed.add(key)
                        else:
                            self._pressed.discard(key)
                    if value == self.KEY_PRESS:
                        cb = self._callbacks.get(key)
                        if cb:
                            cb()
            except Exception:
                pass

    def is_held(self, key: str) -> bool:
        with self._lock:
            return key.upper() in self._pressed

    def get_axis(self, pos_key: str, neg_key: str) -> float:
        return float(self.is_held(pos_key)) - float(self.is_held(neg_key))

    def add_callback(self, key: str, func) -> None:
        with self._lock:
            self._callbacks[key.upper()] = func

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=0.3)
        for fd in self._fds:
            try:
                fd.close()
            except Exception:
                pass


class TerminalKeyboard:
    """备用：从终端 stdin 读取（需要终端窗口保持激活状态）。"""

    SENSITIVITY: float = 0.05

    def __init__(self, hold_sec: float = 0.12):
        self._hold_sec = hold_sec
        self._pressed: dict[str, float] = {}
        self._callbacks: dict[str, callable] = {}
        self._lock = threading.Lock()
        self._running = True
        self._fd = sys.stdin.fileno()
        self._old_term = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        try:
            while self._running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue
                ch = sys.stdin.read(1).upper()
                now = time.monotonic()
                with self._lock:
                    self._pressed[ch] = now
                cb = self._callbacks.get(ch)
                if cb:
                    cb()
        except Exception:
            pass

    def is_held(self, key: str) -> bool:
        with self._lock:
            t = self._pressed.get(key.upper(), 0.0)
        return (time.monotonic() - t) < self._hold_sec

    def get_axis(self, pos_key: str, neg_key: str) -> float:
        return float(self.is_held(pos_key)) - float(self.is_held(neg_key))

    def add_callback(self, key: str, func) -> None:
        with self._lock:
            self._callbacks[key.upper()] = func

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=0.3)
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
        except Exception:
            pass


class PynputKeyboard:
    """pynput X11 全局键盘（需要 X11 环境，无需 /dev/input 权限）。"""

    SENSITIVITY: float = 0.05
    _KEYS = set('WASDZXQETGCVIO MRP'.replace(" ", ""))

    def __init__(self):
        from pynput import keyboard as pk
        self._pressed: set[str] = set()
        self._callbacks: dict[str, callable] = {}
        self._lock = threading.Lock()

        def on_press(key):
            try:
                ch = key.char.upper()
            except AttributeError:
                return
            if ch not in self._KEYS:
                return
            with self._lock:
                self._pressed.add(ch)
                cb = self._callbacks.get(ch)
            if cb:
                cb()

        def on_release(key):
            try:
                ch = key.char.upper()
            except AttributeError:
                return
            with self._lock:
                self._pressed.discard(ch)

        self._listener = pk.Listener(on_press=on_press, on_release=on_release, suppress=False)
        self._listener.daemon = True
        self._listener.start()

    def is_held(self, key: str) -> bool:
        with self._lock:
            return key.upper() in self._pressed

    def get_axis(self, pos_key: str, neg_key: str) -> float:
        return float(self.is_held(pos_key)) - float(self.is_held(neg_key))

    def add_callback(self, key: str, func) -> None:
        with self._lock:
            self._callbacks[key.upper()] = func

    def close(self) -> None:
        self._listener.stop()


def _create_keyboard(terminal_hold_sec: float):
    """优先 evdev 全局捕获 → pynput X11 → 终端 stdin。"""
    try:
        kb = EvdevKeyboard()
        if not kb._fds:
            raise RuntimeError("无可用键盘设备（请检查上方扫描结果）")
        print(f"[键盘] ✅ evdev 全局模式（{len(kb._fds)} 个设备）— 任意窗口下按键均有效", flush=True)
        return kb
    except Exception as e:
        print(f"[键盘] evdev 失败: {e}", flush=True)

    # pynput 在 isaaclab 环境下 X11 监听不可靠，直接跳到 stdin 模式

    print(
        f"[键盘] ⚠️  终端模式 — 按键时请保持此终端窗口激活（hold_sec={terminal_hold_sec:.2f}s）",
        flush=True,
    )
    return TerminalKeyboard(hold_sec=terminal_hold_sec)


class SpaceMouseTeleop:
    """脚本内 SpaceMouse 适配器。

    单设备同时支持：
    - 底盘模式：平移 x/y + 旋转 z
    - 机械臂模式：6 轴分别映射到 6 个关节增量

    按键约定：
    - 左键 `L`：切换模式
    - 右键 `R`：重置环境
    - 左右同时按 `LR`：推进任务阶段
    """

    SENSITIVITY: float = 0.05

    def __init__(self):
        try:
            import hid as hid_module
            from isaaclab.devices.spacemouse.utils import convert_buffer as convert_buffer_fn
        except ImportError as exc:
            raise ImportError(
                "SpaceMouse teleop requires the Python HID dependency. "
                "Install hidapi/hid and then rerun with --teleop_device spacemouse."
            ) from exc

        self._hid = hid_module
        self._convert_buffer = convert_buffer_fn
        self._device = self._hid.device()
        self._device_name = ""
        self._callbacks: dict[str, callable] = {}
        self._lock = threading.Lock()
        self._running = True
        self._find_device()
        self._translation = np.zeros(3, dtype=np.float32)
        self._rotation = np.zeros(3, dtype=np.float32)
        self._base_twist = np.zeros(3, dtype=np.float32)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _find_device(self) -> None:
        supported_devices = {
            "SpaceMouse Compact",
            "SpaceMouse Wireless",
            "3Dconnexion Universal Receiver",
        }
        last_error = None
        for _ in range(5):
            for device in self._hid.enumerate():
                if device.get("product_string") in supported_devices:
                    try:
                        self._device.close()
                    except Exception:
                        pass
                    try:
                        # 优先按 hidraw 路径打开，避免同 VID/PID 下打开到错误接口。
                        if device.get("path"):
                            path = device["path"]
                            if isinstance(path, str):
                                path = path.encode("utf-8")
                            self._device.open_path(path)
                        else:
                            self._device.open(device["vendor_id"], device["product_id"])
                        self._device_name = device["product_string"]
                        return
                    except OSError as exc:
                        path = device.get("path", b"")
                        if isinstance(path, bytes):
                            path = path.decode("utf-8", errors="replace")
                        last_error = (
                            f"{device.get('product_string')} "
                            f"(vid=0x{device.get('vendor_id', 0):04x}, pid=0x{device.get('product_id', 0):04x}, "
                            f"path={path}) -> {exc}"
                        )
            time.sleep(1.0)
        if last_error is not None:
            raise OSError(
                "Found a supported SpaceMouse but failed to open it. "
                "This is usually a /dev/hidraw permission issue. "
                f"Last error: {last_error}"
            )
        raise OSError("No supported SpaceMouse found. Check USB connection and /dev/hidraw permissions.")

    def __str__(self) -> str:
        return f"SpaceMouse({self._device_name})"

    def add_callback(self, key: str, func) -> None:
        with self._lock:
            self._callbacks[key.upper()] = func

    def get_base_axes(self) -> tuple[float, float, float]:
        with self._lock:
            x, y, yaw = self._base_twist.tolist()
        return x, y, yaw

    def get_arm_axes(self) -> np.ndarray:
        with self._lock:
            return np.concatenate([self._translation, self._rotation]).copy()

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=0.3)
        try:
            self._device.close()
        except Exception:
            pass

    def _emit_callback(self, key: str) -> None:
        with self._lock:
            callback = self._callbacks.get(key)
        if callback:
            callback()

    def _reader(self) -> None:
        while self._running:
            try:
                if self._device_name == "3Dconnexion Universal Receiver":
                    data = self._device.read(13)
                else:
                    data = self._device.read(7)
                if not data:
                    continue

                if self._device_name == "3Dconnexion Universal Receiver":
                    if data[0] == 1:
                        with self._lock:
                            self._translation[1] = self._convert_buffer(data[1], data[2])
                            self._translation[0] = self._convert_buffer(data[3], data[4])
                            self._translation[2] = -self._convert_buffer(data[5], data[6])
                            self._rotation[1] = self._convert_buffer(data[7], data[8])
                            self._rotation[0] = self._convert_buffer(data[9], data[10])
                            self._rotation[2] = -self._convert_buffer(data[11], data[12])
                            self._base_twist[1] = self._translation[1]
                            self._base_twist[0] = self._translation[0]
                            self._base_twist[2] = self._rotation[0]
                else:
                    if data[0] == 1:
                        with self._lock:
                            self._translation[1] = self._convert_buffer(data[1], data[2])
                            self._translation[0] = self._convert_buffer(data[3], data[4])
                            self._translation[2] = -self._convert_buffer(data[5], data[6])
                            self._base_twist[1] = self._translation[1]
                            self._base_twist[0] = self._translation[0]
                    elif data[0] == 2:
                        with self._lock:
                            self._rotation[1] = self._convert_buffer(data[1], data[2])
                            self._rotation[0] = self._convert_buffer(data[3], data[4])
                            self._rotation[2] = -self._convert_buffer(data[5], data[6])
                            self._base_twist[2] = self._rotation[0]

                if data[0] == 3:
                    if data[1] == 1:
                        self._emit_callback("L")
                    elif data[1] == 2:
                        self._emit_callback("R")
                    elif data[1] == 3:
                        self._emit_callback("LR")
            except Exception:
                time.sleep(0.01)


def _create_teleop(device_name: str, keyboard_hold_sec: float):
    if device_name == "keyboard":
        return _create_keyboard(keyboard_hold_sec)
    if device_name == "spacemouse":
        teleop = SpaceMouseTeleop()
        print(f"[SpaceMouse] ✅ 已连接 {teleop}", flush=True)
        return teleop
    raise ValueError(f"Unsupported teleop device: {device_name}")


def print_mobile_arm_teleop_help(
    teleop_device: str,
    *,
    use_prismatic_gripper_keyboard: bool,
) -> None:
    print("\n[Mobile Arm Teleoperation]")
    print(f"[设备] {teleop_device}")
    if teleop_device == "keyboard":
        print("  ← 直接在此终端按键即可，无需点击 Isaac Sim 窗口")
        print()
        print("  M    — 切换控制模式（底盘 ↔ 机械臂）")
        print()
        print("  [底盘模式]")
        print("    W/S  — 前进/后退      A/D  — 左平移/右平移      Z/X  — 左转/右转")
        print()
        print("  [机械臂模式]（增量叠加）")
        print("    W/S  — joint1(底座)   A/D  — joint2(肩)   Q/E  — joint3(肘)")
        print("    Z/X  — joint4(腕转)   T/G  — joint5(腕仰)  C/V  — joint6(末端)")
        if use_prismatic_gripper_keyboard:
            print("    I/O  — prismatic 夹爪闭合/张开")
        print()
        print("  R    — 重置环境      P    — 推进任务阶段      Ctrl+C — 退出\n")
    else:
        print("  左键  — 切换控制模式（底盘 ↔ 机械臂）")
        print("  右键  — 重置环境")
        print("  双键  — 推进任务阶段")
        print()
        print("  [底盘模式]")
        print("    平移 x/y  — 前后/横移      旋转 z  — 转向")
        print()
        print("  [机械臂模式]（增量叠加）")
        print("    平移 x/y/z  — joint1/joint2/joint3")
        print("    旋转 x/y/z  — joint4/joint5/joint6")
        print()
        print("  Ctrl+C — 退出\n")
    print("[初始模式] → 底盘控制\n")
