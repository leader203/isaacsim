# Isaac Lab allstar_toilet 运行说明

本仓库基于 NVIDIA Isaac Lab，当前主要运行入口在：

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet
```

运行命令以这里为准：

- [scripts_runtime.md](source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/scripts_runtime.md)


## 运行范围

这套环境用于运行：

- `usdandurdf/generate_robotbuilder_link2plus_collision_urdf.py`
- `scripts/visualize_curobo_mesh_obstacle.py`
- `scripts/run_curobo_pose_plan_test.py`
- `scripts/run_record_mobile_arm.py`
- `curl` 调用 `/tools/invoke` 和 `/tools/jobs/<request_id>` FastAPI 工具服务

## 基础要求

- Linux x86_64，推荐 Ubuntu 22.04。
- NVIDIA GPU 和可用驱动，目标机器上 `nvidia-smi` 必须正常。
- Conda 或 Miniconda。
- Python 3.10。
- Isaac Sim `4.5.0.0` pip 版。
- PyTorch `2.5.1` / torchvision `0.20.1`，CUDA 12.4 wheel。

## 环境镜像配置

如果当前网络可以直接访问 GitHub、conda、PyPI、`pypi.nvidia.com` 和 PyTorch wheel index，可以跳过本节。公司内网环境可参考下面配置。

### GitHub 镜像

```bash
git config --global url."http://mirrors.sangfor.org/github/".insteadOf "https://github.com/"
```

### Conda 镜像

```bash
conda config --set auto_activate_base false
conda config --remove-key default_channels 2>/dev/null || true

conda config --add default_channels http://mirrors.sangfor.com/nexus/repository/anaconda-tsinghua/pkgs/main
conda config --add default_channels http://mirrors.sangfor.com/nexus/repository/anaconda-tsinghua/pkgs/r
conda config --add default_channels http://mirrors.sangfor.com/nexus/repository/anaconda-tsinghua/pkgs/msys2

conda config --set custom_channels.conda-forge http://mirrors.sangfor.com/nexus/repository/anaconda-tsinghua/cloud
conda config --set custom_channels.pytorch http://mirrors.sangfor.com/nexus/repository/anaconda-tsinghua/cloud

conda config --add channels conda-forge
conda config --set show_channel_urls true
conda config --set remote_connect_timeout_secs 40.0
conda config --set remote_read_timeout_secs 100.0
```

### Pip 镜像

```bash
pip3 config --user set global.index-url \
  http://mirrors.sangfor.com/nexus/repository/pypi-proxy-tencent/simple

pip3 config --user set global.extra-index-url \
  http://mirrors.sangfor.com/nexus/repository/pypi-nvidia

pip3 config --user set global.retries 60
pip3 config --user set install.trusted-host mirrors.sangfor.com
```

注意：Isaac Sim 和 CUDA PyTorch wheel 仍需要能访问 NVIDIA/PyTorch 对应源，或者公司镜像里已经同步了这些包。

## 安装流程

### 1. 进入仓库

```bash
cd /home/stardust/haozhao/pro/IsaacLab
```

如果是从 GitLab clone，替换成自己的仓库路径即可。

### 2. 创建 conda 环境

```bash
conda env create -f environment-allstar-toilet-runtime.yml
conda activate isaaclab
python -m pip install -U pip "setuptools<82.0.0" wheel
```

如果 `isaaclab` 环境已经存在：

```bash
conda activate isaaclab
python -m pip install -U pip "setuptools<82.0.0" wheel
```

确认 Python 路径：

```bash
which python
python -V
```

期望是 conda 环境内的 Python 3.10。

### 3. 安装运行依赖

```bash
python -m pip install -r requirements-allstar-toilet-runtime.txt
```

该文件固定了当前可运行组合的关键版本：

```text
Python              3.10
isaacsim            4.5.0.0
torch               2.5.1
torchvision         0.20.1
numpy               1.26.4
nvidia-curobo       本仓库 editable 安装
```

### 4. 安装本地 Isaac Lab 包

必须使用 `--no-deps`：

```bash
python -m pip install --no-deps -e source/isaaclab
python -m pip install --no-deps -e source/isaaclab_assets
python -m pip install --no-deps -e source/isaaclab_contrib
python -m pip install --no-deps -e source/isaaclab_mimic
python -m pip install --no-deps -e source/isaaclab_rl
python -m pip install --no-deps -e source/isaaclab_tasks
```

原因：当前仓库部分 `setup.py` 声明了 `torch>=2.7`，但本项目运行环境基于 Isaac Sim 4.5，实际需要保留 `torch==2.5.1`。不加 `--no-deps` 时，pip 可能升级或替换 torch，导致 Isaac Sim/cuRobo 运行异常。

### 5. 安装本地 cuRobo

```bash
python -m pip install --no-build-isolation --no-deps -e \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/src/nvidia-curobo
```

`--no-build-isolation` 用于避免 pip 创建临时构建环境并重新下载另一套 PyTorch。cuRobo 依赖已经放在 `requirements-allstar-toilet-runtime.txt` 中。

## 验证环境

### 1. 检查关键包

```bash
python - <<'PY'
from importlib import metadata
import torch

names = [
    "isaacsim",
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_contrib",
    "isaaclab_mimic",
    "isaaclab_rl",
    "isaaclab_tasks",
    "nvidia-curobo",
    "torch",
    "torchvision",
    "trimesh",
    "fastapi",
    "uvicorn",
    "opencv-python",
]

for name in names:
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}: not installed")

print("torch.__version__=", torch.__version__)
print("torch.version.cuda=", torch.version.cuda)
print("torch.cuda.is_available()=", torch.cuda.is_available())
PY
```

### 2. 检查 allstar_toilet 任务注册

```bash
TERM=xterm ./isaaclab.sh -p - <<'PY'
import isaaclab_tasks.manager_based.manipulation.allstar_toilet
print("allstar_toilet import ok")
PY
```

## 运行 smoke test

### 可视化 cuRobo obstacle mesh

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/visualize_curobo_mesh_obstacle.py \
  --save /tmp/toilet_curobo_obstacle.png \
  --no-show
```

### 启动 cuRobo FastAPI 工具服务

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_curobo_pose_plan_test.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Common-CuroboPosePlanTest-Dirty-OpenSeat-v0 \
  --initial_robot_id gripper_prismatic \
  --tool_api_server \
  --tool_api_host 127.0.0.1 \
  --tool_api_port 8000
```

另开一个终端测试接口：

```bash
curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"list_robot_profiles","parameters":{}}'
```


### 运行录制脚本

这个脚本运行时会输出底盘位姿，机械臂关节角以及末端执行器位姿，方便设置机器人初始位姿，机械臂特殊关节角以及为curobo规划提供末端执行器目标位姿。


录制脚本是交互式的，建议在有 GUI/viewport 的会话中运行：

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_record_mobile_arm.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5BrushRolling0512-BrushRolling0512-Dirty-OpenSeat-v0 \
  --num_demos 1 \
  --dataset_file /tmp/mecanum730_xms5_brush_rolling_0512_smoke.hdf5 \
  --speed_scale 150.0 \
  --arm_scale 0.05 \
  --episode_length_s 60 \
  --arm_debug_interval 1 \
  --save_obs False
```

更多命令见 [scripts_runtime.md](source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/scripts_runtime.md)。

## 常见问题

### pip 试图升级 torch

安装本地 Isaac Lab 包时必须使用：

```bash
python -m pip install --no-deps -e source/isaaclab
```

以及其他 `source/isaaclab_*` 包同样加 `--no-deps`。否则 pip 可能根据 `setup.py` 里的 `torch>=2.7` 替换当前 `torch==2.5.1`。

### cuRobo 安装卡在 Installing build dependencies

停止当前安装，改用：

```bash
python -m pip install --no-build-isolation --no-deps -e \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/src/nvidia-curobo
```

### `torch.cuda.is_available()` 是 False

先确认目标机器：

```bash
nvidia-smi
```

如果 `nvidia-smi` 不可用，先修 NVIDIA driver。Isaac Sim 和 cuRobo 规划都依赖 GPU 环境。

### 键盘或 SpaceMouse 不响应

键盘默认会尝试 `/dev/input`、`pynput X11` 和终端 stdin fallback。SpaceMouse 需要 `hidapi`，还可能需要当前用户有 `/dev/hidraw*` 权限。

### GitLab clone 后缺少模型或场景文件

检查这些目录是否完整：

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/data/robots/chassis_arm
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/data/scenes
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/configs/curobo
```

如果使用 Git LFS，需要在运行前执行：

```bash
git lfs install
git lfs pull
```

## GitLab 上传检查清单

- 提交 `environment-allstar-toilet-runtime.yml`、`requirements-allstar-toilet-runtime.txt` 和本 README。
- 确认 `scripts_runtime.md` 中引用的 URDF/USD/mesh/scene/config 文件已提交，或已放入 Git LFS。
- 不要提交当前机器的完整 `pip freeze`。
- 不要提交带 token 的私有 Git URL。
- 不要提交生成的 HDF5 demos、日志、截图、cuRobo obstacle dump，除非它们是必须共享的数据。
