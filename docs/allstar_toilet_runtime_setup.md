# allstar_toilet runtime setup

This setup is scoped to the commands in:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/scripts_runtime.md
```

It is not a full dump of the local `isaaclab` conda environment. The current local environment contains unrelated ROS, Jupyter, and vision packages, and `pip freeze` also contains historical editable GitLab URLs. Do not publish that full freeze file.

## Target runtime

- OS: Linux x86_64, Ubuntu 22.04 recommended.
- GPU: NVIDIA GPU with a working driver. `nvidia-smi` must work on the target machine.
- Python: 3.10.
- Isaac Sim: `4.5.0.0` installed from pip.
- PyTorch: `2.5.1` / `torchvision==0.20.1`, CUDA 12.4 wheel index.
- Main local package path:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet
```

## Commands covered

This environment is intended for:

- `usdandurdf/generate_robotbuilder_link2plus_collision_urdf.py`
- `scripts/visualize_curobo_mesh_obstacle.py`
- `scripts/run_curobo_pose_plan_test.py`
- `scripts/run_record_mobile_arm.py`
- `curl` calls to the `/tools/invoke` and `/tools/jobs/<request_id>` FastAPI service

## System packages

Install basic build and runtime libraries first:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  curl \
  libgl1 \
  libglib2.0-0 \
  libegl1 \
  libx11-6 \
  libxext6 \
  libxrender1 \
  libsm6 \
  libxi6 \
  libxrandr2 \
  libxinerama1 \
  libxcursor1 \
  libnss3 \
  libasound2
```

For keyboard teleop, the terminal fallback works without special device permissions. For SpaceMouse or `/dev/input` capture, the user may need access to `input` or `plugdev` groups depending on the machine.

## Create conda env

From the repo root:

```bash
conda env create -f environment-allstar-toilet-runtime.yml
conda activate isaaclab
python -m pip install -U pip "setuptools<82.0.0" wheel
```

If `isaaclab` already exists:

```bash
conda activate isaaclab
python -m pip install -U pip "setuptools<82.0.0" wheel
```

## Install Python runtime dependencies

```bash
python -m pip install -r requirements-allstar-toilet-runtime.txt
```

If your network needs company mirrors, configure those mirrors before this step. Keep the NVIDIA and PyTorch wheel indexes available, because Isaac Sim and CUDA PyTorch wheels are not served by normal PyPI mirrors.

## Install local Isaac Lab packages

Install local packages editable, but do not let pip resolve their dependencies:

```bash
python -m pip install --no-deps -e source/isaaclab
python -m pip install --no-deps -e source/isaaclab_assets
python -m pip install --no-deps -e source/isaaclab_contrib
python -m pip install --no-deps -e source/isaaclab_mimic
python -m pip install --no-deps -e source/isaaclab_rl
python -m pip install --no-deps -e source/isaaclab_tasks
```

The `--no-deps` flag is important. Some local `setup.py` files declare `torch>=2.7`, while this runtime is based on Isaac Sim 4.5 and the known-working `torch==2.5.1` stack.

## Install local cuRobo

```bash
python -m pip install --no-build-isolation --no-deps -e \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/src/nvidia-curobo
```

The cuRobo dependencies are already listed in `requirements-allstar-toilet-runtime.txt`. `--no-build-isolation` avoids pip creating a temporary build env and downloading a different PyTorch.

## Verify

Run from the repo root:

```bash
python - <<'PY'
from importlib import metadata
import torch

names = [
    "isaacsim",
    "isaaclab",
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

Then verify the Isaac Lab launcher can import the custom task package:

```bash
TERM=xterm ./isaaclab.sh -p - <<'PY'
import isaaclab_tasks.manager_based.manipulation.allstar_toilet
print("allstar_toilet import ok")
PY
```

## Smoke-test commands

Mesh viewer in headless mode:

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/visualize_curobo_mesh_obstacle.py \
  --save /tmp/toilet_curobo_obstacle.png \
  --no-show
```

Start the cuRobo FASTAPI tool server:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/allstar_toilet/scripts/run_curobo_pose_plan_test.py \
  --task Isaac-Allstar-Toilet-Mecanum0730-XMS5Common-CuroboPosePlanTest-Dirty-OpenSeat-v0 \
  --initial_robot_id gripper_prismatic \
  --tool_api_server \
  --tool_api_host 127.0.0.1 \
  --tool_api_port 8000
```

In another terminal:

```bash
curl -sS -X POST http://127.0.0.1:8000/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool_name":"list_robot_profiles","parameters":{}}'
```

Recorder commands are interactive and need a GUI/non-headless session if you want viewport and keyboard behavior:

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

## GitLab upload checklist

- Keep robot and scene assets required by `scripts_runtime.md` in Git or Git LFS, especially `data/robots/chassis_arm`, `data/scenes`, and `configs/curobo`.
- Put generated logs, HDF5 demos, screenshots, and cuRobo obstacle dumps in `.gitignore` unless they are intentionally shared.
- Do not commit `pip freeze` output from the current machine; it can include private GitLab tokens and unrelated packages.
- If you use Git LFS, document `git lfs install` and `git lfs pull` in the GitLab README before running the smoke tests.
