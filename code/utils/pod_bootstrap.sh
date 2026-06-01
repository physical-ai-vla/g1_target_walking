#!/usr/bin/env bash
# Bootstrap the RunPod RTX 6000 Ada pod for G1Nav.
# Idempotent-ish: safe to re-run. Logs to /workspace/bootstrap.log
set -u
LOG=/workspace/bootstrap.log
exec > >(tee -a "$LOG") 2>&1
echo "===== bootstrap start $(date -u) ====="

export DEBIAN_FRONTEND=noninteractive
export MUJOCO_GL=egl
export PIP_DISABLE_PIP_VERSION_CHECK=1

cd /workspace || exit 1

echo "--- apt: system libs for EGL/GL offscreen render + git-lfs ---"
apt-get update -qq
apt-get install -y -qq libegl1 libgl1 libglew2.2 libosmesa6 ffmpeg git-lfs python3-venv >/dev/null 2>&1
git lfs install >/dev/null 2>&1
echo "apt done"

# Ubuntu 24.04 is PEP-668 externally-managed. Use a dedicated venv that
# inherits the system torch (which already has CUDA 12.8) via system-site-packages.
VENV=/workspace/venv
if [ ! -d "$VENV" ]; then
  echo "--- creating venv with system site packages (keeps system torch+cuda) ---"
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "python: $(which python)  ($(python --version 2>&1))"
python -c "import torch;print('torch in venv:',torch.__version__, torch.cuda.is_available())" 2>&1 | head -1

echo "--- pip: core sim + RL (JAX CUDA, MuJoCo, playground) ---"
pip install -q --upgrade pip
# MuJoCo (CPU+GPU render) and MJX
pip install -q mujoco==3.4.0 mujoco-mjx==3.4.0
# JAX with CUDA12 (matches driver 580 / cu12.x)
pip install -q "jax[cuda12]"
# RL stack used by mujoco_playground
pip install -q brax flax optax distrax
pip install -q playground   # mujoco_playground PyPI name
# media + utils
pip install -q mediapy imageio imageio-ffmpeg tqdm wandb pyyaml

echo "--- pip: VLA stack (transformers, peft, accelerate, etc.) ---"
pip install -q "transformers>=4.45" accelerate peft safetensors einops timm sentencepiece

echo "--- versions ---"
python - <<'PY'
import importlib
for m in ["mujoco","mujoco_mjx","jax","flax","optax","brax","transformers","peft","accelerate","mediapy"]:
    try:
        mod=importlib.import_module(m); print(f"{m:14s}{getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m:14s}FAIL {e}")
import jax
print("jax devices:", jax.devices())
import torch
print("torch cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY

echo "VENV_PATH=$VENV"
echo "to use: source $VENV/bin/activate"
echo "===== bootstrap done $(date -u) ====="
