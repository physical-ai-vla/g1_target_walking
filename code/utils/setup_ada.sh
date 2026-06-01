#!/usr/bin/env bash
# Lean setup for a fresh RunPod RTX 6000 Ada pod — WALKING TRAINING ONLY.
# Skips the 6.5GB GR00T download (VLM cognition is already done & verified).
# Same EXACT version pins + cuSolver-fix CUDA libs we hardened on H100.
#   bash /workspace/setup_ada.sh 2>&1 | tee /workspace/setup.log
set -u
export DEBIAN_FRONTEND=noninteractive
VENV=/workspace/venv

echo "===== [1/4] apt: render + tmux ====="
apt-get update -qq
apt-get install -y -qq libegl1 libgl1 libglew2.2 libosmesa6 ffmpeg python3-venv tmux >/dev/null 2>&1

echo "===== [2/4] venv (inherits system torch+cuda) ====="
[ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip

echo "===== [3/4] pinned RL + sim stack (the combo that WORKS) ====="
pip install -q \
  'playground==0.1.0' 'mujoco==3.4.0' 'mujoco-mjx==3.4.0' \
  'jax[cuda12]==0.5.3' 'jaxlib==0.5.3' 'brax==0.14.2' 'flax==0.10.6' \
  'optax==0.2.4'
# CUDA libs matched to jaxlib 0.5.3 (the combo that fixed the cuSolver INTERNAL
# error on Ampere). RTX 6000 Ada is Ada(sm_89) so cuSolver is likely fine, but
# pin for safety / reproducibility.
pip install -q \
  'nvidia-cublas-cu12==12.6.4.1' 'nvidia-cusolver-cu12==11.7.1.2' \
  'nvidia-cusparse-cu12==12.5.4.2' 'nvidia-cuda-runtime-cu12==12.6.77' \
  'nvidia-cuda-cupti-cu12==12.6.80' 'nvidia-cuda-nvrtc-cu12==12.6.77' \
  'nvidia-cudnn-cu12==9.5.1.17' 'nvidia-cufft-cu12==11.3.0.4' \
  'nvidia-curand-cu12==10.3.7.77'
pip install -q 'nvidia-nvjitlink-cu12==12.8.93'   # keep torch importable

echo "===== [4/4] render/util stack (for in-process rollout mp4) ====="
pip install -q scipy mediapy imageio imageio-ffmpeg tqdm

echo "===== verify (jax devices + cuSolver) ====="
python - <<PY
import jax, importlib.metadata as m
print("jax", jax.__version__, "devices", jax.devices())
import jax.scipy.linalg as sla, jax.numpy as jnp
print("cuSolver test:", sla.solve(jnp.eye(3)*2., jnp.array([1.,2.,3.])))
for p in ["brax","mujoco","mujoco-mjx","playground"]:
    print(p, m.version(p))
PY
echo "===== SETUP DONE ====="
