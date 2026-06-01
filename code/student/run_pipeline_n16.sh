#!/usr/bin/env bash
# Parameterized single-run of the full G1Nav N1.6 pipeline, for the interactive
# dashboard. Args:  $1 = seed (int)   $2 = instruction (free text)
#   1) build the integrated scene for the seed, render the ego frame
#   2) GR00T N1.6 Eagle grounds the instruction -> target label
#   3) navigate.py walks the trained PPO policy to that target (physics only),
#      recording ego||third-person side-by-side video with 2s pre/post-roll.
# Prints machine-readable result lines:  RESULT_JSON=...  VIDEO=...
set -uo pipefail
SEED="${1:-0}"
INSTR="${2:-go to the orange cylinder}"
cd /workspace
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EGV=/workspace/venv_groot   # transformers 4.51.3 (Eagle)
RLV=/workspace/venv         # jax/brax/mujoco_playground (policy + render)
export G1NAV_EAGLE=/workspace/ckpt/eagle-n16
CKPT=/workspace/ckpt/walk_t4_latest.pkl
EGO=/workspace/ckpt/ui_ego_seed${SEED}.png
RES=/workspace/ckpt/ui_cortex_seed${SEED}.json
VID=/workspace/ckpt/ui_nav_seed${SEED}.mp4

echo "STAGE=scene"
"$RLV/bin/python" - "$SEED" "$EGO" <<'PY' 2>/dev/null
import sys, os
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/workspace/code/student"); sys.path.insert(0, "/workspace/code/envs")
import mujoco
from PIL import Image
import integrated_scene as isc
seed = int(sys.argv[1]); out = sys.argv[2]
path, scene = isc.build_integrated_xml(seed)
m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 240, 320); r.update_scene(d, camera="ego")
Image.fromarray(r.render()).save(out)
open("/workspace/ckpt/ui_labels.txt", "w").write(",".join(o.label for o in scene.objects))
PY
LABELS=$(cat /workspace/ckpt/ui_labels.txt 2>/dev/null)
echo "OBJECTS=$LABELS"

echo "STAGE=cognition"
"$EGV/bin/python" /workspace/code/student/cortex_eagle.py \
  --image "$EGO" --instruction "$INSTR" --labels "$LABELS" --result "$RES" \
  2>&1 | grep -E "CORTEX_DONE" | head
TARGET=$("$RLV/bin/python" -c "import json;print(json.load(open('$RES'))['vlm_target'])" 2>/dev/null)
echo "TARGET=$TARGET"

echo "STAGE=navigate"
"$RLV/bin/python" /workspace/code/student/navigate.py \
  --ckpt "$CKPT" --seed "$SEED" --target "$TARGET" --seconds 14 --out "$VID" \
  2>&1 | grep -E "^NAV|FELL" | head

echo "RESULT_JSON=$RES"
echo "VIDEO=$VID"
echo "DONE"
