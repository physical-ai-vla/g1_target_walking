#!/usr/bin/env bash
# Render >=3 closed-loop demo videos (ego || third-person), with the target
# chosen by the GR00T N1.6 Eagle cognition. For each seed:
#   1) build the integrated scene, render the ego frame at the home pose
#   2) cortex_eagle.py (N1.6 Eagle) grounds the instruction -> target label
#   3) navigate.py walks the trained PPO policy to that target, recording
#      ego || third-person side-by-side video (physics only).
# Run on the RunPod pod (GPU). Outputs /workspace/ckpt/nav_n16_seed*.mp4.
set -uo pipefail
cd /workspace
EGV=/workspace/venv_groot          # transformers 4.51.3  (Eagle cognition)
RLV=/workspace/venv                # jax/brax/mujoco_playground (policy + render)
export MUJOCO_GL=egl
CKPT=/workspace/ckpt/walk_t4_latest.pkl
export G1NAV_EAGLE=/workspace/ckpt/eagle-n16

SEEDS=(0 1 2)
for S in "${SEEDS[@]}"; do
  echo "=================== seed $S ==================="
  EGO=/workspace/ckpt/ego_seed${S}.png
  # 1) scene objects + ego frame at home pose
  "$RLV/bin/python" - "$S" "$EGO" <<'PY' 2>/dev/null
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
open("/workspace/ckpt/_labels.txt", "w").write(",".join(o.label for o in scene.objects))
PY
  LABELS=$(cat /workspace/ckpt/_labels.txt)
  echo "[seed $S] objects: $LABELS"
  TGT=$(echo "$LABELS" | cut -d, -f1)
  INSTRUCTION="go to the $TGT"
  echo "[seed $S] instruction: $INSTRUCTION"
  # 2) N1.6 Eagle cognition -> grounded target
  "$EGV/bin/python" /workspace/code/student/cortex_eagle.py \
    --image "$EGO" --instruction "$INSTRUCTION" --labels "$LABELS" \
    --result /workspace/ckpt/cortex_seed${S}.json 2>&1 | grep -E "CORTEX_DONE|Error|Traceback" | head
  NTGT=$("$RLV/bin/python" -c "import json;print(json.load(open('/workspace/ckpt/cortex_seed${S}.json'))['vlm_target'])")
  echo "[seed $S] N1.6 grounded target = $NTGT"
  # 3) closed-loop walk to that target -> ego||3rd video
  "$RLV/bin/python" /workspace/code/student/navigate.py \
    --ckpt "$CKPT" --seed "$S" --target "$NTGT" --seconds 12 \
    --out /workspace/ckpt/nav_n16_seed${S}.mp4 2>&1 | grep -E "^NAV|FELL|sim_dt|Error|Traceback" | head
done
echo "##### ALL DONE #####"
ls -la /workspace/ckpt/nav_n16_seed*.mp4 2>/dev/null
