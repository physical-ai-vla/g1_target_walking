#!/usr/bin/env bash
# Fast scene preview for the interactive dashboard: render the ego frame and list
# the objects for a seed. No GPU cognition, just a MuJoCo render — returns quickly
# so the user can SEE the scene before typing an instruction.
# Arg: $1 = seed (int). Prints:  OBJECTS=...  EGO=<path>  DONE
set -uo pipefail
SEED="${1:-0}"
cd /workspace
export MUJOCO_GL=egl
RLV=/workspace/venv
EGO=/workspace/ckpt/ui_ego_seed${SEED}.png

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
echo "OBJECTS=$(cat /workspace/ckpt/ui_labels.txt 2>/dev/null)"
echo "EGO=$EGO"
echo "DONE"
