#!/usr/bin/env bash
# Run the GR00T N1.6 Eagle cognition on an ego frame and record cortex_result.json.
# Detects the real objects in the frame (color/shape CV) → uses them as labels so
# the grounding is honest, then runs the N1.6 Eagle perception + keyword grounding.
set -uo pipefail
IMG="${1:-/workspace/ckpt/app_ego.png}"
cd /workspace

pick_venv() { for V in /workspace/venv_groot /workspace/venv; do
  "$V/bin/python" -c "import $1" 2>/dev/null && { echo "$V"; return; }; done; }

CVV=$(pick_venv cv2);  echo "[cog] cv2 venv      = ${CVV:-NONE}"
EGV=/workspace/venv_groot
echo "[cog] eagle venv    = $EGV"
"$EGV/bin/python" -c "import transformers,torch;print('[cog] transformers',transformers.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | head -1

# 1) detect the real objects present in the ego frame
"${CVV:-$EGV}/bin/python" - "$IMG" <<'PY' > /workspace/ckpt/cd_labels.txt 2>/tmp/cd.err
import sys
sys.path.insert(0, '/workspace/code/student')
try:
    import color_detect
    d = color_detect.detect(sys.argv[1])
    labels = []
    for o in d:
        if o['label'] not in labels:
            labels.append(o['label'])
    print(','.join(labels))
except Exception as e:
    sys.stderr.write(f'color_detect failed: {e}\n')
    print('')
PY
LABELS=$(cat /workspace/ckpt/cd_labels.txt)
[ -s /tmp/cd.err ] && cat /tmp/cd.err
echo "[cog] detected objects: ${LABELS:-<none>}"

TARGET=$(echo "$LABELS" | cut -d, -f1)
if [ -z "$TARGET" ]; then
  echo "[cog] color_detect found nothing -> fallback label set"
  LABELS="green cube,green ball,green cone"; TARGET="green cone"
fi
INSTR="go to the $TARGET"
echo "[cog] instruction   = $INSTR"
echo "[cog] labels        = $LABELS"

# 2) GR00T N1.6 Eagle cognition
echo "[cog] ===== running cortex_eagle.py (N1.6) ====="
G1NAV_EAGLE=/workspace/ckpt/eagle-n16 "$EGV/bin/python" \
  /workspace/code/student/cortex_eagle.py \
  --image "$IMG" --instruction "$INSTR" --labels "$LABELS" \
  --result /workspace/ckpt/cortex_result.json
RC=$?
echo "[cog] cortex_eagle rc=$RC"
echo "[cog] ===== cortex_result.json ====="
cat /workspace/ckpt/cortex_result.json 2>/dev/null || echo "(no result written)"
