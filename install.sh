#!/usr/bin/env bash
# G1Nav — one-command installer.
#
#   bash install.sh            # auto: GPU detected -> full stack, else dashboard-only
#   bash install.sh gpu        # force the full GPU stack
#   bash install.sh dashboard  # force the lightweight dashboard-only stack
#
# Creates a virtualenv in .venv and installs the right requirements. Afterwards:
#   source .venv/bin/activate
#   bash run_dashboard.sh            # launch the dashboards
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-auto}"
if [ "$MODE" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    MODE="gpu"; echo "[install] NVIDIA GPU detected -> full GPU stack"
  else
    MODE="dashboard"; echo "[install] no GPU -> dashboard-only stack (pipeline runs on a remote pod)"
  fi
fi

PY="${PYTHON:-python3}"
echo "[install] python: $("$PY" --version 2>&1)"
if [ ! -d .venv ]; then
  echo "[install] creating .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip

if [ "$MODE" = "gpu" ]; then
  echo "[install] installing full GPU requirements (this can take a few minutes)…"
  pip install -q "jax[cuda12]==0.5.3" || pip install -q jax==0.5.3
  pip install -q -r requirements-gpu.txt
  echo "[install] done. GPU stack ready — the dashboard will run the pipeline LOCALLY."
else
  echo "[install] installing dashboard-only requirements…"
  pip install -q -r requirements-dashboard.txt
  echo "[install] done. Dashboard ready."
  echo "[install] NOTE: the interactive pipeline needs a GPU; set the pod via"
  echo "          G1NAV_POD_HOST / G1NAV_POD_PORT / G1NAV_POD_KEY to run remotely."
fi

cat <<'EOF'

────────────────────────────────────────────────────────────
 Next:
   source .venv/bin/activate
   bash run_dashboard.sh          # 8501 results  +  8502 interactive
 Then open  http://localhost:8502  (interactive)  /  http://localhost:8501 (results)
────────────────────────────────────────────────────────────
EOF
