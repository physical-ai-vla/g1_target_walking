#!/usr/bin/env bash
# Launch the G1Nav dashboards.
#   8501  app_results.py     — replay recorded results (always works offline)
#   8502  app_interactive.py — drive the pipeline yourself (local GPU or remote pod)
#
#   bash run_dashboard.sh            # both
#   bash run_dashboard.sh interactive   # only 8502
#   bash run_dashboard.sh results       # only 8501
set -uo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate 2>/dev/null || true

WHICH="${1:-both}"
COMMON="--server.headless true --browser.gatherUsageStats false"

if [ "$WHICH" = "both" ] || [ "$WHICH" = "results" ]; then
  echo "[dash] results dashboard  -> http://localhost:8501"
  streamlit run code/student/app_results.py --server.port 8501 $COMMON \
    > /tmp/g1nav_results.log 2>&1 &
fi
if [ "$WHICH" = "both" ] || [ "$WHICH" = "interactive" ]; then
  echo "[dash] interactive (run it yourself) -> http://localhost:8502"
  streamlit run code/student/app_interactive.py --server.port 8502 $COMMON \
    > /tmp/g1nav_interactive.log 2>&1 &
fi

echo "[dash] launched. Logs: /tmp/g1nav_results.log  /tmp/g1nav_interactive.log"
echo "[dash] stop with:  pkill -f 'streamlit run'"
wait
