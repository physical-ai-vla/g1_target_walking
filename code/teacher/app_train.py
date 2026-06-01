"""Streamlit dashboard for the PPO walking training.

Shows:
  - live training curve (reward vs step) parsed from *_metrics.jsonl
  - latest checkpoint info (step) and a button to roll out the CURRENT policy
    into a short video so you can SEE how it walks right now
  - the most recent rollout video + survival/distance stats

Run on the pod (separate port from the brain dashboard), e.g. 8080 if free:
    source /workspace/venv/bin/activate && cd /workspace/g1nav
    streamlit run code/teacher/app_train.py --server.port 8080 \
        --server.address 0.0.0.0 --server.headless true --server.fileWatcherType none
"""
import json
import os
import subprocess
import sys

import streamlit as st

st.set_page_config(page_title="G1Nav · PPO Training", layout="wide")

CKPT = os.environ.get("WALK_OUT", "/workspace/ckpt/walk_full")
METRICS = CKPT + "_metrics.jsonl"
LATEST = CKPT + "_latest.pkl"
ROLLOUT_MP4 = "/workspace/ckpt/rollout_dash.mp4"


def load_metrics():
    rows = []
    if os.path.exists(METRICS):
        for line in open(METRICS):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def main():
    st.title("🦿 G1Nav — PPO Walking Training")
    st.caption("Low-level walking policy (the 'spinal cord'). Trained ONCE, reused "
               "for every instruction. Cognition is separate (zero-shot VLM).")

    rows = load_metrics()
    c1, c2, c3 = st.columns(3)
    if rows:
        last = rows[-1]
        c1.metric("Latest step", f"{last['step']:,}")
        c2.metric("Latest reward", f"{last['reward']:.2f}")
        c3.metric("Elapsed", f"{last['elapsed_min']:.1f} min")
    else:
        st.info("No metrics yet — training is still compiling / starting.")

    # ---- training curve ----
    if rows:
        st.subheader("Training curve")
        import numpy as np
        steps = [r["step"] for r in rows]
        rew = [r["reward"] for r in rows]
        try:
            import pandas as pd
            df = pd.DataFrame({"step": steps, "reward": rew}).set_index("step")
            st.line_chart(df)
        except Exception:
            st.line_chart({"reward": rew})

    # ---- checkpoint rollout ----
    st.divider()
    st.subheader("Watch the current policy walk")
    if os.path.exists(LATEST):
        try:
            import pickle
            step = pickle.load(open(LATEST, "rb")).get("step")
            st.write(f"Latest checkpoint at **step {step:,}**")
        except Exception:
            st.write("Latest checkpoint present.")
        cols = st.columns(4)
        vx = cols[0].number_input("vx (m/s)", 0.0, 1.0, 0.5, 0.1)
        wz = cols[1].number_input("yaw rate", -1.0, 1.0, 0.0, 0.1)
        secs = cols[2].number_input("seconds", 2.0, 10.0, 6.0, 1.0)
        if cols[3].button("▶ Roll out", type="primary"):
            with st.spinner("Rolling out current policy (renders video)…"):
                cmd = [sys.executable,
                       "/workspace/g1nav/code/teacher/rollout_walk.py",
                       "--ckpt", LATEST, "--vx", str(vx), "--wz", str(wz),
                       "--seconds", str(secs), "--out", ROLLOUT_MP4]
                # run rollout on CPU so it doesn't fight the GPU-bound PPO trainer
                env = dict(os.environ, MUJOCO_GL="egl", JAX_PLATFORMS="cpu")
                r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                   timeout=600)
            st.code((r.stdout + "\n" + r.stderr)[-1500:])
        if os.path.exists(ROLLOUT_MP4):
            st.video(ROLLOUT_MP4)
    else:
        st.info("No checkpoint saved yet — appears after the first eval "
                "(give it a few minutes after step 0).")

    st.caption("Refresh the page to update the curve / metrics.")


if __name__ == "__main__":
    main()
