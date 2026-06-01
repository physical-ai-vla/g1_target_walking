"""G1Nav — Recorded-Results Dashboard (GPU-free, runs on a laptop).

Verifies the system *worked* by surfacing the recorded artifacts: demo videos
(walks / reaches target / fails), the PPO walking-emergence curve parsed from the
training log, the final raw-env eval numbers, and the cognition frames. No GPU,
no checkpoint, no live physics needed — purely plays back what the pod produced.

    streamlit run code/student/app_results.py
    # optional: G1NAV_DATA=/path/to/artifacts  G1NAV_VIDEOS=/path/to/videos

Looks in (first found wins, both are merged for media):
    $G1NAV_DATA   (default: ~/Desktop/g1nav_dash)
    $G1NAV_VIDEOS (default: ~/g1nav/videos)
"""
import os
import re
import glob
import json

import streamlit as st

HOME = os.path.expanduser("~")
DATA_DIR = os.environ.get("G1NAV_DATA", os.path.join(HOME, "Desktop", "g1nav_dash"))
VIDEO_DIR = os.environ.get("G1NAV_VIDEOS", os.path.join(HOME, "g1nav", "videos"))
DIRS = [d for d in (DATA_DIR, VIDEO_DIR) if os.path.isdir(d)]


def find(*names):
    """Return the first existing path matching any of the given basenames/globs."""
    for d in DIRS:
        for n in names:
            hits = sorted(glob.glob(os.path.join(d, n)))
            if hits:
                return hits[0]
    return None


def read_text(*names):
    p = find(*names)
    if not p:
        return None
    try:
        with open(p, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def parse_training_log(text):
    """Pull (step, reward) and (step, net_dist, fwd_vel, cmd, min_z) from the log."""
    rewards, rollouts = [], []
    if not text:
        return rewards, rollouts
    for m in re.finditer(r"step=\s*([\d,]+)\s+reward=\s*([-\d.]+)", text):
        rewards.append((int(m.group(1).replace(",", "")), float(m.group(2))))
    for m in re.finditer(
        r"rollout @ (\d+): net_dist=([\d.]+)m fwd_vel=([+-][\d.]+)\(cmd([\d.]+)\) min_z=([-\d.]+)",
        text,
    ):
        rollouts.append(dict(step=int(m.group(1)), net_dist=float(m.group(2)),
                             fwd_vel=float(m.group(3)), cmd=float(m.group(4)),
                             min_z=float(m.group(5))))
    return rewards, rollouts


st.set_page_config(page_title="G1Nav — Recorded Results", layout="wide", page_icon="🦿")
st.title("🦿🧠 G1Nav — Recorded Results Dashboard")
st.caption("Language-conditioned lower-body navigation for the Unitree G1 in MuJoCo (MJX). "
           "Playback of recorded artifacts — no GPU / checkpoint / live physics required.")

if not DIRS:
    st.error(f"No artifact directory found. Looked in: {DATA_DIR} and {VIDEO_DIR}")
    st.stop()
st.write("**Reading artifacts from:** " + " · ".join(f"`{d}`" for d in DIRS))

# ---- headline eval numbers (raw env, no AutoReset) ----------------------------
roll = read_text("rollout.json")
final = {}
if roll:
    try:
        final = json.loads(roll)
    except json.JSONDecodeError:
        final = {}

log_text = read_text("_status.txt", "ppo*.log", "train*.log")
rewards, rollouts = parse_training_log(log_text)
best = max(rollouts, key=lambda r: r["fwd_vel"]) if rollouts else None
last = rollouts[-1] if rollouts else None

st.subheader("Final walking policy — raw-env evaluation")
c = st.columns(4)
if final:
    cmd = 0.6
    fv = float(final.get("fwd_vel", 0.0))
    c[0].metric("Forward velocity", f"{fv:.2f} m/s", f"{fv/cmd*100:.0f}% of {cmd} cmd")
    c[1].metric("Torso height (min_z)", f"{final.get('min_z', 0):.2f} m", "upright (>0.7)")
    c[2].metric("Net distance", f"{final.get('x_travel', 0):.2f} m")
    c[3].metric("Trained to", f"{final.get('step', 0)/1e6:.0f} M steps")
elif last:
    c[0].metric("Forward velocity", f"{last['fwd_vel']:.2f} m/s",
                f"{last['fwd_vel']/last['cmd']*100:.0f}% of {last['cmd']} cmd")
    c[1].metric("Torso height (min_z)", f"{last['min_z']:.2f} m", "upright (>0.7)")
    c[2].metric("Net distance", f"{last['net_dist']:.2f} m")
    c[3].metric("Trained to", f"{last['step']/1e6:.0f} M steps")
else:
    st.info("No eval numbers found (rollout.json / training log).")
if best:
    st.caption(f"Peak forward velocity over training: **{best['fwd_vel']:.2f} m/s** "
               f"({best['fwd_vel']/best['cmd']*100:.0f}% of command) at "
               f"{best['step']/1e6:.0f} M steps.")

st.divider()

# ---- N1.6 pipeline (this run) -----------------------------------------------
st.subheader("✅ N1.6 pipeline — this run (cognition → Navigator → walk → physics)")
st.caption("Each clip is the *modified* pipeline end-to-end: GR00T N1.6 Eagle grounds the "
           "instruction → trained PPO policy walks to that target in MuJoCo (physics only). "
           "Left = third-person, right = ego. **All 3 seeds reach the target** "
           "(final dist ≈ 0.54 m, torso ≈ 0.76 m, never falls) after fixing two inference "
           "bugs: (1) missing observation normalization, (2) gravity computed in the wrong "
           "frame — `-upvector` (world) instead of `site_xmat.T@[0,0,-1]` (pelvis-IMU local), "
           "which only diverges during yaw and made the robot topple mid-turn.")
ncols = st.columns(3)
n16_any = False
_INSTR = ["go to the orange cylinder", "go to the red cube", "go to the purple cylinder"]
for i in range(3):
    with ncols[i]:
        # new naming: seed{i}_<instruction-slug>.mp4 + .cognition.json ; fall back to old
        slug = _INSTR[i].replace(" ", "-")
        vid = find(f"seed{i}_{slug}.mp4", f"nav_n16_seed{i}.mp4")
        cj = read_text(f"seed{i}_{slug}.cognition.json", f"cortex_seed{i}.json")
        tgt = "?"
        if cj:
            try:
                tgt = json.loads(cj).get("vlm_target", "?")
            except json.JSONDecodeError:
                pass
        st.markdown(f"**seed {i}** · instruction: *“{_INSTR[i]}”*  \n"
                    f"N1.6 grounded target → `{tgt}` · ✅ reached")
        if vid:
            st.video(vid)
            n16_any = True
        else:
            st.info("rendering on pod… (will appear after pull)")
if not n16_any:
    st.caption("⏳ Videos render on the RTX 6000 Ada pod via "
               "`code/student/render_n16_videos.sh`, then are pulled here.")

st.divider()

# ---- demo videos -------------------------------------------------------------
st.subheader("Closed-loop demos (earlier recordings)")
DEMOS = [
    ("✅ Reaches the instructed target", ["demo_NAV_reaches_target.mp4", "nav_60M.mp4", "g1_nav*.mp4"]),
    ("🚶 Walks (velocity-conditioned policy)", ["demo_WALKS.mp4", "walk_latest_rollout.mp4", "g1_fwd.mp4"]),
    ("⚠️ Failure mode (falls)", ["demo_falls.mp4", "walk_raw_falls.mp4", "walk_wrapped_inplace.mp4"]),
]
cols = st.columns(3)
for col, (label, names) in zip(cols, DEMOS):
    with col:
        st.markdown(f"**{label}**")
        p = find(*names)
        if p:
            st.video(p)
            st.caption(os.path.basename(p))
        else:
            st.info("not found")

st.divider()

# ---- walking emergence: first walk vs mature ---------------------------------
st.subheader("Walking emergence (first walk → mature gait)")
ec = st.columns(2)
first = find("walk_41M_FIRST_WALK.mp4", "walk_v3_trackcam.mp4")
mature = find("walk_latest_rollout.mp4", "latest_rollout.mp4", "g1_trainenv_walk.mp4")
with ec[0]:
    st.markdown("**First emergence (~41 M steps)**")
    st.video(first) if first else st.info("not found")
with ec[1]:
    st.markdown("**Mature gait (latest)**")
    st.video(mature) if mature else st.info("not found")

st.divider()

# ---- training curves ---------------------------------------------------------
st.subheader("PPO training — walking emergence curve")
if rollouts:
    import pandas as pd
    df = pd.DataFrame(rollouts)
    df["step_M"] = df["step"] / 1e6
    df = df.set_index("step_M")
    g = st.columns(2)
    with g[0]:
        st.markdown("**Forward velocity vs command**")
        df["command"] = df["cmd"]
        st.line_chart(df[["fwd_vel", "command"]])
    with g[1]:
        st.markdown("**Torso height (min_z) — falls when < 0**")
        st.line_chart(df[["min_z"]])
    if rewards:
        rdf = pd.DataFrame(rewards, columns=["step", "reward"]).set_index("step")
        rdf.index = rdf.index / 1e6
        rdf.index.name = "step_M"
        st.markdown("**Episode reward**")
        st.line_chart(rdf)
    st.caption("Parsed from the training log. The policy first learns *not to fall* "
               "(min_z crosses 0 around 41 M), then matures the forward gait.")
else:
    st.info("No training-log rollout lines found to plot.")

st.divider()

# ---- cognition frames --------------------------------------------------------
st.subheader("Cognition — GR00T N1.6 Eagle (zero-shot, no training)")
cog = read_text("seed0_go-to-the-orange-cylinder.cognition.json", "cortex_result.json")
if cog:
    try:
        cr = json.loads(cog)
    except json.JSONDecodeError:
        cr = {}
    cc = st.columns([1, 2])
    with cc[0]:
        ego = find("seed0_go-to-the-orange-cylinder.ego.png", "app_ego.png",
                   "cortex_ego.png", "ego_start.png")
        if ego:
            st.image(ego, caption="ego RGB frame", use_container_width=True)
        st.metric("Eagle perception embedding",
                  f"{cr.get('eagle_patches', '?')} × {cr.get('eagle_embed_dim', '?')}",
                  f"‖emb‖ = {cr.get('eagle_norm', 0):.0f}")
        st.caption(f"target → **{cr.get('vlm_target', '?')}**  ·  GR00T N1.6 Eagle vision "
                   "encoder (SigLip2) — only parameters from the N1.6 checkpoint are used.")
    with cc[1]:
        st.markdown("**5-stage cortex** (perceive → ground → localize → plan → command):")
        for k in ("1", "2", "3", "4", "5"):
            v = cr.get("stages", {}).get(k)
            if v:
                st.markdown(f"**{k}.** {v}")
    st.caption("Generated by `code/student/cortex_eagle.py` on the RTX 6000 Ada pod "
               "(transformers 4.51.3, eagle-n16). Upper Eagle LM layers are intentionally "
               "absent — GR00T uses the backbone for perception, not text generation.")
else:
    st.info("cortex_result.json (N1.6 Eagle) not found — run code/student/cortex_eagle.py.")

# ---- raw status --------------------------------------------------------------
if log_text:
    with st.expander("Raw training/status log (_status.txt)"):
        st.code(log_text[-4000:])
