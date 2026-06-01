"""G1Nav — Interactive dashboard (2-step: see the scene, then instruct).

Step 1: pick a scene seed and **Load scene** → renders the ego frame and lists
        the objects in that arena, so you know what you can ask for.
Step 2: type an instruction referring to one of those objects and **Run** → the
        full pipeline runs:
            instruction + ego frame
              -> GR00T N1.6 Eagle cognition (grounds the target)
              -> Navigator -> trained PPO walking policy -> MuJoCo physics
              -> ego || third-person video (physics only)

Two execution modes, auto-detected (override with G1NAV_MODE=local|remote):
  • LOCAL  — if a GPU + the sim deps + the checkpoint are present, it runs the
             pipeline right here (no network). Best for a machine with a GPU.
  • REMOTE — otherwise it runs on an SSH-reachable GPU pod and pulls the video
             back. Configure with G1NAV_POD_HOST / G1NAV_POD_PORT / G1NAV_POD_KEY.

Launch:
    streamlit run code/student/app_interactive.py --server.port 8502
"""
import os
import re
import json
import time
import shlex
import tempfile
import subprocess

import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNNER = os.path.join(ROOT, "code", "student", "g1nav_run.py")

# remote config (only used in REMOTE mode)
HOST = os.environ.get("G1NAV_POD_HOST", "root@195.26.233.54")
PORT = os.environ.get("G1NAV_POD_PORT", "33566")
KEY = os.path.expanduser(os.environ.get("G1NAV_POD_KEY", "~/.ssh/id_ed25519"))
PREVIEW_REMOTE = "/workspace/code/student/preview_scene.sh"
RUN_REMOTE = "/workspace/code/student/run_pipeline_n16.sh"
PREVIEW_LOCAL_SH = os.path.join(ROOT, "code", "student", "preview_scene.sh")
RUN_LOCAL_SH = os.path.join(ROOT, "code", "student", "run_pipeline_n16.sh")
SSH = ["ssh", "-p", PORT, "-i", KEY, "-o", "StrictHostKeyChecking=no",
       "-o", "ConnectTimeout=20", "-o", "BatchMode=yes"]
SCP = ["scp", "-P", PORT, "-i", KEY, "-o", "StrictHostKeyChecking=no",
       "-o", "ConnectTimeout=20"]


def detect_mode():
    forced = os.environ.get("G1NAV_MODE")
    if forced in ("local", "remote"):
        return forced
    # local if sim deps import AND a GPU is visible AND the checkpoint exists
    try:
        import jax  # noqa: F401
        import mujoco_playground  # noqa: F401
        import jax as _jax
        has_gpu = any(d.platform == "gpu" for d in _jax.devices())
    except Exception:  # noqa: BLE001
        has_gpu = False
    has_ckpt = os.path.exists(os.path.join(ROOT, "checkpoint", "walk_t4_latest.pkl"))
    return "local" if (has_gpu and has_ckpt) else "remote"


MODE = detect_mode()


def grab(out, key):
    m = re.search(rf"{key}=(.+)", out)
    return m.group(1).strip() if m else None


# ───────────────────────────── LOCAL backend ──────────────────────────────
def local_preview(seed):
    import sys
    sys.path.insert(0, os.path.join(ROOT, "code", "student"))
    sys.path.insert(0, os.path.join(ROOT, "code", "envs"))
    import g1nav_run
    outdir = os.path.join(ROOT, "outputs")
    os.makedirs(outdir, exist_ok=True)
    ego = os.path.join(outdir, f"ego_seed{seed}.png")
    labels = g1nav_run.render_scene(int(seed), ego)
    return ",".join(labels), (ego if os.path.exists(ego) else None)


def local_run(seed, instruction):
    cmd = ["python", RUNNER, "--seed", str(int(seed)), "--instruction", instruction]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=900)
    out = r.stdout + "\n" + r.stderr
    vid = grab(out, "VIDEO")
    rj = grab(out, "RESULT_JSON")
    return out, (vid if vid and os.path.exists(vid) else None), \
        (rj if rj and os.path.exists(rj) else None)


# ───────────────────────────── REMOTE backend ─────────────────────────────
def ssh_run(cmd, timeout=900):
    return subprocess.run(SSH + [HOST, cmd], capture_output=True, text=True, timeout=timeout)


def push(local, remote):
    subprocess.run(SSH + [HOST, "mkdir -p /workspace/code/student"],
                   capture_output=True, text=True, timeout=30)
    subprocess.run(SCP + [local, f"{HOST}:{remote}"], capture_output=True, text=True, timeout=60)


def scp_from(remote, local, timeout=120):
    subprocess.run(SCP + [f"{HOST}:{remote}", local], capture_output=True, text=True, timeout=timeout)


def remote_preview(seed):
    push(PREVIEW_LOCAL_SH, PREVIEW_REMOTE)
    r = ssh_run(f"bash {PREVIEW_REMOTE} {int(seed)}", timeout=120)
    out = r.stdout + "\n" + r.stderr
    objects = grab(out, "OBJECTS")
    remote_ego = grab(out, "EGO")
    local_ego = None
    if remote_ego:
        tmp = tempfile.mkdtemp(prefix="g1nav_prev_")
        local_ego = os.path.join(tmp, os.path.basename(remote_ego))
        scp_from(remote_ego, local_ego, 60)
        if not os.path.exists(local_ego):
            local_ego = None
    return objects, local_ego


def remote_run(seed, instruction):
    push(RUN_LOCAL_SH, RUN_REMOTE)
    r = ssh_run(f"bash {RUN_REMOTE} {int(seed)} {shlex.quote(instruction)}", timeout=900)
    out = r.stdout + "\n" + r.stderr
    remote_vid = grab(out, "VIDEO")
    remote_json = grab(out, "RESULT_JSON")
    local_vid = local_json = None
    if remote_vid:
        tmp = tempfile.mkdtemp(prefix="g1nav_ui_")
        local_vid = os.path.join(tmp, os.path.basename(remote_vid))
        scp_from(remote_vid, local_vid, 120)
        if remote_json:
            local_json = os.path.join(tmp, os.path.basename(remote_json))
            scp_from(remote_json, local_json, 60)
    return out, (local_vid if local_vid and os.path.exists(local_vid) else None), \
        (local_json if local_json and os.path.exists(local_json) else None)


# ───────────────────────────────── UI ─────────────────────────────────────
st.set_page_config(page_title="G1Nav — Interactive", layout="wide", page_icon="🦿")
st.title("🦿🧠 G1Nav — Interactive (run it yourself)")
st.caption("1) Load a scene to see its objects.  2) Tell the robot which one to go to.  "
           "GR00T N1.6 Eagle grounds the target → the trained PPO policy walks there in "
           "MuJoCo (physics only).")

ss = st.session_state
ss.setdefault("scene_seed", None)
ss.setdefault("objects", None)
ss.setdefault("ego_path", None)

with st.sidebar:
    st.header("Execution")
    st.metric("Mode", MODE.upper())
    if MODE == "local":
        st.caption("Running the pipeline on THIS machine's GPU.")
    else:
        st.caption(f"Running on remote GPU pod `{HOST}` (this machine has no usable GPU).")
        if st.button("Check pod"):
            try:
                r = ssh_run("echo OK; nvidia-smi --query-gpu=memory.free --format=csv,noheader", 25)
                ok = "OK" in r.stdout
                (st.success if ok else st.error)((r.stdout + r.stderr).strip())
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    st.caption("Override with env var G1NAV_MODE=local|remote.")

st.subheader("Step 1 — choose a scene")
c1, c2 = st.columns([1, 1])
with c1:
    seed = st.number_input("Scene seed", min_value=0, max_value=9999, value=0, step=1,
                           help="Each seed is a deterministic 3-object arena.")
with c2:
    st.write(""); st.write("")
    load = st.button("🔄 Load scene", type="secondary")

if load:
    with st.status("Rendering the scene…", expanded=False) as s:
        try:
            objects, ego = (local_preview if MODE == "local" else remote_preview)(int(seed))
            ss.scene_seed = int(seed)
            ss.objects = objects
            ss.ego_path = ego
            s.update(label="Scene loaded", state="complete")
            if not objects:
                st.error("Could not read scene objects.")
        except Exception as e:  # noqa: BLE001
            s.update(label="Error", state="error")
            st.exception(e)

if ss.objects:
    st.success(f"**Scene seed {ss.scene_seed}** — objects in this arena:")
    cc = st.columns([1, 1])
    with cc[0]:
        if ss.ego_path:
            st.image(ss.ego_path, caption="ego view (robot's start view)", use_container_width=True)
    with cc[1]:
        for o in [x.strip() for x in ss.objects.split(",") if x.strip()]:
            st.markdown(f"- **{o}**")
        st.caption("Type an instruction below referring to one of these.")

st.divider()
st.subheader("Step 2 — give an instruction")
ready = ss.objects is not None
default_instr = ""
if ready:
    first = [x.strip() for x in ss.objects.split(",") if x.strip()]
    default_instr = f"go to the {first[0]}" if first else ""
instruction = st.text_input("Instruction", value=default_instr, disabled=not ready,
                            placeholder="Load a scene first (Step 1)…",
                            help="e.g. 'go to the red cube', 'walk to the purple cylinder'")
run = st.button("▶ Run pipeline", type="primary", disabled=not ready)

if run and ready:
    with st.status("Running the full pipeline… (~1–2 min: model load + rollout)",
                   expanded=True) as status:
        try:
            t0 = time.time()
            out, vid, rj = (local_run if MODE == "local" else remote_run)(
                int(ss.scene_seed), instruction)
            dt = time.time() - t0
            target = grab(out, "TARGET")
            navline = next((l for l in out.splitlines() if l.startswith("NAV ")), "")
            fell = "FELL" in out
            status.update(label=f"Done in {dt:.0f}s", state="complete", expanded=False)

            st.subheader("Result")
            r1, r2 = st.columns([3, 2])
            with r1:
                if vid and os.path.getsize(vid) > 0:
                    st.video(vid)
                    st.caption("Left = third-person · Right = ego.  2s stand → walk → 2s stand.")
                else:
                    st.error("No video came back — see raw output.")
            with r2:
                (st.error if fell else st.success)(
                    "⚠️ The robot FELL before reaching the target." if fell
                    else "✅ Reached the target (stayed upright).")
                st.metric("You asked", instruction)
                st.metric("N1.6 grounded target", target or "?")
                if navline:
                    st.code(navline, language="text")
                if rj:
                    with st.expander("Cognition (cortex_result.json)"):
                        st.json(json.load(open(rj)))
            with st.expander("Raw output"):
                st.code(out[-6000:], language="text")
        except subprocess.TimeoutExpired:
            status.update(label="Timed out", state="error")
            st.error("The run timed out. Try again, or check the pod (sidebar).")
        except Exception as e:  # noqa: BLE001
            status.update(label="Error", state="error")
            st.exception(e)
elif not ready:
    st.info("⬆️ Load a scene in Step 1 first — then you'll see its objects and can instruct.")
