"""Portable one-shot G1Nav pipeline: seed + instruction -> video.

Runs the full closed loop with REPO-RELATIVE paths so it works in anyone's
environment (no hard-coded /workspace). Three stages:

    1) build the integrated scene for the seed, render the ego frame
    2) GR00T N1.6 Eagle cognition grounds the instruction -> target label
    3) navigate.py walks the trained PPO policy to that target (physics only),
       recording an ego||third-person video with 2 s pre/post-roll.

Usage:
    python code/student/g1nav_run.py --seed 0 --instruction "go to the orange cylinder"

Paths (override via env or flags):
    --ckpt    default: checkpoint/walk_t4_latest.pkl
    --eagle   default: $G1NAV_EAGLE or checkpoint/eagle-n16   (GR00T N1.6 Eagle encoder)
    --outdir  default: outputs/

Prints machine-readable lines:  OBJECTS=...  TARGET=...  NAV ...  VIDEO=<path>  DONE
so a UI (app_interactive.py) can parse the result. Cognition is skipped
gracefully (falls back to keyword match) if the Eagle encoder isn't present.
"""
import os
import re
import sys
import json
import argparse
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STU = os.path.join(ROOT, "code", "student")
ENV = os.path.join(ROOT, "code", "envs")
for p in (STU, ENV):
    if p not in sys.path:
        sys.path.insert(0, p)


def render_scene(seed, ego_path):
    """Build the integrated scene and render the start ego frame. Returns labels."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image
    import integrated_scene as isc
    path, scene = isc.build_integrated_xml(seed)
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 240, 320)
    r.update_scene(d, camera="ego")
    Image.fromarray(r.render()).save(ego_path)
    return [o.label for o in scene.objects]


def ground_target(ego_path, instruction, labels, eagle_dir, result_path):
    """GR00T N1.6 Eagle cognition. Falls back to keyword match if Eagle absent."""
    have_eagle = eagle_dir and os.path.isdir(eagle_dir)
    if have_eagle:
        env = dict(os.environ, G1NAV_EAGLE=eagle_dir)
        cmd = [sys.executable, os.path.join(STU, "cortex_eagle.py"),
               "--image", ego_path, "--instruction", instruction,
               "--labels", ",".join(labels), "--result", result_path]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if os.path.exists(result_path):
            try:
                return json.load(open(result_path)).get("vlm_target"), r.stdout + r.stderr
            except json.JSONDecodeError:
                pass
    # fallback: color+shape keyword overlap with the instruction (no extra ckpt)
    instr = instruction.lower()
    best = max(labels, key=lambda l: sum(w in instr for w in l.lower().split()),
               default=labels[0] if labels else None)
    json.dump({"vlm_target": best, "labels": labels,
               "note": "Eagle encoder not found; used keyword grounding"},
              open(result_path, "w"), indent=2)
    return best, "eagle encoder not found -> keyword grounding"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruction", default="go to the orange cylinder")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoint", "walk_t4_latest.pkl"))
    ap.add_argument("--eagle", default=os.environ.get(
        "G1NAV_EAGLE", os.path.join(ROOT, "checkpoint", "eagle-n16")))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--seconds", type=float, default=14.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ego = os.path.join(args.outdir, f"ego_seed{args.seed}.png")
    res = os.path.join(args.outdir, f"cortex_seed{args.seed}.json")
    vid = os.path.join(args.outdir, f"nav_seed{args.seed}.mp4")

    print("STAGE=scene", flush=True)
    labels = render_scene(args.seed, ego)
    print("OBJECTS=" + ",".join(labels), flush=True)

    print("STAGE=cognition", flush=True)
    target, _ = ground_target(ego, args.instruction, labels, args.eagle, res)
    print("TARGET=" + (target or "?"), flush=True)

    print("STAGE=navigate", flush=True)
    cmd = [sys.executable, os.path.join(STU, "navigate.py"),
           "--ckpt", args.ckpt, "--seed", str(args.seed),
           "--target", target or labels[0], "--seconds", str(args.seconds),
           "--out", vid]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("NAV ") or line.startswith("FELL"):
            print(line, flush=True)

    print("RESULT_JSON=" + res, flush=True)
    print("VIDEO=" + vid, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
