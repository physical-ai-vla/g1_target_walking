"""Canonical walking eval — UNIFIED with the in-training rollout (save_ckpt).

Same setup as train_walk.py's save_ckpt: RAW registry env (NO wrap_for_brax_training
=> NO AutoResetWrapper, so falls are real, never teleported back to start). Command
is in the robot's LOCAL/heading frame, so we report LOCAL forward velocity (what
tracking_lin_vel rewards) + net horizontal displacement, NOT just world-X.

Usage:
    # single checkpoint:
    MUJOCO_GL=egl python walk_eval.py /workspace/ckpt/walk_t4_latest.pkl
    # full sweep of every numbered checkpoint (전수 조사):
    MUJOCO_GL=egl python walk_eval.py --sweep /workspace/ckpt 'walk_t4_step*.pkl'
"""
import os, sys, glob, json
os.environ.setdefault("MUJOCO_GL", "egl")
import jax, jax.numpy as jp, numpy as np, pickle, imageio, mujoco
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

ENV = "G1JoystickFlatTerrain"
VX = 0.6
N = 250  # < 500 so the env never resamples the command

# NOTE: match whatever reward config the run used (only affects env build, not
# the policy weights). The current run uses OFFICIAL defaults, so don't override.
cfg = registry.get_default_config(ENV)
env = registry.load(ENV, config=cfg)          # RAW env — no wrappers, no auto-reset
ppo_cfg = locomotion_params.brax_ppo_config(ENV)
net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                     preprocess_observations_fn=running_statistics.normalize,
                                     **dict(ppo_cfg.network_factory))
make_policy = ppo_networks.make_inference_fn(net)
jr = jax.jit(env.reset); js = jax.jit(env.step)
mjm = env.mj_model
lv_adr = mjm.sensor_adr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SENSOR, "local_linvel_pelvis")]
CMD = jp.array([VX, 0.0, 0.0])

def evaluate(ckpt, render_path=None):
    with open(ckpt, "rb") as f:
        o = pickle.load(f)
    params = o["params"] if isinstance(o, dict) and "params" in o else o
    step = o.get("step") if isinstance(o, dict) else None
    pol = jax.jit(make_policy(params, deterministic=True))
    st = jr(jax.random.PRNGKey(0)); st.info["command"] = CMD
    xs, ys, zs, fwd, frames = [], [], [], [], []
    if render_path:
        rr = mujoco.Renderer(mjm, 288, 384); mjd = mujoco.MjData(mjm)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 4.0; cam.elevation = -12; cam.azimuth = 135
    for i in range(N):
        st.info["command"] = CMD
        act, _ = pol(st.obs, jax.random.PRNGKey(0)); st = js(st, act)
        q = np.array(st.data.qpos).reshape(-1); sd = np.array(st.data.sensordata).reshape(-1)
        xs.append(float(q[0])); ys.append(float(q[1])); zs.append(float(q[2])); fwd.append(float(sd[lv_adr]))
        if render_path:
            mjd.qpos[:] = q; mjd.qvel[:] = np.array(st.data.qvel).reshape(-1)
            mujoco.mj_forward(mjm, mjd); cam.lookat[:] = [q[0], q[1], 0.5]
            rr.update_scene(mjd, camera=cam); frames.append(rr.render())
    if render_path:
        imageio.mimsave(render_path, frames, fps=50); rr.close()
    dx, dy = xs[-1]-xs[0], ys[-1]-ys[0]
    net = float(np.hypot(dx, dy)); mfwd = float(np.mean(fwd)); mz = float(min(zs))
    v = "WALKS" if (mfwd > 0.3 and mz > 0.4) else ("STEPS/STUCK" if mz > 0.4 else "FALLS")
    return {"ckpt": os.path.basename(ckpt), "step": step, "net_dist": round(net, 2),
            "fwd_vel": round(mfwd, 3), "min_z": round(mz, 2), "verdict": v}

if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
    d, pat = sys.argv[2], sys.argv[3]
    ckpts = sorted(glob.glob(os.path.join(d, pat)))
    print(f"sweeping {len(ckpts)} checkpoints (cmd_vx={VX}, raw env, {N} steps)")
    rows = []
    for c in ckpts:
        r = evaluate(c); rows.append(r)
        print(f"  {r['step'] if r['step'] is not None else '?':>11}  net={r['net_dist']:5.2f}m  "
              f"fwd_vel={r['fwd_vel']:+.2f}  min_z={r['min_z']:5.2f}  {r['verdict']}")
    json.dump(rows, open("/workspace/ckpt/walk_sweep.json", "w"), indent=2)
    best = max(rows, key=lambda r: r["fwd_vel"])
    print(f"BEST fwd_vel: step={best['step']} fwd_vel={best['fwd_vel']} ({best['verdict']})")
else:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/workspace/ckpt/walk_t4_latest.pkl"
    r = evaluate(ckpt, render_path="/workspace/ckpt/walk_eval.mp4")
    print(json.dumps(r, indent=2)); print("video -> /workspace/ckpt/walk_eval.mp4")
