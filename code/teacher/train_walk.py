"""Train the low-level velocity-conditioned walking policy for the G1 teacher.

Uses mujoco_playground's built-in `G1JoystickFlatTerrain` env (MJX) + brax PPO.
The env's action IS a PD joint-position target (motor_targets = default_pose +
action*action_scale), which is exactly the joint-target output the task wants.
The command is (vx, vy, yaw_rate) — the same interface our teacher FSM emits.

This is the highest-risk milestone (biped balance). We start on flat terrain and
keep an eval gate: the policy must track random commands for >=N seconds without
falling before we move on to data collection.

Run on the RunPod A6000 inside the venv:
    source /workspace/venv/bin/activate
    MUJOCO_GL=egl python train_walk.py --timesteps 60000000 --out /workspace/ckpt/walk

Outputs a brax params pickle to --out. Reload for rollouts in collect_bc.py.
"""
from __future__ import annotations
import argparse
import functools
import os
import pickle
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="G1JoystickFlatTerrain")
    ap.add_argument("--timesteps", type=int, default=60_000_000,
                    help="PPO env steps. Full recipe is 200M; 40-80M usually walks.")
    ap.add_argument("--out", default="/workspace/ckpt/walk")
    ap.add_argument("--num_envs", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restore", default=None,
                    help="path to a *_latest.pkl / *.pkl to resume training from")
    ap.add_argument("--termination", type=float, default=-3.0,
                    help="fall penalty scale; default -3 (env default is -100, "
                         "which makes the policy too scared to attempt walking)")
    ap.add_argument("--num_evals", type=int, default=100,
                    help="number of evals across training; each saves a checkpoint")
    ap.add_argument("--track_lin", type=float, default=2.0,
                    help="tracking_lin_vel reward scale (default env=1.0; raise to force forward motion)")
    ap.add_argument("--stand_still", type=float, default=-3.0,
                    help="stand_still penalty scale (default env=-1.0; more negative discourages standing)")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import jax
    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import locomotion_params
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks

    print(f"[walk] jax devices: {jax.devices()}")
    # Stack pinned to mujoco 3.4 + playground 0.1.0 -> MJX uses the classic JAX
    # backend (no warp), so no impl override is needed.
    # Soften the fall penalty so the policy explores real locomotion instead of
    # learning to barely-not-die (which is what -100 + reward+8/no-walking looked like).
    env_cfg = registry.get_default_config(args.env)
    env_cfg.reward_config.scales.termination = args.termination
    # FORWARD INCENTIVE: the policy was learning to stand still (x_travel~0) because
    # standing was a safe reward path. Strengthen command-velocity tracking and the
    # stand-still penalty so it MUST move to maximize reward.
    env_cfg.reward_config.scales.tracking_lin_vel = args.track_lin
    env_cfg.reward_config.scales.stand_still = args.stand_still
    env = registry.load(args.env, config=env_cfg)
    print(f"[walk] termination={env_cfg.reward_config.scales.termination} "
          f"tracking_lin_vel={env_cfg.reward_config.scales.tracking_lin_vel} "
          f"stand_still={env_cfg.reward_config.scales.stand_still}")
    ppo_cfg = locomotion_params.brax_ppo_config(args.env)
    ppo_cfg.num_timesteps = args.timesteps
    ppo_cfg.num_envs = args.num_envs
    ppo_cfg.num_evals = args.num_evals   # more evals -> checkpoint saved more often
    print(f"[walk] env={args.env} obs={env.observation_size} act={env.action_size}")
    print(f"[walk] timesteps={ppo_cfg.num_timesteps:,} num_envs={ppo_cfg.num_envs}")

    # network factory from the config
    net_kwargs = dict(ppo_cfg.network_factory)
    make_networks = functools.partial(ppo_networks.make_ppo_networks, **net_kwargs)

    import json
    metrics_path = args.out + "_metrics.jsonl"
    open(metrics_path, "w").close()  # truncate
    times = [time.monotonic()]
    def progress(step, metrics):
        times.append(time.monotonic())
        rew = float(metrics.get("eval/episode_reward", float("nan")))
        rew_std = float(metrics.get("eval/episode_reward_std", float("nan")))
        elapsed = times[-1] - times[0]
        print(f"[walk] step={step:>12,}  reward={rew:8.3f} +/-{rew_std:6.3f}  "
              f"elapsed={elapsed/60:5.1f}m", flush=True)
        # append to a metrics log the dashboard reads
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": int(step), "reward": rew,
                                "reward_std": rew_std, "elapsed_min": elapsed / 60}) + "\n")

    # save a checkpoint every eval AND render a short rollout in-process (same
    # GPU context as training -> no cuSolver clash). The dashboard reads the mp4.
    import numpy as _np
    render_env = registry.load(args.env, config=env_cfg)   # for obs/dynamics
    _mjm = render_env.mj_model
    def save_ckpt(step, make_policy, params):
        try:
            blob = {"step": int(step), "params": params}
            with open(args.out + "_latest.pkl", "wb") as f:
                pickle.dump(blob, f)
            # also keep a NUMBERED checkpoint so we can sweep every eval later
            # (전수 조사 / full-checkpoint investigation). ~2MB each, ~50 total.
            with open(args.out + f"_step{int(step):010d}.pkl", "wb") as f:
                pickle.dump(blob, f)
        except Exception as e:
            print(f"[walk] ckpt save failed: {e}", flush=True)
            return
        # quick rollout video (forward command) so the dashboard shows progress.
        # UNIFIED TRUTHFUL SETUP: use the RAW env (NO wrap_for_brax_training =>
        # NO AutoResetWrapper). The brax wrapper teleports the robot back to the
        # start pose on every fall (done), which MASKS falls as "stepping in
        # place" (min_z~0.74, x~0). The raw env shows reality. We measure LOCAL
        # forward velocity (what tracking_lin_vel rewards; command is in the
        # robot's heading frame, NOT world-X) + net horizontal displacement.
        # walk_eval.py uses this identical setup for the full-checkpoint sweep.
        try:
            import mujoco, imageio, jax.numpy as jp
            pol = jax.jit(make_policy(params, deterministic=True))
            rst = jax.jit(render_env.reset); stp = jax.jit(render_env.step)
            st = rst(jax.random.PRNGKey(0))            # RAW, unbatched, no auto-reset
            CMD = jp.array([0.6, 0.0, 0.0])
            lv_id = mujoco.mj_name2id(_mjm, mujoco.mjtObj.mjOBJ_SENSOR, "local_linvel_pelvis")
            lv_adr = _mjm.sensor_adr[lv_id]
            mjd = mujoco.MjData(_mjm); rr = mujoco.Renderer(_mjm, 288, 384)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = 4.0; cam.elevation = -12; cam.azimuth = 135
            frames = []; xs = []; ys = []; zs = []; fwd = []
            for _ in range(200):                 # 4 s
                st.info["command"] = CMD
                act, _ = pol(st.obs, jax.random.PRNGKey(0)); st = stp(st, act)
                q = _np.array(st.data.qpos).reshape(-1)
                sd = _np.array(st.data.sensordata).reshape(-1)
                mjd.qpos[:] = q; mjd.qvel[:] = _np.array(st.data.qvel).reshape(-1)
                mujoco.mj_forward(_mjm, mjd); cam.lookat[:] = [q[0], q[1], 0.5]
                rr.update_scene(mjd, camera=cam); frames.append(rr.render())
                xs.append(float(q[0])); ys.append(float(q[1])); zs.append(float(q[2]))
                fwd.append(float(sd[lv_adr]))
            imageio.mimsave(args.out + "_rollout.mp4", frames, fps=50)
            rr.close()
            dx = xs[-1] - xs[0]; dy = ys[-1] - ys[0]
            net = float((dx * dx + dy * dy) ** 0.5)        # net horizontal distance
            mfwd = float(_np.mean(fwd))                     # mean local forward vel (cmd=0.6)
            with open(args.out + "_rollout.json", "w") as jf:
                json.dump({"step": int(step), "x_travel": net, "dx": float(dx),
                           "dy": float(dy), "fwd_vel": mfwd,
                           "min_z": float(min(zs))}, jf)
            print(f"[walk]   rollout @ {int(step)}: net_dist={net:.2f}m "
                  f"fwd_vel={mfwd:+.2f}(cmd0.6) min_z={min(zs):.2f}", flush=True)
        except Exception as e:
            print(f"[walk] rollout failed: {e}", flush=True)

    # optionally resume from a previous checkpoint (continue training)
    restore_params = None
    if args.restore:
        with open(args.restore, "rb") as f:
            ro = pickle.load(f)
        restore_params = ro["params"] if isinstance(ro, dict) and "params" in ro else ro
        print(f"[walk] RESUMING from {args.restore}", flush=True)

    extra = {"restore_params": restore_params} if restore_params is not None else {}
    # CRITICAL (matches the official locomotion.ipynb): pass DOMAIN RANDOMIZATION.
    # Without it, the G1 walking policy overfits and is unstable (reward rises but
    # it can't actually walk). This was the missing piece.
    try:
        randomization_fn = registry.get_domain_randomizer(args.env)
        print(f"[walk] domain randomizer: {randomization_fn is not None}", flush=True)
    except Exception as e:
        randomization_fn = None
        print(f"[walk] no domain randomizer: {e}", flush=True)
    if randomization_fn is not None:
        extra["randomization_fn"] = randomization_fn

    train_fn = functools.partial(
        ppo.train,
        **{k: v for k, v in ppo_cfg.items() if k not in ("network_factory",)},
        network_factory=make_networks,
        # use mujoco_playground's brax wrapper (mjx State is not brax pipeline_state)
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        policy_params_fn=save_ckpt,
        seed=args.seed,
        **extra,
    )

    # separate eval_env (official recipe), with the same termination override
    eval_env = registry.load(args.env, config=env_cfg)
    make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env)

    with open(args.out + ".pkl", "wb") as f:
        pickle.dump(params, f)
    # also save env config for reproducible rollouts
    with open(args.out + "_cfg.pkl", "wb") as f:
        pickle.dump({"env": args.env, "env_cfg": env_cfg.to_dict()}, f)
    print(f"[walk] saved params -> {args.out}.pkl")
    print(f"[walk] total wall-clock: {(time.monotonic()-times[0])/60:.1f} min")


if __name__ == "__main__":
    main()
