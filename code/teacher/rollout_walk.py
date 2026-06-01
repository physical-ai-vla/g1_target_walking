"""Roll out a trained (or mid-training) PPO walking policy and render a video.

Loads a checkpoint saved by train_walk.py (either <out>.pkl final params or
<out>_latest.pkl {step,params}), runs the mujoco_playground G1 env with a fixed
velocity command, and writes an mp4 + returns survival/▒distance stats. Used by
the training dashboard to SEE how the current policy walks.
"""
from __future__ import annotations
import argparse
import os
import pickle
import functools

import numpy as np


def load_params(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "params" in obj:
        return obj["params"], obj.get("step")
    return obj, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to *_latest.pkl or *.pkl")
    ap.add_argument("--env", default="G1JoystickFlatTerrain")
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", default="/workspace/ckpt/rollout.mp4")
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import locomotion_params
    from brax.training.agents.ppo import networks as ppo_networks

    env = registry.load(args.env)
    cfg = registry.get_default_config(args.env)
    ppo_cfg = locomotion_params.brax_ppo_config(args.env)
    net_kwargs = dict(ppo_cfg.network_factory)

    # rebuild the policy network + inference fn, load params
    obs_size = env.observation_size
    act_size = env.action_size
    networks = ppo_networks.make_ppo_networks(obs_size, act_size, **net_kwargs)
    make_policy = ppo_networks.make_inference_fn(networks)
    params, step = load_params(args.ckpt)
    policy = make_policy(params, deterministic=True)
    jit_policy = jax.jit(policy)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    # inject a fixed velocity command
    cmd = jp.array([args.vx, args.vy, args.wz])
    state.info["command"] = cmd

    # render with MuJoCo from the env's mj_model
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=320, width=480)
    # tracking camera that follows the robot from behind-above
    track_cam = mujoco.MjvCamera()
    track_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    track_cam.distance = 3.0
    track_cam.elevation = -20.0
    track_cam.azimuth = 90.0

    frames = []
    n = int(args.seconds / cfg.ctrl_dt)
    z0 = None
    fell = False
    for i in range(n):
        act, _ = jit_policy(state.obs, rng)
        state = jit_step(state, act)
        state.info["command"] = cmd
        # sync mj_data from mjx state for rendering
        qpos = np.array(state.data.qpos)
        mj_data.qpos[:] = qpos
        mj_data.qvel[:] = np.array(state.data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        if z0 is None:
            z0 = qpos[2]
        if qpos[2] < 0.4:
            fell = True
        track_cam.lookat[:] = [qpos[0], qpos[1], 0.5]  # follow robot base
        renderer.update_scene(mj_data, camera=track_cam)
        frames.append(renderer.render())

    try:
        import imageio
        imageio.mimsave(args.out, frames, fps=int(1 / cfg.ctrl_dt))
    except Exception as e:
        print(f"video save failed: {e}")

    final_z = float(np.array(state.data.qpos)[2])
    dist = float(np.linalg.norm(np.array(state.data.qpos)[:2]))
    print(f"ROLLOUT step={step} survived={'NO' if fell else 'YES'} "
          f"final_z={final_z:.2f} traveled={dist:.2f}m frames={len(frames)} -> {args.out}")


if __name__ == "__main__":
    main()
