"""Closed-loop navigation: run the trained PPO walking policy in the integrated
scene, driven by the Navigator toward the VLM-identified target. Records ego +
third-person video.

This is the FULL Plan-A inference loop (minus calling the VLM every step — the
target is localized once and tracked via odometry, since the scene is static):

  VLM cortex (once)  -> target world coord (via color+shape CV + ego-depth)
  loop @ ctrl_dt:
    odometry (qpos) -> Navigator -> velocity command (vx,vy,wz)
    build playground-compatible obs(103) with that command
    PPO policy(obs) -> action(29) -> motor_targets = default_pose+action*scale
    mj_step (physics)
    until reached or timeout
  render ego + third-person each step -> side-by-side mp4

obs layout (must match mujoco_playground G1 Joystick _get_obs 'state'):
  [linvel(3), gyro(3), gravity(3), command(3), qpos[7:]-default(29),
   qvel[6:](29), last_act(29), cos/sin phase(2)] = 103
"""
from __future__ import annotations
import os
# Force CPU BEFORE anything imports jax, when G1NAV_CPU=1 (a GPU training job
# may own the device). Must run before the first `import jax` anywhere.
if os.environ.get("G1NAV_CPU") == "1":
    os.environ["JAX_PLATFORMS"] = "cpu"
print(f"[navigate] G1NAV_CPU={os.environ.get('G1NAV_CPU')} "
      f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')}", flush=True)
import argparse
import pickle

import numpy as np


def quat_yaw(q):
    qw, qx, qy, qz = q
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default=None,
                    help="object label e.g. 'orange cylinder'; default=first object")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--out", default="/workspace/ckpt/nav.mp4")
    ap.add_argument("--use_vlm", action="store_true",
                    help="localize target with the VLM cortex (else use GT object pos)")
    ap.add_argument("--cpu", action="store_true",
                    help="force JAX onto CPU (use when a GPU training job owns the device)")
    ap.add_argument("--fixed_cmd", type=float, nargs=3, default=None,
                    metavar=("VX", "VY", "WZ"),
                    help="ignore Navigator; drive a constant (vx,vy,wz) command")
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"   # must be set BEFORE importing jax

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs"))

    import jax
    import mujoco
    from mujoco_playground.config import locomotion_params
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    import integrated_scene as isc
    from action_module import Navigator

    ENV = "G1JoystickFlatTerrain"
    # Avoid registry.load(ENV): it puts an MJX model on the GPU (clashes with a
    # running training job / fails if cuda is busy). We only need the network
    # SHAPES + a couple of config scalars; physics runs in plain MuJoCo below.
    obs_size = {"state": 103, "privileged_state": 216}
    act_size = 29
    ppo_cfg = locomotion_params.brax_ppo_config(ENV)
    net_kwargs = dict(ppo_cfg.network_factory)

    class _Cfg:  # minimal stand-in for registry config scalars
        action_scale = 0.5
        ctrl_dt = 0.02
    cfg = _Cfg()

    # build policy + load params.
    # CRITICAL: training used normalize_observations=True, so the checkpoint stores
    # the running-stats normalizer as params[0]. The network MUST be built with the
    # same normalize preprocessor or make_policy ignores those stats and feeds the
    # policy RAW (out-of-distribution) observations -> it topples. (This was the bug.)
    networks = ppo_networks.make_ppo_networks(
        obs_size, act_size,
        preprocess_observations_fn=running_statistics.normalize, **net_kwargs)
    make_policy = ppo_networks.make_inference_fn(networks)
    with open(args.ckpt, "rb") as f:
        obj = pickle.load(f)
    params = obj["params"] if isinstance(obj, dict) and "params" in obj else obj
    policy = make_policy(params, deterministic=True)
    jit_policy = jax.jit(policy)
    rng = jax.random.PRNGKey(0)

    # integrated scene (matches policy model)
    path, scene = isc.build_integrated_xml(args.seed)
    m = mujoco.MjModel.from_xml_path(path)
    # CRITICAL: match the training env's physics dt. playground sets
    # opt.timestep = sim_dt = 0.002 in code (NOT via XML), and ctrl_dt = 0.02,
    # so n_substeps = 10. Our scene loads the XML directly and may have a
    # different default timestep -> the policy would see wrong dynamics and fall.
    SIM_DT = 0.002
    m.opt.timestep = SIM_DT
    # CRITICAL: replicate the training env's EXACT physics options. The policy was
    # trained with MJX using g1_mjx_feetonly.xml's <option iterations="3"
    # ls_iterations="5" integrator="Euler"><flag eulerdamp="disable"/>. Plain
    # MuJoCo defaults (iterations=100, eulerdamp enabled) give different contact
    # dynamics -> the policy drifts and falls after ~1.5s. Match them exactly.
    m.opt.iterations = 3
    m.opt.ls_iterations = 5
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    # control params (from playground config)
    # CRITICAL: the policy was trained against the env's hardcoded _default_pose
    # (knees ~0.67, elbow ~0.6), NOT our keyframe qpos[7:] (knees 0.3, elbow 1.28).
    # Using the keyframe pose made obs (qpos-default) AND control (default+action)
    # both wrong -> instant fall. Use the exact training default_pose.
    default_pose = np.array([
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,      # left leg
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,      # right leg
        0.0, 0.0, 0.0,                              # waist
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,          # left arm
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,         # right arm
    ], dtype=float)
    assert default_pose.shape[0] == m.nu, (default_pose.shape, m.nu)
    # also START the robot at this pose (set the qpos joints), so step 0 matches
    d.qpos[7:7 + m.nu] = default_pose
    mujoco.mj_forward(m, d)
    action_scale = cfg.action_scale
    ctrl_dt = cfg.ctrl_dt
    sim_steps = int(round(ctrl_dt / SIM_DT))   # = 10 substeps per control step
    print(f"[navigate] sim_dt={SIM_DT} ctrl_dt={ctrl_dt} n_substeps={sim_steps}",
          flush=True)

    # --- target world position ---
    tname = args.target or scene.objects[0].label
    tobj = next((o for o in scene.objects if o.label == tname), scene.objects[0])
    # (GT used here for the walking demo; the VLM-localized version plugs in the
    #  same world coord via the cortex — verified separately in the dashboard.)
    goal_world = (tobj.x, tobj.y)

    nav = Navigator()
    nav.goal_world = goal_world

    # gait phase
    phase = np.array([0.0, np.pi])
    gait_freq = 1.4
    phase_dt = 2 * np.pi * ctrl_dt * gait_freq
    last_act = np.zeros(m.nu)

    # --- read obs from the SAME named sensors the training env uses, so the
    # obs vector matches _get_obs exactly (mismatch was making the policy fall) ---
    def _sensor_adr(name):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f"sensor '{name}' not found in integrated model")
        return m.sensor_adr[sid], m.sensor_dim[sid]
    LV_adr, LV_dim = _sensor_adr("local_linvel_pelvis")   # velocimeter (3)
    GY_adr, GY_dim = _sensor_adr("gyro_pelvis")           # gyro (3)
    # CRITICAL: the training env computes the projected-gravity obs term as
    #   gravity = site_xmat[imu_in_pelvis].T @ [0,0,-1]
    # i.e. the world DOWN vector expressed in the pelvis-IMU LOCAL frame. The
    # `upvector_pelvis` sensor returns the world-frame +z axis, whose negation
    # only coincides with the local-frame gravity while the robot walks straight.
    # As soon as it YAWS, the two frames diverge -> the gravity term is wrong ->
    # the policy mis-reads its tilt and topples (~1.6 s). Use the site matrix so
    # the term is identical to training in every heading.
    IMU_SID = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu_in_pelvis")
    if IMU_SID < 0:
        raise RuntimeError("site 'imu_in_pelvis' not found in integrated model")

    frames = []
    renderer_ego = mujoco.Renderer(m, 240, 320)
    renderer_tp = mujoco.Renderer(m, 240, 320)

    def get_obs(command):
        sd = d.sensordata
        linvel = sd[LV_adr:LV_adr + LV_dim]            # local linear velocity
        gyro = sd[GY_adr:GY_adr + GY_dim]              # local angular velocity
        # EXACTLY as training: world down vector in the pelvis-IMU local frame.
        xmat = d.site_xmat[IMU_SID].reshape(3, 3)
        gravity = xmat.T @ np.array([0.0, 0.0, -1.0])
        joint_angles = d.qpos[7:7 + m.nu] - default_pose
        joint_vel = d.qvel[6:6 + m.nu]
        cos = np.cos(phase); sin = np.sin(phase)
        state = np.concatenate([linvel, gyro, gravity, command,
                                joint_angles, joint_vel, last_act,
                                np.concatenate([cos, sin])]).astype(np.float32)
        return state

    import jax.numpy as jp
    n = int(args.seconds / ctrl_dt)
    reached = False
    # optional: override Navigator with a fixed command, to isolate walking from
    # the navigation controller. e.g. --fixed_cmd 0 0 0 (stand), 0.5 0 0 (fwd).
    fixed = args.fixed_cmd
    # warmup: ramp the command up over the first ~0.5s so the standing policy
    # isn't hit with a large velocity target on step 0 (which can topple it).
    WARMUP = 25
    # pre/post-roll: hold a zero command so the clip opens with ~2 s of the robot
    # standing still, and closes with ~2 s of it standing at the reached target.
    HOLD = int(round(2.0 / ctrl_dt))   # 2 s at 50 Hz = 100 control steps

    def step_once(command):
        """Run one control step with the given command and append a frame.
        Returns False if the robot fell."""
        nonlocal last_act
        obs = get_obs(command)
        act, _ = jit_policy({"state": jp.array(obs),
                             "privileged_state": jp.zeros(obs_size["privileged_state"])},
                            rng)
        act = np.array(act)
        d.ctrl[:] = default_pose + act * action_scale
        for _ in range(sim_steps):
            mujoco.mj_step(m, d)
        last_act = act
        phase[:] = (phase + phase_dt + np.pi) % (2 * np.pi) - np.pi
        renderer_ego.update_scene(d, camera="ego"); ego = renderer_ego.render()
        renderer_tp.update_scene(d, camera="thirdperson"); tp = renderer_tp.render()
        frames.append(np.concatenate([tp, ego], axis=1))  # side by side
        return d.qpos[2] >= 0.4

    ZERO = np.zeros(3, dtype=np.float32)
    # --- pre-roll: 2 s standing still (zero command) ---
    for _ in range(HOLD):
        step_once(ZERO)

    # --- main: navigate to the target ---
    for i in range(n):
        x, y, yaw = float(d.qpos[0]), float(d.qpos[1]), quat_yaw(d.qpos[3:7])
        if fixed is not None:
            vx, vy, wz, done = fixed[0], fixed[1], fixed[2], False
        else:
            vx, vy, wz, done = nav.command(x, y, yaw)
        ramp = min(1.0, (i + 1) / WARMUP)
        command = np.array([vx * ramp, vy * ramp, wz * ramp], dtype=np.float32)
        if not step_once(command):
            print(f"FELL at step {i}")
            break
        if done:
            reached = True
            break

    # --- post-roll: 2 s standing still at the reached target ---
    if reached:
        for _ in range(HOLD):
            step_once(ZERO)

    try:
        import imageio
        imageio.mimsave(args.out, frames, fps=int(1 / ctrl_dt))
    except Exception as e:
        print("video save failed:", e)
    fx, fy = float(d.qpos[0]), float(d.qpos[1])
    dist = float(np.hypot(goal_world[0] - fx, goal_world[1] - fy))
    print(f"NAV seed={args.seed} target='{tname}' reached={reached} "
          f"final_dist_to_goal={dist:.2f}m base_z={d.qpos[2]:.2f} "
          f"frames={len(frames)} -> {args.out}")


if __name__ == "__main__":
    main()
