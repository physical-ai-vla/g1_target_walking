# G1Nav — Language-Conditioned Lower-Body Navigation for the Unitree G1 (MuJoCo / MJX)

Maps **ego RGB-D + proprioception + history + a fixed English instruction → Unitree G1
lower-body joint-position targets**, in real time in MuJoCo (MJX) on a single consumer GPU.
No kinematic motion at test time — physics only. Reuses only parameters that are part of
the **Isaac GR00T N1.6** checkpoint; no other pretrained checkpoints.

Full write-up (architecture, engineering log, results): **[`report.pdf`](report.pdf)**
(한국어판: **[`report_ko.pdf`](report_ko.pdf)**).

## Install & run in 2 commands (any machine)

```bash
bash install.sh            # auto: GPU → full stack, else dashboard-only
source .venv/bin/activate && bash run_dashboard.sh
```

Then open **http://localhost:8502** (interactive — pick a scene, see its objects,
type an instruction, watch the robot walk there) or **http://localhost:8501**
(replay recorded results). Full setup details, remote-pod mode, and CLI usage:
**[`INSTALL.md`](INSTALL.md)**.

---

## TL;DR

| Layer | What | How | Trained? |
|---|---|---|---|
| **Cognition** (what/where) | identify + localize the instructed target | **GR00T N1.6 Eagle** vision encoder (perception) + color/shape CV grounding + ego-depth | **no — zero-shot** |
| **Control** (how to move) | walk to a velocity command | **PPO** walking policy, MJX `G1JoystickFlatTerrain`, trained **once** | yes, amortized |
| **Navigator** | target world coord → `(vx, vy, ωz)` | `world_to_ego` → `goal_to_command` every 20 ms | — |

> Thesis: **high-level grounding needs no data (GR00T N1.6 perception + CV suffices); only
> low-level locomotion needs RL, trained once and reused across every instruction.** The model
> output is the 29-D lower-body PD joint-target vector — never a velocity or other high-level handle.

## Final results (verified)

- Walking policy trained to **154.8 M steps** → `checkpoint/walk_t4_latest.pkl`
  (numbered `walk_t4_step*.pkl` shipped for the full sweep).
- Forward velocity **0.52 m/s** at the final step, **peak 0.57 m/s** (95 % of the 0.6 m/s command).
- Torso height **0.74 m** (upright); **never falls** in raw-env (no-AutoReset) evaluation.
- Net horizontal distance ≈ **2.1 m** per rollout episode.
- Closed loop verified end-to-end: cognition → Navigator → walking policy → MuJoCo physics
  reaches the instructed target.

## Requirements

- **Training:** 1× Ada-generation GPU (developed on **RTX 6000 Ada**, 48 GB). ~77 min to 155 M
  steps, ~2.2 M env-steps/min. Avoid Ampere (cuSolver bug on jax 0.5.3) and Blackwell/sm_120
  (no kernels in the pinned stack).
- **Inference:** single consumer GPU; the spine (PPO MLP) runs at 50 Hz, the cortex runs once
  per episode (static scene). RAM: 16 GB+ host.
- **Pinned stack:** `jax[cuda12]==0.5.3, brax==0.14.2, flax==0.10.6, mujoco==3.4.0,
  mujoco-mjx==3.4.0, mujoco_playground==0.1.0`, `transformers==4.51.3` for the Eagle cortex.

## Quickstart

```bash
# 1. Environment (Ada GPU)
bash code/utils/setup_ada.sh

# 2. Train the walking policy (official Playground recipe)
python code/teacher/train_walk.py --timesteps 500000000 --num_envs 8192 \
    --termination -100 --track_lin 1.0 --stand_still -1.0 --num_evals 50 \
    --out checkpoint/walk_t4

# 3. Evaluate truthfully on the raw env (or sweep every checkpoint)
python code/teacher/walk_eval.py checkpoint/walk_t4_latest.pkl
python code/teacher/walk_eval.py --sweep checkpoint 'walk_t4_step*.pkl'   # → walk_sweep.json

# 4. Scenes + instructions (seeded, regenerable — see dataset/)
python code/envs/scene_gen.py --seed 0

# 5. Cognition (GR00T N1.6 Eagle, zero-shot)
python code/student/cortex_eagle.py --image ego.png \
    --instruction "go to the orange cylinder" \
    --labels "orange cylinder,yellow ball,purple cube" --result out.json

# 6. Full closed-loop demo (cognition → Navigator → walking → physics → ego ∥ 3rd-person)
python code/student/navigate.py --ckpt checkpoint/walk_t4_latest.pkl --use_eagle \
    --target "orange cylinder"

# 7. Recorded-results dashboard (laptop, no GPU)
streamlit run code/student/app_results.py            # → http://localhost:8501
```

## Repository layout

```
code/
  teacher/   train_walk.py · walk_eval.py · rollout_walk.py        (RL training + truthful eval)
  student/   cortex_eagle.py (GR00T N1.6 cognition) · color_detect.py · navigate.py
             app_results.py (results dashboard) · app_brain.py (cortex-stage viz)
  envs/      scene_gen.py · instructions.py                        (seeded scene/instruction gen)
  utils/     setup_ada.sh · setup_h100.sh · pod_bootstrap.sh · smoke_test.py
checkpoint/  walk_t4_latest.pkl  (+ numbered walk_t4_step*.pkl for the sweep)
dataset/     NOTE.md — no collected dataset by design (zero-shot cognition + online RL);
             only the seeded evaluation scenes/instructions are regenerable
videos/      ≥3 episodes, ego RGB-D ∥ third-person
report.md    full report
```

## Notes on compliance

- **MuJoCo only** (MJX for training, MuJoCo-C render for some eval paths). No Isaac/Gazebo/Bullet.
- **Pretrained:** only the GR00T N1.6 Eagle vision encoder is reused (as the perception embedding
  GR00T N1.6 itself conditions on); everything else (the PPO policy) is trained from scratch.
- **No kinematic motion:** the demo is rendered from the same physics it is controlled in.
