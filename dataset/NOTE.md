# dataset/ — there is no collected dataset, by design

This project **collects no training data**, because nothing in the pipeline is
trained on a dataset:

- **Cognition (what/where).** The instructed target is identified by the
  **GR00T N1.6 Eagle** vision encoder used **zero-shot** (it is already
  pretrained) plus a color/shape CV pass. We train nothing here, so there is no
  cognition dataset to collect or ship.
- **Control (how to move).** The walking policy is trained by **on-policy RL
  (PPO) entirely inside the MuJoCo (MJX) simulator** — `G1JoystickFlatTerrain`,
  8192 parallel environments generating transitions on the fly each step. There
  is no offline/behavior-cloning dataset; the "data" exists only transiently
  inside the RL rollout buffers during training and is never stored.

So the take-home's *"training data, or a manifest + script that regenerates it
deterministically"* has no training-data branch for us — there is nothing to
regenerate as data. **The substance of the work is evaluation, not data.**

## What *is* deterministic and regenerable: the evaluation suite

The only regenerable artifacts are the **evaluation scenes + instructions**,
produced from integer seeds (no large assets shipped):

```bash
# a scene (3 colored objects, deterministic from the seed)
python code/envs/scene_gen.py --seed 0 --validate

# an instruction for that scene (go-to / follow / turn-after-pass tiers)
python -c "from code.envs.scene_gen import sample_scene; \
import numpy as np, code.envs.instructions as I; \
s=sample_scene(0); print(I.sample_instruction(s, np.random.default_rng(1000)).text)"
```

- `code/envs/scene_gen.py`  — seed → exact scene (objects, poses, cameras, MJCF).
- `code/envs/instructions.py` — seed → instruction text + success criterion.

Evaluation seeds used for the reported numbers: **0–5** (extend the range to test
more). Because everything is seeded, any reviewer regenerates the identical
scenes without us shipping image/trajectory files.

## Where the real results live

- `checkpoint/walk_t4_latest.pkl` — the trained controller (the only learned weights).
- `code/teacher/walk_eval.py` — truthful raw-env evaluation (forward velocity,
  torso height, net distance; no AutoReset masking).
- `videos/` — closed-loop instruction-following episodes (ego ∥ third-person).
