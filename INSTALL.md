# G1Nav — Install & Run (any environment)

Two ways to use this, picked automatically by `install.sh`:

| Your machine | What you get | How the pipeline runs |
|---|---|---|
| **Has an NVIDIA GPU** (Ada-gen ideal) | the full stack | everything runs **locally** |
| **No GPU** (laptop) | the dashboards | pipeline runs on a **remote GPU pod** over SSH |

## 1. Install (one command)

```bash
bash install.sh           # auto-detects GPU; force with: bash install.sh gpu|dashboard
source .venv/bin/activate
```

That creates `.venv` and installs the right dependencies
(`requirements-gpu.txt` or `requirements-dashboard.txt`).

## 2. Run the dashboards

```bash
bash run_dashboard.sh
```

- **http://localhost:8502 — Interactive ("run it yourself")**
  1. pick a **scene seed** → **Load scene** (shows the ego view + the objects in that arena)
  2. type an **instruction** about one of those objects (e.g. `go to the red cube`)
  3. **Run** → GR00T N1.6 Eagle grounds the target → the PPO policy walks there in
     MuJoCo (physics only) → ego ∥ third-person video comes back.
- **http://localhost:8501 — Results** — replays the recorded demo videos, the walking
  curve, and the N1.6 cognition (works offline, no GPU).

The interactive app shows its **Mode** (LOCAL / REMOTE) in the sidebar.

### Remote mode (laptop → GPU pod)
If you have no local GPU, point the app at an SSH-reachable GPU pod (it has the repo
under `/workspace` and the checkpoints; see the pod bootstrap in `code/utils/`):

```bash
export G1NAV_POD_HOST=root@<ip>      # default points at the dev pod
export G1NAV_POD_PORT=<port>
export G1NAV_POD_KEY=~/.ssh/id_ed25519
bash run_dashboard.sh interactive
```

Force a mode regardless of detection: `export G1NAV_MODE=local` (or `remote`).

## 3. Run it from the command line (no UI)

```bash
# one-shot: scene seed + free-form instruction → video in outputs/
python code/student/g1nav_run.py --seed 0 --instruction "go to the orange cylinder"
```

## 4. Reproduce training / eval (GPU)

```bash
# train the walking policy (official Playground recipe)
python code/teacher/train_walk.py --timesteps 500000000 --num_envs 8192 \
    --termination -100 --track_lin 1.0 --stand_still -1.0 --num_evals 50 \
    --out checkpoint/walk_t4

# truthful raw-env evaluation (or sweep every checkpoint)
python code/teacher/walk_eval.py checkpoint/walk_t4_latest.pkl
python code/teacher/walk_eval.py --sweep checkpoint 'walk_t4_step*.pkl'
```

## Requirements / notes

- **GPU:** developed on **RTX 6000 Ada** (48 GB), CUDA 12.x. Avoid Ampere (cuSolver bug
  on jax 0.5.3) and Blackwell/sm_120 (no kernels in the pinned stack). RAM 16 GB+.
- **Cognition checkpoint:** the GR00T **N1.6 Eagle** vision encoder is the only reused
  pretrained model. Put it at `checkpoint/eagle-n16/` or set `G1NAV_EAGLE=/path/to/eagle-n16`.
  Without it, `g1nav_run.py` falls back to color/shape keyword grounding so the demo still runs.
- **No GPU at all and no pod?** Use the **Results** dashboard (8501) to view the recorded
  runs — it needs only the dashboard requirements.

See **`report.pdf`** for the full approach, architecture, and engineering log.
