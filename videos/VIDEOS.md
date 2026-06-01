# videos/ — demo episodes (each filename contains its instruction)

Three closed-loop episodes. Each is the full pipeline run end-to-end:
**GR00T N1.6 Eagle cognition → Navigator → trained PPO walking policy → MuJoCo
physics** (physics only, no kinematic motion). Every clip is **ego (right) ∥
third-person (left)** side-by-side, with **2 s standing still → walk → 2 s
standing at the target**.

Each episode ships a matched triple (same `seed{N}_<instruction>` prefix):
`*.mp4` (video) · `*.cognition.json` (the N1.6 Eagle grounding result) ·
`*.ego.png` (the start ego frame the cognition saw).

| file prefix | instruction | scene objects (seed) | N1.6 grounded target | result | final dist | torso z |
|---|---|---|---|:---:|---:|---:|
| `seed0_go-to-the-orange-cylinder` | "go to the orange cylinder" | orange cylinder, yellow ball, purple cube | orange cylinder | ✅ reached | 0.39 m | 0.75 m |
| `seed1_go-to-the-red-cube` | "go to the red cube" | red cube, purple cube, yellow cone | red cube | ✅ reached | 0.41 m | 0.76 m |
| `seed2_go-to-the-purple-cylinder` | "go to the purple cylinder" | purple cylinder, purple ball, orange cone | purple cylinder | ✅ reached | 0.54 m | 0.75 m |

All three reach the instructed object and stay upright (torso ≈ 0.75 m, never
falls). The grounded target in every `*.cognition.json` matches the instruction
in its filename; the Eagle perception embedding is 286 patches × 2048-d.

Regenerate any episode (on a GPU pod with the checkpoints):

```bash
bash code/student/run_pipeline_n16.sh 0 "go to the orange cylinder"
# or portable:  python code/student/g1nav_run.py --seed 0 --instruction "go to the orange cylinder"
```
