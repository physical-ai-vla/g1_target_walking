"""Instruction templates + success criteria for G1Nav.

An *instruction* is a free-form English command tied to a scene. Each template
knows how to:
  - render text (possibly with paraphrases) given a target object,
  - decide success given a trajectory of robot poses + object positions,
  - expose the high-level goal the teacher FSM should pursue.

Everything is deterministic given (scene, seed) so dataset regeneration is exact.

The instruction set deliberately spans difficulty tiers so Results can be broken
down by type (and so complex ones can serve as honest failure modes):

  TIER 1 (single goal):     "go to the <obj>"        / "approach the <obj>"
  TIER 2 (tracking):        "follow the <obj>"       (static here -> == go-to, but kept distinct)
  TIER 3 (compositional):   "go straight and turn <dir> after passing the <obj>"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


# ----- geometry helpers ----------------------------------------------------

def _xy(p):
    return np.asarray(p[:2], dtype=float)


def dist_xy(a, b) -> float:
    return float(np.linalg.norm(_xy(a) - _xy(b)))


# ----- instruction spec ----------------------------------------------------

@dataclass
class Instruction:
    template_id: str
    text: str                 # the rendered English instruction
    target_label: str         # e.g. "red ball"
    target_xy: tuple          # ground-truth target position (teacher-only)
    params: dict              # template-specific (e.g. {"turn": "right"})
    # success(traj, objects) -> bool ; traj is list of robot (x,y,yaw)
    success_fn: Callable

    def succeeded(self, traj, objects) -> bool:
        return bool(self.success_fn(traj, objects, self))


# ----- success criteria ----------------------------------------------------

REACH_RADIUS = 0.6     # within this xy distance of target counts as "reached"
PASS_RADIUS = 0.7      # within this -> considered "passed" the object


def _success_goto(traj, objects, instr) -> bool:
    """Reached if any pose along the trajectory comes within REACH_RADIUS."""
    t = instr.target_xy
    return any(dist_xy(p, t) <= REACH_RADIUS for p in traj)


def _success_turn_after_pass(traj, objects, instr) -> bool:
    """Must (1) pass near the object, then (2) change heading by ~90 deg in the
    commanded direction *after* passing."""
    t = instr.target_xy
    turn = instr.params["turn"]  # "left" or "right"
    pass_idx = None
    for i, p in enumerate(traj):
        if dist_xy(p, t) <= PASS_RADIUS:
            pass_idx = i
            break
    if pass_idx is None or pass_idx >= len(traj) - 1:
        return False
    yaw0 = traj[pass_idx][2]
    yaw1 = traj[-1][2]
    dyaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))  # wrapped
    if turn == "left":
        return dyaw > np.deg2rad(60)
    else:
        return dyaw < -np.deg2rad(60)


# ----- templates -----------------------------------------------------------

_GOTO_PARAPHRASES = [
    "go to the {obj}",
    "walk to the {obj}",
    "approach the {obj}",
    "head toward the {obj}",
    "move to the {obj}",
]
_FOLLOW_PARAPHRASES = [
    "follow the {obj}",
    "go follow the {obj}",
    "track the {obj}",
]
_TURN_PARAPHRASES = [
    "go straight and turn {turn} after passing the {obj}",
    "walk forward, then turn {turn} once you pass the {obj}",
    "pass the {obj} and then turn {turn}",
]


def sample_instruction(scene, rng) -> Instruction:
    """Pick a template + target object for a given scene, deterministically."""
    objs = scene.objects
    target = objs[int(rng.integers(len(objs)))]
    label = target.label
    txy = (target.x, target.y)

    tier = rng.choice(["goto", "follow", "turn"], p=[0.5, 0.2, 0.3])

    if tier == "goto":
        text = rng.choice(_GOTO_PARAPHRASES).format(obj=label)
        return Instruction("goto", text, label, txy, {}, _success_goto)

    if tier == "follow":
        text = rng.choice(_FOLLOW_PARAPHRASES).format(obj=label)
        # static scene -> follow reduces to reach-and-stay; reuse goto success
        return Instruction("follow", text, label, txy, {}, _success_goto)

    # turn-after-pass
    turn = rng.choice(["left", "right"])
    text = rng.choice(_TURN_PARAPHRASES).format(obj=label, turn=turn)
    return Instruction("turn_after_pass", text, label, txy, {"turn": turn},
                       _success_turn_after_pass)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from scene_gen import sample_scene

    for seed in range(6):
        scene = sample_scene(seed)
        rng = np.random.default_rng(1000 + seed)
        instr = sample_instruction(scene, rng)
        print(f"seed={seed:2d} [{instr.template_id:15s}] target={instr.target_label:16s} "
              f"@({instr.target_xy[0]:+.2f},{instr.target_xy[1]:+.2f}) :: \"{instr.text}\"")
