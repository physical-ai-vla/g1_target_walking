"""Privileged teacher planner (FSM) for G1Nav.

Given GROUND-TRUTH robot pose + object positions + the instruction, emit a
high-level velocity command (vx, vy, yaw_rate) in the robot's base frame. This
command drives the low-level velocity-conditioned walking policy (the MJX/PPO
teacher), whose joint targets become the BC labels for the student VLA.

The student NEVER sees this planner or the GT — it only imitates the resulting
joint targets from pixels+sensors+text. The planner is "privileged".

Design: a small finite-state machine per instruction type.
  goto / follow      : APPROACH -> STOP
  turn_after_pass    : APPROACH -> (passed?) -> TURN -> GO_STRAIGHT -> STOP
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# command limits (must match what the walking policy was trained to track)
VX_MAX = 0.6     # m/s forward
VY_MAX = 0.3     # m/s lateral
WZ_MAX = 0.8     # rad/s yaw


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _to_base_frame(dx, dy, yaw):
    """World displacement -> robot base frame (x forward, y left)."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    return c * dx - s * dy, s * dx + c * dy


@dataclass
class PlannerState:
    phase: str = "APPROACH"
    passed: bool = False
    turn_done: bool = False
    yaw_at_pass: float = 0.0
    steps_in_phase: int = 0


@dataclass
class TeacherPlanner:
    instruction: object               # Instruction (has target_xy, params, template_id)
    reach_radius: float = 0.5
    pass_radius: float = 0.7
    state: PlannerState = field(default_factory=PlannerState)

    def reset(self):
        self.state = PlannerState()

    def command(self, robot_pose) -> tuple:
        """robot_pose = (x, y, yaw) in world. Returns (vx, vy, wz, done)."""
        x, y, yaw = robot_pose
        tx, ty = self.instruction.target_xy
        dx, dy = tx - x, ty - y
        d = float(np.hypot(dx, dy))
        bx, by = _to_base_frame(dx, dy, yaw)        # target in base frame
        heading_err = _wrap(np.arctan2(dy, dx) - yaw)

        s = self.state
        tid = self.instruction.template_id

        # ---- simple go-to / follow ----
        if tid in ("goto", "follow"):
            if d <= self.reach_radius:
                return 0.0, 0.0, 0.0, True
            return self._approach(bx, by, heading_err, d)

        # ---- turn-after-pass ----
        if tid == "turn_after_pass":
            return self._turn_after_pass(robot_pose, bx, by, heading_err, d)

        # default: stand
        return 0.0, 0.0, 0.0, True

    # -- behaviors ----------------------------------------------------------

    def _approach(self, bx, by, heading_err, d):
        """Walk toward target: turn to face it, then drive forward."""
        wz = float(np.clip(2.0 * heading_err, -WZ_MAX, WZ_MAX))
        # only move forward fast once roughly facing the target
        facing = abs(heading_err) < np.deg2rad(35)
        vx = VX_MAX if facing else 0.15 * VX_MAX
        vx = float(np.clip(min(vx, 1.2 * d), 0, VX_MAX))  # ease in near goal
        vy = 0.0
        return vx, vy, wz, False

    def _turn_after_pass(self, robot_pose, bx, by, heading_err, d):
        s = self.state
        x, y, yaw = robot_pose
        turn = self.instruction.params.get("turn", "left")

        if s.phase == "APPROACH":
            if d <= self.pass_radius:
                s.phase = "TURN"
                s.yaw_at_pass = yaw
                s.steps_in_phase = 0
            return self._approach(bx, by, heading_err, max(d, 1.0))  # keep moving

        if s.phase == "TURN":
            sign = +1.0 if turn == "left" else -1.0
            turned = _wrap(yaw - s.yaw_at_pass) * sign
            if turned >= np.deg2rad(85):
                s.phase = "GO_STRAIGHT"
                s.steps_in_phase = 0
                return 0.3 * VX_MAX, 0.0, 0.0, False
            # pivot turn (small forward to keep gait stable)
            return 0.1 * VX_MAX, 0.0, sign * WZ_MAX, False

        if s.phase == "GO_STRAIGHT":
            s.steps_in_phase += 1
            if s.steps_in_phase > 40:   # walk straight a bit, then done
                return 0.0, 0.0, 0.0, True
            return 0.6 * VX_MAX, 0.0, 0.0, False

        return 0.0, 0.0, 0.0, True


if __name__ == "__main__":
    # smoke test: drive a point-robot kinematically with the planner commands
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
    from scene_gen import sample_scene
    from instructions import sample_instruction

    for seed in [0, 5]:
        scene = sample_scene(seed)
        rng = np.random.default_rng(1000 + seed)
        instr = sample_instruction(scene, rng)
        planner = TeacherPlanner(instr)
        planner.reset()
        pose = np.array([0.0, 0.0, scene.robot_yaw])
        dt = 0.05
        traj = [tuple(pose)]
        done = False
        for step in range(2000):
            vx, vy, wz, done = planner.command(tuple(pose))
            # integrate a unicycle as a stand-in for the walking policy
            pose[0] += (vx * np.cos(pose[2]) - vy * np.sin(pose[2])) * dt
            pose[1] += (vx * np.sin(pose[2]) + vy * np.cos(pose[2])) * dt
            pose[2] = _wrap(pose[2] + wz * dt)
            traj.append(tuple(pose))
            if done:
                break
        ok = instr.succeeded([(p[0], p[1], p[2]) for p in traj], scene.objects)
        print(f"seed={seed} [{instr.template_id:15s}] \"{instr.text}\"")
        print(f"   steps={step+1} final=({pose[0]:+.2f},{pose[1]:+.2f},{np.rad2deg(pose[2]):+.0f}deg) "
              f"target=({instr.target_xy[0]:+.2f},{instr.target_xy[1]:+.2f}) SUCCESS={ok}")
