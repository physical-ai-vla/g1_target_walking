"""Action module — closed-loop navigation for G1Nav (Plan A, static scene).

Pipeline (static-scene optimization, "Option A"):
  1. ONCE: the VLM cortex identifies the target and color+shape CV + ego-depth
     give its egocentric (bearing, distance). We convert that to a FIXED WORLD
     coordinate using the robot's spawn pose.
  2. LOOP at control rate: from the robot's CURRENT pose (odometry from MuJoCo
     qpos), recompute bearing/distance to that fixed world point -> a velocity
     command (vx, vy, yaw_rate) -> the PPO walking policy turns it into 29 joint
     PD targets -> physics steps. Stop when within reach radius.

The VLM is NOT called every step (scene is static); it runs once to lock the
goal. This keeps the loop real-time while honoring onboard-only perception.

This module is controller-agnostic about HOW the policy maps command->joints:
plug in the trained brax PPO inference fn via `set_policy`.
"""
from __future__ import annotations

import numpy as np


# ---- geometry: egocentric (bearing,distance) <-> world ---------------------

def ego_to_world(robot_x, robot_y, robot_yaw, bearing_deg, distance_m):
    """Target world (x,y) from robot pose + egocentric bearing(+left)/distance."""
    th = robot_yaw + np.deg2rad(bearing_deg)
    return (robot_x + distance_m * np.cos(th),
            robot_y + distance_m * np.sin(th))


def world_to_ego(robot_x, robot_y, robot_yaw, tx, ty):
    """Egocentric (bearing_deg +left, distance_m) of world point (tx,ty)."""
    dx, dy = tx - robot_x, ty - robot_y
    dist = float(np.hypot(dx, dy))
    bearing = np.arctan2(dy, dx) - robot_yaw
    bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
    return float(np.rad2deg(bearing)), dist


# ---- command policy: (bearing,distance) -> (vx, vy, wz) --------------------

# Kept well inside the walking policy's stable command regime (constant fwd 0.4
# was verified not to fall; a saturated 0.8 rad/s step yaw-rate DID topple it).
VX_MAX, VY_MAX, WZ_MAX = 0.45, 0.3, 0.5
REACH_RADIUS = 0.55


def goal_to_command(bearing_deg, distance_m):
    """Turn-then-go controller. Returns (vx, vy, wz, done)."""
    if distance_m <= REACH_RADIUS:
        return 0.0, 0.0, 0.0, True
    b = np.deg2rad(bearing_deg)
    wz = float(np.clip(1.0 * b, -WZ_MAX, WZ_MAX))      # gentler turn gain
    facing = abs(bearing_deg) < 40.0
    vx = VX_MAX if facing else 0.25 * VX_MAX           # keep moving while turning
    vx = float(np.clip(min(vx, 1.2 * distance_m), 0.0, VX_MAX))
    return vx, 0.0, wz, False


# ---- odometry: read robot base pose from MuJoCo qpos -----------------------

def robot_pose_from_qpos(qpos):
    """qpos[:7] = [x,y,z, qw,qx,qy,qz] (freejoint base). Returns (x,y,yaw)."""
    x, y = float(qpos[0]), float(qpos[1])
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    # yaw from quaternion
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return x, y, float(yaw)


# ---- the navigator ---------------------------------------------------------

class Navigator:
    """Drives the G1 to a fixed world goal using a command-conditioned policy.

    policy_fn(obs_state) -> action (29,) PD joint targets. obs_state must be the
    103-dim playground 'state' vector (with command injected). The caller wires
    obs construction since it depends on the env wrapper used at deploy time.
    """

    def __init__(self):
        self.goal_world = None      # (tx, ty), fixed once
        self.policy_fn = None
        self.reached = False
        self._vx = 0.0              # slew-limited command state
        self._wz = 0.0

    def set_policy(self, policy_fn):
        self.policy_fn = policy_fn
        return self

    def lock_goal_from_ego(self, robot_x, robot_y, robot_yaw,
                           bearing_deg, distance_m):
        """Call ONCE after the VLM cortex localizes the target."""
        self.goal_world = ego_to_world(robot_x, robot_y, robot_yaw,
                                       bearing_deg, distance_m)
        self.reached = False
        return self.goal_world

    def command(self, robot_x, robot_y, robot_yaw):
        """Current velocity command toward the locked goal (odometry-based)."""
        if self.goal_world is None:
            return 0.0, 0.0, 0.0, False
        bearing, dist = world_to_ego(robot_x, robot_y, robot_yaw, *self.goal_world)
        vx, vy, wz, done = goal_to_command(bearing, dist)
        # slew-rate limit both channels: the walking policy topples on a step
        # change in command. (This, together with the obs-normalization fix, is
        # what stops the robot from falling mid-turn.)
        self._vx += float(np.clip(vx - self._vx, -0.04, 0.04))
        self._wz += float(np.clip(wz - self._wz, -0.05, 0.05))
        if done:
            self.reached = True
        return self._vx, vy, self._wz, done


if __name__ == "__main__":
    # kinematic sanity test: a unicycle should reach a locked goal
    nav = Navigator()
    # robot at origin facing +x; target seen at bearing +25 deg, 2.5 m away
    gx, gy = nav.lock_goal_from_ego(0, 0, 0.0, 25.0, 2.5)
    print(f"locked goal world=({gx:.2f},{gy:.2f})")
    pose = np.array([0.0, 0.0, 0.0]); dt = 0.05
    for step in range(400):
        vx, vy, wz, done = nav.command(*pose)
        pose[0] += vx * np.cos(pose[2]) * dt
        pose[1] += vx * np.sin(pose[2]) * dt
        pose[2] += wz * dt
        if done:
            print(f"reached in {step} steps at ({pose[0]:.2f},{pose[1]:.2f})")
            break
    else:
        print(f"did not reach; ended at ({pose[0]:.2f},{pose[1]:.2f})")
