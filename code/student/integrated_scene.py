"""Build an integrated MJCF: the mujoco_playground G1 (whose model EXACTLY matches
the trained PPO policy's obs/action) + our colored objects + ego & third-person
cameras. This lets us run the trained walking policy in a navigation scene.

We start from playground's scene_mjx_feetonly.xml (the base scene that g1 includes)
and inject: an ego camera on the torso, a high overview camera, the colored
objects, and arena walls — then set the robot spawn via the 'home' keyframe.

Object layout reuses scene_gen.sample_scene so seeds match the brain dashboard.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import numpy as np

# locate the installed mujoco_playground G1 xmls on the pod
import mujoco_playground
_PG = os.path.dirname(mujoco_playground.__file__)
G1_XML_DIR = os.path.join(_PG, "_src", "locomotion", "g1", "xmls")
BASE_SCENE = os.path.join(G1_XML_DIR, "scene_mjx_feetonly_flat_terrain.xml")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
from scene_gen import sample_scene, COLORS, SHAPES  # noqa: E402


def _ego_camera():
    return ('<camera name="ego" mode="fixed" pos="0.08 0 0.55" '
            'xyaxes="0 -1 0 0.22 0 1" fovy="95"/>')


def build_integrated_xml(seed: int, out_path: str = None) -> tuple:
    """Return (xml_path, scene). Writes an integrated MJCF for the given seed."""
    scene = sample_scene(seed)
    tree = ET.parse(BASE_SCENE)
    root = tree.getroot()

    # The G1 body lives in an INCLUDED file (g1_mjx_feetonly.xml), not this scene,
    # so ET can't see torso_link here. Make a patched copy of the g1 body file
    # with the ego camera added onto torso_link, and point the include at it.
    for inc in root.iter("include"):
        f = inc.get("file")
        if f and "g1_mjx_feetonly" in f and "restricted" not in f:
            g1_src = f if os.path.isabs(f) else os.path.join(G1_XML_DIR, f)
            g1_tree = ET.parse(g1_src)
            for body in g1_tree.iter("body"):
                if body.get("name") == "torso_link":
                    body.append(ET.fromstring(_ego_camera()))
                    break
            patched = os.path.join(G1_XML_DIR, f"_g1_ego_{seed}.xml")
            g1_tree.write(patched)
            inc.set("file", patched)

    # 1) robot spawn via the 'home' keyframe qpos[:7] = x,y,z,qw,qx,qy,qz
    import math
    for key in root.iter("key"):
        if key.get("name") != "home":
            continue
        vals = key.get("qpos").split()
        z = vals[2]
        hy = scene.robot_yaw / 2.0
        vals[0] = f"{scene.robot_x:.4f}"; vals[1] = f"{scene.robot_y:.4f}"; vals[2] = z
        vals[3] = f"{math.cos(hy):.5f}"; vals[4] = "0"; vals[5] = "0"; vals[6] = f"{math.sin(hy):.5f}"
        key.set("qpos", " ".join(vals))
        break

    wb = root.find("worldbody")

    # (ego camera was injected into the included g1 body file above)

    # 3) third-person overview camera (whole arena)
    h = scene.arena_half
    wb.append(ET.fromstring(
        f'<camera name="thirdperson" mode="fixed" '
        f'pos="0 {-(h + 4.0):.2f} {h + 4.5:.2f}" '
        f'xyaxes="1 0 0 0 0.66 0.75" fovy="55"/>'))

    # 4) arena walls
    wall_h = 0.3
    for i, (cx, cy, sx, sy) in enumerate([
            (0, h, h, 0.05), (0, -h, h, 0.05), (h, 0, 0.05, h), (-h, 0, 0.05, h)]):
        wb.append(ET.fromstring(
            f'<geom name="wall_{i}" type="box" pos="{cx} {cy} {wall_h/2}" '
            f'size="{sx} {sy} {wall_h/2}" rgba="0.5 0.5 0.55 1" '
            f'contype="1" conaffinity="1"/>'))

    # 5) colored objects (static, no freejoint)
    for i, o in enumerate(scene.objects):
        gtype, size, _ = SHAPES[o.shape]
        size_str = " ".join(str(s) for s in size if s > 0) or "0.05"
        rgba = " ".join(str(c) for c in COLORS[o.color])
        wb.append(ET.fromstring(
            f'<geom name="obj_{i}_{o.color}_{o.shape}" type="{gtype}" '
            f'pos="{o.x} {o.y} {o.z}" size="{size_str}" rgba="{rgba}" '
            f'contype="1" conaffinity="1"/>'))

    # write INTO the playground xml dir so all relative meshdir/include paths
    # (assets/*.STL, sensor.xml, g1_mjx_feetonly.xml) resolve correctly.
    out_path = out_path or os.path.join(G1_XML_DIR, f"_integrated_{seed}.xml")
    tree.write(out_path)
    return out_path, scene


if __name__ == "__main__":
    import mujoco
    for seed in [0, 1]:
        path, scene = build_integrated_xml(seed)
        m = mujoco.MjModel.from_xml_path(path)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
        print(f"seed{seed}: VALID nq={m.nq} nu={m.nu} ncam={m.ncam} {cams} "
              f"base=({d.qpos[0]:.2f},{d.qpos[1]:.2f},z={d.qpos[2]:.2f}) "
              f"objs={[o.label for o in scene.objects]}")
