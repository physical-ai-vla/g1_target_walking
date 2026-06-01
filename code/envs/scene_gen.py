"""Procedural arena scene generator for G1Nav.

Builds a MuJoCo MJCF (as a string / temp file) that contains:
  - the Unitree G1 (included from menagerie, with cameras added via a wrapper)
  - a bounded arena floor + low walls
  - N brightly-colored objects (ball / cube / cylinder / cone) at random,
    non-overlapping positions
  - an egocentric RGBD camera on the torso (head height, looking forward)
  - a fixed third-person camera for video

Everything is seeded so a (seed) -> exact scene mapping holds. This is what
makes dataset/ regenerable from a manifest of seeds.

The generator does NOT depend on a GPU. It only needs `mujoco` to validate.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G1_DIR = os.path.join(ROOT, "assets", "mujoco_menagerie", "unitree_g1")
G1_XML = os.path.join(G1_DIR, "g1.xml")

# ---- object palette -------------------------------------------------------
# (name, mujoco geom type, nominal size triplet, default rgba)
COLORS = {
    "red":    (0.85, 0.10, 0.10, 1.0),
    "yellow": (0.90, 0.85, 0.10, 1.0),
    "blue":   (0.10, 0.25, 0.85, 1.0),
    "green":  (0.10, 0.70, 0.20, 1.0),
    "orange": (0.95, 0.45, 0.05, 1.0),
    "purple": (0.55, 0.10, 0.75, 1.0),
}
SHAPES = {
    # shape_name -> (geom_type, size, z_offset_to_rest_on_floor)
    "ball":     ("sphere",   (0.09, 0, 0),        0.09),
    "cube":     ("box",      (0.08, 0.08, 0.08),  0.08),
    "cylinder": ("cylinder", (0.07, 0.12, 0),     0.12),
    "cone":     ("ellipsoid",(0.08, 0.08, 0.14),  0.14),  # MJCF has no cone; ellipsoid stand-in
}


@dataclass
class ObjSpec:
    color: str
    shape: str
    x: float
    y: float
    z: float

    @property
    def label(self) -> str:
        return f"{self.color} {self.shape}"


@dataclass
class SceneSpec:
    seed: int
    objects: list = field(default_factory=list)
    arena_half: float = 3.0          # arena spans [-half, half] in x and y
    robot_yaw: float = 0.0
    robot_x: float = 0.0             # robot spawn position
    robot_y: float = 0.0

    def to_dict(self):
        d = asdict(self)
        return d


def sample_scene(seed: int, n_min: int = 3, n_max: int = 3,
                 arena_half: float = 3.0, min_sep: float = 0.7,
                 max_view_halfangle_deg: float = 36.0,
                 near: float = 1.3, far_frac: float = 0.85) -> SceneSpec:
    """Deterministically sample a scene from a seed.

    The robot starts AT A WALL of the arena, facing inward; the objects are then
    placed at RANDOM positions in front of it (inside the arena, within the ego
    camera's field of view), so all 3 are visible at spawn and lie ahead to walk
    toward.

    Steps:
      1. pick a random wall + random point along it -> robot spawn
      2. robot faces the arena interior (toward the opposite wall), with a small
         random yaw jitter
      3. place 3 objects at random range/bearing in the frontal cone, kept inside
         the arena and non-overlapping
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(n_min, n_max + 1))

    identities = [(c, s) for c in COLORS for s in SHAPES]
    rng.shuffle(identities)
    identities = identities[:n]

    # 1) robot on a random wall, 2) facing inward (+ jitter)
    wall = int(rng.integers(4))          # 0:-x, 1:+x, 2:-y, 3:+y
    margin = 0.35
    span = arena_half - 0.8
    if wall == 0:    # left wall (-x), face +x
        rx, ry, base_yaw = -arena_half + margin, float(rng.uniform(-span, span)), 0.0
    elif wall == 1:  # right wall (+x), face -x
        rx, ry, base_yaw = arena_half - margin, float(rng.uniform(-span, span)), np.pi
    elif wall == 2:  # bottom wall (-y), face +y
        rx, ry, base_yaw = float(rng.uniform(-span, span)), -arena_half + margin, np.pi / 2
    else:            # top wall (+y), face -y
        rx, ry, base_yaw = float(rng.uniform(-span, span)), arena_half - margin, -np.pi / 2
    yaw = float(base_yaw + rng.uniform(-0.12, 0.12))

    # 3) objects in the frontal cone, inside the arena, non-overlapping
    half_view = np.deg2rad(max_view_halfangle_deg)
    # how far ahead we can place things before hitting the far wall
    reach = (2 * arena_half - margin) * far_frac
    objs: list[ObjSpec] = []
    placed: list[tuple[float, float]] = []
    for (color, shape) in identities:
        for _ in range(500):
            rng_d = float(rng.uniform(near, max(near + 0.5, reach)))
            bearing = float(rng.uniform(-half_view, half_view))
            x = rx + rng_d * np.cos(yaw + bearing)
            y = ry + rng_d * np.sin(yaw + bearing)
            if abs(x) > arena_half - 0.4 or abs(y) > arena_half - 0.4:
                continue
            if all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 >= min_sep
                   for px, py in placed):
                break
        placed.append((x, y))
        objs.append(ObjSpec(color=color, shape=shape, x=float(x), y=float(y),
                            z=SHAPES[shape][2]))

    return SceneSpec(seed=seed, objects=objs, arena_half=arena_half,
                     robot_yaw=yaw, robot_x=float(rx), robot_y=float(ry))


# ---- MJCF assembly --------------------------------------------------------

def _ego_camera_xml() -> str:
    """Egocentric RGBD camera, mounted on torso at head height, looking +x."""
    # mounted on torso_link; offset puts it roughly at head, tilted slightly down
    # wide FOV, slight downward tilt so floor-level objects sit in the frame.
    # pos a bit higher (head) and looking ~12 deg down toward the frontal fan.
    return (
        '<camera name="ego" mode="fixed" pos="0.08 0 0.55" '
        'xyaxes="0 -1 0 0.22 0 1" fovy="95"/>'
    )


def build_mjcf(scene: SceneSpec) -> str:
    """Return a full MJCF XML string for the given scene.

    Strategy: load g1.xml, inject ego camera into torso_link, then wrap with a
    worldbody that adds floor, walls, third-person camera, and objects.
    """
    tree = ET.parse(G1_XML)
    root = tree.getroot()

    # 1) inject ego camera onto torso_link
    torso = None
    for body in root.iter("body"):
        if body.get("name") == "torso_link":
            torso = body
            break
    if torso is None:
        raise RuntimeError("torso_link not found in g1.xml")
    cam = ET.fromstring(_ego_camera_xml())
    torso.append(cam)

    # 2) set robot spawn pose (x, y, yaw) by overriding the first 7 qpos of the
    #    'stand' keyframe: [x, y, z, qw, qx, qy, qz]. z stays from the keyframe.
    import math
    for key in root.iter("key"):
        qpos = key.get("qpos")
        if qpos is None:
            continue
        vals = qpos.split()
        if len(vals) < 7:
            continue
        z = vals[2]
        half_yaw = scene.robot_yaw / 2.0
        qw, qz = math.cos(half_yaw), math.sin(half_yaw)
        vals[0] = f"{scene.robot_x:.4f}"
        vals[1] = f"{scene.robot_y:.4f}"
        vals[2] = z
        vals[3] = f"{qw:.5f}"; vals[4] = "0"; vals[5] = "0"; vals[6] = f"{qz:.5f}"
        key.set("qpos", " ".join(vals))
        break

    half = scene.arena_half

    # 3) build the additive worldbody pieces (floor/walls/objects/camera)
    extra = ET.Element("worldbody")

    # third-person camera: a fixed high overview that frames the WHOLE arena
    # (robot + all objects + walls), placed back/above one corner looking down.
    h = scene.arena_half
    tpcam = ET.fromstring(
        f'<camera name="thirdperson" mode="fixed" '
        f'pos="0 {-(h + 4.0):.2f} {h + 4.5:.2f}" '
        f'xyaxes="1 0 0 0 0.66 0.75" fovy="55"/>'
    )
    extra.append(tpcam)

    # walls (4 thin boxes) so objects/robot stay in arena
    wall_t = 0.05
    wall_h = 0.3
    walls = [
        (0,  half, half, wall_t),   # +y
        (0, -half, half, wall_t),   # -y
        ( half, 0, wall_t, half),   # +x
        (-half, 0, wall_t, half),   # -x
    ]
    for i, (cx, cy, sx, sy) in enumerate(walls):
        g = ET.fromstring(
            f'<geom name="wall_{i}" type="box" pos="{cx} {cy} {wall_h/2}" '
            f'size="{sx} {sy} {wall_h/2}" rgba="0.5 0.5 0.55 1" contype="1" conaffinity="1"/>'
        )
        extra.append(g)

    # objects
    for i, o in enumerate(scene.objects):
        gtype, size, _ = SHAPES[o.shape]
        size_str = " ".join(str(s) for s in size if s > 0) or "0.05"
        rgba = " ".join(str(c) for c in COLORS[o.color])
        # static objects: no freejoint (scene is fully static per the task)
        g = ET.fromstring(
            f'<geom name="obj_{i}_{o.color}_{o.shape}" type="{gtype}" '
            f'pos="{o.x} {o.y} {o.z}" size="{size_str}" rgba="{rgba}" '
            f'contype="1" conaffinity="1"/>'
        )
        extra.append(g)

    # 4) assemble final document: reuse g1 asset/option/default, add scene assets
    mj = ET.Element("mujoco", attrib={"model": f"g1nav_scene_{scene.seed}"})

    # visual + skybox/ground assets
    vis = ET.fromstring(
        '<visual>'
        '<headlight diffuse="0.6 0.6 0.6" ambient="0.2 0.2 0.2" specular="0.3 0.3 0.3"/>'
        '<rgba haze="0.15 0.25 0.35 1"/>'
        '<global azimuth="140" elevation="-20" offwidth="640" offheight="480"/>'
        '</visual>'
    )
    mj.append(vis)

    scene_assets = ET.fromstring(
        '<asset>'
        '<texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" '
        'width="512" height="3072"/>'
        '<texture type="2d" name="groundplane" builtin="checker" mark="edge" '
        'rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>'
        '<material name="groundplane" texture="groundplane" texuniform="true" '
        'texrepeat="5 5" reflectance="0.2"/>'
        '</asset>'
    )
    mj.append(scene_assets)

    floor = ET.fromstring(
        '<worldbody><geom name="floor" size="0 0 0.05" type="plane" '
        'material="groundplane" contype="1" conaffinity="1"/></worldbody>'
    )
    mj.append(floor)

    # bring over everything from g1.xml (compiler, option, default, asset, worldbody, actuator...)
    # We must set meshdir to absolute so the wrapped doc finds STL files.
    for child in list(root):
        if child.tag == "compiler":
            child.set("meshdir", os.path.join(G1_DIR, "assets"))
        mj.append(child)

    # append scene extras (objects/walls/cameras) as an extra worldbody
    mj.append(extra)

    return ET.tostring(mj, encoding="unicode")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()

    scene = sample_scene(args.seed)
    print(f"[scene_gen] seed={scene.seed} n_objects={len(scene.objects)} "
          f"robot_yaw={scene.robot_yaw:.2f}")
    for o in scene.objects:
        print(f"    {o.label:18s} @ ({o.x:+.2f},{o.y:+.2f})")

    xml = build_mjcf(scene)
    out = args.out or os.path.join(ROOT, "assets", f"scene_seed{args.seed}.xml")
    with open(out, "w") as f:
        f.write(xml)
    print(f"[scene_gen] wrote {out} ({len(xml)} chars)")

    if args.validate:
        import mujoco
        m = mujoco.MjModel.from_xml_path(out)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        ncam = m.ncam
        cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(ncam)]
        print(f"[scene_gen] VALID: nq={m.nq} nu={m.nu} ncam={ncam} cameras={cams}")
