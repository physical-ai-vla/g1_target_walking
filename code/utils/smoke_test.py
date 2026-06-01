"""Local CPU smoke test: load Unitree G1, step physics, render one offscreen frame.

Validates that the MJCF asset + MuJoCo install work before any GPU work.
Runs on Mac (CPU). Not part of training — just a sanity check.
"""
import os
import sys
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE = os.path.join(ROOT, "assets", "mujoco_menagerie", "unitree_g1", "scene.xml")


def main():
    print(f"[smoke] mujoco {mujoco.__version__}")
    print(f"[smoke] loading {SCENE}")
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    print(f"[smoke] nq={model.nq} nv={model.nv} nu={model.nu} (actuators)")
    print(f"[smoke] nbody={model.nbody} ngeom={model.ngeom}")

    # list actuator (joint) names -> these define the action space later
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    print(f"[smoke] actuators ({model.nu}):")
    for n in names:
        print(f"         {n}")

    # step physics a bit from default keyframe if present
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        print(f"[smoke] reset to keyframe 0")
    mujoco.mj_forward(model, data)

    for _ in range(200):
        mujoco.mj_step(model, data)
    print(f"[smoke] stepped 200 steps OK; base height z={data.qpos[2]:.3f}")

    # offscreen render
    try:
        renderer = mujoco.Renderer(model, height=240, width=320)
        renderer.update_scene(data)
        img = renderer.render()
        out = os.path.join(ROOT, "assets", "smoke_frame.png")
        try:
            import imageio.v2 as imageio
            imageio.imwrite(out, img)
            print(f"[smoke] wrote render {out} shape={img.shape}")
        except Exception as e:
            print(f"[smoke] render OK shape={img.shape} but imageio save failed: {e}")
    except Exception as e:
        print(f"[smoke] OFFSCREEN RENDER FAILED: {e}")
        print("[smoke] (physics is fine; rendering backend may need a GL context)")

    print("[smoke] DONE")


if __name__ == "__main__":
    sys.exit(main())
