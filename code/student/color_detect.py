"""Color-based object detector for G1Nav.

The cognition module (GR00T N1.6 Eagle) is good at IDENTIFYING the target by language ("the
orange cylinder") but unreliable at precise 2D pixel coordinates. Since the
arena objects are saturated single colors, we recover an accurate pixel
location by masking the target's color in the ego RGB image and taking the
largest connected blob's centroid + bbox.

This pairs the VLM's strength (language grounding) with classic CV's strength
(precise color segmentation). Distance still comes from the ego depth map at
the detected centroid; bearing from the centroid's pixel-x.

Color reference RGBs mirror scene_gen.COLORS so detection matches what was
rendered.
"""
from __future__ import annotations

import numpy as np

# canonical object colors (0-255 RGB), matching scene_gen.COLORS
COLOR_RGB = {
    "red":    (217, 26, 26),
    "yellow": (230, 217, 26),
    "blue":   (26, 64, 217),
    "green":  (26, 179, 51),
    "orange": (242, 115, 13),
    "purple": (140, 26, 191),
}

_COLOR_WORDS = list(COLOR_RGB.keys())


def color_from_label(label: str):
    """Extract the color word from a label like 'orange cylinder' / 'yellow sphere'."""
    if not label:
        return None
    s = label.lower()
    for c in _COLOR_WORDS:
        if c in s:
            return c
    return None


def _rgb_to_hsv(arr):
    """arr: HxWx3 uint8 -> HxWx3 float HSV (H in [0,360), S,V in [0,1])."""
    a = arr.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.max(a, axis=-1)
    mn = np.min(a, axis=-1)
    df = mx - mn + 1e-8
    h = np.zeros_like(mx)
    mask = mx == r
    h[mask] = (60 * ((g - b) / df) % 360)[mask]
    mask = mx == g
    h[mask] = (60 * ((b - r) / df) + 120)[mask]
    mask = mx == b
    h[mask] = (60 * ((r - g) / df) + 240)[mask]
    s = df / (mx + 1e-8)
    v = mx
    return np.stack([h, s, v], axis=-1)


# hue centers (deg) for each color; we threshold around them with S,V minimums
_HUE = {"red": 0, "orange": 30, "yellow": 55, "green": 130, "blue": 225, "purple": 285}


def detect_color(rgb, color: str, hue_tol: float = 18.0,
                 s_min: float = 0.45, v_min: float = 0.30,
                 max_area_frac: float = 0.15, all_blobs: bool = False):
    """Find blob(s) of `color` in rgb (HxWx3 uint8).

    Rejects background (sky/floor) by: higher saturation floor, and discarding any
    component larger than `max_area_frac` of the image (objects are small).

    Returns the largest valid blob dict {cx,cy,bbox,area}, or None.
    If all_blobs=True, returns a list of ALL valid blobs (used to disambiguate
    same-colored objects).
    """
    if color not in _HUE:
        return None
    rgb = np.asarray(rgb)
    H, W = rgb.shape[:2]
    max_area = max_area_frac * H * W
    hsv = _rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hc = _HUE[color]
    dh = np.abs((h - hc + 180) % 360 - 180)
    mask = (dh <= hue_tol) & (s >= s_min) & (v >= v_min)
    if mask.sum() < 12:
        return [] if all_blobs else None

    blobs = _components(mask)
    valid = []
    for comp in blobs:
        ys, xs = np.where(comp)
        if xs.size < 12 or xs.size > max_area:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        valid.append({
            "cx": float(xs.mean()), "cy": float(ys.mean()),
            "bbox": [x0, y0, x1, y1], "area": int(xs.size),
            "shape": classify_shape(comp, x0, y0, x1, y1),
        })
    valid.sort(key=lambda b: -b["area"])
    if all_blobs:
        return valid
    return valid[0] if valid else None


def classify_shape(comp, x0, y0, x1, y1):
    """Estimate object shape from a blob mask.

    Uses two cues:
      - aspect = width/height of the bounding box
      - fill   = blob_area / bbox_area  (circularity-ish: a ball fills less of
                 its box corners than a cube; a cylinder/cone is tall)
    Returns dict of scores per shape name in scene_gen.SHAPES vocabulary.
    """
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)
    aspect = w / h
    fill = float(comp[y0:y1 + 1, x0:x1 + 1].mean())
    # heuristic scores (higher = more likely). Tuned for the 4 shapes:
    #   ball: ~square box, low-ish fill (round) ; cube: square box, high fill
    #   cylinder: tall (aspect<1), high fill ; cone: tall, tapered (low-mid fill)
    scores = {
        "ball":     _g(aspect, 1.0, 0.35) * _g(fill, 0.78, 0.18),
        "cube":     _g(aspect, 1.0, 0.35) * _g(fill, 0.95, 0.15),
        "cylinder": _g(aspect, 0.6, 0.30) * _g(fill, 0.92, 0.18),
        "cone":     _g(aspect, 0.6, 0.30) * _g(fill, 0.70, 0.20),
    }
    best = max(scores, key=scores.get)
    return {"best": best, "scores": scores, "aspect": round(aspect, 2),
            "fill": round(fill, 2)}


def _g(x, mu, sig):
    """Unnormalized Gaussian similarity."""
    return float(np.exp(-((x - mu) ** 2) / (2 * sig * sig)))


def _components(mask):
    """Return a list of boolean masks, one per 4-connected component."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        return [lab == i for i in range(1, n + 1)]
    except Exception:
        return [mask]  # no scipy: treat as single blob


def shape_from_label(label: str):
    """Extract the shape word from a label like 'orange cylinder'."""
    if not label:
        return None
    s = label.lower()
    for sh in ("cylinder", "cube", "ball", "sphere", "cone"):
        if sh in s:
            return "ball" if sh == "sphere" else sh
    return None


def locate_target(rgb, target_label: str):
    """Locate the instruction's target object precisely in the ego RGB.

    1. mask the target COLOR -> candidate blobs
    2. if exactly one blob: use it
    3. if several (same-color objects): pick the blob whose SHAPE best matches
       the target's shape word; if shape is unknown, fall back to the largest.
    Returns (det, info) where det has cx,cy,bbox,area,shape; info explains choice.
    """
    color = color_from_label(target_label)
    if color is None:
        return None, {"reason": "no color word in target"}
    blobs = detect_color(rgb, color, all_blobs=True)
    if not blobs:
        return None, {"reason": f"color '{color}' not visible"}
    if len(blobs) == 1:
        return blobs[0], {"reason": "single blob of that color", "color": color}

    # several same-colored blobs -> disambiguate by shape
    want = shape_from_label(target_label)
    if want is None:
        return blobs[0], {"reason": "same-color, no shape word -> largest",
                          "color": color, "n_blobs": len(blobs)}
    scored = sorted(blobs, key=lambda b: -b["shape"]["scores"].get(want, 0.0))
    return scored[0], {"reason": f"same-color disambiguated by shape '{want}'",
                       "color": color, "n_blobs": len(blobs),
                       "shape_score": round(scored[0]["shape"]["scores"].get(want, 0), 3)}


def detect_all(rgb):
    """Detect every known color present; returns list of (color, det)."""
    out = []
    for c in COLOR_RGB:
        det = detect_color(rgb, c)
        if det is not None:
            out.append((c, det))
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
    import scene_gen as sg
    import mujoco
    for seed in range(3):
        sc = sg.sample_scene(seed)
        open(f"/tmp/cd{seed}.xml", "w").write(sg.build_mjcf(sc))
        m = mujoco.MjModel.from_xml_path(f"/tmp/cd{seed}.xml")
        d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 240, 320); r.update_scene(d, camera="ego"); rgb = r.render()
        print(f"seed{seed}: objects={[o.label for o in sc.objects]}")
        for c, det in detect_all(rgb):
            print(f"   {c:8s} centroid=({det['cx']:.0f},{det['cy']:.0f}) bbox={det['bbox']} area={det['area']}")
