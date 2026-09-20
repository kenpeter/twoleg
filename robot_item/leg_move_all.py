#!/usr/bin/env python3
"""Render an MP4 per leg with all three servo joints moving together.

The physical joint axes (measured from the assembly / servo outputs):

    left hand side                       right hand side (mirrored in Y)
    ankle  X at (-0.007,  0.010, 0.01675)   (-0.007, -0.010, 0.01675)
    knee   Y at ( 0.0175,-0.014, 0.07875)   ( 0.0175, 0.014, 0.07875)
    hip    Y at ( 0.017, -0.014, 0.15394)   ( 0.017,  0.014, 0.15394)

The static assembly is re-rigged into a chain
``foot -> ankle_link -> knee_link -> hip_link`` and each hinge is placed on the
servo output / little-U wall hole, so a correct render means the model's
kinematics match the hardware.

Rigid segments (a fold only happens where a servo is bolted):

    ankle servo (on the foot)  -> drives the shin (小U)
    knee  servo (on the thigh) -> drives the shin (its horn is in the 小U wall)
    hip   servo (on the thigh) -> drives the top (长U)

Everything else -- in particular the two 多功能支架 -- is bolted solid and must
stay rigid.

Run:
    uv run --with numpy --with mujoco --with imageio --with imageio-ffmpeg \
        python robot_item/leg_move_all.py            # both legs
    uv run --with numpy --with mujoco --with imageio --with imageio-ffmpeg \
        python robot_item/leg_move_all.py left        # one leg
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))

FPS = 30
SECONDS = 8.0
SWING_DEG = 35.0

SIDES = {
    "left": {
        "src": "leg_assembly_left.xml",
        "out": "leg_assembly_left_all_move.mp4",
        "ankle": np.array([-0.0070, 0.0100, 0.01675]),
        "knee": np.array([0.0175, -0.0140, 0.07875]),
        "hip": np.array([0.0170, -0.0140, 0.15394]),
        "sign": 1.0,
        "az": 35.0,
    },
    "right": {
        "src": "leg_assembly_right.xml",
        "out": "leg_assembly_right_all_move.mp4",
        "ankle": np.array([-0.0070, -0.0100, 0.01675]),
        "knee": np.array([0.0175, 0.0140, 0.07875]),
        "hip": np.array([0.0170, 0.0140, 0.15394]),
        "sign": -1.0,
        "az": -35.0,
    },
}

# Rigid segments shared by both (mirrored) assemblies.
SHIN = ["little_u", "horn", "horn_5",
        "knee_pivot_shaft", "knee_pivot_bearing"]
THIGH = ["multi_3", "servo_4", "multi_0", "servo_1",
         "hip_pivot_shaft", "hip_pivot_bearing"]
TOP = ["little_u_top", "horn_2"]
FOOT = ["ankle_pivot_shaft", "ankle_pivot_bearing"]


def _vec(a):
    return " ".join(f"{v:.6f}" for v in a)


def _take(wb, foot, name):
    """Detach a body from the worldbody (or from the foot body) and return it."""
    for parent in (wb, foot):
        for b in parent.findall("body"):
            if b.get("name") == name:
                parent.remove(b)
                return b
    raise KeyError(name)


def _reparent(el, parent, origin):
    pos = np.array([float(v) for v in el.get("pos", "0 0 0").split()])
    el.set("pos", _vec(pos - origin))
    parent.append(el)


def build(side, dst):
    cfg = SIDES[side]
    ankle_p, knee_p, hip_p = cfg["ankle"], cfg["knee"], cfg["hip"]

    tree = ET.parse(os.path.join(HERE, cfg["src"]))
    root = tree.getroot()
    root.find("compiler").set("meshdir", HERE)
    opt = root.find("option")
    if opt is None:
        opt = ET.Element("option")
        root.insert(1, opt)
    opt.set("gravity", "0 0 0")
    for g in root.iter("geom"):
        g.set("contype", "0")
        g.set("conaffinity", "0")

    wb = root.find("worldbody")
    foot = next(b for b in wb.findall("body") if b.get("name") == "foot")

    ankle = ET.Element("body", {"name": "ankle_link", "pos": _vec(ankle_p)})
    ET.SubElement(ankle, "joint", {"name": "ankle_test", "type": "hinge",
                                   "axis": "1 0 0", "damping": "0.05"})
    for nm in SHIN:
        _reparent(_take(wb, foot, nm), ankle, ankle_p)
    foot.append(ankle)

    for nm in FOOT:
        _reparent(_take(wb, foot, nm), foot, np.zeros(3))

    knee = ET.Element("body", {"name": "knee_link", "pos": _vec(knee_p - ankle_p)})
    ET.SubElement(knee, "joint", {"name": "knee_test", "type": "hinge",
                                  "axis": "0 1 0", "damping": "0.05"})
    for nm in THIGH:
        _reparent(_take(wb, foot, nm), knee, knee_p)
    ankle.append(knee)

    hip = ET.Element("body", {"name": "hip_link", "pos": _vec(hip_p - knee_p)})
    ET.SubElement(hip, "joint", {"name": "hip_test", "type": "hinge",
                                 "axis": "0 1 0", "damping": "0.05"})
    for nm in TOP:
        _reparent(_take(wb, foot, nm), hip, hip_p)
    knee.append(hip)

    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def render(side, path, out):
    import imageio.v2 as imageio
    cfg = SIDES[side]
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
           for j in ("ankle_test", "knee_test", "hip_test")]
    r = mujoco.Renderer(m, 720, 540)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0175, 0.0, 0.105]
    cam.distance = 0.44
    cam.azimuth = cfg["az"]
    cam.elevation = -6

    amp = np.radians(SWING_DEG) * cfg["sign"]
    n = int(FPS * SECONDS)
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=1) as w:
        for i in range(n):
            t = (i / FPS) / SECONDS
            d.qpos[adr[0]] = amp * np.sin(2 * np.pi * t)                    # ankle
            d.qpos[adr[1]] = amp * np.sin(2 * np.pi * t - 2 * np.pi / 3)    # knee
            d.qpos[adr[2]] = amp * np.sin(2 * np.pi * t - 4 * np.pi / 3)    # hip
            mujoco.mj_forward(m, d)
            r.update_scene(d, cam)
            w.append_data(r.render())
    r.close()
    print(f"wrote {out}  ({n} frames, {SECONDS:.0f}s @ {FPS}fps)")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    sides = list(SIDES) if which == "both" else [which]
    for side in sides:
        dst = f"/tmp/opencode/_leg_{side}_all.xml"
        out = os.path.join(HERE, SIDES[side]["out"])
        render(side, build(side, dst), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
