#!/usr/bin/env python3
"""Render left_arm.mp4 — the left arm (tricep + forearm) moving.

Source: ``left_arm.xml`` = ``left_tricep.xml`` (upper arm: 多功能 + 舵机 + 金属舵盘 +
长U) joined to ``left_forearm.xml`` (lower arm: 多功能 + 舵机 + 金属舵盘 + 一字) at the
elbow.

The static assembly is re-rigged as ``shoulder_link -> elbow_link -> forearm``
with both hinges on the measured world-Y axes:

    shoulder  Y through (0.012750, -0.007000,  0.010000)   # tricep horn
    elbow     Y through (0.013700,  0.000500, -0.080800)   # 长U prong pivot shaft

so a correct render means the kinematics match the hardware.

Run:
    uv run --with numpy --with mujoco --with imageio --with imageio-ffmpeg \
        python robot_item/arm_move.py            # left_arm.mp4
        python robot_item/arm_move.py physics    # left_arm_physics.mp4
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))

FPS = 30
SECONDS = 8.0
SHOULDER_DEG = 25.0
ELBOW_DEG = 50.0

SHOULDER_P = np.array([0.012750, -0.007000, 0.010000])
ELBOW_P = np.array([0.013700, 0.000500, -0.080800])
TRICEP_BODIES = ["multi_0", "servo_1", "horn_2", "longU_down"]


def _vec(a):
    return " ".join(f"{v:.6f}" for v in a)


def _shift(el, origin, parent):
    pos = np.array([float(v) for v in el.get("pos", "0 0 0").split()])
    el.set("pos", _vec(pos - origin))
    parent.append(el)


def build(dst, physics=False):
    # XML comments here contain runs of '-', which strict parsers reject.
    txt = re.sub(r"<!--.*?-->", "", open(os.path.join(HERE, "left_arm.xml")).read(),
                 flags=re.S)
    root = ET.fromstring(txt)
    root.find("compiler").set("meshdir", HERE)
    opt = ET.SubElement(root, "option")
    opt.set("gravity", "0 0 -9.81" if physics else "0 0 0")
    if physics:
        opt.set("timestep", "0.001")
    wb = root.find("worldbody")

    shoulder = ET.Element("body", {"name": "shoulder_link", "pos": _vec(SHOULDER_P)})
    ET.SubElement(shoulder, "joint", {"name": "shoulder_test", "type": "hinge",
                                      "axis": "0 1 0", "damping": "0.1"})
    elbow = ET.Element("body", {"name": "elbow_link", "pos": _vec(ELBOW_P - SHOULDER_P)})
    ET.SubElement(elbow, "joint", {"name": "elbow_test", "type": "hinge",
                                   "axis": "0 1 0", "damping": "0.1"})

    for b in list(wb.findall("body")):
        if b.get("name") in TRICEP_BODIES:
            wb.remove(b)
            _shift(b, SHOULDER_P, shoulder)
        elif b.get("name") == "forearm":
            wb.remove(b)
            _shift(b, ELBOW_P, elbow)
    # top-level geoms (screws, elbow bearing/shaft) belong to the tricep side
    for g in list(wb.findall("geom")):
        wb.remove(g)
        _shift(g, SHOULDER_P, shoulder)

    shoulder.append(elbow)
    wb.append(shoulder)

    if physics:
        act = ET.SubElement(root, "actuator")
        for n in ("shoulder_test", "elbow_test"):
            ET.SubElement(act, "position", {"name": n, "joint": n, "kp": "6",
                                            "dampratio": "1", "ctrlrange": "-1.4 1.4"})

    ET.ElementTree(root).write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def render(path, out, physics=False):
    import imageio.v2 as imageio
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    qa = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
          for j in ("shoulder_test", "elbow_test")}
    act = {j: m.actuator(j).id for j in ("shoulder_test", "elbow_test")} if physics else {}
    r = mujoco.Renderer(m, 620, 620)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.006, -0.003, -0.050]
    cam.distance = 0.34
    cam.azimuth = 300
    cam.elevation = -8

    sub = max(1, int(round(1.0 / (FPS * m.opt.timestep))))
    n = int(FPS * SECONDS)
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=1) as w:
        for i in range(n):
            t = (i / FPS) / SECONDS
            tgt = {
                "shoulder_test": np.radians(SHOULDER_DEG) * np.sin(2 * np.pi * t),
                "elbow_test": np.radians(ELBOW_DEG) * np.sin(2 * np.pi * t - np.pi / 2),
            }
            if physics:
                for j, v in tgt.items():
                    d.ctrl[act[j]] = v
                for _ in range(sub):
                    mujoco.mj_step(m, d)
            else:
                for j, v in tgt.items():
                    d.qpos[qa[j]] = v
                mujoco.mj_forward(m, d)
            r.update_scene(d, cam)
            w.append_data(r.render())
    r.close()
    print(f"wrote {out}  ({n} frames, {SECONDS:.0f}s @ {FPS}fps"
          f"{', physics' if physics else ''})")


def main():
    physics = "physics" in sys.argv[1:] or "--physics" in sys.argv[1:]
    tag = "_physics" if physics else ""
    render(build(f"/tmp/opencode/_arm_move{tag}.xml", physics),
           os.path.join(HERE, f"left_arm{tag}.mp4"), physics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
