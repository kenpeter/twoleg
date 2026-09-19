#!/usr/bin/env python3
"""Render MP4s of the left leg joints moving.

Two clips, driven by a position servo on a throw-away hinge:
  * leg_assembly_left_move.mp4       - knee / little-U upper joint
  * leg_assembly_left_ankle_move.mp4 - ankle / bottom servo joint

Run:  uv run --with numpy --with mujoco --with imageio --with imageio-ffmpeg \
          python robot_item/leg_move_video.py
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_leg_align import build_hinged_copy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
SECONDS = 4.0
SWING_DEG = 35.0


def _disable_collisions(root):
    for g in root.iter("geom"):
        g.set("contype", "0")
        g.set("conaffinity", "0")


def add_actuator_and_write(tree, dst, joint):
    root = tree.getroot()
    root.find("compiler").set("meshdir", HERE)
    act = ET.SubElement(root, "actuator")
    ET.SubElement(act, "position", {"joint": joint, "kp": "8",
                                    "dampratio": "1", "ctrlrange": "-1.2 1.2"})
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def build_ankle(dst):
    """Rotate the shin (little-U + everything above) about the ankle axis."""
    P = np.array([-0.007, 0.010, 0.01675])          # ankle disc centre / axis point
    tree = ET.parse(os.path.join(HERE, "leg_assembly_left.xml"))
    root = tree.getroot()
    _disable_collisions(root)
    wb = root.find("worldbody")
    foot = next(b for b in wb.findall("body") if b.get("name") == "foot")

    link = ET.Element("body", {"name": "ankle_link", "pos": " ".join(f"{v:.6f}" for v in P)})
    ET.SubElement(link, "joint", {"name": "ankle_test", "type": "hinge", "axis": "1 0 0",
                                  "damping": "0.02"})

    def reparent(el, parent):
        pos = np.array([float(v) for v in el.get("pos", "0 0 0").split()])
        el.set("pos", " ".join(f"{v:.6f}" for v in (pos - P)))
        parent.remove(el)
        link.append(el)

    for nm in ("shortU_up", "shortU_down"):
        reparent(next(b for b in foot.findall("body") if b.get("name") == nm), foot)
    for g in list(foot.findall("geom")):
        if g.get("name", "").startswith(("screw_", "nut_")):
            reparent(g, foot)
    for nm in ("multi_0", "servo_1", "horn_2", "multi_3", "servo_4", "horn_5",
               "knee_pivot_shaft", "knee_pivot_bearing"):
        reparent(next(b for b in wb.findall("body") if b.get("name") == nm), wb)

    wb.insert(0, link)
    return add_actuator_and_write(tree, dst, "ankle_test")


def build_knee(dst):
    tree = ET.parse(build_hinged_copy("/tmp/opencode/_leg_knee.xml"))
    return add_actuator_and_write(tree, dst, "knee_test")


def render(path, out, lookat, dist, az, el, joint):
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 720, 540)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.azimuth = az
    cam.elevation = el

    import imageio.v2 as imageio
    n = int(FPS * SECONDS)
    amp = np.radians(SWING_DEG)
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as w:
        for i in range(n):
            d.ctrl[0] = amp * np.sin(2 * np.pi * (i / FPS) / SECONDS)
            for _ in range(max(1, int(round(1.0 / (FPS * m.opt.timestep))))):
                mujoco.mj_step(m, d)
            r.update_scene(d, cam)
            w.append_data(r.render())
    r.close()
    print(f"wrote {out}  ({n} frames, {SECONDS:.0f}s @ {FPS}fps)")


def main():
    render(build_knee("/tmp/opencode/_knee_v.xml"),
           os.path.join(HERE, "leg_assembly_left_move.mp4"),
           [0.0175, -0.005, 0.06], 0.30, 230, -6, "knee_test")
    render(build_ankle("/tmp/opencode/_ankle_v.xml"),
           os.path.join(HERE, "leg_assembly_left_ankle_move.mp4"),
           [0.0175, 0.010, 0.045], 0.34, 200, 2, "ankle_test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
