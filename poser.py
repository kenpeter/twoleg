#!/usr/bin/env python3
"""poser.py: toy-figure poser for chest_both_arms_legs.xml.

Bend any joint by hand and see problems early, before training.
No physics runs here, so the robot cannot fall over. Posing only.

Usage (run from repo root, plain python3 is fine):
  python3 poser.py check
      Early-problem scan at zero pose: contacts, foot gap,
      balance point vs feet, motor coverage. No window.
  python3 poser.py set L_knee_test=0.5 L_hand_1=0.3
      Bend joints, print the same scan for that pose,
      save poser_pose.png (front view). Add --view side|top.
  python3 poser.py --gui
      Window with one slider per joint. Needs a screen.

A joint is a bend point. An actuator (motor) is what moves a joint.
"""
import os
import sys
from pathlib import Path

try:
    import mujoco  # noqa: F401
except ImportError:
    here = Path(__file__).resolve().parent
    for cand in (here / "training" / ".venv" / "bin" / "python",):
        if cand.is_file() and os.access(cand, os.X_OK):
            os.execv(str(cand), [str(cand), str(here / "poser.py"),
                                 *sys.argv[1:]])
    sys.exit("error: no mujoco found. Expected training/.venv/bin/python.")

import numpy as np

import mujoco

MODEL = str(Path(__file__).resolve().parent
            / "robot_item" / "xml" / "chest_both_arms_legs.xml")
J = mujoco.mjtObj.mjOBJ_JOINT
G = mujoco.mjtObj.mjOBJ_GEOM
B = mujoco.mjtObj.mjOBJ_BODY

POSE_TARGETS = (
    "L_hip_roll_test", "L_hip_test", "L_knee_test", "L_ankle_test",
    "R_hip_roll_test", "R_hip_test", "R_knee_test", "R_ankle_test",
    "head_test",
    "L_hand_1", "L_hand_2", "L_hand_3",
    "R_hand_1", "R_hand_2", "R_hand_3",
)


def hinge_names(m):
    out = []
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            out.append(mujoco.mj_id2name(m, J, j))
    return out


def scan(m, d, label):
    print(f"\n--- {label} ---")
    print(f"contacts at rest: {d.ncon}")
    for i in range(d.ncon):
        a = mujoco.mj_id2name(m, G, d.contact[i].geom1)
        b = mujoco.mj_id2name(m, G, d.contact[i].geom2)
        print(f"  touch: {a} <-> {b} "
              f"depth={-d.contact[i].dist * 1000:.1f}mm")
    spans = {}
    for i in range(m.ngeom):
        n = mujoco.mj_id2name(m, G, i)
        if n in ("left_foot_collision", "right_foot_collision"):
            e = np.abs(d.geom_xmat[i].reshape(3, 3)) @ m.geom_size[i]
            c = d.geom_xpos[i]
            spans[n] = (c[0] - e[0], c[0] + e[0], c[1] - e[1], c[1] + e[1])
    if len(spans) == 2:
        gap = (spans["right_foot_collision"][0]
               - spans["left_foot_collision"][1]) * 1000
        print(f"foot inner-edge gap: {gap:.1f} mm "
              f"({'OK' if gap >= 5 else 'TOO CLOSE'})")
    com = d.subtree_com[0]
    print(f"balance point x={com[0] * 1000:.0f} y={com[1] * 1000:.0f} mm, "
          f"feet y span [{spans['left_foot_collision'][2] * 1000:.0f},"
          f"{spans['left_foot_collision'][3] * 1000:.0f}] mm")
    edge = spans["left_foot_collision"][3]
    margin = (edge - com[1]) * 1000
    print(f"balance margin to foot edge: {margin:.0f} mm "
          f"({'OK' if margin > 10 else 'TIPPY, feet behind balance point'})")


def render(m, d, view, out):
    os.environ.setdefault("MUJOCO_GL", "egl")
    r = mujoco.Renderer(m, 720, 540)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if view == "top":
        cam.lookat[:] = np.array([0, -0.08, -0.25])
        cam.distance = 0.6
        cam.azimuth = 90
        cam.elevation = -89
    elif view == "side":
        cam.lookat[:] = np.array([0, -0.05, 0.0])
        cam.distance = 1.0
        cam.azimuth = 90
        cam.elevation = -5
    else:
        cam.lookat[:] = np.array([0, -0.05, 0.0])
        cam.distance = 1.0
        cam.azimuth = 0
        cam.elevation = -8
    r.update_scene(d, cam)
    from PIL import Image
    Image.fromarray(r.render()).save(out)
    print(f"picture saved: {out}")


def cmd_check():
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    names = set(hinge_names(m))
    injected = sum(1 for t in POSE_TARGETS if t in names)
    print(f"bodies={m.nbody} joints={m.njnt} motors={m.nu} in file "
          f"+ {injected} added by trainer sensors={m.nsensor}")
    scan(m, d, "zero pose")


def cmd_set(pairs, view, out):
    m = mujoco.MjModel.from_xml_path(MODEL)
    names = set(hinge_names(m))
    d = mujoco.MjData(m)
    qa = {}
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, J, j)
        if n:
            qa[n] = (j, m.jnt_qposadr[j])
    for p in pairs:
        name, _, val = p.partition("=")
        if name not in names:
            close = [n for n in names if name.lower() in n.lower()]
            sys.exit(f"error: no joint '{name}'. "
                     f"did you mean: {close[:3]}")
        v = float(val)
        lo, hi = m.jnt_range[qa[name][0]]
        if not (lo <= v <= hi):
            sys.exit(f"error: {name}={v} outside range [{lo:.2f},{hi:.2f}]")
        d.qpos[qa[name][1]] = v
        print(f"{name} = {v}")
    mujoco.mj_forward(m, d)
    scan(m, d, "posed")
    render(m, d, view, out)


def cmd_gui():
    import tkinter as tk
    m = mujoco.MjModel.from_xml_path(MODEL)
    names = hinge_names(m)
    pose = {n: 0.0 for n in names}
    d = mujoco.MjData(m)
    qa = {}
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, J, j)
        if n:
            qa[n] = (j, m.jnt_qposadr[j])
    root = tk.Tk()
    root.title("twoleg poser (no physics, cannot fall)")
    sliders = {}

    def redraw(_=None):
        for n, s in sliders.items():
            d.qpos[qa[n][1]] = s.get()
        mujoco.mj_forward(m, d)
        scan(m, d, "posed")

    def snap():
        render(m, d, "front", "poser_pose.png")

    for n in names:
        lo, hi = (float(m.jnt_range[qa[n][0]][0]),
                  float(m.jnt_range[qa[n][0]][1]))
        f = tk.Frame(root)
        f.pack(fill="x")
        tk.Label(f, text=n, width=18, anchor="w").pack(side="left")
        s = tk.Scale(f, from_=lo, to=hi, resolution=0.01,
                     orient="horizontal", length=300, command=redraw)
        s.pack(side="left")
        sliders[n] = s
    tk.Button(root, text="save picture",
              command=snap).pack(pady=8)
    redraw()
    root.mainloop()


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "check":
        cmd_check()
    elif a[0] == "set":
        view = "front"
        rest = []
        for x in a[1:]:
            if x.startswith("--view="):
                view = x.split("=", 1)[1]
            else:
                rest.append(x)
        cmd_set(rest, view, "poser_pose.png")
    elif a[0] == "--gui":
        cmd_gui()
    else:
        sys.exit("usage: poser.py check | set NAME=val ... [--view=side|top|front] | --gui")
