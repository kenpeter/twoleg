#!/usr/bin/env python3
"""status.py: dummy-proof inspector for chest_both_arms_legs.xml.

Shows the robot in 3D with data points, actuators, and which part can move.

Usage (run from repo root):
  training/.venv/bin/python status.py --table
      Print the data table only: every joint, its body, axis, range,
      whether it has a motor, plus sensors and foot gap. No window.
  training/.venv/bin/python status.py
      Open the interactive 3D window AND auto-tour: each movable joint
      wiggles on its own for 4 seconds while its name prints in the
      terminal. Drag with the mouse to rotate. Right panel has one
      slider per motor for manual play.

A joint is a bend point. An actuator (motor) is what moves a joint.
A sensor is how the robot feels tilt and motion.
"""
import math
import os
import sys
from pathlib import Path

try:
    import mujoco  # noqa: F401
except ImportError:
    here = Path(__file__).resolve().parent
    for cand in (here / "training" / ".venv" / "bin" / "python",):
        if cand.is_file() and os.access(cand, os.X_OK):
            os.execv(str(cand), [str(cand), str(here / "status.py"),
                                 *sys.argv[1:]])
    sys.exit("error: no mujoco found. Expected training/.venv/bin/python "
             "next to status.py at the repo top.")

import numpy as np

import mujoco

MODEL = str(Path(__file__).resolve().parent
            / "robot_item" / "xml" / "chest_both_arms_legs.xml")

POSE_TARGETS = (
    "L_hip_roll_test", "L_hip_test", "L_knee_test", "L_ankle_test",
    "R_hip_roll_test", "R_hip_test", "R_knee_test", "R_ankle_test",
    "head_test",
    "L_hand_1", "L_hand_2", "L_hand_3",
    "R_hand_1", "R_hand_2", "R_hand_3",
)

def load_model():
    spec = mujoco.MjSpec.from_file(MODEL)
    names = {j.name for j in spec.joints}
    for t in POSE_TARGETS:
        if t not in names:
            continue
        a = spec.add_actuator()
        a.name = t
        a.target = t
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm = np.zeros(10)
        a.gainprm[0] = 40.0
        a.biasprm = np.zeros(10)
        a.biasprm[1] = -40.0
        a.biasprm[2] = -0.5
        a.ctrllimited = True
        a.ctrlrange = [-1.4, 1.4]
        a.forcerange = [-20.0, 20.0]
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[5, 5, 0.1])
    return spec.compile()


def joint_rows(m):
    J = mujoco.mjtObj.mjOBJ_JOINT
    A = mujoco.mjtObj.mjOBJ_ACTUATOR
    motors = {mujoco.mj_id2name(m, A, i) for i in range(m.nu)}
    rows = []
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, J, j)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            rows.append((name, "whole robot", "moves in space", "-", "free base"))
        else:
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                     m.jnt_bodyid[j])
            axis = [round(float(v), 3) for v in m.jnt_axis[j]]
            rng = [round(float(v), 3) for v in m.jnt_range[j]]
            rows.append((name, body, axis, rng,
                         "MOTOR" if name in motors else "frozen"))
    return rows


def print_table(m):
    print(f"\nmodel: {MODEL}")
    print(f"bodies={m.nbody} joints={m.njnt} motors={m.nu} "
          f"sensors={m.nsensor}")
    print(f"{'joint':<18}{'on body':<22}{'axis':<18}{'range'}{chr(10)}"
          + "-" * 80)
    for name, body, axis, rng, state in joint_rows(m):
        print(f"{name:<18}{body:<22}{str(axis):<18}{rng}  {state}")
    print("-" * 80)
    print("tour order:", ", ".join(t for t in POSE_TARGETS))
    d = mujoco.MjData(m)
    d.qpos[2] = 0.253
    mujoco.mj_forward(m, d)
    G = mujoco.mjtObj.mjOBJ_GEOM
    spans = {}
    for i in range(m.ngeom):
        n = mujoco.mj_id2name(m, G, i)
        if n in ("left_foot_collision", "right_foot_collision"):
            e = np.abs(d.geom_xmat[i].reshape(3, 3)) @ m.geom_size[i]
            spans[n] = (d.geom_xpos[i][0] - e[0], d.geom_xpos[i][0] + e[0])
    gap = (spans["right_foot_collision"][0]
           - spans["left_foot_collision"][1]) * 1000
    print(f"foot inner-edge gap: {gap:.1f} mm, rest contacts: {d.ncon}")


def tour_sequence():
    return list(POSE_TARGETS)


def mirror_ctrl_to_qpos(m, d, act_index, qadr):
    A = mujoco.mjtObj.mjOBJ_ACTUATOR
    for name, i in act_index.items():
        if name in qadr:
            j = qadr[name]
            lo, hi = m.jnt_range[j]
            d.qpos[m.jnt_qposadr[j]] = min(hi, max(lo, d.ctrl[i]))


def run_viewer():
    import mujoco.viewer
    m = load_model()
    d = mujoco.MjData(m)
    d.qpos[2] = 0.253
    J = mujoco.mjtObj.mjOBJ_JOINT
    A = mujoco.mjtObj.mjOBJ_ACTUATOR
    qadr = {}
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, J, j)
        if n:
            qadr[n] = j
    act_index = {mujoco.mj_id2name(m, A, i): i for i in range(m.nu)}
    mujoco.mj_forward(m, d)
    print_table(m)
    print("\nLIVE POSE: drag the right-panel sliders, "
          "the joints follow. No physics runs, so it cannot fall. "
          "Drag mouse to rotate, wheel to zoom.")
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = np.array([0.0, -0.05, 0.0])
        v.cam.distance = 1.2
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        while v.is_running():
            mirror_ctrl_to_qpos(m, d, act_index, qadr)
            mujoco.mj_forward(m, d)
            v.sync()


if __name__ == "__main__":
    if "--table" in sys.argv:
        print_table(load_model())
    elif "--tour-list" in sys.argv:
        print("\n".join(tour_sequence()))
    else:
        run_viewer()
