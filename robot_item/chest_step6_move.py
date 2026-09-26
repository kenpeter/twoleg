#!/usr/bin/env python3
"""Render chest_assembly_step6.mp4 / _physics.mp4 — chest + arms moving.

Source: ``chest_assembly_step6.xml`` (chest step5 plus the left arm seated on the
slantU_L prong yoke).  The file is a pose, so it is re-rigged here into the servo
chain the hardware actually has.  Every hinge sits on a measured axis: the pivot is
each 金属舵盘's disc centre and the axis is the disc's own 5 mm direction, so a
correct render means the kinematics match the hardware.

Chain, from the world outward:

    chest_servo_L / chest_servo_R   X-axis abduction.  The rotor is the horn, and
                                   the 斜U wing is bolted through the web bore onto
                                   it, so the whole arm side swings with it.
    waist_servo_L / waist_servo_R   Y-axis.  The legs are not part of step6, so the
                                   rotor is just the horn (it still turns).
    head_servo                      Z-axis head tilt; nothing is mounted on its
                                   output in this file, so the head box rocks.
    left arm (inside left_arm):
        shoulder    Y through the prong-bore line, world X=-0.09805 Z=0.08905
        elbow       Y through the 长U prong bore
        wrist       Y through the forearm's 金属舵盘

horn_2 (its four retaining screws and the shoulder shaft) stays welded to the
shoulder yoke: it is the shoulder servo's output, bolted to the fixed prong ear, so
the servo housing + multi_0 + 长U + forearm are what swing about it.  Likewise
horn_2_f is the wrist servo's output and carries the 一字, so the wrist hinge turns
the horn and the 一字 while the forearm holds still.

Run:
    uv run --with numpy --with mujoco --with imageio --with imageio-ffmpeg \
        python robot_item/chest_step6_move.py            # chest_assembly_step6.mp4
        python robot_item/chest_step6_move.py kinematic  # chest_assembly_step6_kinematic.mp4
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "chest_assembly_step6.xml")

FPS = 30
SECONDS = 8.0
SETTLE_SECONDS = 1.0     # physics only: let the chain reach its t=0 pose before frame 0

# Camera. Azimuth is degrees around the world Z axis: 55 looks the robot in the face,
# 235 is the same shot from behind. Override with AZIMUTH=... python chest_step6_move.py
CAM_AZIMUTH = float(os.environ.get("AZIMUTH", "55"))
CAM_ELEVATION = float(os.environ.get("ELEVATION", "-10"))
CAM_DISTANCE = float(os.environ.get("DISTANCE", "0.52"))
CAM_LOOKAT = [-0.030, -0.010, 0.040]
SHOULDER_DEG = 20.0
ELBOW_DEG = 30.0
WRIST_DEG = 25.0
CHEST_DEG = 15.0        # shoulder flex/extend, world X, both wings in phase
WAIST_DEG = 15.0
HEAD_DEG = 12.0

# ---- measured hinge frames -------------------------------------------------
# Each entry: hinge name, joint axis, pivot (world, metres), and the bodies/geoms
# the rotor carries.  Pivots are 金属舵盘 disc centres; axes are the disc normals.
CHEST_HINGES = {
    # abduction: horn disc normal is world X; the 斜U wing is bolted onto the horn
    "abduct_L": dict(axis=(1.0, 0.0, 0.0), pivot=(-0.06796, -0.003966, 0.067034),
                     bodies=("chest_servo_L_horn", "slantU_L"), geoms=()),
    "abduct_R": dict(axis=(1.0, 0.0, 0.0), pivot=(0.06796, -0.004034, 0.067034),
                     bodies=("chest_servo_R_horn", "slantU_R"), geoms=()),
    # waist: horn disc normal is world Y; the legs live in other files
    "waist_L": dict(axis=(0.0, 1.0, 0.0), pivot=(-0.037034, -0.02896, -0.011966),
                    bodies=("waist_servo_L_horn",), geoms=()),
    "waist_R": dict(axis=(0.0, 1.0, 0.0), pivot=(0.036966, -0.02896, -0.011966),
                    bodies=("waist_servo_R_horn",), geoms=()),
    # head: rocks about the Z-normal horn_center disc
    "head": dict(axis=(0.0, 0.0, 1.0), pivot=(-0.000034, 0.013884, 0.08054),
                 bodies=("head_servo",), geoms=()),
}

# left arm, in the left_arm body's own frame (see chest_assembly_step6.xml header)
SHOULDER_P = np.array([0.016087, 0.0, -0.001945])   # on the prong-bore line
ELBOW_P = np.array([0.013250, 0.000481, -0.081250])
# The wrist pivot is deliberately NOT a constant here. The elbow and shoulder sit in the
# left_arm frame, but the wrist disc is bolted to the forearm, so the same point has two
# different coordinate expressions and hardcoding one of them silently mis-places the
# hinge by the forearm offset. rig() reads both from the tree instead.

# bodies / geoms carried by each arm hinge
SHOULDER_BODIES = ("multi_0", "servo_1", "longU_down", "elbow_bearing", "far_bearing")
SHOULDER_GEOMS = ("base_screw_1", "base_screw_2", "base_screw_3", "base_screw_4",
                  "base_nut_1", "base_nut_2", "base_nut_3", "base_nut_4",
                  "elbow_shaft", "far_shaft")
ELBOW_BODIES = ("forearm",)
# The wrist servo's output is horn_2_f and nothing else.  straight_3_f (the 一字)
# is fixed to the forearm: it sits 14.6..72.8 mm off the wrist axis, so putting it
# on the rotor swung it through a 73 mm arc and tore the hand off the arm.
WRIST_BODIES = ("horn_2_f",)
WRIST_GEOMS = ("horn_s0_f", "horn_s1_f", "horn_s2_f", "horn_s3_f")
# grounded to the shoulder yoke (the shoulder servo's output)
GROUNDED = ("horn_2",)

ACTUATORS = ("abduct_L", "abduct_R", "waist_L", "waist_R", "head",
             "shoulder_test", "elbow_test", "wrist_test")


def _vec(a):
    return " ".join(f"{v:.6f}" for v in a)


def _pos(el):
    return np.array([float(v) for v in el.get("pos", "0 0 0").split()])


def _prepare(physics):
    """Parse step6, kill the pose-only bits, and set up dynamics."""
    # XML comments here contain runs of '-', which strict parsers reject.
    txt = re.sub(r"<!--.*?-->", "", open(SRC, encoding="utf-8").read(), flags=re.S)
    root = ET.fromstring(txt)
    root.find("compiler").set("meshdir", HERE)
    opt = ET.SubElement(root, "option")
    if physics:
        opt.set("gravity", "0 0 -9.81")
        opt.set("timestep", "0.001")
        opt.set("iterations", "100")
    else:
        opt.set("gravity", "0 0 0")
    # the chest and each arm are bolted together, so nothing needs to collide;
    # leaving the meshes out of the contact set avoids self-collision jitter
    # (same choice chest_move.py makes for the welded parts).
    for g in root.iter("geom"):
        g.set("contype", "0")
        g.set("conaffinity", "0")
    return root


def _take(parent, name, origin):
    """Move a body/geom out of the tree under `parent` and re-base its pos.

    The element's pos is relative to its current parent's origin; once it hangs
    under a hinge at `origin` it has to be expressed relative to that hinge, or it
    snaps away from where it belongs (this is what arm_move.py's _shift does).
    `origin` is in the same frame the pos was authored in (world for chest-level
    bodies, the left_arm frame for arm parts).
    """
    for parent_el in parent.iter():
        for ch in list(parent_el):
            if ch.get("name") != name:
                continue
            parent_el.remove(ch)
            pos = np.array([float(v) for v in ch.get("pos", "0 0 0").split()])
            ch.set("pos", _vec(pos - origin))
            return ch
    raise KeyError(name)


def _hinge(name, axis, pivot, parent, origin_world):
    body = ET.Element("body", {"name": f"{name}_link", "pos": _vec(pivot)})
    ET.SubElement(body, "joint", {"name": name, "type": "hinge",
                                  "axis": _vec(axis), "damping": "0.05"})
    return body


def rig(root):
    wb = root.find("worldbody")
    arm = wb.find("body[@name='left_arm']")

    # Read both wrist frames before any _take re-bases the tree. forearm_pos is the
    # forearm origin in the left_arm frame; horn_pos is the wrist disc centre in the
    # forearm frame. The hinge is placed with their sum (left_arm frame) while the disc
    # and its screws are re-based with horn_pos (forearm frame). Reading both from the
    # tree is what keeps the disc on the servo instead of 91 mm away from it.
    forearm = arm.find("body[@name='forearm']")
    forearm_pos = _pos(forearm)
    horn_pos = _pos(forearm.find("body[@name='horn_2_f']"))
    wrist_p = forearm_pos + horn_pos

    # --- chest / waist / head hinges, all children of the world -------------
    for name in ("waist_L", "waist_R", "head", "abduct_R"):
        cfg = CHEST_HINGES[name]
        pivot = np.array(cfg["pivot"])
        h = _hinge(name, cfg["axis"], pivot, wb, pivot)
        for nm in cfg["bodies"] + cfg["geoms"]:
            h.append(_take(wb, nm, pivot))
        wb.append(h)
    # abduct_L must also carry the whole left arm, so build it last and move the
    # arm in as one rigid child.
    cfg = CHEST_HINGES["abduct_L"]
    pivot = np.array(cfg["pivot"])
    abduct_L = _hinge("abduct_L", cfg["axis"], pivot, wb, pivot)
    for nm in cfg["bodies"] + cfg["geoms"]:
        abduct_L.append(_take(wb, nm, pivot))
    abduct_L.append(_take(wb, "left_arm", pivot))
    wb.append(abduct_L)

    # --- left-arm hinges, inside left_arm ----------------------------------
    shoulder = _hinge("shoulder_test", (0, 1, 0), SHOULDER_P, arm, SHOULDER_P)
    for nm in SHOULDER_BODIES + SHOULDER_GEOMS:
        shoulder.append(_take(arm, nm, SHOULDER_P))

    elbow = _hinge("elbow_test", (0, 1, 0), ELBOW_P - SHOULDER_P, arm, SHOULDER_P)
    for nm in ELBOW_BODIES:
        elbow.append(_take(arm, nm, ELBOW_P))

    wrist = _hinge("wrist_test", (0, 1, 0), wrist_p - ELBOW_P, arm, SHOULDER_P)
    fw = elbow.find("body[@name='forearm']")
    for nm in WRIST_BODIES + WRIST_GEOMS:
        wrist.append(_take(fw, nm, horn_pos))

    elbow.append(wrist)
    shoulder.append(elbow)
    arm.append(shoulder)
    return root


def build(dst, physics):
    root = _prepare(physics)
    rig(root)
    if physics:
        act = ET.SubElement(root, "actuator")
        for n in ACTUATORS:
            ET.SubElement(act, "position", {"name": n, "joint": n, "kp": "6",
                                            "dampratio": "1",
                                            "ctrlrange": "-1.4 1.4"})
    ET.ElementTree(root).write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def _targets(t):
    """Servo commands, one full cycle over the clip.

    Both chest hinges rotate about world X and both waist hinges about world Y.
    The left and right wings are mirror images across X=0, and a rotation about X
    (or Y) moves a point and its mirror identically, so each L/R pair takes the
    SAME sign.  Running them in antiphase swings one wing up while the other goes
    down, which is not a motion this robot can do.
    """
    return {
        "abduct_L": np.radians(CHEST_DEG) * np.sin(2 * np.pi * t),
        "abduct_R": np.radians(CHEST_DEG) * np.sin(2 * np.pi * t),
        "waist_L": np.radians(WAIST_DEG) * np.sin(2 * np.pi * t - np.pi / 2),
        "waist_R": np.radians(WAIST_DEG) * np.sin(2 * np.pi * t - np.pi / 2),
        "head": np.radians(HEAD_DEG) * np.sin(2 * np.pi * t + np.pi / 2),
        "shoulder_test": np.radians(SHOULDER_DEG) * np.sin(2 * np.pi * t - np.pi / 2),
        "elbow_test": np.radians(ELBOW_DEG) * np.sin(2 * np.pi * t),
        "wrist_test": np.radians(WRIST_DEG) * np.sin(2 * np.pi * t + np.pi / 2),
    }


def render(path, out, physics):
    import imageio.v2 as imageio
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    qadr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ACTUATORS}
    act = ({j: m.actuator(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j)).id
            for j in ACTUATORS} if physics else {})

    r = mujoco.Renderer(m, 900, 620)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = CAM_LOOKAT
    cam.distance = CAM_DISTANCE
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION

    n = int(FPS * SECONDS)
    sub = max(1, int(round(1.0 / (FPS * m.opt.timestep)))) if physics else 1
    err = {j: 0.0 for j in ACTUATORS}
    if physics:
        # The chain starts at rest against gravity with every servo at zero, so the
        # shoulder hangs 9 deg off its command until the position servo winds it in.
        # Settle on the t=0 command before the first frame, otherwise the clip opens
        # with a visible lurch that is a start-up artefact, not the motion.
        for j, v in _targets(0.0).items():
            d.ctrl[act[j]] = v
        for _ in range(int(SETTLE_SECONDS / m.opt.timestep)):
            mujoco.mj_step(m, d)
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=1) as w:
        for i in range(n):
            t = (i / FPS) / SECONDS
            target = _targets(t)
            if physics:
                for j, v in target.items():
                    d.ctrl[act[j]] = v
                for _ in range(sub):
                    mujoco.mj_step(m, d)
                for j, v in target.items():
                    err[j] = max(err[j], abs(d.qpos[qadr[j]] - v))
            else:
                for j, v in target.items():
                    d.qpos[qadr[j]] = v
                mujoco.mj_forward(m, d)
            r.update_scene(d, cam)
            w.append_data(r.render())
    r.close()
    print(f"wrote {out}  ({n} frames, {SECONDS:.0f}s @ {FPS}fps"
          f"{', physics' if physics else ''})")
    print(f"  {len(ACTUATORS)} driven servos, mass {m.body_mass.sum():.4f} kg, "
          f"{m.nbody} bodies")
    if physics:
        worst = max(err.items(), key=lambda kv: kv[1])
        print(f"  max tracking error under dynamics: {np.degrees(worst[1]):.2f} deg "
              f"({worst[0]})")


def main():
    kinematic = "kinematic" in sys.argv[1:] or "--kinematic" in sys.argv[1:]
    physics = not kinematic
    tag = "_kinematic" if kinematic else ""
    render(build(f"/tmp/opencode/_chest_step6{tag}.xml", physics),
           os.path.join(HERE, f"chest_assembly_step6{tag}.mp4"), physics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
