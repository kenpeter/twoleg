#!/usr/bin/env python3
"""Render joint-motion videos for the chest + leg assemblies.

Each leg is re-rigged as a chain ``chest -> waist_link -> foot -> ankle_link ->
knee_link -> hip_link`` with a hinge on every servo output / little-U wall hole
(waist abduction axis is world Y through the measured back-prong pivot hole,
``x = +/-0.037449, z = -0.0115``). Everything else stays rigid, so a correct
render means the model's kinematics match the hardware.

Outputs:
  * chest_left_leg.mp4, chest_right_leg.mp4  (one side)
  * chest_both_legs.mp4                      (both legs, symmetric)

Append ``physics`` to build a true dynamic clip instead: gravity, a position
servo on every hinge and ``mj_step`` (left/right legs get separate collision
groups so they cannot pass through each other), e.g. chest_both_legs_physics.mp4.

Run:
    uv run --with numpy --with mujoco --with imageio --with imageio-ffmpeg \
        python robot_item/chest_move.py both            # chest_both_legs.mp4
        python robot_item/chest_move.py both physics    # chest_both_legs_physics.mp4
        python robot_item/chest_move.py left            # chest_left_leg.mp4
        python robot_item/chest_move.py right
        python robot_item/chest_move.py all [physics]   # every clip
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leg_move_all import SIDES, SHIN, THIGH, TOP, FOOT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

FPS = 30
SECONDS = 8.0
WAIST_SWING_DEG = 20.0
SWING_DEG = 35.0

# Waist (abduction) axis: world Y through this point (measured: back-prong
# pivot hole of the leg's top short-U == waist servo shaft line).
WAIST = {
    "left": np.array([-0.037449, -0.001000, -0.011500]),
    "right": np.array([0.037449, -0.001000, -0.011500]),
}
CHEST = {
    "left": "chest_left_leg.xml",
    "right": "chest_right_leg.xml",
}
LEG = {"left": "leg_left", "right": "leg_right"}
CAM = {  # lookat, azimuth; right is the mirror of left
    "left": ([-0.030, -0.010, -0.160], 40.0),
    "right": ([0.030, -0.010, -0.160], -40.0),
}
OUT = {s: os.path.join(HERE, f"chest_{s}_leg.mp4") for s in ("left", "right")}


def _vec(a):
    return " ".join(f"{v:.6f}" for v in a)


def _setup(tree, physics=False):
    root = tree.getroot()
    root.find("compiler").set("meshdir", HERE)
    opt = root.find("option")
    if opt is None:
        opt = ET.Element("option")
        root.insert(1, opt)
    opt.set("gravity", "0 0 -9.81" if physics else "0 0 0")
    if physics:
        opt.set("timestep", "0.001")
        opt.set("iterations", "100")
    for g in root.iter("geom"):
        g.set("contype", "0")
        g.set("conaffinity", "0")
    return root


def _add_actuators(root, names):
    """Position-servo every hinge so the chain is driven under dynamics."""
    act = ET.SubElement(root, "actuator")
    for n in names:
        ET.SubElement(act, "position", {"name": n, "joint": n, "kp": "40",
                                        "dampratio": "1",
                                        "ctrlrange": "-1.4 1.4"})


def _leg_collision(el, contype, conaffinity):
    """Give a leg's solid geoms its own collision group (legs hit each other)."""
    for g in el.iter("geom"):
        if g.get("density") is not None:
            g.set("contype", str(contype))
            g.set("conaffinity", str(conaffinity))


def _take(wb, foot, name):
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


def make_chain(side, leg_pos, leg_quat, sfx):
    """Re-rig one leg as ``waist -> top -> hip -> knee -> ankle -> foot``.

    The top little-U is bolted to the waist horn (rigid with the waist), the hip
    servo folds the thigh below it, the knee servo folds the shin, the ankle
    servo folds the foot. Body/joint names are prefixed with ``sfx``.
    """
    cfg = SIDES[side]
    waist_p = WAIST[side]
    ankle_p, knee_p, hip_p = cfg["ankle"], cfg["knee"], cfg["hip"]

    leg_root = _setup(ET.parse(os.path.join(HERE, cfg["src"])))
    leg_wb = leg_root.find("worldbody")
    leg_foot = next(b for b in leg_wb.findall("body") if b.get("name") == "foot")

    top = ET.Element("body", {"name": "leg_top", "pos": "0 0 0"})
    for nm in TOP:
        _reparent(_take(leg_wb, leg_foot, nm), top, np.zeros(3))

    hip = ET.Element("body", {"name": "hip_link", "pos": _vec(hip_p)})
    ET.SubElement(hip, "joint", {"name": "hip_test", "type": "hinge",
                                 "axis": "0 1 0", "damping": "0.05"})
    for nm in THIGH:
        _reparent(_take(leg_wb, leg_foot, nm), hip, hip_p)
    top.append(hip)

    knee = ET.Element("body", {"name": "knee_link", "pos": _vec(knee_p - hip_p)})
    ET.SubElement(knee, "joint", {"name": "knee_test", "type": "hinge",
                                  "axis": "0 1 0", "damping": "0.05"})
    for nm in SHIN:
        _reparent(_take(leg_wb, leg_foot, nm), knee, knee_p)
    hip.append(knee)

    ankle = ET.Element("body", {"name": "ankle_link",
                                "pos": _vec(ankle_p - knee_p)})
    ET.SubElement(ankle, "joint", {"name": "ankle_test", "type": "hinge",
                                   "axis": "1 0 0", "damping": "0.05"})
    for nm in FOOT:
        _reparent(_take(leg_wb, leg_foot, nm), ankle, ankle_p)
    # Whatever is left in the foot (foot mesh, multi bracket, ankle servo)
    # folds as one rigid piece about the ankle axis.
    for b in list(leg_foot.findall("body")):
        leg_foot.remove(b)
        _reparent(b, ankle, ankle_p)
    leg_foot.set("pos", _vec(-ankle_p))
    ankle.append(leg_foot)
    knee.append(ankle)

    # Hang the chain off the waist hinge; the top stays seated in the waist U
    # while the thigh/shin/foot articulate below it.
    waist = ET.Element("body", {"name": "waist_link", "pos": _vec(waist_p)})
    ET.SubElement(waist, "joint", {"name": "waist_test", "type": "hinge",
                                   "axis": "0 1 0", "damping": "0.05"})
    top.set("pos", _vec(leg_pos - waist_p))
    top.set("quat", leg_quat)
    waist.append(top)

    if sfx:
        for el in waist.iter():
            if el.get("name"):
                el.set("name", sfx + el.get("name"))
    return waist


def _detach_leg(wb, side):
    leg = next(b for b in wb.findall("body") if b.get("name") == LEG[side])
    leg_pos = np.array([float(v) for v in leg.get("pos").split()])
    leg_quat = leg.get("quat")
    wb.remove(leg)
    return leg_pos, leg_quat


def _write(root, dst):
    ET.ElementTree(root).write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def build(dst, side, physics=False):
    root = _setup(ET.parse(os.path.join(HERE, CHEST[side])), physics)
    wb = root.find("worldbody")
    leg_pos, leg_quat = _detach_leg(wb, side)
    chain = make_chain(side, leg_pos, leg_quat, "")
    wb.append(chain)
    if physics:
        _leg_collision(chain, 1, 2)
        _add_actuators(root, ["waist_test", "ankle_test", "knee_test", "hip_test"])
    return _write(root, dst)


def build_both(dst, physics=False):
    root = _setup(ET.parse(os.path.join(HERE, "chest_both_legs.xml")), physics)
    wb = root.find("worldbody")
    geom = {}
    for side in SIDES:
        geom[side] = _detach_leg(wb, side)
    bits = {"left": (1, 2), "right": (2, 1)}
    names = []
    for side in SIDES:
        leg_pos, leg_quat = geom[side]
        sfx = side[0].upper() + "_"
        chain = make_chain(side, leg_pos, leg_quat, sfx)
        if physics:
            _leg_collision(chain, *bits[side])
            names += [sfx + j for j in ("waist_test", "ankle_test",
                                        "knee_test", "hip_test")]
        wb.append(chain)
    if physics:
        _add_actuators(root, names)
    return _write(root, dst)


def _writer(out):
    import imageio.v2 as imageio
    return imageio.get_writer(out, fps=FPS, codec="libx264", quality=8,
                              macro_block_size=1)


def _qadr(m, name):
    return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]


def _substep(m):
    return max(1, int(round(1.0 / (FPS * m.opt.timestep))))


def _act_id(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def render(path, out, side, physics=False):
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    joints = ("waist_test", "ankle_test", "knee_test", "hip_test")
    adr = {j: _qadr(m, j) for j in joints}
    act = {j: _act_id(m, j) for j in joints}
    r = mujoco.Renderer(m, 720, 540)
    cam = mujoco.MjvCamera()
    lookat, az = CAM[side]
    cam.lookat[:] = lookat
    cam.distance = 0.62
    cam.azimuth = az
    cam.elevation = -4

    sign = SIDES[side]["sign"]
    w_amp = np.radians(WAIST_SWING_DEG) * sign
    amp = np.radians(SWING_DEG) * sign
    n = int(FPS * SECONDS)
    sub = _substep(m)
    with _writer(out) as w:
        for i in range(n):
            t = (i / FPS) / SECONDS
            target = {
                "waist_test": w_amp * np.sin(2 * np.pi * t),
                "ankle_test": amp * np.sin(2 * np.pi * t - np.pi / 2),
                "knee_test": amp * np.sin(2 * np.pi * t - np.pi),
                "hip_test": amp * np.sin(2 * np.pi * t - 3 * np.pi / 2),
            }
            if physics:
                for j, v in target.items():
                    d.ctrl[act[j]] = v
                for _ in range(sub):
                    mujoco.mj_step(m, d)
            else:
                for j, v in target.items():
                    d.qpos[adr[j]] = v
                mujoco.mj_forward(m, d)
            r.update_scene(d, cam)
            w.append_data(r.render())
    r.close()
    print(f"wrote {out}  ({n} frames, {SECONDS:.0f}s @ {FPS}fps"
          f"{', physics' if physics else ''})")


def render_both(path, out, physics=False):
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    joints = ("waist_test", "ankle_test", "knee_test", "hip_test")
    adr, act = {}, {}
    for side in SIDES:
        p = side[0].upper() + "_"
        for j in joints:
            adr[side, j] = _qadr(m, p + j)
            act[side, j] = _act_id(m, p + j)
    r = mujoco.Renderer(m, 900, 620)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, -0.010, -0.150]
    cam.distance = 0.82
    cam.azimuth = 270
    cam.elevation = -6

    n = int(FPS * SECONDS)
    sub = _substep(m)
    # Mirror map: to make the right leg a mirror image of the left, the right
    # waist/ankle hinge signs are negated while knee/hip keep the left sign
    # (verified: right foot == Mx(left foot) for all joint values).
    RM = {"waist_test": -1.0, "ankle_test": -1.0,
          "knee_test": 1.0, "hip_test": 1.0}
    with _writer(out) as w:
        for i in range(n):
            t = (i / FPS) / SECONDS
            base = {
                # waist moves outward only (never adducts past centre), so the
                # two legs always stay apart and can never cross.
                "waist_test": np.radians(WAIST_SWING_DEG) * (0.5 + 0.5 * np.sin(2 * np.pi * t)),
                "ankle_test": np.radians(SWING_DEG) * np.sin(2 * np.pi * t - np.pi / 2),
                "knee_test": np.radians(SWING_DEG) * np.sin(2 * np.pi * t - np.pi),
                "hip_test": np.radians(SWING_DEG) * np.sin(2 * np.pi * t - 3 * np.pi / 2),
            }
            for j, v in base.items():
                if physics:
                    d.ctrl[act["left", j]] = v
                    d.ctrl[act["right", j]] = RM[j] * v
                else:
                    d.qpos[adr["left", j]] = v
                    d.qpos[adr["right", j]] = RM[j] * v
            if physics:
                for _ in range(sub):
                    mujoco.mj_step(m, d)
            else:
                mujoco.mj_forward(m, d)
            r.update_scene(d, cam)
            w.append_data(r.render())
    r.close()
    print(f"wrote {out}  ({n} frames, {SECONDS:.0f}s @ {FPS}fps"
          f"{', physics' if physics else ''})")


def main():
    args = sys.argv[1:]
    physics = "physics" in args or "--physics" in args
    which = next((a for a in args if a not in ("physics", "--physics")), "both")
    tag = "_physics" if physics else ""
    if which == "both":
        render_both(build_both(f"/tmp/opencode/_chestmove_both{tag}.xml", physics),
                    os.path.join(HERE, f"chest_both_legs{tag}.mp4"), physics)
    elif which == "all":
        for side in SIDES:
            render(build(f"/tmp/opencode/_chestmove_{side}{tag}.xml", side, physics),
                   OUT[side][:-4] + f"{tag}.mp4", side, physics)
        render_both(build_both(f"/tmp/opencode/_chestmove_both{tag}.xml", physics),
                    os.path.join(HERE, f"chest_both_legs{tag}.mp4"), physics)
    elif which in SIDES:
        render(build(f"/tmp/opencode/_chestmove_{which}{tag}.xml", which, physics),
               OUT[which][:-4] + f"{tag}.mp4", which, physics)
    else:
        raise SystemExit(f"usage: {sys.argv[0]} [both|all|left|right] [physics]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
