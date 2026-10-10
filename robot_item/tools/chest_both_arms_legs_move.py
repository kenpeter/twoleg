#!/usr/bin/env python3
"""Render chest_both_arms_legs(_physics).mp4 — the whole humanoid moving.

Source: ``chest_both_arms_legs.xml`` = ``chest_assembly_step6`` (chest + both
arms) with both legs hung on the waist pivots.  It is a pose file, so it is
re-rigged into the servo chains the hardware actually has, taking each half from
the script that already measures it:

    upper body   chest_step6_move.rig(): abduct_L/R (chest servo horns, world X),
                 waist_L/R (waist servo horns, world Y), head (world Z), and the
                 shoulder/elbow/wrist hinge of both arms, every pivot on a
                 measured 金属舵盘 disc centre.
    legs         chest_move.make_chain(): re-groups the model's own leg subtrees
                 into waist -> top -> hip -> knee -> ankle -> foot with the
                 pivots on the measured prong-bore/shaft lines.  It re-groups
                 rather than re-mounts, so the zero-angle pose is exact for
                 whatever mirror the model used (chest_move.MIRRORS).

The waist horns carry the legs, so they are driven from the leg waist joints
instead of chest_step6's own waist wave, and the right leg runs the calibrated
mirror map: measured per joint, right leg == Mx(left leg) to 0.000 mm.

Run:
    uv run --with numpy --with mujoco --with imageio --with imageio-ffmpeg \
        python robot_item/tools/chest_both_arms_legs_move.py            # physics
    python robot_item/tools/chest_both_arms_legs_move.py kinematic
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chest_move as cm                      # noqa: E402
import chest_step6_move as csm               # noqa: E402
from leg_move_all import SIDES               # noqa: E402

SRC = os.path.join(HERE, "chest_both_arms_legs.xml")

FPS = 30
SECONDS = 8.0
SETTLE_SECONDS = 1.0     # physics only: reach the t=0 pose before frame 0
LEG_KP = 40.0            # the legs carry every gram of leg under gravity
TARGET_FILL = 0.80       # humanoid fraction of the short frame side
LEG_JOINTS = ("waist_test", "ankle_test", "knee_test", "hip_test")
# Mirror map for the right leg, calibrated joint by joint (chest_move.RM).
RM = cm.RM
ACTUATORS = list(csm.ACTUATORS) + [f"{s}_{j}" for s in ("L", "R")
                                   for j in LEG_JOINTS]


def build(dst, physics):
    root = cm._setup(ET.parse(SRC), physics)
    wb = root.find("worldbody")
    # Detach the legs first, so the arm/chest rig cannot reach into them.
    legs = {side: cm._leg(wb, side) for side in SIDES}
    csm.rig(root)
    bits = {"left": (1, 2), "right": (2, 1)}   # legs collide, arms are welded
    names = []
    for side in SIDES:
        leg, prefix, leg_pos, leg_quat = legs[side]
        sfx = side[0].upper() + "_"
        chain = cm.make_chain(side, leg_pos, leg_quat, sfx, leg, prefix)
        if physics:
            cm._leg_collision(chain, *bits[side])
            names += [sfx + j for j in LEG_JOINTS]
        wb.append(chain)
    if physics:
        act = ET.SubElement(root, "actuator")
        for n in csm.ACTUATORS:
            ET.SubElement(act, "position", {"name": n, "joint": n, "kp": "6",
                                            "dampratio": "1",
                                            "ctrlrange": "-1.4 1.4"})
        for n in names:
            ET.SubElement(act, "position", {"name": n, "joint": n,
                                            "kp": f"{LEG_KP:g}", "dampratio": "1",
                                            "ctrlrange": "-1.4 1.4"})
    ET.ElementTree(root).write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def _targets(t):
    """Servo commands: chest_step6's upper body plus a marching pair of legs.

    The waist horns and the leg tops are bolted together, so the horn commands
    are simply the leg waist commands (the two hinges share the world Y axis,
    0.6 mm apart on the shaft line).
    """
    out = csm._targets(t)
    base = {
        # waist swings outward only (never adducts past centre), so the legs stay
        # apart and can never cross
        "waist_test": np.radians(cm.WAIST_SWING_DEG)
        * (0.5 + 0.5 * np.sin(2 * np.pi * t)),
        "ankle_test": np.radians(cm.SWING_DEG) * np.sin(2 * np.pi * t - np.pi / 2),
        "knee_test": np.radians(cm.SWING_DEG) * np.sin(2 * np.pi * t - np.pi),
        "hip_test": np.radians(cm.SWING_DEG) * np.sin(2 * np.pi * t - 3 * np.pi / 2),
    }
    for j, v in base.items():
        out[f"L_{j}"] = v
        out[f"R_{j}"] = RM[j] * v
    out["waist_L"] = out["L_waist_test"]
    out["waist_R"] = out["R_waist_test"]
    return out


def _camera(m, d):
    """Frame the shot from the rendered silhouette, not from stat.center.

    Two things make the model's own bounding box useless here: m.stat.center
    sits well below the robot's visual middle (aiming at it crops the head),
    and the marching pose is wider and shorter than the rest pose.  So: set a
    few choreography poses, render each, take the union of the ink, aim the
    camera so that union is centred, and scale the distance until it fills
    TARGET_FILL of the frame; cut the frame to the union's aspect and refit.
    Override the shot with AZIMUTH=... ELEVATION=... python ...
    """
    az = float(os.environ.get("AZIMUTH", "130"))
    el = float(os.environ.get("ELEVATION", "-10"))
    fovy = m.vis.global_.fovy
    qadr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ACTUATORS}
    q0 = d.qpos.copy()
    poses = [0.0, 0.25, 0.5, 0.75]     # choreography phases, one full cycle

    def cam(distance):
        c = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(c)
        c.lookat[:] = m.stat.center
        c.distance = distance
        c.azimuth = az
        c.elevation = el
        return c

    def ink(r, c, w, h):
        x0, y0, x1, y1 = w, h, 0, 0
        for t in poses:
            for j, v in _targets(t).items():
                d.qpos[qadr[j]] = v
            mujoco.mj_forward(m, d)
            r.update_scene(d, c)
            ys, xs = np.nonzero(r.render().max(axis=-1) > 10)
            if len(xs):
                x0, x1 = min(x0, xs.min()), max(x1, xs.max())
                y0, y1 = min(y0, ys.min()), max(y1, ys.max())
        if x1 <= x0:
            return None
        return x0, x1, y0, y1

    def fit(r, c, w, h, rounds):
        box = None
        for _ in range(rounds):
            box = ink(r, c, w, h)
            if box is None:
                return None
            x0, x1, y0, y1 = box
            gl = r.scene.camera[0]
            right = np.cross(gl.forward, gl.up)
            n = np.linalg.norm(right)
            if n > 1e-9:
                right /= n
            unit = 2.0 * c.distance * np.tan(np.radians(fovy) / 2) / h
            c.lookat += right * ((x0 + x1) / 2 - w / 2) * unit
            c.lookat -= gl.up * ((y0 + y1) / 2 - h / 2) * unit
            fill = max((x1 - x0) / w, (y1 - y0) / h)
            if fill > 1e-6:
                c.distance *= fill / TARGET_FILL
        return box

    # probe on a square to learn the union silhouette's aspect
    r = mujoco.Renderer(m, 480, 480)
    try:
        c = cam(2.0 * m.stat.extent)
        box = fit(r, c, 480, 480, 8)
    finally:
        r.close()
        d.qpos[:] = q0
        mujoco.mj_forward(m, d)
    aspect = ((box[1] - box[0]) / max(1, box[3] - box[2])) if box else 1.0
    long_side = 880
    if aspect >= 1.0:
        w, h = long_side, int(2 * round(long_side / aspect / 2))
    else:
        h, w = long_side, int(2 * round(long_side * aspect / 2))
    w, h = max(w, 360), max(h, 360)

    # refit at the final aspect (Renderer takes height first)
    r = mujoco.Renderer(m, h, w)
    try:
        fit(r, c, w, h, 6)
    finally:
        r.close()
        d.qpos[:] = q0
        mujoco.mj_forward(m, d)
    return c, (w, h)


def render(path, out, physics):
    import imageio.v2 as imageio
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    qadr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ACTUATORS}
    act = ({j: m.actuator(j).id for j in ACTUATORS} if physics else {})

    cam, (w, h) = _camera(m, d)
    r = mujoco.Renderer(m, h, w)
    n = int(FPS * SECONDS)
    sub = cm._substep(m) if physics else 1
    err = {j: 0.0 for j in ACTUATORS}
    if physics:
        for j, v in _targets(0.0).items():
            d.ctrl[act[j]] = v
        for _ in range(int(SETTLE_SECONDS / m.opt.timestep)):
            mujoco.mj_step(m, d)
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=1) as w_:
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
            w_.append_data(r.render())
    r.close()
    print(f"wrote {out}  ({n} frames, {SECONDS:.0f}s @ {FPS}fps"
          f"{', physics' if physics else ''}, {w}x{h})")
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
    render(build(f"/tmp/opencode/_humanoid{tag}.xml", physics),
           os.path.join(HERE, f"chest_both_arms_legs{tag}.mp4"), physics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
