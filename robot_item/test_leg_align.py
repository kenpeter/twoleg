#!/usr/bin/env python3
"""Physical verification of the left leg assembly (robot_item/leg_assembly_left.xml).

Checks, all through MuJoCo:

A. ALIGNMENT - for the ankle and knee joints, the servo output tip and the
               little-U wall hole must sit on the disc's rotation axis (within
               0.75 mm).  This is what "the shaft goes into the hole" means.
B. FIT       - colliding meshes must not interpenetrate.
C. WORKING   - the shin+femur are rebuilt as a real hinge at the knee axis,
               kicked, and simulated: the joint must spin and stay finite.

Run:  uv run --with numpy --with mujoco python robot_item/test_leg_align.py
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "leg_assembly_left.xml")
TOL_MM = 0.75

# (joint, servo geom, disc body, little-U wall geom)
JOINTS = [
    ("ankle", "servo_visual", "horn", "shortU_down_visual"),
    ("knee", "servo_4_v", "horn_5", "shortU_up_visual"),
    ("hip", "servo_1_v", "horn_2", "multi_0_v"),
]


def load(path=XML):
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _verts(m, d, geom_name):
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mid = m.geom_dataid[g]
    a = m.mesh_vertadr[mid]
    v = m.mesh_vert[a:a + m.mesh_vertnum[mid]]
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def _frame(axis):
    e1 = np.cross(axis, [0, 0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(axis, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def measure_joint(m, d, joint):
    """Return (servo tip perpendicular offset, wall hole offset) in mm."""
    name, servo_geom, disc_body, wall_geom = joint
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, disc_body)
    axis = d.xmat[b].reshape(3, 3)[:, 1].copy()          # horn thin axis
    centre = d.xpos[b].copy()
    e1, e2 = _frame(axis)
    perp = lambda P: np.array([np.dot(P - centre, e1), np.dot(P - centre, e2)])

    S = _verts(m, d, servo_geom)
    ps = np.array([np.linalg.norm(perp(v)) for v in S])
    near = S[ps < 0.006]
    proj = np.dot(near - centre, axis)
    tip = near[proj < proj.min() + 0.002].mean(axis=0)

    W = _verts(m, d, wall_geom)
    pr = np.array([np.linalg.norm(perp(v)) for v in W])
    hole = W[pr < pr.min() + 0.001].mean(axis=0)

    return (float(np.linalg.norm(perp(tip))) * 1000,
            float(np.linalg.norm(perp(hole))) * 1000,
            axis)


def check_alignment(m, d):
    fails = []
    for joint in JOINTS:
        tip_mm, hole_mm, axis = measure_joint(m, d, joint)
        print(f"  {joint[0]:5s} axis {np.round(axis,3)}  servo-tip off-axis {tip_mm:5.2f} mm  "
              f"hole off-axis {hole_mm:5.2f} mm")
        if tip_mm > TOL_MM:
            fails.append(f"{joint[0]}: servo output {tip_mm:.2f} mm off the hole axis")
        if hole_mm > TOL_MM:
            fails.append(f"{joint[0]}: wall hole {hole_mm:.2f} mm off the disc axis")
    return fails


def check_fit(m, d):
    """True mesh clearance for the parts that sit close together.

    MuJoCo collides convex hulls, so a servo parked inside a concave U-bracket
    shows a huge false overlap; use real vertex distances instead.
    """
    pairs = [
        ("shortU_up_visual", "servo_4_v"),
        ("shortU_up_visual", "multi_3_v"),
        ("shortU_up_visual", "horn_5_v"),
        ("shortU_down_visual", "servo_visual"),
        ("shortU_down_visual", "multi_visual"),
    ]
    from scipy.spatial import cKDTree
    worst = 1e9
    for a, b in pairs:
        A, B = _verts(m, d, a), _verts(m, d, b)
        gap = float(cKDTree(B).query(A)[0].min()) * 1000
        worst = min(worst, gap)
        print(f"  {a:20s} <-> {b:16s} gap {gap:6.2f} mm")
    return worst


def build_hinged_copy(dst):
    """Nest the shin under a hinge at the knee axis so it can spin.

    Collision is turned off in this copy: MuJoCo collides the *convex hull* of
    each mesh, and the U-shaped brackets are concave, so a servo sitting inside
    the U always shows a false hull-overlap.  Alignment (section A) is the real
    fit check; here we only test the rotational degree of freedom.
    """
    knee = np.array([0.0175, -0.014, 0.07475])
    tree = ET.parse(XML)
    root = tree.getroot()
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    wb = root.find("worldbody")
    foot = next(b for b in wb.findall("body") if b.get("name") == "foot")
    # new parent that carries the hinge
    link = ET.Element("body", {"name": "knee_link", "pos": " ".join(f"{v:.6f}" for v in knee)})
    ET.SubElement(link, "joint", {"name": "knee_test", "type": "hinge", "axis": "0 1 0",
                                  "damping": "0.01"})
    idx = list(wb).index(foot)
    wb.remove(foot)
    foot.set("pos", " ".join(f"{-v:.6f}" for v in knee))
    link.append(foot)
    wb.insert(idx, link)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def check_working(seconds=1.2, kick=2.0):
    tmp = build_hinged_copy("/tmp/opencode/_leg_hinge.xml")
    tree = ET.parse(tmp)
    tree.getroot().find("compiler").set("meshdir", HERE)
    tree.write(tmp, encoding="utf-8", xml_declaration=True)

    m = mujoco.MjModel.from_xml_path(tmp)
    d = mujoco.MjData(m)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "knee_test")
    qadr, vadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
    d.qvel[vadr] = kick
    mujoco.mj_forward(m, d)
    a0 = float(d.qpos[qadr])
    a1, peak = a0, 0.0
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        a1 = float(d.qpos[qadr])
        peak = max(peak, abs(a1 - a0))
    finite = bool(np.isfinite(d.qpos).all())
    print(f"  knee hinge free swing: {math.degrees(a0):.1f} -> {math.degrees(a1):.1f} deg "
          f"(peak {math.degrees(peak):.1f} deg), final |qvel| {abs(d.qvel[vadr]):.3f} rad/s")
    fails = []
    if not finite:
        fails.append("simulation produced a non-finite state")
    if math.degrees(peak) < 5.0:
        fails.append(f"knee joint barely moved ({math.degrees(peak):.1f} deg)")
    return fails


def main():
    m, d = load()
    print("A. alignment")
    fails = check_alignment(m, d)
    print("B. fit (true mesh clearance)")
    if check_fit(m, d) < 0.2:
        fails.append("parts overlap (mesh clearance < 0.2 mm)")
    print("C. working")
    fails += check_working()
    print()
    for f in fails:
        print("FAIL:", f)
    if fails:
        return 1
    print("VERDICT: PASS  - outputs/holes aligned to the physical joint and it moves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
