"""Render top + front views of the TwoLeg XML so the stance can be eyeballed
against the real-robot photo. Colors: feet orange, legs gray, servos red.

Usage: cd training && .venv/bin/python scripts/render_gap.py [out_prefix]
"""
import math
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

XML = os.environ.get("XML", "/home/kenpeter/work/twoleg/robot_item/xml/robot_twoleg.xml")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/gap"
CAMS = ("top", "front", "iso")
src = open(XML).read().replace('meshdir=".."', 'meshdir="/home/kenpeter/work/twoleg/robot_item"')
src = src.replace("<worldbody>", "<worldbody>\n" + "".join(f'<camera name="{c}"/>\n' for c in CAMS))
m = mujoco.MjModel.from_xml_string(src)
d = mujoco.MjData(m)
m.vis.global_.offwidth = 900
m.vis.global_.offheight = 900
mujoco.mj_forward(m, d)

LROOT = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "leg_left")


def in_subtree(g):
    p = m.geom_bodyid[g]
    while p != 0:
        if p == LROOT or p == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "leg_right"):
            return True
        p = m.body_parentid[p]
    return False


for g in range(m.ngeom):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if not in_subtree(g):
        m.geom_rgba[g] = [0.8, 0.8, 0.85, 0.0]
    elif "_foot" in nm:
        m.geom_rgba[g] = [0.95, 0.45, 0.10, 1.0]
    elif "servo" in nm:
        m.geom_rgba[g] = [0.85, 0.1, 0.1, 1.0]
    else:
        m.geom_rgba[g] = [0.7, 0.7, 0.75, 1.0]

r = mujoco.Renderer(m, 900, 900)
views = {
    "top":   (np.array([0.0, 0.0, 0.75]), np.array([0.001, 0.0, 1.0])),
    "front": (np.array([0.0, -0.65, 0.0]), np.array([0.0, 1.0, 0.0])),
    "iso":   (np.array([0.5, -0.5, 0.35]), np.array([0.0, 0.0, 1.0])),
}
for name, (pos, fwd) in views.items():
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    if abs(fwd @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    tup = np.cross(fwd, right)
    R = np.eye(3); R[:, 0] = right; R[:, 1] = tup; R[:, 2] = fwd
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        cq = [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    else:
        cq = [1.0, 0.0, 0.0, 0.0]
    m.cam_pos[cid] = pos
    m.cam_quat[cid] = np.array(cq)
    r.update_scene(d, camera=name)
    imageio.imwrite(f"{OUT}_{name}.png", r.render())
    print("wrote", f"{OUT}_{name}.png")
