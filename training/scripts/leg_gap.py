"""Measure the TwoLeg lateral stance from the compiled MJCF.

Trustworthy method: use d.geom_xpos / d.geom_xmat directly (MuJoCo has the
mesh alignment already baked into them). Do NOT re-apply mesh_quat/mesh_pos —
that double-transforms and was the source of a bogus 58 mm "overlap" earlier.

Usage: cd training && .venv/bin/python scripts/leg_gap.py [path/to.xml]
"""
from __future__ import annotations

import sys

import mujoco
import numpy as np

DEFAULT = "/home/kenpeter/work/twoleg/robot_item/xml/robot_twoleg.xml"


def geom_verts(m, d, g):
    mid = m.geom_dataid[g]
    if mid < 0:
        return None
    a, c = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    V = m.mesh_vert[a:a + c]
    return V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def subtree(m, root_name):
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_name)
    out = []
    for i in range(root, m.nbody):
        p = i
        while p != 0:
            if p == root:
                out.append(i)
                break
            p = m.body_parentid[p]
    return out


def report(path):
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    print(f"file {path}")
    print(f"njnt={m.njnt} nq={m.nq} nbody={m.nbody} ngeom={m.ngeom} mass={m.body_subtreemass[0]:.6f}kg colliding={len(coll)}")
    lx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "leg_left")
    rx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "leg_right")
    lat = int(np.argmax(np.abs(d.xpos[lx] - d.xpos[rx])))
    print(f"lateral axis = world {'xyz'[lat]}")
    out = {}
    for tag, root, foot in (("L", "leg_left", "L_foot"), ("R", "leg_right", "R_foot")):
        fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{foot}_collision")
        W = geom_verts(m, d, fid)
        flo, fhi = W.min(0), W.max(0)
        print(f"[{tag}] {foot} plate lateral {flo[lat]*1000:.2f}..{fhi[lat]*1000:.2f} "
              f"(w {(fhi[lat]-flo[lat])*1000:.2f}mm) bottom_z {flo[2]*1000:.2f}mm")
        out[f"{tag}_plate"] = (float(flo[lat]), float(fhi[lat]))
        ti = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{tag}_shortU_top")
        hi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{tag}_longU_top")
        for nm, bi in ((f"{tag}_shortU_top", ti), (f"{tag}_longU_top", hi)):
            if bi >= 0:
                print(f"     {nm} x={d.xpos[bi][0]*1000:.2f} z={d.xpos[bi][2]*1000:.2f} (leg-top attach)")
    L, R = out["L_plate"], out["R_plate"]
    inner_gap = R[0] - L[1]           # R inner(left) minus L inner(right): >0 => apart
    print(f"\nfoot inner-edge gap = {inner_gap*1000:.2f} mm   (>0 apart, <0 overlap)")
    print(f"support polygon width = {(R[1]-L[0])*1000:.2f} mm")
    out["inner_gap"] = float(inner_gap)
    out["support_w"] = float(R[1] - L[0])
    out["mass"] = float(m.body_subtreemass[0])
    return out


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
