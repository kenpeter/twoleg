#!/usr/bin/env python3
"""Score candidate placements of left_arm.xml onto chest_assembly_step5.xml.

Constraint set (all measured, see bores3.py output):
  * slantU ear bores: 2 x D8 on the world line X=-0.09805 Z=0.08905, axis world Y,
    ear mid-planes at Y=-0.02758 (near) and Y=+0.02441 (far).
  * each ear carries 4 x D3 on a 14 mm diamond (slantU mesh frame) about the D8.
  * arm shoulder: horn_2 4 x D3 on a 14 mm diamond at arm (12.75,-8.5,10) mm,
    disc axis = arm Y; servo_1 body spans arm Y -4.5..38.5 mm.

Candidates: the horn seats on the outer face of the near or far ear (2 choices of
arm-Y direction) x 4 rolls that keep the horn diamond on the ear diamond.  Each
is scored on bolt RMS, prong/servo interference and gap to every chest part.
"""
import itertools
import os
import sys

import numpy as np
import mujoco
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
CHEST = os.path.join(HERE, "chest_assembly_step5.xml")
ARM = os.path.join(HERE, "left_arm.xml")

# world, metres
EAR_BORE = np.array([-0.09805, 0.08905])          # X, Z of the D8 line
EAR_MID = {"near": -0.02758, "far": 0.02441}       # ear plate mid-plane, world Y
EAR_HALF = 0.0005
# 4 x D3 on the near ear, world metres (bores3 on slantU_L_c)
EAR_M3 = np.array([[-0.09311, 0.09399], [-0.10301, 0.09398],
                   [-0.09310, 0.08409], [-0.10300, 0.08408]])
# arm frame, metres
HORN_BORES = np.array([[0.00575, -0.0085, 0.010], [0.01275, -0.0085, 0.003],
                       [0.01975, -0.0085, 0.010], [0.01275, -0.0085, 0.017]])
HORN_FACES_Y = (-0.0095, -0.0045)                  # arm-frame Y of the disc faces
SERVO_Y = (-0.0045, 0.0385)                        # arm-frame Y of the servo body


def quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q


def build_candidates():
    """(label, R, t) with world = R @ arm + t."""
    d1 = np.array([0.70771, 0.0, -0.70651])   # slantU mesh X in world
    d2 = np.array([-0.70651, 0.0, -0.70771])  # slantU mesh Y in world
    yaxis = np.array([0.0, 1.0, 0.0])
    out = []
    for xi, dx in enumerate((d1, d2)):
        for ys in (1.0, -1.0):
            X = dx * (1.0 if xi == 0 else 1.0)
            Y = yaxis * ys
            Z = np.cross(X, Y)
            R = np.column_stack([X, Y, Z])
            for side in ("near", "far"):
                # put the horn's bore mid-plane on the ear's mid-plane
                ymid = EAR_MID[side]
                bore_arm_y = HORN_BORES[0, 1]
                if ys > 0:
                    t_y = ymid - bore_arm_y
                else:
                    t_y = ymid + bore_arm_y
                # X,Z fixed by the ear bore line
                t = np.array([EAR_BORE[0], t_y, EAR_BORE[1]]) - R @ np.array(
                    [HORN_BORES[0, 0], 0.0, HORN_BORES[0, 2]])
                out.append((f"roll{xi}_y{'p' if ys > 0 else 'm'}_{side}", R, t))
    return out


def arm_points(R, t, pts):
    return (R @ np.asarray(pts).T).T + t


def main():
    mc = mujoco.MjModel.from_xml_path(CHEST)
    dc = mujoco.MjData(mc)
    mujoco.mj_forward(mc, dc)
    ma = mujoco.MjModel.from_xml_path(ARM)
    da = mujoco.MjData(ma)
    mujoco.mj_forward(ma, da)

    def wverts(m, d, g):
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]

    chest_parts = {}
    for g in range(mc.ngeom):
        if mc.geom_group[g] != 3:
            continue
        chest_parts[mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_GEOM, g)] = wverts(mc, dc, g)
    arm_parts = {}
    for g in range(ma.ngeom):
        if ma.geom_group[g] != 3:
            continue
        arm_parts[mujoco.mj_id2name(ma, mujoco.mjtObj.mjOBJ_GEOM, g)] = wverts(ma, da, g)
    arm_local = dict(arm_parts)   # already in the arm root frame

    print(f"{'candidate':22s} {'boltRMS':>8s} {'servoY':>17s} {'minGap':>8s} "
          f"{'armZdir':>22s} {'arm bbox mm':>26s}")
    rows = []
    for label, R, t in build_candidates():
        hb = arm_points(R, t, HORN_BORES)
        # bolt RMS: each horn bore to the nearest ear M3 in the ear plane
        plane = np.array([EAR_MID["near"]]) if "near" in label else np.array([EAR_MID["far"]])
        e = EAR_M3.copy()
        d = np.linalg.norm(hb[:, None][:, :, [0, 2]] - e[None, :, :], axis=2)
        rms = float(np.sqrt((d.min(axis=1) ** 2).mean())) * 1000
        sy = arm_points(R, t, [[0, SERVO_Y[0], 0], [0, SERVO_Y[1], 0]])[:, 1]
        lo, hi = min(plane[0], min(EAR_MID.values())), max(plane[0], max(EAR_MID.values()))
        # gap: every arm collision vertex against every chest part
        worst, worst_name = 1e9, ""
        for an, av in arm_local.items():
            aw = av @ R.T + t
            for cn, cv in chest_parts.items():
                if cn.startswith("slantU"):
                    pass
                g = float(cKDTree(cv).query(aw)[0].min()) * 1000
                if g < worst:
                    worst, worst_name = g, f"{an}/{cn}"
        allv = np.vstack([v @ R.T + t for v in arm_local.values()])
        blo, bhi = allv.min(0) * 1000, allv.max(0) * 1000
        zdir = R @ np.array([0.0, 0.0, -1.0])
        rows.append((label, rms, (sy.min() * 1000, sy.max() * 1000), worst,
                     worst_name, zdir, blo, bhi, R, t))
        print(f"{label:22s} {rms:8.2f} [{sy.min()*1000:7.2f},{sy.max()*1000:7.2f}] "
              f"{worst:8.2f} {str(np.round(zdir,3)):>22s} "
              f"[{np.round(blo,0).tolist()} .. {np.round(bhi,0).tolist()}]  {worst_name}")
    print("\nprong gap = [%.2f, %.2f] mm" % (min(EAR_MID.values()) * 1000 - 0.5,
                                             max(EAR_MID.values()) * 1000 + 0.5))
    for label, rms, sy, worst, wn, zdir, blo, bhi, R, t in rows:
        print(f"\n## {label}  quat={np.round(quat(R),6).tolist()} pos={np.round(t,6).tolist()}")


if __name__ == "__main__":
    sys.exit(main())
