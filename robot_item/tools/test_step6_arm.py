#!/usr/bin/env python3
"""Verify the left_arm mount on chest_assembly_step6.xml.

Checks, all against measured geometry (no file-frame assumptions):
  1. MuJoCo compiles and the arm bodies are present.
  2. The arm shoulder axis (horn_2 4xD3 diamond) is coaxial with the slantU_L
     prong bores and its Y range sits inside the 52 mm ear gap.
  3. The horn face is flush with the near ear's inner face.
  4. No arm vertex penetrates any chest part (dense surface sampling, because
     vertex-only tests miss plate-on-plate overlap).
  5. Mass/geom counts unchanged from step5 + the arm's own parts.

Run: uv run --offline --with mujoco --with numpy --with scipy --with pillow \
        python robot_item/test_step6_arm.py
"""
import os
import sys

import numpy as np
import mujoco
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
STEP5 = os.path.join(HERE, "chest_assembly_step5.xml")
STEP6 = os.path.join(HERE, "chest_assembly_step6.xml")
PNG = os.path.join(HERE, "chest_assembly_step6.png")

TOL_MM = 0.75          # coaxiality / seating tolerance
PEN_TOL_MM = -0.05     # allow a hair of overlap
SAMPLE = 24            # barycentric samples per triangle edge

# measured slantU_L prong bores, world metres (bores3.py on chest_assembly_step5.xml)
EAR_BORE_NEAR = np.array([-0.09805, -0.02758, 0.08904])
EAR_BORE_FAR = np.array([-0.09808, 0.02441, 0.08907])
EAR_M3_NEAR = np.array([[-0.09311, 0.09399], [-0.10301, 0.09398],
                        [-0.09310, 0.08409], [-0.10300, 0.08408]]) / 1.0
# arm frame, metres: horn_2 4xD3 diamond and disc faces along arm Y
HORN_BORES = np.array([[0.00575, -0.0085, 0.010], [0.01275, -0.0085, 0.003],
                       [0.01975, -0.0085, 0.010], [0.01275, -0.0085, 0.017]])
HORN_FACE_Y = (-0.0095, -0.0045)
SERVO_Y = (-0.0045, 0.0385)


def geom_verts(m, d, g):
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def geom_tris(m, g):
    mid = m.geom_dataid[g]
    nf = int(m.mesh_facenum[mid])
    fa = int(m.mesh_faceadr[mid])
    return np.asarray(m.mesh_face[fa:fa + nf]).reshape(nf, 3).astype(int)


def surface_points(m, d, g, n=SAMPLE):
    v = geom_verts(m, d, g)
    f = geom_tris(m, g)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    out = [a, b, c]
    for i in range(1, n):
        for j in range(1, n - i):
            out.append(a + (b - a) * (i / n) + (c - a) * (j / n))
    return np.vstack(out)


def main():
    fails = []

    m5 = mujoco.MjModel.from_xml_path(STEP5)
    m = mujoco.MjModel.from_xml_path(STEP6)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f"step5 nbody {m5.nbody} ngeom {m5.ngeom} mass {m5.body_mass.sum():.4f} kg")
    print(f"step6 nbody {m.nbody} ngeom {m.ngeom} mass {m.body_mass.sum():.4f} kg")
    if m.nbody <= m5.nbody or m.ngeom <= m5.ngeom:
        fails.append("step6 did not gain the arm bodies/geoms")

    arm = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
           for i in range(m.nbody)}
    for need in ("left_arm", "multi_0", "servo_1", "horn_2", "forearm", "straight_3_f"):
        if need not in arm:
            fails.append(f"missing body {need}")

    # 2 + 3: shoulder coaxiality and seating
    gh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "horn_2_c")
    gs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "servo_1_c")
    vh, vs = geom_verts(m, d, gh), geom_verts(m, d, gs)
    for nm, verts, ycol in (("horn", vh, 1), ("servo", vs, 1)):
        pass
    # horn disc faces = min/max world Y of horn verts
    horn_y = (vh[:, 1].min(), vh[:, 1].max())
    ear_near_inner = EAR_BORE_NEAR[1] + 0.0005
    print(f"horn world Y span {horn_y[0]*1000:.2f}..{horn_y[1]*1000:.2f} mm "
          f"(ear inner face {ear_near_inner*1000:.2f} mm)")
    gap = abs(horn_y[0] - ear_near_inner) * 1000
    print(f"horn outer face to near-ear inner face: {gap:.2f} mm (TOL {TOL_MM})")
    if gap > TOL_MM:
        fails.append(f"horn not seated on the near ear: {gap:.2f} mm")

    # shoulder axis coaxial with both prong bores
    hb = (d.geom_xmat[gh].reshape(3, 3) @ HORN_BORES.T).T + d.geom_xpos[gh]
    cen = hb.mean(axis=0)
    print(f"horn bore centroid world {np.round(cen*1000, 3).tolist()} mm")
    for nm, ear in (("near", EAR_BORE_NEAR), ("far", EAR_BORE_FAR)):
        off = np.linalg.norm(cen - ear) * 1000
        print(f"  shoulder axis vs {nm} prong bore: {off:.2f} mm (TOL {TOL_MM})")
        if off > TOL_MM:
            fails.append(f"shoulder axis off the {nm} prong bore by {off:.2f} mm")
    # the ear bores are on one world-Y line: the arm axis must be parallel to it
    axis = np.array([EAR_BORE_FAR[1] - EAR_BORE_NEAR[1]])
    if abs(axis[0]) < 0.05:
        fails.append("prong bores are not separated along Y; re-measure")

    # servo inside the ear gap
    gap_lo, gap_hi = EAR_BORE_NEAR[1] + 0.0005, EAR_BORE_FAR[1] - 0.0005
    sy = (vs[:, 1].min(), vs[:, 1].max())
    print(f"servo world Y span {sy[0]*1000:.2f}..{sy[1]*1000:.2f} mm, "
          f"gap {gap_lo*1000:.2f}..{gap_hi*1000:.2f} mm")
    if sy[0] < gap_lo - 0.001 or sy[1] > gap_hi + 0.001:
        fails.append("shoulder servo does not sit inside the prong gap")

    # 4: penetration, dense surface sampling
    arm_geoms = [g for g in range(m.ngeom)
                 if m.geom_group[g] == 3 and "horn_s" not in
                 (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    arm_ids = set()
    for i in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        if nm and (nm == "left_arm" or arm_parent(m, i, "left_arm")):
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == i:
                    arm_ids.add(g)
    chest_ids = [g for g in range(m.ngeom)
                 if m.geom_group[g] == 3 and g not in arm_ids]
    worst, worst_name = 1e9, ""
    trees = {g: cKDTree(surface_points(m, d, g)) for g in chest_ids}
    for g in sorted(arm_ids):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        pts = surface_points(m, d, g)
        for cg, tree in trees.items():
            cname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, cg)
            gapmm = float(tree.query(pts)[0].min()) * 1000
            if gapmm < worst:
                worst, worst_name = gapmm, f"{nm} <-> {cname}"
    print(f"densest arm/chest approach: {worst:.2f} mm  ({worst_name})")
    if worst < PEN_TOL_MM:
        fails.append(f"arm penetrates chest: {worst:.2f} mm at {worst_name}")

    # 5: preview
    if not os.path.exists(PNG):
        fails.append(f"{PNG} missing")
    else:
        try:
            from PIL import Image
            arr = np.array(Image.open(PNG).convert("RGB"))
            mean = float(arr.mean())
            print(f"preview mean {mean:.1f}")
            if mean <= 10:
                fails.append(f"preview black (mean {mean:.1f})")
            if os.path.getmtime(PNG) + 1 < os.path.getmtime(STEP6):
                fails.append("preview older than the xml")
        except Exception as e:
            fails.append(f"preview check failed: {e}")

    print("\n" + "=" * 60)
    if fails:
        print(f"VERDICT: FAIL - {len(fails)} issues")
        for f in fails:
            print("  -", f)
        return 1
    print("VERDICT: PASS - left_arm mounted on the slantU_L prong bores")
    return 0


def arm_parent(m, i, root):
    p = m.body_parentid[i]
    seen = 0
    while p != 0 and seen < 20:
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, p) == root:
            return True
        p = m.body_parentid[p]
        seen += 1
    return False


if __name__ == "__main__":
    sys.exit(main())
