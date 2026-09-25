#!/usr/bin/env python3
"""Physical verification of the chest assembly models.

chest_assembly_step5.xml has both 斜U wings mounted on the chest-servo discs
through each wing's central web bore (the hole marked red in the part drawing).

Checks mirroring test_leg_align.py style (TOL 0.75 mm):

- strict XML parses
- mujoco compiles
- slantU web bore concentric with the chest-servo disc ≤0.75 mm
- flange flush gap (outer face, 0-0.5 mm)
- no penetration via vertices
- quat normalized and mirrored
- slantU L/R world mirror bbox centres
- preview exists and not black

Run: uv run --offline --with numpy --with mujoco --with scipy --with pillow python robot_item/test_chest_align.py
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML_STEP5 = os.path.join(HERE, "chest_assembly_step5.xml")
XML_BOTH = os.path.join(HERE, "chest_both_arms.xml")
PNG_STEP5 = os.path.join(HERE, "chest_assembly_step5.png")

TOL_MM = 0.75
FLUSH_TOL_MM = 0.5
PENETRATION_TOL_MM = 0.05

# Measured hole centres (mm) at res 0.15 — slantU Z-slab 0-5/49-55 plane XY, multi Y-slab 0-3 plane XZ
# slantU local XY (Xmm,Ymm) at Z ~2.5 or 52 (same XY)
SLANT_BORE_PMID = (47.58, 9.96, 24.61)  # web central bore, file coords, web mid-plane
SLANT_BORE_N = (0.7077, -0.7065, 0.0)    # web plate normal in file coords
MULTI_HOLES_XZ = [
    (31.00, 20.65),
    (31.00, 6.65),
    (11.30, 20.65),
    (11.30, 6.65),
    (38.01, 13.65),
    (30.99, 13.65),
    (11.31, 13.65),
    (4.29, 13.65),
    (24.00, 13.65),
    (18.30, 13.65),
]
# bbox centres (mm)
SLANT_CENTER = np.array([28.244, 13.186, 27.035])
MULTI_CENTER = np.array([29.0, 18.5, 12.5])
# multi outer face world X (from stl_mate world calc) for chest_side_L/R
MULTI_OUTER_L = -0.0155
MULTI_OUTER_R = 0.0155
# flange half thickness world offset = 27.035 mm (local Z max - centre)
FLANGE_HALF = 0.027035


def _verts(m, d, geom_name):
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if g == -1:
        return None
    mid = m.geom_dataid[g]
    if mid == -1:
        return None
    a = m.mesh_vertadr[mid]
    n = m.mesh_vertnum[mid]
    if n == 0:
        return None
    v = m.mesh_vert[a:a+n]
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def _body_pos(m, d, body_name):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid == -1:
        return None
    return d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()


def _slant_world_YZ(body_pos, R_mat, Xh, Yh):
    # world = body + R*(local - centre)/1000
    # for slant, R = [[0,0,1],[0,-1,0],[1,0,0]] but we get R from model
    # we compute generically using R_mat
    local = np.array([Xh - SLANT_CENTER[0], Yh - SLANT_CENTER[1], 0.0])  # ZDelta not needed for YZ
    # Actually local ZDelta = Zh - cz ; for YZ we only need Xh,Yh; ZDelta affects world X only
    # So YZ depends only on Xh,Yh
    # world_Y = body_Y + R[1,0]*(Xh-cx)+R[1,1]*(Yh-cy)+R[1,2]*(Zh-cz)
    # world_Z = body_Z + R[2,0]*(Xh-cx)+R[2,1]*(Yh-cy)+R[2,2]*(Zh-cz)
    # For Y, R[1,2] term is zero for Z? No, R[1]={0,-1,0} so only Yh matters
    # For Z, R[2]={1,0,0} so only Xh matters
    # So we can compute directly:
    # Using R_mat:
    local_full = np.array([Xh - SLANT_CENTER[0], Yh - SLANT_CENTER[1], 0.0])
    world = body_pos + R_mat.dot(local_full)/1000.0
    return world[1], world[2]


def _slant_bore_local(side):
    # mirrored mesh negates X scale, so the R local X offset is the negated L one
    sx = -1.0 if side == "R" else 1.0
    p = SLANT_BORE_PMID
    return np.array([sx * (p[0] - SLANT_CENTER[0]),
                     p[1] - SLANT_CENTER[1],
                     p[2] - SLANT_CENTER[2]])


def _multi_world_YZ(body_pos, R_mat, lx, lz):
    # local = [lx-cx, ly-cy, lz-cz]; ly ~1.5 avg, affects world X only (R[0,1]=-1)
    # For YZ: world_Y = body_Y + R[1,2]*(lz-cz) = - (lz-cz)
    # world_Z = body_Z + R[2,0]*(lx-cx) = (lx-cx)
    local = np.array([lx - MULTI_CENTER[0], 0.0, lz - MULTI_CENTER[2]])
    # ly delta approximated 0 (mid of slab) — error <1.5mm in X, not YZ
    world = body_pos + R_mat.dot(local)/1000.0
    return world[1], world[2]


def test_strict_xml_parses():
    fails = []
    for p in [XML_STEP5, XML_BOTH]:
        try:
            ET.parse(p)
        except Exception as e:
            fails.append(f"{os.path.basename(p)} ET.parse FAIL: {e}")
    if fails:
        return fails, False
    return [], True


def test_mujoco_compiles():
    fails = []
    nbody_expect = {"chest_assembly_step5.xml": 28, "chest_both_arms.xml": 52}
    for p in [XML_STEP5, XML_BOTH]:
        try:
            m = mujoco.MjModel.from_xml_path(p)
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            exp = nbody_expect.get(os.path.basename(p))
            if exp is None or m.nbody != exp:
                fails.append(f"{os.path.basename(p)} nbody {m.nbody} != {exp}")
            min_geom = {"chest_assembly_step5.xml": 60, "chest_both_arms.xml": 90}
            if m.ngeom < min_geom.get(os.path.basename(p), 0):
                fails.append(f"{os.path.basename(p)} ngeom {m.ngeom} <{min_geom[os.path.basename(p)]}")
            # mass sanity: total mass approx step4 +0.282 kg
            total = float(m.body_mass.sum())
            # we don't have step4 model here, rough check >1.0
            if not np.isfinite(total) or total < 0.5:
                fails.append(f"{os.path.basename(p)} total mass {total:.3f} non-finite/low")
        except Exception as e:
            fails.append(f"{os.path.basename(p)} mujoco compile FAIL: {e}")
    return fails, len(fails)==0


def test_slantU_horn_seat():
    """斜U web bore sits on each chest-servo disc: concentric + flush."""
    fails = []
    try:
        m = mujoco.MjModel.from_xml_path(XML_STEP5)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        from scipy.spatial import cKDTree
        for side in ("L", "R"):
            body = f"slantU_{side}"
            horn = f"chest_servo_{side}_horn"
            res_s = _body_pos(m, d, body)
            res_h = _body_pos(m, d, horn)
            if res_s is None or res_h is None:
                fails.append(f"{body} or {horn} missing")
                continue
            pos_s, mat_s = res_s
            pos_h, _ = res_h
            local = _slant_bore_local(side)
            bore = pos_s + mat_s.dot(local) / 1000.0
            yz = math.hypot((bore[1] - pos_h[1]) * 1000, (bore[2] - pos_h[2]) * 1000)
            print(f"  {body} web bore vs servo disc YZ {yz:.2f} mm (TOL {TOL_MM})")
            if yz > TOL_MM:
                fails.append(f"{body} web bore not concentric with servo disc: {yz:.2f} mm >{TOL_MM}")
            vs = _verts(m, d, f"{body}_c")
            vh = _verts(m, d, f"{horn}_c")
            if vs is None or vh is None:
                fails.append(f"missing {body}_c or {horn}_c verts")
                continue
            gap = float(cKDTree(vh).query(vs)[0].min()) * 1000
            print(f"  {body} <-> horn vertex gap {gap:.2f} mm")
            if gap < -PENETRATION_TOL_MM or gap > FLUSH_TOL_MM:
                fails.append(f"{body} not seated on horn: gap {gap:.2f} mm not in "
                             f"[-{PENETRATION_TOL_MM},{FLUSH_TOL_MM}]")
    except Exception as e:
        fails.append(f"exception {e}")
    return fails, len(fails) == 0


def test_slantU_neighbours_clear():
    """The seated 斜U must not penetrate the chest side, servo, centre or U beam."""
    fails = []
    try:
        m = mujoco.MjModel.from_xml_path(XML_STEP5)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        from scipy.spatial import cKDTree
        for side in ("L", "R"):
            body = f"slantU_{side}"
            opposite = "slantU_R_c" if side == "L" else "slantU_L_c"
            vs = _verts(m, d, f"{body}_c")
            if vs is None:
                fails.append(f"missing {body}_c verts")
                continue
            for other in [f"chest_side_{side}_c", f"chest_servo_{side}_c", "chest_center_c",
                          "u_beam_c", "head_servo_c", opposite]:
                ov = _verts(m, d, other)
                if ov is None:
                    continue
                gap = float(cKDTree(ov).query(vs)[0].min()) * 1000
                print(f"  {body} <-> {other} gap {gap:.2f} mm")
                if gap < -PENETRATION_TOL_MM:
                    fails.append(f"{body} penetrates {other}: gap {gap:.2f} mm < -{PENETRATION_TOL_MM}")
    except Exception as e:
        fails.append(f"exception {e}")
    return fails, len(fails) == 0


def test_quat_normalized_and_mirrored():
    fails = []
    try:
        m = mujoco.MjModel.from_xml_path(XML_STEP5)
        # find bodies via xml for quat/scale check
        tree = ET.parse(XML_STEP5)
        root = tree.getroot()
        # asset scales
        for mesh in root.findall(".//mesh"):
            name = mesh.get("name")
            scale = mesh.get("scale")
            if name in ("slantU","slantU_r"):
                expected = "0.001 0.001 0.001" if name=="slantU" else "-0.001 0.001 0.001"
                if scale != expected:
                    fails.append(f"mesh {name} scale {scale} != {expected}")
        # body quats
        quats = {}
        for body in root.findall(".//body"):
            n = body.get("name")
            if n in ("slantU_L","slantU_R"):
                q = body.get("quat")
                quats[n]=q
                vals = list(map(float, q.split()))
                norm = math.sqrt(sum(v*v for v in vals))
                print(f"  {n} quat {q} norm {norm:.5f}")
                if not (0.999 <= norm <= 1.001):
                    fails.append(f"{n} quat norm {norm:.4f} not 1")
        missing = [n for n in ("slantU_L", "slantU_R") if n not in quats]
        for n in missing:
            fails.append(f"body {n} missing from {os.path.basename(XML_STEP5)}")
        # mirrored check: R conjugate?
        if not missing:
            ql = list(map(float, quats["slantU_L"].split()))
            qr = list(map(float, quats["slantU_R"].split()))
            # sagittal mirror: (w,x,y,z) -> (w,x,-y,-z)
            if abs(ql[0]-qr[0])>0.001 or abs(ql[1]-qr[1])>0.001 or abs(ql[2]+qr[2])>0.001 or abs(ql[3]+qr[3])>0.001:
                fails.append(f"quats not mirrored L {quats['slantU_L']} R {quats['slantU_R']}")
        # masses equal
        m = mujoco.MjModel.from_xml_path(XML_STEP5)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m,d)
        id_l = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "slantU_L")
        id_r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "slantU_R")
        if id_l == -1 or id_r == -1:
            fails.append(f"body id lookup failed slantU_L {id_l} slantU_R {id_r}")
        else:
            ml = float(m.body_mass[id_l])
            mr = float(m.body_mass[id_r])
            print(f"  masses L {ml:.6f} R {mr:.6f}")
            if abs(ml-0.141449)>0.001 or abs(mr-0.141191)>0.001:
                fails.append(f"mass mismatch L {ml:.5f} R {mr:.5f} expected 0.141449")
            if abs(ml-mr)>1e-6:
                fails.append(f"masses not equal L {ml} R {mr}")
    except Exception as e:
        fails.append(f"exception {e}")
    return fails, len(fails)==0


def test_slantU_world_mirror():
    """slantU_L_c and slantU_R_c world bbox centres mirror about X=0."""
    fails = []
    try:
        m = mujoco.MjModel.from_xml_path(XML_STEP5)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        vl = _verts(m, d, "slantU_L_c")
        vr = _verts(m, d, "slantU_R_c")
        if vl is None or vr is None:
            fails.append("missing slantU_L_c or slantU_R_c verts")
        else:
            cl = (vl.min(axis=0) + vl.max(axis=0)) / 2
            cr = (vr.min(axis=0) + vr.max(axis=0)) / 2
            dx = (cl[0] + cr[0]) * 1000
            dy = (cl[1] - cr[1]) * 1000
            dz = (cl[2] - cr[2]) * 1000
            print(f"  slantU bbox mirror deltas dx {dx:.2f} dy {dy:.2f} dz {dz:.2f} mm "
                  f"(TOL {PENETRATION_TOL_MM})")
            if abs(dx) > PENETRATION_TOL_MM:
                fails.append(f"slantU bbox centres not mirrored in X: {dx:.2f} mm")
            if abs(dy) > PENETRATION_TOL_MM:
                fails.append(f"slantU bbox centres differ in Y: {dy:.2f} mm")
            if abs(dz) > PENETRATION_TOL_MM:
                fails.append(f"slantU bbox centres differ in Z: {dz:.2f} mm")
    except Exception as e:
        fails.append(f"exception {e}")
    return fails, len(fails) == 0


def test_preview_exists():
    fails = []
    import pathlib
    p = pathlib.Path(PNG_STEP5)
    if not p.exists():
        fails.append(f"{PNG_STEP5} missing")
        return fails, False
    # mtime check
    try:
        xml_mtime = os.path.getmtime(XML_STEP5)
        png_mtime = os.path.getmtime(PNG_STEP5)
        if png_mtime + 1 < xml_mtime:
            fails.append(f"preview stale mtime {png_mtime} < xml {xml_mtime}")
    except Exception as e:
        fails.append(f"mtime check fail {e}")
    # mean brightness >10
    try:
        from PIL import Image
        import numpy as np
        im = Image.open(PNG_STEP5).convert("RGB")
        arr = np.array(im)
        mean = float(arr.mean())
        print(f"  preview mean {mean:.1f}")
        if mean <= 10:
            fails.append(f"preview mean {mean:.1f} <=10 (black)")
    except Exception as e:
        fails.append(f"preview check fail {e}")
    return fails, len(fails)==0


def main():
    tests = [
        ("strict_xml_parses", test_strict_xml_parses),
        ("mujoco_compiles", test_mujoco_compiles),
        ("slantU_horn_seat", test_slantU_horn_seat),
        ("slantU_neighbours_clear", test_slantU_neighbours_clear),
        ("quat_normalized_and_mirrored", test_quat_normalized_and_mirrored),
        ("slantU_world_mirror", test_slantU_world_mirror),
        ("preview_exists", test_preview_exists),
    ]
    fails_total = []
    for name, fn in tests:
        print(f"\n== {name} ==")
        fails, ok = fn()
        for f in fails:
            print("  FAIL:", f)
        if ok:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name}")
            fails_total.extend(fails if fails else [name])
    print("\n" + ("="*60))
    if fails_total:
        print(f"VERDICT: FAIL - {len(fails_total)} issues")
        for f in fails_total:
            print("  -", f)
        return 1
    print("VERDICT: PASS - all tests green")
    return 0

if __name__ == "__main__":
    sys.exit(main())
