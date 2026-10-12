"""Widen the TwoLeg stance by hip ABDUCTION (rotate each leg about its top).

Each leg hangs rigidly from the waist servo horn. Sliding the feet would tear
the leg off the waist. The physical lever is abduction: rotate each leg about
a fore-aft (world-y) axis through its top attach point (L/R_shortU_top, at
|x|=37.20mm, z=-21.54mm), swinging the foot outward while the waist
attachment stays exactly where it is.

The rotation is applied to every body pose in the leg subtree, so the whole
leg stays rigid and correctly oriented; only its placement changes.

Self-verifying: after writing, it re-measures the foot inner-edge gap with
leg_gap.py and prints the delta so the result is not taken on trust.

Usage:
  cd training
  .venv/bin/python scripts/abduct_legs.py in.xml out.xml [--target-gap 0.020]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = __file__.rsplit("/", 1)[0]


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q = [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def fvec(s, default):
    return np.array([float(v) for v in s.split()]) if s else default


def collect(body, acc):
    acc.append(body)
    for ch in body:
        if ch.tag == "body":
            collect(ch, acc)


def rot_y(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def subtree_world_poses(el, parent_pos, parent_R, acc):
    Rl = quat_to_mat(fvec(el.get("quat"), np.array([1., 0, 0, 0])))
    pos = parent_pos + parent_R @ fvec(el.get("pos"), np.zeros(3))
    R = parent_R @ Rl
    acc[el] = (pos, R)
    for ch in el:
        if ch.tag == "body":
            subtree_world_poses(ch, pos, R, acc)


def rewrite_local(el, new_world_pos, new_world_R, parent_pos, parent_R):
    lp = parent_R.T @ (new_world_pos - parent_pos)
    lR = parent_R.T @ new_world_R
    el.set("pos", " ".join(f"{v:.6f}" for v in lp))
    el.set("quat", " ".join(f"{v:.6f}" for v in mat_to_quat(lR)))
    Rl = quat_to_mat(fvec(el.get("quat"), np.array([1., 0, 0, 0])))
    cw = new_world_R @ Rl.T  # not used; recompute children from stored new poses below


def solve_alpha(dx, dz, out):
    """Rotation rot_y(alpha) maps foot offset (dx,dz) so its x grows by -out
    (left: x more negative) -> we solve rot_y(alpha) @ (dx,dz) has x = dx - out.
    rot_y: x' = c*dx + s*dz ; z' = -s*dx + c*dz."""
    f = lambda a: (np.cos(a) * dx + np.sin(a) * dz) - (dx - out)
    lo, hi = -0.6, 0.6
    if f(lo) * f(hi) > 0:
        return np.sign(dx) * (out / abs(dz))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def abduct_leg(leg_el, pivot, alpha):
    Ry = rot_y(alpha)
    poses = {}
    subtree_world_poses(leg_el, np.zeros(3), np.eye(3), poses)
    new_world = {}
    for el, (pos, R) in poses.items():
        new_world[el] = (pivot + Ry @ (pos - pivot), Ry @ R)

    def rec(el, parent_pos, parent_R):
        npos, nR = new_world[el]
        lp = parent_R.T @ (npos - parent_pos)
        lR = parent_R.T @ nR
        el.set("pos", " ".join(f"{v:.6f}" for v in lp))
        el.set("quat", " ".join(f"{v:.6f}" for v in mat_to_quat(lR)))
        for ch in el:
            if ch.tag == "body":
                rec(ch, npos, nR)
    rec(leg_el, np.zeros(3), np.eye(3))


def measure(path):
    out = subprocess.run([sys.executable, f"{HERE}/leg_gap.py", path],
                         capture_output=True, text=True)
    gap = sup = None
    for line in out.stdout.splitlines():
        if line.startswith("foot inner-edge gap"):
            gap = float(line.split("=")[1].split("mm")[0].strip())
        if line.startswith("support polygon width"):
            sup = float(line.split("=")[1].split("mm")[0].strip())
    return gap, sup, out.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--target-gap", type=float, default=0.020, help="foot inner-edge gap target, m")
    ap.add_argument("--leg-top-x", type=float, default=0.03720)
    ap.add_argument("--leg-top-z", type=float, default=-0.02154)
    ap.add_argument("--foot-center-x", type=float, default=0.03420)
    ap.add_argument("--foot-z", type=float, default=-0.250)
    args = ap.parse_args()

    cur_gap, _, _ = measure(args.inp)
    if cur_gap is None:
        raise SystemExit("could not read baseline gap")
    out_per_foot = (args.target_gap - cur_gap * 0.001) / 2.0
    print(f"baseline gap {cur_gap:.2f}mm -> target {args.target_gap*1000:.1f}mm, "
          f"out/foot {out_per_foot*1000:.2f}mm")

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(args.inp, parser=parser)
    wb = tree.getroot().find("worldbody")
    legs = {el.get("name"): el for el in wb.iter("body")
            if el.get("name") in ("leg_left", "leg_right")}
    if len(legs) != 2:
        raise SystemExit(f"need leg_left+leg_right, got {list(legs)}")

    dx = args.foot_center_x - args.leg_top_x          # ~ -0.003 (foot slightly inboard of top)
    dz = args.foot_z - args.leg_top_z                 # ~ -0.228 (foot below top)
    alpha = solve_alpha(dx, dz, out_per_foot)         # left: +alpha (solves dx -> dx-out)
    print(f"|alpha| = {abs(np.degrees(alpha)):.4f} deg  (dx={dx*1000:.2f}mm dz={dz*1000:.1f}mm)")

    abduct_leg(legs["leg_left"], np.array([-args.leg_top_x, 0.0, args.leg_top_z]), alpha)
    abduct_leg(legs["leg_right"], np.array([args.leg_top_x, 0.0, args.leg_top_z]), -alpha)
    tree.write(args.out, encoding="unicode", xml_declaration=True)

    new_gap, new_sup, _ = measure(args.out)
    print(f"RESULT: gap {new_gap:.2f}mm (target {args.target_gap*1000:.1f}), "
          f"support width {new_sup:.2f}mm")


if __name__ == "__main__":
    main()
