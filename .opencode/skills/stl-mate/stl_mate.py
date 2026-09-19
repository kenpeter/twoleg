#!/usr/bin/env python3
"""stl-mate — deterministic STL/MJCF geometry helpers (see SKILL.md).

Run from the repo root, e.g.:
    uv run --with numpy --with mujoco python .opencode/skills/stl-mate/stl_mate.py bbox robot_item/舵机/舵机.STL
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np


# ---------------------------------------------------------------- STL I/O
def read_stl(path):
    """Return triangles as an (N,3,3) float array (mm). Binary or ASCII."""
    data = open(path, "rb").read()
    if len(data) < 84:
        raise ValueError(f"STL too short: {path}")
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + 50 * n == len(data) and n > 0:
        arr = np.frombuffer(data[84:84 + 50 * n], dtype=np.uint8)
        arr = arr.reshape(n, 50)
        f = arr[:, :48].copy().view("<f4").reshape(n, 12)
        return f[:, 3:].reshape(n, 3, 3).astype(float)
    # ASCII
    verts = []
    for line in data.decode("utf-8", "replace").splitlines():
        s = line.split()
        if len(s) == 4 and s[0].lower() == "vertex":
            verts.append([float(s[1]), float(s[2]), float(s[3])])
    return np.asarray(verts, float).reshape(-1, 3, 3)


def write_binary_stl(path, tris):
    tris = np.asarray(tris, np.float32)
    with open(path, "wb") as fh:
        fh.write(b"stl-mate binary STL".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tris)))
        for t in tris:
            fh.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for v in t:
                fh.write(struct.pack("<3f", *v))
            fh.write(struct.pack("<H", 0))


# ---------------------------------------------------------------- helpers
def _point_in_tri(px, py, a, b, c):
    d1 = (px - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (py - b[1])
    d2 = (px - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (py - c[1])
    d3 = (px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def _holes_in_grid(filled, res, ax_names):
    """Flood-fill from the border; enclosed free regions are holes."""
    h, w = filled.shape
    free = ~filled
    out = np.zeros_like(free, dtype=np.int32)
    border = set(out[0, :]) | set(out[-1, :]) | set(out[:, 0]) | set(out[:, -1])
    from collections import deque
    comp = 0
    for sy in range(h):
        for sx in range(w):
            if free[sy, sx] and out[sy, sx] == 0:
                comp += 1
                q = deque([(sy, sx)])
                out[sy, sx] = comp
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and free[ny, nx] and out[ny, nx] == 0:
                            out[ny, nx] = comp
                            q.append((ny, nx))
    border_comps = set(out[0, :]) | set(out[-1, :]) | set(out[:, 0]) | set(out[:, -1])
    res_holes = []
    for i in range(1, comp + 1):
        if i in border_comps:
            continue
        ys, xs = np.where(out == i)
        if len(ys) < 8:
            continue
        res_holes.append((len(ys) * res * res, xs.mean() * res, ys.mean() * res, len(ys)))
    res_holes.sort(reverse=True)
    return res_holes, ax_names


# ---------------------------------------------------------------- commands
def cmd_bbox(args):
    v = read_stl(args.file).reshape(-1, 3)
    mn, mx = v.min(0), v.max(0)
    print(f"file   {args.file}")
    print(f"min    {mn[0]:.3f} {mn[1]:.3f} {mn[2]:.3f}")
    print(f"max    {mx[0]:.3f} {mx[1]:.3f} {mx[2]:.3f}")
    print(f"size   {mx[0]-mn[0]:.3f} {mx[1]-mn[1]:.3f} {mx[2]-mn[2]:.3f}")
    print(f"center {((mn[0]+mx[0])/2):.3f} {((mn[1]+mx[1])/2):.3f} {((mn[2]+mx[2])/2):.3f}")


def cmd_axes(args):
    v = read_stl(args.file).reshape(-1, 3)
    for ax, name in enumerate("XYZ"):
        hist, edges = np.histogram(v[:, ax], bins=12)
        print(f"{name}: [{v[:, ax].min():.2f},{v[:, ax].max():.2f}] span {v[:, ax].max()-v[:, ax].min():.2f}")
        for c, e0, e1 in zip(hist, edges[:-1], edges[1:]):
            print(f"   [{e0:7.2f},{e1:7.2f}) {c}")


def cmd_holes(args):
    tris = read_stl(args.file)
    ax = "xyz".index(args.axis.lower())
    keep = tris[(tris[:, :, ax].min(1) >= args.min - 1e-4) & (tris[:, :, ax].max(1) <= args.max + 1e-4)]
    proj = [i for i in range(3) if i != ax]
    names = "".join("xyz"[i] for i in proj).upper()
    if len(keep) == 0:
        print(f"no triangles in slab {args.axis}[{args.min},{args.max}]")
        return
    p = keep[:, :, proj]
    lo = p.reshape(-1, 2).min(0)
    hi = p.reshape(-1, 2).max(0)
    res = args.res
    w = int(np.ceil((hi[0] - lo[0]) / res)) + 3
    h = int(np.ceil((hi[1] - lo[1]) / res)) + 3
    filled = np.zeros((h, w), bool)
    for t in p:
        pts = (t - lo) / res + 1.0
        x0, x1 = int(pts[:, 0].min()), int(np.ceil(pts[:, 0].max()))
        y0, y1 = int(pts[:, 1].min()), int(np.ceil(pts[:, 1].max()))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        if x1 < x0 or y1 < y0:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        a, b, c = pts[0], pts[1], pts[2]
        for py, px in zip(yy.ravel(), xx.ravel()):
            if _point_in_tri(px, py, a, b, c):
                filled[py, px] = True
    holes, _ = _holes_in_grid(filled, res, names)
    print(f"slab {args.axis.upper()} in [{args.min},{args.max}]  plane={names}  tris={len(keep)}  res={res}")
    print(f"holes (area mm^2, {names[0]} mm, {names[1]} mm, px):")
    for area, u, vv, px in holes[:12]:
        print(f"   {area:7.2f}   {u:7.2f}   {vv:7.2f}   {px}")


def cmd_mirror(args):
    tris = read_stl(args.file)
    ax = "xyz".index(args.axis.lower())
    tris = tris.copy()
    tris[:, :, ax] *= -1.0
    tris = tris[:, ::-1, :]  # reverse winding so normals stay outward
    write_binary_stl(args.out, tris)
    print(f"wrote {args.out} (mirrored {args.axis.upper()})")


def cmd_world(args):
    import mujoco
    m = mujoco.MjModel.from_xml_path(args.mjcf)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for i in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"BODY {n:18s} mass={m.body_mass[i]:.6f} ipos={np.round(m.body_ipos[i],5)}")
    for i in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        print(f"GEOM {n:18s} xpos={np.round(d.geom_xpos[i],5)}")


def cmd_render(args):
    import mujoco
    from PIL import Image
    m = mujoco.MjModel.from_xml_path(args.mjcf)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    m.vis.global_.offwidth = 1280
    m.vis.global_.offheight = 960
    r = mujoco.Renderer(m, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = args.lookat
    cam.distance = args.dist
    cam.azimuth = args.az
    cam.elevation = args.el
    r.update_scene(d, cam)
    Image.fromarray(r.render()).save(args.out)
    print(f"wrote {args.out}")


def cmd_previews(args):
    """Render each matched MJCF to a PNG alongside it (auto-framed).

    Each render runs in its own subprocess: reusing one process across models
    lets the GL context die after the first render (black frames).
    """
    import glob
    import subprocess
    import mujoco

    files = sorted(glob.glob(args.glob))
    if not files:
        print("no files match", args.glob)
        return
    ok = 0
    for f in files:
        out = os.path.splitext(f)[0] + ".png"
        try:
            m = mujoco.MjModel.from_xml_path(f)
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            c = np.array([d.geom_xpos[i] for i in range(m.ngeom)])
            r = np.array([m.geom_rbound[i] for i in range(m.ngeom)])
            lo = (c - r[:, None]).min(0)
            hi = (c + r[:, None]).max(0)
            lookat = (lo + hi) / 2
            dist = max(2.0 * float((hi - lo).max()), 0.06)
        except Exception as e:
            print("SKIP", f, "->", e)
            continue
        cmd = [sys.executable, os.path.abspath(__file__), "render", f, out,
               "--az", str(args.az), "--el", str(args.el), "--dist", str(dist),
               "--lookat", *[f"{v:.6f}" for v in lookat],
               "--width", str(args.width), "--height", str(args.height)]
        rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if rc == 0 and os.path.exists(out):
            ok += 1
            print("wrote", out)
        else:
            print("FAIL", f)
    print(f"{ok}/{len(files)} previews written")


def cmd_check(args):
    """Physics/assembly sanity check on a compiled MJCF (deterministic verdict)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(args.mjcf)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    ok = True
    print(f"CHECK {args.mjcf}")
    print(f"compile: OK nbody={m.nbody} ngeom={m.ngeom} njnt={m.njnt} nq={m.nq} nu={m.nu}")

    collision_bodies = set()
    for i in range(m.ngeom):
        if m.geom_contype[i] or m.geom_conaffinity[i]:
            collision_bodies.add(int(m.geom_bodyid[i]))

    for i in range(1, m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        mass = float(m.body_mass[i])
        flag = ""
        if mass <= 0 and i in collision_bodies:
            ok = False
            flag = "  <-- FAIL collision body mass<=0"
        print(f"body {n:18s} mass={mass:.6f} ipos={np.round(m.body_ipos[i], 5)}{flag}")

    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        ax = m.jnt_axis[j]
        rng = np.round(m.jnt_range[j], 4).tolist() if m.jnt_limited[j] else "unlimited"
        print(f"joint {str(n):16s} type={int(m.jnt_type[j])} axis={np.round(ax,3)} range={rng}")
        if float(np.linalg.norm(ax)) < 1e-6:
            ok = False
            print("   <-- FAIL zero joint axis")

    deepest = 0.0
    pairs = {}
    for i in range(d.ncon):
        c = d.contact[i]
        deepest = min(deepest, float(c.dist))
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1])
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2])
        pairs[(b1, b2)] = pairs.get((b1, b2), 0) + 1
    pen_mm = -deepest * 1000.0
    print(f"contacts: ncon={d.ncon} deepest_penetration={pen_mm:.3f} mm")
    for (b1, b2), k in sorted(pairs.items(), key=lambda x: -x[1])[:12]:
        print(f"   {b1} <-> {b2}: {k}")
    if pen_mm > args.max_pen:
        ok = False
        print(f"   <-- FAIL penetration {pen_mm:.2f}mm > {args.max_pen}mm")

    print(f"COM (world): {np.round(d.subtree_com[0], 5)}")
    print("VERDICT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(prog="stl_mate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bbox"); b.add_argument("file"); b.set_defaults(func=cmd_bbox)
    a = sub.add_parser("axes"); a.add_argument("file"); a.set_defaults(func=cmd_axes)

    h = sub.add_parser("holes")
    h.add_argument("file")
    h.add_argument("--axis", required=True, choices=list("xyzXYZ"))
    h.add_argument("--min", type=float, required=True)
    h.add_argument("--max", type=float, required=True)
    h.add_argument("--res", type=float, default=0.25)
    h.set_defaults(func=cmd_holes)

    mi = sub.add_parser("mirror")
    mi.add_argument("file"); mi.add_argument("out")
    mi.add_argument("--axis", default="y", choices=list("xyzXYZ"))
    mi.set_defaults(func=cmd_mirror)

    w = sub.add_parser("world"); w.add_argument("mjcf"); w.set_defaults(func=cmd_world)

    r = sub.add_parser("render")
    r.add_argument("mjcf"); r.add_argument("out")
    r.add_argument("--az", type=float, default=45)
    r.add_argument("--el", type=float, default=-20)
    r.add_argument("--dist", type=float, default=0.25)
    r.add_argument("--lookat", type=float, nargs=3, default=[0, 0, 0])
    r.add_argument("--width", type=int, default=900)
    r.add_argument("--height", type=int, default=700)
    r.set_defaults(func=cmd_render)

    ck = sub.add_parser("check", help="physics/assembly sanity check on a compiled MJCF")
    ck.add_argument("mjcf")
    ck.add_argument("--max-pen", type=float, default=3.0, help="max allowed penetration in mm")
    ck.set_defaults(func=cmd_check)

    pv = sub.add_parser("previews", help="render each matched MJCF to a PNG alongside it")
    pv.add_argument("glob")
    pv.add_argument("--az", type=float, default=45)
    pv.add_argument("--el", type=float, default=-20)
    pv.add_argument("--width", type=int, default=700)
    pv.add_argument("--height", type=int, default=700)
    pv.set_defaults(func=cmd_previews)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
