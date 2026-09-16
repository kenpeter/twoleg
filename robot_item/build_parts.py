#!/usr/bin/env python3
"""Generate a standalone MuJoCo model.xml for every part in robot_item/.

Each part folder (e.g. robot_item/舵机/) gets a model.xml next to its .STL.
Meshes are scaled mm->m (0.001), recentered so the body origin sits at the
part's bounding-box centre, and given visual (group 2) + collision (group 3)
geoms.  A combined robot_item/all_parts.xml preview and a manifest json are
also written so the parts can be assembled later.

Run:  python3 robot_item/build_parts.py
"""
import json
import os
import struct

ROOT = os.path.dirname(os.path.abspath(__file__))

# folder -> (slug, material, density kg/m^3)
PARTS = {
    "L型":     ("L_bracket", "bracket", 2700),
    "U型梁":   ("U_beam",    "bracket", 2700),
    "一字":    ("straight",  "bracket", 2700),
    "多功能":  ("multi",     "bracket", 2700),
    "大脚板":  ("foot",      "bracket", 2700),
    "斜U":     ("slantU",    "bracket", 2700),
    "法兰轴承": ("bearing",   "steel",   7800),
    "短U":     ("shortU",    "bracket", 2700),
    "舵机":    ("servo",     "dark",    1600),
    "金属舵盘": ("horn",      "horn",    2700),
    "长U":     ("longU",     "bracket", 2700),
}

MATERIALS = {
    "bracket": "0.75 0.75 0.78 1",
    "dark":    "0.18 0.18 0.20 1",
    "steel":   "0.55 0.56 0.60 1",
    "horn":    "0.80 0.80 0.82 1",
}


def stl_bbox(path):
    """Return (min, max) lists in file units for binary or ascii STL."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 84:
        raise ValueError("STL too short: " + path)
    n = struct.unpack("<I", data[80:84])[0]
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    if 84 + 50 * n == len(data) and n > 0:
        for i in range(n):
            off = 84 + 50 * i
            v = struct.unpack("<12f", data[off:off + 48])
            for t in (v[3:6], v[6:9], v[9:12]):
                for k in range(3):
                    mn[k] = min(mn[k], t[k])
                    mx[k] = max(mx[k], t[k])
    else:  # ascii
        for line in data.decode("utf-8", "replace").splitlines():
            s = line.split()
            if len(s) == 4 and s[0] == "vertex":
                for k in range(3):
                    val = float(s[k + 1])
                    mn[k] = min(mn[k], val)
                    mx[k] = max(mx[k], val)
    if mn[0] == float("inf"):
        raise ValueError("no triangles in " + path)
    return mn, mx


def fmt(v):
    return " ".join(f"{x:.6f}" for x in v)


def find_stl(folder):
    hits = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(".stl")]
    if len(hits) != 1:
        raise ValueError(f"expected 1 STL in {folder}, found {hits}")
    return hits[0]


def write_part(folder, slug, material, density):
    stl = find_stl(folder)
    mn, mx = stl_bbox(os.path.join(folder, stl))
    center_m = [c * 0.001 for c in ((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2)]
    offset = [-c for c in center_m]
    size = [mx[k] - mn[k] for k in range(3)]
    xml = f"""<?xml version='1.0' encoding='utf-8'?>
<mujoco model="part_{slug}">
  <compiler angle="radian" meshdir="." autolimits="true"/>
  <visual>
    <global offwidth="1024" offheight="1024"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>
  </visual>
  <default>
    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/></default>
    <default class="collision"><geom type="mesh" group="3"/></default>
  </default>
  <asset>
    <mesh name="{slug}" file="{stl}" scale="0.001 0.001 0.001"/>
    <material name="{material}" rgba="{MATERIALS[material]}"/>
  </asset>
  <worldbody>
    <!-- source {stl}: bbox min {fmt(mn)} max {fmt(mx)} mm, size {fmt(size)} mm -->
    <body name="{slug}">
      <geom name="{slug}_visual" mesh="{slug}" class="visual" material="{material}" pos="{fmt(offset)}"/>
      <geom name="{slug}_collision" mesh="{slug}" class="collision" pos="{fmt(offset)}" density="{density}"/>
    </body>
  </worldbody>
</mujoco>
"""
    out = os.path.join(folder, "model.xml")
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    return {
        "slug": slug,
        "folder": os.path.relpath(folder, os.path.dirname(ROOT)),
        "stl": stl,
        "model": os.path.relpath(out, os.path.dirname(ROOT)),
        "bbox_mm": {"min": [round(v, 3) for v in mn], "max": [round(v, 3) for v in mx],
                    "size": [round(v, 3) for v in size]},
        "center_mm": [round(v, 3) for v in ((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2)],
        "material": material,
        "density": density,
    }


def write_preview(manifest, spacing=0.20):
    cols = 4
    repo = os.path.dirname(ROOT)
    meshes = "\n".join(
        f'    <mesh name="{p["slug"]}" file="{os.path.relpath(os.path.join(repo, p["folder"]), ROOT)}/{p["stl"]}" scale="0.001 0.001 0.001"/>'
        for p in manifest
    )
    mats = "\n".join(f'    <material name="{m}" rgba="{r}"/>' for m, r in MATERIALS.items())
    bodies = []
    for i, p in enumerate(manifest):
        r, c = divmod(i, cols)
        x = (c - (cols - 1) / 2) * spacing
        y = (r - (len(manifest) - 1) // cols) * spacing
        sz = max(p["bbox_mm"]["size"]) / 1000
        bodies.append(
            f'    <body name="{p["slug"]}" pos="{x:.4f} {y:.4f} 0">\n'
            f'      <geom name="{p["slug"]}_v" type="mesh" mesh="{p["slug"]}" class="visual" material="{p["material"]}"/>\n'
            f'      <geom name="{p["slug"]}_c" type="mesh" mesh="{p["slug"]}" class="collision" density="{p["density"]}"/>\n'
            f'    </body>'
        )
    xml = f"""<?xml version='1.0' encoding='utf-8'?>
<mujoco model="robot_item_parts">
  <compiler angle="radian" meshdir="." autolimits="true"/>
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>
  </visual>
  <default>
    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/></default>
    <default class="collision"><geom type="mesh" group="3" density="0"/></default>
  </default>
  <asset>
{meshes}
{mats}
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.5 0.5 0.5"/>
{bodies}
  </worldbody>
</mujoco>
"""
    out = os.path.join(ROOT, "all_parts.xml")
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    return out


def main():
    manifest = []
    for folder_name, (slug, material, density) in PARTS.items():
        folder = os.path.join(ROOT, folder_name)
        if not os.path.isdir(folder):
            raise SystemExit("missing folder: " + folder)
        manifest.append(write_part(folder, slug, material, density))
        print(f"wrote {folder_name}/model.xml  ({slug})")
    with open(os.path.join(ROOT, "parts_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    prev = write_preview(manifest)
    print("wrote", os.path.relpath(prev, os.path.dirname(ROOT)))
    print(f"{len(manifest)} parts generated")


if __name__ == "__main__":
    main()
