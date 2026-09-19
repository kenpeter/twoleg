---
name: stl-mate
description: Deterministic STL/MJCF geometry helpers for mating parts — bounding boxes, hole detection on a planar wall, principal axes, sagittal mirroring, world placement of raw mesh points, and offscreen verification renders. Use when assembling MJCF models from STL parts in robot_item/ (foot, ankle, thigh, xiaou, servo, horn), aligning holes/bores, mirroring left/right, or checking where a mesh point lands in world after a body pos/quat. Triggers include "align", "mate", "hole", "bore", "mirror", "left/right", "bbox", "shaft axis", "place part".
---

# stl-mate — STL/MJCF mating helpers

All geometry is in the STL's native units (mm). MJCF uses metres. The script is
dependency-light: `numpy` for parsing/rastering and `mujoco` only for `world`/`render`.

Run everything through uv from the repo root:

```
uv run --with numpy --with mujoco python .opencode/skills/stl-mate/stl_mate.py <cmd> ...
```

## Commands

| Command | Use for |
|---|---|
| `bbox FILE` | min/max/size/centre (mm) — feed these into `geom pos` recentering |
| `holes FILE --axis X --min A --max B [--res 0.2]` | find holes through a slab (e.g. a bracket wall); prints area + centre in the two projected axes |
| `axes FILE` | vertex distribution per axis — find a part's long/thin axes |
| `mirror FILE OUT --axis y` | write a properly-wound mirrored STL (negate axis + reverse triangle order) |
| `world MJCF` | print every body/geom world position of a compiled MJCF (verify placement) |
| `render MJCF OUT.png [--az --el --dist --lookat x y z]` | offscreen PNG for visual verification |
| `previews GLOB` | render each matched MJCF to a `<name>.png` **alongside** it (auto-framed) so every `*.xml` has a viewable image |

**Keep a preview next to every assembly.** After creating/editing any MJCF, run:

```
uv run --with numpy --with mujoco --with pillow python .opencode/skills/stl-mate/stl_mate.py previews 'robot_item/*.xml'
uv run --with numpy --with mujoco --with pillow python .opencode/skills/stl-mate/stl_mate.py previews 'robot_item/*/model.xml'
```

`previews` renders each file in its own subprocess — reusing one process lets the GL
context die after the first frame (black images). If a PNG comes out black, that is why.

## Core recipes

**Recenter a part so the body origin is its bbox centre** (matches `build_parts.py`):
`center = bbox mid` → `geom pos = -center/1000`.

**Mirror left → right (sagittal, Y→−Y).** Two valid options:
1. Negative-Y mesh scale in the XML (`scale="0.001 -0.001 0.001"`), plus negate all
   body/geom Y positions and conjugate rotations `(w,x,y,z)→(w,-x,y,z)`.
2. Pre-mirror the STL (`mirror FILE OUT --axis y`), keep positive scale, negate Y
   positions and conjugate quats. Verify masses match and `body_ipos` Y flips
   (`world MJCF`) — that is the mirror check.

**Mate a hole to a shaft.** Use `holes` on the wall slab to get the bore centre
`(u,v)` in the wall's plane, and `world MJCF` to get the shaft axis point. Shift
the body in the plane so `u,v` match the shaft's `(y,z)` (or `(x,z)` etc.). Then
`render` straight down the shaft axis to confirm the bore is concentric.

**Pick the shaft axis of a servo.** `axes` shows the long/thin dimensions; the
output shaft is on the axis with the high-Z round boss (see `robot_item/舵机`).
The raw servo shaft is `+Z`, offset toward the `Y=0` end.

## Rules

- Never eyeball a placement when a hole/axis can be measured — measure, place,
  then `render` to confirm.
- Mirroring is only exact with a mirrored mesh; a placement-only mirror leaves
  chiral parts (servo offset, bracket tabs) un-flipped.
- Keep rendered proof in `/tmp`, not the repo.
