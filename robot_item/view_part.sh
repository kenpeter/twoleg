#!/usr/bin/env bash
# View any MuJoCo XML in robot_item/ in the live viewer.
#
#   ./robot_item/view_part.sh                      # main model (robot_twoleg.xml)
#   ./robot_item/view_part.sh --list               # every model, with nq/nu/nkey/mass
#   ./robot_item/view_part.sh robot_twoleg         # by name fragment
#   ./robot_item/view_part.sh 舵机                  # part folder -> that part's model.xml
#   ./robot_item/view_part.sh 3                    # by number from --list
#   ./robot_item/view_part.sh robot_twoleg STAND   # ... posed at a keyframe (name or index)
#
# Windowed only. EGL and OSMesa both fail on this NVIDIA setup, so rendering
# a PNG without a display is not supported here.
set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL=glfw
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1   # keep prompts visible when stdout is a pipe

# launch_passive needs mjpython on macOS; plain python elsewhere.
PYBIN="python"
[[ "$(uname -s)" == "Darwin" ]] && PYBIN="mjpython"

uv run --offline --with mujoco "$PYBIN" - "$@" <<'PY' 2> >(grep -v -E "libdecor|libEGL|pci id|dri2 screen|No plugins found|no decorations|^$" >&2)
import glob
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
try:
    import glfw

    glfw.ERROR_REPORTING = "ignore"  # silence the expected Wayland window-position notice
except Exception:
    pass

import mujoco
import mujoco.viewer

MODEL_DIR = "robot_item"
DEFAULT_XML = f"{MODEL_DIR}/robot_twoleg.xml"


def models():
    return sorted(glob.glob(f"{MODEL_DIR}/*.xml"))


def probe(path):
    try:
        m = mujoco.MjModel.from_xml_path(path)
        keys = ",".join(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) or "" for k in range(m.nkey)
        )
        return (
            f"nq={m.nq:<3} nu={m.nu:<3} nkey={m.nkey:<3} "
            f"mass={m.body_mass.sum():6.2f}kg  {keys or '-'}"
        )
    except Exception as e:
        return "LOAD FAILS: " + str(e).splitlines()[-1][:46]


def show_list(found):
    print(f"\n  {len(found)} models in {MODEL_DIR}/")
    print("  " + "-" * 84)
    for i, f in enumerate(found, 1):
        print(f"  {i:2}) {os.path.basename(f):<34} {probe(f)}")
    print("  " + "-" * 84)
    print("  open one : ./robot_item/view_part.sh <number|fragment> [keyframe]\n")


def resolve(sel, found):
    if os.path.isfile(sel):
        return sel
    if sel.isdigit() and 1 <= int(sel) <= len(found):
        return found[int(sel) - 1]
    part = f"{MODEL_DIR}/{sel}/model.xml"
    if os.path.isfile(part):
        return part
    hits = [f for f in found if sel in os.path.basename(f)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        sys.exit(f"error: no model matching '{sel}'")
    sys.exit(f"error: '{sel}' is ambiguous:\n  " + "\n  ".join(hits))


def keyframe_index(m, key):
    if m.nkey == 0:
        print(f"warning: model has no keyframes; showing the default pose")
        return None
    idx = int(key) if key.lstrip("-").isdigit() else mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_KEY, key
    )
    if idx < 0:
        names = ", ".join(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) or "" for k in range(m.nkey)
        )
        sys.exit(f"error: no keyframe '{key}'. available: {names}")
    return idx


argv = sys.argv[1:]
found = models()
if not found:
    sys.exit(f"error: no *.xml found in {MODEL_DIR}/")

if not argv or argv[0] in ("--list", "-l"):
    show_list(found)
    sys.exit(0)

positional = [a for a in argv if not a.startswith("-")]
if len(positional) > 2:
    sys.exit("error: too many arguments (expected [model] [keyframe])")
unknown = [a for a in argv if a.startswith("-") and a not in ("--list", "-l")]
if unknown:
    sys.exit(f"error: unknown option {unknown[0]}")

path = resolve(positional[0], found) if positional else DEFAULT_XML

m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)
if len(positional) > 1:
    idx = keyframe_index(m, positional[1])
    if idx is not None:
        mujoco.mj_resetDataKeyframe(m, d, idx)
        print(f"keyframe '{positional[1]}' loaded")
mujoco.mj_forward(m, d)

print(
    f"viewing {path}  nbody={m.nbody} ngeom={m.ngeom} "
    f"nq={m.nq} nu={m.nu} nkey={m.nkey}"
)
print("close the window or press Ctrl+C to exit")
with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.lookat[:] = m.stat.center
    v.cam.distance = 2.6 * m.stat.extent
    v.cam.azimuth = 120
    v.cam.elevation = -20
    v.sync()
    while v.is_running():
        v.sync()
        time.sleep(0.05)
print("viewer closed.")
PY