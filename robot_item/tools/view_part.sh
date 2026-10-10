#!/usr/bin/env bash
# View any MuJoCo XML in the repo in the live viewer.
#
# Single model root: robot_item/xml/. The RL training model and the viewer
# both read robot_item/xml/robot_twoleg.xml from here.
#
#   ./robot_item/tools/view_part.sh                      # main model (robot_twoleg.xml)
#   ./robot_item/tools/view_part.sh --list               # every model, with nq/nu/nkey/mass
#   ./robot_item/tools/view_part.sh robot_twoleg         # by name fragment
#   ./robot_item/tools/view_part.sh robot_item/left_arm # by path fragment
#   ./robot_item/tools/view_part.sh 舵机                  # part folder -> that part's model.xml
#   ./robot_item/tools/view_part.sh 3                    # by number from --list
#   ./robot_item/tools/view_part.sh robot_twoleg STAND   # ... posed at a keyframe (name or index)
#
# Windowed only. EGL and OSMesa both fail on this NVIDIA setup, so rendering
# a PNG without a display is not supported here.
set -euo pipefail
cd "$(dirname "$0")/../.."

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

MODEL_DIR = "robot_item/xml"          # the one model root
PART_ROOT = "robot_item"             # CAD part folders, each with its own model.xml
MODEL_DIRS = (MODEL_DIR,)
DEFAULT_XML = f"{MODEL_DIR}/robot_twoleg.xml"

# The single source of truth declares no actuators (nu=0) because mjlab injects
# its own at env build. The viewer has no such step, so add them here; without
# this every hinge is locked and the model cannot be posed.
POSE_TARGETS = (
    "L_hip_roll_test", "L_hip_test", "L_knee_test", "L_ankle_test",
    "R_hip_roll_test", "R_hip_test", "R_knee_test", "R_ankle_test",
)


def load_with_actuators(path):
    spec = mujoco.MjSpec.from_file(path)
    if not spec.actuators and spec.joints:
        names = {j.name for j in spec.joints}
        import numpy as np

        for t in POSE_TARGETS:
            if t not in names:
                continue
            a = spec.add_actuator()
            a.name = t
            a.target = t
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.gainprm = np.zeros(10)
            a.gainprm[0] = 40.0
            a.biasprm = np.zeros(10)
            a.biasprm[1] = -40.0
            a.biasprm[2] = -0.5
            a.ctrllimited = True
            a.ctrlrange = [-1.4, 1.4]
            a.forcerange = [-20.0, 20.0]
    return spec.compile()


def models():
    return sorted(f for d in MODEL_DIRS for f in glob.glob(f"{d}/*.xml"))


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
    print(f"\n  {len(found)} models in {' and '.join(MODEL_DIRS)}/")
    print("  " + "-" * 96)
    for i, f in enumerate(found, 1):
        print(f"  {i:2}) {f:<52} {probe(f)}")
    print("  " + "-" * 96)
    print("  open one : ./robot_item/tools/view_part.sh <number|fragment> [keyframe]\n")


def resolve(sel, found):
    if os.path.isfile(sel):
        return sel
    if sel.isdigit() and 1 <= int(sel) <= len(found):
        return found[int(sel) - 1]
    part = f"{PART_ROOT}/{sel}/model.xml"
    if os.path.isfile(part):
        return part
    hits = [f for f in found if sel in f]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        sys.exit(f"error: no model matching '{sel}'")
    sys.exit(
        f"error: '{sel}' is ambiguous:\n  "
        + "\n  ".join(hits)
        + "\n  narrow it with a path fragment, e.g. 'twoleg/robot_twoleg' or 'robot_item/robot_twoleg'"
    )


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
    sys.exit(f"error: no *.xml found in {' or '.join(MODEL_DIRS)}")

if argv and argv[0] in ("--list", "-l"):
    show_list(found)
    sys.exit(0)

positional = [a for a in argv if not a.startswith("-")]
if len(positional) > 2:
    sys.exit("error: too many arguments (expected [model] [keyframe])")
unknown = [a for a in argv if a.startswith("-") and a not in ("--list", "-l")]
if unknown:
    sys.exit(f"error: unknown option {unknown[0]}")

path = resolve(positional[0], found) if positional else DEFAULT_XML

m = load_with_actuators(path)
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