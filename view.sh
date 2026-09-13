#!/usr/bin/env bash
# Live 3D viewer for the twoleg model (paused at STAND keyframe)
# NOTE: stays on Wayland on purpose — forcing X11 crashes with
# the current NVIDIA 595.84 kernel vs 595.91 userspace mismatch.
# The libdecor / libEGL / dri2 warnings below are harmless and filtered.
cd "$(dirname "$0")"
export MUJOCO_GL=glfw
export PYTHONWARNINGS=ignore
export MJ_KEY="${1:-0}"  # keyframe index: 0=STAND, 1=ARMS_STRAIGHT (or a keyframe name)
echo "Opening MuJoCo viewer (Wayland, keyframe '${MJ_KEY}')... close the window or press Ctrl+C to exit."
uv run --with mujoco python << 'PY' 2> >(grep -v -E "libdecor|libEGL|pci id|dri2 screen|No plugins found|no decorations|^$" >&2)
import os
import warnings
warnings.filterwarnings("ignore")
try:
    import glfw
    glfw.ERROR_REPORTING = "ignore"  # silence "Wayland: ... window position" (expected, harmless)
except Exception:
    pass
import mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path('twoleg_mjcf/robot_twoleg.xml')
d = mujoco.MjData(m)
key = os.environ.get('MJ_KEY', '0')
try:
    idx = int(key)
except ValueError:
    idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key)
mujoco.mj_resetDataKeyframe(m, d, idx)
mujoco.mj_forward(m, d)
print(f"Viewer running (keyframe {key}).", flush=True)
with mujoco.viewer.launch_passive(m, d) as v:
    import time
    while v.is_running():
        v.sync()
        time.sleep(0.05)
print("Viewer closed.", flush=True)
PY
