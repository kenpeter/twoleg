#!/usr/bin/env bash
# View a single robot_item part (or all of them) in the MuJoCo viewer.
#
#   ./robot_item/view_part.sh               # all parts laid out
#   ./robot_item/view_part.sh servo         # by slug
#   ./robot_item/view_part.sh 舵机           # by folder name
#   ./robot_item/view_part.sh --list        # list available parts
#
# Uses glfw (Wayland) like twoleg_mjcf/view.sh; EGL/headless crashes on this
# NVIDIA setup, so keep the windowed path.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--list" || "${1:-}" == "-l" ]]; then
  echo "all"
  ls robot_item/*/model.xml | sed 's#robot_item/##; s#/model.xml##'
  ls robot_item/*_assembly.xml 2>/dev/null | sed 's#robot_item/##; s#\.xml##'
  exit 0
fi

NAME="${1:-all}"
case "$NAME" in
  all) XML="robot_item/all_parts.xml" ;;
  *.xml) XML="$NAME" ;;
  *) XML="robot_item/$NAME/model.xml" ;;
esac
if [[ ! -f "$XML" && -f "robot_item/$NAME.xml" ]]; then XML="robot_item/$NAME.xml"; fi
if [[ ! -f "$XML" ]]; then
  echo "No model at '$XML'. Available:" >&2
  "$0" --list >&2
  exit 1
fi

export MUJOCO_GL=glfw
export PYTHONWARNINGS=ignore
echo "Opening $XML  (close window or Ctrl+C to exit)"
uv run --offline --with mujoco python - "$XML" <<'PY' 2> >(grep -v -E "libdecor|libEGL|pci id|dri2 screen|No plugins found|no decorations|^$" >&2)
import sys, time, warnings
warnings.filterwarnings("ignore")
try:
    import glfw; glfw.ERROR_REPORTING = "ignore"
except Exception:
    pass
import mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path(sys.argv[1])
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.lookat[:] = m.stat.center
    v.cam.distance = 2.6 * m.stat.extent
    v.cam.azimuth = 120
    v.cam.elevation = -20
    v.sync()
    while v.is_running():
        v.sync()
        time.sleep(0.05)
PY
