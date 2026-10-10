#!/usr/bin/env bash
# View the TwoLeg training model in the MuJoCo viewer.
#
#   ./view.sh                # the model in THIS directory (robot_twoleg.xml)
#   ./view.sh --list         # every model in robot_item/xml/
#   ./view.sh robot_twoleg   # any model by name, index or path fragment
#
# Canonical model lives at robot_item/xml/robot_twoleg.xml. The copy beside this
# script is a mirror for convenience; edit the canonical file, not this one.
# Real viewer: robot_item/tools/view_part.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../.." && pwd)"
VIEWER="$REPO/robot_item/tools/view_part.sh"

if [[ ! -x "$VIEWER" ]]; then
  echo "error: viewer not found at $VIEWER" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  exec "$VIEWER" "$HERE/robot_twoleg.xml"
fi
exec "$VIEWER" "$@"