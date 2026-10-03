#!/usr/bin/env bash
# Play the newest training video from the newest TwoLeg run.
#
#   ./watch_video.sh              newest video
#   ./watch_video.sh 4000         the video for env step 4000
#   ./watch_video.sh --list       list what exists, newest last
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../logs/rsl_rl/velocity" && pwd)"
PLAYER="${VIDEO_PLAYER:-vlc}"

# Newest run that actually has a rendered video, so an older long run does not
# win the "newest video" slot.
run_dir() {
  find "$ROOT" -mindepth 1 -maxdepth 1 -type d -exec test -d '{}/videos' \; -print |
    sort |
    tail -1
}

list() {
  local dir
  dir="$(run_dir)"
  [[ -n "$dir" ]] || return 0
  find "$dir" -name 'rl-video-step-*.mp4' -type f |
    sed -E 's/.*rl-video-step-([0-9]+)\.mp4/\1 &/' |
    sort -n -k1,1 |
    cut -d' ' -f2-
}

if [[ "${1:-}" == "--list" ]]; then
  list | tail -20
  exit 0
fi

mapfile -t videos < <(list)
if [[ ${#videos[@]} -eq 0 ]]; then
  echo "no videos rendered yet under $ROOT" >&2
  echo "training must be launched with --video True" >&2
  exit 1
fi

target="${videos[-1]}"
if [[ -n "${1:-}" ]]; then
  target="$(list | grep -- "step-$1.mp4" | tail -1 || true)"
  if [[ -z "$target" ]]; then
    echo "no video for env step $1. available:" >&2
    list >&2
    exit 1
  fi
fi

echo "playing $target"
echo "($(ls "${videos[@]}" | wc -l) videos available, newest last)"
exec "$PLAYER" "$target"
