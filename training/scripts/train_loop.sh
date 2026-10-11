#!/usr/bin/env bash
# Keep TwoLeg training running.
#
# Each round resumes from the numerically newest checkpoint under
# logs/rsl_rl/<exp>/<run>/. Order is numeric on the model_<iter> suffix on
# purpose: mjlab's own get_checkpoint_path resolves --agent.load-run and
# --agent.load-checkpoint as regexes and takes the highest ALPHABETICAL match,
# which picks model_900 over model_1100 in the same run.
#
# After every round, scripts/latest.py collapses the experiment back to one
# model (KEEP, default 1) and one clip at training/latest_walk.mp4.
#
# Bounded on purpose: after MAX_STALE_ROUNDS rounds that fail to advance the
# iteration counter, it exits instead of spinning forever.
set -u

cd "$(dirname "$0")/.."
PY=.venv/bin/python
TASK=TwoLeg-Velocity-Flat
EXPERIMENT=${EXPERIMENT:-fresh500}
NUM_ENVS=${NUM_ENVS:-2048}
MAX_ITERATIONS=${MAX_ITERATIONS:-1000000}
MAX_STALE_ROUNDS=${MAX_STALE_ROUNDS:-3}
KEEP=${KEEP:-1}
LOG=${LOG:-/tmp/opencode/train_loop.log}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export WARP_USE_LEGACY_CODEGEN=${WARP_USE_LEGACY_CODEGEN:-1}

newest() {
  "$PY" - <<'EOF'
import sys
from pathlib import Path
root = Path("logs/rsl_rl")
best = None
for ck in set(root.glob("*/model_*.pt")) | set(root.glob("*/*/model_*.pt")):
    try:
        it = int(ck.stem.rsplit("_", 1)[1])
    except ValueError:
        continue
    key = (ck.parent.parent.name, it)
    if best is None or key > best[0]:
        best = (key, ck)
if best is None:
    print("none none 0")
else:
    print(best[1].parent.name, best[1].name, best[0][1])
EOF
}

stale=0
while true; do
  read -r run ckpt iter <<<"$(newest)"
  echo "[loop] $(date +%H:%M:%S) resuming run=$run ckpt=$ckpt iter=$iter" >> "$LOG"
  if [ "$ckpt" = "none" ]; then
    "$PY" train_twoleg.py "$TASK" \
      --env.scene.num-envs "$NUM_ENVS" \
      --agent.max-iterations "$MAX_ITERATIONS" \
      --agent.experiment-name "$EXPERIMENT" >> "$LOG" 2>&1
  else
    "$PY" train_twoleg.py "$TASK" \
      --env.scene.num-envs "$NUM_ENVS" \
      --agent.max-iterations "$MAX_ITERATIONS" \
      --agent.experiment-name "$EXPERIMENT" \
      --agent.resume True \
      --agent.load-run "$run" \
      --agent.load-checkpoint "$ckpt" >> "$LOG" 2>&1
  fi
  status=$?
  # Collapse to one model and one clip, then render a fresh clip if it moved on.
  "$PY" scripts/latest.py --keep "$KEEP" >> "$LOG" 2>&1
  read -r _ _ iter_after <<<"$(newest)"
  echo "[loop] $(date +%H:%M:%S) round exited status=$status iter_after=$iter_after" >> "$LOG"
  if [ "$iter_after" -le "$iter" ]; then
    stale=$((stale + 1))
    echo "[loop] no progress ($stale/$MAX_STALE_ROUNDS)" >> "$LOG"
    if [ "$stale" -ge "$MAX_STALE_ROUNDS" ]; then
      echo "[loop] stopping: $MAX_STALE_ROUNDS rounds with no new checkpoint" >> "$LOG"
      exit 1
    fi
  else
    stale=0
  fi
  sleep 20
done