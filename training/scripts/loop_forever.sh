#!/usr/bin/env bash
# Keep the TwoLeg agentic walk loop running non-stop.
# Re-invokes training/scripts/walk_loop.py in a while-true; walk_loop.py resumes
# from .audit/walk_loop_state.json each call and advances one strategy per round.
# Stops only when the biped WALKS (loop exits 0) or a round hard-fails (exit 2).
# GPU power cap 150W is set via nvidia-smi (done once, persists until reboot).
set -u
cd /home/kenpeter/work/twoleg/training
LOG=/tmp/opencode/loop_forever.log
echo "[loop_forever] start $(date -Is)" >> "$LOG"
while true; do
    echo "[loop_forever] invoke $(date -Is)" >> "$LOG"
    uv run --no-sync python scripts/walk_loop.py --rounds 1 --no-reflect >> "$LOG" 2>&1
    rc=$?
    echo "[loop_forever] walk_loop exited rc=$rc $(date -Is)" >> "$LOG"
    if [ "$rc" -eq 0 ]; then
        echo "[loop_forever] WALKS -> stopping loop" >> "$LOG"
        break
    fi
    if [ "$rc" -eq 2 ]; then
        echo "[loop_forever] hard fail -> sleep 60s then retry" >> "$LOG"
        sleep 60
        continue
    fi
    # rc==1: strategies exhausted or round budget spent -> state saved, re-invoke
    sleep 5
done
echo "[loop_forever] end $(date -Is)" >> "$LOG"
