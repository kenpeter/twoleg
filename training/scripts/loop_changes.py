#!/usr/bin/env python
"""Whitelisted reward-weight editor for the TwoLeg training loop.

The loop calls this ONLY with approved (whitelisted) changes so the agent can
never silently mutate arbitrary code. Each change is a small reversible bump to
a known reward term that targets a measured failure mode.

Usage:
  python scripts/loop_changes.py <change_key> [--round N]

Change keys (whitelist):
  bump_knee_flex       -> increase knee_flexion weight by 0.25 (targets stiff shuffle; cap 1.5)
  strengthen_grounding -> increase contact_continuity weight by 0.5 (targets flight-heavy gait; cap 3.0)
Each change refuses when already at cap, so re-running converges instead of drifting.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CFG = Path(__file__).resolve().parents[1] / "twoleg_rl/tasks/velocity/config/twoleg/env_cfgs.py"

WHITELIST = {"bump_knee_flex", "strengthen_grounding"}

KEY_RE = re.compile(
    r'(cfg\.rewards\["knee_flexion"\]\s*=\s*RewardTermCfg\(\s*\n\s*func=knee_flexion,\s*\n\s*weight=)([\d.]+)',
)

GROUND_RE = re.compile(
    r'(cfg\.rewards\["contact_continuity"\]\s*=\s*RewardTermCfg\(\s*\n\s*func=contact_continuity,\s*\n\s*weight=)([\d.]+)',
)


def bump_knee_flex() -> float:
    text = CFG.read_text()
    m = KEY_RE.search(text)
    if not m:
        raise SystemExit("knee_flexion term not found in cfg")
    old = float(m.group(2))
    if old >= 1.5:
        print(f"bump_knee_flex refused: already at cap ({old} >= 1.5)")
        return old
    new = round(old + 0.25, 4)
    new_text = KEY_RE.sub(lambda mm: f"{mm.group(1)}{new}", text, count=1)
    CFG.write_text(new_text)
    return new


def strengthen_grounding() -> float:
    text = CFG.read_text()
    m = GROUND_RE.search(text)
    if not m:
        raise SystemExit("contact_continuity term not found in cfg")
    old = float(m.group(2))
    if old >= 3.0:
        print(f"strengthen_grounding refused: already at cap ({old} >= 3.0)")
        return old
    new = round(old + 0.5, 4)
    new_text = GROUND_RE.sub(lambda mm: f"{mm.group(1)}{new}", text, count=1)
    CFG.write_text(new_text)
    return new


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in WHITELIST:
        print(f"unknown/forbidden change; allowed: {sorted(WHITELIST)}")
        return 2
    key = sys.argv[1]
    if key == "bump_knee_flex":
        new_w = bump_knee_flex()
        print(f"bump_knee_flex -> knee_flexion weight now {new_w}")
    elif key == "strengthen_grounding":
        new_w = strengthen_grounding()
        print(f"strengthen_grounding -> contact_continuity weight now {new_w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
