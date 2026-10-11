"""Keep exactly one model and one walk clip: the latest.

Two jobs, in this order:

  prune   delete every model_*.pt except the newest `keep`, and drop run
          directories that are left with neither a checkpoint nor a tensorboard
          event file. Checkpoint order is numeric on the model_<iter> suffix;
          a plain lexicographic sort returns model_900 ahead of model_1100.
  record  render training/latest_walk.mp4 from that same newest checkpoint,
          unless latest_walk.json already names it and the clip still decodes.

Checkpoints are found both directly under <log-root>/<experiment>/ and inside
<log-root>/<experiment>/<run>/, so a run that is mid-training and an already
collapsed experiment directory both resolve.

Usage:
  uv run --no-sync python scripts/latest.py              # prune to 1, then record
  uv run --no-sync python scripts/latest.py --keep 3     # keep the 3 newest
  uv run --no-sync python scripts/latest.py --no-prune   # record only
  uv run --no-sync python scripts/latest.py --dry-run    # show the plan
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = ROOT / "logs" / "rsl_rl"
RECORDER = ROOT / "record_clip.py"
VIDEO = ROOT / "latest_walk.mp4"
SIDECAR = ROOT / "latest_walk.json"
MODEL_LINK = ROOT / "latest_model.pt"
DURATION_TOLERANCE = 0.9


def ckpt_iter(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def all_ckpts(log_root: Path) -> list[tuple[int, Path]]:
    found = []
    for ck in set(log_root.glob("*/model_*.pt")) | set(log_root.glob("*/*/model_*.pt")):
        try:
            found.append((ckpt_iter(ck), ck))
        except ValueError:
            continue
    return sorted(found, key=lambda item: item[0], reverse=True)


def probe_duration(path: Path) -> float | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return None
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def decodes_clean(path: Path) -> bool:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return out.returncode == 0 and not out.stderr.strip()


def video_is_current(ckpt: Path, secs: float) -> tuple[bool, str]:
    if not VIDEO.exists():
        return False, "no clip"
    if not SIDECAR.exists():
        return False, "no sidecar"
    duration = probe_duration(VIDEO)
    if duration is None or duration < secs * DURATION_TOLERANCE:
        return False, "missing or short"
    if not decodes_clean(VIDEO):
        return False, "decode errors"
    try:
        meta = json.loads(SIDECAR.read_text())
    except json.JSONDecodeError:
        return False, "bad sidecar"
    if Path(meta.get("ckpt", "")) != ckpt.resolve():
        return False, f"stale, shows {Path(meta.get('ckpt', '?')).name}"
    return True, f"{duration:.2f}s from {ckpt.name}"


def link_model(ckpt: Path) -> None:
    target = os.path.relpath(ckpt.resolve(), ROOT)
    if MODEL_LINK.is_symlink() or MODEL_LINK.exists():
        MODEL_LINK.unlink()
    MODEL_LINK.symlink_to(target)


def prune(ckpts: list[tuple[int, Path]], keep: int, dry_run: bool) -> int:
    doomed = ckpts[keep:]
    touched = {ck.parent for _, ck in doomed}
    for _, ck in doomed:
        if dry_run:
            print(f"[latest] would delete {ck}", flush=True)
            continue
        ck.unlink(missing_ok=True)
        print(f"[latest] deleted {ck.name} from {ck.parent.name}", flush=True)

    for run in sorted(touched):
        if list(run.glob("model_*.pt")):
            continue
        if list(run.glob("events.out.tfevents*")):
            continue
        if dry_run:
            print(f"[latest] would remove emptied run dir {run}", flush=True)
            continue
        shutil.rmtree(run)
        print(f"[latest] removed emptied run dir {run.name}", flush=True)
    return len(doomed)


def record(ckpt: Path, secs: float, task: str) -> bool:
    env = {**os.environ, "MUJOCO_GL": "egl", "WARP_USE_LEGACY_CODEGEN": "1"}
    proc = subprocess.run(
        [sys.executable, str(RECORDER),
         "--ckpt", str(ckpt.resolve()),
         "--out", str(VIDEO.resolve()),
         "--secs", str(secs),
         "--task", task],
        cwd=ROOT, env=env,
    )
    if proc.returncode != 0:
        return False
    duration = probe_duration(VIDEO)
    if duration is None or duration < secs * DURATION_TOLERANCE:
        return False
    SIDECAR.write_text(json.dumps({
        "ckpt": str(ckpt.resolve()),
        "iteration": ckpt_iter(ckpt),
        "video": VIDEO.name,
        "duration_s": round(duration, 3),
        "task": task,
        "rendered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    ap.add_argument("--keep", type=int, default=1, help="checkpoints to keep (default 1)")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-render even if current")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--task", default="TwoLeg-Velocity-Flat")
    args = ap.parse_args()

    log_root = Path(args.log_root).resolve()
    if not log_root.is_dir():
        print(f"[latest] no log root at {log_root}", flush=True)
        return 1
    ckpts = all_ckpts(log_root)
    if not ckpts:
        print(f"[latest] no checkpoints under {log_root}", flush=True)
        return 1

    newest_iter, newest = ckpts[0]
    print(f"[latest] newest checkpoint {newest} (iteration {newest_iter})", flush=True)

    if not args.no_prune:
        removed = prune(ckpts, max(args.keep, 1), args.dry_run)
        print(f"[latest] pruned {removed} checkpoint(s), kept {min(args.keep, len(ckpts))}", flush=True)

    if not args.dry_run:
        link_model(newest)

    if args.no_record:
        return 0

    ok, why = (False, "forced") if args.force else video_is_current(newest, args.secs)
    if ok:
        print(f"[latest] clip current ({why})", flush=True)
        return 0
    if args.dry_run:
        print(f"[latest] PLAN render {newest.name} ({why}) -> {VIDEO.name}", flush=True)
        return 0

    print(f"[latest] rendering {newest.name} ({why})", flush=True)
    started = time.time()
    if record(newest, args.secs, args.task):
        duration = probe_duration(VIDEO)
        print(f"[latest] wrote {VIDEO.name} {duration:.2f}s in {time.time()-started:.0f}s", flush=True)
        return 0
    print(f"[latest] render FAILED after {time.time()-started:.0f}s", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())