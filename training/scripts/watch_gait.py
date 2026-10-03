"""Live gait progress for the newest (or a named) TwoLeg training run.

Reads the run's TensorBoard event file and prints the terms that decide whether
the policy walks, not the terms that go up for free. Refreshes until Ctrl-C.

    uv run python scripts/watch_gait.py
    uv run python scripts/watch_gait.py --run 2026-09-28_21-53-46_velocity
    uv run python scripts/watch_gait.py --watch-videos

Targets, from microduck_rl's own velocity run at iteration 2499:
track_linear_velocity 1.184, air_time 0.837, error_vel_xy 0.440, upright 1.621.
The pre-alignment TwoLeg run plateaued at error_vel_xy 0.46 with no air_time
term at all, which is the failure this script exists to make visible.
"""

import argparse
import glob
import os
import time

from tensorboard.backend.event_processing import event_accumulator

LOG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
    "rsl_rl",
    "velocity",
)

TERMS = (
    ("Episode_Reward/track_linear_velocity", "1.0"),
    ("Episode_Reward/air_time", "0.5"),
    ("Episode_Reward/upright", "1.5"),
    ("Episode_Reward/pose", "0.5"),
    ("Episode_Reward/foot_clearance", "< 0.05"),
    ("Episode_Reward/foot_slip", "< 0.05"),
    ("Metrics/twist/error_vel_xy", "< 0.40"),
    ("Metrics/air_time_mean", "> 0.05"),
    ("Metrics/slip_velocity_mean", "< 0.05"),
    ("Episode_Termination/fell_over", "< 5"),
    ("Episode_Termination/nan_state", "0"),
    ("Train/mean_episode_length", "> 800"),
    ("Train/mean_reward", "-"),
)


def newest_run(root):
    runs = sorted(
        (d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)),
        key=os.path.getmtime,
    )
    if not runs:
        raise SystemExit(f"no runs under {root}")
    return runs[-1]


def event_file(run):
    found = glob.glob(os.path.join(run, "events.out.tfevents.*"))
    if not found:
        raise SystemExit(f"no event file in {run}")
    return found[0]


def tail_of(path):
    ea = event_accumulator.EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    out = {}
    for tag, target in TERMS:
        if tag not in tags:
            out[tag] = (None, target)
            continue
        series = ea.Scalars(tag)
        values = [s.value for s in series]
        window = values[-max(1, len(values) // 20) :] or values[-1:]
        out[tag] = (sum(window) / len(window), target)
    return out, (series[-1].step if series else 0)


def satisfied(value, target):
    if value is None or target == "-":
        return None
    if target.startswith("<"):
        return value < float(target[1:])
    if target.startswith(">"):
        return value > float(target[1:])
    return value >= float(target)


def render(run, rows, step):
    print(f"\n{os.path.basename(run.rstrip('/'))}   iteration {step}")
    print(f"{'term':40s} {'last 5%':>10s} {'target':>9s}")
    print("-" * 63)
    for tag, target in TERMS:
        value, tgt = rows[tag]
        shown = "absent" if value is None else f"{value:10.4g}"
        flag = ""
        if value is not None and tgt != "-":
            flag = "ok " if satisfied(value, tgt) else "  -"
        print(f"{tag:40s} {shown:>10s} {tgt:>9s} {flag}")


def watch_videos(run):
    videos = sorted(
        glob.glob(os.path.join(run, "videos", "**", "*.mp4"), recursive=True)
    )
    if not videos:
        print("no videos rendered yet")
        return
    latest = videos[-1]
    print(f"latest training video: {latest}")
    for path in videos[-5:]:
        print(f"  {os.path.relpath(path, run)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="run directory name under logs/rsl_rl/velocity")
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--watch-videos", action="store_true")
    args = parser.parse_args()

    run = (
        os.path.join(LOG_ROOT, args.run)
        if args.run
        else newest_run(LOG_ROOT)
    )
    print(f"watching {run}")

    while True:
        try:
            rows, step = tail_of(event_file(run))
            render(run, rows, step)
            if args.watch_videos:
                watch_videos(run)
        except Exception as exc:  # the run may still be writing
            print(f"waiting for scalars: {exc}")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
