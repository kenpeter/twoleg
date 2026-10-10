import os, glob
run = "/home/kenpeter/work/twoleg/training/logs/rsl_rl/fresh500/2026-10-10_23-24-16"
ev = sorted(glob.glob(os.path.join(run, "events.out*")))[-1]
import tensorboard.backend.event_processing.event_accumulator as ea
acc = ea.EventAccumulator(ev)
acc.Reload()
tags = acc.Tags().get("scalars", [])
print("SCALAR TAGS:", tags[:60])
for tg in tags[:40]:
    try:
        vals = acc.Scalars(tg)
        if vals:
            last = vals[-1].value
            first = vals[0].value
            print("  {0:50s} first={1:.3f} last={2:.3f} n={3}".format(tg, first, last, len(vals)))
    except Exception as e:
        print("  {0} ERR {1}".format(tg, e))
