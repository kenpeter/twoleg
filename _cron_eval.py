import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
d = "training/logs/rsl_rl/velocity/2026-10-07_09-33-45_velocity"
f = os.path.join(d, "events.out.tfevents.1791326030.kenpeter-ubuntu.476431.0")
ea = EventAccumulator(f)
ea.Reload()
tags = ea.Tags().get("scalars", [])
print("TAGS:", tags)
def last(tag, n=3):
    try:
        s = ea.Scalars(tag)
        return [(x.step, round(x.value,4)) for x in s[-n:]]
    except Exception as e:
        return "ERR:"+str(e)
for t in tags:
    print("---", t, last(t,3))
