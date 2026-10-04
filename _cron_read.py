import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

f = sys.argv[1]
ea = EventAccumulator(f)
ea.Reload()
scalars = ea.Tags().get('scalars', [])
print("NUM_SCALAR_TAGS", len(scalars))
for tag in sorted(scalars):
    if any(t in tag.lower() for t in ['reward','episode','length','track','command','lin_vel','ang_vel','error','upright','orient','nan','mean','base','height']):
        s = ea.Scalars(tag)
        last_step, last_val = s[-1].step, s[-1].value
        print(f"{tag} | step={last_step} | {last_val:.4f}")
