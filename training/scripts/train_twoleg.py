"""TwoLeg training entry.

Imports our task registration (twoleg_rl) so TwoLeg-Velocity-Flat is in the
registry, then reuses unitree_rl_mjlab's train flow.
"""

import sys
import os
from pathlib import Path

# Make our package + the cloned unitree_rl_mjlab repo importable.
ROOT = Path(__file__).resolve().parents[1]
UNITREE_REPO = Path("/home/kenpeter/work/gh-repohub/unitree_rl_mjlab")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UNITREE_REPO))  # exposes `scripts.train`
import twoleg_rl.tasks.velocity.config.twoleg  # noqa: F401  (registers tasks)
from scripts.train import main

# The installed pydantic rejects wandb.Settings(start_method="thread") (extra_forbidden),
# and we have no W&B account. Disable W&B and relax the Settings kwarg so rsl_rl's
# WandbSummaryWriter works offline (TensorBoard backend still records).
os.environ.setdefault("WANDB_MODE", "disabled")
try:
    import wandb

    _orig_settings_init = wandb.Settings.__init__

    def _relaxed_settings_init(self, *args, **kwargs):
        kwargs.pop("start_method", None)
        _orig_settings_init(self, *args, **kwargs)

    wandb.Settings.__init__ = _relaxed_settings_init
except Exception as _e:  # pragma: no cover
    print(f"[warn] could not patch wandb.Settings: {_e}")

if __name__ == "__main__":
    main()
