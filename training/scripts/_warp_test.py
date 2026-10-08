import sys
sys.path.insert(0, ".")
import os
os.environ["WARP_USE_LEGACY_CODEGEN"] = "1"
import torch
import twoleg_rl.tasks.velocity.config.twoleg  # noqa
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

cfg = load_env_cfg("TwoLeg-Velocity-Flat", play=True)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
venv = RslRlVecEnvWrapper(env)
o, _ = venv.reset()
for _ in range(10):
    a = torch.zeros(venv.action_space.shape, device="cpu")
    o, _, _, _ = venv.step(a)
print("STEP OK (no checkpoint load)")
env.close()
