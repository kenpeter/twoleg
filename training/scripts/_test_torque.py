import os, sys
REPO = "/home/kenpeter/work/twoleg/training"
TRA = "/home/kenpeter/work/torch-rl-algorithms"
for p in (TRA, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WARP_USE_LEGACY_CODEGEN", "1")

import numpy as np
import torch
from scripts.twoleg_gym_env import TwoLegGymEnv
from twoleg_rl.tasks.velocity.config.twoleg.dc_motor_action import clip_effort_dcmotor, EFFORT_LIMIT

# 1) envelope sanity
e = torch.tensor([[3.0, -3.0, 0.5, 0.0]])
v = torch.tensor([[0.0, 4.7, 2.0, 0.0]])
print("EFFORT_LIMIT", EFFORT_LIMIT)
print("clip:", clip_effort_dcmotor(e, v).tolist())

# 2) env builds + steps with torque control
env = TwoLegGymEnv(device="cpu", num_envs=1)
print("obs_space:", {k: tuple(x.shape) for k, x in env.observation_space.spaces.items()})
print("act_space:", tuple(env.action_space.shape))
obs = env.start()
print("start obs:", {k: tuple(x.shape) for k, x in obs.items()})
# push max torque on all joints for a few steps
acts = np.full((1, 8), 1.0, dtype=np.float32)
for i in range(5):
    obs2, infos = env.step(acts)
    r = float(infos["rewards"].item()) if hasattr(infos["rewards"], "item") else float(infos["rewards"])
    print(f"step {i}: reward={r:.3f} resets={bool(infos['resets'].item())} actor={tuple(obs2['actor'].shape)}")
env.close()
print("TORQUE ADAPTER OK")
