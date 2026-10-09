import sys
sys.path.insert(0, ".")
import numpy as np
import torch
from scripts.twoleg_gym_env import TwoLegGymEnv

env = TwoLegGymEnv(device="cpu", num_envs=1, enable_mirroring=False)
print("obs_space:", {k: tuple(v.shape) for k, v in env.observation_space.spaces.items()})
print("act_space:", tuple(env.action_space.shape))
obs = env.start()
print("start obs shapes:", {k: tuple(v.shape) for k, v in obs.items()})
acts = np.random.uniform(-1, 1, size=(1, 8)).astype(np.float32)
for i in range(5):
    obs2, infos = env.step(acts)
    r = infos["rewards"].item() if hasattr(infos["rewards"], "item") else float(infos["rewards"])
    print(f"step {i}: reward={r:.3f} term={bool(infos['terminated'].item())} actor={tuple(obs2['actor'].shape)}")
env.close()
print("ADAPTER OK")
