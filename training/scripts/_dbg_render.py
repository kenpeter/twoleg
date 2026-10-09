import sys, torch
sys.path.insert(0, "/home/kenpeter/work/torch-rl-algorithms")
sys.path.insert(0, "/home/kenpeter/work/twoleg/training")
from scripts.twoleg_gym_env import TwoLegGymEnv
import mujoco

env = TwoLegGymEnv(device="cuda:0", num_envs=1, max_episode_steps=250, enable_mirroring=False)
obs = env.start()

# Access underlying mjlab env to render
mjenv = env.env
# find the mujoco data/site for rendering
print("env type:", type(mjenv))
print("has render:", hasattr(mjenv, 'render'))
# Inspect scene robot base height over a few steps with ZERO action
a = torch.zeros((1, env.action_space.shape[0]), device="cuda:0")
for i in range(10):
    o, infos = env.step(a)
    r = infos.get("resets")
    print(f"step {i}: resets.any={bool(r.any()) if r is not None else 'NONE'}")
print("done")
