import sys, torch
sys.path.insert(0, "/home/kenpeter/work/torch-rl-algorithms")
sys.path.insert(0, "/home/kenpeter/work/twoleg/training")
from scripts.twoleg_gym_env import TwoLegGymEnv

env = TwoLegGymEnv(device="cuda:0", num_envs=4, max_episode_steps=250, enable_mirroring=False)
obs = env.start()
print("start obs[actor] shape:", obs["actor"].shape)
# peek at base height from critic obs if present, else actor
import numpy as np
a = torch.zeros((4, env.action_space.shape[0]), device="cuda:0")
for i in range(5):
    o, infos = env.step(a)
    r = infos.get("resets")
    print(f"step {i}: resets.any={bool(r.any()) if r is not None else 'NONE'} | resets sum={int(r.sum()) if r is not None else -1}")
    # try to read base height: actor obs layout [lin(3),ang(3),grav(3),jointpos(8),...]
    # projected gravity z gives tilt; base height not in obs. Use env internals:
    try:
        bz = env.env.scene["robot"].data.root_pos_w[:, 2].cpu().numpy()
        print(f"   base_z = {bz}")
    except Exception as e:
        print("   base_z read failed:", e)
print("action_space:", env.action_space.shape)
