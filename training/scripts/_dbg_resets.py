import sys, torch
sys.path.insert(0, "/home/kenpeter/work/torch-rl-algorithms")
sys.path.insert(0, "/home/kenpeter/work/twoleg/training")
from scripts.twoleg_gym_env import TwoLegGymEnv

env = TwoLegGymEnv(device="cuda:0", num_envs=8, max_episode_steps=250, enable_mirroring=False)
obs = env.start()
print("start obs type:", type(obs), "keys:", list(obs.keys()) if isinstance(obs, dict) else obs)
A = env.action_space.shape[0]
print("action dim A =", A, "obs[actor] shape:", obs["actor"].shape)

reset_events = 0
for i in range(600):
    a = torch.zeros((8, A), device="cuda:0")
    o, infos = env.step(a)
    r = infos.get("resets")
    if r is None:
        print(f"step {i}: infos has no 'resets' key! keys={list(infos.keys())}")
        break
    if bool(r.any()):
        reset_events += 1
        print(f"step {i}: RESET fired (any={bool(r.any())}), resets shape={tuple(r.shape)}")
        break
    if i == 599:
        print(f"step {i}: NO reset after 600 steps. resets all-false? sample={r[:3].tolist()}")
print("reset_events =", reset_events)
