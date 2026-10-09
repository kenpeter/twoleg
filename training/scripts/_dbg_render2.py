"""Render the CURRENT twoleg.xml model: spawn at STANDING_KEYFRAME, hold a small
crouch (knees bent) with zero action, save ~4s to training/latest_walk.mp4 so we
can SEE whether the robot stands or tips. Also dumps CoM/feet each 30 steps."""
import sys, torch
sys.path.insert(0, "/home/kenpeter/work/torch-rl-algorithms")
sys.path.insert(0, "/home/kenpeter/work/twoleg/training")
from scripts.twoleg_gym_env import TwoLegGymEnv

env = TwoLegGymEnv(device="cuda:0", num_envs=1, max_episode_steps=250, enable_mirroring=False)
obs = env.start()
mjenv = env.env
robot = mjenv.scene["robot"]
# apply a crouch by stepping with PD target? simpler: set joint pos via reset already at keyframe.
# Hold zero action; record CoM vs feet.
import numpy as np
a = torch.zeros((1, env.action_space.shape[0]), device="cuda:0")
frames = []
print("robot base z at spawn:", float(robot.data.root_link_pos_w[0,2]))
for i in range(120):
    o, infos = env.step(a)
    if hasattr(mjenv, "render"):
        try:
            f = mjenv.render()  # may return an image
            if f is not None:
                frames.append(f)
        except Exception as e:
            pass
    if i % 20 == 0:
        bp = robot.data.body_com_pos_w[0].cpu().numpy()
        names = list(robot.body_names)
        fi = [j for j,n in enumerate(names) if n in ("L_foot","R_foot")]
        z = float(robot.data.root_link_pos_w[0,2])
        print(f"step {i}: base_z={z:.3f} feet_z={[round(bp[j,2],3) for j in fi]} feet_y={[round(bp[j,1],3) for j in fi]}")
print("done; frames captured:", len(frames))
