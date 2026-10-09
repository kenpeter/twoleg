"""Diagnose WHY the robot flips: measure spawn geometry. Is the root at 0.251
with feet NOT touching ground (spawning in air -> crash)? Or feet penetrating?
Report base_z, min foot height, foot_x/y spread (CoM over feet?), leg lengths.
"""
import sys, torch
sys.path.insert(0, "/home/kenpeter/work/torch-rl-algorithms")
sys.path.insert(0, "/home/kenpeter/work/twoleg/training")
from mjlab.utils.torch import configure_torch_backends
configure_torch_backends()
import twoleg_rl.tasks.velocity.config.twoleg  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

env_cfg = load_env_cfg("TwoLeg-Velocity-Flat")
env_cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
env_wrapped = RslRlVecEnvWrapper(env, clip_actions=False)
obs, _ = env_wrapped.reset()
robot = env.scene["robot"]
import numpy as np
base_z = robot.data.root_link_pos_w[:, 2].item()
print(f"base_z (root height) at spawn = {base_z:.4f}")
# foot body positions
for b in ["L_foot", "R_foot"]:
    try:
        body = env.scene[b]
        p = body.data.body_pos_w[:, 0, :].cpu().numpy()[0]
        print(f"  {b}: x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f}")
    except Exception as e:
        print(f"  {b}: {e}")
# all bodies lowest point
bodies = env.scene["robot"].data.body_com_pos_w.cpu().numpy()[0]
minz = bodies[:, 2].min()
print(f"min body z (lowest point of robot) = {minz:.4f}")
print(f"CoM (root) xy = {robot.data.root_link_pos_w[:, :2].cpu().numpy()[0]}")
# foot body names
for b in ["robot/L_foot", "robot/R_foot"]:
    try:
        body = env.scene[b]
        p = body.data.body_com_pos_w[:, 0, :].cpu().numpy()[0]
        print(f"  {b}: x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f}")
    except Exception as e:
        # fallback: print all body names with z
        print(f"  (listing bodies)")
env.close()
