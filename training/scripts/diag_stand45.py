"""Render robot at CORRECT spawn height (root_z=0.45, feet should be on ground)
under ZERO action, term disabled, so we can SEE why it still flips. This is the
ground-truth visual of the model problem.
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
env_cfg.terminations = {}
robot_cfg = env_cfg.scene.entities["robot"]
robot_cfg.init_state.pos = (0.0, 0.0, 0.45)
robot_cfg.init_state.joint_pos = {".*_hip_test":0.0,".*_knee_test":0.0,".*_ankle_test":0.0,".*_hip_roll_test":0.0}
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode="rgb_array")
env_wrapped = RslRlVecEnvWrapper(env, clip_actions=False)
obs, _ = env_wrapped.reset()
import mediapy as media, numpy as np
frames = []
for i in range(120):
    a = torch.zeros(1, 8)
    obs, r, d, info = env_wrapped.step(a)
    f = env.render()
    if f is not None:
        ff = f[0] if f.ndim == 4 else f
        frames.append(np.asarray(ff))
env.close()
if frames:
    clip = [x if x.dtype == np.uint8 else (np.clip(x,0,1)*255).astype(np.uint8) for x in frames]
    media.write_video("diag_stand_z045.mp4", clip, fps=30)
    print(f"[diag] wrote {len(clip)} frames -> diag_stand_z045.mp4")
