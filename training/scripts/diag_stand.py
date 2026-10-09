"""Render the robot's initial pose under ZERO action with terminations DISABLED,
so we can SEE how it's configured and why it tips >70deg in 26 steps. No
termination = it falls to the ground, visually revealing the instability.
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
env_cfg.terminations = {}  # disable fall termination -> robot falls to ground, visible
robot_cfg = env_cfg.scene.entities["robot"]
robot_cfg.init_state.joint_pos = {
    ".*_hip_test": 0.2, ".*_knee_test": -0.4, ".*_ankle_test": 0.2, ".*_hip_roll_test": 0.0
}
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
    clip = [x if x.dtype == np.uint8 else (np.clip(x, 0, 1) * 255).astype(np.uint8) for x in frames]
    media.write_video("diag_stand_no_term.mp4", clip, fps=30)
    print(f"[diag] wrote {len(clip)} frames -> diag_stand_no_term.mp4")
