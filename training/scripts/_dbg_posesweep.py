"""Sweep candidate standing poses under ZERO action (term disabled) and measure
max base tilt (acos of projected gravity z) over 60 steps. Report which pose
stays upright (tilt < ~20deg = stable enough to learn from).
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

POSES = {
    "bent_knee_-0.4":  {"hip":0.2,"knee":-0.4,"ankle":0.2,"roll":0.0},
    "straight_0":      {"hip":0.0,"knee":0.0,"ankle":0.0,"roll":0.0},
    "knee_-0.2_hip_0": {"hip":0.0,"knee":-0.2,"ankle":0.0,"roll":0.0},
    "knee_-0.6_hip_0.3":{"hip":0.3,"knee":-0.6,"ankle":0.3,"roll":0.0},
    "knee_-0.5_hip_-0.3":{"hip":-0.3,"knee":-0.6,"ankle":0.3,"roll":0.0},
    "knee_-0.3_hip_0.15_ank_0.15":{"hip":0.15,"knee":-0.3,"ankle":0.15,"roll":0.0},
}

def tilt(pose):
    env_cfg = load_env_cfg("TwoLeg-Velocity-Flat")
    env_cfg.scene.num_envs = 1
    env_cfg.terminations = {}
    robot_cfg = env_cfg.scene.entities["robot"]
    robot_cfg.init_state.joint_pos = {
        ".*_hip_test": pose["hip"], ".*_knee_test": pose["knee"],
        ".*_ankle_test": pose["ankle"], ".*_hip_roll_test": pose["roll"],
    }
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=False)
    obs, _ = env_wrapped.reset()
    robot = env.scene["robot"]
    max_tilt = 0.0
    for i in range(60):
        a = torch.zeros(1, 8)
        obs, r, d, info = env_wrapped.step(a)
        pg = robot.data.projected_gravity_b[:, 2]
        ang = torch.acos(torch.clamp(-pg, -1.0, 1.0)).abs().item()
        max_tilt = max(max_tilt, ang)
    env.close()
    return max_tilt

for name, pose in POSES.items():
    try:
        t = tilt(pose)
        deg = t * 180 / 3.14159
        print(f"{name}: max_tilt={deg:.1f}deg  {'STABLE' if deg < 20 else 'tips'}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
