"""Test standing at the CORRECT spawn height (root_z=0.21, feet on ground).
PD position control, zero action, measure max tilt over 300 steps."""
import sys, torch
sys.path.insert(0, "/home/kenpeter/work/torch-rl-algorithms")
sys.path.insert(0, "/home/kenpeter/work/twoleg/training")
from mjlab.utils.torch import configure_torch_backends
configure_torch_backends()
import twoleg_rl.tasks.velocity.config.twoleg  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

ec = load_env_cfg("TwoLeg-Velocity-Flat")
ec.scene.num_envs = 1; ec.terminations = {}
robot_cfg = ec.scene.entities["robot"]
robot_cfg.init_state.pos = (0.0, 0.0, 0.21)
robot_cfg.init_state.joint_pos = {".*_hip_test":0.0,".*_knee_test":0.0,".*_ankle_test":0.0,".*_hip_roll_test":0.0}
env = ManagerBasedRlEnv(cfg=ec, device="cpu")
w = RslRlVecEnvWrapper(env, clip_actions=False)
obs, _ = w.reset()
robot = env.scene["robot"]
maxt = 0.0; minz = 9
for i in range(300):
    o, r, d, info = w.step(torch.zeros(1, 8))
    ang = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:,2], -1, 1)).abs().item()
    maxt = max(maxt, ang)
    minz = min(minz, robot.data.root_link_pos_w[:,2].item())
deg = maxt * 180 / 3.14159
print(f"root_z=0.21 straight PD-hold: max_tilt={deg:.1f}deg  min_base_z={minz:.4f}  -> {'STABLE' if deg<15 else 'TIPS'}")
env.close()
