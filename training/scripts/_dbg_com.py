"""Definitive stability check: is the robot's center of mass within the foot
support polygon at the standing pose? If CoM xy is outside the feet, the robot
is statically unstable -> it MUST tip no matter the control mode. This is THE
test that separates 'pose/balance problem' from 'control problem'."""
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
ec.scene.num_envs = 1
ec.terminations = {}
robot_cfg = ec.scene.entities["robot"]
robot_cfg.init_state.pos = (0.0, 0.0, 0.45)
robot_cfg.init_state.joint_pos = {".*_hip_test":0.0,".*_knee_test":0.0,".*_ankle_test":0.0,".*_hip_roll_test":0.0}
env = ManagerBasedRlEnv(cfg=ec, device="cpu")
w = RslRlVecEnvWrapper(env, clip_actions=False)
obs, _ = w.reset()
robot = env.scene["robot"]

# total mass and per-body mass
masses = robot.root_physx_view.get_masses().cpu().numpy()[0]  # per-body mass
total = masses.sum()
print(f"TOTAL MASS = {total:.4f} kg  (robot spec = 4.5 kg)  bodies={len(masses)}")

# body positions at spawn
body_pos = robot.data.body_com_pos_w.cpu().numpy()[0]   # (N,3)
body_names = robot.body_names
# find foot bodies by name
foot_idx = [i for i,n in enumerate(body_names) if "foot" in n.lower()]
print("foot bodies:", [(body_names[i], body_pos[i]) for i in foot_idx])

# CoM xy (mass-weighted)
com = (body_pos * masses[:,None]).sum(0) / total
print(f"CoM xyz = {com}")

# foot support polygon in xy
if foot_idx:
    fx = [body_pos[i,0] for i in foot_idx]
    fy = [body_pos[i,1] for i in foot_idx]
    print(f"foot x range=[{min(fx):.3f},{max(fx):.3f}]  y range=[{min(fy):.3f},{max(fy):.3f}]")
    print(f"CoM x={com[0]:.3f} y={com[1]:.3f}")
    inside = (min(fx) <= com[0] <= max(fx)) and (min(fy) <= com[1] <= max(fy))
    print(f"CoM INSIDE foot polygon? {inside}  -> {'STATICALLY STABLE' if inside else 'STATICALLY UNSTABLE (CoM off feet)'}")
env.close()
