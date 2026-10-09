"""CoM vs foot-polygon — array-safe. Decisive static-stability test."""
import sys, numpy as np, torch
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
body_pos = robot.data.body_com_pos_w.cpu().numpy()[0]
names = list(robot.body_names)
mm = np.asarray(env.sim.model.body_mass)
mm = mm.reshape(-1) if mm.ndim==2 else mm          # (124,)
bi = np.asarray(getattr(robot, "body_indices", np.arange(1,1+len(names))))
masses = mm[bi].reshape(-1)
assert len(masses)==len(names)==len(body_pos), (len(masses),len(names),len(body_pos))
print(f"TOTAL MASS = {masses.sum():.4f} kg")
com = (body_pos * masses[:,None]).sum(0)/masses.sum()
print(f"CoM xyz = ({com[0]:.3f},{com[1]:.3f},{com[2]:.3f})")
foot_idx = [i for i,n in enumerate(names) if "foot" in n.lower()]
print("feet:", [(names[i], tuple(body_pos[i].round(3))) for i in foot_idx])
if foot_idx:
    fx=[body_pos[i,0] for i in foot_idx]; fy=[body_pos[i,1] for i in foot_idx]
    inside=(min(fx)<=com[0]<=max(fx)) and (min(fy)<=com[1]<=max(fy))
    print(f"foot x=[{min(fx):.3f},{max(fx):.3f}] y=[{min(fy):.3f},{max(fy):.3f}]")
    print(f"CoM x={com[0]:.3f} y={com[1]:.3f} -> INSIDE? {inside} {'STABLE' if inside else 'UNSTABLE (CoM off feet)'}")
else:
    print("no foot bodies; sample names:", names[:15])
env.close()
