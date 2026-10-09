"""Find the right mass API on the mjlab articulation, then compute CoM vs foot
polygon at the standing pose. Definitive static-stability test."""
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
d = robot.data
# discover available mass attrs
cands = [a for a in dir(d) if "mass" in a.lower() or "com" in a.lower()]
print("mass/com attrs on robot.data:", cands)
print("body_names[:12]:", robot.body_names[:12])
env.close()
