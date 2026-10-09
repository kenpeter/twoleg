"""Confirm feet symmetric left/right and straight-leg PD-hold stability after
the clean leg rebuild."""
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
ec.scene.num_envs = 1; ec.terminations = {}
rc = ec.scene.entities["robot"]
rc.init_state.pos = (0.0, 0.0, 0.0)
rc.init_state.joint_pos = {".*_hip_test":0.0,".*_knee_test":0.0,".*_ankle_test":0.0,".*_hip_roll_test":0.0}
env = ManagerBasedRlEnv(cfg=ec, device="cpu")
w = RslRlVecEnvWrapper(env, clip_actions=False)
obs, _ = w.reset()
robot = env.scene["robot"]
bp = robot.data.body_com_pos_w.cpu().numpy()[0]
names = list(robot.body_names)
root = robot.data.root_link_pos_w.cpu().numpy()[0]
mm = np.asarray(env.sim.model.body_mass).reshape(-1)
bi = np.asarray(getattr(robot, "body_indices", np.arange(1,1+len(names))))
masses = mm[bi].reshape(-1)
com = (bp * masses[:,None]).sum(0)/masses.sum()
for side in ["L","R"]:
    fi = [i for i,n in enumerate(names) if n==f"{side}_foot"][0]
    print(f"{side}_foot body pos = ({bp[fi,0]:.3f},{bp[fi,1]:.3f},{bp[fi,2]:.3f})")
print(f"ROOT pos = ({root[0]:.3f},{root[1]:.3f},{root[2]:.3f})")
print(f"CoM = ({com[0]:.3f},{com[1]:.3f},{com[2]:.3f})")
# stability test
maxt=0.0; minz=9
for i in range(300):
    o,r,d,info = w.step(torch.zeros(1,8))
    ang = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:,2],-1,1)).abs().item()
    maxt=max(maxt,ang); minz=min(minz, robot.data.root_link_pos_w[:,2].item())
deg=maxt*180/3.14159
print(f"straight PD-hold: max_tilt={deg:.1f}deg min_base_z={minz:.4f} -> {'STABLE' if deg<15 else 'TIPS'}")
env.close()
