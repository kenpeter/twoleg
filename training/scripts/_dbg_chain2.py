"""Clean combined diagnostic at root_z=0.21:
- root/torso world pos
- both foot body positions
- mass-weighted CoM
- joint axes + ranges (via mjlab model API)
This confirms whether feet are off-center (geometry bug) and shows joint axes."""
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
rc.init_state.pos = (0.0, 0.0, 0.21)
rc.init_state.joint_pos = {".*_hip_test":0.0,".*_knee_test":0.0,".*_ankle_test":0.0,".*_hip_roll_test":0.0}
env = ManagerBasedRlEnv(cfg=ec, device="cpu")
w = RslRlVecEnvWrapper(env, clip_actions=False)
obs, _ = w.reset()
robot = env.scene["robot"]
bp = robot.data.body_com_pos_w.cpu().numpy()[0]
names = list(robot.body_names)
mm = np.asarray(env.sim.model.body_mass).reshape(-1)
bi = np.asarray(getattr(robot, "body_indices", np.arange(1,1+len(names))))
masses = mm[bi].reshape(-1)
com = (bp * masses[:,None]).sum(0)/masses.sum()
root = robot.data.root_link_pos_w.cpu().numpy()[0]
print(f"ROOT/torso pos = ({root[0]:.3f},{root[1]:.3f},{root[2]:.3f})")
for side in ["L","R"]:
    fi = [i for i,n in enumerate(names) if n==f"{side}_foot"][0]
    print(f"{side}_foot body pos = ({bp[fi,0]:.3f},{bp[fi,1]:.3f},{bp[fi,2]:.3f})")
print(f"CoM = ({com[0]:.3f},{com[1]:.3f},{com[2]:.3f})")
print("feet x range:", [round(bp[[i for i,n in enumerate(names) if n==f"{s}_foot"][0],0],3) for s in ['L','R']],
      "y range:", [round(bp[[i for i,n in enumerate(names) if n==f"{s}_foot"][0],1],3) for s in ['L','R']])
# joint axes via mjlab
jm = env.sim.model
print("\njoint axes:")
for jname in ["L_hip_roll_test","L_hip_test","L_knee_test","L_ankle_test","R_hip_roll_test","R_hip_test","R_knee_test","R_ankle_test"]:
    jid = int(np.where(np.asarray(jm.joint_names)==jname)[0][0]) if hasattr(jm,"joint_names") else None
    if jid is None:
        print(f"  {jname}: (names api missing)"); continue
    ax = np.asarray(jm.jnt_axis[jid])
    rng = np.asarray(jm.jnt_range[jid])
    print(f"  {jname:16s} axis=({ax[0]:.2f},{ax[1]:.2f},{ax[2]:.2f}) range=({rng[0]:.2f},{rng[1]:.2f})")
env.close()
