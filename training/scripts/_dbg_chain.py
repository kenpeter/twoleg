"""Inspect the leg kinematic chain: for each leg, list joints (axis, range), and
the world positions of hip/knee/ankle/foot bodies at the neutral pose (root_z=0.21).
This tells me whether the joint axes form a proper sagittal-plane biped or are
mis-oriented (which would explain why PD can't hold the legs)."""
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

# joints in order (L then R)
for side in ["L", "R"]:
    print(f"\n=== {side} leg ===")
    for jname in [f"{side}_hip_roll_test", f"{side}_hip_test", f"{side}_knee_test", f"{side}_ankle_test"]:
        # find body owning the joint by name
        ji = env.sim.model.joint_name2id(jname) if hasattr(env.sim.model, "joint_name2id") else None
    # just report body positions of key bodies
    for key in ["hip_link", "knee_link", "ankle_link", "foot"]:
        cand = [i for i,n in enumerate(names) if n==f"{side}_{key}"]
        if cand:
            i = cand[0]
            print(f"  {side}_{key:11s} pos=({bp[i,0]:.3f},{bp[i,1]:.3f},{bp[i,2]:.3f})")
# joint axes
print("\n=== joint axes / ranges (from model) ===")
jm = env.sim.model
for jname in ["L_hip_roll_test","L_hip_test","L_knee_test","L_ankle_test"]:
    jid = jm.joint_name2id(jname)
    axis = jm.jnt_axis[jid]
    rng = (jm.jnt_range[jid,0], jm.jnt_range[jid,1])
    print(f"  {jname:16s} axis=({axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}) range=({rng[0]:.2f},{rng[1]:.2f})")
env.close()
