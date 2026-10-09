"""Measure foot-body height as a function of root_z (NO simulation, just initial
pose) to find the spawn height where feet touch ground (foot_z ~ 0). The earlier
'0.45 fix' made the robot float 25cm -> crash -> flip. This finds the real value."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/kenpeter/work/torch-rl-algorithms")
sys.path.insert(0, "/home/kenpeter/work/twoleg/training")
from mjlab.utils.torch import configure_torch_backends
configure_torch_backends()
import twoleg_rl.tasks.velocity.config.twoleg  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

for rz in [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
    ec = load_env_cfg("TwoLeg-Velocity-Flat")
    ec.scene.num_envs = 1; ec.terminations = {}
    rc = ec.scene.entities["robot"]
    rc.init_state.pos = (0.0, 0.0, rz)
    rc.init_state.joint_pos = {".*_hip_test":0.0,".*_knee_test":0.0,".*_ankle_test":0.0,".*_hip_roll_test":0.0}
    env = ManagerBasedRlEnv(cfg=ec, device="cpu")
    w = RslRlVecEnvWrapper(env, clip_actions=False)
    obs, _ = w.reset()
    robot = env.scene["robot"]
    bp = robot.data.body_com_pos_w.cpu().numpy()[0]
    names = list(robot.body_names)
    fi = [i for i,n in enumerate(names) if "foot" in n.lower() and "collision" not in n]
    fz = [bp[i,2] for i in fi]
    # also lowest body overall
    status = 'FLOATS' if min(fz)>0.02 else ('PENETRATES' if min(fz)<-0.02 else 'ON GROUND ~ok')
    print(f"root_z={rz:.3f}: foot_z={[round(x,3) for x in fz]} min_foot={min(fz):.4f} -> {status}")
    env.close()
