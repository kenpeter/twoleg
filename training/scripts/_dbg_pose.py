"""Test whether a BENT-KNEE standing init pose keeps the robot upright under
zero action (stable), vs the current straight-leg pose. We patch the init
joint_pos via env cfg and measure base height drift + whether it falls.
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

def test_pose(knee, hip, ankle, label):
    env_cfg = load_env_cfg("TwoLeg-Velocity-Flat")
    env_cfg.scene.num_envs = 1
    # override init joint_pos on the robot entity
    robot_cfg = env_cfg.scene.entities["robot"]
    robot_cfg.init_state.joint_pos = {
        ".*_hip_test": hip, ".*_knee_test": knee, ".*_ankle_test": ankle, ".*_hip_roll_test": 0.0
    }
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=False)
    obs, _ = env_wrapped.reset()
    # read base height via mjlab data
    robot = env.scene["robot"]
    z0 = robot.data.root_link_pos_w[:, 2].clone()
    fall = 0
    for i in range(150):
        a = torch.zeros(1, 8)
        obs, r, d, info = env_wrapped.step(a)
        z = robot.data.root_link_pos_w[:, 2]
        if bool(d[0]):
            fall = i
            break
    zend = robot.data.root_link_pos_w[:, 2]
    print(f"{label}: knee={knee} hip={hip} ankle={ankle} | z0={z0.item():.3f} zend={zend.item():.3f} fell_at={fall}")
    env.close()

# current straight-leg
test_pose(0.0, 0.0, 0.0, "STRAIGHT")
# bent knee attempts
test_pose(-0.4, 0.2, 0.2, "BENT-A")
test_pose(-0.6, 0.3, 0.3, "BENT-B")
test_pose(-0.3, 0.15, 0.15, "BENT-C")
