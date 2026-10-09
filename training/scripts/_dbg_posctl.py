"""Test: with PD JOINT-POSITION control, does zero action hold the bent-knee
standing pose stably (survive >> 26 steps)? If yes, position control is the fix
for the cold-start collapse (torque control can't stand -> can't learn)."""
import sys, torch
sys.path.insert(0, "/home/kenpeter/work/torch-rl-algorithms")
sys.path.insert(0, "/home/kenpeter/work/twoleg/training")
from mjlab.utils.torch import configure_torch_backends
configure_torch_backends()
import twoleg_rl.tasks.velocity.config.twoleg  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

def test(mode, knee, hip, ankle):
    env_cfg = load_env_cfg("TwoLeg-Velocity-Flat")
    env_cfg.scene.num_envs = 1
    if mode == "pos":
        env_cfg.actions["joint_pos"] = JointPositionActionCfg(
            entity_name="robot", actuator_names=(r"^(L|R)_(hip_roll|hip|knee|ankle)_test$",),
            scale=0.3, use_default_offset=True
        )
    robot_cfg = env_cfg.scene.entities["robot"]
    robot_cfg.init_state.joint_pos = {
        ".*_hip_test": hip, ".*_knee_test": knee, ".*_ankle_test": ankle, ".*_hip_roll_test": 0.0
    }
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=False)
    obs, _ = env_wrapped.reset()
    robot = env.scene["robot"]
    z0 = robot.data.root_link_pos_w[:, 2].clone()
    fall = 0
    for i in range(200):
        a = torch.zeros(1, 8)
        obs, r, d, info = env_wrapped.step(a)
        if bool(d[0]):
            fall = i
            break
    zend = robot.data.root_link_pos_w[:, 2]
    print(f"{mode} knee={knee}: z0={z0.item():.3f} zend={zend.item():.3f} fell_at={fall} (survived {200 if fall==0 else fall})")
    env.close()

test("pos", -0.4, 0.2, 0.2)
test("pos", -0.3, 0.15, 0.15)
