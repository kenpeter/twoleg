"""Sweep root spawn height; find height where the robot's lowest body sits ~0
(feet on ground, not penetrating) AND it stays upright under zero action.
The current 0.251 spawns feet 19cm underground -> explodes -> flips to 178deg.
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

for rz in [0.30, 0.40, 0.45, 0.49, 0.52, 0.55, 0.60]:
    env_cfg = load_env_cfg("TwoLeg-Velocity-Flat")
    env_cfg.scene.num_envs = 1
    env_cfg.terminations = {}
    robot_cfg = env_cfg.scene.entities["robot"]
    robot_cfg.init_state.pos = (0.0, 0.0, rz)
    # neutral straight pose
    robot_cfg.init_state.joint_pos = {
        ".*_hip_test": 0.0, ".*_knee_test": 0.0,
        ".*_ankle_test": 0.0, ".*_hip_roll_test": 0.0,
    }
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=False)
    obs, _ = env_wrapped.reset()
    robot = env.scene["robot"]
    bodies = robot.data.body_com_pos_w.cpu().numpy()[0]
    minz = bodies[:, 2].min()
    # step 40 with zero action, measure max tilt
    max_tilt = 0.0
    for i in range(40):
        a = torch.zeros(1, 8)
        obs, r, d, info = env_wrapped.step(a)
        pg = robot.data.projected_gravity_b[:, 2]
        ang = torch.acos(torch.clamp(-pg, -1.0, 1.0)).abs().item()
        max_tilt = max(max_tilt, ang)
    env.close()
    deg = max_tilt * 180 / 3.14159
    print(f"root_z={rz:.3f}: min_body_z={minz:.4f}  max_tilt_after40={deg:.1f}deg  {'STABLE' if deg<20 and minz>-0.02 else ''}")
