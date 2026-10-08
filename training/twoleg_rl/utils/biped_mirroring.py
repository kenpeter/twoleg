"""Left-right mirroring helpers, ported from BipedRobot (MIT).

Source: /home/kenpeter/work/BipedRobot/envs/utils/mirroring.py
Copyright (c) 2025 Pablo Gomez Martinez (MIT, see BipedRobot/LICENSE.txt).

Status: reference implementation, NOT wired into the mjlab PPO loop yet.
BipedRobot's layout is a flat MuJoCo obs (7 + n_joints + 6 + n_joints) with a
12-DOF action; our loop runs a vectorized mjlab ManagerBasedRlEnv with a 6-DOF
duck and symmetry handled (if at all) via the runner's symmetry_cfg. Wiring
this in means adapting the index maps below to the duck's joint order
(L_hip/knee/ankle, R_hip/knee/ankle) and the mjlab obs layout, then proving
symmetric rollouts stay symmetric. Kept here so sim-to-real work can pick it
up without re-deriving the sign conventions.
"""

import numpy as np


def mirror_observation(obs, n_joints):
    """Mirror a MuJoCo observation over the Y-axis (BipedRobot convention)."""
    obs = obs.copy()

    quat_start = 3
    joint_start = 7
    vel_start = 7 + n_joints
    joint_vel_start = vel_start + 6

    obs[1] *= -1

    qx, qy, qz, qw = obs[quat_start:quat_start + 4]
    obs[quat_start:quat_start + 4] = [-qx, qy, -qz, qw]

    joint_angles = obs[joint_start:vel_start]
    half = n_joints // 2
    obs[joint_start:vel_start] = np.concatenate([joint_angles[half:], joint_angles[:half]])

    obs[joint_start + 0] *= -1
    obs[joint_start + 2] *= -1
    obs[joint_start + 5] *= -1
    obs[joint_start + 6] *= -1
    obs[joint_start + 8] *= -1
    obs[joint_start + 11] *= -1

    obs[vel_start + 1] *= -1

    obs[vel_start + 4] *= -1
    obs[vel_start + 5] *= -1

    joint_vels = obs[joint_vel_start:]
    obs[joint_vel_start:] = np.concatenate([joint_vels[half:], joint_vels[:half]])

    obs[joint_vel_start + 0] *= -1
    obs[joint_vel_start + 2] *= -1
    obs[joint_vel_start + 5] *= -1
    obs[joint_vel_start + 6] *= -1
    obs[joint_vel_start + 8] *= -1
    obs[joint_vel_start + 11] *= -1

    return obs


def mirror_action(action, n_joints):
    """Mirror a MuJoCo action over the Y-axis (BipedRobot convention)."""
    action = action.copy()

    half = n_joints // 2
    action = np.concatenate([action[half:], action[:half]])
    action[0] *= -1
    action[2] *= -1
    action[5] *= -1
    action[6] *= -1
    action[8] *= -1
    action[11] *= -1

    return action
