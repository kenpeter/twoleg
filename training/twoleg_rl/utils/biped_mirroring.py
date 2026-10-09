"""Left-right (Y-axis) mirroring for the TwoLeg mjlab obs/action layout.

Ported from BipedRobot's mirroring (MIT, Pablo Gomez Martinez) but rewritten for
our 36/48-dim mjlab observation groups and 8-DOF action (NOT the BipedRobot flat
12-DOF layout the old version assumed).

Joint order (from env): [L_hip_roll, L_hip, L_knee, L_ankle,
                         R_hip_roll, R_hip, R_knee, R_ankle]
A Y-axis reflection swaps L<->R and inverts the sign of every joint angle/velocity
(reflection flips handedness, so all joint rotations reverse). Same for actions.

Actor obs (36): [0:3] lin_vel, [3:6] ang_vel, [6:9] projected_gravity,
                [9:17] joint_pos, [17:25] joint_vel, [25:33] actions, [33:36] command
Critic obs (48) = actor(36) + [36:38] foot_height, [38:40] foot_air_time,
                  [40:42] foot_contact, [42:48] foot_contact_forces
"""

import numpy as np

# Joint index map: L half (0-3) <-> R half (4-7)
_L = slice(0, 4)
_R = slice(4, 8)

# Under a Y-axis reflection, angular velocity y-component (and the y parts of
# linear velocity / gravity) flip; the x and z components swap sign too only for
# cross products — but for body-frame vectors the rule is: reflect over Y flips
# the y component of a vector, leaves x,z, and inverts the sign of rotations
# about x and z while rotations about y are also inverted. We handle each.


def mirror_observation(obs: np.ndarray) -> np.ndarray:
    """Mirror a flat actor or critic observation over the Y-axis (L<->R)."""
    obs = obs.copy()
    # --- vectors (lin_vel, ang_vel, projected_gravity) = first 9 dims ---
    # Reflection over Y: (x, y, z) -> (x, -y, z) for a position-like vector,
    # but for angular velocity/axis rotations the Y component flips and the
    # X/Z rotational components invert. Simplest consistent rule for body-frame
    # vectors measured in the (mirrored) body frame: flip y, flip x and z too
    # is WRONG; correct is flip y only for translational, and for the
    # projected_gravity (torso up vector) flip y. We flip y on all three vectors.
    obs[1] *= -1.0   # lin_vel y
    obs[4] *= -1.0   # ang_vel y
    obs[7] *= -1.0   # projected_gravity y

    # --- joint_pos [9:17] / joint_vel [17:25] / actions [25:33] ---
    for start in (9, 17, 25):
        j = obs[start:start + 8]
        # swap L<->R then negate (reflection inverts all joint rotations)
        swapped = np.concatenate([j[_R], j[_L]])
        obs[start:start + 8] = -swapped

    # --- command [33:36] (lin_x, lin_y, ang_z): mirror y and yaw ---
    obs[34] *= -1.0  # lin_y
    obs[35] *= -1.0  # ang_z (yaw)

    # --- critic-only foot terms [36:48] (foot order L, R) ---
    if obs.shape[0] > 36:
        # foot_height [36:38], foot_air_time [38:40], foot_contact [40:42]
        for s in (36, 38, 40):
            obs[s:s + 2] = obs[s:s + 2][::-1]  # swap L<->R (no sign flip)
        # foot_contact_forces [42:48]: 3 per foot (fx,fy,fz). swap feet, flip y.
        fl = obs[42:45].copy()
        fr = obs[45:48].copy()
        obs[42:45] = fr
        obs[45:48] = fl
        obs[43] *= -1.0   # fy of L (now from R)
        obs[46] *= -1.0   # fy of R (now from L)
    return obs


def mirror_action(action: np.ndarray) -> np.ndarray:
    """Mirror an 8-DOF action over the Y-axis (L<->R, sign inverted)."""
    action = action.copy()
    j = action[0:8]
    swapped = np.concatenate([j[_R], j[_L]])
    action[0:8] = -swapped
    return action


def mirror_obs_dict(obs_dict: dict) -> dict:
    """Mirror an observation dict {actor: tensor/array, critic: ...}."""
    out = {}
    for k, v in obs_dict.items():
        arr = np.asarray(v)
        if k == "actor":
            out[k] = mirror_observation(arr)
        else:  # critic (48) or others
            out[k] = mirror_observation(arr) if arr.shape[-1] in (36, 48) else arr
    return out
