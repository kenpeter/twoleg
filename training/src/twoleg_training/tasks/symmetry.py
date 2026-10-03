"""Left-right mirror tables for the TwoLeg velocity task.

The mechanism is symmetric about the YZ plane (left-right is X in this model).
The mirrored policy sees the mirror image, which forces both legs to develop
equally instead of letting one leg do all the work.

Signs are measured, not assumed. The leg signs come from the reference
(``leg_move_all`` RM: waist -1, ankle +1, knee -1, hip -1, measured to
0.000 mm). The arm signs come from ``chest_step6_move._targets``: abduct keeps
it negates because theirs are parallel. The arm-spread abduct hinges were
removed (no real servos); 15 actuated joints remain.

Two index orders are in play and must not be mixed. ``joint_pos`` and
``joint_vel`` in the observations follow the MJCF hinge order, while
``actions`` and the actuator list follow the actuator (JOINT_NAMES) order.
Each has its own permutation below.
"""

from dataclasses import dataclass

import torch
from tensordict import TensorDict

from mjlab.rl import RslRlPpoAlgorithmCfg


@dataclass
class PpoWithSymmetryCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm config extended with an optional symmetry_cfg field."""

    symmetry_cfg: dict | None = None


SYMMETRY_CFG = {
    "use_data_augmentation": False,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 0.5,
    "data_augmentation_func": "twoleg_training.tasks.symmetry.twoleg_vel_symmetry",
}

# MJCF hinge order: head, shoulder, elbow, wrist, shoulder_R, elbow_R, wrist_R,
# L_waist, L_ankle, L_knee, L_hip, R_waist, R_ankle, R_knee, R_hip.
# Pairings: shoulder/elbow/wrist opposite; leg waist/hip/knee opposite;
# ankle same sign.
_OBS_JOINT_PERM: list[int] = [0, 4, 5, 6, 1, 2, 3, 11, 12, 13, 14, 7, 8, 9, 10]
_OBS_JOINT_SIGN: list[float] = [
    1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1,
]

# Actuator (JOINT_NAMES) order: head, shoulder, elbow, wrist, shoulder_R,
# elbow_R, wrist_R, L_waist, L_ankle, L_knee, L_hip, R_waist, R_ankle,
# R_knee, R_hip.
_ACT_PERM: list[int] = [0, 4, 5, 6, 1, 2, 3, 11, 12, 13, 14, 7, 8, 9, 10]
_ACT_SIGN: list[float] = [
    1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1,
]

# 57-dim actor obs. Left-right is X, forward is Y, up is Z.
# base_lin_vel: negate lateral x.  base_ang_vel is a pseudovector: it keeps x
# and negates y and z.  Gravity is a plain vector.  The twist command is
# (lateral, forward, yaw), so it negates x and z.
_OBS_PERM: list[int] = (
    [0, 1, 2]                                   # base_lin_vel
    + [3, 4, 5]                                 # base_ang_vel
    + [6, 7, 8]                                 # projected_gravity
    + [9 + j for j in _OBS_JOINT_PERM]          # joint_pos
    + [24 + j for j in _OBS_JOINT_PERM]         # joint_vel
    + [39 + j for j in _ACT_PERM]                   # actions
    + [54, 55, 56]                              # command
)

_OBS_SIGN: list[float] = (
    [-1.0, 1.0, 1.0]                            # base_lin_vel
    + [1.0, -1.0, -1.0]                         # base_ang_vel
    + [-1.0, 1.0, 1.0]                          # projected_gravity
    + _OBS_JOINT_SIGN                            # joint_pos
    + _OBS_JOINT_SIGN                            # joint_vel
    + _ACT_SIGN                                 # actions
    + [-1.0, 1.0, -1.0]                         # command
)

_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_tensors(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if device not in _cache:
        obs_perm = torch.tensor(_OBS_PERM, dtype=torch.long, device=device)
        obs_sign = torch.tensor(_OBS_SIGN, dtype=torch.float32, device=device)
        act_perm = torch.tensor(_ACT_PERM, dtype=torch.long, device=device)
        act_sign = torch.tensor(_ACT_SIGN, dtype=torch.float32, device=device)
        _cache[device] = (obs_perm, obs_sign, act_perm, act_sign)
    return _cache[device]


def twoleg_vel_symmetry(
    env,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Bilateral symmetry mirror for the TwoLeg vel env.

    Returns [original, mirrored] concatenated along the batch dimension.
    Compatible with the rsl_rl PPO ``symmetry_cfg`` interface.
    """
    aug_obs: TensorDict | None = None
    aug_actions: torch.Tensor | None = None

    if obs is not None:
        actor_orig: torch.Tensor = obs["actor"]
        obs_perm, obs_sign, _, _ = _get_tensors(actor_orig.device)
        actor_sym = actor_orig[:, obs_perm] * obs_sign
        critic_orig: torch.Tensor = obs["critic"]
        critic_repeated = torch.cat([critic_orig, critic_orig], dim=0)
        aug_obs = TensorDict(
            {
                "actor": torch.cat([actor_orig, actor_sym], dim=0),
                "critic": critic_repeated,
            },
            batch_size=[actor_orig.shape[0] * 2],
            device=actor_orig.device,
        )

    if actions is not None:
        _, _, act_perm, act_sign = _get_tensors(actions.device)
        actions_sym = actions[:, act_perm] * act_sign
        aug_actions = torch.cat([actions, actions_sym], dim=0)

    return aug_obs, aug_actions


def self_check(device: str = "cpu") -> None:
    """Mirroring twice must return the original. A wrong table trains on lies."""
    dev = torch.device(device)
    obs_perm, obs_sign, act_perm, act_sign = _get_tensors(dev)
    assert len(_OBS_PERM) == 63 == len(_OBS_SIGN), (len(_OBS_PERM), len(_OBS_SIGN))
    assert len(_ACT_PERM) == 17 == len(_ACT_SIGN), (len(_ACT_PERM), len(_ACT_SIGN))
    x = torch.arange(63, dtype=torch.float32, device=dev)
    back = (x[obs_perm] * obs_sign)[obs_perm] * obs_sign
    assert torch.equal(back, x), "obs tables are not an involution"
    y = torch.arange(17, dtype=torch.float32, device=dev)
    back_a = (y[act_perm] * act_sign)[act_perm] * act_sign
    assert torch.equal(back_a, y), "action tables are not an involution"
    print("symmetry tables: mirror-twice returns the input (63D obs, 17D actions)")


if __name__ == "__main__":
    self_check()
