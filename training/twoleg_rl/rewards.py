"""TwoLeg-specific reward terms not provided by the base mjlab/unitree recipe.

These target gait-quality failure modes our rigorous eval surfaced (e.g. knees
not bending -> stiff-legged shuffle). Added only when a measured failure has no
existing reward term, per the standing rule against blind weight rotation.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def joint_pos_reg(
    env: "ManagerBasedRlEnv",
    asset_cfg: "SceneEntityCfg",  # noqa: F821
    default_joint_pos: dict[str, float],
    sigma: float = 0.25,
) -> torch.Tensor:
    """Penalize deviation of each joint from a stable STANDING target pose.

    The iw biped cannot stand passively (zero torque lets it topple within ~26
    steps, and random torque topples it in ~1 step -> PPO only sees 1-step
    episodes and never learns gait). Pulling joints toward a compliant bent-knee
    stance gives the policy a stable attractor so early random actions are
    corrected, the robot survives, and balance then walking can be learned.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [N, J]
    # Build target tensor matching joint_ids order.
    names = [asset.joint_names[i] for i in asset_cfg.joint_ids]
    target = torch.tensor(
        [default_joint_pos.get(n.replace("_test", ""), 0.0) for n in names],
        dtype=joint_pos.dtype, device=joint_pos.device,
    ).unsqueeze(0)
    err = joint_pos - target
    return torch.exp(-torch.sum(err * err, dim=1) / sigma**2)


def both_feet_air(
    env: "ManagerBasedRlEnv",
    sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize both feet being off the ground at once (kills hopping/bouncing).

    A real alternating walk keeps at least one foot planted; a hop/spring has
    both airborne. Returns 1.0 when both feet are air, 0.0 otherwise, gated off
    at rest.
    """
    sensor = env.scene[sensor_name]
    found = sensor.data.found[:].float()  # [N, num_feet]
    both_air = (found.sum(dim=-1) < 0.5).float()  # [N]
    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.norm(cmd[:, :3], dim=-1)
    gate = (cmd_mag > command_threshold).float()
    return both_air * gate


def knee_flexion(
    env: "ManagerBasedRlEnv",
    asset_cfg,
    command_name: str = "twist",
    command_threshold: float = 0.1,
    target: float = 0.6,
    weight_low: float = 0.15,
    weight_high: float = 1.3,
) -> torch.Tensor:
    """Reward both knees sitting in a bent (flexed) range while moving.

    A smooth window around ``target`` (default 0.6 rad ~ 34 deg) encourages the
    legs to articulate instead of locking straight. Below ``weight_low`` or above
    ``weight_high`` the reward tapers to zero so it never fights standing.

    Args:
        env: The environment instance.
        asset_cfg: Entity/joint selector resolving to the 6 duck joints
            (hip, knee, ankle per leg) in order.
        command_name: Velocity command term name (gates reward at rest).
        command_threshold: Below this command magnitude the reward is suppressed.
        target: Desired average knee-flexion angle (rad).
        weight_low: Lower bend bound (rad) where reward starts.
        weight_high: Upper bend bound (rad) where reward ends.

    Returns:
        Per-env reward tensor in [0, 1].
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos  # [N, J]
    # asset_cfg.joint_ids is populated from joint_names at construction.
    knee = joint_pos[:, asset_cfg.joint_ids].abs().mean(dim=-1)  # [N]

    # Smooth window: 1 inside [weight_low, weight_high], tapering outside.
    lo = torch.sigmoid((knee - weight_low) * 20.0)
    hi = torch.sigmoid((weight_high - knee) * 20.0)
    window = lo * hi

    # Distance-from-target inside the window.
    shape = torch.exp(-((knee - target) ** 2) / (2 * 0.25 ** 2))

    reward = window * shape

    # Gate at rest.
    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.norm(cmd[:, :3], dim=-1)
    gate = (cmd_mag > command_threshold).float()
    return reward * gate


def vertical_velocity_penalty(
    env: "ManagerBasedRlEnv",
    command_name: str = "twist",
    command_threshold: float = 0.1,
    scale: float = 2.0,
) -> torch.Tensor:
    """Penalize vertical COM velocity (kills bouncing/launch momentum).

    A walking biped keeps its body roughly level; hopping launches it upward.
    With no term opposing vertical motion, the light duck springs. This directly
    penalizes |vz| of the root, so a spring costs reward every step it's airborne.
    """
    vz = env.scene["robot"].data.root_link_lin_vel_w[:, 2].abs()  # [N]
    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.norm(cmd[:, :3], dim=-1)
    gate = (cmd_mag > command_threshold).float()
    return (scale * vz) * gate


def com_height_cap(
    env: "ManagerBasedRlEnv",
    command_name: str = "twist",
    command_threshold: float = 0.1,
    cap: float = 0.12,
    scale: float = 10.0,
) -> torch.Tensor:
    """Hard structural anti-hop: penalize the root rising above ``cap``.

    The light 1.26 kg duck launches to ~0.96 m when bouncing (measured 87%
    flight at model_999) while its standing root height is only 0.044 m. Any
    reward term that only nudges |vz| loses to a single big spring. Capping the
    root height physically makes launching unrewarding: above ``cap`` the
    penalty grows quadratically, so the optimum is to stay grounded and step.

    Args:
        env: The environment instance.
        command_name: Velocity command term name (gates at rest).
        command_threshold: Below this command magnitude the term is suppressed.
        cap: Maximum allowable root height (m). Above this, penalty applies.
        scale: Quadratic penalty gain above the cap.

    Returns:
        Per-env penalty tensor in [0, +inf) (reward weight is negative).
    """
    root_z = env.scene["robot"].data.root_link_pos_w[:, 2]  # [N]
    excess = torch.clamp(root_z - cap, min=0.0)
    penalty = scale * excess ** 2
    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.norm(cmd[:, :3], dim=-1)
    gate = (cmd_mag > command_threshold).float()
    return penalty * gate


def contact_continuity(
    env: "ManagerBasedRlEnv",
    sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward keeping at least one foot on the ground each step.

    Directly opposes flight: 1.0 when >=1 foot grounded, 0.0 when both airborne.
    This is the structural anti-hop term; `both_feet_air` only penalizes the air
    phase, while this rewards the grounded phase so the policy is pulled toward
    stance rather than merely pushed away from hops.
    """
    found = env.scene[sensor_name].data.found[:].float()  # [N, num_feet]
    grounded = (found.sum(dim=-1) >= 0.5).float()  # [N], >=1 foot down
    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.norm(cmd[:, :3], dim=-1)
    gate = (cmd_mag > command_threshold).float()
    return grounded * gate


# ---------------------------------------------------------------------------
# Ported from BipedRobot (MIT, (c) 2025 Pablo Gomez Martinez,
# /home/kenpeter/work/BipedRobot/envs/isaaclab/rewards/rewards.py).
# Adapted from single-env numpy+torch helpers to mjlab ManagerBasedRlEnv
# vectorized signature. Math is unchanged; only data sources differ
# (site_pos_w / contact sensor / command manager).
# NOT ported (need per-step touchdown buffers or missing DOFs):
# step_length, knee_bend_on_touchdown (stateful), joint_deviation_hip /
# joint_deviation_ankle_roll (need 12-DOF hip/ankle axes this 6-DOF duck
# lacks), track_joint_pos/vel_exp (need Isaac Lab + retargeted FBX .npz).
# ---------------------------------------------------------------------------


def biped_torso_centering(
    env: "ManagerBasedRlEnv",
    asset_cfg,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
    command_threshold: float = 0.1,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Keep the torso centered over the feet midpoint.

    BipedRobot `torso_centering_reward`: exp(-sigma * |base_xy - feet_mid_xy|^2).
    Order-invariant (midpoint), so foot-sensor ordering does not matter.
    """
    del sensor_name  # midpoint needs positions only, not contact state
    asset = env.scene[asset_cfg.name]
    feet_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [N, F, 3]
    feet_mid = feet_pos[:, :, :2].mean(dim=1)  # [N, 2]
    base_xy = env.scene["robot"].data.root_link_pos_w[:, :2]  # [N, 2]
    dist_sq = torch.sum(torch.square(base_xy - feet_mid), dim=1)
    reward = torch.exp(-sigma * dist_sq)
    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.norm(cmd[:, :3], dim=-1)
    gate = (cmd_mag > command_threshold).float()
    return reward * gate


def biped_swing_height(
    env: "ManagerBasedRlEnv",
    asset_cfg,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
    command_threshold: float = 0.1,
    min_height: float = 0.05,
    max_height: float = 0.15,
) -> torch.Tensor:
    """Reward swing-foot clearance inside a height window.

    BipedRobot `swing_foot_height`: clamp(h - min, 0, max - min) summed over
    airborne feet. Complements (not replaces) the existing `foot_clearance`
    term, which weights clearance by foot speed.
    """
    asset = env.scene[asset_cfg.name]
    foot_h = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [N, F]
    contact = env.scene[sensor_name].data.found[:].float()  # [N, F]
    swing = 1.0 - contact
    window = torch.clamp(foot_h - min_height, min=0.0)
    window = torch.clamp(window, max=max_height - min_height)
    reward = torch.sum(window * swing, dim=1)
    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.norm(cmd[:, :3], dim=-1)
    gate = (cmd_mag > command_threshold).float()
    return reward * gate
