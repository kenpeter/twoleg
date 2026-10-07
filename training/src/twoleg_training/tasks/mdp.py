"""Reward terms specific to the TwoLeg gait.

``feet_air_time`` in mjlab pays for any foot in the air, so a policy can
collect the whole term by hopping on one leg and never loading the other.
That is what happened here: the right foot carried 39 N while the left
carried 0, for the entire run. This module pays only when both feet swing.
"""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor


def duty_balance(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.04,
    command_name: str | None = None,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Force BOTH feet to lift in turn; break the one-leg freeze.

    H5: the policy collapsed to a one-leg standstill (right foot glued down,
    left foot never lifts, contact_alt ~3.4). air_time/both_feet_air_time only
    pay swing *in a narrow window*, so a dead foot that never lifts scores 0 but
    is never penalized. This term rewards EACH foot's own air_time above a low
    floor independently, so the dead foot MUST lift to earn anything. Summed
    across feet, a true two-legged gait (both feet airborne periodically) earns
    ~2x a one-leg freeze (only the live foot scores). Gated on command so it
    only applies when told to move.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None
    found = sensor.data.found
    assert found is not None
    lifted = (current_air_time > threshold_min).float()  # [N, legs]
    reward = lifted.sum(dim=1)  # each lifted foot contributes 1.0
    env.extras["log"]["Metrics/duty_balance"] = reward.mean()
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
            reward = reward * (total > command_threshold).float()
    return reward


def no_fly(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Anti-idle / anti-double-stance penalty (ported from legged_gym Cassie).

    H6: the loop found a SYMMETRIC SQUAT-FREEZE -- both feet planted
    (duty ~0.39 each), no stance transfer (switch_hz 0.0), double_support 0.0,
    deep knee bend (1.32 rad), tiny forward speed (0.058). duty_balance only
    rewards lifting, so standing still scores 0 but is never penalized and is a
    stable local optimum. Cassie's no_fly term punishes BOTH feet being airborne;
    the dual failure here is BOTH feet being ON THE GROUND with no stepping. So
    we penalize the idle double-stance: reward is high when exactly ONE foot is
    in contact (true single support) and low when both are down (freeze) or both
    up (hop). This forces the policy off the planted squat and into alternation.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    left_down = (found[:, 0] > 0).float()
    right_down = (found[:, 1] > 0).float()
    # single support = exactly one foot down -> 1.0 ; double stance or double
    # flight -> 0.0 (penalized via being the absence of reward).
    single_support = (left_down + right_down == 1.0).float()
    reward = single_support
    env.extras["log"]["Metrics/no_fly_single_support"] = reward.mean()
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
            reward = reward * (total > command_threshold).float()
    return reward


def feet_moving(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.04,
    command_name: str | None = None,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Reward EITHER foot being airborne (any lift), gated on command.

    Companion to no_fly: together they push the policy into alternating single
    support (no_fly rewards exactly-one-down, feet_moving rewards any-up). A
    planted squat earns neither; a real gait earns both on alternating phases.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None
    lifted = (current_air_time > threshold_min).any(dim=1).float()
    reward = lifted
    env.extras["log"]["Metrics/feet_moving"] = reward.mean()
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
            reward = reward * (total > command_threshold).float()
    return reward


def legs_energy(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    weight_scale: float = 1.0,
) -> torch.Tensor:
    """Mechanical power penalty (ported from robust_robot_walker, ICRA 2025).

    H7: penalize torque * joint_velocity (mechanical power) to force EFFICIENT,
    non-flailing motion. robust_robot_walker uses weight -1e-6 to -2e-5
    (torque^2 * vel^2 is large, so the scale must be tiny). Discourages the
    high-torque thrash / hop our early fresh runs show; an efficient alternating
    gait costs less. Uses qfrc_actuator (PD actuator force in joint space) x
    joint_vel, summed over leg DOFs.
    """
    asset = env.scene[asset_cfg.name]
    torque = asset.data.qfrc_actuator  # [B, nv]
    jvel = asset.data.joint_vel  # [B, nv]
    # restrict to the asset's joint indices (legs only if asset_cfg selects them)
    idx = asset_cfg.joint_ids if hasattr(asset_cfg, "joint_ids") else slice(None)
    power = (torque[:, idx] * jvel[:, idx]) ** 2
    reward = -weight_scale * power.sum(dim=1)
    env.extras["log"]["Metrics/legs_energy"] = reward.mean()
    return reward


def reward_weight(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    weight_stages: list[dict],
) -> torch.Tensor:
    """Step-staged reward weight curriculum (ported from microduck_rl).

    mjlab 1.3.0 dropped the built-in mdp.reward_weight helper, so microduck
    provides its own. weight_stages is a list of {"step": int, "weight": float}
    dicts; the weight of the latest stage whose step has elapsed is applied.
    Mutates the live RewardManager term cfg (not env.cfg, which is a deepcopy at
    manager init).
    """
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in weight_stages:
        if env.common_step_counter > stage["step"]:
            term_cfg.weight = stage["weight"]
    return torch.tensor([term_cfg.weight])


def standing_envs_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    standing_stages: list[dict],
) -> torch.Tensor:
    """Ramp the fraction of standing environments over training (microduck_rl).

    Microduck starts at rel_standing_envs=0.02 and ramps to 0.25 by iter ~2000,
    only AFTER a gait exists -- "walk first, stand later". Mutates the live
    command term cfg.
    """
    del env_ids
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from typing import cast

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"
    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
    for stage in standing_stages:
        if env.common_step_counter > stage["step"]:
            cfg.rel_standing_envs = stage["rel_standing_envs"]
    return torch.tensor([cfg.rel_standing_envs])


def feet_gait(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """Phase-locked alternating-gait reward (ported from unitree_rl_mjlab).

    H9: this is the PROVEN gait shaper that replaces air-time rewards. It pays
    when each foot's ACTUAL contact matches the EXPECTED contact phase: left
    offset 0.0, right offset 0.5 (half-period apart), stance = leg_phase <
    threshold. A hop/flail (both feet airborne together, or one foot never
    planting) earns ~0 because actual contact desynchronizes from the phase.
    This structurally prevents the hop attractor that air-time rewards caused.
    Needs only ContactSensor.current_contact_time + env.episode_length_buf/
    step_dt, all present in our mjlab.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward = reward * scale
    env.extras["log"]["Metrics/feet_gait"] = reward.mean()
    return reward


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation from default when command ~0 (unitree_rl_mjlab).

    H9: prevents idle thrash / camping at zero command. Only active when the
    command norm is below threshold, so it never taxes walking. Forces the
    policy to hold a calm default pose when told to stand.
    """
    asset = env.scene[asset_cfg.name]
    jids = asset_cfg.joint_ids if hasattr(asset_cfg, "joint_ids") and asset_cfg.joint_ids else slice(None)
    diff_angle = asset.data.joint_pos[:, jids] - asset.data.default_joint_pos[:, jids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward = reward * scale
    env.extras["log"]["Metrics/stand_still"] = reward.mean()
    return reward


def body_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward flat base orientation / upright torso (unitree_rl_mjlab).

    H9: complement to our `upright` term. Uses the body's projected gravity
    (z = -1 when upright) to penalize tilt, keeping the torso upright during
    gait. Our mjlab exposes projected_gravity_b on the articulation data.
    """
    asset = env.scene[asset_cfg.name]
    proj = asset.data.projected_gravity_b[:, 2]  # -1 upright, +1 inverted
    reward = -torch.square(proj - (-1.0))
    env.extras["log"]["Metrics/body_orientation_l2"] = reward.mean()
    return reward

