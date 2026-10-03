"""Reward terms specific to the TwoLeg gait.

``feet_air_time`` in mjlab pays for any foot in the air, so a policy can
collect the whole term by hopping on one leg and never loading the other.
That is what happened here: the right foot carried 39 N while the left
carried 0, for the entire run. This module pays only when both feet swing.
"""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor


def both_feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.04,
    threshold_max: float = 0.30,
    command_name: str | None = None,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Reward a foot's swing only while the other foot is in stance.

    H3: the old minimum-across-feet form only pays when both feet are
    simultaneously in-window, which is the double-flight phase of a hop, so
    raising its weight bought hopping (both-up 23 pct), not walking. Gating
    each foot's swing on the other foot's contact pays single support: an
    alternating gait earns on every phase, a hop earns nothing, and a skim
    never reaches the window. Takes the max across feet.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None
    found = sensor.data.found
    assert found is not None
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    other_down = torch.stack([(found[:, 1] > 0), (found[:, 0] > 0)], dim=1)
    reward = torch.max((in_range & other_down).float(), dim=1).values
    env.extras["log"]["Metrics/min_air_time"] = reward.mean()
    asymmetry = torch.abs(current_air_time[:, 0] - current_air_time[:, 1])
    env.extras["log"]["Metrics/foot_air_asymmetry"] = asymmetry.mean()
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
            reward = reward * (total > command_threshold).float()
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
