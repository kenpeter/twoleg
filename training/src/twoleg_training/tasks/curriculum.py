"""Curricula for the TwoLeg velocity task.

Step-staged parameter curricula, following the same shape as
``mjlab_microduck.tasks.mdp.reward_weight``: mutate the live manager term cfg,
never ``env.cfg``, which managers deep-copy at init.
"""

import torch

# Steps are env steps: iteration * num_steps_per_env (24).
STEPS_PER_ITER = 24


def iter_to_step(iteration: int) -> int:
    return iteration * STEPS_PER_ITER


def air_time_window(
    env,
    env_ids: torch.Tensor,
    reward_name: str = "air_time",
    stage_list: list[dict] | None = None,
) -> torch.Tensor:
    """Ramp the feet_air_time reward window wider as the gait firms up.

    microduck_rl's window is 0.125-0.30 s. At TwoLeg's 0.02 s control step that
    needs 6.25 consecutive steps of flight, and TwoLeg's feet manage about 3.7,
    so microduck's window pays nothing here and the policy never learns to step.
    The first stage is reachable on iteration 0; the last stage is microduck's
    exact window, reached inside microduck's own gait budget.
    """
    del env_ids
    stages = stage_list if stage_list is not None else AIR_TIME_STAGES
    # H4: the both-feet term was built with a fixed 0.04-0.10 s window while
    # this curriculum widened air_time to 0.125-0.30, and the feet now average
    # 0.13 s aloft. Move both windows together so a real slow swing can score.
    term_names = (reward_name, "both_feet_air_time")
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in stages:
        if env.common_step_counter > stage["step"]:
            for name in term_names:
                cfg = env.reward_manager.get_term_cfg(name)
                cfg.params["threshold_min"] = stage["threshold_min"]
                cfg.params["threshold_max"] = stage["threshold_max"]
    return torch.tensor([term_cfg.params["threshold_min"]])


AIR_TIME_STAGES = [
    {"step": iter_to_step(0), "threshold_min": 0.04, "threshold_max": 0.10},
    {"step": iter_to_step(250), "threshold_min": 0.06, "threshold_max": 0.15},
    {"step": iter_to_step(500), "threshold_min": 0.08, "threshold_max": 0.20},
    {"step": iter_to_step(1000), "threshold_min": 0.10, "threshold_max": 0.25},
    {"step": iter_to_step(1500), "threshold_min": 0.125, "threshold_max": 0.30},
]


def walk_command_ramp(
    env,
    env_ids: torch.Tensor,
    command_name: str = "twist",
    stage_list: list | None = None,
) -> torch.Tensor:
    """Walking curriculum: ramp the commanded forward speed from a tiny value
    up to the full slow-walk range as the gait firms up.

    TwoLeg collapses onto one leg at the full 0.0-0.3 m/s envelope (left_duty
    0.0). Asking for full-speed walking on iteration 0 gives the policy no easy
    intermediate: it falls. Start near-zero (balance + weight shift), then widen
    the forward command so stepping is required only once the stance is stable.
    Mirrors air_time_window but on the command, not the reward window.
    """
    del env_ids
    stages = stage_list if stage_list is not None else WALK_CMD_STAGES
    command_term = env.command_manager.get_term(command_name)
    cfg = command_term.cfg
    for stage in stages:  # noqa: SIM110
        if env.common_step_counter > stage["step"]:
            cfg.ranges.lin_vel_y = (stage["y_min"], stage["y_max"])
    return torch.tensor([cfg.ranges.lin_vel_y[1]])


WALK_CMD_STAGES = [
    # step 0: basically stand + shift weight -- tiny forward bias only
    {"step": iter_to_step(0), "y_min": 0.0, "y_max": 0.05},
    {"step": iter_to_step(250), "y_min": 0.0, "y_max": 0.10},
    {"step": iter_to_step(500), "y_min": 0.0, "y_max": 0.15},
    {"step": iter_to_step(1000), "y_min": 0.0, "y_max": 0.20},
    {"step": iter_to_step(1500), "y_min": 0.0, "y_max": 0.30},  # full range
]
