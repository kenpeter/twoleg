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
