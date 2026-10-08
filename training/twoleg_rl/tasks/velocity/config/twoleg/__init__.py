"""Register TwoLeg velocity tasks with the mjlab task registry."""

from mjlab.rl import MjlabOnPolicyRunner as VelocityOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
    unitree_twoleg_flat_env_cfg,
    unitree_twoleg_rough_env_cfg,
)
from .rl_cfg import unitree_twoleg_ppo_runner_cfg

register_mjlab_task(
    task_id="TwoLeg-Velocity-Rough",
    env_cfg=unitree_twoleg_rough_env_cfg(),
    play_env_cfg=unitree_twoleg_rough_env_cfg(play=True),
    rl_cfg=unitree_twoleg_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="TwoLeg-Velocity-Flat",
    env_cfg=unitree_twoleg_flat_env_cfg(),
    play_env_cfg=unitree_twoleg_flat_env_cfg(play=True),
    rl_cfg=unitree_twoleg_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
