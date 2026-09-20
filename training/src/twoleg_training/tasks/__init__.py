from mjlab.tasks.registry import register_mjlab_task

from .twoleg_velocity_env_cfg import TwoLegRlCfg, make_twoleg_velocity_env_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-TwoLeg",
    env_cfg=make_twoleg_velocity_env_cfg(),
    play_env_cfg=make_twoleg_velocity_env_cfg(play=True),
    rl_cfg=TwoLegRlCfg,
    runner_cls=None,
)
