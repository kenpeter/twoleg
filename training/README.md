# TwoLeg training

RL training for the TwoLeg biped, built on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo Warp + rsl_rl). This is a small, self-contained first pass modelled on
[`microduck_rl`](../../microduck_rl): one velocity (walking) task, flat ground,
no sim2real domain-randomization stack yet.

## Task

`Mjlab-Velocity-Flat-TwoLeg` — track a slow twist command:

| command | range |
|---|---|
| forward velocity `lin_vel_x` | `0.0 … 0.3` m/s |
| lateral `lin_vel_y` | `-0.1 … 0.1` m/s |
| yaw rate `ang_vel_z` | `-0.4 … 0.4` rad/s |

Rewards: linear/angular velocity tracking, upright torso, joint posture
(standing vs walking), joint-limit and action-rate penalties. All 15 joints are
actuated by the XML `<position>` servos.

## Robot model

`assets/twoleg.xml` is a copy of `../twoleg_mjcf/robot_twoleg.xml` with an IMU
site and the three built-in sensors mjlab expects (`imu_lin_vel`,
`imu_ang_vel`, `root_angmom`). Body/geom/joint layout is otherwise unchanged, and
meshes are read from `../twoleg_mjcf/assets`.

## Quickstart

Requires a CUDA GPU (mjlab runs through MuJoCo Warp) and [uv](https://docs.astral.sh/uv/).

```bash
cd training
uv sync

# train
uv run train Mjlab-Velocity-Flat-TwoLeg --env.scene.num-envs 4096

# watch a checkpoint
uv run play Mjlab-Velocity-Flat-TwoLeg --checkpoint <path/to/model.pt>
```

Logs go to `logs/rsl_rl/velocity/<timestamp>/` (TensorBoard). Scale
`--env.scene.num-envs` and `--agent.max-iterations` to taste.

## Layout

```
training/
├── assets/twoleg.xml                     # robot + IMU sensors
├── src/twoleg_training/
│   ├── robot/twoleg_constants.py         # EntityCfg, HOME pose, XML actuators
│   └── tasks/
│       ├── __init__.py                   # task registration (mjlab.tasks entry point)
│       └── twoleg_velocity_env_cfg.py    # env cfg + PPO config
└── pyproject.toml
```
