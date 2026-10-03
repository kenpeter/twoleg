# TwoLeg training

RL training for the TwoLeg biped, built on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo Warp + rsl_rl). This is a small, self-contained first pass modelled on
[`microduck_rl`](../../microduck_rl): one velocity (walking) task, flat ground,
no sim2real domain-randomization stack yet.

## Task

`Mjlab-Velocity-Flat-TwoLeg` — track a slow twist command:

| command | range |
|---|---|
| forward velocity `lin_vel_y` | `0.0 … 0.3` m/s |
| lateral `lin_vel_x` | `-0.1 … 0.1` m/s |
| yaw rate `ang_vel_z` | `-0.4 … 0.4` rad/s |

Forward is `lin_vel_y`: this model faces `+y`, its left-right axis is `+x`
(feet at `x = +-0.034` with equal `y`, foot box 12.4 cm along `y`), and mjlab
compares the command to body-frame velocity channel by channel.

Rewards: linear/angular velocity tracking, upright torso, joint posture
(standing vs walking), joint-limit and action-rate penalties. All 15 joints are
actuated by the XML `<position>` servos.

## Robot model

`robot_item/chest_both_arms_legs.xml` plus the `chest_move` rig is the source
of truth for the mechanism. It is a world-pinned puppet, not a free body, so
`scripts/generate_twoleg_model.py` rebuilds it into one connected tree and
writes both consumers:

| output | used by | meshdir |
|---|---|---|
| `../twoleg_mjcf/robot_twoleg.xml` | viewing | `../robot_item` |
| `assets/twoleg.xml` | mjlab | `../../robot_item` |

The generator re-parents the limb chains under the torso, recomputes every
body's `pos` and `quat` from its world pose, drops the duplicate waist hinges,
and carries the geoms over untouched so mass and inertia follow. Regenerate
after any change to the reference:

```bash
uv run python training/scripts/generate_twoleg_model.py
```

Never hand-edit either output. Verify the result against the reference:

```bash
uv run python training/scripts/check_model_vs_reference.py
```

It should report 121 bodies, 262 geoms, a mass ratio of 1.000, and 0.00 mm on
every shared joint anchor.

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
├── assets/twoleg.xml                     # GENERATED, mjlab-ready model
├── scripts/generate_twoleg_model.py      # rebuilds the model from the reference
├── scripts/check_model_vs_reference.py   # verifies it against the reference
├── src/twoleg_training/
│   ├── robot/twoleg_constants.py         # EntityCfg, HOME pose, XML actuators
│   └── tasks/
│       ├── __init__.py                   # task registration (mjlab.tasks entry point)
│       ├── curriculum.py                 # air-time window ramp
│       └── twoleg_velocity_env_cfg.py    # env cfg + PPO config
└── pyproject.toml
```
