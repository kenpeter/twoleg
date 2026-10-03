# TwoLeg training abstraction

## Scope and how to read this file

This file answers three questions about the training runs under
`training/logs/rsl_rl/velocity/`:

1. Did the twoleg runs follow the `microduck_rl` recipe?
2. How much training did `microduck_rl` give its duck before it walked?
3. Why does the twoleg policy stand still?

Every number below was read from an artifact on disk. Each claim carries a label:

- **measured**: read directly from a log file, `params/*.yaml`, or a rendered video frame.
- **derived**: arithmetic on measured values, with the arithmetic shown.
- **inference**: my reading of the evidence. Not a measurement.

No training was run and no policy was evaluated for this file. The
`preflight_3d.py --capability train` gate reports `BLOCKED` on this host
(Python 3.14.4, no `torch`, `mujoco`, `gymnasium`, or `stable_baselines3`), so
every statement below is a reading of a past run, not a fresh rollout.

## How training works

```
robot_item/chest_both_arms_legs.xml   (reference assembly, world-pinned puppet)
        │  training/scripts/generate_twoleg_model.py
        │    re-parent limbs under torso, recompute pos/quat from world poses,
        │    drop the 2 duplicate waist hinges, add freejoint + imu + foot sites
        v
twoleg_mjcf/robot_twoleg.xml          (connected tree, 121 bodies, 17 joints)
training/assets/twoleg.xml            (GENERATED, mjlab-ready, relative meshdir)
        │  robot/twoleg_constants.py:17-22
        v
EntityCfg -> make_twoleg_velocity_env_cfg(play=False)  [tasks/twoleg_velocity_env_cfg.py:33]
  ├─ Robot: 17 XML <position> servos   [generated model]
  │          4.4818 kg, 121 bodies, 262 geoms, kp 6 arms / 40 legs
  ├─ Terrain: plane, extent 2.0 m, no generator
  ├─ Sensors: foot_height_scan, feet_ground_contact, self_collision
  ├─ Commands: twist(3) = lin_vel_x 0..0.3, lin_vel_y -0.1..0.1, ang_vel_z -0.4..0.4
  ├─ 13 reward terms, 1 curriculum (air-time window ramp)
  └─ 3 terminations: time_out, fell_over, nan_state
        │
        v
  MuJoCo Warp via mjlab 1.3.0, 4096 envs, decimation 4, timestep 0.005 -> 50 Hz
        │  rollout 24 steps per env per iteration
        v
  PPO via rsl_rl  [tasks/twoleg_velocity_env_cfg.py:137-174]
        │  actor/critic 512-256-128 ELU
        v
  logs/rsl_rl/velocity/<timestamp>/model_*.pt  -> TensorBoard
```

## The runs on disk

`training/logs/rsl_rl/velocity/` holds 8 runs. Six are smoke tests. Two carry
real training.

**measured** (2026-10-03): only `2026-10-03_08-28-43_velocity` remains
on disk; the earlier 2026-10-01/02/03 run dirs were deleted.

| run | max checkpoint | `num_envs` | `max_iterations` | wall clock |
|---|---|---|---|---|
| `2026-09-27_23-59-17_velocity` | `model_24999.pt` | 4096 | 25000 | 6 h 09 m 54 s |
| `2026-09-28_10-37-56_velocity` | `model_299.pt` | 1024 | 300 | about 2 m |
| `2026-09-27_23-48-25_velocity` | `model_149.pt` | not resolved | not resolved | under 1 m |
| `2026-09-27_23-46-05` / `-23-46-25` / `2026-09-27_23-58-35` | `model_2` / `model_9` / `model_4` | not resolved | not resolved | under 1 m |
| `2026-09-28_10-36-51` / `10-37-23` | `model_2.pt` | not resolved | not resolved | under 1 m |

Wall clock for the 25,000-iteration run is derived from the mtime of
`model_0.pt` (2026-09-27 23:59:23) to the mtime of `model_24999.pt`
(2026-09-28 06:09:17).

**derived** sample budget for the 25,000-iteration run:

```
4096 envs x 24 steps/env/iter x 25000 iters = 2,457,600,000 env-steps
25000 iters x 0.8833 s/iter              = 6.13 h   (logged Perf/* mean)
```

The 0.8833 s/iter is the mean of `Perf/collection_time` (0.816 s) plus
`Perf/learning_time` (0.059 s) over the last 10% of the run. The
`Perf/total_fps` of 1.12e5 agrees: `4096 x 24 / 0.816 = 1.20e5` during
collection.

The run directory also holds 105 training videos and a `git/twoleg.diff`
recording commit `8ee9bc3` as the provenance.

## What was copied from microduck_rl

`training/README.md:4-6` states the intent: "a small, self-contained first pass
modelled on `microduck_rl`". The logs show the intent was carried into the
agent configuration and stopped there.

**measured**, comparing `params/agent.yaml` in each run directory:

| setting | twoleg 25k run | microduck velocity run | same? |
|---|---|---|---|
| `class_name` | `PPO` via `OnPolicyRunner` | `PPO` via `OnPolicyRunner` | yes |
| `num_steps_per_env` | 24 | 24 | yes |
| `num_learning_epochs` | 5 | 5 | yes |
| `num_mini_batches` | 4 | 4 | yes |
| `learning_rate` | 0.001, `adaptive` | 0.001, `adaptive` | yes |
| `gamma` / `lam` | 0.99 / 0.95 | 0.99 / 0.95 | yes |
| `clip_param` | 0.2 | 0.2 | yes |
| `entropy_coef` | 0.01 | 0.01 | yes |
| `desired_kl` | 0.01 | 0.01 | yes |
| `max_grad_norm` | 1.0 | 1.0 | yes |
| `value_loss_coef` | 1.0 | 1.0 | yes |
| `seed` | 42 | 42 | yes |
| `actor.hidden_dims` | 256, 128, 64 | 512, 256, 128 | no |
| `critic.hidden_dims` | 256, 128, 64 | 512, 256, 128 | no |
| `max_iterations` | 25000 | 2500 | no |
| `save_interval` | 500 | 2500 | no |
| `logger` | `tensorboard` | `wandb` | no |
| `algorithm.symmetry_cfg` | key absent | `null` | no |

Every PPO hyperparameter is identical. The network is half as wide at every
layer and one layer shallower. The iteration budget is ten times larger.

## What was dropped

**measured**, comparing `params/env.yaml` in each run directory. The twoleg
reward block has 8 terms, the microduck one has 16.

| reward term | twoleg weight | microduck weight | status |
|---|---|---|---|
| `track_linear_velocity` | 2.0 | 2.0 | both |
| `track_angular_velocity` | 1.5 | 2.0 | both, weakened |
| `upright` | 1.0 | 2.0 | both, weakened |
| `pose` | 1.0 | 1.0 | both |
| `action_rate_l2` | -0.1 | -0.1 | both |
| `body_ang_vel` | -0.05 | -0.05 | both |
| `angular_momentum` | -0.02 | -0.02 | both |
| `dof_pos_limits` | -1.0 | -1.0 | both |
| `air_time` | absent | 3.0 | dropped |
| `foot_clearance` | absent | -2.0 | dropped |
| `foot_swing_height` | absent | -0.25 | dropped |
| `foot_slip` | absent | -0.1 | dropped |
| `self_collisions` | absent | -1.0 | dropped |
| `head_pose_tracking` | absent | 2.0 | dropped |
| `head_pose_bias` | absent | 0.0 | dropped |
| `body_pose_tracking` | absent | 0.0 | dropped |

Four of the eight dropped terms are foot-gait shaping. `air_time` at weight
3.0 is the only positive reward in the microduck velocity task that pays for
lifting a foot. None of the four survive in twoleg.

Other structural differences, **measured** from the same two files:

| | twoleg | microduck |
|---|---|---|
| `terrain_type` | `plane` | `generator`, 4 difficulty levels |
| curricula active in the log | none | 8, logged as 9 `Curriculum/*` tags |
| domain-randomization events | 6 | 13 |
| domain randomization of mass, armature, joint friction | absent | present |
| IMU misalignment (up to 6 deg, random axis) and 0 to 1 step lag | absent | present |
| terminations | 2 (`time_out`, `fell_over`) | 4, plus `nan_state` with a contact-sensor check |
| scalar tags logged | 26 | 46 |
| `nan_guard.enabled` | `false` | not present in the resolved file |
| command ranges | `lin_vel_x 0..0.3`, `ang_vel_z -0.4..0.4` | `lin_vel_x -0.4..0.4`, `ang_vel_z -1.0..1.0` |

`head_pose_tracking`, `head_pose_bias`, and `body_pose_tracking` are not
available to twoleg by construction. Its MJCF has no head-pose command block,
so dropping them is correct. The other five drops are not forced by the robot.

## How much microduck_rl trained its duck to walk

Three measurements, in increasing order of difficulty.

**The shipped walking policy took about 3,750 iterations.**
`microduck_rl/AGENTS.md:224-225` records the deployed walk and stand policies
by wandb run id and iteration: "walk = 441tzs6d@3750, stand =
69u48n8l@9750". That line exists because Hub ONNX files ship with
`run_path=None`, so the run had to be recovered by matching the last-layer
weights against the wandb checkpoints.

**The author's own budget for a gait is 4,000 to 6,000 iterations.**
`microduck_rl/AGENTS.md:231-232`: "Budgets: simple episodic tricks ~1000 iters
at 4096 envs; gaits and curriculum-heavy recovery need 4000-6000." The same
file gives the 1 to 2 hour figure that `README.md:34` repeats: "~1-2 h for a
usable gait at 4096 envs".

**The local velocity logs hold 6 runs totalling 19,994 iterations.**
Recovered by taking the highest `model_*.pt` symlink in each
`wandb/offline-run-*/files/` directory, which survives after checkpoints are
pruned. The highest single run reached 4,999 iterations at 2,048 envs.

| task | runs | iterations summed | longest run |
|---|---|---|---|
| `running` | 25 | 901,250 | 76,500 |
| `sprint` | 19 | 395,953 | 34,250 |
| `velstand` | 13 | 31,491 | 4,999 |
| `velocity` (the walking task) | 6 | 19,994 | 4,999 |
| `ground_pick` | 12 | 14,996 | 4,999 |
| all 14 tasks | 99 | 1,413,666 | 76,500 |

**derived**: `1,413,666 x 24 x 2048 = 69,484,511,232` env-steps across the
whole repo. Walking specifically: `19,994 x 24 x 2048 = 982,745,088`.

**Fast gait work went into `running`, not `velocity`.** The longest single run
is `2026-09-12_16-16-33_running-max-speed`: 15,000 iterations from checkpoint
61,500 to 76,500 in 6 h 26 m 43 s, at 4,096 envs on a flat plane, with
`max_iterations: 80000` configured.

**The only speed number in either repo is a world-displacement measurement.**
`microduck_rl/hf_upload_README.md:21-27` reports, for
`model_76500.pt`, evaluated with `scripts/eval_sprint_speed.py`:

| command | measured forward speed |
|---|---|
| 0.4 m/s | 0.11 m/s (16 resets) |
| 0.8 m/s | 1.27 m/s |
| 1.2 m/s | 1.55 m/s |
| 1.6 m/s | 1.61 m/s |
| 2.0 m/s | 1.66 m/s (p10 1.49, p90 1.85, max 1.93) |

`AGENTS.md:233-238` and `hf_upload_README.md:30-34` both insist on this
method: "Reward/curriculum metrics are NEVER reported as speed."

**measured**, the `running` task's resolved reward block, for contrast with
the `velocity` task:

| term | weight | term | weight |
|---|---|---|---|
| `forward_progress` | 5.0 | `dof_pos_limits` | -1.0 |
| `track_linear_velocity` | 2.0 | `foot_clearance` | -2.0 |
| `heading_hold` | 1.5 | `self_collisions` | -1.0 |
| `upright` | 0.75 | `foot_swing_height` | -0.25 |
| `pose` | 0.15 | `action_rate_l2` | -0.02 |
| `track_angular_velocity` | 0.5 | `foot_slip` | -0.05 |
| `body_ang_vel` | -0.01 | `air_time` | 0.0 |
| `angular_momentum` | -0.005 | | |

`forward_progress` at 5.0 outweighs `track_linear_velocity` at 2.0 by two and a
half times, and it is the only term in either repo that pays for covering
ground rather than for matching a command. `air_time` is present but weighted
0.0, so the running gait was not bought with a swing-phase bonus. It was bought
with a term that pays for displacement. Twoleg has no equivalent term at all.

## What the twoleg 25,000-iteration run produced

**measured**, mean of the last 10% of iterations unless noted.

| tag | twoleg 25k | microduck velocity 2.5k |
|---|---|---|
| `Train/mean_reward` | 80.18 (last) | 93.3 (last) |
| `Episode_Reward/track_linear_velocity` | 1.378 | 1.184 |
| `Episode_Reward/track_angular_velocity` | 1.239 | 0.454 |
| `Episode_Reward/upright` | 0.991 | 1.621 |
| `Episode_Reward/pose` | 0.879 | 0.572 |
| `Episode_Reward/air_time` | tag absent | 0.837 |
| `Episode_Termination/fell_over` | 0.027 | 0.441 |
| `Train/mean_episode_length` | 997 | 916 |
| `Metrics/twist/error_vel_xy` | 0.454 | 0.440 |
| `Metrics/twist/error_vel_yaw` | 0.402 | 1.571 |
| `Policy/mean_std` | 0.336 | 0.209 |
| `Loss/entropy` | 4.45 | -2.19 |
| `Perf/collection_time` (s) | 0.816 | 1.939 |
| `Perf/total_fps` | 1.124e5 | 2.456e4 |
| scalar tags | 26 | 46 |

Two rows carry the answer.

`Metrics/twist/error_vel_xy` is 0.454 at iteration 24,999 and was already
0.460 at iteration 100. The policy did not improve velocity tracking in
24,899 iterations.

`Train/mean_episode_length` is 997 out of a 1000-step cap, and
`Episode_Termination/fell_over` is 0.027 percent. The robot never falls and
never gets cut short. It holds one pose for the full 20-second episode.

**measured**, the last training video. `videos/train/rl-video-step-599040.mp4`
is 100 frames at 50 fps, 2.0 s, and its step number is `24960 x 24`. Frames 0,
25, 50, 75, and 99 show the humanoid standing with both feet flat, knees
together, no stepping, and no displacement against the floor grid. The same
check on `rl-video-step-17280.mp4` (iteration 720) shows the same standing
posture with a slight forward lean.

**inference**: the twoleg policy learned to stand still.

## Why it stands still

Read the reward table against the measured curve.

The policy gets 0.991 of a possible 1.0 on `upright` and 0.879 of 1.0 on
`pose`. Both are Gaussian terms that peak at perfect posture. Standing still
maximises both, and `pose` is weighted 1.0.

`track_linear_velocity` reaches 1.378 out of a possible 2.0. It is a Gaussian
on the velocity error with `std = sqrt(0.1) = 0.316` m/s, so the reward is
already above half scale while the robot sits inside roughly 0.2 m/s of the
command. The twoleg command range is `lin_vel_x 0.0 to 0.3` m/s, half the
magnitude of microduck's `-0.4 to 0.4`. Doing nothing earns most of that
term's mass without moving.

`air_time` is the only term that pays for a swing phase, and it is absent.
`foot_clearance`, `foot_swing_height`, and `foot_slip` are absent too, so
nothing opposes foot drag and nothing demands a lift. A biped with no reward
for unweighting one leg has no reason to unweight one leg. `action_rate_l2` at
-0.1 actively rewards holding still, because a still robot has a zero action
derivative.

That is the whole gap. `track_linear_velocity` is a tracking term, so it is
satisfied by a robot that sits near its command, and a robot that sits still
sits near a `0.0 to 0.3` m/s command. `microduck_rl` learned to run with
`air_time` at weight 0.0, and it still learned a gait, but only because
`forward_progress` at 5.0 was there to pay for ground covered. Neither a
swing-phase bonus nor a displacement bonus exists in the twoleg run.

**inference**: the reward function the twoleg run used has a global optimum at
standing still. Ten times microduck's iteration budget cannot leave that
optimum, because nothing in the gradient points out of it. This is a reward
design problem, not a training budget problem.

Two things make it worse than a neutral result. `upright` and
`track_angular_velocity` were both weakened relative to microduck, which
removes part of the pressure that keeps a biped stepping. The learning rate also
trends down: `Loss/learning_rate` reads 7.6e-5 at iteration 10,000, having
started at 1e-3, because the adaptive schedule hits the 0.01 `desired_kl`
target. Over iterations 10,000 to 24,999 it averages 5.11e-4, about half the
initial rate, and it reaches its 1e-5 floor at every instability step listed
below. It is not monotonic, and it recovers to 1.95e-3, so this is a weaker
signal than the reward table and should not be leaned on.

## Numerical instability in the same run

**measured**: 120 of 25,000 logged iterations, 0.48 percent, have a negative
`Train/mean_reward`. The worst is `-1.121e9` at iteration 24,417. The
correlated tags at that step are `Episode_Reward/action_rate_l2 = -1.168e8` and
`Loss/value = 1.288e9`.

`action_rate_l2` is the squared change in the action vector, weighted -0.1.
With `clip: null` and 15 actions, a finite bound of about -6.0 exists. A value
of -1.168e8 requires a non-finite or overflowing action. `Loss/value` reaching
1.13e13 in the same window confirms the value head diverged with the actor.
`Loss/learning_rate` reads 1e-5, its floor, at every one of these steps, then
recovers on the next step. `Episode_Termination/fell_over` stays between 0.04
and 0.17 percent throughout, so this is not a physical fall.

**inference**: the twoleg env had no guard to catch this. The microduck env
terminates on `nan_state`, checked against the `feet_ground_contact` sensor, and
logs `Episode_Termination/nan_state` every iteration. The twoleg env has
`nan_guard.enabled: false` and `nan_policy: disabled`. These 120 iterations
were silent.

## Gotchas for anyone reading these logs

- `Episode_Reward/<term>` is the weighted value, not the raw signal. A term at
  weight 0 logs 0 whatever the policy does. `AGENTS.md:229-230` says this.
- Total reward can rise on regularizers alone while the task never happens.
  `AGENTS.md:226-228` says this. The twoleg `mean_reward` sits at 80 to 86
  from iteration 100 onward, and none of the movement metrics move with it.
- `Metrics/twist/error_vel_xy` is the honest progress signal here, and it is
  flat. Watch it, not `mean_reward`.
- Twoleg logs no `air_time`, `foot_clearance`, `foot_slip`, or
  `foot_swing_height` tag at all. The terms were removed from the config, not
  merely weighted to zero. An absent tag and a zero tag mean different things.
- wandb checkpoint directories in `microduck_rl/wandb/offline-run-*/files/`
  are symlinks. When the run directory is pruned, the symlink dangles, but its
  target path still records the run name and the final iteration. That is the
  only surviving record of iteration counts for pruned runs.
- `Task/mean_reward` for twoleg has 120 outliers spanning -1.1e9 to -3.9e6. Any
  statistic over the full run is dominated by them. The last-10% mean of
  -4.5e5 is not a training signal, it is the arithmetic of 8 outliers landing
  in that window.

## The training robot is not the reference robot

`training/assets/twoleg.xml` is generated from `twoleg_mjcf/robot_twoleg.xml`,
which is hand-built and has no provenance link to
`robot_item/chest_both_arms_legs.xml`. Git confirms they are independent:
`robot_twoleg.xml` landed in `7931963` and `59a2a98`;
`chest_both_arms_legs.xml` landed later, in `d1591b5`.

**measured**, `training/scripts/check_model_vs_reference.py`:

| | reference | training |
|---|---|---|
| source | `chest_both_arms_legs.xml` via the `chest_move` rig | `twoleg_mjcf/robot_twoleg.xml` |
| bodies | 121 | 32 |
| joints | 19 | 16 (15 actuated plus the freejoint) |
| actuators | 19 | 15 |
| geoms | 262 | 94 |
| mass, compiled | 4.4818 kg | 2.0625 kg |
| world-pinned bodies | 32 | 1 |
| freejoints | 0 | 1 |
| shared joint names | **none** | **none** |

The joint name sets are disjoint. `waist_test`, `hip_test`, `knee_test`,
`ankle_test`, `abduct_*`, `shoulder_test`, `elbow_test`, `wrist_test` in the
reference against `hip_roll`, `hip_pitch`, `knee`, `ankle`,
`shoulder_pitch`, `shoulder_roll`, `elbow`, `head_yaw` in training.

**The reference is a puppet, not a free body.** 32 of its 121 bodies attach
directly to the world, including `L_waist_link`, `R_waist_link`,
`abduct_L_link`, `abduct_R_link`, and `head_link`. Every limb hangs from the
world rather than from the torso. Adding a freejoint to the root `u_beam`
would move only the chest plate and leave the limbs pinned in the air.

**measured**: the reference build emits no `freejoint`. Its physics mode works
because the chains are pinned to the world, which is why
`chest_both_arms_legs_move.py:46` can say `LEG_KP = 40.0` because "the legs
carry every gram of leg under gravity". The legs swing from a fixed world
anchor. Nothing falls over, and no foot ever needs to touch the ground.

**measured**: the reference's collision masks are an inter-leg setup. Left-leg
geoms are `contype=1 conaffinity=2`, right-leg geoms are `contype=2
conaffinity=1`, so the two legs collide with each other and with nothing else.
A ground plane at `contype=1 conaffinity=1` is not in either mask. Dropped into
a free-body simulation with a floor, the reference robot falls through it.

The reference is authoritative for two things: the world pose of every part,
and the measured pivot of every hinge (`leg_move_all.SIDES`, `chest_move.WAIST`,
commented as measured to 0.000 mm). It is not authoritative for articulation,
because it has none that survives a free body.

**Consequence**: no reward change can make the old training robot move like the
reference video, because they were different mechanisms with disjoint joint
sets.

**The rebuild is done.** `training/scripts/generate_twoleg_model.py` reads the
reference assembly and its rig, re-parents the limb chains under the torso,
recomputes each body's `pos` and `quat` from its world pose at the zero pose,
drops the two duplicate waist hinges, and adds the freejoint, the `imu` site,
and two foot sites at the soles. It writes both consumers in one pass, so
`sync_twoleg_xml.py` is deleted.

**measured**, `training/scripts/check_model_vs_reference.py` after the rebuild:

| | reference | training |
|---|---|---|
| bodies | 121 | 121 |
| geoms | 262 | 262 |
| mass | 4.4818 kg | 4.4818 kg (ratio 1.000) |
| world-pinned bodies | 32 | 1 |
| freejoints | 0 | 1 |
| shared joint anchors | | **0.00 mm on all 17** |

The only joint differences are the two duplicate waist hinges the reference
models twice, and the added freejoint. The training model now has the
mechanism's true 17 actuated joints: 4 per arm (`abduct`, `shoulder_test`,
`elbow_test`, `wrist_test`), 4 per leg (`L_waist_test`, `L_hip_test`,
`L_knee_test`, `L_ankle_test`), and `head`.

Two things had to be fixed that the reference never exercised, because its
puppet never fell over or touched a floor:

* **Collision masks.** The reference's leg geoms are `contype=1 conaffinity=2`
  on the left and `2/1` on the right, an inter-leg setup that leaves a ground
  plane at `1/1` in neither mask. The generator adds the ground bit, and a drop
  test now lands the robot on its feet and settles instead of falling through.
* **Contact capacity.** 26 collision geoms against the old model's two foot
  boxes overflows mjlab's default `nconmax`, so the env sets it to 200.

Cost: the faithful model is heavier to simulate. At 64 envs the old model ran
5,783 steps/s and the rebuilt one runs 878, about 6.5x slower, because it is
121 bodies and 262 geoms instead of 32 and 94.

## Result: it walks slowly

**measured**, `training/scripts/eval_walk.py` on the final checkpoint of run
`2026-09-28_23-48-21`, holding a fixed 0.30 m/s forward command for 500 control
steps (10 s) across 16 environments:

| | value |
|---|---|
| net displacement | 2.081 m in 10 s (p10 1.517, p90 2.367) |
| net speed | **0.208 m/s** |
| mean speed, direction ignored | 0.212 m/s |
| tracking of the command | 69 percent |

Mean speed and net speed agree, so the motion is forward travel rather than
oscillation. The video at `videos/train/rl-video-step-142000.mp4` shows a
stepping gait with the floor grid advancing.

**measured**, the same run's last 5 percent of iterations:

| term | value | target |
|---|---|---|
| `Episode_Reward/track_linear_velocity` | 1.689 | ≥ 1.0 |
| `Episode_Reward/air_time` | 1.361 | ≥ 0.5 |
| `Metrics/twist/error_vel_xy` | 0.220 | < 0.40 |
| `Episode_Reward/upright` | 1.913 | ≥ 1.5 |
| `Episode_Termination/fell_over` | 0.058 | < 10 |
| `Train/mean_episode_length` | 984 | > 800 |

For comparison, microduck's own velocity run at iteration 2499 reached
`error_vel_xy` 0.440 and `air_time` 0.837. The twoleg run is better on both.

Two bugs had to be fixed to get here, and both were silent:

1. **The root frame.** The chest body carries a 90 degree rotation of its own.
   Making it the free body's root left mjlab reading `projected_gravity` as
   `[9.81, 0, 0]` instead of `[0, 0, -9.81]` while the robot stood, so the
   `upright` reward and the `fell_over` termination both measured the wrong
   axis. The policy learned to lie down, and reported `upright` 0.065 and
   `air_time_mean` 9.76 s. The generator now inserts an empty world-aligned
   base body as the root and hangs the chest from it.
2. **Joint regularisation.** The reference leaves `damping`, `frictionloss` and
   `armature` unset and has massless link bodies, so a hinge could rotate
   near-zero inertia and the solver returned NaN within milliseconds of a free
   fall. The generator sets `armature=0.01`, `damping=0.05`,
   `frictionloss=0.02`.

Also deliberate: the 26 mesh collision geoms became one box per foot, because
mesh-mesh contact on the near-parallel leg brackets destabilises the solver.
The kinematics, mass and pivots are untouched.

## Verify this file

```bash
cd /home/kenpeter/work/twoleg

# reward and weight diff
python3 - <<'PY'
import re
def weights(p):
    out=[];inr=False
    for l in open(p).read().splitlines():
        if re.match(r'^rewards:\s*$',l): inr=True; continue
        if inr and re.match(r'^[a-z_]+:\s*$',l): break
        if inr:
            m=re.match(r'^  ([a-z_0-9]+):\s*$',l)
            if m: out.append([m.group(1),None]); continue
            m=re.match(r'^\s+weight:\s*(\S+)',l)
            if m and out: out[-1][1]=m.group(1)
    return out
A=weights("training/logs/rsl_rl/velocity/2026-09-27_23-59-17_velocity/params/env.yaml")
B=weights("/home/kenpeter/work/microduck_rl/logs/rsl_rl/velocity/2026-09-05_13-08-20_auto5000/params/env.yaml")
for k in sorted(set(dict(A))|set(dict(B))):
    print(f"{k:30s} twoleg={dict(A).get(k,'-'):>6s} microduck={dict(B).get(k,'-'):>6s}")
PY

# the video claim
ffmpeg -v error -i training/logs/rsl_rl/velocity/2026-09-27_23-59-17_velocity/videos/train/rl-video-step-599040.mp4 \
  -vf "select='eq(n\,0)+eq(n\,25)+eq(n\,50)+eq(n\,75)+eq(n\,99)',tile=5x1" -frames:v 1 /tmp/twoleg_last.png
```

Scalar tables come from the two tfevents files with each project's own venv,
which already has tensorboard:

```bash
/home/kenpeter/work/twoleg/training/.venv/bin/python - <<'PY'
from tensorboard.backend.event_processing import event_accumulator as ea
a=ea.EventAccumulator("training/logs/rsl_rl/velocity/2026-09-27_23-59-17_velocity/"
  "events.out.tfevents.1790517561.kenpeter-ubuntu.23723.0", size_guidance={'scalars':0})
a.Reload()
print(sorted(a.Tags()['scalars']))
for t in ("Train/mean_reward","Metrics/twist/error_vel_xy","Loss/learning_rate"):
    s=a.Scalars(t); print(t, "n=",len(s), "last=",s[-1].value, "step100=",s[100].value)
PY
```

## Walk loop (agentic, video-gated)

`/pp` runs this loop. The loop condition is a picture, not a reward: the
policy must walk slowly on two legs in a rendered rollout. Contact numbers
cross-check, the frames decide.

```
 /pp <walk task>  (pp/.opencode/commands/pp.md routes to poteto-mode)
        │
        v
 walk_loop.py  (training/scripts/walk_loop.py, strategy registry inside)
        │  state: .audit/walk_loop_state.json   trail: .audit/twoleg-walk.tsv
        v
 ┌─ round ─────────────────────────────────────────────┐
 │ 1. resume-train, bounded (1000 iters, 2048 envs)     │
 │ 2. render rollout at held 0.1 m/s  (MUJOCO_GL=glfw)  │
 │ 3. verify_walk.py watches the frames:                │
 │      pink feet -> left/right clusters per frame      │
 │      vertical traces correlated (out of phase=walk)  │
 │ 4. PASS? stop, report video.  FAIL? log row, next    │
 │    strategy. Table empty? stop, do not spin.         │
 └─────────────────────────────────────────────────────┘

 PASS predicate (scripts/verify_walk.py):
   speed within 0.05 of command  AND  each foot down >= 20 pct
   AND contact alternation <= -0.2  AND frame alternation <= -0.2
 Anything inconclusive fails. Exit 0 walks, 1 does not, 2 error.

 Strategy registry (only these run; the loop invents nothing):
   continue-1000: resume-train 1000 iters, same config (H1-H5 stack)

 Run it:
   uv run python scripts/walk_loop.py --from-run <run> --from-checkpoint <pt> --rounds 1
   uv run python scripts/walk_loop.py --verify-only --from-run <run> --from-checkpoint <pt>
```

**measured**: the verifier self-test reads alt -0.998 on synthetic
alternating frames and +1.000 on together frames. On `model_10993.pt` it
reads speed +0.070, duties 0.07/0.95, frame alt +0.42, verdict NO-WALK,
which matches the human-watched training video frame by frame.

## 2026-10-03: forward-axis bug, fixed; loop still NO-WALK

**measured** (this session)

- The command envelope had the facing axis swapped: it sampled
  `lin_vel_x` as the forward command, but on this model forward is +Y.
  Evidence, all from `training/assets/twoleg.xml` loaded in the training
  venv (`mujoco 3.14.0`) at spawn pose:
  - `L_ankle_link` / `R_ankle_link` at (±0.0442, -0.0255, -0.2332):
    left-right is X.
  - Knee world axes are ±X, so knee flexion steps the foot along Y:
    forward is Y. Matches `symmetry.py:55` ("Left-right is X, forward is Y").
  - Consequence of the old config, measured earlier in-session comment:
    holding `lin_vel_x = 0.10` moved the policy 0.080 m/s along body+x
    and 0.001 m/s along body+y — a side-step, not a walk.
- **Fixed** in `twoleg_velocity_env_cfg.py` (uncommitted):
  `LIN_VEL_X = (-0.1, 0.1)` lateral, `LIN_VEL_Y = (0.0, 0.3)` forward.
  Run `2026-10-03_08-28-43_velocity/params/env.yaml` already carries the
  fixed ranges.
- Stale camera comment at the same file's viewer block was corrected
  (it claimed the front is +x; the value `azimuth = 180` was already
  correct for a +y-facing robot).

**measured**, walk loop status (`training/scripts/walk_loop.py`,
`/tmp/opencode/loop_infinite3.log`):

```
round 5 NO-WALK -> HARD EVIDENCE fails: frame_alt=-1.0 L=1.598 R=0.901
roll=0.0 -> keep training -> continuing (infinite loop)
```

**measured**, `videos/train/rl-video-step-22000.mp4` of the same run,
8 frames at 2 fps: robot stands in place, sways, one foot lifts slightly.
No stepping, no forward travel. The run is at `model_7500.pt` (~2048 envs,
resumed from the deleted `2026-10-03_07-25-04_velocity` run's
`model_6500.pt`). That resume source directory has been removed from disk,
so restarting the loop process will fail on `--agent.load-run`; only the
live process still holds the checkpoint.

**inference**: the resumed checkpoint carries the old-axis command
distribution, and the NO-WALK metrics are unchanged across 5 loop rounds.
A clean from-scratch run with the fixed axis is the next verifiable unit;
resume state is suspect.

**measured**, actuator inventory (knee question): each leg has 4 actuated
joints — `{L,R}_waist_test`, `{L,R}_hip_test`, `{L,R}_knee_test`,
`{L,R}_ankle_test`. Each knee is a single hinge (`axis=±X`,
`ctrlrange=-1.4 1.4`). There is no second knee DOF to wire; if the
physical knee uses two servos, they drive this one knee coordinate.
