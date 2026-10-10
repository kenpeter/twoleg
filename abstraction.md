# BipedRobot adoption abstraction

## Scope and how to read this file

This file answers two questions:

1. What is in `/home/kenpeter/work/BipedRobot`, where did it come from, and
   how does it work?
2. What did twoleg take from it, and what still does not transfer?

Every claim carries a label:

- **measured**: read directly from a file, git log, or config on disk.
- **derived**: arithmetic or structural conclusion from measured values.
- **inference**: reading of the evidence. Not a measurement.

Nothing here was trained or evaluated. All statements are readings of the
checkout at `/home/kenpeter/work/BipedRobot` (HEAD `932d42a`, 94 commits).

## What is in ~/work/BipedRobot

**measured**: it is a clean clone of `git@github.com:AsterisCrack/BipedRobot.git`,
HEAD `932d42a` ("More randomization and noise to better match reality"), no
local modifications. License: MIT (`LICENSE.txt`), so copying with
attribution is allowed.

**measured**, top-level layout:

```
config/                  # YAML config + Pydantic schema
  config.yaml            # live training config (SAC, BipedV2_SAC, rough terrain)
  schema.py              # typed fields and defaults
envs/
  isaaclab/              # primary env: BipedEnv (DirectRLEnv), rewards, mdp
  mujoco/                # secondary env: MujocoEnv, base_env, distributed
  rewards/mujoco_reward.py
  utils/mirroring.py     # left-right obs/action mirroring
  utils/randomizer.py    # physics randomization
  assets/robot, robotV2  # MuJoCo XML, URDF, USD
src/
  isaaclab/              # train.py, play.py, export_onnx.py, verify_onnx_bundle.py
  mujoco/                # train.py, train_lstm.py, test_model.py, sim2real.py
torch_rl_algorithms/     # PPO, SAC, DDPG, D4PG, MPO; MLP/CNN/LSTM/transformer
  algorithms/{ppo,sac,ddpg,d4pg,mpo}
robot_motion_reference/  # FBX -> NPZ motion imitation pipeline
  processed/             # only joint_map_template.json shipped; no .npz
tests/                   # test_env.py, test_transformer.py
media/                   # robot.jpg, WalkingVideo.gif
```

**measured**, the README's advertised story matches most of the tree: custom
RL algorithms, swappable backbones, Isaac Lab primary with a MuJoCo twin that
shares observation/action/reward code, 18 configurable reward terms, FBX
motion imitation, and sim-to-real randomization plus left-right mirroring.

**measured**, where the README is stale: quickstart cites
`config/final/train_config_sac.yaml`, which does not exist; the real configs
are `config/config.yaml` (+ `config/isaac/`). The motion-imitation
`processed/` dir ships only a template, no usable NPZ.

## How it works

**measured**, from `config/config.yaml` and the env source:

```
config/config.yaml  (SAC, model BipedV2_SAC, steps 500k,
                     actor_obs "normal", critic_obs "privileged",
                     history 5, normalize_obs, symmetry_augmentation,
                     use_rough_terrain)
        │
        v
envs/isaaclab/biped_env.py  BipedEnv (DirectRLEnv, 4096 parallel envs, GPU)
  observation_space = 48        [biped_env_cfg.py:180]
  obs noise / sensor lag applied per control step
        │  shared with MuJoCo twin via identical obs/action/reward layout
        v
envs/isaaclab/rewards/rewards.py   (shared pure-torch terms, 27 functions)
envs/mujoco/mujoco_env.py          MujocoEnv: nu=12 actuators
        │  12 DOF action space: per leg hip_z/hip_x/hip_y/knee/ankle_y/ankle_x
        v
torch_rl_algorithms/algorithms/sac  (off-policy SAC; also PPO/DDPG/D4PG/MPO)
        │  CNN/LSTM/transformer backbones selectable
        v
export_onnx.py -> standalone ONNX bundle for hardware
```

**measured**, reward config in `config/config.yaml` (abridged, weights live):
`track_lin_vel_xy_exp 2.0`, `track_ang_vel_z_exp 1.5`,
`termination_penalty -0.2`, `flat_orientation_l2 -2.0`, `lin_vel_z_l2 -2`,
`dof_pos_limits -1.0`, gait-shaping terms phased from `0.0` upward,
`symmetry_augmentation` on, and comments documenting three past gait-clock
defects (inverted feet indices, mirrored phase clock, wrong stride length).

**measured**, the physical robot is ~25 cm scale with 25 kg-class servos;
sim and CAD masses were compared on 2026-10-08; the compiled twoleg MJCF
is ~4.5 kg while BipedRobot compiles at 1.468 kg (see ledger below).

## What twoleg adopted

**measured** (2026-10-10, superseding the 2026-10-08 audit):

- **Withdrawn**: the previously claimed "phased gait-weight curriculum (`0.0`
  until stable, then ramp)". Reading their `config/config.yaml` directly
  shows no such ramp exists — see the row 16 correction in the ledger.
- `termination_penalty` at `-10.0` and `dof_pos_limits` at `-1.0`, both now
  live in the resolved config and verified.
- PD gains aligned to their measured `kp 21.1`, `damping 0.0`;
  `frictionloss 0.03`; joint limits raised from 2/9 to 8/9 with spans taken
  from measured flexion direction rather than copied sign-for-blind.
- The stance itself was solved locally, not copied: their
  `default_joint_pos` is all zeros, so their GIF shows a policy holding a
  pose rather than a hand-set one. Our standing stance is an ankle
  correction (`-0.5`), and every knee above 0 falls.
- Small portable utilities: `envs/utils/mirroring.py`, `randomizer.py`.
  Their leg geometry is one-sided (`knee` range `0..2.36`); see
  `.opencode/skills/leg-align/` for the DOF/strength/mass checklist.

**measured**, what does not transfer yet:

- **DOF gap**: BipedRobot actuates 12 joints (6/leg, `nu=12`); twoleg's
  mjlab model actuates 8 (4/leg: hip_roll, hip, knee, ankle). Their
  policy, env, and reward tensors assume 12; copying the brain onto a
  smaller action space cannot reproduce their gait. The two axes twoleg
  lacks outright are hip abduction and ankle pitch, and both are hardware
  rather than tuning.
- **Algorithm gap**: they train SAC (off-policy); twoleg runs on-policy PPO
  under mjlab/rsl_rl. Their SAC code is usable only as a second algorithm
  after the DOF gap closes.
- **Motion imitation**: their FBX->NPZ pipeline needs Isaac Lab (not
  installed here) and ships no processed NPZ.
- **`robot_motion_reference` checkout** is read-only reference material.

**inference**: the highest-value portable ideas are the phased-in gait
weights, the alternation-gated gait clock (their comments say this exact
term had three separate sign/index bugs — the same failure class twoleg
hit), and the symmetry augmentation. None of them require matching their
12-DOF body.

## Alignment ledger (2026-10-10, measured)

Every figure below is read from the compiled MJCF or the resolved config, not
from a comment. "Was" is the pre-alignment value on this branch.

| # | Item | BipedRobot | Was | Now | Status |
|---|---|---|---|---|---|
| 1 | Spawn height `z` | — | `0.0` (soles 0.251 m inside floor) | `0.251` | fixed |
| 2 | Stance ankle | `0` (policy-held) | `0.0` | `-0.5` | **stands** |
| 3 | Stance knee | `0` | `0.0` | `0.0` | verified |
| 4 | PD gain `kp` | `21.1` | `40.0` | `21.1` | aligned |
| 5 | PD `damping` | `0.0` | `0.5` | `0.0` | aligned |
| 6 | `effort_limit` | `±5.0` (204% of their servo) | `1.0` | `2.45` | aligned to our hardware |
| 7 | Joints limited | `12/13` | `2/9` | `8/9` | aligned |
| 8 | Ankle range | `-0.436…1.571` | none | `±1.571` | re-derived |
| 9 | Hip range | `±2.00713` | none | `±1.2` | aligned |
| 10 | Knee range | `0…2.35619` | none | `-0.15…1.9` | sign measured |
| 11 | Hip-roll range | `-0.698…1.571` | `±0.785` | `±0.785` | kept |
| 12 | `frictionloss` | `0.03` | `0.02` | `0.03` | aligned |
| 13 | `dof_pos_limits` | `-1.0` | absent | `-1.0` | matched |
| 14 | `termination_penalty` | `-10.0` | absent | `-10.0` | written, verified |
| 15 | Upright gate in eval | — | none | `projected_gravity_b z < -0.9` | added |
| 16 | Swing foot height | `1.5` (their one ACTIVE shaper) | `0.5` | `1.5` | aligned |
| 16b | `step_length` / `knee_bend_touchdown` / `torso_centering` | `0.0` — OFF | `0.15` / `1.0` / `1.0` — ON | kept on | not aligned, see note |
| 17 | Hip abduction (axis X) | `hip_y` | absent | absent | hardware |
| 18 | Ankle pitch (axis Y) | `ankle_y` | absent | absent | hardware |
| 19 | Actuators | `12` (6/leg) | `8` (4/leg) | `8` | hardware |
| 20 | Compiled mass | `1.468 kg` | `4.482 kg` | `4.482 kg` | kept CAD |
| 21 | Biped | `123` | `15` | `123` | ours is richer |
| 22 | Colliding geoms | `18 ngeom` | `2` | `8` | aligned |
| 23 | Algorithm | SAC + PPO | PPO | PPO | **settled: stay on-policy** |
| 24 | Sim backend | Isaac Lab / MuJoCo | mjlab + warp | mjlab + warp | gap |
| 25 | Motion imitation | FBX→NPZ | none | none | not portable |

Notes on the non-obvious rows:

- **Row 6, effort aligned to hardware, not to their sim.** Our servo is a
  DS3245 at **45 kg·cm = 4.41 N·m**; theirs is 25 kg·cm = 2.45 N·m. We now run
  2.45, which is 56% of our stall. Their sim runs 5.0, which is **204% of
  their own servo**, so copying their number would put us at 113% of ours.
  Effort does not affect standing or recovery (at equilibrium the PD error is
  ~0, so no torque is requested; measured identical 0.74 deg tilt at every
  level from 1.0 to 5.0, and identical outcomes under kicks up to 20 rad/s).
  It does bound large pose changes: commanding the knee 0.0 to 0.8 rad
  reached 0.710 at 1.0 N·m (11% short) versus 0.808 at 2.45. That is what
  walking does, which is why this row is now aligned rather than deferred.
- **Row 8, ankle re-derived rather than copied.** Mapping the ankle over
  -2.5…2.0 showed two flat-standing branches (-2.0…-0.4 and +1.0…+2.0) with a
  toe-edge dead zone between. An earlier `-0.9…0.4` clipped one branch and
  enclosed the dead zone. `±1.571` is the servo envelope and contains both.
- **Row 2, the stance is an ankle correction.** Every knee above 0 still
  falls (65–142 deg tilt). 20 s at `-0.5` holds 0.74 deg tilt, 7 contacts.
- **Row 15.** `verify_walk.py` gated on contact, speed, duty and alternation
  with no upright check, so a robot walking on its side passed. Root height
  barely moves during a side-fall, so height-based gates are blind to the
  dominant failure mode.
- **Row 23, settled by decision 2026-10-10.** We stay on PPO. mjlab exports
  only on-policy runners (`MjlabOnPolicyRunner`, `RslRlBaseRunner`,
  `RslRlOnPolicyRunner`); SAC would need a new off-policy runner or a second
  training script through the existing gym adapter. Nothing indicates SAC is
  why BipedRobot walks — the likelier causes are 12 DOF and 1.468 kg, both of
  which are hardware. This is a choice, not a gap.
- **Row 16, correction.** An earlier version of this ledger claimed
  BipedRobot ramps gait weights from `0.0`. **It does not.** Their
  `config/config.yaml` ships `swing_foot_height: 1.5` with
  `step_length`, `knee_bend_touchdown` and `torso_centering` all at `0.0`,
  and nothing mutates that table at runtime. The only curriculum in their
  config is domain randomisation (COM/payload 0 to 100% over 40k steps), not
  reward weights. There was no ramp to copy.

Rule of thumb: everything reward-shaped and conceptual ports; everything that
assumes 12 DOFs, SAC, or Isaac Lab does not.

### Stance evidence (2026-10-10)

Standalone 20 s rollout, gate final tilt < 15 deg:

```
knee=0, hip=0, hip_roll=0, ankle=-0.5
  dz=-0.0033 m   TILT=0.74 deg   ncon=7   STANDS
```

Real runtime, `ManagerBasedRlEnv(cfg, "cpu")`, 8 envs, 600 steps at zero
action:

```
upright_frac @100..600 steps = 1.00 throughout
terminated frac = 0.0        STANCE_HOLDS
```

`termination_penalty` cases: post-reset `0.0`, terminated `1.0` (-> -10.0),
truncated `0.0`, both `0.0`. The term excludes time-outs, so falling costs
more than running out the clock.

## TwoLeg stack (2026-10-10)

TwoLeg trains with mjlab 1.2 + MuJoCo Warp + rsl_rl PPO on an 8-actuator
(4-per-leg) model compiled from `robot_item/xml/robot_twoleg.xml`, on-policy.
The model is the single source of truth for both the viewer and the trainer;
it is declared `nu=0` in XML because mjlab injects one
`BuiltinPositionActuatorCfg` over 8 targets (mjlab appends actuators rather
than clearing, so an XML actuator would give 14).

The walk-gate is `upright + both-feet load + signed speed + stance-switch
rate + knee/hip articulation`, where upright is `projected_gravity_b z < -0.9`.
BipedRobot contributes patterns and reference implementations, not a drop-in
replacement.

**measured** state as of this commit: the stance holds (0.74 deg tilt over
20 s standalone, `upright_frac` 1.00 over 600 steps in the real runtime). No
PPO iteration has been run against this model yet.
