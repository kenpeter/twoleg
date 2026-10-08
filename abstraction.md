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

**measured** (2026-10-08, from the prior adoption audit):

- Phased gait-weight curriculum pattern (`0.0` until stable, then ramp) —
  the lever twoleg's always-on reward terms lacked.
- `termination_penalty` and `dof_pos_limits` semantics.
- Small portable utilities: `envs/utils/mirroring.py`, `randomizer.py`.
- The BipedRobot leg geometry is one-sided (`knee` range `0..2.36`), matching
  twoleg's measured flexion direction; see `.opencode/skills/leg-align/` for
  the DOF/strength/mass comparison checklist.

**measured**, what does not transfer yet:

- **DOF gap**: BipedRobot actuates 12 joints (6/leg, `nu=12`); twoleg's
  mjlab model actuates the reduced set aligned to its servo count. Their
  policy, env, and reward tensors assume 12; copying the brain onto a
  smaller action space cannot reproduce their gait.
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

## Alignment ledger (2026-10-08, measured)

Aligned:

- **Reward shape, H9**: `foot_gait` 0.15 phased gait term, `stand_still`,
  `body_orientation_l2`, no air-time jackpots — mirrors the BipedRobot
  gait-reward philosophy in `envs/rewards/mujoco_reward.py`.
- **Knee flexion direction**: their knee range `0..2.35619` (one-sided)
  matches twoleg's measured flexion direction, so flexion sign ports
  directly.
- **Patterns imported**: `dof_pos_limits`, `termination` semantics, phased
  gait weights, mirroring/randomizer utilities (copied from
  `envs/utils/`).

Not aligned:

- **DOF count**: BipedRobot 12 actuators (6/leg: hip_z, hip_y, hip_x, knee,
  ankle_y, ankle_x); twoleg H9 has 6 (3/leg: hip pitch, knee, ankle roll),
  nu=6. Missing: hip yaw, hip roll, ankle pitch. Their policy/env/reward
  tensors assume 12 and cannot transfer.
- **Joint ranges**: theirs hip_x ±2.00713, knee 0..2.35619, hip_z
  -1.5708..0.698, hip_y -1.5708..0.785, ankle_y ±1.91986, ankle_x
  -1.5708..0.436. Ours still duck defaults: hip ±1.5708, knee
  -1.5708..0, ankle ±1.5708 (twoleg.xml:362,386,419). The widening was
  scoped but never applied.
- **Actuator numbers**: theirs position kp 21.1, forcerange ±5, damping
  1.084, armature 0.045, frictionloss 0.03 (robot_mujoco.xml:31-32);
  robotV2 damping 0, armature 0.04, frictionloss 0.2. Ours
  stiffness 40.0, damping 0.5, effort 1.0, armature 0.01
  (twoleg_constants.py:35-40), XML damping 0.05, frictionloss 0.02.
  Effort 1.0 vs their 2.94 N·m rated is not reconciled.
- **Solver/env**: they train Isaac Lab (primary) + a MuJoCo twin with SAC;
  twoleg trains mjlab + mujoco_warp with PPO. No Isaac Lab here.
- **Motion imitation**: their FBX->NPZ pipeline ships no processed NPZ and
  needs Isaac Lab; not portable.
- **Mass**: theirs 1.468 kg compiled; our H9 MJCF compiles at ~4.5 kg
  (2026-10-08 comparison), so a true 1:1 match still pending.

Rule of thumb: everything reward-shaped and conceptual ports; everything
that assumes 12 DOFs, SAC, or Isaac Lab does not.

## TwoLeg stack in one paragraph (unchanged scope)

TwoLeg trains with mjlab 1.3 + MuJoCo Warp + rsl_rl PPO on a 6-actuator
(3-per-leg) H9 model, 4096 envs, on-policy, with the walk-gate
(`upright + both-feet load + signed speed + stance-switch rate + knee/hip
articulation`) as the predicate. BipedRobot contributes patterns and
reference implementations, not a drop-in replacement.
