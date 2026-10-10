# TwoLeg RL — clean-slate rebuild (aligned to unitree_rl_mjlab)

This is a from-scratch codebase built on the cloned `unitree_rl_mjlab` recipe
(`/home/kenpeter/work/gh-repohub/unitree_rl_mjlab`), per the user's "start empty,
write new code" directive. The old microduck-port pipeline was wiped; the only
survivors are the duck robot asset (`twoleg.xml` + meshes at
`/home/kenpeter/work/twoleg/robot_item`) and the rigorous human-gait eval idea.

## Layout
Everything lives flat under `twoleg_rl/` — one level, no subdirectories.
- `twoleg_rl/robot.py` — robot asset wiring (EntityCfg, 8 actuated joints:
  L/R hip_roll/hip/knee/ankle; foot sites `left_foot`/`right_foot`). The model
  itself is the single source of truth at `robot_item/xml/robot_twoleg.xml`.
- `twoleg_rl/env_cfgs.py`, `rewards.py`, `rl_cfg.py`, `dc_motor_action.py` —
  `TwoLeg-Velocity-Flat` / `TwoLeg-Velocity-Rough`, cloned from Unitree's
  `unitree_g1_flat_env_cfg`. The gait shaper is `feet_gait` (phase-locked
  alternating contact, weight 0.5) — NOT air-time rewards (those caused the
  one-foot flail). Plus `foot_clearance`, `foot_slip`, `soft_landing`,
  `stand_still`, `body_orientation_l2`, `self_collisions`, and duck `pose` stds.
- `twoleg_rl/register.py` — registers both tasks with mjlab. Import it for its
  side effect before `load_env_cfg`.
- `twoleg_rl/train_twoleg.py` — training entry (imports our task
  registration, reuses Unitree's train flow; patches W&B for offline run).
- `twoleg_rl/verify_walk.py` — headless rigorous human-gait eval (returns
  JSON verdict WALKS / NO-WALK).

## Run
```
cd /home/kenpeter/work/twoleg/training
# one training round
uv run --no-sync python twoleg_rl/train_twoleg.py TwoLeg-Velocity-Flat \
  --env.scene.num-envs 4096 --agent.max-iterations 1000
# eval a checkpoint
uv run --no-sync python twoleg_rl/verify_walk.py --ckpt logs/.../model_NNNN.pt
```

## Framework
Installed editable: `unitree-rl-mjlab` (its `src` package) + `mjlab==1.2.0`,
`mujoco-warp==3.8.1`. The gait reward functions `feet_gait`/`stand_still`/
`body_orientation_l2` come from that installed package (`src.tasks.velocity.mdp`).
