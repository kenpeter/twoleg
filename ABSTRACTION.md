# TwoLeg PPO — VRAM-maxing training config (RTX 4070 Ti, 12 GB)

## Principle
GPU VRAM is the limiting resource; env count and rollout horizon scale linearly with VRAM usage. Only ~1.4 GB / 12 GB is used at `num_envs=2048` — headroom for ~8× more envs.

## Recommended max-VRAM launch

```
export MUJOCO_GL=egl WARP_USE_LEGACY_CODEGEN=1
PYTHONPATH=/home/kenpeter/work/torch-rl-algorithms:/home/kenpeter/work/twoleg/training \
  training/.venv/bin/python training/train_ppo.py \
    --steps 2000000 \
    --device cuda:0 \
    --num_envs 8192 \
    --max_episode_steps 250
```

Config knobs in `training/train_ppo.py` (`build_config`):
- `num_envs` — parallel envs. Primary throughput lever. `epoch_steps = num_envs * rollout_horizon`.
- `rollout_horizon` / `num_steps` (default 48) — buffer capacity per PPO update. Larger = fewer, bigger updates; more stable gradient estimates.
- `save_steps` (default 1000) — checkpoint interval (overwrites old via `replace_checkpoint=True`).
- `ppo_epoch` (default 4), `num_mini_batches` (default 4) — 16 gradient steps per rollout.

## VRAM budget guide (RTX 4070 Ti, 12 GB)

| num_envs | horizon | buffer transitions | est. VRAM | iter/s | est. wall time (2M steps) |
|----------|---------|-------------------|-----------|--------|---------------------------|
| 1        | 48      | 48                | ~0.5 GB   | 1.8    | ~13 days (unusable)       |
| 2048     | 48      | 98k               | ~1.4 GB   | 1.35   | ~12 min                   |
| 8192     | 48      | 393k              | ~5 GB     | ~4-6   | ~3-5 min                  |
| 8192     | 256     | 2.1M              | ~8-10 GB  | ~2-3   | ~6-8 min                  |

## Rules
1. **Never run num_envs=1** — 440 ms/step, GPU underutilized (~8%).
2. **Keep horizon ≥ 48** — small horizons cause noisy advantage estimates; flat reward.
3. **Memory is bounded by horizon × num_envs**, not by total env steps — safe to scale num_envs up until VRAM ~10 GB.
4. **Checkpoints overwrite** (`replace_checkpoint=True`) — no accumulation.
5. **Resume**: `--resume` needs `trainer_state.pt`; fresh start uses `--model_path <ckpt>.pt`.

## Verify VRAM use
`nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader` — target 8–10 GB used during training.
