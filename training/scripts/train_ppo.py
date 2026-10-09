"""Train the TwoLeg biped with the from-scratch on-policy PPO from
torch-rl-algorithms (Pablo Gomez Martinez's repo, on your machine at
/home/kenpeter/work/torch-rl-algorithms).

Wiring:
  - TwoLegGymEnv wraps the mjlab TwoLeg env and implements the
    torch-rl-algorithms Trainer contract (start/step + observation_space/
    action_space + device), feeding obs as a dict {actor, critic} and
    infos with rewards/resets/next_observations.
  - PPO(env, config=..., device=...) is the same API as examples/train_gymnasium.py.

Run:
  export MUJOCO_GL=egl WARP_USE_LEGACY_CODEGEN=1
  PYTHONPATH=/home/kenpeter/work/torch-rl-algorithms:<this dir> \
    python scripts/train_ppo.py --steps 2000000 --device cuda:0
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
REPO = str(Path(__file__).resolve().parents[1])

import torch

from scripts.twoleg_gym_env import TwoLegGymEnv
from config import NetworkConfig, NetworkType  # torch-rl-algorithms

try:
    from algorithms.ppo.model import PPO
except ImportError:
    from torch_rl_algorithms.algorithms.ppo.model import PPO


def build_config(actions, hidden=(256, 256), lr=3e-4, steps=2_000_000, num_envs=1):
    # epoch_steps must scale with num_envs or the parallel envs are wasted:
    # each iteration should roll out ~num_envs * rollout_horizon env-steps so the
    # GPU buffer saturates. 48 env-steps/env/iter is a standard mjlab/IsaacLab
    # ratio. With num_envs=1 this stays a small 48-step rollout (safe for tests).
    rollout_per_env = 48
    epoch_steps = max(5000, num_envs * rollout_per_env)
    return {
        "train": {
            "steps": steps,
            "epoch_steps": epoch_steps,
            "checkpoint_path": str(Path(REPO) / "logs" / "twoleg_ppo"),
            "save_steps": 10_000,
            "test_episodes": 1,
            "show_progress": True,
            "replace_checkpoint": False,
            "log": True,
            "log_dir": str(Path(REPO) / "logs" / "twoleg_ppo" / "tb"),
            "log_name": "twoleg_ppo",
        },
        "model": {
            "actor_config": NetworkConfig(
                network_type=NetworkType.MLP, hidden_sizes=list(hidden)
            ).model_dump(),
            "critic_config": NetworkConfig(
                network_type=NetworkType.MLP, hidden_sizes=list(hidden)
            ).model_dump(),
        },
        "ppo": {
            "clip_param": 0.2,
            "ppo_epoch": 4,
            "num_mini_batches": 4,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "num_steps": 2048,
        },
        "actor_lr": lr,
        "critic_lr": lr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--hidden", type=str, default="256,256")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enable_mirroring", action="store_true")
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--play", action="store_true")
    args = ap.parse_args()

    hidden = tuple(int(x) for x in args.hidden.split(","))
    device = args.device if torch.cuda.is_available() else "cpu"

    env = TwoLegGymEnv(
        device=device,
        num_envs=args.num_envs,
        enable_mirroring=args.enable_mirroring,
    )

    cfg = build_config(env.action_space.shape[0], hidden=hidden, lr=args.lr, steps=args.steps, num_envs=args.num_envs)

    model = PPO(
        env,
        model_path=args.model_path,
        device=device,
        config=cfg,
    )

    if args.play:
        model.play() if hasattr(model, "play") else model.test()
        return

    print(f"[twoleg PPO] device={device} steps={args.steps} hidden={hidden}")
    model.train(steps=args.steps)


if __name__ == "__main__":
    main()
