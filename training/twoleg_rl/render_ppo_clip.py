"""Render a headless clip of the newest torch-rl-algorithms PPO policy.

The new stack (scripts/train_ppo.py + TwoLegGymEnv) saves raw model state
dicts (logs/twoleg_ppo/step_N.pt), which record_clip.py (rsl_rl runner)
cannot load. This script builds the same ActorCritic, loads the state dict,
rolls out deterministically with render=True, and writes an mp4.

Usage:
  PYTHONPATH=/home/kenpeter/work/torch-rl-algorithms:<training dir> \
    .venv/bin/python scripts/render_ppo_clip.py \
    --ckpt logs/twoleg_ppo/step_61440000.pt --out latest_ppo_walk.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, "/home/kenpeter/work/torch-rl-algorithms")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import mediapy as media

from twoleg_rl.twoleg_gym_env import TwoLegGymEnv
from twoleg_rl.train_ppo import build_config

try:
    from algorithms.ppo.model import PPO
except ImportError:
    from torch_rl_algorithms.algorithms.ppo.model import PPO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="latest_ppo_walk.mp4")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--hidden", type=str, default="256,256")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    hidden = tuple(int(x) for x in args.hidden.split(","))
    env = TwoLegGymEnv(device=device, num_envs=1, max_episode_steps=250, render=True)
    cfg = build_config(env.action_space.shape[0], hidden=hidden)
    model = PPO(env, device=device, config=cfg)
    model.model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.model.eval()

    obs = env.start()
    frames = []
    for _ in range(args.steps):
        with torch.no_grad():
            act = model.model.actor.get_action(obs["actor"])
        obs, infos = env.step(act)
        frame = env.env.render()
        if frame is not None:
            frames.append(frame)
        resets = infos.get("resets")
        if resets is not None and bool(resets.any()):
            obs = env.start()
    media.write_video(args.out, frames, fps=args.fps)
    print(f"[render] wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
