"""Render the robot under a RANDOM policy for ~3s to visually diagnose why
episodes die in ~1 step during early training. No checkpoint needed.

Mirrors record_clip.py's headless EGL render path but uses random actions so we
can see the cold-start collapse (per the 'watch the video' verification rule).
"""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WARP_USE_LEGACY_CODEGEN", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/kenpeter/work/torch-rl-algorithms")

import argparse
import mediapy as media
import numpy as np
import torch
from mjlab.utils.torch import configure_torch_backends
configure_torch_backends()
import twoleg_rl.tasks.velocity.config.twoleg  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "diag_collapse.mp4"))
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--task", default="TwoLeg-Velocity-Flat")
    ap.add_argument("--policy", default="random", choices=["random", "zero"])
    args = ap.parse_args()

    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode="rgb_array")
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=False)
    fps = env.metadata.get("render_fps", 30)

    obs, _ = env_wrapped.reset()
    A = 8  # 8 actuated leg joints (L/R: hip_roll, hip, knee, ankle)
    frames = []
    step = 0
    n = int(args.secs * fps) + 30
    while step < n:
        if args.policy == "random":
            actions = torch.randn(1, A)
        else:
            actions = torch.zeros(1, A)
        obs, rewards, dones, infos = env_wrapped.step(actions)
        frame = env.render()
        if frame is not None:
            f = frame[0] if frame.ndim == 4 else frame
            frames.append(np.asarray(f))
        step += 1
    env.close()
    if frames:
        clip = [f if f.dtype == np.uint8 else (np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames]
        media.write_video(args.out, clip, fps=fps)
        print(f"[diag] wrote {len(clip)} frames -> {args.out}")
    else:
        print("[diag] no frames")

if __name__ == "__main__":
    main()
