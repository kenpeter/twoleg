"""Record a headless clip of the latest TwoLeg policy (last ~5 seconds).

Loads the newest checkpoint from logs/rsl_rl/twoleg_velocity/, rolls out env[0]
with the inference policy, renders offscreen (MUJOCO_GL=egl), and writes the last
5 seconds of frames to an mp4. This is the ground-truth walk clip the user watches.

Usage:
  uv run --no-sync python scripts/record_clip.py [--ckpt PATH] [--out PATH] [--secs 5]
"""

import os
import sys
from pathlib import Path

# Headless EGL rendering.
os.environ.setdefault("MUJOCO_GL", "egl")
# Warp 1.18.0 broke mujoco_warp sensor kernel codegen; legacy codegen path works.
os.environ.setdefault("WARP_USE_LEGACY_CODEGEN", "1")

ROOT = Path(__file__).resolve().parents[1]
UNITREE_REPO = Path("/home/kenpeter/work/gh-repohub/unitree_rl_mjlab")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UNITREE_REPO))

import argparse  # noqa: E402
import mediapy as media  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dataclasses import asdict  # noqa: E402

# Initialize warp/CUDA the same way train.py does (mjlab sim needs wp.context).
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

configure_torch_backends()

import register  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402

from mjlab.rl import MjlabOnPolicyRunner  # noqa: E402

LOGS = ROOT / "logs" / "rsl_rl" / "twoleg_velocity"


def newest_ckpt():
    runs = sorted([p for p in LOGS.iterdir() if p.is_dir()], reverse=True)
    for run in runs:
        pts = sorted(run.glob("model_*.pt"), reverse=True)
        if pts:
            return pts[0]
    raise SystemExit(f"No checkpoints found under {LOGS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default newest)")
    ap.add_argument("--out", default=str(ROOT / "latest_walk.mp4"))
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--task", default="TwoLeg-Velocity-Flat")
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else newest_ckpt()
    print(f"[record] ckpt = {ckpt}")

    # Build env in rgb_array mode (CPU avoids mjlab's wp.context init race on
    # this warp version; rendering still works headless via MUJOCO_GL=egl).
    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = 1
    device = "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=False)
    fps = env.metadata.get("render_fps", 30)
    dt = 1.0 / fps if fps else 0.033

    # Load runner + checkpoint, get normalized inference policy.
    agent_cfg = asdict(load_rl_cfg(args.task))
    runner = MjlabOnPolicyRunner(env_wrapped, agent_cfg, str(ckpt.parent), device)
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=device)

    obs, _ = env_wrapped.reset()
    frames = []
    roll = []  # rolling buffer of last `secs` frames
    max_frames = int(args.secs * fps)
    step = 0
    n_record = int(args.secs * fps) + 30
    while step < n_record:
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, infos = env_wrapped.step(actions)
        frame = env.render()
        if frame is not None:
            f = frame[0] if frame.ndim == 4 else frame
            roll.append(np.asarray(f))
            if len(roll) > max_frames:
                roll.pop(0)
        terminated = dones
        if bool(terminated[0]):
            obs, _ = env.reset()
        step += 1

    env.close()
    if roll:
        clip = [f if f.dtype == np.uint8 else (np.clip(f, 0, 1) * 255).astype(np.uint8) for f in roll]
        media.write_video(args.out, clip, fps=fps)
        print(f"[record] wrote {len(clip)} frames ({len(clip)/fps:.1f}s) -> {args.out}")
    else:
        print("[record] no frames captured (renderer produced None)")


if __name__ == "__main__":
    main()
