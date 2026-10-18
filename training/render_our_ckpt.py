"""Render a walk clip from a torch-rl-algorithms PPO step_*.pt checkpoint.

record_clip.py expects the mjlab/rsl_rl runner format. This run was trained
with algorithms.ppo.model.PPO, whose checkpoints are bare model state_dicts,
so load them straight into the same ActorCritic.

Key detail: TwoLegGymEnv only enables offscreen rendering when constructed
with render=True (that is what sets scene.render_mode="rgb_array" and passes
render_mode to ManagerBasedRlEnv). Without it env.render() returns None.
"""
import os, sys, argparse
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WARP_USE_LEGACY_CODEGEN", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, "/home/kenpeter/work/torch-rl-algorithms")

import numpy as np
import torch
import mediapy as media

import register  # noqa: F401
from twoleg_gym_env import TwoLegGymEnv
from train_ppo import build_config
from algorithms.ppo.model import PPO

BROWSER_SAFE = ["-profile:v", "baseline", "-level", "3.0", "-bf", "0",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def to_frame(x):
    """Normalise an env render result into an uint8 HxWx3 array, or None."""
    if x is None:
        return None
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    # ManagerBasedRlEnv.render() can return a numpy *string* (a status/mode
    # marker) when offscreen capture is unavailable; that is not an image.
    if arr.dtype.kind in ("U", "S"):
        return None
    if arr.dtype == object:
        try:
            arr = np.stack([np.asarray(i) for i in arr])
        except Exception:
            return None
        if arr.dtype.kind in ("U", "S"):
            return None
    if arr.ndim == 2:                     # grayscale -> RGB
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(ROOT / "training" / "logs" / "walk.mp4"))
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--task", default="TwoLeg-Velocity-Flat")
    ap.add_argument("--cmd", type=float, default=0.3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="seconds to run before starting capture")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--distance", type=float, default=1.15)
    ap.add_argument("--lookat-z", type=float, default=0.20)
    ap.add_argument("--elevation", type=float, default=-12.0)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    print(f"[clip] ckpt = {ckpt}")

    # render=True is what enables offscreen rgb_array frames.
    env = TwoLegGymEnv(device="cpu", num_envs=1, max_episode_steps=1000,
                       enable_mirroring=False, render=True, task=args.task)
    cfg_env = env.env
    try:
        cfg_env.scene.viewer.width = args.width
        cfg_env.scene.viewer.height = args.height
    except Exception:
        pass

    nact = env.action_space.shape[0]
    cfg = build_config(nact, hidden=(256, 256), lr=3e-4, steps=100,
                       num_envs=1, max_episode_steps=1000)
    model = PPO(env, model_path=None, device="cpu", config=cfg)
    missing, unexpected = model.model.load_state_dict(
        torch.load(str(ckpt), map_location="cpu"), strict=False)
    print(f"[clip] loaded: missing={len(missing)} unexpected={len(unexpected)}")
    model.model.eval()

    # forward velocity command
    try:
        tw = cfg_env.command_manager.get_term("twist")
        c = torch.zeros_like(tw.command)
        c[:, 0] = args.cmd
        tw.command[:] = c
        print(f"[clip] command forward vel = {args.cmd} m/s")
    except Exception as e:
        print(f"[clip] no twist command ({type(e).__name__})")

    fps = args.fps
    n_keep = int(args.secs * fps)
    n_total = int((args.secs + args.warmup) * fps)

    def act_from(ob):
        """MLPActor.get_action() returns the deterministic mean action."""
        return model.model.actor.get_action(ob)

    # NOTE: TwoLegGymEnv.start() returns a 1-tuple: (obs_dict,)
    start_out = env.start()
    obs = start_out[0] if isinstance(start_out, tuple) else start_out
    frames, heights, xpos = [], [], []
    nsteps_total = int((args.secs + args.warmup) * fps)

    with torch.no_grad():
        for i in range(nsteps_total):
            ob = obs["actor"] if isinstance(obs, dict) else obs
            if not torch.is_tensor(ob):
                ob = torch.as_tensor(np.asarray(ob), dtype=torch.float32)
            else:
                ob = ob.detach().to(torch.float32)
            act = act_from(ob)

            step_out = env.step(act)
            obs = step_out[0] if isinstance(step_out, (tuple, list)) else step_out

            robot = cfg_env.scene["robot"].data
            heights.append(float(robot.root_link_pos_w[0, 2]))
            xpos.append(float(robot.root_link_pos_w[0, 0]))
            f = to_frame(cfg_env.render())
            if f is not None:
                frames.append(f)

    print(f"[clip] rendered frames = {len(frames)} / {n_total}")
    if heights:
        print(f"[clip] torso z: start={heights[0]:.3f} min={min(heights):.3f} max={max(heights):.3f}")
        print(f"[clip] forward x: start={xpos[0]:.3f} end={xpos[-1]:.3f} travelled={xpos[-1]-xpos[0]:+.3f} m")
    fell = min(heights) < 0.10 if heights else True

    keep = frames[-n_keep:] if len(frames) > n_keep else frames
    if not keep:
        print("[clip] NO FRAMES - rendering unavailable")
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    media.write_video(args.out, np.stack(keep), fps=fps, ffmpeg_args=BROWSER_SAFE)
    size = Path(args.out).stat().st_size
    print(f"[clip] WROTE {args.out} ({len(keep)} frames, {size} bytes)")
    print(f"[clip] VERDICT: {'FALL/COLLAPSE detected' if fell else 'stayed upright'}")


if __name__ == "__main__":
    main()
