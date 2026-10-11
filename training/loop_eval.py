"""Loop eval: the OUTER/INNER loop condition.

Combines two signals into one verdict:
  1. HEADLESS NUMBERS  (verify_walk gate)  -> upright, duty, speed, alternation
  2. LAST 5 SECONDS of latest_walk.mp4     -> a MEASURED end-state signal
     (text-only agent cannot view pixels, so we quantify the clip's final
      window: mean frame-to-frame motion in foot region + torso-band vertical
      position proxy. The montage PNG is written for the human to watch; the
      loop itself uses the scalar signal.)

Exit code: 0 if WALKS (loop may stop), 1 if not (loop must reflect+edit).
Prints JSON: {verdict, numbers:{...}, video:{...}, montage}
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import register  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from dataclasses import asdict
from robot import TORSO_BODY, FOOT_BODIES

TASK = "TwoLeg-Velocity-Flat"
N_ENVS = 1
ROLLOUT = 120
CMD = 0.1
FPS = 30


def run_numbers(ckpt: str, device: str) -> dict:
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = N_ENVS
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    venv = RslRlVecEnvWrapper(env)
    agent_cfg = asdict(load_rl_cfg(TASK))
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(venv, agent_cfg, device=device)
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    twist = env.command_manager.get_term("twist")
    obs, _ = venv.reset()
    for _ in range(30):
        c = torch.zeros_like(twist.command); c[:, 0] = CMD
        twist.command[:] = c
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))
    found, speeds, knee, hip, tilts, footx, footy, torsz = [], [], [], [], [], [], [], []
    for _ in range(ROLLOUT):
        c = torch.zeros_like(twist.command); c[:, 0] = CMD
        twist.command[:] = c
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))
        f = env.scene["feet_ground_contact"].data.found[:].clone()
        found.append(f)
        v = env.scene["robot"].data.root_link_lin_vel_b
        speeds.append(v[:, 0].detach().clone())
        jp = env.scene["robot"].data.joint_pos[:, :6]
        knee.append((jp[:, 1] - jp[:, 4]).abs())
        hip.append((jp[:, 0] - jp[:, 3]).abs())
        pg = env.scene["robot"].data.projected_gravity_b
        tilts.append(pg[:, 2].detach().clone())
        # L/R foot center-of-mass position (world) -> lateral (Y) + forward (X) sep
        com = env.scene["robot"].data.body_com_pos_w  # [N, B, 3]
        li = env.scene["robot"].body_names.index("L_foot")
        ri = env.scene["robot"].body_names.index("R_foot")
        lc, rc = com[:, li], com[:, ri]
        footx.append((rc[:, 0] - lc[:, 0]).abs())   # forward separation
        footy.append((rc[:, 1] - lc[:, 1]).abs())   # lateral separation
        # torso COM height (anti-fall signal)
        ti = env.scene["robot"].body_names.index(TORSO_BODY)
        torsz.append(com[:, ti, 2])
    found = torch.stack(found); speeds = torch.stack(speeds)
    knee = torch.stack(knee); hip = torch.stack(hip); tilt = torch.stack(tilts)
    footx = torch.stack(footx); footy = torch.stack(footy); torsz = torch.stack(torsz)
    left_duty = found[:, :, 0].float().mean(0)
    right_duty = found[:, :, 1].float().mean(0)
    both_load = (found.sum(-1) > 1).float().mean(0)
    speed = speeds.mean(0).abs()
    knee_flex = knee.max(0).values
    hip_swing = hip.max(0).values
    lf = found[:, :, 0].float(); rf = found[:, :, 1].float()
    lc = lf - lf.mean(0); rc = rf - rf.mean(0)
    denom = (lc.pow(2).sum(0) * rc.pow(2).sum(0)).sqrt() + 1e-6
    corr = (lc * rc).sum(0) / denom
    alternation = (-corr).clamp(0, 1).mean().item()
    articulated = (knee_flex > 0.35).float().mean() + (hip_swing > 0.20).float().mean()
    upright_frac = (tilt.mean(0) < -0.9).float().mean().item()
    # foot separation: lateral (Y) is the crossing/angle metric; forward (X) is stride.
    foot_sep_y_t = footy.mean(0)              # [m] per-env avg lateral gap (tensor)
    foot_sep_x_t = footx.mean(0)             # [m] per-env avg forward gap (tensor)
    foot_sep_y = foot_sep_y_t.mean().item()   # scalar for reporting
    foot_sep_x = footx.mean(0).mean().item()
    feet_collide = (foot_sep_y_t < 0.06).float().mean().item()  # frac of time feet too close
    torso_h = torsz.mean(0).mean().item()   # [m] avg torso COM height (anti-fall)
    torso_low_frac = (torsz.mean(0) < 0.15).float().mean().item()  # frac of time torso near floor
    g_up = torch.tensor(upright_frac > 0.5, dtype=torch.float32)
    valid = (g_up * (left_duty > 0.20).float() * (right_duty > 0.20).float()
             * (speed > 0.05).float() * (both_load > 0.10).float()
             * (both_load < 0.85).float() * (knee_flex > 0.20).float()
             * (foot_sep_y_t > 0.06).float()   # feet must keep a lateral gap (no crossing)
             * articulated.clamp(0, 1) * torch.tensor(alternation))
    valid_frac = valid.mean().item()
    switch_ok = (torch.abs(left_duty - right_duty) < 0.6).float().mean().item()
    verdict = (valid_frac >= 0.60 and switch_ok > 0.5 and articulated > 0.5 and upright_frac > 0.5)
    env.close()
    return {"verdict": "WALKS" if verdict else "NO-WALK", "valid_frac": round(valid_frac, 3),
            "upright_frac": round(upright_frac, 3),
            "left_duty": round(left_duty.mean().item(), 3),
            "right_duty": round(right_duty.mean().item(), 3),
            "speed": round(speed.mean().item(), 3),
            "double_support": round(both_load.mean().item(), 3),
            "knee_flex": round(knee_flex.mean().item(), 3),
            "alternation": round(alternation, 3), "switch_ok": round(switch_ok, 3),
            "foot_sep_y": round(foot_sep_y, 3), "foot_sep_x": round(foot_sep_x, 3),
            "feet_collide_frac": round(feet_collide, 3),
            "torso_h": round(torso_h, 3), "torso_low_frac": round(torso_low_frac, 3)}


def run_video_signal(mp4: str, outdir: Path) -> dict:
    """Quantify the LAST 5 seconds of the clip (the loop-condition video half).

    Returns a scalar motion signal PLUS a fall signal: if the torso band
    collapses (vertical COM drops hard) or the robot topples (horizontal COM
    shifts hard), `fell` is True. The loop treats a fall as NO-WALK even when
    the numbers gate might otherwise look ambiguous.
    """
    import numpy as np
    outdir.mkdir(parents=True, exist_ok=True)
    if not Path(mp4).exists():
        return {"last5_frames": 0, "mean_frame_motion": 0.0,
                "montage": None, "clip_frames": 0, "fell": False, "error": "no clip file"}
    # total frames
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of",
                        "default=nokey=1:noprint_wrappers=1", mp4],
                       capture_output=True, text=True)
    total = int(r.stdout.strip() or 0) or 500
    last5 = min(150, total)
    start = max(0, total - last5)
    # extract last-5s frames at 10 fps for a compact montage + motion profile
    seq = list(range(start, total, max(1, (total - start) // 10)))[:10]
    prev = None
    motions = []
    row_coms = []   # vertical body-band position (0 top, 1 bottom)
    col_coms = []   # horizontal shift (0 left, 1 right)
    for i, fr in enumerate(seq):
        out = outdir / f"last5_{i:02d}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{fr/30:.3f}", "-i", mp4,
                        "-frames:v", "1", "-vf", "scale=200:-1", str(out)],
                       capture_output=True, text=True)
        if not out.exists():
            continue
        cur = _load(out)
        lum = np.asarray(cur, dtype=float).mean(2)
        rc = lum.mean(1); cc = lum.mean(0)
        if rc.sum() > 0:
            row_coms.append(float((np.arange(len(rc)) * rc).sum() / rc.sum()) / len(rc))
        if cc.sum() > 0:
            col_coms.append(float((np.arange(len(cc)) * cc).sum() / cc.sum()) / len(cc))
        if prev is not None:
            a = np.asarray(prev, dtype=float); b = np.asarray(cur, dtype=float)
            motions.append(float(np.mean(np.abs(a - b))))
        prev = cur
    # fall signal: last frame band dropped >5% vertically or shifted >8% horizontally
    fell = False
    if len(row_coms) >= 2:
        vdrop = row_coms[0] - row_coms[-1]   # positive = body moved DOWN
        hshift = abs(col_coms[-1] - col_coms[0]) if len(col_coms) >= 2 else 0.0
        fell = (vdrop > 0.05) or (hshift > 0.08)
    # montage of the last-5s sequence (PIL grid — robust, no ffmpeg filter quirks)
    files = [str(outdir / f"last5_{i:02d}.png") for i in range(10) if (outdir / f"last5_{i:02d}.png").exists()]
    montage = outdir / "last5_montage.png"
    if len(files) >= 2:
        try:
            from PIL import Image
            imgs = [Image.open(f).convert("RGB") for f in files]
            w, h = imgs[0].size
            cols = int(len(imgs) ** 0.5) + 1
            rows = (len(imgs) + cols - 1) // cols
            grid = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
            for idx, im in enumerate(imgs):
                grid.paste(im, ((idx % cols) * w, (idx // cols) * h))
            grid.save(montage)
        except Exception as e:
            montage = None
            print(f"[montage] failed: {e}")
    motion = round(float(sum(motions) / len(motions)), 4) if motions else 0.0
    return {"last5_frames": len(files), "mean_frame_motion": motion,
            "montage": str(montage) if montage.exists() else None,
            "clip_frames": total, "fell": fell,
            "vdrop": round(row_coms[0] - row_coms[-1], 3) if len(row_coms) >= 2 else None,
            "hshift": round(abs(col_coms[-1] - col_coms[0]), 3) if len(col_coms) >= 2 else None}


def _load(p):
    from PIL import Image
    return Image.open(p).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video", default=str(ROOT / "latest_walk.mp4"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-json", default=str(ROOT / "loop_eval.json"))
    args = ap.parse_args()
    nums = run_numbers(args.ckpt, args.device)
    vid = run_video_signal(args.video, ROOT / "scripts" / "_frames")
    # combined verdict: numbers must WALK AND the clip must actually show motion
    # in the last 5s (a frozen/faceplant clip fails even if numbers look ok). A
    # detected fall also fails.
    walks = (nums["verdict"] == "WALKS" and vid["mean_frame_motion"] > 1.0
             and not vid.get("fell", False))
    out = {"verdict": "WALKS" if walks else "NO-WALK",
           "numbers": nums, "video": vid}
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
    sys.exit(0 if walks else 1)


if __name__ == "__main__":
    main()
