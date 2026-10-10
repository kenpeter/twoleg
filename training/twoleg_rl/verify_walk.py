"""Headless rigorous human-gait eval for a trained TwoLeg policy.

Rolled out under ManagerBasedRlEnv with NO render context. Returns a JSON
verdict the training loop can trust. This is the same gate we honed earlier:
the policy must finish the last frames in a real two-footed alternating gait
-- not a hop, not a one-leg flail, not a squat-freeze.

Usage:
  uv run python scripts/verify_walk.py --ckpt logs/.../model_NNNN.pt
  (or set TWOLEG_CKPT env var)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import twoleg_rl.register  # noqa: F401  (registers TwoLeg tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "TwoLeg-Velocity-Flat"
N_ENVS = 4
ROLLOUT_STEPS = 250
CMD = 0.1  # slow forward command


def rolling_ok_last(window: torch.Tensor, thr: float) -> bool:
    """True if the LAST `window` fraction of envs stay upright+loading."""
    return bool(window[-1].item() > thr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--task", type=str, default=TASK)
    args = ap.parse_args()
    import os
    ckpt = args.ckpt or os.environ.get("TWOLEG_CKPT")
    if not ckpt:
        # Fall back to newest checkpoint under the twoleg_velocity log dir.
        from pathlib import Path as _P
        logs = _P(__file__).resolve().parents[1] / "logs" / "rsl_rl" / "twoleg_velocity"
        pts = sorted(logs.glob("*/model_*.pt"), reverse=True)
        ckpt = str(pts[0]) if pts else None
    if not ckpt or not Path(ckpt).exists():
        print(json.dumps({"verdict": "NO-RUN", "reason": "no checkpoint"}))
        return 2

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = N_ENVS
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    venv = RslRlVecEnvWrapper(env)
    agent_cfg = asdict(load_rl_cfg(args.task))
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(venv, agent_cfg, device="cpu")
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    twist = env.command_manager.get_term("twist")
    obs, _ = venv.reset()
    # Training forward is body +X (hip/knee axes are Y, ankle axis is X;
    # twist.lin_vel_x is the forward command). A previous revision commanded
    # body -Y here, which trains on X but evaluates on Y and always NO-WALKs.
    for _ in range(50):
        c = torch.zeros_like(twist.command); c[:, 0] = CMD
        twist.command[:] = c
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))

    found, speeds, knee, hip = [], [], [], []
    for _ in range(ROLLOUT_STEPS):
        c = torch.zeros_like(twist.command); c[:, 0] = CMD
        twist.command[:] = c
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))
        f = env.scene["feet_ground_contact"].data.found[:].clone()  # [N,2]
        found.append(f)
        v = env.scene["robot"].data.root_link_lin_vel_b
        speeds.append(v[:, 0].detach().clone())  # forward = body x
        jp = env.scene["robot"].data.joint_pos[:, :6]
        knee.append((jp[:, 1] - jp[:, 4]).abs())  # knee flexion proxy
        hip.append((jp[:, 0] - jp[:, 3]).abs())   # hip swing proxy

    found = torch.stack(found)        # [T,N,2]
    speeds = torch.stack(speeds)      # [T,N]
    knee = torch.stack(knee)          # [T,N]
    hip = torch.stack(hip)            # [T,N]

    left_duty = found[:, :, 0].float().mean(0)   # [N]
    right_duty = found[:, :, 1].float().mean(0)  # [N]
    both_load = (found.sum(-1) > 1).float().mean(0)  # [N] double support frac
    speed = speeds.mean(0).abs()        # [N]
    knee_flex = knee.max(0).values     # [N]
    hip_swing = hip.max(0).values      # [N]

    # True gait alternation: over time, left and right contact should be
    # ANTI-correlated (one down while the other is up). A squat has both feet
    # planted the whole time -> correlation ~ +1 (fails). A walk -> ~ -1.
    # Mean over envs of the time-series Pearson correlation.
    lf = found[:, :, 0].float()  # [T,N]
    rf = found[:, :, 1].float()  # [T,N]
    lc = lf - lf.mean(0)
    rc = rf - rf.mean(0)
    denom = (lc.pow(2).sum(0) * rc.pow(2).sum(0)).sqrt() + 1e-6
    corr = (lc * rc).sum(0) / denom  # [N], ~ -1 for walking, ~ +1 for squat
    alternation = (-corr).clamp(0, 1).mean().item()  # 1 = good alternation

    # articulation: joints must move (knee/hip swing present)
    articulated = (knee_flex > 0.35).float().mean() + (hip_swing > 0.20).float().mean()
    valid = (
        (left_duty > 0.20).float()
        * (right_duty > 0.20).float()
        * (speed > 0.05).float()
        * (both_load > 0.10).float() * (both_load < 0.85).float()
        * (knee_flex > 0.20).float()
        * articulated.clamp(0, 1)
        * alternation
    )
    valid_frac = valid.mean().item()
    # stance alternation proxy: left/right duty should both be high and similar
    switch_ok = (torch.abs(left_duty - right_duty) < 0.6).float().mean().item()

    verdict = (
        valid_frac >= 0.60
        and switch_ok > 0.5
        and articulated > 0.5
    )
    out = {
        "verdict": "WALKS" if verdict else "NO-WALK",
        "valid_frac": round(valid_frac, 3),
        "left_duty_mean": round(left_duty.mean().item(), 3),
        "right_duty_mean": round(right_duty.mean().item(), 3),
        "speed_mean": round(speed.mean().item(), 3),
        "double_support_frac": round(both_load.mean().item(), 3),
        "knee_flex_mean": round(knee_flex.mean().item(), 3),
        "hip_swing_mean": round(hip_swing.mean().item(), 3),
        "switch_ok": round(switch_ok, 3),
        "alternation": round(alternation, 3),
    }
    env.close()
    print(json.dumps(out))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
