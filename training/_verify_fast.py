"""Fast GPU verify of a TwoLeg checkpoint: same gate as verify_walk.py but
N_ENVS=1, ROLLOUT_STEPS=120, cuda:0 so it finishes in ~60s instead of >400s
on CPU. Prints the same JSON verdict."""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import asdict
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import register  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "TwoLeg-Velocity-Flat"
N_ENVS = 1
ROLLOUT_STEPS = 120
CMD = 0.1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = N_ENVS
    env = ManagerBasedRlEnv(cfg=cfg, device=dev)
    venv = RslRlVecEnvWrapper(env)
    agent_cfg = asdict(load_rl_cfg(TASK))
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(venv, agent_cfg, device=dev)
    runner.load(args.ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
    policy = runner.get_inference_policy(device=dev)

    twist = env.command_manager.get_term("twist")
    obs, _ = venv.reset()
    for _ in range(30):
        c = torch.zeros_like(twist.command); c[:, 0] = CMD
        twist.command[:] = c
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))

    found, speeds, knee, hip, tilts = [], [], [], [], []
    for _ in range(ROLLOUT_STEPS):
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

    found = torch.stack(found); speeds = torch.stack(speeds)
    knee = torch.stack(knee); hip = torch.stack(hip); tilt = torch.stack(tilts)
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
    g_upright = torch.tensor(upright_frac > 0.5, dtype=torch.float32)
    g_left = (left_duty > 0.20).float()
    g_right = (right_duty > 0.20).float()
    g_speed = (speed > 0.05).float()
    g_both_lo = (both_load > 0.10).float()
    g_both_hi = (both_load < 0.85).float()
    g_knee = (knee_flex > 0.20).float()
    valid = g_upright * g_left * g_right * g_speed * g_both_lo * g_both_hi * g_knee * articulated.clamp(0, 1) * torch.tensor(alternation)

    valid_frac = valid.mean().item()
    switch_ok = (torch.abs(left_duty - right_duty) < 0.6).float().mean().item()
    verdict = (valid_frac >= 0.60 and switch_ok > 0.5 and articulated > 0.5 and upright_frac > 0.5)
    out = {"verdict": "WALKS" if verdict else "NO-WALK",
           "valid_frac": round(valid_frac, 3), "upright_frac": round(upright_frac, 3),
           "left_duty_mean": round(left_duty.mean().item(), 3),
           "right_duty_mean": round(right_duty.mean().item(), 3),
           "speed_mean": round(speed.mean().item(), 3),
           "double_support_frac": round(both_load.mean().item(), 3),
           "knee_flex_mean": round(knee_flex.mean().item(), 3),
           "hip_swing_mean": round(hip_swing.mean().item(), 3),
           "switch_ok": round(switch_ok, 3), "alternation": round(alternation, 3)}
    env.close()
    print(json.dumps(out))

if __name__ == "__main__":
    main()
