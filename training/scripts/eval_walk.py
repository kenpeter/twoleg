"""Measure what a TwoLeg checkpoint actually does, in metres.

Reward and curriculum terms are not speed. This loads a checkpoint, holds a
fixed forward command, rolls out, and reports the base's world displacement and
mean velocity. Same posture as microduck_rl's scripts/eval_sprint_speed.py.

    uv run python training/scripts/eval_walk.py --checkpoint <path/to/model.pt>
    uv run python training/scripts/eval_walk.py --checkpoint <pt> --command 0.3 --steps 500
"""

import argparse
import os
from dataclasses import asdict

import torch

TASK = "Mjlab-Velocity-Flat-TwoLeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--command", type=float, default=0.3, help="forward m/s to hold")
    ap.add_argument("--steps", type=int, default=500, help="control steps (50 Hz)")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = args.envs
    agent_cfg = load_rl_cfg(TASK)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    venv = RslRlVecEnvWrapper(env)
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(venv, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    robot = env.scene["robot"]
    # Global heading metric: the model's forward is body +y here, so reset
    # mjlab's +x default to avoid measuring side-sway as forward travel.
    robot.data.forward_vec_b[:] = 0.0
    robot.data.forward_vec_b[:, 1] = 1.0
    twist = env.command_manager.get_term("twist")

    import math

    obs, _ = venv.reset()
    start = robot.data.root_link_pos_w[:, :2].clone()

    # NOTE: each env resets to its own heading, so world-frame x averages out
    # across envs and a signed world-x travel reads near zero even when every
    # env walks forward. Integrate each step's displacement against that step's
    # heading instead. (Verified: matches the body-velocity integral to 1e-3.)
    forward, P, H = [], [], []
    for _ in range(args.steps):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = args.command
        twist.command[:] = cmd
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = venv.step(actions)
        forward.append(robot.data.root_com_lin_vel_w[:, :2].norm(dim=1).mean().item())
        P.append(robot.data.root_link_pos_w[:, :2].clone())
        H.append(robot.data.heading_w.clone())
    P = torch.stack(P)
    H = torch.stack(H)

    d = P[1:] - P[:-1]
    h = H[:-1]
    fwd = torch.stack([torch.cos(h), torch.sin(h)], -1)
    travel = (d * fwd).sum(-1).sum(0)  # per env, metres along its own axis
    elapsed = (P.shape[0] - 1) * env.step_dt
    fwd_speed = (travel / elapsed).mean().item()

    end = robot.data.root_link_pos_w[:, :2]
    displacement = (end - start).norm(dim=1)
    mean_speed = displacement.mean().item() / elapsed

    print(f"checkpoint      : {os.path.basename(args.checkpoint)}")
    print(f"held command    : {args.command:.2f} m/s forward")
    print(f"control steps   : {args.steps}  ({elapsed:.1f} s at {1/env.step_dt:.0f} Hz)")
    print(f"envs            : {args.envs}")
    print()
    print(f"mean |v| xy     : {sum(forward)/len(forward):.3f} m/s   (speed, direction ignored)")
    print(f"net displacement: {displacement.mean().item():.3f} m  "
          f"(p10 {displacement.quantile(0.1).item():.3f}, p90 {displacement.quantile(0.9).item():.3f})")
    print(f"net speed       : {mean_speed:.3f} m/s   (displacement / time)")
    print(f"forward travel  : {travel.mean().item():+.3f} m along own axis  "
          f"(p10 {travel.quantile(0.1).item():+.3f}, p90 {travel.quantile(0.9).item():+.3f})")
    print(f"forward speed   : {fwd_speed:+.3f} m/s")
    print()
    verdict = "WALKS" if fwd_speed > 0.05 else "does not translate forward"
    print(f"verdict         : {verdict}")


if __name__ == "__main__":
    main()
