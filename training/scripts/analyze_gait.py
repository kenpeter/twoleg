"""Diagnose the gait: is the TwoLeg policy walking, hopping, or shuffling?

Reward terms can be satisfied by a hop. This rolls out a checkpoint with a fixed
forward command and reports what the feet actually do: the contact distribution
(both down / one down / both up), per-foot contact runs, and whether the two
feet alternate. Alternating single-foot support is what "walking" means.

    uv run python training/scripts/analyze_gait.py --checkpoint <pt> --command 0.3
"""

import argparse
import os
from dataclasses import asdict

import numpy as np
import torch

TASK = "Mjlab-Velocity-Flat-TwoLeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--command", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--envs", type=int, default=8)
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
    # NOTE: do NOT hoist .data.found out of the loop. ContactSensor reallocates
    # the tensor every step, so a captured handle stays frozen at zeros and the
    # run reads as 100% both-feet-up. Re-read the attribute each step.
    twist = env.command_manager.get_term("twist")

    obs, _ = venv.reset()
    start = robot.data.root_link_pos_w[:, :2].clone()

    hist = np.zeros(3, dtype=np.int64)      # 0,1,2 feet in contact
    lf = []                                    # left foot contact per step
    rf = []
    for _ in range(args.steps):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = -args.command  # forward is body -y (face side)
        twist.command[:] = cmd
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = venv.step(actions)
        c = env.scene["feet_ground_contact"].data.found[:]  # live read, see NOTE above
        n_down = (c > 0.5).sum(1)             # per env 0/1/2
        for k in (0, 1, 2):
            hist[k] += int((n_down == k).sum())
        lf.append((c[:, 0] > 0.5).float().mean().item())
        rf.append((c[:, 1] > 0.5).float().mean().item())

    end = robot.data.root_link_pos_w[:, :2]
    disp = (end - start).norm(dim=1)
    elapsed = args.steps * env.step_dt
    lf = np.array(lf)
    rf = np.array(rf)
    total = hist.sum()
    both_down = hist[2] / total
    one_down = hist[1] / total
    both_up = hist[0] / total
    # alternation: correlation of the two feet's contact traces, low = out of phase
    alt = float(np.corrcoef(lf, rf)[0, 1]) if lf.std() > 1e-6 and rf.std() > 1e-6 else float("nan")

    print(f"checkpoint     : {os.path.basename(args.checkpoint)}")
    print(f"command        : {args.command:.2f} m/s, {args.steps} steps, {args.envs} envs")
    print(f"net speed      : {disp.mean().item()/elapsed:.3f} m/s")
    print()
    print("contact distribution over all (env, step) samples:")
    print(f"  both feet down (standing/slide): {both_down*100:5.1f} %")
    print(f"  ONE foot down  (walking)       : {one_down*100:5.1f} %")
    print(f"  both feet up   (hopping/flying): {both_up*100:5.1f} %")
    print()
    print(f"foot duty factor: left {lf.mean():.2f}  right {rf.mean():.2f}")
    print(f"foot contact correlation: {alt:+.2f}  (near -1 = perfectly alternating = walking)")
    print()
    if one_down > 0.3 and (alt < -0.3 or np.isnan(alt)):
        print("verdict        : WALKING pattern (single-foot support, feet alternate)")
    elif both_up > 0.25:
        print("verdict        : HOPPING (both feet leave the ground together)")
    elif both_down > 0.6:
        print("verdict        : SHUFFLING / standing (weight stays on both feet)")
    else:
        print("verdict        : mixed, not a clean walk")


if __name__ == "__main__":
    main()
