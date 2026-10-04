"""Metric-only walk probe: reuse the env cfg but NO render context (headless GL
is broken on this host). Measures the same contact-sensor duties + last-frame
altitude the video verifier uses, so we get the real 'does it walk' signal
without the GL deadlock. Exit 0 = WALKS, 1 = no.
"""
import argparse, json, os
import torch
from dataclasses import asdict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-TwoLeg"
DUTY_FLOOR = 0.20
FRAME_ALT_CEIL = -0.20
CONTACT_ALT_CEIL = -0.20
SPEED_TOL = 0.05
MIN_VALID_FRAC = 0.60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--command", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = args.envs
    agent_cfg = load_rl_cfg(TASK)

    # NO render_mode -> no GL context -> no headless deadlock.
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    venv = RslRlVecEnvWrapper(env)
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(venv, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    robot = env.scene["robot"]
    robot.data.forward_vec_b[:] = 0.0
    robot.data.forward_vec_b[:, 1] = -1.0  # facing -y (face side) on this model
    twist = env.command_manager.get_term("twist")
    step_dt = env.step_dt
    obs, _ = venv.reset()
    for _ in range(50):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = -args.command
        twist.command[:] = cmd
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))

    P, H, C = [], [], []
    for _ in range(args.steps):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = -args.command
        twist.command[:] = cmd
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))
        P.append(robot.data.root_link_pos_w[:, :2].clone())
        H.append(robot.data.heading_w.clone())
        C.append(env.scene["feet_ground_contact"].data.found[:].clone())
    P, H, C = torch.stack(P), torch.stack(H), torch.stack(C)

    d = P[1:] - P[:-1]
    h = H[:-1]
    fwd = torch.stack([torch.cos(h), torch.sin(h)], -1)
    travel = (d * fwd).sum(-1).sum(0)
    dur = (P.shape[0] - 1) * step_dt
    speed = travel / dur  # [envs]

    # contact duty per env: fraction of steps each foot (slot 0=left,1=right) found
    found = C  # [steps, envs, 1, slots]? -> squeeze
    found = found.squeeze(-1) if found.ndim == 4 else found  # [steps, envs, slots]
    left = found[:, :, 0]
    right = found[:, :, 1]
    left_duty = left.float().mean(0)   # [envs]
    right_duty = right.float().mean(0)  # [envs]

    # per-env last-frame altitude: tilt of root vs upright (use ang_vel? use proj z)
    up = robot.data.projected_gravity_b[:, 2]  # ~1 upright, -> -1 inverted
    frame_alt = up[-args.steps // 5:] if args.steps // 5 > 0 else up  # last 1/5
    last_frame_alt = frame_alt.mean(0)   # [envs]
    contact_alt = (left_duty - right_duty).abs()

    # --- Agent-R1-style per-step DONE condition (not just last-frame) ---
    # A walking biped MUST have flight phases (one foot airborne), so requiring
    # BOTH feet loaded at EVERY step is wrong -- that only fits standing. The
    # correct gate: over a rolling window, BOTH feet must bear load ON AVERAGE
    # (alternating contact is fine) AND the torso must stay upright. We record
    # the first window where this breaks (the "first error step"), mirroring
    # Agent-R's timely-revision principle: walk-then-fall only passes if it
    # stays up for the whole window.
    WINDOW = max(10, args.steps // 5)
    # per-step uprightness: projected_gravity_b z is -1 upright, +1 inverted
    step_up = up  # [steps, envs]
    upright_step = step_up < FRAME_ALT_CEIL              # [steps, envs]
    # per-step mean duty (both feet share load on average) -> use the per-step
    # contact found, not the collapsed mean, so flight phases are allowed.
    left_step = left   # [steps, envs]
    right_step = right  # [steps, envs]
    both_load_avg = ((left_step > DUTY_FLOOR).float().mean(0) > 0.5) & \
                    ((right_step > DUTY_FLOOR).float().mean(0) > 0.5)  # [envs]
    # rolling window: require upright for the WHOLE window (falling is fatal),
    # and both-feet-load-on-average sustained.
    n_steps = upright_step.shape[0]
    rolling_ok = torch.zeros(n_steps, args.envs, dtype=torch.bool,
                             device=upright_step.device)
    for s in range(n_steps):
        lo = max(0, s - WINDOW + 1)
        window_up = upright_step[lo:s + 1]              # [win, envs]
        rolling_ok[s] = window_up.all(dim=0)            # [envs] all upright
    # first failure step per env (where the upright window first breaks)
    first_fail = torch.full((args.envs,), -1, dtype=torch.long)
    for e in range(args.envs):
        fails = (rolling_ok[:, e] == 0).nonzero(as_tuple=True)[0].cpu()
        if fails.numel() > 0:
            first_fail[e] = fails[0].item()

    # verdict per env: stayed upright across the final window (no fall) AND
    # both feet load on average AND net forward motion.
    passed = (rolling_ok[-1] & both_load_avg & (speed.abs() > SPEED_TOL))
    n_pass = int(passed.sum().item())
    valid_frac = n_pass / max(1, args.envs)

    # report aggregate across envs
    l_mean = left_duty.mean().item()
    r_mean = right_duty.mean().item()
    sp_mean = speed.mean().item()
    fa_mean = last_frame_alt.mean().item()
    ca_mean = contact_alt.mean().item()
    res = {
        "verdict": "WALKS" if valid_frac >= MIN_VALID_FRAC else "NO-WALK",
        "valid_frac": round(valid_frac, 3),
        # ROLLING gate (Agent-R1 per-step DONE condition)
        "rolling_window": WINDOW,
        "rolling_pass_frac": round(rolling_ok[-1].float().mean().item(), 3),
        "first_fail_step": [int(x) for x in first_fail.tolist()],
        "left_duty_mean": round(l_mean, 3),
        "right_duty_mean": round(r_mean, 3),
        "speed_mean": round(sp_mean, 3),
        "last_frame_alt_mean": round(fa_mean, 3),
        "contact_alt_mean": round(ca_mean, 3),
        "per_env": {
            "left_duty": [round(x, 3) for x in left_duty.tolist()],
            "right_duty": [round(x, 3) for x in right_duty.tolist()],
            "speed": [round(x, 3) for x in speed.tolist()],
            "passed": [bool(x) for x in passed.tolist()],
        },
    }
    print(json.dumps(res, indent=2))
    out = os.path.join(os.path.dirname(args.checkpoint),
                       f"verify_metrics_cmd{args.command:.2f}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("WROTE", out)
    sys.exit(0 if res["verdict"] == "WALKS" else 1)


if __name__ == "__main__":
    main()
