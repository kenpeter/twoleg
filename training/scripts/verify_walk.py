"""Video-watching gait verifier: does the policy walk slowly on two legs?

Renders a rollout at a held forward command, then judges what is visible in
the frames, not what the reward says. Feet render pink, so each frame is split
into a left and a right foot cluster by pink-pixel mass and the two vertical
traces are correlated. Out of phase means stepping. In phase means hopping.

Contact-sensor duties are measured on the same rollout as a cross-check, but
the verdict is driven by the picture. Inconclusive counts as FAIL.

    uv run python scripts/verify_walk.py --checkpoint <path/to/model.pt>
    uv run python scripts/verify_walk.py --checkpoint <pt> --command 0.1 --self-test

Needs MUJOCO_GL=glfw on this host (egl and osmesa both fail here).
Exit code 0 = WALKS, 1 = does not walk, 2 = error. A JSON sidecar lands next
to the video for loop runners.
"""

import argparse
import csv
import json
import os
import statistics as st
import sys
from dataclasses import asdict

import numpy as np
import torch

TASK = "Mjlab-Velocity-Flat-TwoLeg"

# PASS predicate. min-duty kills the timing-only false positive seen at H8
# (alt -0.39 with the left foot down 8 pct of steps).
DUTY_FLOOR = 0.20
FRAME_ALT_CEIL = -0.20
CONTACT_ALT_CEIL = -0.20
SPEED_TOL = 0.05
MIN_VALID_FRAC = 0.60


def foot_traces(frames):
    """Per-frame vertical centroid of the left and right pink foot clusters.

    Returns two lists (may contain None) plus the fraction of usable frames.
    Pure numpy so the self-test runs without a simulator.
    """
    left, right = [], []
    for img in frames:
        h, w, _ = img.shape
        lower = img[h // 2 :, :, :].astype(np.int32)
        r, g, b = lower[:, :, 0], lower[:, :, 1], lower[:, :, 2]
        mask = (r > 140) & (b > 140) & (g < 130)
        ys, xs = np.nonzero(mask)
        if len(xs) < 80:
            left.append(None)
            right.append(None)
            continue
        xmid = float(np.median(xs))
        l = ys[xs < xmid]
        r_ = ys[xs >= xmid]
        if len(l) < 40 or len(r_) < 40:
            left.append(None)
            right.append(None)
            continue
        left.append(float(np.mean(l)))
        right.append(float(np.mean(r_)))
    return left, right


def alt_corr(a, b):
    """Correlation of two traces over frames where both are present."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 20:
        return float("nan"), 0.0
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    if xs.std() < 1e-6 or ys.std() < 1e-6:
        return float("nan"), len(pairs) / max(len(a), 1)
    return float(np.corrcoef(xs, ys)[0, 1]), len(pairs) / len(a)


def verdict(speed, cmd, lrate, rrate, contact_alt, frame_alt, valid_frac):
    """PASS needs forward motion, both feet sharing the work, and the picture
    showing alternation. Anything inconclusive fails."""
    reasons = []
    if not (abs(speed - cmd) <= SPEED_TOL and speed > 0.03):
        reasons.append(f"speed {speed:+.3f} not within {SPEED_TOL} of {cmd:.2f}")
    if min(lrate, rrate) < DUTY_FLOOR:
        reasons.append(f"min duty {min(lrate, rrate):.2f} below {DUTY_FLOOR}")
    if not (contact_alt == contact_alt and contact_alt <= CONTACT_ALT_CEIL):
        reasons.append(f"contact alt {contact_alt:+.2f} above {CONTACT_ALT_CEIL}")
    if valid_frac < MIN_VALID_FRAC:
        reasons.append(f"usable frames {valid_frac:.0%} below {MIN_VALID_FRAC:.0%}")
    elif not (frame_alt == frame_alt and frame_alt <= FRAME_ALT_CEIL):
        reasons.append(f"frame alt {frame_alt:+.2f} above {FRAME_ALT_CEIL}")
    return (not reasons), reasons


def self_test():
    """Synthetic pink-blob frames: alternating must read ~-1, together ~+1."""
    h, w = 64, 64
    n = 60

    def frame(ly, ry):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[0:5, 0:5] = (229, 51, 229)
        img[int(ly) : int(ly) + 6, 10:20] = (229, 51, 229)
        img[int(ry) : int(ry) + 6, 44:54] = (229, 51, 229)
        return img

    t = np.arange(n)
    alt_frames = [frame(40 + 8 * np.sin(2 * np.pi * i / 20),
                         40 + 8 * np.sin(2 * np.pi * i / 20 + np.pi)) for i in t]
    hop_frames = [frame(40 + 8 * np.sin(2 * np.pi * i / 20),
                         40 + 8 * np.sin(2 * np.pi * i / 20)) for i in t]
    la, ra = foot_traces(alt_frames)
    lh, rh = foot_traces(hop_frames)
    ca, _ = alt_corr(la, ra)
    ch, _ = alt_corr(lh, rh)
    print(f"self-test alternating frames: alt {ca:+.3f} (want near -1)")
    print(f"self-test together frames    : alt {ch:+.3f} (want near +1)")
    ok = ca < -0.8 and ch > 0.8
    print("self-test:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--command", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)

    import imageio.v2 as imageio

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = args.envs
    agent_cfg = load_rl_cfg(TASK)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
    venv = RslRlVecEnvWrapper(env)
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(venv, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    robot = env.scene["robot"]
    # This model's forward is body -y (face side). mjlab hardcodes forward_vec_b to
    # body+x, so point it at the real forward axis; otherwise the travel
    # metric scores side-sway like a side-stepping gait.
    robot.data.forward_vec_b[:] = 0.0
    robot.data.forward_vec_b[:, 1] = -1.0
    twist = env.command_manager.get_term("twist")
    step_dt = env.step_dt
    obs, _ = venv.reset()
    for _ in range(50):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = -args.command  # forward is body -y (face side)
        twist.command[:] = cmd
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))

    out = args.out or os.path.join(
        os.path.dirname(args.checkpoint), f"verify_cmd{args.command:.2f}.mp4")
    writer = imageio.get_writer(out, fps=int(1 / step_dt), macro_block_size=1)
    frames, P, H, C = [], [], [], []
    for _ in range(args.steps):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = -args.command  # forward is body -y (face side)
        twist.command[:] = cmd
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))
        img = env.render()
        frame = img[0] if img.ndim == 4 else img
        writer.append_data(frame)
        frames.append(frame)
        P.append(robot.data.root_link_pos_w[:, :2].clone())
        H.append(robot.data.heading_w.clone())
        C.append(env.scene["feet_ground_contact"].data.found[:].clone())
    writer.close()
    P, H, C = torch.stack(P), torch.stack(H), torch.stack(C)

    d = P[1:] - P[:-1]
    h = H[:-1]
    fwd = torch.stack([torch.cos(h), torch.sin(h)], -1)
    travel = (d * fwd).sum(-1).sum(0)
    dur = (P.shape[0] - 1) * step_dt
    speed = (travel / dur).mean().item()

    left = (C[..., 0] > 0.5)
    right = (C[..., 1] > 0.5)
    lrate = left.float().mean().item()
    rrate = right.float().mean().item()
    lf = left.float().mean(1).cpu().numpy()
    rf = right.float().mean(1).cpu().numpy()
    contact_alt = (float(np.corrcoef(lf, rf)[0, 1])
                   if lf.std() > 1e-6 and rf.std() > 1e-6 else float("nan"))

    ly, ry = foot_traces(frames)
    frame_alt, valid_frac = alt_corr(ly, ry)

    # ---- LAST-FRAMES CHECK (end-state grounding) ----
    # The loop must not pass on a mean that hides a late collapse (or a late
    # recovery). Require the final N frames to be upright (frame_alt ok) AND
    # both feet sharing load (duty floor). This is what a human sees in the
    # last frames of the video.
    LAST_N = max(20, args.steps // 5)
    lf_last = lf[-LAST_N:]
    rf_last = rf[-LAST_N:]
    lrate_last = float(np.mean(lf_last)) if len(lf_last) else 0.0
    rrate_last = float(np.mean(rf_last)) if len(rf_last) else 0.0
    ly_last = ly[-LAST_N:]; ry_last = ry[-LAST_N:]
    frame_alt_last, valid_frac_last = alt_corr(ly_last, ry_last)
    last_ok = (
        min(lrate_last, rrate_last) >= DUTY_FLOOR
        and frame_alt_last == frame_alt_last
        and frame_alt_last <= FRAME_ALT_CEIL
    )

    passed, reasons = verdict(speed, args.command, lrate, rrate,
                              contact_alt, frame_alt, valid_frac)
    if not last_ok:
        reasons.append(
            f"last{LAST_N} frames: min duty {min(lrate_last, rrate_last):.2f} "
            f"below {DUTY_FLOOR} or frame alt {frame_alt_last:+.2f} above {FRAME_ALT_CEIL} "
            f"(end-state not walking)")
    sidecar = {"verdict": "WALKS" if passed else "NO-WALK",
               "reasons": reasons, "speed": round(speed, 4),
               "command": args.command, "left_duty": round(lrate, 3),
               "right_duty": round(rrate, 3),
               "contact_alt": round(contact_alt, 3),
               "frame_alt": round(frame_alt, 3),
               "valid_frames": round(valid_frac, 3),
               "last_ok": bool(last_ok),
               "last_min_duty": round(min(lrate_last, rrate_last), 3),
               "last_frame_alt": round(frame_alt_last, 3),
               "video": out, "checkpoint": args.checkpoint}
    with open(os.path.splitext(out)[0] + ".json", "w") as fh:
        json.dump(sidecar, fh, indent=1)

    print(f"checkpoint : {os.path.basename(args.checkpoint)}")
    print(f"command    : {args.command:.2f} m/s, {args.steps} steps, {args.envs} envs")
    print(f"speed      : {speed:+.3f} m/s along own axis")
    print(f"duty       : left {lrate:.2f}  right {rrate:.2f}")
    print(f"contact alt: {contact_alt:+.2f}   frame alt: {frame_alt:+.2f} "
          f"({valid_frac:.0%} frames usable)")
    print(f"video      : {out}")
    print(f"verdict    : {'WALKS' if passed else 'NO-WALK'}")
    for r in reasons:
        print(f"  because: {r}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
