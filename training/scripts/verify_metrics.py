"""Metric-only walk probe: reuse the env cfg but NO render context (headless GL
is broken on this host). Measures whether the policy WALKS LIKE A HUMAN, not just
moves. A human gait has a recognizable signature:

  1. Upright torso, sustained (no walk-then-fall, no wild sway).
  2. Alternating single-support with brief double-support (not both feet sliding,
     not hopping). Stance transfers left<->right at a human-like cadence.
  3. KNEE BENDS DURING STANCE -- humans flex the support knee to absorb load;
     a straight-leg shuffle is robot-like and rejected.
  4. Sagittal leg swing -- hips flex/extend fore-aft, not just splay sideways.
  5. Smooth forward progression at the COMMANDED speed (signed, along heading),
     with stable velocity (no lurching / trembling).
  6. Left/right symmetry (mirror_loss should enforce this).

Exit 0 = WALKS (human-like), 1 = no.
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
DUTY_FLOOR = 0.20          # a foot "loaded" if contact fraction > this
FRAME_ALT_CEIL = -0.20     # proj_gravity_b z < -0.20  => upright (z=-1 upright)
SPEED_TOL = 0.05           # min forward speed (signed, along heading)
MIN_VALID_FRAC = 0.60       # >=60% of envs must pass
MIN_SWITCH_HZ = 0.40       # stance must transfer L<->R at least this often
MIN_KNEE_FLEX_RAD = 0.35    # support/ swing knee must actually bend
MIN_HIP_SWING_RAD = 0.20    # legs must swing fore-aft
MIN_DOUBLE_SUPPORT = 0.10   # some double-support (human ~0.2-0.4); pure hop fails
MAX_DOUBLE_SUPPORT = 0.85   # but not mostly both-feet-sliding
MIN_STANCE_KNEE_BEND = 0.20 # knee bent WHILE that foot is the support foot
MAX_SPEED_CV = 0.60         # velocity stability: std/mean of forward speed

KNEE_JOINTS = ("L_knee_test", "R_knee_test")
HIP_JOINTS = ("L_hip_test", "R_hip_test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--command", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = args.envs
    agent_cfg = load_rl_cfg(TASK)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    venv = RslRlVecEnvWrapper(env)
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(venv, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    robot = env.scene["robot"]
    robot.data.forward_vec_b[:] = 0.0
    robot.data.forward_vec_b[:, 1] = -1.0  # facing -y on this model
    twist = env.command_manager.get_term("twist")
    step_dt = env.step_dt
    obs, _ = venv.reset()
    for _ in range(50):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = -args.command
        twist.command[:] = cmd
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))

    joint_cols = {n: robot.joint_names.index(n) for n in KNEE_JOINTS + HIP_JOINTS}
    knee_cols = [joint_cols[n] for n in KNEE_JOINTS]
    hip_cols = [joint_cols[n] for n in HIP_JOINTS]

    P, H, C, U, K, A = [], [], [], [], [], []
    for _ in range(args.steps):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = -args.command
        twist.command[:] = cmd
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))
        P.append(robot.data.root_link_pos_w[:, :2].clone())
        H.append(robot.data.heading_w.clone())
        C.append(env.scene["feet_ground_contact"].data.found[:].clone())
        U.append(robot.data.projected_gravity_b[:, 2].clone())
        jp = robot.data.joint_pos
        K.append(jp[:, knee_cols].clone())
        A.append(jp[:, hip_cols].clone())
    P, H, C = torch.stack(P), torch.stack(H), torch.stack(C)
    U, K, A = torch.stack(U), torch.stack(K), torch.stack(A)  # [steps, envs, legs]

    # --- forward travel along each env's own heading (signed) ---
    d = P[1:] - P[:-1]
    h = H[:-1]
    fwd = torch.stack([torch.cos(h), torch.sin(h)], -1)
    travel = (d * fwd).sum(-1).sum(0)
    dur = (P.shape[0] - 1) * step_dt
    speed = travel / dur  # [envs]

    # --- contact duty per foot ---
    found = C.squeeze(-1) if C.ndim == 4 else C  # [steps, envs, slots]
    left = found[:, :, 0]
    right = found[:, :, 1]
    left_duty = left.float().mean(0)
    right_duty = right.float().mean(0)
    contact_alt = (left_duty - right_duty).abs()

    # --- upright per step (z=-1 upright) ---
    up = U  # [steps, envs]
    last_frame_alt = up[-args.steps // 5:].mean(0) if args.steps // 5 > 0 else up.mean(0)

    # --- sustained upright rolling window (no walk-then-fall) ---
    WINDOW = max(10, args.steps // 5)
    upright_step = up < FRAME_ALT_CEIL
    n_steps = upright_step.shape[0]
    rolling_ok = torch.zeros(n_steps, args.envs, dtype=torch.bool, device=up.device)
    for s in range(n_steps):
        lo = max(0, s - WINDOW + 1)
        rolling_ok[s] = upright_step[lo:s + 1].all(dim=0)
    first_fail = torch.full((args.envs,), -1, dtype=torch.long)
    for e in range(args.envs):
        fails = (rolling_ok[:, e] == 0).nonzero(as_tuple=True)[0].cpu()
        if fails.numel() > 0:
            first_fail[e] = fails[0].item()

    # --- both feet share load ON AVERAGE (alternating contact ok) ---
    both_load_avg = ((left > DUTY_FLOOR).float().mean(0) > 0.5) & \
                   ((right > DUTY_FLOOR).float().mean(0) > 0.5)

    # --- stance transfer + double-support fraction ---
    stance = torch.where(left > 0, 0, torch.where(right > 0, 1, -1))
    switched = (stance[1:] != stance[:-1]) & (stance[1:] >= 0) & (stance[:-1] >= 0)
    switch_hz = switched.float().sum(0) / dur
    double_support = ((left > 0) & (right > 0)).float().mean(0)  # fraction of steps
    single_support_frac = ((left > 0) ^ (right > 0)).float().mean(0)

    # --- joint articulation: peak-to-peak over time, worse of two legs ---
    knee_flex = (K.amax(0) - K.amin(0)).min(1).values      # [envs]
    hip_swing = (A.amax(0) - A.amin(0)).min(1).values      # [envs]
    articulated = (knee_flex > MIN_KNEE_FLEX_RAD) & (hip_swing > MIN_HIP_SWING_RAD)

    # --- knee bent DURING stance (support knee flexes, human-like) ---
    # For each foot, average its knee angle while THAT foot is the support foot.
    l_knee = K[:, :, 0]; r_knee = K[:, :, 1]
    l_stance_knee = l_knee[left > 0].mean() if (left > 0).any() else torch.tensor(0.0)
    r_stance_knee = r_knee[right > 0].mean() if (right > 0).any() else torch.tensor(0.0)
    stance_knee_bend = min(l_stance_knee.item(), r_stance_knee.item())
    stance_knee_ok = stance_knee_bend > MIN_STANCE_KNEE_BEND

    # --- velocity smoothness (no lurching/trembling) ---
    speed_cv = speed.std().item() / (abs(speed.mean().item()) + 1e-6)

    # --- HUMAN-LIKE GATE: all must hold ---
    passed = (rolling_ok[-1] & both_load_avg & (speed > SPEED_TOL)
              & (switch_hz > MIN_SWITCH_HZ)
              & (double_support > MIN_DOUBLE_SUPPORT) & (double_support < MAX_DOUBLE_SUPPORT)
              & articulated
              & torch.tensor(stance_knee_ok)
              & (speed_cv < MAX_SPEED_CV))
    n_pass = int(passed.sum().item())
    valid_frac = n_pass / max(1, args.envs)

    if args.envs == 1:
        # cv needs a batch; for single-env use per-step speed std instead
        step_speed = (d * fwd).sum(-1)  # [steps-1]
        cv1 = step_speed.std().item() / (abs(step_speed.mean().item()) + 1e-6)
        speed_cv = cv1

    res = {
        "verdict": "WALKS" if valid_frac >= MIN_VALID_FRAC else "NO-WALK",
        "valid_frac": round(valid_frac, 3),
        "rolling_window": WINDOW,
        "rolling_pass_frac": round(rolling_ok[-1].float().mean().item(), 3),
        "first_fail_step": [int(x) for x in first_fail.tolist()],
        "left_duty_mean": round(left_duty.mean().item(), 3),
        "right_duty_mean": round(right_duty.mean().item(), 3),
        "speed_mean": round(speed.mean().item(), 3),
        "last_frame_alt_mean": round(last_frame_alt.mean().item(), 3),
        "contact_alt_mean": round(contact_alt.mean().item(), 3),
        "switch_hz_mean": round(float(switch_hz.mean().item()), 3),
        "double_support_frac": round(float(double_support.mean().item()), 3),
        "single_support_frac": round(float(single_support_frac.mean().item()), 3),
        "knee_flex_min_rad": round(float(knee_flex.mean().item()), 3),
        "hip_swing_min_rad": round(float(hip_swing.mean().item()), 3),
        "stance_knee_bend_rad": round(stance_knee_bend, 3),
        "speed_cv": round(float(speed_cv), 3),
        "articulated_frac": round(float(articulated.float().mean().item()), 3),
        "per_env": {
            "left_duty": [round(x, 3) for x in left_duty.tolist()],
            "right_duty": [round(x, 3) for x in right_duty.tolist()],
            "speed": [round(x, 3) for x in speed.tolist()],
            "switch_hz": [round(x, 3) for x in switch_hz.tolist()],
            "knee_flex_rad": [round(x, 3) for x in knee_flex.tolist()],
            "hip_swing_rad": [round(x, 3) for x in hip_swing.tolist()],
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
