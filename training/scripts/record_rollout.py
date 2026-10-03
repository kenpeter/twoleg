"""Rollout recorder: render a real video of a policy + dump a per-step
duty/altitude CSV, so we can SEE how the robot falls before changing rewards.

Unlike verify_walk.py (which deadlocks on headless GL) this renders the same
way training does -- under xvfb-run with MUJOCO_GL=glfw. Run it wrapped:

  xvfb-run -a uv run --no-sync python scripts/record_rollout.py \
      --checkpoint <pt> --out /tmp/opencode/rollout.mp4
"""
import argparse, csv, json, os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="/tmp/opencode/rollout.mp4")
    ap.add_argument("--command", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--envs", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = args.envs
    agent_cfg = load_rl_cfg(TASK)

    # Render context (glfw) -- needs DISPLAY, so run under xvfb-run.
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device,
                            render_mode="rgb_array")
    venv = RslRlVecEnvWrapper(env)
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(venv, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    robot = env.scene["robot"]
    robot.data.forward_vec_b[:] = 0.0
    robot.data.forward_vec_b[:, 1] = -1.0
    twist = env.command_manager.get_term("twist")
    step_dt = env.step_dt
    obs, _ = venv.reset()
    for _ in range(50):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = -args.command
        twist.command[:] = cmd
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))

    import imageio.v2 as imageio
    writer = imageio.get_writer(args.out, fps=int(1 / step_dt),
                                 macro_block_size=1)
    rows = []
    for s in range(args.steps):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = -args.command
        twist.command[:] = cmd
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))
        img = env.render()
        frame = img[0] if img.ndim == 4 else img
        writer.append_data(frame)
        found = env.scene["feet_ground_contact"].data.found[:]
        found = found.squeeze(-1) if found.ndim == 4 else found  # [envs, slots]
        up = robot.data.projected_gravity_b[:, 2]  # [envs]
        # env 0 only (single-env record)
        rows.append({
            "step": s,
            "left_contact": float(found[0, 0].item()),
            "right_contact": float(found[0, 1].item()),
            "upright": float(up[0].item()),
            "torso_z": float(robot.data.root_link_pos_w[0, 2].item()),
        })
    writer.close()

    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "left_contact",
                                            "right_contact", "upright", "torso_z"])
        w.writeheader()
        w.writerows(rows)
    # first-fail (fall) step
    first_fail = next((r["step"] for r in rows
                       if r["upright"] <= FRAME_ALT_CEIL), None)
    print(json.dumps({
        "video": args.out, "csv": csv_path,
        "first_fall_step": first_fail,
        "torso_z_start": rows[0]["torso_z"],
        "torso_z_end": rows[-1]["torso_z"],
        "upright_end": rows[-1]["upright"],
    }, indent=2))


if __name__ == "__main__":
    main()
