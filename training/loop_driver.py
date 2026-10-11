"""TwoLeg agentic training loop (Hermes agent 2, simple version).

OUTER loop: repeat inner cycles until WALKS (eval = numbers + last 5s video).
INNER loop (one ~30-min cycle):
  1. train  (~22 min, resume from newest ckpt, fixed iter budget)
  2. record last 5s -> latest_walk.mp4
  3. loop_eval.py -> verdict (numbers + last-5s motion signal)
  4. if NO-WALK: ONE small reversible edit to env_cfgs.py rewards, based on the
     failing metric; persist the change to loop_log.jsonl
  5. repeat

The agent (this script) IS the worker + evaluator. Coordination spec is persisted
to loop_spec.json so the "team" improves across cycles (MASS-lite: we evolve the
workflow, not fine-tune a model).

Usage:
  python loop_driver.py --cycle-min 30 --iters 55 --exp fresh500 --num-envs 2048
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path("/home/kenpeter/work/twoleg/training")
REPO = Path("/home/kenpeter/work/gh-repohub/unitree_rl_mjlab")
LOGS = ROOT / "logs" / "rsl_rl" / "fresh500"
SPEC = ROOT / "loop_spec.json"
LOGL = ROOT / "loop_log.jsonl"

ENV = dict(os.environ)
ENV.update({"MUJOCO_GL": "egl", "WARP_USE_LEGACY_CODEGEN": "1",
            "WANDB_MODE": "disabled",
            "PYTHONPATH": f"{REPO}:{ROOT}"})


def newest_ckpt() -> Path:
    pts = sorted(LOGS.glob("*/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0] if pts else None


def run(cmd, timeout_s, name):
    print(f"\n### [{name}] {' '.join(cmd[-6:])}")
    p = subprocess.run(cmd, cwd=str(ROOT), env=ENV, timeout=timeout_s,
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(f"!!! [{name}] rc={p.returncode}\n{p.stderr[-800:]}")
    return p


def train(iters, num_envs, exp):
    ck = newest_ckpt()
    load_run = ck.parent.name if ck else ".*"
    load_ck = ck.name if ck else "model_.*.pt"
    cmd = [f"{ROOT}/.venv/bin/python", "train_twoleg.py", "TwoLeg-Velocity-Flat",
           "--agent.experiment-name", exp, "--agent.resume", "True",
           "--agent.load-run", load_run, "--agent.load-checkpoint", load_ck,
           "--agent.max-iterations", str(iters), "--env.scene.num-envs", str(num_envs)]
    return run(cmd, iters * 90 + 600, "train")


def record():
    ck = newest_ckpt()
    cmd = [f"{ROOT}/.venv/bin/python", "record_clip.py", "--secs", "5.0",
           "--out", str(ROOT / "latest_walk.mp4")]
    if ck is not None:
        cmd += ["--ckpt", str(ck)]
    return run(cmd, 900, "record")


def evaluate(ckpt):
    cmd = [f"{ROOT}/.venv/bin/python", "loop_eval.py", "--ckpt", str(ckpt),
           "--device", "cuda:0", "--out-json", str(ROOT / "loop_eval.json")]
    p = run(cmd, 600, "eval")
    try:
        return json.loads(Path(ROOT / "loop_eval.json").read_text())
    except Exception:
        return {"verdict": "NO-WALK", "numbers": {}, "video": {}}


def reflect_and_edit(eval_out: dict) -> str:
    """One small reversible reward edit keyed on the failing metric. Returns a note."""
    n = eval_out.get("numbers", {})
    spec = json.loads(SPEC.read_text()) if SPEC.exists() else {}
    edits = spec.get("edits", {})
    # pick the lever with the most headroom
    if n.get("torso_low_frac", 0) > 0.1:
        target, delta = "torso_height", -1.0
        note = "torso dropping / falling -> strengthen torso_height anti-fall"
    elif n.get("feet_collide_frac", 0) > 0.3:
        target, delta = "foot_separation", -1.0
        note = "feet still colliding -> strengthen foot_separation (widen stance)"
    elif n.get("foot_sep_y", 1) < 0.10:
        target, delta = "foot_separation", -0.5
        note = "feet too close laterally -> strengthen foot_separation"
    elif n.get("speed", 1) < 0.05 and n.get("double_support", 0) > 0.85:
        # only nudge air-time a little; large boosts re-collapse the feet into a
        # triangle (in-place hop). Cap the increment.
        target, delta = "feet_air_time", +0.3
        note = "no forward motion -> small feet_air_time nudge (capped to avoid triangle)"
    elif n.get("upright_frac", 1) < 0.5:
        target, delta = "body_orientation_l2", -1.0
        note = "not upright -> strengthen body_orientation_l2"
    elif n.get("alternation", 1) < 0.3:
        target, delta = "foot_gait", +0.15
        note = "no gait alternation -> strengthen foot_gait phase reward"
    elif n.get("knee_flex", 1) < 0.2:
        target, delta = "knee_flexion", +0.5
        note = "no knee flex -> strengthen knee_flexion reward"
    else:
        target, delta = "track_lin_vel", +0.5
        note = "soft fail -> nudge track_lin_vel"
    # apply via a tiny sed-free python patch on env_cfgs.py
    apply_edit(target, delta)
    edits[target] = edits.get(target, 0) + (delta if isinstance(delta, (int, float)) else 1)
    spec["edits"] = edits
    SPEC.write_text(json.dumps(spec, indent=2))
    return note


def apply_edit(target, delta):
    """Minimal edit of the reward weight for `target` in env_cfgs.py."""
    f = ROOT / "env_cfgs.py"
    src = f.read_text()
    import re
    # Match the target reward block's own `weight=NUMBER` (first weight= after the
    # key, non-greedy so we don't touch a nested param). group(1)=prefix, group(2)=num.
    pat = re.compile(
        r'(cfg\.rewards\["' + re.escape(target) + r'"\] = RewardTermCfg\(.*?weight\s*=\s*)([-0-9.eE+]+)',
        re.S)
    m = pat.search(src)
    if not m:
        return
    old_w = float(m.group(2))
    new_w = old_w + delta if isinstance(delta, (int, float)) and delta != 0 else old_w
    new_src = src[:m.start(2)] + repr(new_w) + src[m.end(2):]
    f.write_text(new_src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle-min", type=int, default=30)
    ap.add_argument("--iters", type=int, default=55)
    ap.add_argument("--exp", default="fresh500")
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--max-cycles", type=int, default=999)
    args = ap.parse_args()

    cycle = 0
    while cycle < args.max_cycles:
        cycle += 1
        t0 = time.time()
        print(f"\n========== CYCLE {cycle} ==========")
        train(args.iters, args.num_envs, args.exp)
        ck = newest_ckpt()
        if ck is None:
            print("NO CHECKPOINT - abort"); break
        record()
        ev = evaluate(ck)
        verdict = ev.get("verdict")
        print(f"CYCLE {cycle} verdict={verdict} numbers={ev.get('numbers')}")
        LOGL.open("a").write(json.dumps({"cycle": cycle, "ckpt": str(ck),
                                           "verdict": verdict, "eval": ev}) + "\n")
        if verdict == "WALKS":
            print(f"*** WALKS at cycle {cycle}, ckpt {ck} — loop stops. ***")
            # send montage to user
            montage = ev.get("video", {}).get("montage")
            print(f"MONTAGE:{montage}")
            break
        note = reflect_and_edit(ev)
        print(f"REFLECT: {note}; next cycle resumes from {ck}")
        # safety: if a cycle ran way over budget, still continue (non-stop)
        elapsed = (time.time() - t0) / 60
        print(f"cycle {cycle} took {elapsed:.1f} min")


if __name__ == "__main__":
    main()
