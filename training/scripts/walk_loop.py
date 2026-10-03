"""Reflective walk loop: train, watch the end of the video, reflect, small change, repeat.

This is the agentic loop for /pp (TwoLeg biped). Each round:
  1. resume-train a bounded iteration budget from the current checkpoint,
  2. render a rollout and run scripts/verify_walk.py (which now judges the
     LAST frames: upright + both feet sharing load = walking),
  3. if WALKS  -> STOP (success),
  4. if NO-WALK -> the REFLECTOR reads the verdict + end-frame pixels, writes
     ONE small reversible change as a new strategy, and the loop repeats.

There is no fixed strategy table and no STRATEGIES_EXHAUSTED stop: the loop
only ends on WALKS or a hard training failure. The reflector is the brain; it
must keep each change small (one reward term, one servo sign, one hyper) so
the search stays local and reversible. State persists in
.audit/walk_loop_state.json so a later /pp call resumes.

    uv run python scripts/walk_loop.py --rounds 1
    uv run python scripts/walk_loop.py --reflect-only --from-run <r> --from-checkpoint <pt>
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
REPO = os.path.dirname(TRAINING)
AUDIT = os.path.join(REPO, ".audit")
STATE = os.path.join(AUDIT, "walk_loop_state.json")
STRAT = os.path.join(AUDIT, "walk_loop_strategies.jsonl")  # append-only reflector log
LOG_TSV = os.path.join(AUDIT, "twoleg-walk.tsv")
LOG_SH = ("/home/kenpeter/work/pp/.opencode/skills/show-me-your-work/scripts/log.sh")
TASK = "Mjlab-Velocity-Flat-TwoLeg"

# Reflection registry lives in code as the FIRST hypothesis; after that the
# reflector appends. Seed with the standing open questions so round 1 has a
# direction if state is fresh.
SEED_STRATEGY = {
    "name": "fresh-symmetry-gate-1000", "kind": "resume-train", "iters": 1000, "envs": 4096,
    "why": "H12: last verdict left_duty 0.0 / right_duty 0.936 -> collapses onto "
           "right leg, left leg does nothing. Hypothesis: leg servos asymmetric "
           "(sign/scale mismatch or unmirrored position target). Add a left/right "
           "duty-balance reward term so both legs must bear load; predict symmetric "
           "stance and stepping. Revert if it flails.",
}


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def log_row(phase, decision, why, evidence, result):
    sh(["bash", LOG_SH, LOG_TSV, phase, decision, why, evidence, result])


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as fh:
            return json.load(fh)
    return {"rounds_done": 0, "run": None, "checkpoint": None,
            "last_verdict": None, "last_video": None}


def save_state(s):
    os.makedirs(AUDIT, exist_ok=True)
    with open(STATE, "w") as fh:
        json.dump(s, fh, indent=1)


def append_strategy(st):
    os.makedirs(AUDIT, exist_ok=True)
    with open(STRAT, "a") as fh:
        fh.write(json.dumps(st) + "\n")


def run_dirs():
    root = os.path.join(TRAINING, "logs", "rsl_rl", "velocity")
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))


def newest_model(run):
    d = os.path.join(TRAINING, "logs", "rsl_rl", "velocity", run)
    if not os.path.isdir(d):
        return None
    pts = sorted(f for f in os.listdir(d)
                 if f.startswith("model_") and f.endswith(".pt"))
    if not pts:
        return None
    return max(pts, key=lambda f: os.path.getmtime(os.path.join(d, f)))


def train_round(strategy, run, ckpt, fresh=False):
    """Launch a bounded training run detached, poll until it exits. Returns the
    new run dir, or None on failure.

    fresh=True  -> train from random init (NO --agent.resume / load flags).
    fresh=False -> resume from (run, ckpt). ANY run we launch is always
    recorded in state with its newest checkpoint, so it can be resumed later.
    """
    iters, envs = strategy["iters"], strategy["envs"]
    before = set(run_dirs())
    logf = f"/tmp/opencode/loop_{strategy['name']}.log"
    if os.path.exists(logf):
        os.remove(logf)
    resume_part = ""
    if not fresh:
        resume_part = (f"--agent.resume True --agent.load-run {run} "
                       f"--agent.load-checkpoint {ckpt} ")
    # xvfb-run wraps train: --video True needs a GL context, and this host has
    # no real display. Without it the child dies at renderer init (caught as
    # "no new run dir appeared").
    cmd = (f"setsid nohup xvfb-run -a -s \"-screen 0 800x600x24\" "
           f"uv run --no-sync train {TASK} "
           f"--env.scene.num-envs {envs} --agent.max-iterations {iters} "
           f"{resume_part}--video True "
           f"> {logf} 2>&1 < /dev/null &")
    subprocess.Popen(cmd, shell=True, cwd=TRAINING,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Generous wall-clock budget: ~1.0s/iter on this 150W-capped 4070 Ti plus
    # startup (xvfb+uv+cuda) slack. Never pkill a healthy run that is still
    # making progress -- only give up if the proc actually died. The earlier
    # iters*2.0+600 budget was too tight and killed a resume at iter 3255.
    deadline = time.time() + iters * 1.0 + 1800.0
    launched = False
    while time.time() < deadline:
        time.sleep(20)
        r = sh(["pgrep", "-f", "train Mjlab"])
        alive = bool(r.stdout.strip())
        if alive:
            launched = True
        elif launched:
            break  # proc exited on its own -> run finished (or crashed)
    if time.time() >= deadline and launched:
        # Only reached if the proc is STILL alive past budget. Log but do not
        # kill; let it keep training and pick up the newer checkpoint.
        print(f"train round still running past budget ({iters} iters); "
              f"continuing to poll for checkpoint")
    after = set(run_dirs())
    new = sorted(after - before)
    if not new:
        print("no new run dir appeared; check", logf)
        return None
    print(f"round done, new run dir: {new[-1]}")
    return new[-1]


def verify(run, ckpt):
    """Run the metric verifier (headless, no GL) which reads the same contact
    duties + last-frame altitude the video verifier used. Returns (passed, sidecar).
    Returns (passed, sidecar). Returns a dict with 'verdict' in sidecar."""
    if not ckpt:
        raise RuntimeError(f"verify: no checkpoint in run {run} (training died?)")
    ckpt_path = os.path.join(TRAINING, "logs", "rsl_rl", "velocity", run, ckpt)
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"verify: checkpoint missing: {ckpt_path}")
    env = dict(os.environ)  # no MUJOCO_GL needed (verify_metrics has no render)
    r = subprocess.run(
        ["uv", "run", "--no-sync", "python", "scripts/verify_metrics.py",
         "--checkpoint", ckpt_path, "--command", "0.1",
         "--steps", "250", "--envs", "4", "--device", "cuda:0"],
        cwd=TRAINING, capture_output=True, text=True, env=env, timeout=300)
    print(r.stdout[-1500:] if r.stdout else "")
    if r.stderr:
        print("ERR:", r.stderr[-800:])
    sidecar = {}
    sp = os.path.join(TRAINING, "logs", "rsl_rl", "velocity", run,
                      f"verify_metrics_cmd0.10.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            sidecar = json.load(fh)
    return sidecar.get("verdict") == "WALKS", sidecar


def reflect(state):
    """The brain. Read the last verdict + end-frame pixels, propose ONE small
    reversible change. Returns a strategy dict. Never invents a rewrite."""
    v = state.get("last_verdict") or {}
    reasons = "; ".join(v.get("reasons", [])) or "no verdict"
    left = v.get("left_duty", "?")
    right = v.get("right_duty", "?")
    speed = v.get("speed", "?")
    last_min = v.get("last_min_duty", "?")
    last_alt = v.get("last_frame_alt", "?")
    video = state.get("last_video") or v.get("video") or ""
    round_n = state["rounds_done"] + 1

    prompt = (
        f"You are the reflection brain for a TwoLeg biped RL training loop. "
        f"Round {round_n} FAILED to walk.\n\n"
        f"VERIFIER VERDICT (last-frames aware):\n"
        f"  reasons: {reasons}\n"
        f"  duty left={left} right={right}  speed={speed}\n"
        f"  LAST-frames min duty={last_min}  last frame_alt={last_alt}\n\n"
        f"WATCH THE VIDEO before deciding: {video}\n"
        f"Open it and look at the FINAL frames. Is the robot upright? Are BOTH "
        f"feet on the ground bearing load, or did it collapse onto one leg / fall? "
        f"Describe what the last frames actually show, then explain the most "
        f"likely cause.\n\n"
        f"Then propose EXACTLY ONE small, reversible change to make it walk on "
        f"two legs (no rewrites):\n"
        f" - add/adjust a reward term (duty-balance so both legs share load, "
        f"foot-contact alternation, torso-upright bonus)\n"
        f" - fix a leg servo sign/scale or mirror bug in the reference XML\n"
        f" - adjust one hyper (learning rate, iters, action clip)\n"
        f"Return JSON only: {{\"name\": str, \"kind\": \"resume-train\", \"iters\": 2000, "
        f"\"envs\": 2048, \"why\": str, \"change\": str}}"
    )

    # Use the running opencode instance in poteto mode to reflect + watch the
    # end-frame video (it owns the model/XML context). Pin muse-spark-1.3
    # (free) per direction. If opencode is unavailable, fall back to a
    # duty-balance nudge. NOTE: opencode can hang (keeps pipe open past
    # timeout) so we launch it in its own process group and killpg hard.
    import subprocess as _sp
    try:
        proc = _sp.Popen(
            ["opencode", "run", "--model",
             "opencode/muse-spark-1.3-contributor-free", prompt],
            cwd="/home/kenpeter/work/pp",
            stdout=_sp.PIPE, stderr=_sp.PIPE, text=True,
            start_new_session=True)
        try:
            out, err = proc.communicate(timeout=180)
        except _sp.TimeoutExpired:
            proc.kill(); _sp.run(["pkill", "-9", "-f", "opencode run"],
                                  capture_output=True)
            raise RuntimeError("opencode timed out after 180s")
        txt = (out or "") + (err or "")
        j = txt[txt.find("{"):txt.rfind("}")+1]
        st = json.loads(j)
        st.setdefault("iters", 1000); st.setdefault("envs", 4096)
        st.setdefault("kind", "resume-train")
        return st
    except Exception as e:
        print(f"reflect: opencode unavailable ({e}); fallback duty-balance")
        return {"name": f"duty-balance-{round_n}", "kind": "resume-train",
                "iters": 1000, "envs": 4096,
                "why": f"fallback: still collapsing (last min duty {last_min}). "
                       f"Add duty-balance reward so both legs bear load.",
                "change": "add left/right duty-balance term to reward"}


def decide(state, new_run, new_ckpt):
    """Agentic loop-condition with evidence-first priority:

    1. HARD EVIDENCE (deterministic, from contact sensors) is the PRIMARY gate.
       - STOP  if the measured evidence clearly shows walking: torso upright at
         the end (frame_alt > -0.2), BOTH feet share load (duty > 0.2 each),
         forward motion (speed > 0.05), and the rolling upright window held.
       - CONTINUE if evidence clearly shows failure (inverted/fallen/one-leg/
         collapsed).
    2. AMBIGUOUS ZONE: evidence is decent but not conclusive (e.g. upright but
       one foot under-loaded, or walks then stumbles). Only HERE do we ask the
       subagent judge STOP vs CONTINUE from the verdict + rollout trace.
    3. Subagent unavailable -> fall back to the evidence (CONTINUE unless the
       hard evidence already said STOP).

    The agent never overrides clear evidence; it only breaks ties.
    """
    info = state.get("last_verdict") or {}
    fa = info.get("last_frame_alt_mean", -1.0)
    l_d = info.get("left_duty_mean", 0.0)
    r_d = info.get("right_duty_mean", 0.0)
    sp = info.get("speed_mean", 0.0)
    roll = info.get("rolling_pass_frac", 0.0)

    walks_evidence = (fa > -0.2) and (l_d > 0.2) and (r_d > 0.2) and (sp > 0.05) and (roll > 0.5)
    fails_evidence = (fa < -0.5) or (l_d < 0.2) or (r_d < 0.2) or (roll < 0.5)

    if walks_evidence:
        return "stop", f"HARD EVIDENCE walks: frame_alt={fa} L={l_d} R={r_d} spd={sp} roll={roll}"
    if fails_evidence:
        return "continue", (f"HARD EVIDENCE fails: frame_alt={fa} L={l_d} "
                            f"R={r_d} roll={roll} -> keep training")

    # Ambiguous zone: measured numbers are borderline. Ask the subagent judge,
    # but only as a tie-breaker (it cannot override the hard gates above).
    verdict_json = json.dumps(info, indent=2)
    prompt = (
        f"You are the STOP/CONTINUE tie-breaker for a TwoLeg biped RL loop.\n"
        f"The HARD measured gate was INCONCLUSIVE, so judge from nuance.\n"
        f"MEASURED VERDICT:\n{verdict_json}\n\n"
        f"- STOP only if at the LAST frames the torso is upright (frame_alt near 0) "
        f"AND both feet share load (duty > 0.2 each) AND it moves forward.\n"
        f"- CONTINUE if it falls, is inverted, hops one-legged, or walks then "
        f"collapses.\n"
        f"Reply exactly one word: STOP or CONTINUE, then a short reason.\n"
    )
    try:
        import subprocess as _sp
        proc = _sp.Popen(
            ["opencode", "run", "--model",
             "opencode/muse-spark-1.3-contributor-free", prompt],
            cwd="/home/kenpeter/work/pp",
            stdout=_sp.PIPE, stderr=_sp.PIPE, text=True,
            start_new_session=True)
        try:
            out, err = proc.communicate(timeout=180)
        except _sp.TimeoutExpired:
            proc.kill()
            _sp.run(["pkill", "-9", "-f", "opencode run"], capture_output=True)
            raise RuntimeError("subagent judge timed out")
        txt = (out or "") + (err or "")
        if "STOP" in txt[:60].upper():
            return "stop", f"subagent tie-break: {txt[:200]}"
        return "continue", f"subagent tie-break: {txt[:200]}"
    except Exception as e:
        # Subagent down in the ambiguous zone -> default CONTINUE (keep training).
        return "continue", f"subagent down, default continue ({e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", default=None)
    ap.add_argument("--from-checkpoint", default=None)
    ap.add_argument("--rounds", type=int, default=0,
                    help="max rounds; 0 = infinite loop until WALKS (default)")
    ap.add_argument("--reflect-only", action="store_true")
    ap.add_argument("--no-reflect", action="store_true",
                    help="skip the opencode reflector; always use the fallback "
                         "strategy (pure resume, no blocking on muse)")
    args = ap.parse_args()

    state = load_state()
    run = args.from_run or state["run"]
    ckpt = args.from_checkpoint or state["checkpoint"]
    if not run or not ckpt:
        if state.get("rounds_done", 0) == 0 and state.get("run") is None:
            run, ckpt = None, None  # first-ever round trains fresh from init
        else:
            print("need --from-run and --from-checkpoint (or a saved state)")
            sys.exit(2)

    if args.reflect_only:
        st = reflect(state)
        print("REFLECT ->", json.dumps(st))
        sys.exit(0)

    def pick_strategy(state):
        if args.no_reflect:
            # Reflector disabled: use the safe fallback (continue the same
            # microduck-exact config, pure resume). Never blocks on muse.
            return {"name": f"continue-{state['rounds_done'] + 1}",
                    "kind": "resume-train", "iters": 1000, "envs": 4096,
                    "why": "reflector disabled; continuing microduck-exact resume",
                    "change": "none (no reflector)"}
        if state.get("rounds_done", 0) == 0 and "strategy" not in state:
            return SEED_STRATEGY
        return reflect(state)

    for step in range(args.rounds) if args.rounds and args.rounds > 0 else iter(int, 1):
        st = pick_strategy(state)
        append_strategy(st)
        fresh = (state.get("run") is None)  # first ever run = fresh from init
        log_row("loop", f"round {state['rounds_done'] + 1}: {st['name']} "
                f"{'FRESH' if fresh else 'from ' + ckpt}",
                st.get("why", ""), f"logs/rsl_rl/velocity/{run}/{ckpt}",
                "training")
        new_run = train_round(st, run, ckpt, fresh=fresh)
        if new_run is None:
            log_row("loop", f"{st['name']} failed to produce a run", "see loop log",
                    f"/tmp/opencode/loop_{st['name']}.log", "failed")
            sys.exit(2)
        new_ckpt = newest_model(new_run)
        if new_ckpt is None:
            log_row("loop", f"{st['name']} produced no checkpoint (training died)",
                    "see loop log", f"/tmp/opencode/loop_{st['name']}.log", "failed")
            sys.exit(2)
        try:
            passed, info = verify(new_run, new_ckpt)
        except RuntimeError as e:
            log_row("loop", f"verify failed: {e}", "training died early",
                    f"logs/rsl_rl/velocity/{new_run}", "failed")
            sys.exit(2)
        # ANY run we launched is now recorded so it can always be resumed.
        state.update({"rounds_done": state["rounds_done"] + 1,
                      "run": new_run, "checkpoint": new_ckpt,
                      "last_verdict": info, "last_video": info.get("video", "")})
        save_state(state)
        run, ckpt = new_run, new_ckpt
        # Agentic loop-condition: subagent decides STOP (walks) vs CONTINUE.
        decision, why = decide(state, new_run, new_ckpt)
        if decision == "stop":
            print(f"LOOP VERDICT: WALKS at {new_run}/{new_ckpt} ({why})")
            log_row("loop", f"WALKS at {new_ckpt}",
                    why, f"logs/rsl_rl/velocity/{new_run}/{new_ckpt}", "done")
            sys.exit(0)
        print(f"round {state['rounds_done']} NO-WALK -> {why} -> continuing (infinite loop)")
        # brief pause so we never spin on a bad state; then loop forever.
        time.sleep(5)


if __name__ == "__main__":
    main()
