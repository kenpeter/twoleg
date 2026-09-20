#!/usr/bin/env python3
"""GEPA-style reflect loop for gen1 prompts — deterministic, no LM calls.

Failure text (e.g. a verify FAIL or reflection 'failed:' line) is mined for
a concrete constraint via regex miners; the constraint is appended to a copy
of the base phase prompt as an UNREVIEWED candidate; the candidate is scored
with score_prompts.py mode A and kept only if score >= base score.

  python3 gepa_reflect.py --phase verify --failure "tip gap 27mm FAIL ..." \\
      --base ../../agents/verify.md --out /tmp/verify_candidate.md
  python3 gepa_reflect.py --phase action --failure-file /tmp/fail.txt

Exit 0 = candidate kept (written to --out), 1 = candidate rejected or
unscorable, 2 = config/usage error.
With a real LM, replace propose_constraint() with an LM call that reads the
failure + base prompt and drafts the constraint; the mine/score/keep loop
stays identical.
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_prompts import load_schemas, phase_prompt_score  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_SCHEMAS = EVAL_DIR.parent / "agents" / "schemas.json"

# (pattern, constraint template). {m} = matched excerpt (<=80 chars).
MINERS = [
    (r"gap[^.\n]{0,80}\d+\.?\d*\s*mm",
     "validate attachment with the explicit tip metric (Euclidean tip-to-center "
     "minus half-thickness, with threshold) — never a single-axis gap alone"),
    (r"quat[^.\n]{0,60}|angle[^.\n]{0,30}\d+\.?\d*\s*(deg|°)",
     "verify orientation by quat angle-diff against the analytic target AND a "
     "render down the decisive axis before claiming flat/aligned"),
    (r"missing|not found|no such file|does not exist",
     "check-exists-first: stat every file/folder before read/overwrite; merge "
     "docs instead of silent overwrite"),
    (r"FAIL|failed|Traceback|AssertionError|assert",
     "TDD red-green: reproduce the failure with a named failing test first, "
     "apply the smallest reversible patch, re-run the suite green"),
    (r"timeout|timed out|524|gateway",
     "retry the op once; on repeat timeout split along the natural seam and "
     "dispatch parallel subagents, then merge — never loop the monolith"),
    (r"secret|token|api_key|sk-|password",
     "secrets via env vars only; grep for key patterns before every commit; "
     "never commit credentials"),
    (r"overwrite|clobber|wiped|lost",
     "back up (.bak) before live patches; dry-run on a .test artifact first"),
]
FALLBACK = ("state the blocker with verbatim evidence (log lines, numbers, "
            "filenames), write reflection.md, and loop back to discover with "
            "the failure context attached")


def propose_constraint(failure):
    for pattern, template in MINERS:
        m = re.search(pattern, failure, re.IGNORECASE)
        if m:
            excerpt = re.sub(r"\s+", " ", m.group(0)).strip()[:80]
            return template, excerpt
    return FALLBACK, re.sub(r"\s+", " ", failure).strip()[:80]


def main(argv=None):
    ap = argparse.ArgumentParser(description="GEPA-style mine/score/keep loop for gen1 prompts.")
    ap.add_argument("--schemas", default=str(DEFAULT_SCHEMAS))
    ap.add_argument("--phase", required=True, help="discover|plan|action|verify")
    ap.add_argument("--failure", default="", help="failure text to mine")
    ap.add_argument("--failure-file", default="", help="file holding failure text")
    ap.add_argument("--base", default="", help="base prompt file (default agents/<phase>.md)")
    ap.add_argument("--out", default="", help="candidate output path (default /tmp/gen1_<phase>_candidate.md)")
    args = ap.parse_args(argv)

    try:
        schemas = load_schemas(args.schemas)
    except (OSError, ValueError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if args.phase not in schemas["phases"]:
        print(f"usage error: --phase must be one of {list(schemas['phases'])}", file=sys.stderr)
        return 2

    failure = args.failure
    if args.failure_file:
        try:
            failure = Path(args.failure_file).read_text() + "\n" + failure
        except OSError as e:
            print(f"config error: {e}", file=sys.stderr)
            return 2
    if not failure.strip():
        print("usage error: empty failure text (--failure or --failure-file)", file=sys.stderr)
        return 2

    base_path = Path(args.base) if args.base else EVAL_DIR.parent / "agents" / f"{args.phase}.md"
    try:
        base_text = base_path.read_text()
    except OSError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    base_score, base_missing = phase_prompt_score(schemas, args.phase, base_text)
    constraint, excerpt = propose_constraint(failure)
    today = datetime.date.today().isoformat()
    candidate = (base_text.rstrip() + "\n\n"
                 f"## Candidate constraint (auto-mined {today}, GEPA-style, UNREVIEWED)\n"
                 f"- `candidate-constraint-{args.phase}-{today}`: {constraint} "
                 f'[mined from failure: "{excerpt}"]\n')
    cand_score, cand_missing = phase_prompt_score(schemas, args.phase, candidate)

    print(f"BASELINE [{args.phase}] score={base_score:.2f} missing={base_missing}")
    print(f"MINED constraint: {constraint}")
    print(f"CANDIDATE [{args.phase}] score={cand_score:.2f} missing={cand_missing}")
    if cand_score >= base_score:
        out = Path(args.out) if args.out else Path(f"/tmp/gen1_{args.phase}_candidate.md")
        out.write_text(candidate)
        print(f"VERDICT: KEEP -> {out} (review before copying into agents/{args.phase}.md)")
        return 0
    print("VERDICT: REJECT (candidate scores below base; not written)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
