#!/usr/bin/env python3
"""DSPy-lesson scorer for gen1 prompts (stdlib only, no LM, no API key).

Mode A — prompt conformance (scores a prompt variant vs schemas.json):
  python3 score_prompts.py --baseline
  python3 score_prompts.py --variant /tmp/candidate.md --phase discover

Mode B — report lint (scores a produced report vs a phase output_schema):
  python3 score_prompts.py --report /tmp/gen1_discover.md --phase discover

Exit 0 = pass, 1 = fail, 2 = config/usage error.
A PASS here is a cheap proxy lint, NOT execution evidence
(sdcard-evidence-before-claim): it never replaces test_gen1.py,
test_sdlc_agents.py, MuJoCo compile, or EGL verification.
"""
import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent  # .opencode/eval -> .opencode -> repo
DEFAULT_SCHEMAS = EVAL_DIR.parent / "agents" / "schemas.json"
DEFAULT_AGENTS = EVAL_DIR.parent / "agents"


def load_schemas(path):
    with open(path) as f:
        data = json.load(f)
    if "phases" not in data:
        raise ValueError(f"{path} has no 'phases' key")
    return data


def check_all(text, tokens):
    """Case-insensitive substring check. Returns (score, missing)."""
    low = text.lower()
    missing = [t for t in tokens if t.lower() not in low]
    score = (len(tokens) - len(missing)) / max(len(tokens), 1)
    return score, missing


def check_any_group(text, alt_group):
    """Group like 'PASS|FAIL' passes if ANY alternative is present (case-insensitive)."""
    low = text.lower()
    return any(a.strip().lower() in low for a in alt_group.split("|") if a.strip())


def phase_prompt_score(schemas, phase, variant_text):
    tokens = schemas["phases"][phase]["prompt_must_contain"]
    return check_all(variant_text, tokens)


def phase_report_lint(schemas, phase, report_text):
    groups = schemas["phases"][phase]["output_schema"]
    missing = [g for g in groups if not check_any_group(report_text, g)]
    score = (len(groups) - len(missing)) / max(len(groups), 1)
    return score, missing


def cmd_baseline(schemas, agents_dir):
    total, n, ok = 0.0, 0, True
    for phase in schemas["phases"]:
        p = agents_dir / f"{phase}.md"
        if not p.exists():
            print(f"[{phase}] MISSING {p} score=0.00")
            ok = False
            continue
        score, missing = phase_prompt_score(schemas, phase, p.read_text())
        total += score
        n += 1
        flag = "OK  " if score == 1.0 else "FAIL"
        if score != 1.0:
            ok = False
        print(f"[{phase}] {flag} score={score:.2f} missing={missing}")
    print(f"BASELINE total={total / max(n, 1):.2f}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score gen1 prompt variants / reports (proxy lint).")
    ap.add_argument("--schemas", default=str(DEFAULT_SCHEMAS))
    ap.add_argument("--agents-dir", default=str(DEFAULT_AGENTS))
    ap.add_argument("--baseline", action="store_true", help="score current agents/*.md")
    ap.add_argument("--variant", help="path to candidate prompt file (mode A)")
    ap.add_argument("--report", help="path to produced report file (mode B)")
    ap.add_argument("--phase", help="discover|plan|action|verify")
    args = ap.parse_args(argv)

    try:
        schemas = load_schemas(args.schemas)
    except (OSError, ValueError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    agents_dir = Path(args.agents_dir)

    if args.baseline:
        return cmd_baseline(schemas, agents_dir)
    if args.variant:
        if args.phase not in schemas["phases"]:
            print(f"usage error: --phase must be one of {list(schemas['phases'])}", file=sys.stderr)
            return 2
        try:
            text = Path(args.variant).read_text()
        except OSError as e:
            print(f"config error: {e}", file=sys.stderr)
            return 2
        score, missing = phase_prompt_score(schemas, args.phase, text)
        print(f"[{args.phase}] variant score={score:.2f} missing={missing}")
        return 0 if score == 1.0 else 1
    if args.report:
        if args.phase not in schemas["phases"]:
            print(f"usage error: --phase must be one of {list(schemas['phases'])}", file=sys.stderr)
            return 2
        try:
            text = Path(args.report).read_text()
        except OSError as e:
            print(f"config error: {e}", file=sys.stderr)
            return 2
        score, missing = phase_report_lint(schemas, args.phase, text)
        print(f"[{args.phase}] report lint score={score:.2f} missing={missing}")
        return 0 if score == 1.0 else 1
    print("usage error: one of --baseline, --variant FILE --phase P, --report FILE --phase P", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
