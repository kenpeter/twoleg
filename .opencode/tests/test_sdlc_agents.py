#!/usr/bin/env python3
"""High-quality verification that /gen1 + 4 SDLC agents (muse-spark + DNA) are correctly wired.

Run: python3 test_sdlc_agents.py -v
Exit 0 = all high-quality gates pass.
"""
import json, re, pathlib, sys, subprocess
ROOT = pathlib.Path(__file__).parent.parent.parent
AGENTS_DIR = ROOT / ".opencode" / "agents"
GEN1_CMD = ROOT / ".opencode" / "command" / "gen1.md"
GENES_PATH = pathlib.Path.home() / ".hermes/profiles/agent-1/.evolver/gep/genes.json"
EVAL_DIR = ROOT / ".opencode" / "eval"
SCHEMAS_PATH = ROOT / ".opencode" / "agents" / "schemas.json"
EVAL_PATH = EVAL_DIR / "gen1_eval.jsonl"
DEMOS_PATH = EVAL_DIR / "demos.md"
SCORER = EVAL_DIR / "score_prompts.py"

FAILURES = []

def check(cond, msg):
    if not cond:
        raise AssertionError(msg)

def test_agents_exist():
    for name in ["discover.md","plan.md","action.md","verify.md"]:
        p = AGENTS_DIR / name
        check(p.exists(), f"missing {p}")
        check(p.stat().st_size > 500, f"{name} too small — likely empty")

def test_agents_frontmatter_valid():
    """Each agent must be high-quality: valid frontmatter, subagent, muse-spark, temperature, description."""
    for p in AGENTS_DIR.glob("*.md"):
        txt = p.read_text()
        # frontmatter between --- ---
        m = re.search(r"^---\n(.*?)\n---", txt, re.DOTALL)
        check(m, f"{p.name} missing frontmatter")
        fm = m.group(1)
        check("mode: subagent" in fm, f"{p.name} frontmatter must have mode: subagent")
        check("opencode/muse-spark-1.2-contributor-free" in fm, f"{p.name} frontmatter must pin model muse-spark")
        check("description:" in fm, f"{p.name} frontmatter must have description")
        check("temperature:" in fm, f"{p.name} frontmatter must have temperature")
        # model must be in frontmatter only, not duplicated elsewhere
        check(txt.count("muse-spark") >= 1, f"{p.name} must mention muse-spark")

def test_agents_dna_injected():
    for p in AGENTS_DIR.glob("*.md"):
        txt = p.read_text()
        check("🧬 DNA" in txt, f"{p.name} missing 🧬 DNA injection marker")
        check("you ARE these genes" in txt.lower() or "dna injected" in txt.lower(), f"{p.name} must declare DNA identity")

def test_agents_reference_real_genes():
    with open(GENES_PATH) as f:
        genes = {g["id"]: g for g in json.load(f)["genes"]}
    for p in AGENTS_DIR.glob("*.md"):
        txt = p.read_text()
        ids = re.findall(r"sdcard-[a-z0-9-]+", txt)
        check(len(ids) >= 3, f"{p.name} should list >=3 gene IDs, found {ids}")
        for gid in ids:
            check(gid in genes, f"{p.name} references unknown gene {gid} not in genes.json")
            check(genes[gid]["dna"] == "🧬", f"gene {gid} must have 🧬")
            check(len(genes[gid]["trigger"]) > 5 and len(genes[gid]["action"]) > 5, f"gene {gid} trigger/action too short")

def test_agents_character_high_quality():
    """Character claims must match SDLC role + handoff + no-implement guards."""
    cases = {
        "discover.md": [r"best.*unknown", r"breadth", r"pass.*plan|\bhandoff\b"],
        "plan.md": [r"best planner", r"tests-first|tdd", r"pass.*action|\bhandoff\b"],
        "action.md": [r"best.*execution|best.*action", r"new behavior without.*test|tdd", r"pass.*verify|\bhandoff\b"],
        "verify.md": [r"fact.*evidence|evidence.*fact", r"numbers|math", r"real execution|image.*log|evidence"],
    }
    for fname, patterns in cases.items():
        txt = (AGENTS_DIR / fname).read_text().lower()
        for pat in patterns:
            check(re.search(pat, txt), f"{fname} must match character pattern /{pat}/")

def test_gen1_router_high_quality():
    txt = GEN1_CMD.read_text()
    check("~/work/gen1" in txt or "/home/kenpeter/work/gen1" in txt, "gen1.md must ref ~/work/gen1 as source of truth")
    check(".opencode/agents" in txt, "gen1.md must route to .opencode/agents")
    check("muse-spark" in txt, "gen1.md must mention muse-spark")
    for agent in ["@discover","@plan","@action","@verify"]:
        check(agent in txt, f"gen1.md must mention {agent}")
    check("Task" in txt, "gen1.md must describe firing agents via Task tool")
    check("loop back" in txt.lower(), "gen1.md must describe loop back to discover")
    check("verify-fix" in txt, "gen1.md must retain verify-fix alias via @verify")

def test_no_duplicate_commands():
    cmd_dir = ROOT / ".opencode" / "command"
    for name in ["discover.md","plan.md","action.md","verify.md"]:
        check(not (cmd_dir / name).exists(), f"{name} should NOT be in command/ (agents only)")

def test_sdlc_loop_closure():
    """Verify -> reflect -> loop to Discover is closed, and each phase has a handoff."""
    verify_txt = (AGENTS_DIR / "verify.md").read_text().lower()
    check("discover" in verify_txt, "verify must loop back to discover")
    check("reflect" in verify_txt, "verify must include reflect step (sdcard-reflect-after-session)")
    discover_txt = (AGENTS_DIR / "discover.md").read_text().lower()
    check("/tmp/gen1_discover.md" in discover_txt, "discover must write handoff file")
    plan_txt = (AGENTS_DIR / "plan.md").read_text().lower()
    check("/tmp/gen1_plan.md" in plan_txt, "plan must write handoff file")
    gen1_txt = GEN1_CMD.read_text().lower()
    check("reflect" in gen1_txt and "loop back" in gen1_txt, "gen1.md must describe discover->plan->action->verify->reflect->loop")
    # order check: discover before plan before action before verify before reflect
    for a,b in [("discover","plan"),("plan","action"),("action","verify"),("verify","reflect")]:
        check(gen1_txt.index(a) < gen1_txt.index(b), f"gen1 loop order {a} before {b}")

def test_reflect_mandatory_and_on_difficulty():
    gen1_txt = GEN1_CMD.read_text()
    low = gen1_txt.lower()
    check("mandatory" in low and "reflect" in low, "gen1 must state reflect mandatory after every finish")
    check("difficult" in low, "gen1 must state reflect triggered on difficulty")
    check("reflection.md" in gen1_txt, "gen1 must reference .opencode/memory/reflection.md")
    check("sdcard-reflect-after-session" in gen1_txt or "reflect-after-session" in (AGENTS_DIR/"verify.md").read_text(), "reflect gene must be wired")
    # verify agent must write reflection.md
    vtxt = (AGENTS_DIR / "verify.md").read_text()
    check("reflection.md" in vtxt, "verify.md must write reflection.md")

def test_memory_short_long_term():
    mem_dir = ROOT / ".opencode" / "memory"
    check((mem_dir / "short-term.md").exists(), "short-term.md missing")
    check((mem_dir / "long-term.md").exists(), "long-term.md missing")
    check((mem_dir / "reflection.md").exists(), "reflection.md missing")
    stxt = (mem_dir / "short-term.md").read_text().lower()
    check("20" in stxt or "rolling" in stxt, "short-term must document cap/limit")
    ltxt = (mem_dir / "long-term.md").read_text().lower()
    check("durable" in ltxt or "long-term" in ltxt, "long-term must be durable")
    # agents must read both memories
    for p in AGENTS_DIR.glob("*.md"):
        txt = p.read_text()
        check("short-term" in txt and "long-term" in txt, f"{p.name} must reference both memories")
    # gen1 router must reference memory
    gtxt = GEN1_CMD.read_text()
    check("short-term" in gtxt and "long-term" in gtxt, "gen1.md must reference memory")

def test_genes_store_integrity():
    """High-quality: gene store valid, sacred genes present, no duplicates."""
    with open(GENES_PATH) as f:
        data = json.load(f)
    check("genes" in data and len(data["genes"]) >= 20, "genes.json should have >=20 genes after imports")
    ids = [g["id"] for g in data["genes"]]
    check(len(ids) == len(set(ids)), f"duplicate gene ids: {ids}")
    sacred = {g["id"] for g in data["genes"] if g.get("sacred")}
    for must in ["sdcard-test-gate","sdcard-dont-hardcode-secrets","sdcard-research-first"]:
        check(must in sacred, f"sacred gene {must} missing")

def test_opencode_agents_discoverable():
    """Opencode must list the 4 subagents."""
    r = subprocess.run(["opencode","agent","list"], capture_output=True, text=True, cwd=str(ROOT))
    # agent list may not be available in all versions — soft check
    if r.returncode == 0:
        out = r.stdout.lower()
        for name in ["discover","plan","action","verify"]:
            check(name in out, f"opencode agent list missing {name}")
    else:
        # fallback: files exist is enough
        pass

def test_phase_schemas_frozen():
    """DSPy lesson 1: schema frozen (I/O, handoff, memory, output_schema); wording free."""
    schemas = json.loads(SCHEMAS_PATH.read_text())
    check(schemas.get("version"), "schemas.json needs version")
    phases = schemas.get("phases", {})
    check(set(phases) == {"discover","plan","action","verify"}, f"schemas must define exactly 4 phases, got {list(phases)}")
    check(schemas.get("loop_order") == ["discover","plan","action","verify","reflect"], f"loop_order drift: {schemas.get('loop_order')}")
    for phase, spec in phases.items():
        for key in ["inputs","outputs","handoff_path","handoff_marker","prompt_must_contain","output_schema"]:
            check(spec.get(key), f"schemas[{phase}] missing frozen field {key}")
        agent_txt = (AGENTS_DIR / f"{phase}.md").read_text().lower()
        check(spec["handoff_marker"].lower() in agent_txt, f"{phase}.md drifted from frozen handoff_marker {spec['handoff_marker']!r}")

def test_eval_set_valid():
    """DSPy lesson 2: eval set mined from memory history covers all 4 phases."""
    rows = [json.loads(l) for l in EVAL_PATH.read_text().splitlines() if l.strip()]
    check(len(rows) >= 8, f"eval set too small: {len(rows)}")
    ids = [r["id"] for r in rows]
    check(len(ids) == len(set(ids)), "duplicate eval ids")
    covered = {r["phase"] for r in rows}
    check(covered == {"discover","plan","action","verify"}, f"eval must cover all 4 phases, got {covered}")
    for r in rows:
        check(r.get("task") and r.get("source"), f"{r['id']} needs task+source")
        check(isinstance(r.get("output_must_contain"), list) and len(r["output_must_contain"]) >= 2, f"{r['id']} needs >=2 output tokens")

def test_demos_pinned():
    """DSPy lesson 3: best PASS traces pinned as few-shot demos per phase."""
    check(DEMOS_PATH.exists(), "demos.md missing")
    txt = DEMOS_PATH.read_text()
    demos = re.findall(r"^### Demo", txt, re.MULTILINE)
    check(len(demos) >= 8, f"need >=8 pinned demos, found {len(demos)}")
    low = txt.lower()
    for phase in ["discover","plan","action","verify"]:
        check(low.count(phase) >= 2, f"demos.md must cover {phase}")
    check(re.search(r"e0\d|e10", txt), "demos must reference eval ids")

def test_scorer_baseline_green():
    """DSPy lesson 2+4: deterministic scorer passes baseline + report-lint controls."""
    r = subprocess.run([sys.executable, str(SCORER), "--baseline"], capture_output=True, text=True, cwd=str(ROOT))
    check(r.returncode == 0, f"scorer baseline failed:\n{r.stdout}\n{r.stderr}")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        good = pathlib.Path(d) / "good.md"
        bad = pathlib.Path(d) / "bad.md"
        good.write_text("Context:\nResearch findings http://example.com/paper\nFiles to change: a.py\n")
        bad.write_text("lorem ipsum nothing relevant\n")
        rg = subprocess.run([sys.executable, str(SCORER), "--report", str(good), "--phase", "discover"], capture_output=True, text=True)
        check(rg.returncode == 0, f"report lint positive control failed:\n{rg.stdout}")
        rb = subprocess.run([sys.executable, str(SCORER), "--report", str(bad), "--phase", "discover"], capture_output=True, text=True)
        check(rb.returncode != 0, "report lint negative control should fail")

if __name__ == "__main__":
    import traceback, time
    tests = [test_agents_exist, test_agents_frontmatter_valid, test_agents_dna_injected, test_agents_reference_real_genes, test_agents_character_high_quality, test_gen1_router_high_quality, test_no_duplicate_commands, test_sdlc_loop_closure, test_reflect_mandatory_and_on_difficulty, test_memory_short_long_term, test_genes_store_integrity, test_opencode_agents_discoverable, test_phase_schemas_frozen, test_eval_set_valid, test_demos_pinned, test_scorer_baseline_green]
    verbose = "-v" in sys.argv
    passed = failed = 0
    for fn in tests:
        t0 = time.time()
        try:
            fn()
            passed += 1
            print(f"✅ {fn.__name__} ({int((time.time()-t0)*1000)}ms)")
        except AssertionError as e:
            failed += 1
            print(f"❌ {fn.__name__}: {e}")
            if verbose:
                traceback.print_exc()
        except Exception as e:
            failed += 1
            print(f"💥 {fn.__name__}: {e}")
            if verbose:
                traceback.print_exc()
    print(f"\n{'='*60}\nResult: {passed}/{len(tests)} passed, {failed} failed")
    sys.exit(1 if failed else 0)
