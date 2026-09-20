# Gen1 pinned demos (few-shot material, mined from PASS history)

> Usage: when evolving wording in `.opencode/agents/<phase>.md`, paste that
> phase's demos below as few-shot examples. Excerpts are verbatim-compressed
> from memory; full traces in `.opencode/memory/*.md`. Eval index:
> `.opencode/eval/gen1_eval.jsonl`.

## Discover (research-first, breadth, no implementation)

### Demo D-discover-1 (eval e01)
- Task: fix disc/shaft head servo — gap 17mm re-confirmed (servo world 0.280 vs horn 0.263, STL horn 20x5x20 servo 19.5x54x43)
- Output: arXiv/GitHub/web no MJCF auto-align (3 irrelevant papers, 0 repos); evolver_scan OpenGL 0x502 only; no 3D gene/skill exists (27 genes/2 caps); propose `sdcard-mjcf-attachment-constraint` + `mjcf_align.py`, blast radius 5 → `/tmp/gen1_discover.md`
- Evidence: root cause named (hard-coded pos/quat, no STL mating compute); compatibility risks listed; files-to-change enumerated
- Source: short-term.md 2026-09-06 04:34

### Demo D-discover-2 (eval e02)
- Task: fix tip axis to Z and re-align disc to shaft tip -0.030 (Loop2, CONDITIONAL PASS loop1: Z 0.5mm PASS, tip 27mm FAIL)
- Output: prior heuristic longest-Y bug documented (shaft nub Z 43mm not Y 54mm); verify nudge -0.030 0 0.0155 = tip 27 + half 2.5 + clearance 0.5 = 30mm; gene 28 needs shaft_axis override Z param; blast radius 5 → `/tmp/gen1_discover.md` Loop2
- Evidence: failure context from reflection reused as input; quantitative nudge derivation
- Source: short-term.md 2026-09-06 14:55

### Demo D-discover-3 (eval e03)
- Task: image_to_mjcf pipelines skill
- Output: 5 wrappers enumerated (rigyd fallback placeholder, TRELLIS-2/Hunyuan3D→obj2mjcf, CRM→obj2mjcf, Meshroom/COLMAP/OpenSfM→mesh→MJCF, SAM2+Metric3D vision fix vs `mjcf_align.py` solve_mate); offline fallback demos per wrapper
- Evidence: install notes (torch+diffusers+trimesh+mujoco, CoACD), env-vars-only secrets, path-traversal guard
- Source: long-term.md 2026-09-06 Skill image_to_mjcf

## Plan (compat checklist, TDD test plan, user gate)

### Demo D-plan-1 (eval e04)
- Task: fix disc/shaft head servo
- Output: `/tmp/gen1_plan.md` (166 lines); compat checklist 12 items (training N/A, MuJoCo compile/scale/quat must verify); design 5 files blast radius 5; TDD T1-T6 with T1 gap<1mm failing pre-fix; secure SDLC OWASP A01/A08 backup+allowlist → await user go-ahead for /action
- Evidence: every behavior mapped to a named failing-first test; explicit gate, no code touched
- Source: short-term.md 2026-09-06 04:50

### Demo D-plan-2 (eval e05)
- Task: flatten horn disc horizontal
- Output: `/tmp/gen1_plan2.md` (249 lines); quat 0.5 0.5 -0.5 0.5 → 0.5 0.5 -0.5 -0.5 (R_servo*RotX(-90), Y→+Z 0° vs 90° vertical); pos 0 0 -0.0095 preserved; TDD plane<5° fails pre-fix red (90°→green 0°); compat checklist 10 items → auto-approved handoff /action
- Evidence: STL-grounded derivation (Y thin 20x5x20, normal Y 676 vs Z mistaken); single-sign-flip minimal change
- Source: short-term.md 2026-09-06 22:15

## Action (TDD red→green, minimal patch, conventional commit)

### Demo D-action-1 (eval e06)
- Task: fix disc/shaft head servo Loop1
- Output: skill `twoleg_mjcf/mjcf_align.py` (stl_bbox scale 0.001, shaft axis Y, R_horn=R_servo*RotY90); tests T1 gap 17.00→0.50mm PASS, T5 quat 90°→0° PASS; dry-run `robot_twoleg.test.xml` + render; live patch horn pos 0 0 -0.002 → 0 0 0.0155 with `.bak`; mujoco compile OK nbody 15 → handoff /verify
- Evidence: baseline `test_gen1.py` green first; secrets grep clean; dry-run before live patch; backup kept
- Source: short-term.md 2026-09-06 14:39

### Demo D-action-2 (eval e07)
- Task: zero-gap touching horn disc (pos -0.0095→-0.009, quat unchanged)
- Output: TDD 6 FAIL red pre-fix → patched `mjcf_align.py` clearance param (None→0.0005 safe, 0.0 touching, offset 24.0mm face 0) + `validate_gap` face_gap; gene act14→15 touching 24.0mm doc; dry-run face 0 tip 2.5mm; live patch; 13/13 twoleg PASS 26/26 gen1 PASS; commit 27cdddda conventional → handoff /verify
- Evidence: red-green shown; tests + code committed together; only intended files staged
- Source: short-term.md 2026-09-06 23:22

## Verify (real execution, numbers, mandatory reflection)

### Demo D-verify-1 (eval e08)
- Task: verify disc/shaft head servo Loop1
- Output: XML OK; MjModel OK nbody=15 njnt=16 nu=15; gap Z 0.50mm PASS (<1mm) BUT tip-to-horn 27.00mm FAIL; R_horn angle 0.00° PASS; EGL renders `/tmp/verify_head_closeup.png` (mean 45.4) + full (26.6); verdict CONDITIONAL → reflection written with root cause (longest-Y heuristic, Z-only metric insufficient) as Loop2 input
- Evidence: FAIL stated with numbers, not hidden by green tests (6/6 twoleg + 26/26 gen1 still green); next-try nudge (-0.030) derived
- Source: long-term.md 2026-09-06 Verify + short-term 04:44

### Demo D-verify-2 (eval e09)
- Task: verify disc CONNECTS zero gap loop3 (pos 0 0 -0.009 face 0, quat 0.5 0.5 -0.5 -0.5 flat)
- Output: XML OK; MjModel OK nbody15 njnt16 ngeorm46; face_gap 5.96e-11 <0.1mm, horn_top 0.2585 == tip 0.2585, offset 24.0mm; quat norm 1 angle 0° R_servo*RotX(-90) Y→+Z dot 1 plane 0°; EGL 1024 sideways mean 56.2 + head_closeup mean 55.6 bore centered touching; 13/13 + 26/26; gene act16 → PASS
- Evidence: every claim has a measured number; renders confirm touching; reflection + long-term distill written
- Source: reflection.md 2026-09-06 23:35 + long-term.md loop3

### Demo D-verify-3 (eval e10)
- Task: verify horn disc flat parallel to ground loop2
- Output: XML OK quat 0.5 0.5 -0.5 -0.5 norm 1 angle 0° vs RotX-90 (120° before); R_horn col1 [0,0,1] dot +Z 1.0, plane 0°<5° PASS (was 90° vertical); pos 0 0 -0.0095 preserved, tip Z 3.00mm<5mm; EGL head_closeup mean 50.2 flat circle + top-down mean 209.7 circle proof; 11/11 PASS → PASS
- Evidence: top-down close (dist 0.08 elev 90) chosen as decisive flatness proof; before/after angles stated
- Source: long-term.md 2026-09-06 Verify horizontal
