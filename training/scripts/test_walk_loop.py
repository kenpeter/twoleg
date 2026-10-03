"""Tests that the agentic walk loop fires and verifies. No pytest needed.

Usage (from the repo root, training venv, no installs):
    training/.venv/bin/python training/scripts/test_walk_loop.py          # unit only, no GPU, safe anytime
    training/.venv/bin/python training/scripts/test_walk_loop.py --artifacts
        # read-only check that a fired round left state + decision-trail rows
    training/.venv/bin/python training/scripts/test_walk_loop.py --gpu \
        --run <run_dir> --ckpt <model.pt>
        # fires walk_loop --verify-only as a subprocess (needs GPU + checkpoint)

Exit 0 = all selected tests pass, 1 = a failure, 2 = bad usage.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import walk_loop  # noqa: E402  (module has no import-time side effects)


class RegistryTest(unittest.TestCase):
    def test_entries_are_bounded_resume_rounds(self):
        self.assertTrue(walk_loop.STRATEGIES, "registry must not be empty")
        for st in walk_loop.STRATEGIES:
            for key in ("name", "kind", "iters", "envs", "why"):
                self.assertIn(key, st)
            self.assertEqual(st["kind"], "resume-train")
            self.assertIsInstance(st["iters"], int)
            self.assertIsInstance(st["envs"], int)
            self.assertGreater(st["iters"], 0)
            self.assertLessEqual(st["envs"], 2048)  # 4096 OOMs on this 12 GB host


class StateTest(unittest.TestCase):
    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = walk_loop.STATE
            walk_loop.STATE = os.path.join(tmp, "state.json")
            try:
                want = {"strategy_index": 0, "rounds_done": 2,
                        "run": "r", "checkpoint": "c.pt"}
                walk_loop.save_state(want)
                self.assertEqual(walk_loop.load_state(), want)
            finally:
                walk_loop.STATE = old

    def test_missing_state_gives_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = walk_loop.STATE
            walk_loop.STATE = os.path.join(tmp, "absent.json")
            try:
                st = walk_loop.load_state()
                self.assertEqual(st["strategy_index"], 0)
            finally:
                walk_loop.STATE = old


class NewestModelTest(unittest.TestCase):
    def test_picks_latest_by_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = walk_loop.TRAINING
            walk_loop.TRAINING = tmp
            try:
                d = os.path.join(tmp, "logs", "rsl_rl", "velocity", "runA")
                os.makedirs(d)
                for i, name in enumerate(("model_100.pt", "model_200.pt")):
                    p = os.path.join(d, name)
                    open(p, "w").write("x")
                    os.utime(p, (1000 + i, 1000 + i))
                self.assertEqual(walk_loop.newest_model("runA"), "model_200.pt")
                self.assertIsNone(walk_loop.newest_model("no_such_run"))
            finally:
                walk_loop.TRAINING = old


class ExhaustedTest(unittest.TestCase):
    def test_empty_registry_stops_instead_of_spinning(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_state, old_log, old_argv = (
                walk_loop.STATE, walk_loop.LOG_TSV, sys.argv[:])
            walk_loop.STATE = os.path.join(tmp, "state.json")
            walk_loop.LOG_TSV = os.path.join(tmp, "t.tsv")
            sys.argv = ["walk_loop.py", "--rounds", "1"]
            try:
                with open(walk_loop.STATE, "w") as fh:
                    json.dump({"strategy_index": 99, "rounds_done": 9,
                               "run": "r", "checkpoint": "c.pt"}, fh)
                with self.assertRaises(SystemExit) as cm:
                    walk_loop.main()
                self.assertEqual(cm.exception.code, 1)
                with open(walk_loop.LOG_TSV) as fh:
                    body = fh.read()
                self.assertIn("strategies exhausted", body)
            finally:
                walk_loop.STATE, walk_loop.LOG_TSV = old_state, old_log
                sys.argv = old_argv


class ArtifactsTest(unittest.TestCase):
    """Read-only: a fired round must leave state + trail rows. Safe anytime."""

    def test_fired_round_left_state_and_trail(self):
        repo = os.path.dirname(TRAINING)
        with open(os.path.join(repo, ".audit", "twoleg-walk.tsv")) as fh:
            body = fh.read()
        self.assertIn("loop", body)
        state_p = os.path.join(repo, ".audit", "walk_loop_state.json")
        if os.path.exists(state_p):
            # Round finished: state must point at a run and checkpoint.
            with open(state_p) as fh:
                st = json.load(fh)
            self.assertIn("run", st)
            self.assertIn("checkpoint", st)
        else:
            # Round still in flight: the fire-time row proves the loop fired.
            self.assertIn("training", body)


def run_verify_only(run, ckpt):
    """Fire the loop's verification condition as a subprocess. Needs GPU."""
    env = dict(os.environ, MUJOCO_GL="glfw")
    r = subprocess.run(
        ["uv", "run", "--no-sync", "python", "scripts/walk_loop.py",
         "--verify-only", "--from-run", run, "--from-checkpoint", ckpt],
        cwd=TRAINING, capture_output=True, text=True, env=env, timeout=1500)
    print(r.stdout[-1200:] if r.stdout else "")
    if r.stderr:
        print(r.stderr[-800:])
    assert r.returncode in (0, 1), f"verify-only crashed: {r.returncode}"
    assert "LOOP VERDICT:" in r.stdout, "loop did not reach its verdict line"
    stem = os.path.splitext(ckpt)[0]
    sidecar = f"/tmp/opencode/loop_verify_{stem}.json"
    assert os.path.exists(sidecar), f"missing sidecar {sidecar}"
    with open(sidecar) as fh:
        info = json.load(fh)
    for key in ("verdict", "reasons", "speed", "left_duty", "right_duty",
                "contact_alt", "frame_alt"):
        assert key in info, f"sidecar missing {key}"
    print(f"VERIFY-ONLY PROOF: verdict={info['verdict']} sidecar={sidecar}")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    if args.gpu and not (args.run and args.ckpt):
        print("need --run and --ckpt with --gpu", file=sys.stderr)
        return 2

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(RegistryTest))
    suite.addTests(loader.loadTestsFromTestCase(StateTest))
    suite.addTests(loader.loadTestsFromTestCase(NewestModelTest))
    suite.addTests(loader.loadTestsFromTestCase(ExhaustedTest))
    if args.artifacts or args.gpu:
        suite.addTests(loader.loadTestsFromTestCase(ArtifactsTest))
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    if not res.wasSuccessful():
        return 1
    if args.gpu:
        return run_verify_only(args.run, args.ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
