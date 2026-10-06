"""Safe, reversible code-editing for the TwoLeg agentic training loop.

The loop body may propose ONE structured change after a non-walking verdict.
Changes are applied ONLY through this module, which:
  * touches a whitelist of known targets (reward weights, symmetry coeff,
    env-config params, training hyperparams),
  * commits the edit to git (so it is reversible),
  * records the rollback commit hash in state,
  * runs the structural/regression test suite after applying; if tests fail it
    auto-reverts.

No free-form file rewriting / shell injection is permitted. The reflector must
emit a change dict matching the schema in apply_change().
"""
from __future__ import annotations

import json
import subprocess
import os
import re
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # training/

REWARD_WEIGHT_FILE = "src/twoleg_training/tasks/twoleg_velocity_env_cfg.py"
SYMMETRY_FILE = "src/twoleg_training/tasks/symmetry.py"
ENV_CFG_FILE = "src/twoleg_training/tasks/twoleg_velocity_env_cfg.py"

# Reward term names we allow the loop to rescale (cfg.rewards["<name>"].weight).
ALLOWED_REWARD_TERMS = {
    "air_time", "foot_clearance", "foot_swing_height", "upright",
    "track_lin_vel", "track_ang_vel", "action_rate", "action_smoothness",
    "standing_envs", "gait_contact", "both_feet_air_time", "penalize_held_foot",
}

# Env-config scalar params we allow the loop to tune.
ALLOWED_ENV_PARAMS = {
    "fell_over_limit_angle", "command_threshold", "foot_target_height",
    "rel_standing_envs", "learning_rate", "entropy_coef", "clip_param",
}

# Algorithm hyperparams editable via target "hyperparam".
ALLOWED_HYPERPARAMS = {
    "learning_rate", "entropy_coef", "clip_param",
}


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + list(args), cwd=REPO,
                         capture_output=True, text=True)


def git_commit(msg: str) -> str:
    """Commit all current changes; return the new HEAD short hash (or '' on fail)."""
    git("add", "-A")
    r = git("commit", "-m", msg)
    if r.returncode != 0:
        return ""
    return git("rev-parse", "--short", "HEAD").stdout.strip()


def run_tests() -> bool:
    """Return True if the structural/regression suite passes.

    Skips tests marked @pytest.mark.pin (those assert microduck-exact constant
    values and must NOT block the loop from tuning reward weights/symmetry).
    """
    r = subprocess.run(
        [".venv/bin/python", "-m", "pytest", "tests/", "-q", "-m", "not pin"],
        cwd=REPO, capture_output=True, text=True)
    return r.returncode == 0


def _set_reward_weight(term: str, weight: float) -> None:
    if term not in ALLOWED_REWARD_TERMS:
        raise ValueError(f"reward term not allowed: {term}")
    path = os.path.join(REPO, REWARD_WEIGHT_FILE)
    with open(path) as fh:
        src = fh.read()
    pat = re.compile(
        r'(cfg\.rewards\["' + re.escape(term) + r'"\]\.weight\s*=\s*)([-0-9.eE+]+)')
    m = pat.search(src)
    if not m:
        raise ValueError(f"could not find weight assignment for reward '{term}'")
    new_src = pat.sub(lambda mm: f"{mm.group(1)}{weight!r}", src, count=1)
    with open(path, "w") as fh:
        fh.write(new_src)


def _set_symmetry_coeff(field: str, value: Any) -> None:
    if field not in {"mirror_loss_coeff", "use_mirror_loss", "use_data_augmentation"}:
        raise ValueError(f"symmetry field not allowed: {field}")
    path = os.path.join(REPO, SYMMETRY_FILE)
    with open(path) as fh:
        src = fh.read()
    pat = re.compile(r'("' + re.escape(field) + r'":\s*)([^,\n]+)')
    m = pat.search(src)
    if not m:
        raise ValueError(f"could not find symmetry field '{field}'")
    new_src = pat.sub(lambda mm: f'{mm.group(1)}{json.dumps(value)}', src, count=1)
    with open(path, "w") as fh:
        fh.write(new_src)


def _set_env_param(param: str, value: Any) -> None:
    if param not in ALLOWED_ENV_PARAMS:
        raise ValueError(f"env param not allowed: {param}")
    path = os.path.join(REPO, ENV_CFG_FILE)
    with open(path) as fh:
        src = fh.read()
    if param == "fell_over_limit_angle":
        pat = re.compile(
            r'(cfg\.terminations\["fell_over"\]\.params\["limit_angle"\]\s*=\s*math\.radians\()([0-9.]+)')
        if not pat.search(src):
            raise ValueError("could not find fell_over limit_angle assignment")
        src = pat.sub(lambda mm: f"{mm.group(1)}{float(value)}", src, count=1)
    elif param == "command_threshold":
        pat = re.compile(r'(COMMAND_THRESHOLD\s*=\s*)([0-9.eE+-]+)')
        if not pat.search(src):
            raise ValueError("could not find COMMAND_THRESHOLD")
        src = pat.sub(lambda mm: f"{mm.group(1)}{float(value)!r}", src, count=1)
    elif param == "foot_target_height":
        pat = re.compile(r'(FOOT_TARGET_HEIGHT\s*=\s*)([0-9.eE+-]+)')
        if not pat.search(src):
            raise ValueError("could not find FOOT_TARGET_HEIGHT")
        src = pat.sub(lambda mm: f"{mm.group(1)}{float(value)!r}", src, count=1)
    elif param == "rel_standing_envs":
        pat = re.compile(
            r'(\{"step":\s*0,\s*"rel_standing_envs":\s*)([0-9.eE+-]+)')
        if not pat.search(src):
            raise ValueError("could not find standing envs stage 0")
        src = pat.sub(lambda mm: f"{mm.group(1)}{float(value)!r}", src, count=1)
    else:
        raise ValueError(f"env param handler missing: {param}")
    with open(path, "w") as fh:
        fh.write(src)


def _set_hyperparam(param: str, value: Any) -> None:
    if param not in ALLOWED_HYPERPARAMS:
        raise ValueError(f"hyperparam not allowed: {param}")
    path = os.path.join(REPO, REWARD_WEIGHT_FILE)
    with open(path) as fh:
        src = fh.read()
    pat = re.compile(r'(' + re.escape(param) + r'\s*=\s*)([0-9.eE+-]+)')
    if not pat.search(src):
        raise ValueError(f"could not find hyperparam '{param}'")
    src = pat.sub(lambda mm: f"{mm.group(1)}{float(value)!r}", src, count=1)
    with open(path, "w") as fh:
        fh.write(src)


def apply_change(change: dict) -> dict:
    """Apply ONE structured change. Returns a result dict.

    change schema (all keys optional except 'target'):
      {"target": "reward_weight", "term": "air_time", "weight": 1.0}
      {"target": "symmetry", "field": "mirror_loss_coeff", "value": 0.8}
      {"target": "env_param", "param": "fell_over_limit_angle", "value": 30.0}
      {"target": "hyperparam", "param": "learning_rate", "value": 5.0e-4}
      {"target": "none"}   # no code change (e.g. just retrain)

    On success: {"ok": True, "commit": <hash>, "summary": ...}
    On failure / test break: {"ok": False, "reverted": True, "reason": ...}
    """
    target = change.get("target", "none")
    try:
        if target == "none":
            return {"ok": True, "commit": "", "summary": "no code change"}
        elif target == "reward_weight":
            term = change["term"]; weight = float(change["weight"])
            _set_reward_weight(term, weight)
            summary = f"reward {term} -> {weight}"
        elif target == "symmetry":
            field = change["field"]; value = change["value"]
            _set_symmetry_coeff(field, value)
            summary = f"symmetry {field} -> {value}"
        elif target == "env_param":
            param = change["param"]; value = change["value"]
            _set_env_param(param, value)
            summary = f"env {param} -> {value}"
        elif target == "hyperparam":
            param = change["param"]; value = float(change["value"])
            _set_hyperparam(param, value)
            summary = f"hyperparam {param} -> {value}"
        else:
            return {"ok": False, "reverted": False,
                    "reason": f"unknown target: {target}"}

        if not run_tests():
            git("checkout", "--", ".")
            return {"ok": False, "reverted": True,
                    "reason": "tests failed after change; reverted"}
        commit = git_commit(f"loop auto-change: {summary}")
        return {"ok": True, "commit": commit, "summary": summary}
    except Exception as e:
        git("checkout", "--", ".")
        return {"ok": False, "reverted": True, "reason": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    print("tests pass (structural):", run_tests())
    print(json.dumps(apply_change({"target": "none"}), indent=2))
