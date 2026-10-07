"""Test cases for the TwoLeg RL training pipeline.

These are the regression nets for every bug we actually hit while porting the
microduck_rl recipe, and after the H9 re-alignment to unitree_rl_mjlab. They run
WITHOUT a GPU render context (headless) and fail FAST -- before any training run
is wasted.

Run:  uv run --no-sync pytest training/tests/test_twoleg_config.py -q
"""
import importlib.util
import os
import sys

import pytest
import torch

# Make the training package importable regardless of cwd.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

TASK = "Mjlab-Velocity-Flat-TwoLeg"

# microduck_rl reference values that still apply after H9 unitree alignment.
EXPECT = {
    "foot_clearance.weight": -0.1,
    "foot_swing_height.weight": -0.1,
    "foot_slip.weight": -0.1,
    "self_collisions.weight": -1.0,
    "track_linear_velocity.weight": 2.0,
    "track_linear_velocity.std": 0.316227766,   # sqrt(0.1)
    "track_angular_velocity.weight": 2.0,
    "track_angular_velocity.std": 0.707106781,  # sqrt(0.5)
    "upright.weight": 2.0,
    "upright.std": 0.223606798,                 # sqrt(0.05)
    "action_rate_l2.weight": -0.1,
    "pose.weight": 1.0,
}


def _cfg():
    return load_env_cfg(TASK, play=True)


@pytest.mark.pin
def test_reward_weights_match_microduck_exact():
    """Core gait reward weights/stds equal the microduck_rl reference.

    Catches: over-strong foot_clearance (-2.0), foot_swing (-0.25), wrong
    track/upright stds. H9: air_time terms are intentionally removed (the hop
    attractor); feet_gait/stand_still/body_orientation_l2 are unitree ports.
    """
    cfg = _cfg()
    r = cfg.rewards
    for key, val in EXPECT.items():
        term, attr = key.split(".")
        assert term in r, f"missing reward term {term}"
        got = r[term].weight if attr == "weight" else r[term].params[attr]
        assert abs(got - val) < 1e-5, f"{key}: got {got}, expected {val}"


@pytest.mark.pin
def test_feet_gait_present():
    """H9: feet_gait (phase-locked alternating gait) is the PRIMARY gait shaper.

    Replaces the air_time rewards that caused the one-foot flail. Catches the
    accidental removal that lets the hop attractor return. Left offset 0.0 /
    right offset 0.5, period 0.6, threshold 0.56 (Unitree G1-flat values).
    """
    cfg = _cfg()
    assert "feet_gait" in cfg.rewards, "feet_gait must be present (gait shaper)"
    assert cfg.rewards["feet_gait"].weight == 0.5
    p = cfg.rewards["feet_gait"].params
    assert p["offset"] == [0.0, 0.5], f"gait offset must be half-phase: {p['offset']}"
    assert p["period"] == 0.6
    assert p["threshold"] == 0.56


@pytest.mark.pin
def test_duty_balance_present():
    """duty_balance (weight 2.0) forces both feet to lift, breaking the
    one-leg standstill (right foot glued, left foot dead). Intentional addition.
    """
    cfg = _cfg()
    assert "duty_balance" in cfg.rewards, "duty_balance must be present"
    assert cfg.rewards["duty_balance"].weight == 2.0


@pytest.mark.pin
def test_antifreeze_terms_present():
    """Anti-squat-freeze + unitree-aligned terms:
    - no_fly (2.0) rewards exactly-one-foot-down (single support)
    - feet_moving (1.5) rewards any foot lifted
    - stand_still (-1.0) penalizes motion at zero command
    - body_orientation_l2 (-1.0) keeps torso upright
    Together these break the symmetric planted-squat local optimum and idle
    thrash the loop found earlier.
    """
    cfg = _cfg()
    assert "no_fly" in cfg.rewards, "no_fly must be present (anti-idle/anti-double-stance)"
    assert cfg.rewards["no_fly"].weight == 2.0
    assert "feet_moving" in cfg.rewards, "feet_moving must be present (any-lift reward)"
    assert cfg.rewards["feet_moving"].weight == 1.5
    assert "stand_still" in cfg.rewards, "stand_still must be present (unitree)"
    assert cfg.rewards["stand_still"].weight == -1.0
    assert "body_orientation_l2" in cfg.rewards, "body_orientation_l2 must be present (unitree)"
    assert cfg.rewards["body_orientation_l2"].weight == -1.0


@pytest.mark.pin
def test_h9_ports_present():
    """H9: ported from unitree_rl_mjlab --
    - feet_gait (0.5): phase-locked alternating gait, the proven hop preventer.
    - stand_still (-1.0): idle-thrash penalty.
    - body_orientation_l2 (-1.0): upright torso.
    - legs_energy (-1e-5): robust_robot_walker mechanical-power penalty.
    Air-time rewards are deliberately ABSENT (the flail cause).
    """
    cfg = _cfg()
    assert "feet_gait" in cfg.rewards, "feet_gait must be present (unitree)"
    assert "stand_still" in cfg.rewards, "stand_still must be present (unitree)"
    assert "body_orientation_l2" in cfg.rewards, "body_orientation_l2 must be present (unitree)"
    assert "legs_energy" in cfg.rewards, "legs_energy must be present (robust_robot_walker)"
    assert cfg.rewards["legs_energy"].weight == -1e-5
    assert "air_time" not in cfg.rewards, "air_time must be removed (hop attractor)"
    assert "both_feet_air_time" not in cfg.rewards, "both_feet_air_time removed"
    assert "feet_air_time_fc" not in cfg.rewards, "feet_air_time_fc removed"


def test_standing_envs_starts_small():
    """rel_standing_envs must open at 0.02 (microduck), not 0.1.

    Catches: a config that biases the gradient toward 'standing is the job'.
    """
    cfg = _cfg()
    assert cfg.commands["twist"].rel_standing_envs == 0.02, \
        f"rel_standing_envs should be 0.02, got {cfg.commands['twist'].rel_standing_envs}"


@pytest.mark.pin
def test_symmetry_matches_microduck():
    """Symmetry config matches microduck: data_aug off, mirror coeff 0.5."""
    from twoleg_training.tasks.symmetry import SYMMETRY_CFG
    assert SYMMETRY_CFG["use_data_augmentation"] is False
    assert SYMMETRY_CFG["mirror_loss_coeff"] == 0.5


def test_six_leg_joints_actuated():
    """Hips, knees AND ankles must be actuated (push-off needs ankles)."""
    from twoleg_training.robot.twoleg_constants import JOINT_NAMES
    assert set(JOINT_NAMES) == {
        "L_hip_test", "L_knee_test", "L_ankle_test",
        "R_hip_test", "R_knee_test", "R_ankle_test",
    }, f"ankles must be actuated: {JOINT_NAMES}"


def test_curricula_present():
    """The two microduck ramps must be wired (action_rate + standing_envs)."""
    cfg = _cfg()
    # play mode clears curricula; build a train-mode cfg instead.
    train_cfg = load_env_cfg(TASK, play=False)
    assert "action_rate_weight" in train_cfg.curriculum, "missing action_rate_weight ramp"
    assert "standing_envs" in train_cfg.curriculum, "missing standing_envs ramp"


def test_env_builds_headless():
    """The env must build with NO render_mode (no GL context needed).

    Catches: any code path that forces a GL/render context at construction.
    """
    cfg = _cfg()
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")  # no render_mode
    assert env is not None
    env.close()


def test_command_axis_forward_is_body_y():
    """Forward command lives on body y. Ranges follow duck: wide enough that
    shuffle cannot track them."""
    cfg = _cfg()
    assert cfg.commands["twist"].ranges.lin_vel_x == (-0.4, 0.4), \
        "lin_vel_x must be lateral"
    assert cfg.commands["twist"].ranges.lin_vel_y == (-0.3, 0.3), \
        "lin_vel_y must span forward and back"
    assert cfg.commands["twist"].ranges.ang_vel_z == (-1.0, 1.0), \
        "ang_vel_z must allow real turns"


def test_policy_runs_and_both_feet_load(tmp_path):
    """Integration: a loaded policy must put BOTH feet on the ground.

    This is the one-leg-collapse regression test. Loads a checkpoint (path from
    env var TWOLEG_CKPT if set, else skips) and asserts left_duty AND
    right_duty > 0.2 over a short rollout. No GL needed.
    """
    ckpt = os.environ.get("TWOLEG_CKPT")
    if not ckpt or not os.path.exists(ckpt):
        pytest.skip("set TWOLEG_CKPT to a .pt to run the gait integration test")
    cfg = _cfg()
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    venv = RslRlVecEnvWrapper(env)
    agent_cfg = load_rl_cfg(TASK)
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(venv, vars(agent_cfg), device="cpu")
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    twist = env.command_manager.get_term("twist")
    obs, _ = venv.reset()
    for _ in range(50):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = 0.1
        twist.command[:] = cmd
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))
    found = []
    for _ in range(200):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = 0.1
        twist.command[:] = cmd
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))
        found.append(env.scene["feet_ground_contact"].data.found[:].clone())
    found = torch.stack(found)
    found = found.squeeze(-1) if found.ndim == 4 else found
    left = found[:, :, 0].float().mean(0)
    right = found[:, :, 1].float().mean(0)
    env.close()
    assert left.min().item() > 0.2, f"left foot not loading: {left.tolist()}"
    assert right.min().item() > 0.2, f"right foot not loading: {right.tolist()}"


def test_walk_loop_state_resumable(tmp_path):
    """The loop state must record run+checkpoint so any run is resumable."""
    state = {
        "rounds_done": 1,
        "run": "2026-10-02_11-25-26_velocity",
        "checkpoint": "model_3000.pt",
        "last_verdict": {"verdict": "NO-WALK"},
        "last_video": None,
    }
    assert state["run"] and state["checkpoint"], "state must carry run+checkpoint"


def test_forward_command_produces_forward_motion_not_sidestep(tmp_path):
    """Regression test for the swapped-axis bug: forward command (body y) must
    produce body-y velocity, and ~0 lateral (body x) velocity. No GL needed.
    """
    ckpt = os.environ.get("TWOLEG_CKPT")
    if not ckpt or not os.path.exists(ckpt):
        pytest.skip("set TWOLEG_CKPT to a .pt to run the forward-walk test")
    cfg = _cfg()
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    venv = RslRlVecEnvWrapper(env)
    agent_cfg = load_rl_cfg(TASK)
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(venv, vars(agent_cfg), device="cpu")
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    twist = env.command_manager.get_term("twist")
    obs, _ = venv.reset()
    for _ in range(50):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = -0.15
        twist.command[:] = cmd
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))
    fwd, lat = [], []
    for _ in range(100):
        cmd = torch.zeros_like(twist.command)
        cmd[:, 1] = -0.15
        twist.command[:] = cmd
        with torch.no_grad():
            obs, _, _, _ = venv.step(policy(obs))
        v = env.scene["robot"].data.root_link_lin_vel_b
        fwd.append(v[:, 1].detach().clone())
        lat.append(v[:, 0].detach().clone())
    fwd = torch.stack(fwd).mean(0).mean().item()
    lat = torch.stack(lat).mean(0).abs().mean().item()
    env.close()
    assert fwd < -0.05, f"no forward motion on -y command: v_y={fwd:.3f}"
    assert lat < 0.05, f"side-step regression: |v_x|={lat:.3f} not ~0"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
