"""The walk gate must reject gaits that are NOT human-like.

Each case builds the per-step tensors the gate reads and asserts the verdict.
The old gate was `rolling_ok[-1] & both_load_avg & (speed.abs() > SPEED_TOL)`,
with no stance-switch, no articulation, and no double-support check -- so it
passed a policy that walked backwards, one that dragged straight legs, one that
slid on both planted feet, and one that hopped. The test pins the replacement
human-gait gate using the SAME tensor orientation as scripts/verify_metrics.py:
`left`/`right` are SEPARATE [steps, envs] contact tensors (one leg each), not a
stacked [steps, envs, legs].
"""
import unittest

import torch

SPEED_TOL = 0.05
MIN_SWITCH_HZ = 0.40
MIN_KNEE_FLEX_RAD = 0.35
MIN_HIP_SWING_RAD = 0.20
MIN_DOUBLE_SUPPORT = 0.10
MAX_DOUBLE_SUPPORT = 0.85
MIN_STANCE_KNEE_BEND = 0.20
MAX_SPEED_CV = 0.60


def gate(left, right, speed, knee, hip, double_support, stance_knee_bend, speed_cv,
         upright=None, dur=5.0):
    """Verbatim the conjunction in scripts/verify_metrics.py (human-gait gate).

    left/right: [steps, envs] contact (one leg each). knee/hip: [steps, envs, 2]
    per-leg. speed/double_support/stance_knee_bend/speed_cv: [envs].
    """
    if upright is None:
        upright = torch.ones(left.shape[1], dtype=torch.bool)
    both_load_avg = ((left > 0.20).float().mean(0) > 0.5) & \
                   ((right > 0.20).float().mean(0) > 0.5)   # [envs]
    stance = torch.where(left > 0, 0, torch.where(right > 0, 1, -1))
    switched = (stance[1:] != stance[:-1]) & (stance[1:] >= 0) & (stance[:-1] >= 0)
    switch_hz = switched.float().sum(0) / dur               # [envs]
    knee_flex = (knee.amax(0) - knee.amin(0)).min(1).values  # [envs]
    hip_swing = (hip.amax(0) - hip.amin(0)).min(1).values     # [envs]
    articulated = (knee_flex > MIN_KNEE_FLEX_RAD) & (hip_swing > MIN_HIP_SWING_RAD)
    ds = double_support
    passed = (upright & both_load_avg & (speed > SPEED_TOL)
              & (switch_hz > MIN_SWITCH_HZ)
              & (ds > MIN_DOUBLE_SUPPORT) & (ds < MAX_DOUBLE_SUPPORT)
              & articulated
              & (stance_knee_bend > MIN_STANCE_KNEE_BEND)
              & (speed_cv < MAX_SPEED_CV))
    return passed


class TestWalkGate(unittest.TestCase):
    def setUp(self):
        self.T = 100
        self.up = torch.ones(1, dtype=torch.bool)  # [envs] all upright

    def _tensors(self, left_pat, right_pat, speed, knee_flex, hip_swing,
                 double_support, stance_knee, speed_cv):
        # left/right are SEPARATE [T,1] contact tensors (one leg each)
        left = left_pat.float()
        right = right_pat.float()
        # knee/hip [T,1,2]: per-leg peak-to-peak = knee_flex / hip_swing
        knee = torch.full((self.T, 1, 2), (knee_flex + 0.5))
        knee[0, 0, :] -= knee_flex
        knee[-1, 0, :] += knee_flex
        hip = torch.full((self.T, 1, 2), (hip_swing + 0.5))
        hip[0, 0, :] -= hip_swing
        hip[-1, 0, :] += hip_swing
        speed_t = torch.tensor([speed])
        ds = torch.tensor([double_support])
        return left, right, speed_t, knee, hip, ds, stance_knee, speed_cv

    def test_backwards_fails(self):
        # alternating gait, articulated, but speed NEGATIVE (walking backwards)
        L = torch.tensor([[1.]] * 50 + [[0.]] * 50)
        R = torch.tensor([[0.]] * 50 + [[1.]] * 50)
        args = self._tensors(L, R, -0.5, 0.5, 0.3, 0.3, 0.3, 0.2)
        self.assertFalse(bool(gate(*args, upright=self.up).item()))

    def test_straight_legs_fail(self):
        # alternating gait, good speed, but knees never bend (knee_flex < min)
        L = torch.tensor([[1.]] * 50 + [[0.]] * 50)
        R = torch.tensor([[0.]] * 50 + [[1.]] * 50)
        args = self._tensors(L, R, 0.5, 0.1, 0.3, 0.3, 0.3, 0.2)  # knee 0.1 < 0.35
        self.assertFalse(bool(gate(*args, upright=self.up).item()))

    def test_sliding_on_planted_feet_fails(self):
        # both feet planted the whole time (double_support ~1.0) + sliding fwd
        L = torch.ones(self.T, 1)
        R = torch.ones(self.T, 1)
        args = self._tensors(L, R, 0.5, 0.5, 0.3, 0.99, 0.3, 0.2)  # ds 0.99 > 0.85
        self.assertFalse(bool(gate(*args, upright=self.up).item()))

    def test_hop_fails(self):
        # never both feet down (double_support ~0) -> pure hop, not walk
        L = torch.tensor([[1.]] * 50 + [[0.]] * 50)
        R = torch.tensor([[0.]] * 50 + [[1.]] * 50)
        args = self._tensors(L, R, 0.5, 0.5, 0.3, 0.0, 0.3, 0.2)  # ds 0 < 0.10
        self.assertFalse(bool(gate(*args, upright=self.up).item()))

    def test_human_gait_passes(self):
        # alternating with OVERLAP and MULTIPLE cycles: L down in [0,70] but also
        # toggle every 25 steps so the stance transfers several times (switch_hz
        # high). R mirrors with offset. Each foot down ~70%, double-support window.
        steps = 100
        L = torch.zeros(steps, 1)
        R = torch.zeros(steps, 1)
        for i in range(0, steps, 25):
            L[i:i+18] = 1.0   # L support for 18 steps
            R[i+12:i+25] = 1.0  # R support for 13 steps, overlapping (double-support)
        args = self._tensors(L, R, 0.5, 0.5, 0.3, 0.3, 0.3, 0.2)
        self.assertTrue(bool(gate(*args, upright=self.up).item()))


if __name__ == "__main__":
    unittest.main()
