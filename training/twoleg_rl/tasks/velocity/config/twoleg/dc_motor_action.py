"""Torque-controlled joint action with a DC-motor torque-speed envelope.

Mirrors BipedRobot's biped_env_v2._clip_effort_dcmotor (a port of Isaac Lab's
DCMotor._clip_effort). The iw leg servos are STS3215-class: torque collapses
toward zero as joint speed approaches no-load speed, exactly when a swing leg
needs torque most. A flat symmetric clip flatters the sim servo and lets the
policy cheat at high speed, so we apply the real four-quadrant envelope.

This replaces mjlab's JointPositionAction (position/PD control) with effort
(torque) control, matching BipedRobot's actuator model. The policy now commands
torque directly; the envelope is a hard physical constraint.
"""

from __future__ import annotations

import torch

from mjlab.envs.mdp.actions import JointEffortAction, JointEffortActionCfg
from mjlab.envs.mdp.actions.actions import BaseActionCfg, BaseAction
from mjlab.actuator.actuator import TransmissionType


# --- iw DC-motor constants (45 kg-class servos) ------------------------------
# BipedRobot uses STS3215-class (25 kg) servos: stall 2.94, rated 0.98,
# no-load 4.71 rad/s. Our iw servos are 45 kg-class -> scale x1.8.
#   stall: 2.94 * 1.8 = 5.29 N-m
#   rated: 0.98 * 1.8 = 1.76 N-m
#   no-load speed: 4.71 * 1.8 = 8.48 rad/s
SATURATION_EFFORT = 5.29   # N-m, stall @ 12 V (physical torque-speed curve)
EFFORT_LIMIT = 1.76        # N-m, rated continuous @ 12 V (drive clamp)
VELOCITY_LIMIT = 8.48      # rad/s, no-load speed (45 kg-class servo)
_VEL_AT_EFFORT_LIM = VELOCITY_LIMIT * (1.0 + EFFORT_LIMIT / SATURATION_EFFORT)


def clip_effort_dcmotor(effort: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
    """Four-quadrant DC-motor torque-speed clip (vectorized, [N, J]).

    Envelope: flat +-EFFORT_LIMIT up to the corner at _VEL_AT_EFFORT_LIM,
    then linear droop to zero at +-VELOCITY_LIMIT.
    """
    vel = torch.clamp(joint_vel, -_VEL_AT_EFFORT_LIM, _VEL_AT_EFFORT_LIM)
    torque_speed_top = SATURATION_EFFORT * (1.0 - vel / VELOCITY_LIMIT)
    torque_speed_bottom = SATURATION_EFFORT * (-1.0 - vel / VELOCITY_LIMIT)
    max_effort = torch.minimum(torque_speed_top, torch.full_like(vel, EFFORT_LIMIT))
    min_effort = torch.maximum(torque_speed_bottom, torch.full_like(vel, -EFFORT_LIMIT))
    return torch.clamp(effort, min_effort, max_effort)


class DCMotorEffortActionCfg(BaseActionCfg):
    """Configuration for DC-motor torque-controlled joints."""

    def __post_init__(self):
        self.transmission_type = TransmissionType.JOINT

    def build(self, env) -> "DCMotorEffortAction":
        return DCMotorEffortAction(self, env)


class DCMotorEffortAction(BaseAction):
    """Torque control + four-quadrant DC-motor torque-speed envelope."""

    def __init__(self, cfg: DCMotorEffortActionCfg, env):
        super().__init__(cfg=cfg, env=env)

    def apply_actions(self) -> None:
        effort = self._processed_actions  # [N, J], already scaled by cfg.scale
        joint_vel = self._entity.data.joint_vel[:, self._target_ids]  # [N, J]
        clipped = clip_effort_dcmotor(effort, joint_vel)
        self._entity.set_joint_effort_target(clipped, joint_ids=self._target_ids)
