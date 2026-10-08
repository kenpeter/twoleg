"""Physics randomization helper, ported from BipedRobot (MIT).

Source: /home/kenpeter/work/BipedRobot/envs/utils/randomizer.py
Copyright (c) 2025 Pablo Gomez Martinez (MIT, see BipedRobot/LICENSE.txt).

Status: reference implementation, NOT wired into the mjlab loop yet. Our loop
already randomizes foot friction, encoder bias, and base COM offset via mjlab
domain-randomization events (see params/env.yaml events block), which covers
the same ground through the mjlab API. Wire this class in only if a future
sim-to-real pass needs BipedRobot's exact per-episode scaling semantics
(friction / damping / mass / inertia) on a raw mujoco.Model.
"""

import numpy as np


class Randomizer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        self.original_friction = model.geom_friction.copy()
        self.original_damping = model.dof_damping.copy()
        self.original_mass = model.body_mass.copy()
        self.original_inertia = model.body_inertia.copy()

    def randomize(self):
        cfg = self.config
        if not cfg:
            return

        if cfg.randomize_dynamics:
            if cfg.friction:
                self.model.geom_friction[:] = self.original_friction * np.random.uniform(
                    cfg.friction[0], cfg.friction[1], size=self.model.geom_friction.shape
                )

            if cfg.joint_damping:
                self.model.dof_damping[:] = self.original_damping * np.random.uniform(
                    cfg.joint_damping[0], cfg.joint_damping[1],
                    size=self.model.dof_damping.shape,
                )

            if cfg.mass:
                self.model.body_mass[:] = self.original_mass * np.random.uniform(
                    cfg.mass[0], cfg.mass[1], size=self.model.body_mass.shape
                )

            if cfg.inertia:
                self.model.body_inertia[:] = self.original_inertia * np.random.uniform(
                    cfg.inertia[0], cfg.inertia[1], size=self.model.body_inertia.shape
                )

    def get_noise(self):
        """Return imu and velocity noise if enabled."""
        imu_noise = 0.0
        vel_noise = 0.0
        if getattr(self.config, "randomize_sensors", False):
            imu_noise = getattr(self.config, "imu_noise", 0.0)
            vel_noise = getattr(self.config, "vel_noise", 0.0)
        return imu_noise, vel_noise
