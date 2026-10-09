"""Gym-style adapter wrapping the mjlab TwoLeg env for torch-rl-algorithms.

torch-rl-algorithms' Trainer expects an environment with:
    start()           -> observations  (dict {group: tensor[N,D]} or tensor[N,D])
    step(actions)     -> (observations, infos)
    observation_space / action_space   (gym spaces; used by the model)
    device             (optional)

This adapter drives the mjlab ManagerBasedRlEnv (wrapped by RslRlVecEnvWrapper,
which exposes the rsl_rl VecEnv API: obs_dict, rewards, dones, infos). We run a
single vectorized env (num_envs=1) and treat it as the one "worker" the Trainer
expects. Actions are normalized in [-1, 1] and scaled by TWOLEG_ACTION_SCALE.

Mirroring (left/right symmetry) is wired through twoleg_rl.utils.biped_mirroring
so the trainer's symmetry_augmentation option works.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import twoleg_rl.tasks.velocity.config.twoleg  # noqa: F401  (registers the task)
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from twoleg_rl.assets.robots.twoleg.twoleg_constants import TWOLEG_ACTION_SCALE


class TwoLegGymEnv:
    """Single-worker Gym adapter for the TwoLeg mjlab env.

    Implements the torch-rl-algorithms Trainer contract (start/step) while also
    exposing standard gym observation_space/action_space so the algorithm models
    (ActorCritic / ActorTwinCriticWithTargets) can read input/output dims.
    """

    metadata = {"render_modes": [], "render_fps": 100}

    def __init__(
        self,
        task: str = "TwoLeg-Velocity-Flat",
        device: str = "cpu",
        num_envs: int = 1,
        max_episode_steps: int = 1000,
        enable_mirroring: bool = False,
        render: bool = False,
    ):
        self.task = task
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.enable_mirroring = enable_mirroring
        self.render_mode = render

        cfg = load_env_cfg(task, play=False)
        cfg.scene.num_envs = num_envs
        if render:
            cfg.scene.render_mode = "rgb_array"
        self._cfg = cfg

        self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
        self.venv = RslRlVecEnvWrapper(self.env, clip_actions=cfg.agent.clip_actions if hasattr(cfg, "agent") else 1.0)

        # --- spaces -------------------------------------------------------
        actor_dim = int(self.env.observation_manager.group_obs_dim["actor"][0])
        critic_dim = int(self.env.observation_manager.group_obs_dim["critic"][0])
        action_dim = int(sum(self.env.action_manager.action_term_dim))

        self.observation_space = spaces.Dict(
            {
                "actor": spaces.Box(-np.inf, np.inf, (actor_dim,), np.float32),
                "critic": spaces.Box(-np.inf, np.inf, (critic_dim,), np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (action_dim,), np.float32)
        self._action_dim = action_dim
        self._episode_steps = torch.zeros(num_envs, dtype=torch.long)

    # --- Trainer contract -------------------------------------------------
    def start(self):
        obs, _ = self.venv.reset()
        self._episode_steps.zero_()
        return self._to_trainer_obs(obs)

    def step(self, actions):
        # actions: numpy [N, A] (from the Trainer). Scale normalized -> radians.
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        scaled = self._scale(actions_t)
        obs, rewards, dones, infos = self.venv.step(scaled)
        self._episode_steps += 1

        # timeout termination (mjlab don't separate truncated; fold into infos)
        truncated = self._episode_steps >= self.max_episode_steps
        term = dones.bool() | truncated

        infos_out = {
            # torch-rl-algorithms PPO agent.update(**infos) expects these keys:
            "rewards": rewards,        # tensor [N]
            "resets": term,           # tensor [N] (GAE "dones")
            "next_observations": obs,  # dict {actor,critic} for GAE bootstrap
            "next_observations_actor": obs["actor"],
            "next_observations_critic": obs["critic"],
        }
        # forward any extra mjlab infos (e.g. episode metrics) as numpy
        for k, v in (infos or {}).items():
            try:
                infos_out[k] = v
            except Exception:
                pass
        return self._to_trainer_obs(obs), infos_out

    # --- helpers ---------------------------------------------------------
    def _scale(self, actions_t):
        # actions_t: [N, A] in [-1, 1]; per-joint scale to radians.
        scales = torch.tensor(
            [TWOLEG_ACTION_SCALE[j] for j in sorted(TWOLEG_ACTION_SCALE)],
            dtype=torch.float32,
            device=self.device,
        )
        # TwoLeg_ACTION_SCALE is keyed by joint name; align to action order.
        return actions_t * scales.to(actions_t.device)

    def _to_trainer_obs(self, obs_dict):
        # mjlab RslRlVecEnvWrapper returns dict {group: tensor[N,D]} (already
        # concatenated). Pass through; Trainer accepts dict observations.
        return {k: v for k, v in obs_dict.items()}

    # --- gym compat (optional, not required by Trainer) -------------------
    def reset(self, seed=None, options=None):
        obs = self.start()
        return obs, {}

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# --- symmetry hook for torch-rl-algorithms SAC/PPO mirror augmentation ------
def mirror_obs(obs, actions=None):
    """Left/right mirror of a flat obs/action batch (BipedRobot convention)."""
    from twoleg_rl.utils import biped_mirroring

    if isinstance(obs, dict):
        return {k: biped_mirroring.mirror_observation(v, 8) for k, v in obs.items()}
    return biped_mirroring.mirror_observation(obs, 8)


def mirror_action(actions):
    from twoleg_rl.utils import biped_mirroring

    return biped_mirroring.mirror_action(actions, 8)
