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

Mirroring (left/right symmetry) is wired through twoleg_rl.biped_mirroring
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

import twoleg_rl.register  # noqa: F401  (registers the task)
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from twoleg_rl.robot import TWOLEG_ACTION_SCALE


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

        self.env = ManagerBasedRlEnv(
            cfg=cfg, device=device, render_mode="rgb_array" if render else None
        )
        self.venv = RslRlVecEnvWrapper(self.env, clip_actions=cfg.agent.clip_actions if hasattr(cfg, "agent") else 1.0)

        # --- spaces -------------------------------------------------------
        # Stock torch-rl-algorithms PPO reads obs["actor"] and feeds the SAME
        # tensor to both actor and critic (agent.step: `obs = observation["actor"]`).
        # So actor and critic must share one dimension. Our critic (48) = actor
        # (36) + 12 privileged. We declare the actor space at the FULL critic dim
        # (48) and feed the 48-dim obs as "actor" (see _to_trainer_obs), so the
        # actor network input = 48 and the critic normalizer matches. Privileged
        # dims reaching the actor are valid state info.
        critic_dim = int(self.env.observation_manager.group_obs_dim["critic"][0])
        actor_dim = critic_dim
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
        self._mirror_active = bool(self.enable_mirroring and np.random.rand() < 0.5)
        return self._to_trainer_obs(self._maybe_mirror_obs(obs))

    def step(self, actions):
        # actions: numpy [N, A] (from the Trainer) OR a torch tensor (PPO may
        # return CUDA tensors directly). Normalize to host numpy first.
        if hasattr(actions, "cpu"):
            actions_np = np.asarray(actions.detach().cpu().numpy(), dtype=np.float32)
        else:
            actions_np = np.asarray(actions, dtype=np.float32)
        if self._mirror_active:
            actions_np = self._mirror_action(actions_np)
        actions_t = torch.tensor(actions_np, dtype=torch.float32, device=self.device)
        scaled = self._scale(actions_t)
        obs, rewards, dones, infos = self.venv.step(scaled)
        self._episode_steps += 1

        # timeout termination (mjlab don't separate truncated; fold into infos)
        # keep on the same device as dones (cuda when device=cuda)
        truncated = (self._episode_steps >= self.max_episode_steps).to(dones.device)
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
        return self._to_trainer_obs(self._maybe_mirror_obs(obs)), infos_out

    # --- mirroring -------------------------------------------------------
    def _maybe_mirror_obs(self, obs_dict):
        if not self._mirror_active:
            return obs_dict
        from twoleg_rl.biped_mirroring import mirror_obs_dict
        return mirror_obs_dict(obs_dict)

    def _mirror_action(self, actions_np):
        from twoleg_rl.biped_mirroring import mirror_action
        return mirror_action(actions_np)

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
        # mjlab returns dict {group: tensor[N,D]}. The stock torch-rl-algorithms
        # PPO only reads obs["actor"] and feeds that SAME tensor to the critic
        # (agent.step does `obs = observation["actor"]`; critic(obs_batch)). So
        # actor and critic obs MUST share one dimension. Our critic obs (48) is
        # actor obs (36) + 12 privileged (foot height/airtime/contact/forces),
        # and actor is a prefix of critic. We expose the full 48-dim as "actor"
        # so the critic normalizer matches; the actor network simply takes 48 in.
        # (Privileged dims reaching the actor are valid state info, not a leak.)
        critic = obs_dict.get("critic", obs_dict.get("actor"))
        actor = obs_dict.get("actor", critic)
        out = {}
        for k, v in obs_dict.items():
            out[k] = v
        out["actor"] = critic  # single shared dim for stock PPO
        return out

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
    from twoleg_rl import biped_mirroring

    if isinstance(obs, dict):
        return {k: biped_mirroring.mirror_observation(v, 8) for k, v in obs.items()}
    return biped_mirroring.mirror_observation(obs, 8)


def mirror_action(actions):
    from twoleg_rl import biped_mirroring

    return biped_mirroring.mirror_action(actions, 8)
