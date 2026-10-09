import os, sys

# Set PYTHONPATH inside the script (avoids shell env hijack heuristic).
REPO = "/home/kenpeter/work/twoleg/training"
TRA = "/home/kenpeter/work/torch-rl-algorithms"
for p in (TRA, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WARP_USE_LEGACY_CODEGEN", "1")

import torch
from scripts.twoleg_gym_env import TwoLegGymEnv
from config import NetworkConfig, NetworkType
try:
    from algorithms.ppo.model import PPO
except ImportError:
    from torch_rl_algorithms.algorithms.ppo.model import PPO

env = TwoLegGymEnv(device="cpu", num_envs=1)
cfg = {
    "train": {"steps": 5000,
              "epoch_steps": 5000,
              "checkpoint_path": "/home/kenpeter/work/twoleg/training/logs/twoleg_ppo",
              "save_steps": 100000, "test_episodes": 1, "show_progress": True,
              "replace_checkpoint": False, "log": True,
              "log_dir": "/home/kenpeter/work/twoleg/training/logs/twoleg_ppo/tb",
              "log_name": "twoleg_ppo"},
    "model": {"actor_config": NetworkConfig(network_type=NetworkType.MLP, hidden_sizes=[256,256]).model_dump(),
              "critic_config": NetworkConfig(network_type=NetworkType.MLP, hidden_sizes=[256,256]).model_dump()},
    "ppo": {"clip_param":0.2,"ppo_epoch":4,"num_mini_batches":4,"value_loss_coef":0.5,
            "entropy_coef":0.01,"gamma":0.99,"gae_lambda":0.95,"max_grad_norm":0.5,"num_steps":2048},
    "actor_lr": 3e-4, "critic_lr": 3e-4,
}
print("[twoleg PPO] building model on cpu")
model = PPO(env, model_path=None, device="cpu", config=cfg)
print("[twoleg PPO] training 5000 steps")
model.train(steps=5000)
print("PPO SMOKE OK")
