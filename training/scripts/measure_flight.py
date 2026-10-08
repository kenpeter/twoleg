import sys
sys.path.insert(0, ".")
# Warp 1.18.0 broke mujoco_warp sensor kernel codegen; legacy codegen path works.
# Must be set BEFORE any mjlab/warp import.
import os
os.environ["WARP_USE_LEGACY_CODEGEN"] = "1"
import torch
from mjlab.utils.torch import configure_torch_backends

configure_torch_backends()
import twoleg_rl.tasks.velocity.config.twoleg  # noqa
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from dataclasses import asdict

CKPT = sys.argv[1]
cfg = load_env_cfg("TwoLeg-Velocity-Flat", play=True)
cfg.scene.num_envs = 4
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
venv = RslRlVecEnvWrapper(env)
runner = MjlabOnPolicyRunner(venv, asdict(load_rl_cfg("TwoLeg-Velocity-Flat")), "logs", device="cpu")
runner.load(CKPT, load_cfg={"actor": True}, map_location="cpu")
pol = runner.get_inference_policy(device="cpu")
twist = env.command_manager.get_term("twist")
o, _ = venv.reset()
for _ in range(60):
    c = torch.zeros_like(twist.command); c[:, 0] = 0.1; twist.command[:] = c
    with torch.no_grad():
        o, _, _, _ = venv.step(pol(o))
T = 300
flight, z, vz, spd = [], [], [], []
for _ in range(T):
    c = torch.zeros_like(twist.command); c[:, 0] = 0.1; twist.command[:] = c
    with torch.no_grad():
        o, _, _, _ = venv.step(pol(o))
    f = env.scene["feet_ground_contact"].data.found[:].float()
    flight.append((f.sum(-1) < 0.5).float().mean().item())
    z.append(env.scene["robot"].data.root_link_pos_w[:, 2].mean().item())
    vz.append(env.scene["robot"].data.root_link_lin_vel_w[:, 2].abs().mean().item())
    spd.append(env.scene["robot"].data.root_link_lin_vel_b[:, 0].abs().mean().item())
print("FLIGHT_FRAC:", round(sum(flight) / len(flight), 3),
      "COM_range:", round(max(z) - min(z), 3),
      "mean|vz|:", round(sum(vz) / len(vz), 3),
      "mean_fwd_speed:", round(sum(spd) / len(spd), 3))
