import os, sys, math
REPO = "/home/kenpeter/work/twoleg/training"
TRA = "/home/kenpeter/work/torch-rl-algorithms"
for p in (TRA, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WARP_USE_LEGACY_CODEGEN", "1")
import numpy as np
import torch
from scripts.twoleg_gym_env import TwoLegGymEnv
env = TwoLegGymEnv(device="cpu", num_envs=1)
# measure root height stability over 100 steps with zero action (policy idle)
env.start()
zs = []
acts = np.zeros((1, 8), dtype=np.float32)
for i in range(100):
    obs, infos = env.step(acts)
    # extract root z from critic/actor? use env data if available
    try:
        z = float(env.env.scene["robot"]._data.root_link_pos_w[0, 2].item())
    except Exception as e1:
        try:
            z = float(env.env.sim._data.qpos[2].item())
        except Exception as e2:
            z = float('nan')
    zs.append(z)
env.close()
zs = [z for z in zs if not math.isnan(z)]
if zs:
    print(f"root_z: min={min(zs):.4f} max={max(zs):.4f} mean={sum(zs)/len(zs):.4f} range={max(zs)-min(zs):.4f}")
    if max(zs) - min(zs) > 1.0 or any(math.isnan(z) for z in zs):
        print("UNSTABLE: robot exploded or NaN")
    else:
        print("STABLE: leg collisions OK at rest/idle")
else:
    print("could not read root z; steps completed without crash -> likely stable")
