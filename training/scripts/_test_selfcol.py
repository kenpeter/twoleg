import os, sys, numpy as np, torch
REPO = "/home/kenpeter/work/twoleg/training"
for p in (REPO,):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WARP_USE_LEGACY_CODEGEN", "1")
from scripts.twoleg_gym_env import TwoLegGymEnv

env = TwoLegGymEnv(device="cpu", num_envs=1)
env.start()
art = env.env.scene["robot"]
# Force both legs to identical angles (cross them) and step; check contacts.
# Push strong symmetric torque that would make legs swing toward each other.
acts = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
max_contacts = 0
for i in range(40):
    o, inf = env.step(acts)
    # count active contacts involving a capsule geom
    d = env.env.sim.data
    m = env.env.sim.model
    ncon = int(d.nacon)
    cap_contacts = 0
    for c in range(ncon):
        con = d.acon[c]
        g1 = int(torch.as_tensor(con.geom1).item())
        g2 = int(torch.as_tensor(con.geom2).item())
        nm1 = m.geom(g1).name if g1 < m.ngeom else ""
        nm2 = m.geom(g2).name if g2 < m.ngeom else ""
        if "_capsule" in nm1 or "_capsule" in nm2:
            cap_contacts += 1
    max_contacts = max(max_contacts, cap_contacts)
print("max capsule-involved contacts over 40 steps:", max_contacts)
print("SELF_COLLISION_ACTIVE" if max_contacts > 0 else "NO_SELF_COLLISION_DETECTED")
env.close()
