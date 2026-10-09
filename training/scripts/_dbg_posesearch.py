"""Two-stage search for a HOLDABLE standing pose at correct height (root_z=0.45):
Stage 1: for a grid of (hip_pitch, knee, ankle, hip_roll) compute static CoM
         margin inside the foot polygon (no simulation -> fast).
Stage 2: simulate PD-hold (zero action) for the top-K by margin; measure max tilt.
Pick the pose that survives upright. This is the pose we set as STANDING_KEYFRAME,
so training starts from a robot that can actually stand.
"""
import sys, numpy as np, torch
sys.path.insert(0, "/home/kenpeter/work/torch-rl-algorithms")
sys.path.insert(0, "/home/kenpeter/work/twoleg/training")
from mjlab.utils.torch import configure_torch_backends
configure_torch_backends()
import twoleg_rl.tasks.velocity.config.twoleg  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

def com_margin(pose):
    ec = load_env_cfg("TwoLeg-Velocity-Flat")
    ec.scene.num_envs = 1; ec.terminations = {}
    rc = ec.scene.entities["robot"]
    rc.init_state.pos = (0.0, 0.0, 0.45)
    rc.init_state.joint_pos = {
        ".*_hip_test": pose[0], ".*_knee_test": pose[1],
        ".*_ankle_test": pose[2], ".*_hip_roll_test": pose[3]}
    env = ManagerBasedRlEnv(cfg=ec, device="cpu")
    w = RslRlVecEnvWrapper(env, clip_actions=False)
    obs, _ = w.reset()
    robot = env.scene["robot"]
    bp = robot.data.body_com_pos_w.cpu().numpy()[0]
    names = list(robot.body_names)
    mm = np.asarray(env.sim.model.body_mass).reshape(-1)
    bi = np.asarray(getattr(robot, "body_indices", np.arange(1,1+len(names))))
    masses = mm[bi].reshape(-1)
    com = (bp * masses[:,None]).sum(0)/masses.sum()
    fi = [i for i,n in enumerate(names) if "foot" in n.lower()]
    fx=[bp[i,0] for i in fi]; fy=[bp[i,1] for i in fi]
    foot_w = max(fx)-min(fx); foot_h = max(fy)-min(fy)
    # margin: how far CoM is from polygon edge (normalized by half-span)
    mx = (min(fx)+max(fx))/2 - com[0]; my = (min(fy)+max(fy))/2 - com[1]
    margin = min(abs(mx)/(foot_w/2+1e-6), abs(my)/(foot_h/2+1e-6))
    env.close()
    return dict(com=com, fx=fx, fy=fy, foot_w=foot_w, foot_h=foot_h, margin=margin,
                inside=(min(fx)<=com[0]<=max(fx)) and (min(fy)<=com[1]<=max(fy)))

# Stage 1: coarse grid
grid = {
    "hip_pitch": [0.0, 0.15, 0.3],
    "knee":      [0.0, -0.2, -0.4, -0.6],
    "ankle":     [0.0, 0.2, 0.4],
    "hip_roll":  [0.0, 0.1, 0.2],
}
import itertools
cands = []
for hp in grid["hip_pitch"]:
    for kn in grid["knee"]:
        for an in grid["ankle"]:
            for hr in grid["hip_roll"]:
                pose=(hp,kn,an,hr)
                try:
                    r = com_margin(pose)
                    if r["inside"]:
                        cands.append((r["margin"], pose, r))
                except Exception as e:
                    pass
cands.sort(key=lambda x:-x[0])
print(f"[stage1] {len(cands)} inside-polygon poses found; top 8 by margin:")
for m,pose,r in cands[:8]:
    print(f"  margin={m:.2f} pose={tuple(round(x,2) for x in pose)} foot({r['foot_w']:.2f}x{r['foot_h']:.2f}) CoM=({r['com'][0]:.2f},{r['com'][1]:.2f})")

# Stage 2: simulate PD-hold for top 5
top = [p for _,p,_ in cands[:5]]
print(f"\n[stage2] simulating PD-hold for {len(top)} poses (150 steps):")
for pose in top:
    ec = load_env_cfg("TwoLeg-Velocity-Flat")
    ec.scene.num_envs=1; ec.terminations={}
    rc = ec.scene.entities["robot"]
    rc.init_state.pos=(0.0,0.0,0.45)
    rc.init_state.joint_pos={".*_hip_test":pose[0],".*_knee_test":pose[1],".*_ankle_test":pose[2],".*_hip_roll_test":pose[3]}
    env = ManagerBasedRlEnv(cfg=ec, device="cpu")
    w = RslRlVecEnvWrapper(env, clip_actions=False)
    obs,_ = w.reset()
    robot = env.scene["robot"]
    maxt=0.0
    for i in range(150):
        o,r,d,info = w.step(torch.zeros(1,8))
        ang = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:,2],-1,1)).abs().item()
        maxt=max(maxt,ang)
    env.close()
    deg=maxt*180/3.14159
    print(f"  pose={tuple(round(x,2) for x in pose)} max_tilt={deg:.1f}deg {'STABLE' if deg<15 else 'tips'}")
