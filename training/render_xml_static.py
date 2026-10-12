"""Static kinematic showcase of the TwoLeg XML (no dynamics -> cannot topple).

Loads chest_both_arms_legs.xml directly with mujoco, sets the standing keyframe on the
legs (locked), and animates the head + hands so the viewer can SEE all 15 DOF
move. Because we override qpos every frame (kinematic hold), the robot holds a
perfect two-foot stance for the whole clip -- this is a model showcase, not a
control demo, so it isolates "is the 15-DOF XML correct" from "can the untrained
policy balance" (which training learns).

Usage:  .venv/bin/python render_xml_static.py --out FILE --secs 4
"""
import argparse
import os
import tempfile
import numpy as np
import mujoco
import imageio.v2 as imageio

XML_PATH = os.path.join(os.path.dirname(__file__), "..", "robot_item", "xml", "chest_both_arms_legs.xml")

# Keyframe: legs held at the designed standing pose (mirrors robot.py STANDING_KEYFRAME
# minus head/hands which we animate). Ankle -0.5, hips/knees 0, hip_roll 0.
KEYFRAME = {
    "L_hip_roll_test": 0.0, "L_hip_test": 0.0, "L_knee_test": 0.0, "L_ankle_test": -0.5,
    "R_hip_roll_test": 0.0, "R_hip_test": 0.0, "R_knee_test": 0.0, "R_ankle_test": -0.5,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/xml_static.mp4")
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=50)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(os.path.abspath(XML_PATH))
    # ensure offscreen buffer is large enough for our render size
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)

    jid = {model.joint(i).name: i for i in range(model.njnt)}
    # map joint name -> qpos slice (all our joints are hinge => nq=1)
    qslice = {}
    for j in range(model.njnt):
        qslice[model.joint(j).name] = (model.jnt_qposadr[j], 1)

    # initial standing pose
    mujoco.mj_resetData(model, data)
    for name, val in KEYFRAME.items():
        if name in qslice:
            adr, nq = qslice[name]
            data.qpos[adr:adr+nq] = val
    mujoco.mj_forward(model, data)

    # render-only camera: place it and orient it to look at the robot's torso
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
    cam_pos = np.array([1.6, -1.6, 0.9])
    target = np.array([0.0, 0.0, 0.15])
    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    true_up = np.cross(fwd, right)
    R = np.eye(3)
    R[:, 0] = right; R[:, 1] = true_up; R[:, 2] = fwd
    tr = R[0,0] + R[1,1] + R[2,2]
    import math
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        qw, qx, qy, qz = 0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    else:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    model.cam_pos[cam_id] = cam_pos
    model.cam_quat[cam_id] = np.array([qw, qx, qy, qz])

    tmp = tempfile.mkdtemp(prefix="xml_frames_")
    fpaths = []
    renderer = mujoco.Renderer(model, width=1280, height=720)
    frames = int(args.secs * args.fps)
    try:
        for f in range(frames):
            t = f / args.fps
            for name, val in KEYFRAME.items():
                adr, nq = qslice[name]
                data.qpos[adr:adr+nq] = val
            if "head_test" in qslice:
                adr, nq = qslice["head_test"]
                data.qpos[adr:adr+nq] = 0.3 * np.sin(2*np.pi*0.5*t)
            for h in ("L_hand_1", "R_hand_1", "L_hand_2", "R_hand_2", "L_hand_3", "R_hand_3"):
                if h in qslice:
                    adr, nq = qslice[h]
                    amp = 0.4 if "hand_1" in h else 0.3
                    ph = np.pi if h.startswith("R") else 0.0
                    data.qpos[adr:adr+nq] = amp * np.sin(2*np.pi*0.8*t + ph)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="demo_cam")
            img = renderer.render()
            fp = os.path.join(tmp, f"f{f:04d}.png")
            imageio.imwrite(fp, img)
            fpaths.append(fp)
    except Exception as e:
        print("render-loop error:", repr(e))

    if not fpaths:
        print("NO FRAMES RENDERED")
        return
    # assemble to mp4 with ffmpeg
    import subprocess
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i",
           os.path.join(tmp, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"wrote {args.out} ({len(fpaths)} frames, {args.secs}s)")

if __name__ == "__main__":
    main()
