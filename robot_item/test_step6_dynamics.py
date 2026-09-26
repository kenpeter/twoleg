#!/usr/bin/env python3
"""Verify that every servo in the step6 rig actually reaches its commanded angle.

chest_assembly_step6.xml is a pose, so chest_step6_move.py re-rigs it into the servo
chain the hardware has and drives it with position actuators. This test exercises that
same rig under MuJoCo dynamics and asserts the servos track, which is the property a
kinematic render cannot show: a kinematic clip writes qpos directly and would look
identical even if the gains, damping or joint axes were wrong.

Checks:
  1. The rig compiles with one hinge and one position actuator per driven servo.
  2. At zero joint angle every body sits where chest_assembly_step6.xml puts it. The
     rig re-parents bodies between frames, and a pivot constant expressed in the wrong
     frame moves a part without any simulation running: the wrist disc sat 91 mm from
     the servo it is bolted to and orbited that wrong point.
  3. The wrist disc stays on the wrist axis for the whole clip.
  4. The two arms stay mirror images of each other in the world YZ plane for the whole
     clip, which pins the relative phase of the L/R arm servos.
  5. The chain reaches its t=0 command before frame 0 (a start-up lurch is an artefact
     of recording before the servo has wound in, not motion).
  6. Every servo tracks its command to within TRACK_TOL_DEG over the whole clip.
  7. qpos and qvel stay finite for the whole clip.

Contact state is reported, not asserted: chest_step6_move.py takes every geom out of the
contact set because the bolted parts interpenetrate by design, so the rig simulates
gravity, inertia and servo dynamics but no contact. Turning contacts on stalls the
servos at 15-20 deg of error, so that is a modelling change, not a flag.

Run: uv run --offline --with numpy --with mujoco python robot_item/test_step6_dynamics.py
"""
import math
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chest_step6_move as RIG  # noqa: E402

RIGGED = "/tmp/opencode/_step6_dynamics.xml"
TRACK_TOL_DEG = 3.0
SETTLE_TOL_DEG = 1.5
FPS = RIG.FPS
SECONDS = RIG.SECONDS


def _load():
    RIG.build(RIGGED, physics=True)
    m = mujoco.MjModel.from_xml_path(RIGGED)
    return m, mujoco.MjData(m)


def _ids(m):
    qadr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in RIG.ACTUATORS}
    ctrl = {j: m.actuator(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j)).id
            for j in RIG.ACTUATORS}
    return qadr, ctrl


def test_rig_compiles():
    """One hinge and one position actuator per driven servo, or a servo in the video is
    driven by nothing and the tracking check below would pass on a dead joint."""
    fails = []
    try:
        m, d = _load()
    except Exception as exc:
        return [f"rig does not compile: {exc}"], False
    if m.nq != len(RIG.ACTUATORS):
        fails.append(f"nq {m.nq} != {len(RIG.ACTUATORS)} driven servos")
    if m.nu != len(RIG.ACTUATORS):
        fails.append(f"nu {m.nu} != {len(RIG.ACTUATORS)} position actuators")
    for j in RIG.ACTUATORS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
        if jid == -1:
            fails.append(f"{j} has no joint")
        elif m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            fails.append(f"{j} is not a hinge")
        if aid == -1:
            fails.append(f"{j} has no actuator")
    return fails, not fails


POSE_TOL_M = 1e-6
MIRROR_TOL_M = 3e-3     # 1.67 mm measured; in-phase arm servos give 88.9 mm


def _body_xpos(m, d):
    out = {}
    for b in range(1, m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if nm and nm not in out:
            out[nm] = d.xpos[b].copy()
    return out


def test_rig_preserves_source_pose():
    """The rig re-parents bodies between the world, left_arm and forearm frames. At zero
    joint angle that must be a no-op: every part has to land where the pose file already
    had it. A pivot constant written in the forearm frame but used as a left_arm frame
    point is invisible in the XML tree arithmetic and shows up only as a part sitting
    somewhere it was never bolted to."""
    fails = []
    try:
        m, d = _load()
        mujoco.mj_forward(m, d)
        src = mujoco.MjModel.from_xml_path(RIG.SRC)
        ds = mujoco.MjData(src)
        mujoco.mj_forward(src, ds)
        rig_pos, src_pos = _body_xpos(m, d), _body_xpos(src, ds)
        shared = sorted(set(rig_pos) & set(src_pos))
        if len(shared) < 30:
            fails.append(f"only {len(shared)} body names shared with the source; "
                         f"the rig no longer mirrors the pose file")
        for nm in shared:
            off = float(np.linalg.norm(rig_pos[nm] - src_pos[nm]))
            if off > POSE_TOL_M:
                fails.append(f"{nm} is {off * 1000:.3f} mm from where "
                             f"{os.path.basename(RIG.SRC)} puts it at qpos=0 "
                             f"(tol {POSE_TOL_M * 1000:.3f} mm); rig="
                             f"{np.round(rig_pos[nm] * 1000, 2)} source="
                             f"{np.round(src_pos[nm] * 1000, 2)}")
    except Exception as exc:
        fails.append(f"exception {exc}")
    return fails, not fails


def test_wrist_disc_stays_on_axis():
    """The wrist 金属舵盘 is bolted to the forearm servo, so it must sit on the wrist axis
    at every frame rather than orbiting a point 91 mm away from it."""
    fails = []
    worst = 0.0
    try:
        m, d = _load()
        qadr, ctrl = _ids(m)
        mujoco.mj_forward(m, d)
        for j, v in RIG._targets(0.0).items():
            d.ctrl[ctrl[j]] = v
        for _ in range(int(RIG.SETTLE_SECONDS / m.opt.timestep)):
            mujoco.mj_step(m, d)
        sub = max(1, int(round(1.0 / (FPS * m.opt.timestep))))
        disc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "horn_2_f")
        axis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_test_link")
        if disc == -1 or axis == -1:
            return [f"horn_2_f or wrist_test_link missing from the rig"], False
        for i in range(int(FPS * SECONDS)):
            t = (i / FPS) / SECONDS
            for j, v in RIG._targets(t).items():
                d.ctrl[ctrl[j]] = v
            for _ in range(sub):
                mujoco.mj_step(m, d)
            worst = max(worst, float(np.linalg.norm(d.xpos[disc] - d.xpos[axis])))
        if worst > POSE_TOL_M:
            fails.append(f"wrist disc drifts {worst * 1000:.3f} mm off the wrist axis "
                         f"(tol {POSE_TOL_M * 1000:.3f} mm)")
    except Exception as exc:
        fails.append(f"exception {exc}")
    print(f"  worst disc-to-axis distance over the clip: {worst * 1000:.4f} mm")
    return fails, not fails


def test_arms_stay_mirror_images():
    """The two arms must remain reflections of each other in the world YZ plane for the
    whole clip. This is what pins the sign of the arm servos: a joint axis is written in
    its own body's frame and the right arm's frame is the mirror of the left's, so
    whether the pair runs in phase or antiphase decides whether the arms swing together
    or one forward and one back. Driving them in phase puts the worst mirror error at
    88.9 mm; antiphase holds it under 2 mm."""
    fails = []
    worst = 0.0
    worst_name = ""
    pairs = []
    try:
        m, d = _load()
        qadr, ctrl = _ids(m)
        mujoco.mj_forward(m, d)
        for b in range(1, m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            if not nm or nm.endswith("_R"):
                continue
            r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm + "_R")
            if r >= 0:
                pairs.append((b, r, nm))
        if len(pairs) < 8:
            return [f"only {len(pairs)} mirrored body pairs; the right arm is missing "
                    f"or not generated from the left"], False
        for j, v in RIG._targets(0.0).items():
            d.ctrl[ctrl[j]] = v
        for _ in range(int(RIG.SETTLE_SECONDS / m.opt.timestep)):
            mujoco.mj_step(m, d)
        sub = max(1, int(round(1.0 / (FPS * m.opt.timestep))))
        for i in range(int(FPS * SECONDS)):
            t = (i / FPS) / SECONDS
            for j, v in RIG._targets(t).items():
                d.ctrl[ctrl[j]] = v
            for _ in range(sub):
                mujoco.mj_step(m, d)
            for b, r, nm in pairs:
                p = d.xpos[r]
                err = float(np.linalg.norm(
                    np.array([-p[0], p[1], p[2]]) - d.xpos[b]))
                if err > worst:
                    worst, worst_name = err, nm
        if worst > MIRROR_TOL_M:
            fails.append(f"{worst_name} is {worst * 1000:.3f} mm from its mirror image "
                         f"(tol {MIRROR_TOL_M * 1000:.1f} mm); the L/R servos are probably "
                         f"driven in the wrong relative phase")
    except Exception as exc:
        fails.append(f"exception {exc}")
    print(f"  {len(pairs)} mirrored body pairs, worst mirror error {worst * 1000:.3f} mm")
    return fails, not fails


def test_settles_before_recording():
    """The chain starts at rest under gravity with every command at zero, so it must be
    given time to wind in before frame 0 or the clip opens with a lurch."""
    fails = []
    try:
        m, d = _load()
        qadr, ctrl = _ids(m)
        mujoco.mj_forward(m, d)
        target = RIG._targets(0.0)
        for j, v in target.items():
            d.ctrl[ctrl[j]] = v
        for _ in range(int(RIG.SETTLE_SECONDS / m.opt.timestep)):
            mujoco.mj_step(m, d)
        for j, v in target.items():
            err = math.degrees(abs(d.qpos[qadr[j]] - v))
            if err > SETTLE_TOL_DEG:
                fails.append(f"{j} still {err:.2f} deg from its t=0 command after "
                             f"{RIG.SETTLE_SECONDS:.1f}s settle (tol {SETTLE_TOL_DEG})")
    except Exception as exc:
        fails.append(f"exception {exc}")
    return fails, not fails


def test_servos_track_under_dynamics():
    """The property a kinematic render cannot show. Every servo is stepped against a
    sinusoidal command with gravity on and the joint angle read back from the solver."""
    fails = []
    worst = {}
    try:
        m, d = _load()
        qadr, ctrl = _ids(m)
        mujoco.mj_forward(m, d)
        target = RIG._targets(0.0)
        for j, v in target.items():
            d.ctrl[ctrl[j]] = v
        for _ in range(int(RIG.SETTLE_SECONDS / m.opt.timestep)):
            mujoco.mj_step(m, d)
        sub = max(1, int(round(1.0 / (FPS * m.opt.timestep))))
        err = {j: 0.0 for j in RIG.ACTUATORS}
        for i in range(int(FPS * SECONDS)):
            t = (i / FPS) / SECONDS
            target = RIG._targets(t)
            for j, v in target.items():
                d.ctrl[ctrl[j]] = v
            for _ in range(sub):
                mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                fails.append(f"qpos/qvel went non-finite at frame {i}")
                return fails, False
            for j, v in target.items():
                err[j] = max(err[j], abs(d.qpos[qadr[j]] - v))
        for j in RIG.ACTUATORS:
            deg = math.degrees(err[j])
            worst[j] = deg
            if deg > TRACK_TOL_DEG:
                fails.append(f"{j} tracking error {deg:.2f} deg > {TRACK_TOL_DEG}")
    except Exception as exc:
        fails.append(f"exception {exc}")
    for j in RIG.ACTUATORS:
        if j in worst:
            print(f"  {j:14s} max tracking error {worst[j]:6.2f} deg")
    return fails, not fails


def report_contacts():
    m, _ = _load()
    live = int((m.geom_contype != 0).sum())
    print(f"  geoms in the contact set: {live}/{m.ngeom}")
    print("  NOTE: 0 means no contact physics. The bolted parts interpenetrate by design,")
    print("        so the rig simulates gravity, inertia and servo dynamics only.")


def main():
    tests = [
        ("rig_compiles", test_rig_compiles),
        ("rig_preserves_source_pose", test_rig_preserves_source_pose),
        ("wrist_disc_stays_on_axis", test_wrist_disc_stays_on_axis),
        ("arms_stay_mirror_images", test_arms_stay_mirror_images),
        ("settles_before_recording", test_settles_before_recording),
        ("servos_track_under_dynamics", test_servos_track_under_dynamics),
    ]
    fails_total = []
    for name, fn in tests:
        print(f"\n== {name} ==")
        fails, ok = fn()
        for f in fails:
            print("  FAIL:", f)
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
        if not ok:
            fails_total.extend(fails or [name])
    print("\n== contact state (reported, not asserted) ==")
    report_contacts()
    print("\n" + "=" * 60)
    if fails_total:
        print(f"VERDICT: FAIL - {len(fails_total)} issues")
        for f in fails_total:
            print("  -", f)
        return 1
    print("VERDICT: PASS - all tests green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
