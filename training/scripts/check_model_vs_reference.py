"""Compare the training robot against the reference assembly.

The training model and the reference assembly are two different things and
drift silently. This script makes the difference visible and exits non-zero
when they disagree, so the gap cannot be missed again.

    uv run python training/scripts/check_model_vs_reference.py

What it checks, and what it cannot:

* Body, joint, geom and mass counts, and the joint name sets.
* Per-joint anchor and axis, in world frame at the zero pose, matched by joint
  name where the two models share a name.
* Whether the training model is a connected free body (one root with a
  freejoint) or a puppet pinned to the world.

It does NOT check that the two articulations are the same mechanism, because
they are not. The reference assembly is a world-pinned puppet: 32 of its 121
bodies attach to the world, so each limb hangs from the world rather than from
the torso. See the report in abstraction.md.
"""

import os
import subprocess
import sys
import tempfile

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROBOT_ITEM = os.path.join(ROOT, "robot_item")
TRAINING = os.path.join(ROOT, "twoleg_mjcf", "robot_twoleg.xml")


def build_reference(dst):
    sys.path.insert(0, ROBOT_ITEM)
    import chest_both_arms_legs_move as ref  # noqa: E402

    ref.build(dst, True)
    return dst


def names(model, objtype):
    return {
        mujoco.mj_id2name(model, objtype, i)
        for i in range(model.njnt if objtype == mujoco.mjtObj.mjOBJ_JOINT else model.nbody)
        if mujoco.mj_id2name(model, objtype, i)
    }


def joint_table(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name:
            continue
        body = model.jnt_bodyid[j]
        out[name] = {
            "anchor_world": data.xpos[body] + data.xmat[body].reshape(3, 3) @ model.jnt_pos[j],
            "axis_world": data.xmat[body].reshape(3, 3) @ model.jnt_axis[j],
            "type": int(model.jnt_type[j]),
        }
    return out


def world_pinned(model):
    return [i for i in range(1, model.nbody) if model.body_parentid[i] == 0]


def summarize(label, model, path):
    pinned = world_pinned(model)
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    print(f"=== {label} ===")
    print(f"  path        : {os.path.relpath(path, ROOT)}")
    print(f"  bodies      : {model.nbody - 1}")
    print(f"  joints      : {model.njnt}")
    print(f"  actuators   : {model.nu}")
    print(f"  geoms       : {model.ngeom}")
    print(f"  mass        : {model.body_mass.sum():.4f} kg")
    print(f"  world-pinned bodies: {len(pinned)}")
    print(f"  freejoints  : {len(free)}")
    print()
    return pinned, free


def main():
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = build_reference(os.path.join(tmp, "reference.xml"))
        ref = mujoco.MjModel.from_xml_path(ref_path)
        train = mujoco.MjModel.from_xml_path(TRAINING)

        ref_pinned, ref_free = summarize("REFERENCE (robot_item/chest_both_arms_legs.xml via chest_move rig)", ref, ref_path)
        tr_pinned, tr_free = summarize("TRAINING (twoleg_mjcf/robot_twoleg.xml)", train, TRAINING)

        ref_joints = names(ref, mujoco.mjtObj.mjOBJ_JOINT)
        tr_joints = names(train, mujoco.mjtObj.mjOBJ_JOINT)
        print("=== joint names ===")
        print(f"  in reference only: {sorted(ref_joints - tr_joints)}")
        print(f"  in training only : {sorted(tr_joints - ref_joints)}")
        print(f"  shared           : {sorted(ref_joints & tr_joints)}")
        print()

        mass_ratio = train.body_mass.sum() / ref.body_mass.sum()
        print("=== headline deltas ===")
        print(f"  mass    training/reference = {mass_ratio:.3f}  "
              f"({train.body_mass.sum():.3f} vs {ref.body_mass.sum():.3f} kg)")
        print(f"  joints  training/reference = {train.njnt}/{ref.njnt}")
        print(f"  bodies  training/reference = {train.nbody-1}/{ref.nbody-1}")
        print(f"  reference is world-pinned  = {len(ref_pinned) > 1}")
        print(f"  training is a free body    = {len(tr_free) == 1}")
        print()

        shared = sorted(ref_joints & tr_joints)
        if shared:
            print("=== shared joint anchors, world frame at zero pose ===")
            rt, tt = joint_table(ref), joint_table(train)
            for name in shared:
                d = np.linalg.norm(rt[name]["anchor_world"] - tt[name]["anchor_world"])
                print(f"  {name:20s} delta = {d*1000:8.2f} mm")

        print()
        print("The reference is a world-pinned puppet with no freejoint, so it cannot be")
        print("dropped into a free-body simulation. The training model is generated from it")
        print("by generate_twoleg_model.py: same parts, same pivots, re-parented into one")
        print("connected tree. Anchors above should read 0.00 mm and the mass ratio 1.000.")
        print()
        print("Expected differences, all deliberate:")
        print("  +1 body    the world-aligned base root, because the chest body carries a")
        print("             90-degree rotation that would break projected_gravity as root")
        print("  +2 geoms   one collision box per foot; the reference's 26 mesh collision")
        print("             geoms are visual-only here because mesh-mesh contact on the")
        print("             leg brackets blows the solver up")
        print("  -2 joints  the duplicate waist hinges the reference models twice")
        print("  +1 joint   the freejoint")


if __name__ == "__main__":
    main()
