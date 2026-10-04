"""Generate the training-ready TwoLeg model from the reference assembly.

The reference assembly (``robot_item/chest_both_arms_legs.xml`` plus the
articulation that ``chest_move`` rigs onto it) is authoritative for geometry:
where every part sits and where every hinge pivots. It is not usable as a
free body, because 32 of its 121 bodies attach directly to the world and each
limb therefore hangs from the world instead of from the torso.

This script rebuilds the same parts into a connected kinematic tree:

* one root body, ``torso``, carrying a ``freejoint``
* the limb chains re-parented under ``torso``
* every body's ``pos`` and ``quat`` recomputed from its world pose at the zero
  pose, so the assembled geometry is unchanged
* the duplicate waist hinges dropped. The reference models each waist servo as
  two hinges on one shaft (``waist_L`` on the horn branch, ``L_waist_test`` on
  the leg branch), which would give the leg two serial hinges in one place. The
  leg-side hinge is kept and the horn branch is welded into the leg, giving the
  mechanism's true 17 actuated joints.
* geoms, and therefore mass and inertia, carried over untouched. The reference
  has no explicit ``<inertial>`` blocks, so MuJoCo infers mass from the geom
  densities (aluminium 2700, servo bodies 1600, visual geoms 0), which totals
  4.4818 kg.

It writes both consumers in one pass, so nothing has to patch the XML
afterwards:

* ``twoleg_mjcf/robot_twoleg.xml`` for viewing, meshes relative to that dir
* ``training/assets/twoleg.xml`` for mjlab, meshes relative to that dir

    uv run python training/scripts/generate_twoleg_model.py
    uv run python training/scripts/generate_twoleg_model.py --check
"""

import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROBOT_ITEM = os.path.join(ROOT, "robot_item")
VIEW_OUT = os.path.join(ROOT, "twoleg_mjcf", "robot_twoleg.xml")
TRAIN_OUT = os.path.join(ROOT, "training", "assets", "twoleg.xml")

OUTPUTS = (
    (VIEW_OUT, "../robot_item"),
    (TRAIN_OUT, "../../robot_item"),
)

# Joint regularisation. The reference leaves damping, frictionloss and armature
# unset, which is fine for a puppet whose limbs are pinned to the world: the
# only bodies a hinge moves are its own subtree, and nothing has to hold the
# body up. In a free body it blows up. Several link bodies are massless, so a
# hinge can rotate near-zero inertia, the mass matrix goes singular, and the
# solver returns NaN within a few milliseconds. Armature adds reflected rotor
# inertia, which is physically the right quantity for a geared servo and also
# bounds the matrix away from singular.
JOINT_DAMPING = "0.05"
JOINT_FRICTIONLOSS = "0.02"
JOINT_ARMATURE = "0.01"

# The chest body is the top of the assembly, but it carries a 90-degree
# rotation of its own. Making it the free body's root would leave mjlab reading
# projected_gravity as [9.81, 0, 0] while the robot stands, which silently
# breaks the upright reward and the fell_over termination. So the root is an
# empty world-aligned frame, and the chest hangs from it with its own pose.
ROOT_BODY = "u_beam"
BASE_BODY = "torso"

# The reference's limb chains already hang together below these bodies; they
# only need re-attaching to the torso. Everything else keeps its parent.
LIMB_ROOTS = (
    "head_link",
    "abduct_L_link",
    "abduct_R_link",
    "L_waist_link",
    "R_waist_link",
)

# Torso shell and hardware, currently world children.
TORSO_PARTS = (
    "chest_center", "s1_screws", "chest_side_L", "chest_side_R",
    "horn_center", "horn_base_L", "horn_base_R", "s2_screws",
    "low_fin_0", "low_fin_1", "low_fin_2", "low_fin_3",
    "low_plate_0", "low_plate_1", "s3_nuts",
    "chest_servo_L", "chest_servo_R", "waist_servo_L", "waist_servo_R",
    "waist_pivot_shaft_L", "waist_pivot_bearing_L",
    "waist_pivot_shaft_R", "waist_pivot_bearing_R", "sh_screws",
)

# The horn branch of each waist hinge, welded into the leg it is bolted to.
WELDED_WAIST = {"waist_L_link": "L_leg_top", "waist_R_link": "R_leg_top"}
DUPLICATE_WAIST_JOINTS = ("waist_L", "waist_R")

# Servo count check: the real robot has 15 servos -- 4 per leg, 3 per arm, 1
# head. The arm-spread hinges abduct_L/R belong to no real actuator, so the
# abduct links are welded to the torso and their actuators dropped.
WELDED_STATIC = ("abduct_L_link", "abduct_R_link")
DROP_ACTUATORS = ("abduct_L", "abduct_R")

# 6-servo stance: hips, knees and ankles stay actuated (ankles give the
# push-off a gait needs). The other hinges are welded (joint elements removed)
# and every other actuator is dropped.
KEEP_JOINTS = ("L_hip_test", "R_hip_test", "L_knee_test", "R_knee_test",
               "L_ankle_test", "R_ankle_test")
WELD = True

# Collision geometry. The reference enables 26 mesh geoms across the two legs,
# masked so the legs hit each other and nothing else. Mesh-mesh contact on
# long, thin, near-parallel brackets is the usual source of solver blow-up in
# legged RL, and the reference never needed ground contact at all because its
# limbs were pinned to the world. So the meshes stay as visual geometry and the
# feet get one box each, which is what the previous model did and what legged
# RL normally does. The kinematics, mass and pivots are untouched.
FOOT_BOX_MIN_HALF_Z = 0.005
FOOT_COLLISION_RGBA = "0.9 0.2 0.9 1"


def reference_xml(dst):
    sys.path.insert(0, ROBOT_ITEM)
    import chest_both_arms_legs_move as ref  # noqa: E402

    return ref.build(dst, True)


def world_poses(path):
    """World position and rotation of every body at the zero pose."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            out[name] = (data.xpos[i].copy(), data.xmat[i].reshape(3, 3).copy())
    return out


def quat_from_mat(mat):
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    return q


def vec_str(v):
    return " ".join(f"{x:.9g}" for x in v)


def reparent(poses, child, parent):
    """Child's pos/quat expressed in the parent's frame, from world poses."""
    cpos, cmat = poses[child]
    ppos, pmat = poses[parent]
    rel_pos = pmat.T @ (cpos - ppos)
    rel_mat = pmat.T @ cmat
    return vec_str(rel_pos), vec_str(quat_from_mat(rel_mat))


def build_tree(root):
    bodies = {}
    parent = {}
    for b in root.iter("body"):
        name = b.get("name")
        bodies[name] = b
    for b in root.iter("body"):
        for c in b.findall("body"):
            parent[c.get("name")] = b.get("name")

    final = {}
    for name in bodies:
        if name in WELDED_WAIST:
            final[name] = WELDED_WAIST[name]
        elif name in LIMB_ROOTS or name in TORSO_PARTS:
            final[name] = ROOT_BODY
        elif name in parent:
            final[name] = parent[name]
    return bodies, final


# Foot sites for the clearance and slip rewards. The reference has no sites, so
# they are placed at the sole of each foot: the lowest point of the foot's own
# collision mesh, expressed in the foot body frame.
FOOT_BODIES = {"L_foot": "left_foot", "R_foot": "right_foot"}


def sole_sites(path):
    """Body-frame AABB of each foot body's own geoms.

    Taken from the mesh vertices rather than the geom's bounding sphere, so the
    box lands on the real footprint whatever rotation the body and the geom
    carry. This model's left-right axis is X, not Y, so guessing the axes put
    the two foot boxes on top of each other and blew the solver up.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(path)
    out = {}
    for body_name, site_name in FOOT_BODIES.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid:
                continue
            mid = model.geom_dataid[g]
            if mid < 0:
                continue
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = model.mesh_vert[adr : adr + num]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[g])
            local = verts @ rot.reshape(3, 3).T + model.geom_pos[g]
            lo = np.minimum(lo, local.min(0))
            hi = np.maximum(hi, local.max(0))
        if not np.isfinite(lo).all():
            raise SystemExit(f"no mesh geoms on {body_name}")
        out[body_name] = (site_name, lo, hi)
    return out


def generate(dst, mesh_dir):
    ref_path = reference_xml("/tmp/_twoleg_ref_build.xml")
    tree = ET.parse(ref_path)
    root = tree.getroot()
    poses = world_poses(ref_path)

    bodies, final_parent = build_tree(root)
    if ROOT_BODY not in bodies:
        raise SystemExit(f"root body {ROOT_BODY!r} not found")
    missing = [n for n in bodies if n != ROOT_BODY and n not in final_parent]
    if missing:
        raise SystemExit(f"bodies with no assigned parent: {missing}")

    # Reposition every body that changes parent, before copying anything.
    old_parent = {}
    for b in root.iter("body"):
        for c in b.findall("body"):
            old_parent[c.get("name")] = b.get("name")

    for name, parent_name in final_parent.items():
        if old_parent.get(name) != parent_name:
            pos, quat = reparent(poses, name, parent_name)
            el = bodies[name]
            el.set("pos", pos)
            el.set("quat", quat)

    # Rebuild from copies so no element is ever in two places.
    flat = {}
    for name, el in bodies.items():
        new = ET.Element("body", dict(el.attrib))
        for child in el:
            if child.tag != "body":
                new.append(copy.deepcopy(child))
        flat[name] = new

    kids = {name: [] for name in bodies}
    for name, parent_name in final_parent.items():
        kids[parent_name].append(name)

    def attach(parent_el, parent_name):
        for child_name in kids[parent_name]:
            parent_el.append(flat[child_name])
            attach(flat[child_name], child_name)

    worldbody = root.find("worldbody")
    for child in list(worldbody):
        worldbody.remove(child)
    base = ET.Element("body", {"name": BASE_BODY, "pos": "0 0 0", "quat": "1 0 0 0"})
    base.append(ET.Element("freejoint", {"name": "root_joint"}))
    base.append(ET.Element("site", {"name": "imu", "pos": "0 0 0.04", "size": "0.001"}))
    base.append(flat[ROOT_BODY])
    # Front indicator: red arrow welded to the free root, pointing at the
    # model's forward (-y, face side) so a side-on vs front-on render is unambiguous.
    base.append(ET.Element("geom", {
        "name": "front_shaft", "type": "cylinder", "size": "0.004 0.07",
        "pos": "0 -0.10 0.25", "quat": "0.7071068 0.7071068 0 0",
        "rgba": "1 0.1 0.1 1", "contype": "0", "conaffinity": "0",
    }))
    base.append(ET.Element("geom", {
        "name": "front_tip", "type": "cylinder", "size": "0.012 0.015",
        "pos": "0 -0.19 0.25", "quat": "0.7071068 0.7071068 0 0",
        "rgba": "1 0.1 0.1 1", "contype": "0", "conaffinity": "0",
    }))
    worldbody.append(base)
    attach(flat[ROOT_BODY], ROOT_BODY)

    # Drop the duplicate waist hinges and their actuators. Each waist servo is
    # modelled twice in the reference, once on the horn branch (waist_L) and
    # once on the leg branch (L_waist_test), 0.6 mm apart on one shaft. Keeping
    # both would give the leg two serial hinges in the same place. The leg-side
    # hinge stays; the horn branch is welded into the leg.
    #
    # Then weld everything except the four hip/ankle hinges: the 4-servo
    # configuration used in this debugging/teaching run. Every mubljoint
    # transformation you kept is still valid (joint transforms stay in the
    # body's pos/quat wrt parent); a nested joint removed just makes the
    # subtree rigid with the parent's frame.
    for body_name in list(WELDED_WAIST) + list(WELDED_STATIC):
        el = flat.get(body_name)
        if el is None:
            continue
        for j in list(el):
            if j.tag == "joint":
                el.remove(j)
    for act in root.findall("actuator"):
        for a in list(act):
            drop_names = DUPLICATE_WAIST_JOINTS + DROP_ACTUATORS
            if a.get("name") in drop_names or (WELD and a.get("name") not in KEEP_JOINTS):
                act.remove(a)

    # Compiler and model tag. Drop the reference's solver options: mjlab owns
    # the timestep and integrator and warns when the model disagrees.
    for old in root.findall("option"):
        root.remove(old)
    compiler = root.find("compiler")
    compiler.set("meshdir", mesh_dir)
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    root.set("model", "twoleg")

    if WELD:
        for body in root.iter("body"):
            for j in list(body):
                if j.tag == "joint" and j.get("name") not in KEEP_JOINTS:
                    body.remove(j)
    # Regularise every hinge. See JOINT_ARMATURE above.
    for joint in root.iter("joint"):
        if joint.get("type", "hinge") == "freejoint":
            continue
        joint.set("damping", JOINT_DAMPING)
        joint.set("frictionloss", JOINT_FRICTIONLOSS)
        joint.set("armature", JOINT_ARMATURE)

    # Collision: clear every geom, then give each foot one box at the sole.
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    # Foot site and collision box from each foot's real body-frame AABB.
    for body_name, (site_name, lo, hi) in sole_sites(ref_path).items():
        el = flat[body_name]
        for old in list(el):
            if old.tag == "site":
                el.remove(old)
        center = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        # The plate should read long along the walk direction (+y). The raw
        # mesh AABB is long along x due to how the part was exported, which
        # would make both pink slabs overlap about 5 cm at spawn.
        half[0], half[1] = half[1], half[0]
        # Thicken the slab so a 2 mm plate cannot tunnel through the floor.
        half[2] = max(half[2], FOOT_BOX_MIN_HALF_Z)
        center[2] = lo[2] + half[2]
        el.append(ET.Element("site", {"name": site_name, "pos": vec_str([center[0], center[1], lo[2]])}))
        el.append(
            ET.Element(
                "geom",
                {
                    "name": f"{site_name}_collision",
                    "type": "box",
                    "size": vec_str(half),
                    "pos": vec_str(center),
                    "contype": "1",
                    "conaffinity": "1",
                    "density": "0",
                    "rgba": FOOT_COLLISION_RGBA,
                },
            )
        )

    # mjlab sensors.
    for old in root.findall("sensor"):
        root.remove(old)
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "gyro", {"name": "imu_ang_vel", "site": "imu"})
    ET.SubElement(sensor, "velocimeter", {"name": "imu_lin_vel", "site": "imu"})
    ET.SubElement(sensor, "subtreeangmom", {"name": "root_angmom", "body": BASE_BODY})

    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def report(path):
    import mujoco

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pinned = [i for i in range(1, model.nbody) if model.body_parentid[i] == 0]
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    print(f"  bodies      : {model.nbody - 1}")
    print(f"  joints      : {model.njnt}  (freejoints {len(free)})")
    print(f"  actuators   : {model.nu}")
    print(f"  geoms       : {model.ngeom}")
    print(f"  mass        : {model.body_mass.sum():.4f} kg")
    print(f"  world-pinned: {len(pinned)}")
    lowest = min(
        data.geom_xpos[g][2]
        - (
            model.geom_size[g][2]
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            else model.geom_rbound[g]
        )
        for g in range(model.ngeom)
    )
    print(f"  lowest point: {lowest:.4f} m  (spawn the torso near {-lowest:.4f} m)")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    print(f"  joints      : {sorted(n for n in names if n)}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify without writing")
    args = ap.parse_args()
    for dst, mesh_dir in OUTPUTS:
        target = dst + ".tmp" if args.check else dst
        generate(target, mesh_dir)
        print(f"wrote {os.path.relpath(target, ROOT)}  (meshdir={mesh_dir})")
        if not args.check:
            print("compiled:")
            report(target)
    if args.check:
        for dst, _ in OUTPUTS:
            if os.path.exists(dst + ".tmp"):
                os.remove(dst + ".tmp")


if __name__ == "__main__":
    main()
