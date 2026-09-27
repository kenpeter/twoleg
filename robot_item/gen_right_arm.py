#!/usr/bin/env python3
"""Generate the right arm models in this repo from their left arm sources.

Target 1, the chest: mirror the left_arm body inside chest_assembly_step6.xml into a
right_arm body in the same file.  The step6 file used to carry one arm.  Adding the
right one by hand is how the two copies drift apart, and this corpus has already paid
for that once: the left arm lost horn_s0..s3 and shoulder_shaft* and the right arm did
not, which is exactly the asymmetry test_lr_twin_equivalent exists to catch.  So the
right arm is generated, not typed, and this script is the record of how.

The chest's right arm cannot be copied from chest_both_arms.xml: that file mounts its
arms at pos x=+/-0.065, while step6 mounts the left one at x=-0.114137 and z=0.090995
with a different orientation.  A copy would land the right arm nowhere near its own
shoulder.  The only correct source is step6's own left arm.

Target 2, the part: mirror left_arm.xml into the standalone right_arm.xml part model.
Its source is the part file, not the chest, and it follows the neighbouring part files
instead: names stay unsuffixed (right_tricep.xml, thigh_assembly_right.xml) and both
sides share the same STL files.

One reflection, two planes.  A reflection is S = diag(s1, s2, s3) with each si in
{+1, -1}.  Composing a rotation with a reflection keeps it a rotation but reverses the
angle, so for every pos and quat in the subtree:

    pos  t     -> S t
    quat (w, v) -> (w, det(S) * S * v)

det(S) is the factor that is easy to lose, so it is written out instead of being folded
into the signs by hand.  The sign triple drives both targets:

    S = (-1, +1, +1)  the chest   pos -> (-x,  y,  z)   quat (w, x, y, z) -> (w, x, -y, -z)
    S = (+1, -1, +1)  the part    pos -> ( x, -y,  z)   quat (w, x, y, z) -> (w, -x, y, -z)

Each target mirrors its whole subtree, top body and children both, because a child's
transform is relative to its parent: mirroring only the top body would leave the
children un-mirrored inside a mirrored frame.

The parts are chiral, so a mirrored subtree needs the mesh scale negated on the reflection
normal.  A geom draws its world vertices as (S . v) . R_geom' + p, and reflecting the
subtree conjugates each local transform by R, so the mirrored geom's vertices land where
the left side's land under R only where S == R.  S == I is a rotated copy of the part
wearing the right name, and that is the bug this docstring used to sit on.  diag(1, -1, 1)
turns scale="0.001 0.001 0.001" into "0.001 -0.001 0.001"; diag(-1, 1, 1) negates x.

The two targets need that negation in different places, because they share their meshes
differently.  The part's <asset> block is its own, so the part flips its scales in place
and keeps one mesh name per STL, which is what the neighbouring part files do.  The
chest's <asset> block belongs to the whole chest model, where the left arm and the 斜U
wings use those same names, so flipping them in place would un-mirror the left arm as
well.  The chest therefore gives every mesh the arm references its own twin,

    <mesh name="{mesh}_r" file="{the base mesh's file}" scale="{base scale . CHEST_PLANE}"/>

appended to the end of the asset block, and re-points the right arm's geoms at the twins.
The twins share their STL file, so the chest gains no geometry, and the rule is keyed on
the arm's own mesh set, so every chest mesh the arm does not reference, slantU_r
included, is left alone.  That is the convention chest_both_arms.xml already follows by
hand for the same six parts.

Every name in the chest's subtree gets the _R suffix, matching chest_both_arms.xml.
The part keeps its source names, matching the other part files.

The part's header comment is PART_HEADER below, because xml.etree.ElementTree drops comments
on parse, so the record has to be authored as a string and emitted ahead of the serialized
tree.  The chest has no such string: ET.parse discards its comments too, header included,
so a regenerated chest carries none.  XML 1.0 forbids "--" inside a comment and expat
enforces it, so that header names the mode in words rather than spelling the two-hyphen
flag.

Run: python3 gen_right_arm.py                # write the chest
     python3 gen_right_arm.py --check        # report the chest, write nothing
     python3 gen_right_arm.py --part         # write right_arm.xml
     python3 gen_right_arm.py --part --check
"""
import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "chest_assembly_step6.xml")
PART_SOURCE = os.path.join(HERE, "left_arm.xml")
PART_TARGET = os.path.join(HERE, "right_arm.xml")
CHEST_PLANE = (-1, 1, 1)
PART_PLANE = (1, -1, 1)
SUFFIX = "_R"
MESH_SUFFIX = "_r"
MIRRORED_ATTRS = ("body", "geom", "site", "freejoint")
XML_DECL = "<?xml version='1.0' encoding='utf-8'?>\n"

PART_HEADER = """  <!-- right_arm: the sagittal mirror of the left arm part, S = diag(1, -1, +1),
       i.e. Y -> -Y, applied to left_arm.xml by gen_right_arm.py in part mode.
       Do not hand-edit this file.  Edit left_arm.xml or the generator and rerun it.
       Transform rules, applied to every body and geom in the worldbody tree:
         pos  (x, y, z) -> (x, -y, z)
         quat (w, x, y, z) -> (w, -x, y, -z)
         mesh scale (sx, sy, sz) -> (sx, -sy, sz)
       Body and geom names are unchanged and every mesh keeps its left_arm.xml name
       and file=, so this part shares the STL files with the left arm.  That is the
       convention the neighbouring part files already use.  It is not the convention
       of the chest mirror in gen_right_arm.py, which reflects about the other plane,
       diag(-1, 1, 1), and suffixes every name with _R, so do not read one target's
       naming or numbers as the other's.
       工程记录, every Y value mirrored from left_arm.xml.  长U bracket: base plate on
       top bolted under the multi, prongs down.  M3 screws + nuts through the 长U
       base-plate holes into the 多功能 bottom (visual only).  Horn retaining screws
       (visual only), through the 4 disc holes.  Shoulder pivot shaft on the arm-Y
       axis at X=0.01275 Z=0.010, shaped to the 金属舵盘 central socket so the disc is
       centred on the 舵机 output and the servo can rotate the arm out/in without the
       horn cocking in the bore.  Measured socket: D9 (R4.5) Y 4.5..7.5, D6 (R3.0)
       Y 7.5..8.0, D3 (R1.5) Y 8.0..9.5.  The shaft is a D9 shoulder filling the
       mouth, a D6 pilot in the step, and a D3 tip running through to protrude
       1.1 mm past the horn's outer face.  Elbow pivot (Y axis) at the 长U prong A
       bore (X=0.01325, Z=-0.08125): flange bearing centred in the prong and a 4 mm
       shaft through prong + the 金属舵盘 bolted on its outer face.  The other prong
       (B) carries a matching bearing and a 4 mm shaft from the 多功能 far face
       (Y=-0.040) through prong B (Y -0.043..-0.049), so both sides of the U connect
       to the forearm hardware.  The 多功能 sits inside the 长U prongs
       (Y -0.040..-0.003, clear of both prongs) and the 金属舵盘 bolts flat onto prong
       A's outer face (Y 0.005..0.010), its 4x M3 diamond on that prong's bore axis
       (X=0.01325, Z=-0.08125) with the 舵机 body spanning the gap between them.
       Right arm = the mirrored left_tricep unit (upper arm: 多功能 + 舵机 + 金属舵盘 +
       长U) joined to the mirrored left_forearm unit (多功能 + 舵机 + 金属舵盘 + 一字)
       per manual p.10 "三.1 左右手臂的拼接 · 右手".  The elbow is the 长U prong bore at
       Y=-0.001..0.005 (X=0.01325, Z=-0.08125, axis along Y): the forearm's 多功能
       stands inside the prongs on that bore axis, and its 金属舵盘 lines its 4x M3
       diamond up with the prong's identical diamond.  The 一字 keeps the disc's other
       face and both units stay in their mirrored standalone frames, so right_arm =
       tricep + T_elbow(forearm).
       Parts are dimensioned in millimetres and MJCF is in metres, hence every mesh
       scale of 0.001, with the Y component negated for this mirror. -->
"""


def _vector(el, attr, count):
    """Parse a fixed-width numeric attribute, or fail with the offending text."""
    text = el.get(attr)
    if text is None:
        sys.exit(f"{el.tag} name={el.get('name')!r} has no {attr} attribute")
    parts = text.split()
    if len(parts) != count:
        sys.exit(f"{el.tag} {attr}={text!r} needs {count} numbers, got {len(parts)}")
    return [float(v) for v in parts]


def _fmt(values):
    return " ".join(f"{v:.6f}" for v in values)


def _reflect(values, signs):
    return [s * v for s, v in zip(signs, values)]


def mirror_pos(el, signs):
    if el.get("pos") is None:
        return
    el.set("pos", _fmt(_reflect(_vector(el, "pos", 3), signs)))


def mirror_quat(el, signs):
    if el.get("quat") is None:
        return
    w, vx, vy, vz = _vector(el, "quat", 4)
    det = signs[0] * signs[1] * signs[2]
    el.set("quat", _fmt([w] + [det * c for c in _reflect((vx, vy, vz), signs)]))


def mirror_tree(el, signs):
    """Mirror this element's own transform and everything under it, in place."""
    mirror_pos(el, signs)
    mirror_quat(el, signs)
    for child in el:
        if child.tag in MIRRORED_ATTRS:
            mirror_tree(child, signs)
    return el


def mirror_mesh_scales(root, signs):
    """Reflect each mesh scale, which negates the component on the reflection normal."""
    for mesh in root.iter("mesh"):
        if mesh.get("scale") is None:
            continue
        mesh.set("scale", _fmt(_reflect(_vector(mesh, "scale", 3), signs)))


def mirror_mesh_names(body):
    """The distinct mesh names any geom in this subtree references, sorted for determinism."""
    return sorted({g.get("mesh") for g in body.iter("geom") if g.get("mesh")})


def mirror_meshes(root, body, signs):
    """Declare a mirrored twin per mesh this subtree uses and re-point its geoms at them.

    A geom's world vertices are (S . v) . R_geom' + p and the reflection conjugates each
    local transform by R, so the mirrored geom matches the left side under R only where
    S == R.  So a twin is the base mesh's own file at the base scale reflected by signs.
    Twins are keyed on the subtree's own mesh set, so meshes it does not reference are not
    touched, and an existing twin is re-scaled where it stands rather than duplicated,
    because MuJoCo rejects a duplicate asset name at compile time.

    Returns the twin names, sorted, so the run can say what it mirrored."""
    asset = root.find("asset")
    if asset is None:
        sys.exit(f"{os.path.basename(TARGET)} has no <asset> block to mirror meshes into")
    declared, dupes = {}, set()
    for mesh in asset.findall("mesh"):
        name = mesh.get("name")
        if name is None:
            continue
        if name in declared:
            dupes.add(name)
        declared[name] = mesh
    if dupes:
        sys.exit(f"{os.path.basename(TARGET)}: <asset> declares {', '.join(sorted(dupes))} "
                 f"more than once, which MuJoCo rejects at compile time")

    renames = {}
    for name in mirror_mesh_names(body):
        base = declared.get(name)
        if base is None:
            sys.exit(f"{os.path.basename(TARGET)}: geom mesh {name!r} has no <asset> entry, "
                     f"so there is no file= for a twin to share.  Add a <mesh name={name!r} "
                     f"file=... /> line to the asset block and rerun.")
        if not base.get("file"):
            sys.exit(f"{os.path.basename(TARGET)}: asset mesh {name!r} declares no file=, "
                     f"so a twin cannot share one")
        twin = name + MESH_SUFFIX
        scale = _fmt(_reflect(_vector(base, "scale", 3), signs))
        if twin not in declared:
            declared[twin] = ET.SubElement(asset, "mesh",
                                           {"name": twin, "file": base.get("file")})
        declared[twin].set("scale", scale)
        renames[name] = twin

    for geom in body.iter("geom"):
        if geom.get("mesh") in renames:
            geom.set("mesh", renames[geom.get("mesh")])
    return sorted(renames.values())


def suffix_names(el):
    if el.get("name"):
        el.set("name", el.get("name") + SUFFIX)
    for child in el:
        if child.tag in MIRRORED_ATTRS:
            suffix_names(child)
    return el


def existing_right_arm(wb):
    for body in wb.findall("./body"):
        if body.get("name") == "right_arm":
            return body
    return None


def worldbody(root, source):
    wb = root.find("worldbody")
    if wb is None:
        sys.exit(f"{os.path.basename(source)} has no worldbody to mirror")
    return wb


def read_on_disk(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return fh.read()


def report_drift(path, before, data):
    """Report the generated text against what was on disk before this run."""
    name = os.path.basename(path)
    if before is None:
        print(f"{name}: not on disk yet, so no drift to report")
    elif before == data:
        print(f"{name}: on disk matches the generated text, no drift")
    else:
        print(f"{name}: on disk DIFFERS from the generated text, drift")


def build(apply_changes):
    if not os.path.exists(TARGET):
        sys.exit(f"missing {TARGET}")
    tree = ET.parse(TARGET)
    root = tree.getroot()
    wb = worldbody(root, TARGET)
    left = next((b for b in wb.findall("./body")
                 if b.get("name") == "left_arm"), None)
    if left is None:
        sys.exit("chest_assembly_step6.xml has no left_arm to mirror")

    prior = existing_right_arm(wb)
    if prior is not None:
        wb.remove(prior)
        print("removed the previous right_arm so this run is idempotent")

    right = suffix_names(mirror_tree(copy.deepcopy(left), CHEST_PLANE))
    right.set("name", "right_arm")
    wb.insert(list(wb).index(left) + 1, right)
    twins = mirror_meshes(root, right, CHEST_PLANE)

    bodies = sum(1 for e in right.iter() if e.tag == "body")
    geoms = sum(1 for e in right.iter() if e.tag == "geom")
    print(f"right_arm: pos={right.get('pos')} quat={right.get('quat')}")
    print(f"  {bodies} bodies, {geoms} geoms, all names suffixed {SUFFIX!r}")
    print(f"  {len(twins)} mesh(es) mirrored as {', '.join(twins)}")

    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    before = read_on_disk(TARGET)
    if apply_changes:
        with open(TARGET, "wb") as fh:
            fh.write(data)
        print(f"wrote {os.path.basename(TARGET)}")
    else:
        print("--check: nothing written")
    report_drift(TARGET, before, data)
    return right


def build_part(apply_changes):
    if not os.path.exists(PART_SOURCE):
        sys.exit(f"missing {PART_SOURCE}")
    root = ET.parse(PART_SOURCE).getroot()
    wb = worldbody(root, PART_SOURCE)
    mirror_tree(wb, PART_PLANE)
    mirror_mesh_scales(root, PART_PLANE)
    root.set("model", "right_arm")

    # Count inside worldbody only: the <default> block also holds two <geom> tags.
    bodies = sum(1 for e in wb.iter() if e.tag == "body")
    geoms = sum(1 for e in wb.iter() if e.tag == "geom")
    print(f"right_arm: mirrored {os.path.basename(PART_SOURCE)}, S = diag{PART_PLANE}")
    print(f"  {bodies} bodies, {geoms} geoms, names unchanged")

    tree_text = ET.tostring(root, encoding="unicode")
    data = (XML_DECL + PART_HEADER + tree_text).encode("utf-8")
    before = read_on_disk(PART_TARGET)
    if apply_changes:
        with open(PART_TARGET, "wb") as fh:
            fh.write(data)
        print(f"wrote {os.path.basename(PART_TARGET)}")
    else:
        print("--check: nothing written")
    report_drift(PART_TARGET, before, data)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="build in memory and report, write nothing")
    ap.add_argument("--part", action="store_true",
                    help="target the standalone part model right_arm.xml from "
                         "left_arm.xml instead of the chest")
    args = ap.parse_args()
    if args.part:
        build_part(not args.check)
    else:
        build(not args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
