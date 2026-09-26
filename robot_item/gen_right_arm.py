#!/usr/bin/env python3
"""Generate the right arm in chest_assembly_step6.xml by mirroring its left arm.

The step6 file used to carry one arm. Adding the right one by hand is how the two copies
drift apart, and this corpus has already paid for that once: the left arm lost
horn_s0..s3 and shoulder_shaft* and the right arm did not, which is exactly the
asymmetry test_lr_twin_equivalent exists to catch. So the right arm is generated, not
typed, and this script is the record of how.

The right arm cannot be copied from chest_both_arms.xml: that file mounts its arms at
pos x=+/-0.065, while step6 mounts the left one at x=-0.114137 and z=0.090995 with a
different orientation. A copy would land the right arm nowhere near its own shoulder.
The only correct source is step6's own left arm.

Reflection is about the world YZ plane, M = diag(-1, 1, 1). Composing a rotation with a
reflection keeps it a rotation but reverses the angle, so for every pos and quat in the
subtree:

    pos  (x, y, z) -> (-x,  y,  z)
    quat (w, x, y, z) -> (w, x, -y, -z)

Both the arm body and its children are mirrored, because a child's transform is relative
to its parent: mirroring only the arm would leave the children un-mirrored inside a
mirrored frame.

Every name in the subtree gets the _R suffix, matching chest_both_arms.xml.

Run: uv run --offline --with numpy --with mujoco python robot_item/gen_right_arm.py
     gen_right_arm.py --check   # verify only, write nothing
"""
import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "chest_assembly_step6.xml")
SUFFIX = "_R"
MIRRORED_ATTRS = ("body", "geom", "site", "freejoint")


def _floats(text):
    return [float(v) for v in text.split()]


def mirror_pos(el):
    if el.get("pos") is None:
        return
    x, y, z = _floats(el.get("pos"))
    el.set("pos", f"{-x:.6f} {y:.6f} {z:.6f}")


def mirror_quat(el):
    if el.get("quat") is None:
        return
    w, x, y, z = _floats(el.get("quat"))
    el.set("quat", f"{w:.6f} {x:.6f} {-y:.6f} {-z:.6f}")


def mirror_tree(el):
    """Mirror this element's own transform and everything under it, in place."""
    mirror_pos(el)
    mirror_quat(el)
    for child in el:
        if child.tag in MIRRORED_ATTRS:
            mirror_tree(child)
    return el


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


def build(apply_changes):
    if not os.path.exists(TARGET):
        sys.exit(f"missing {TARGET}")
    tree = ET.parse(TARGET)
    root = tree.getroot()
    wb = root.find("worldbody")
    left = next((b for b in wb.findall("./body")
                 if b.get("name") == "left_arm"), None)
    if left is None:
        sys.exit("chest_assembly_step6.xml has no left_arm to mirror")

    prior = existing_right_arm(wb)
    if prior is not None:
        wb.remove(prior)
        print("removed the previous right_arm so this run is idempotent")

    right = suffix_names(mirror_tree(copy.deepcopy(left)))
    right.set("name", "right_arm")
    wb.insert(list(wb).index(left) + 1, right)

    bodies = sum(1 for e in right.iter() if e.tag == "body")
    geoms = sum(1 for e in right.iter() if e.tag == "geom")
    print(f"right_arm: pos={right.get('pos')} quat={right.get('quat')}")
    print(f"  {bodies} bodies, {geoms} geoms, all names suffixed {SUFFIX!r}")

    if apply_changes:
        tree.write(TARGET, encoding="utf-8", xml_declaration=True)
        print(f"wrote {os.path.basename(TARGET)}")
    else:
        print("--check: nothing written")
    return right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="build in memory and report, write nothing")
    args = ap.parse_args()
    build(not args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
