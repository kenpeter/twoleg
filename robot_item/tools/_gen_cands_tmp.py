#!/usr/bin/env python3
"""Write chest_assembly_step5.xml + left_arm.xml as step6 candidates.

Each candidate is chest_assembly_step5.xml with the arm bodies from
left_arm.xml wrapped in one <body name="left_arm"> carrying the candidate
pos/quat.  Prints the transform table so the winner can be pasted into
chest_assembly_step6.xml by hand.
"""
import copy
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHEST = os.path.join(HERE, "chest_assembly_step5.xml")
ARM = os.path.join(HERE, "left_arm.xml")

EAR_BORE_XZ = np.array([-0.09805, 0.08905])
EAR_MID = {"near": -0.02758, "far": 0.02441}
HORN_BORES = np.array([[0.00575, -0.0085, 0.010], [0.01275, -0.0085, 0.003],
                       [0.01975, -0.0085, 0.010], [0.01275, -0.0085, 0.017]])
HORN_CENTROID = HORN_BORES.mean(axis=0)
NEED = {"straight", "longU", "bearing"}


def quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return [0.25 * s, (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0] * 4
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q


def candidates_hang():
    """Arm hangs straight down: arm Z -> world +Z, arm Y along the prong-bore axis.

    R_A: horn fills the near ear's 5 mm, servo fills the remaining 43 mm.
    R_B: mirrored in Y (horn against the far ear).
    """
    for tag, sx in (("hA", 1.0), ("hB", -1.0)):
        R = np.diag([sx, sx, 1.0])
        face = EAR_MID["near"] + 0.0005 if sx > 0 else EAR_MID["far"] - 0.0005
        ty = face + 0.0095 if sx > 0 else face - 0.0095
        anchor = np.array([HORN_CENTROID[0], 0.0, HORN_CENTROID[2]])
        t = np.array([EAR_BORE_XZ[0], ty, EAR_BORE_XZ[1]]) - R @ anchor
        yield tag, R, t


def candidates():
    d1 = np.array([0.70771, 0.0, -0.70651])
    d2 = np.array([-0.70651, 0.0, -0.70771])
    y = np.array([0.0, 1.0, 0.0])
    for xi, X in enumerate((d1, d2)):
        for ys in (1.0, -1.0):
            Y = y * ys
            Z = np.cross(X, Y)
            R = np.column_stack([X, Y, Z])
            for side in ("near", "far"):
                # seat the horn's OUTER face (arm Y = -0.0095) on the ear's outer face
                face = EAR_MID[side] + (-0.0005 if side == "near" else 0.0005)
                t_y = face + 0.0095 if ys > 0 else face - 0.0095
                anchor = np.array([HORN_CENTROID[0], 0.0, HORN_CENTROID[2]])
                t = np.array([EAR_BORE_XZ[0], t_y, EAR_BORE_XZ[1]]) - R @ anchor
                yield f"c{xi}{'p' if ys > 0 else 'm'}_{side}", R, t


def _sanitize(text):
    """XML comments may not contain '--'; left_arm.xml uses dashed rules."""
    import re

    def fix(mo):
        return "<!--" + re.sub(r"-{2,}", "", mo.group(1)) + "-->"
    return re.sub(r"<!--(.*?)-->", fix, text, flags=re.S)


def build(R, t, out):
    ct = ET.parse(CHEST)
    croot = ct.getroot()
    croot.set("model", os.path.splitext(os.path.basename(out))[0])
    aroot = ET.fromstring(_sanitize(open(ARM, encoding="utf-8").read()))
    have = {m.get("name") for m in croot.findall(".//mesh")}
    for m in aroot.findall(".//mesh"):
        if m.get("name") in NEED and m.get("name") not in have:
            croot.find("asset").append(copy.deepcopy(m))
            have.add(m.get("name"))
    wrap = ET.Element("body", {"name": "left_arm",
                               "pos": " ".join(f"{v:.6f}" for v in t),
                               "quat": " ".join(f"{v:.6f}" for v in quat(R))})
    wb = aroot.find("worldbody")
    for child in list(wb):
        wrap.append(copy.deepcopy(child))
    croot.find("worldbody").append(wrap)
    ct.write(out, encoding="utf-8", xml_declaration=True)


def main():
    for label, R, t in list(candidates()) + list(candidates_hang()):
        out = os.path.join(HERE, f"_cand_{label}.xml")
        build(R, t, out)
        down = R @ np.array([0.0, 0.0, -1.0])
        print(f"{label}: pos=\"{' '.join(f'{v:.6f}' for v in t)}\" "
              f"quat=\"{' '.join(f'{v:.6f}' for v in quat(R))}\"  arm-down={np.round(down,4).tolist()}")


if __name__ == "__main__":
    sys.exit(main())
