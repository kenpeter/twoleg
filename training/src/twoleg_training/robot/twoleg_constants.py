"""TwoLeg robot configuration for mjlab.

Wraps ``training/assets/twoleg.xml`` as an mjlab ``EntityCfg``. That file is
generated from the reference assembly by
``training/scripts/generate_twoleg_model.py``; do not edit it by hand. The
XML's own ``<position>`` actuators are reused via ``XmlActuatorCfg``, so no
gains are re-specified here.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

TWOLEG_XML: Path = Path(__file__).resolve().parents[3] / "assets" / "twoleg.xml"
assert TWOLEG_XML.exists(), f"XML not found: {TWOLEG_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(TWOLEG_XML))


# All 15 actuated joints: head, both arms (shoulder, elbow, wrist) and both
# legs (waist, hip, knee, ankle). Order matches the generated actuator list.
# Everything else is welded in training/assets/twoleg.xml; see
# training/scripts/generate_twoleg_model.py KEEP_JOINTS/WELD.
JOINT_NAMES: tuple[str, ...] = (
    "head",
    "shoulder_test",
    "elbow_test",
    "wrist_test",
    "shoulder_test_R",
    "elbow_test_R",
    "wrist_test_R",
    "L_waist_test",
    "L_ankle_test",
    "L_knee_test",
    "L_hip_test",
    "R_waist_test",
    "R_ankle_test",
    "R_knee_test",
    "R_hip_test",
)

# HOME pose = the model's own zero pose (straight legs, arms down), with the
# torso at the height that puts the feet on the ground. The generated model's
# lowest point sits at -0.3197 m from the torso origin.
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.32),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)

# Collision geoms live in the XML (the reference assembly's leg and foot
# meshes, with the ground bit added), so no CollisionCfg is needed.
# Target the joints by name: a catch-all ".*" also matches the imu/foot sites and
# makes mjlab warn that the actuator config may be aiming at the wrong namespace.
TWOLEG_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=JOINT_NAMES),),
    soft_joint_pos_limit_factor=0.9,
)


def get_twoleg_robot_cfg() -> EntityCfg:
    """Fresh TwoLeg robot config (avoids shared-mutation issues)."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(),
        spec_fn=get_spec,
        articulation=TWOLEG_ARTICULATION,
    )


if __name__ == "__main__":
    spec = get_spec()
    spec.compile()
    print(f"TwoLeg spec OK: {len(spec.joints)} joints, {len(spec.actuators)} actuators")
