"""TwoLeg robot configuration for mjlab.

Wraps ``training/assets/twoleg.xml`` (a copy of ``twoleg_mjcf/robot_twoleg.xml``
plus an IMU site and built-in sensors) as an mjlab ``EntityCfg``. The XML's own
``<position>`` actuators are reused via ``XmlActuatorCfg``, so no gains are
re-specified here.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

TWOLEG_XML: Path = Path(__file__).resolve().parents[3] / "assets" / "twoleg.xml"
assert TWOLEG_XML.exists(), f"XML not found: {TWOLEG_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(TWOLEG_XML))


# 15 actuated joints, in XML actuator order.
JOINT_NAMES: tuple[str, ...] = (
    "head_yaw",
    "left_shoulder_pitch",
    "left_elbow",
    "left_shoulder_roll",
    "right_shoulder_pitch",
    "right_elbow",
    "right_shoulder_roll",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

# HOME pose = the STAND keyframe of robot_twoleg.xml: straight legs, arms tucked.
# Regex keys are matched in order; the final ".*" is the catch-all.
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.165),
    joint_pos={
        r"left_shoulder_roll": -0.4,
        r"right_shoulder_roll": 0.4,
        r"left_elbow": -0.62,
        r"right_elbow": 0.62,
        r".*": 0.0,
    },
    joint_vel={".*": 0.0},
)

# The XML already gives the two foot boxes collision geoms and leaves the meshes
# as visual-only, so no CollisionCfg is needed.
TWOLEG_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=(r".*",)),),
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
