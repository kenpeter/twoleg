"""TwoLeg (0.2 m duck biped) robot asset wiring for the mjlab velocity task.

Mirrors unitree_rl_mjlab's ``unitree_g1`` constants: provide the robot EntityCfg
(via a spec_fn returning a mujoco.MjSpec) and the per-joint action scale. The
duck has 6 actuated joints (L/R hip, knee, ankle) and two foot bodies
(L_foot / R_foot) with contact-collision geoms and sites ``left_foot`` /
``right_foot``.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

TWOLEG_XML: Path = Path(__file__).resolve().parent / "twoleg.xml"
# The xml references meshes via "/home/kenpeter/work/twoleg/robot_item" (absolute).
TWOLEG_MESH_DIR: Path = Path("/home/kenpeter/work/twoleg/robot_item")

# Foot site / body names used by the gait rewards + contact sensor.
FOOT_SITES = ("left_foot", "right_foot")
FOOT_BODIES = ("L_foot", "R_foot")
TORSO_BODY = "torso"

# Duck servo actuator group: all 6 leg joints share one BuiltinPositionActuator
# (mirrors G1's per-group BuiltinPositionActuatorCfg). Stiffness/damping/effort
# are duck-scaled (small 0.2 m, 1.26 kg robot).
TWOLEG_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "L_hip_test", "L_knee_test", "L_ankle_test",
        "R_hip_test", "R_knee_test", "R_ankle_test",
    ),
    stiffness=40.0,
    damping=0.5,
    effort_limit=1.0,
    armature=0.01,
)

TWOLEG_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(TWOLEG_ACTUATOR,),
)

# Per-joint action scale (rad). Unitree uses 0.25 as the base; the duck's small
# servos stay within the same range and the policy learns the gait from there.
TWOLEG_ACTION_SCALE: dict[str, float] = {
    "L_hip_test": 0.25,
    "L_knee_test": 0.25,
    "L_ankle_test": 0.25,
    "R_hip_test": 0.25,
    "R_knee_test": 0.25,
    "R_ankle_test": 0.25,
}


def get_spec() -> "mujoco.MjSpec":
    """Load the TwoLeg MJCF into a mujoco.MjSpec with meshes resolved."""
    spec = mujoco.MjSpec.from_file(str(TWOLEG_XML))
    return spec


def get_twoleg_robot_cfg() -> EntityCfg:
    """Build the TwoLeg entity config from twoleg.xml."""
    return EntityCfg(
        spec_fn=get_spec,
        articulation=TWOLEG_ARTICULATION,
    )

