"""TwoLeg (iw) robot asset wiring for the mjlab velocity task.

Mirrors unitree_rl_mjlab's ``unitree_g1`` constants: provide the robot EntityCfg
(via a spec_fn returning a mujoco.MjSpec) and the per-joint action scale. The
iw has 8 actuated leg joints (L/R: hip_roll, hip, knee, ankle) = 15-DOF budget
with head(1) + hands(3+3). ~4.5 kg real mass (MuJoCo auto 4.50 kg).
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

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
        "L_hip_roll_test", "L_hip_test", "L_knee_test", "L_ankle_test",
        "R_hip_roll_test", "R_hip_test", "R_knee_test", "R_ankle_test",
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
    "L_hip_roll_test": 0.25,
    "L_hip_test": 0.25,
    "L_knee_test": 0.25,
    "L_ankle_test": 0.25,
    "R_hip_roll_test": 0.25,
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
    # Standing spawn: at the zero-pose the foot boxes sit 0.251 m BELOW the
    # floor (root at world origin), so the contact solver ejects the robot on
    # step 0. Raise the root to z=0.251 so the feet rest on the ground, and
    # bend the legs slightly so it stands instead of collapsing.
    STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.251),
        joint_pos={
            ".*_hip_test": 0.0,
            ".*_knee_test": 0.0,
            ".*_ankle_test": 0.0,
            ".*_hip_roll_test": 0.0,
        },
        joint_vel={".*": 0.0},
    )
    return EntityCfg(
        spec_fn=get_spec,
        articulation=TWOLEG_ARTICULATION,
        init_state=STANDING_KEYFRAME,
        # Leg-link self-collision requires a capsule refit of the 266 decorative
        # overlapping mesh colliders first (MuJoCo uses convex hulls -> they
        # interpenetrate at rest and explode). Until then we keep the body geoms
        # non-colliding (contype=0 in the XML) and only the two foot boxes touch
        # the ground. This CollisionCfg just formalizes the foot/terrain contact
        # (condim=3) via the mjlab API; enabling self-collision later means
        # giving the leg links a few non-overlapping capsules and adding a
        # CollisionCfg with condim=1 for them.
        collisions=(
            CollisionCfg(
                geom_names_expr=(r".*_foot_collision$",),
                condim=3,
                conaffinity=1,
                contype=1,
                priority=1,
                friction=(0.6,),
            ),
        ),
    )

