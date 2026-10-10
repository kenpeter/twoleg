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

# Single source of truth for the robot model. The same file backs the viewer
# (robot_item/view_part.sh), so physics edits must not be forked per consumer.
TWOLEG_XML: Path = Path(__file__).resolve().parents[1] / "robot_item" / "xml" / "robot_twoleg.xml"
# The xml references meshes via "/home/kenpeter/work/twoleg/robot_item" (absolute).
TWOLEG_MESH_DIR: Path = Path("/home/kenpeter/work/twoleg/robot_item")

# Foot site / body names used by the gait rewards + contact sensor.
FOOT_SITES = ("left_foot", "right_foot")
FOOT_BODIES = ("L_foot", "R_foot")
TORSO_BODY = "torso"

# Duck servo actuator group: all 8 leg joints share one BuiltinPositionActuator
# (mirrors G1's per-group BuiltinPositionActuatorCfg). Stiffness/damping/effort
# are duck-scaled (small 0.2 m, 1.26 kg robot).
# kp/damping aligned to BipedRobot's measured actuator gains (kp 21.1, bias_damp
# 0.0). Effort stays at 1.0: raising it to 5.0 moved root height under 2 cm in a
# 3 s rollout, so effort is not what stops this robot standing.
TWOLEG_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "L_hip_roll_test", "L_hip_test", "L_knee_test", "L_ankle_test",
        "R_hip_roll_test", "R_hip_test", "R_knee_test", "R_ankle_test",
    ),
    stiffness=21.1,
    damping=0.0,
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
    """Build the TwoLeg entity config from robot_twoleg.xml."""
    # Spawn height: the lowest colliding geom (the foot boxes) bottoms out at
    # -0.2510 m with the root at the world origin, so the root starts 0.251 m up
    # or the contact solver ejects the robot on step 0.
    #
    # The stance is an ANKLE correction, not a knee bend. Measured over a 3 s
    # rollout (gate: final tilt < 15 deg), zero-pose tops out at 0.7 deg and holds
    # for 10 s at dz -0.0033 m, while any knee > 0 falls (65-142 deg). The ankle
    # value is tolerant to about +-0.05 before it tips past 60 deg, so -0.5 sits
    # in the middle of the standing basin.
    STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.251),
        joint_pos={
            ".*_hip_test": 0.0,
            ".*_knee_test": 0.0,
            ".*_ankle_test": -0.5,
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
            # Foot/terrain contact: full friction (condim=3), highest priority.
            CollisionCfg(
                geom_names_expr=(r".*_foot_collision$",),
                condim=3,
                conaffinity=1,
                contype=1,
                priority=1,
                friction=(0.6,),
            ),
            # Leg-link self-collision: thin capsules (one per main link) get
            # condim=1 so legs can't interpenetrate each other. The decorative
            # mesh colliders stay contype=0 (XML default), so only these
            # capsules collide -- no overlap-at-rest explosion.
            CollisionCfg(
                geom_names_expr=(r".*_(hip|knee|ankle)_capsule$",),
                condim=1,
                conaffinity=1,
                contype=1,
                priority=0,
            ),
        ),
    )

