"""TwoLeg velocity (slow-walk) environment.

A deliberately small version of mjlab's velocity task, following the
``/home/kenpeter/work/microduck_rl`` recipe but stripped to the essentials:

* flat plane, per-foot height and contact sensors feeding the gait rewards
* velocity command tracking (slow, forward-biased) + upright + joint posture
* all 15 joints actuated with the XML position servos
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.rewards import self_collision_cost
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from twoleg_training.robot.twoleg_constants import get_twoleg_robot_cfg
from twoleg_training.tasks.curriculum import air_time_window, walk_command_ramp
from twoleg_training.tasks.mdp import reward_weight, standing_envs_curriculum

# Slow-walk command envelope. Forward is body -y (face side: chest servos /
# wide head box): the leg kinematics
# (hip/knee hinge axes local Y, chain quats keep Y world-Y) swing the feet
# in the YZ plane, and +x of this body is across the stance.
LIN_VEL_X = (-0.4, 0.4)  # lateral
LIN_VEL_Y = (-0.3, 0.3)  # forward (body -y, face side)
ANG_VEL_Z = (-1.0, 1.0)

# Gait-shaping constants ported from the microduck_rl velocity task.
FOOT_SITES = ("left_foot", "right_foot")
COMMAND_THRESHOLD = 0.01
AIR_TIME_MIN = 0.04
AIR_TIME_MAX = 0.10
FOOT_TARGET_HEIGHT = 0.02
HEIGHT_SCAN_RADIUS = 0.03
HEIGHT_SCAN_SAMPLES = 6

# microduck_rl's AGENTS.md budgets gaits at 4000-6000 iterations at 4096 envs.
GAIT_ITERATIONS = 5000


def make_twoleg_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    # --- Scene: plane + our robot, foot height scan + contact sensors. ---
    cfg.scene.entities = {"robot": get_twoleg_robot_cfg()}
    height_scan = next(
        sensor
        for sensor in cfg.scene.sensors or ()
        if isinstance(sensor, TerrainHeightSensorCfg)
        and sensor.name == "foot_height_scan"
    )
    height_scan.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in FOOT_SITES
    )
    height_scan.pattern = RingPatternCfg.single_ring(
        radius=HEIGHT_SCAN_RADIUS, num_samples=HEIGHT_SCAN_SAMPLES
    )
    cfg.scene.sensors = (
        height_scan,
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(
                mode="subtree",
                pattern=r"^(L_ankle_link|R_ankle_link)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    )
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # --- Actuation: XML position servos; action = target offset in radians. ---
    cfg.actions["joint_pos"].scale = 1.0

    # --- Solver: the reference assembly carries 26 collision geoms per leg pair
    # against the old model's two foot boxes, so the default contact capacity is
    # too small and mjlab raises nconmax overflow at startup. ---
    cfg.sim.nconmax = 200
    cfg.sim.contact_sensor_maxmatch = 200

    # --- Observations: no height scan. The critic keeps the privileged foot
    # terms microduck_rl gives it, which the restored sensors now feed. ---
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Obs noise: duck tuned these 5-50x below template after audit (lines 587-590
    # of microduck_velocity_env_cfg.py). Template noise punishes dynamic motion.
    from mjlab.utils.noise import UniformNoiseCfg as Unoise
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms["projected_gravity"].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    # --- Rewards: restore the gait terms, sized for a 0.2 m leg on a 1.26 kg body. ---
    cfg.rewards.pop("soft_landing", None)  # Dropped in microduck_rl's reward set too.

    # command_threshold is 0.01, not the base 0.5: these terms gate on
    # norm(cmd_xy) + abs(cmd_yaw), which tops out near 0.7 over our ranges.
    # AIR_TIME window follows microduck_rl EXACTLY: 0.125-0.300 s. Our earlier
    # 0.04-0.10 window was unreachable for TwoLeg's ~3.7-step flight at 0.02s
    # control, so stepping paid nothing and the policy fell into one-leg balance.
    cfg.rewards["air_time"].weight = 4.0
    cfg.rewards["air_time"].params["threshold_min"] = 0.125
    cfg.rewards["air_time"].params["threshold_max"] = 0.300
    cfg.rewards["air_time"].params["command_threshold"] = COMMAND_THRESHOLD

    # foot_clearance / foot_swing_height: microduck_rl uses -0.1 (NOT our -2.0 /
    # -0.25). Our over-strong values penalized any foot lift and killed the gait.
    cfg.rewards["foot_clearance"].weight = -0.1
    cfg.rewards["foot_clearance"].params["target_height"] = FOOT_TARGET_HEIGHT
    cfg.rewards["foot_clearance"].params["command_threshold"] = COMMAND_THRESHOLD
    cfg.rewards["foot_clearance"].params["asset_cfg"] = SceneEntityCfg(
        "robot", site_names=FOOT_SITES
    )

    cfg.rewards["foot_swing_height"].weight = -0.1
    cfg.rewards["foot_swing_height"].params["target_height"] = FOOT_TARGET_HEIGHT
    cfg.rewards["foot_swing_height"].params["command_threshold"] = COMMAND_THRESHOLD

    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = COMMAND_THRESHOLD
    cfg.rewards["foot_slip"].params["asset_cfg"] = SceneEntityCfg(
        "robot", site_names=FOOT_SITES
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )

    # NOTE: both_feet_air_time (our H2 anti-hop term) is REMOVED to follow
    # microduck_rl exactly -- microduck achieves symmetric two-legged gait via the
    # reward + curricula below, not an explicit both-feet term. If one-leg hopping
    # returns, re-add it (weight 5.0, func both_feet_air_time).

    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    cfg.rewards["upright"].weight = 2.0  # microduck exact
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -0.1

    # H11: joint_vel_legs REMOVED. It was an H5 attempt-tax (-0.05 on leg joint
    # velocity) that stayed live from iteration 0 of the fresh run, i.e. it
    # taxed every leg motion before a single step existed. Duck's lore (their
    # AGENTS.md, measured in-repo): attempt-taxes during discovery make doing
    # nothing win; introduce them AFTER the skill. The fresh run's outcome
    # (speed 0.000, duties 0.056/1.00) is exactly that lock-in. Duck has no
    # such term; action_rate_l2 stays at duck's opening -0.1. Re-add only
    # after the walk predicate passes ("walk first, tax later"). Revert this
    # removal if the unloaded flail returns instead of stepping.

    # Joint posture: hold HOME when standing, allow motion when walking.
    cfg.rewards["pose"].weight = 1.0
    cfg.rewards["pose"].params["std_standing"] = {
        r".*hip.*": 0.1,
        r".*knee.*": 0.1,
        r".*ankle.*": 0.1,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25,
    }
    cfg.rewards["pose"].params["std_running"] = cfg.rewards["pose"].params["std_walking"]
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(r".*",))
    cfg.rewards["pose"].params["walking_threshold"] = 0.05

    # --- Commands: fixed slow-walk ranges. Forward lives in channel 1 because
    # this body faces -y (chest servos / wide head box side); see LIN_VEL_Y above. ---
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.ranges.lin_vel_x = LIN_VEL_X
    command.ranges.lin_vel_y = LIN_VEL_Y
    command.ranges.ang_vel_z = ANG_VEL_Z
    # H10: walk first, stand later. Duck hands out standing marks in 2 of 100
    # envs and only raises that to 25 after it already walks; we gave standing
    # away in 10 of 100 from iteration 0, so half the gradient said "standing
    # is the job". Copy duck's opening hand: 2 of 100. The ramp to 0.25 is
    # added only after the walk predicate passes, because "later" has to mean
    # later than a verified walk.
    command.rel_standing_envs = 0.02
    command.rel_heading_envs = 0.0
    command.rel_turn_in_place_envs = 0.15

    # --- Events: point DR at our bodies/geoms, reset at standing height. ---
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.31, 0.33)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "L_foot_collision",
        "R_foot_collision",
    )
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)
    cfg.events["push_robot"].interval_range_s = (3.0, 6.0)
    cfg.events["push_robot"].params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}

    # --- Terminations: time_out + fell_over (mjlab's bad_orientation already
    # catches real flips via acos(-proj_z) > limit_angle). A custom
    # torso_inverted(proj_z<0) was added here but is WRONG for mjlab's
    # convention (gravity in body frame is ~(0,0,-1) for an UPRIGHT robot, so
    # proj_z<0 fires at the correct spawn pose and terminated every env on step
    # 1: ep_len=1, torso_inverted=2048). Removed; keep microduck-exact fell_over.
    cfg.terminations.pop("out_of_terrain_bounds", None)
    # Tighter than microduck's 70deg: TwoLeg is top-heavy and converges to a
    # fully-inverted fixed point (frame_alt=-1.0) that 70deg tolerance permits.
    # Terminating at 25deg forces the policy to stay near-upright and breaks the
    # inverted attractor. Reversible: widen back if it over-constrains.
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(25.0)

    # --- Curriculum (microduck_rl EXACT): no air_time/command widening ramps
    # (microduck uses fixed windows + fixed modest command ranges). Instead it
    # ramps action-rate smoothing (-0.1 -> -1.0 by iter 1500) and standing-env
    # fraction (0.02 -> 0.25 by iter 2000, "walk first, stand later"). ---
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum.pop("air_time_window", None)
    cfg.curriculum.pop("walk_command_ramp", None)
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * 24, "weight": -0.2},
                {"step": 750 * 24, "weight": -0.4},
                {"step": 1000 * 24, "weight": -0.6},
                {"step": 1250 * 24, "weight": -0.8},
                {"step": 1500 * 24, "weight": -1.0},
            ],
        },
    )
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0, "rel_standing_envs": 0.02},
                {"step": 500 * 24, "rel_standing_envs": 0.05},
                {"step": 750 * 24, "rel_standing_envs": 0.1},
                {"step": 1000 * 24, "rel_standing_envs": 0.15},
                {"step": 1500 * 24, "rel_standing_envs": 0.2},
                {"step": 2000 * 24, "rel_standing_envs": 0.25},
            ],
        },
    )

    cfg.viewer.body_name = "torso"
    cfg.viewer.distance = 0.8
    cfg.viewer.elevation = -10.0
    # Face the robot. The model's front is -y (left-right is X, forward is -Y,
    # see symmetry.py). MuJoCo azimuth 0 puts the camera on -y, i.e. behind the
    # robot; azimuth 180 puts it on +y, looking at the face.
    cfg.viewer.azimuth = 180.0

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.terminations.pop("out_of_terrain_bounds", None)

    return cfg


TwoLegRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    logger="tensorboard",
    wandb_project="twoleg",
    experiment_name="velocity",
    run_name="velocity",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=GAIT_ITERATIONS,
)
