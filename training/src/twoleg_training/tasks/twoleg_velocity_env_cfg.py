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
from mjlab.tasks.velocity.mdp.rewards import self_collision_cost, soft_landing
from twoleg_training.tasks.mdp import (
    duty_balance,
    no_fly,
    feet_moving,
    legs_energy,
    feet_gait,
    stand_still,
    body_orientation_l2,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from twoleg_training.robot.twoleg_constants import get_twoleg_robot_cfg
from twoleg_training.tasks.curriculum import walk_command_ramp
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

    # --- Rewards: ALIGNED TO unitree_rl_mjlab (native mjlab velocity recipe). ---
    # H9: the proven gait shaper is feet_gait (phase-locked alternating contact),
    # NOT air-time rewards. Air-time rewards made a hop pay ~2x a step -> the
    # one-foot flail we kept hitting. Unitree's recipe drops air_time entirely
    # and shapes gait via feet_gait + foot_clearance + foot_slip + soft_landing.
    # We port feet_gait/stand_still/body_orientation_l2 and keep our anti-
    # attractor layer (duty_balance/no_fly/feet_moving) as a duck-specific
    # safety net on top.
    cfg.rewards.pop("soft_landing", None)  # re-added below with unitree weight

    # Drop the air-time rewards (the flail cause). Both mjlab air_time and our
    # custom both_feet_air_time / feet_air_time_fc are removed from the active set.
    for _t in ("air_time", "both_feet_air_time", "feet_air_time_fc"):
        cfg.rewards.pop(_t, None)

    # feet_gait: THE phase-locked alternating-gait reward. left offset 0.0,
    # right offset 0.5 (half-period apart), stance threshold 0.56, period 0.6s
    # (same as Unitree G1). Pays when actual contact matches expected phase;
    # a hop/flail earns ~0. This is the structural hop preventer.
    cfg.rewards["feet_gait"] = RewardTermCfg(
        func=feet_gait,
        weight=0.5,
        params={
            "period": 0.6,
            "offset": [0.0, 0.5],
            "threshold": 0.56,
            "command_threshold": COMMAND_THRESHOLD,
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
        },
    )

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

    # soft_landing (unitree -1e-3): penalize hard landings, encourages smooth gait.
    # Uses the base mjlab term (make_velocity_env_cfg already defines it); just
    # set unitree's weight.
    cfg.rewards["soft_landing"] = RewardTermCfg(
        func=soft_landing,
        weight=-1e-3,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": COMMAND_THRESHOLD,
        },
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )

    # stand_still (unitree -1.0): penalize joint deviation from default when
    # command ~0. Prevents idle thrash / camping at zero command.
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=stand_still,
        weight=-1.0,
        params={
            "command_name": "twist",
            "command_threshold": COMMAND_THRESHOLD,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
        },
    )

    # body_orientation_l2 (unitree -1.0): keep torso upright during gait.
    cfg.rewards["body_orientation_l2"] = RewardTermCfg(
        func=body_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso",))},
    )

    # feet_moving + no_fly: duck-specific anti-attractor safety net. feet_moving
    # rewards any lift (breaks planted squat); no_fly rewards exactly-one-down
    # (single support). These complement feet_gait (which handles alternation);
    # kept because our 0.2m duck has weaker symmetry priors than a G1.
    cfg.rewards["no_fly"] = RewardTermCfg(
        func=no_fly,
        weight=2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": COMMAND_THRESHOLD,
        },
    )
    cfg.rewards["feet_moving"] = RewardTermCfg(
        func=feet_moving,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "threshold_min": 0.04,
            "command_name": "twist",
            "command_threshold": COMMAND_THRESHOLD,
        },
    )

    # duty_balance: our one-leg-collapse breaker. Rewards each foot lifting in
    # turn so a dead foot MUST move. Kept as the duck-specific safety net.
    cfg.rewards["duty_balance"] = RewardTermCfg(
        func=duty_balance,
        weight=2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "threshold_min": 0.04,
            "command_name": "twist",
            "command_threshold": COMMAND_THRESHOLD,
        },
    )

    # H7: legs_energy -- mechanical power penalty (robust_robot_walker, ICRA
    # 2025). Penalize qfrc_actuator * joint_vel squared (mechanical power) to
    # force efficient, non-flailing motion. Scale -1e-5 (robust_robot_walker
    # uses -1e-6..-2e-5; torque^2*vel^2 is large so the scale is tiny). Targets
    # ALL leg DOFs so it discourages the high-torque thrash / hop our fresh runs
    # showed; an efficient alternating gait costs less power.
    cfg.rewards["legs_energy"] = RewardTermCfg(
        func=legs_energy,
        weight=-1e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(
                    "L_hip_test", "L_knee_test", "L_ankle_test",
                    "R_hip_test", "R_knee_test", "R_ankle_test",
                ),
            ),
            "weight_scale": 1.0,
        },
    )

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
