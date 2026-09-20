"""TwoLeg velocity (slow-walk) environment.

A deliberately small version of mjlab's velocity task, following the
``/home/kenpeter/work/microduck_rl`` recipe but stripped to the essentials:

* flat plane, no terrain raycast / foot-height sensors
* velocity command tracking (slow, forward-biased) + upright + joint posture
* all 15 joints actuated with the XML position servos
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from twoleg_training.robot.twoleg_constants import get_twoleg_robot_cfg

# Slow-walk command envelope.
LIN_VEL_X = (0.0, 0.3)  # forward only, slow
LIN_VEL_Y = (-0.1, 0.1)
ANG_VEL_Z = (-0.4, 0.4)


def make_twoleg_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    # --- Scene: plane + our robot, no terrain/foot sensors. ---
    cfg.scene.entities = {"robot": get_twoleg_robot_cfg()}
    cfg.scene.sensors = ()
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # --- Actuation: XML position servos; action = target offset in radians. ---
    cfg.actions["joint_pos"].scale = 0.5

    # --- Observations: drop the sensor-driven height scan terms. ---
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    for term in ("foot_height", "foot_air_time", "foot_contact", "foot_contact_forces"):
        cfg.observations["critic"].terms.pop(term, None)

    # --- Rewards: remove everything that needs sensors we dropped. ---
    for reward in (
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
    ):
        cfg.rewards.pop(reward, None)

    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    cfg.rewards["track_angular_velocity"].weight = 1.5
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -0.1

    # Joint posture: hold HOME when standing, allow motion when walking.
    cfg.rewards["pose"].weight = 1.0
    cfg.rewards["pose"].params["std_standing"] = {
        r".*hip.*": 0.1,
        r".*knee.*": 0.1,
        r".*ankle.*": 0.1,
        r".*shoulder.*": 0.2,
        r".*elbow.*": 0.2,
        r".*head.*": 0.2,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25,
        r".*shoulder.*": 0.3,
        r".*elbow.*": 0.3,
        r".*head.*": 0.2,
    }
    cfg.rewards["pose"].params["std_running"] = cfg.rewards["pose"].params["std_walking"]
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(r".*",))
    cfg.rewards["pose"].params["walking_threshold"] = 0.05

    # --- Commands: fixed slow-walk ranges. ---
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.ranges.lin_vel_x = LIN_VEL_X
    command.ranges.lin_vel_y = LIN_VEL_Y
    command.ranges.ang_vel_z = ANG_VEL_Z
    command.rel_standing_envs = 0.1
    command.rel_heading_envs = 0.0

    # --- Events: point DR at our bodies/geoms, reset at standing height. ---
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.165, 0.175)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_col",
        "right_foot_col",
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)
    cfg.events["push_robot"].params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}

    # --- Terminations: fell_over + time_out only (no terrain bounds). ---
    cfg.terminations.pop("out_of_terrain_bounds", None)

    # --- Curriculum: none for the simple task (fixed commands, flat plane). ---
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)

    cfg.viewer.body_name = "torso"
    cfg.viewer.distance = 0.8
    cfg.viewer.elevation = -10.0

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.terminations.pop("out_of_terrain_bounds", None)

    return cfg


TwoLegRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(256, 128, 64),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(256, 128, 64),
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
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2000,
)
