"""TwoLeg velocity (flat) environment configuration.

Aligned to unitree_rl_mjlab's ``unitree_g1_flat_env_cfg``: same base velocity
recipe (make_velocity_env_cfg), same gait shaper (feet_gait phase-locked), same
reward/curriculum/termination structure. Only the robot asset, foot-site
mapping, and duck-scaled action scale differ.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from twoleg_rl.tasks.velocity.config.twoleg.dc_motor_action import (
    DCMotorEffortActionCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.rewards import self_collision_cost, soft_landing
from src.tasks.velocity.mdp.rewards import feet_gait, stand_still, body_orientation_l2
from twoleg_rl.tasks.velocity.config.twoleg.rewards import knee_flexion, both_feet_air, vertical_velocity_penalty, contact_continuity, com_height_cap, biped_torso_centering, biped_swing_height
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from twoleg_rl.assets.robots.twoleg.twoleg_constants import (
    TWOLEG_ACTION_SCALE,
    get_twoleg_robot_cfg,
    FOOT_SITES,
    FOOT_BODIES,
    TORSO_BODY,
)


def unitree_twoleg_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """TwoLeg rough-terrain velocity config (clone of unitree_g1_rough)."""
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 48

    cfg.scene.entities = {"robot": get_twoleg_robot_cfg()}

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(L_foot|R_foot)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        # Re-enabled: the warp 1.18.0/mujoco_warp _sensor_pos codegen is patched,
        # so track_air_time compiles now. soft_landing reward needs it.
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    # --- Torque control (matches BipedRobot's DC-motor effort model) ---------
    # Policy commands torque directly; the four-quadrant STS3215 envelope is
    # applied inside DCMotorEffortAction. scale maps normalized action [-1,1]
    # to the rated continuous torque (EFFORT_LIMIT = 0.98 N-m).
    from twoleg_rl.tasks.velocity.config.twoleg.dc_motor_action import (
        DCMotorEffortActionCfg,
        EFFORT_LIMIT,
    )
    cfg.actions["joint_pos"] = DCMotorEffortActionCfg(
        entity_name="robot",
        actuator_names=(r"^(L|R)_(hip_roll|hip|knee|ankle)_test$",),
        scale=EFFORT_LIMIT,
    )

    cfg.viewer.body_name = TORSO_BODY

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 0.15

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = FOOT_SITES

    # Gait reward: phase-locked alternating contact. WEIGHT KEPT LOW on purpose:
    # for the light 1.26 kg duck, this term rewards "alternating contact timing"
    # which a HOP also satisfies, so a high weight makes bouncing the optimum.
    # The anti-hop terms (contact_continuity / vertical_velocity / no_fly) must
    # outrank it, so grounding is the clear optimum.
    cfg.rewards["foot_gait"] = RewardTermCfg(
        func=feet_gait,
        weight=0.15,
        params={
            "period": 0.6,
            "offset": [0.0, 0.5],
            "threshold": 0.56,
            "command_threshold": 0.1,
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
        },
    )
    # Foot shaping (unitree weights).
    cfg.rewards["foot_clearance"].weight = -0.1
    cfg.rewards["foot_clearance"].params["target_height"] = 0.03
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.1
    cfg.rewards["foot_clearance"].params["asset_cfg"] = SceneEntityCfg(
        "robot", site_names=FOOT_SITES
    )
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.1
    cfg.rewards["foot_slip"].params["asset_cfg"] = SceneEntityCfg(
        "robot", site_names=FOOT_SITES
    )
    cfg.rewards["soft_landing"] = RewardTermCfg(
        func=soft_landing,
        weight=-1e-3,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.1,
        },
    )
    cfg.rewards["knee_flexion"] = RewardTermCfg(
        func=knee_flexion,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*knee_test.*",)),
            "command_name": "twist",
            "command_threshold": 0.1,
            "target": 0.6,
            "weight_low": 0.15,
            "weight_high": 1.3,
        },
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )
    # Anti-hop: THREE hard terms that must outrank feet_gait (0.15) + forward
    # tracking, so grounding is the clear optimum. One bounce must cost more than
    # a full walking episode's forward reward (measured 72-76% flight before).
    # Penalize BOTH feet airborne (kills hops when they do happen).
    cfg.rewards["no_fly"] = RewardTermCfg(
        func=both_feet_air,
        weight=-3.0,
        params={"sensor_name": "feet_ground_contact"},
    )
    # Reward keeping >=1 foot grounded each step (pull toward stance).
    cfg.rewards["contact_continuity"] = RewardTermCfg(
        func=contact_continuity,
        weight=2.0,
        params={"sensor_name": "feet_ground_contact"},
    )
    # HARD penalty on vertical launch velocity (|vz|) — directly kills bounce.
    cfg.rewards["vertical_velocity"] = RewardTermCfg(
        func=vertical_velocity_penalty,
        weight=-3.0,
        params={"scale": 5.0},
    )
    # STRUCTURAL anti-hop: forbid the root from rising above the duck's walking
    # envelope. Standing root height is 0.044 m; a bounce launches to ~0.96 m
    # (measured 87% flight). Capping root_z above 0.12 m with a quadratic
    # penalty makes launching physically unrewarding — the policy MUST stay
    # grounded to score, so stepping (not springing) is the optimum. This is the
    # term that the velocity/|vz| nudges alone could not beat for a 1.26 kg duck.
    cfg.rewards["com_height_cap"] = RewardTermCfg(
        func=com_height_cap,
        weight=-1.0,
        params={"cap": 0.12, "scale": 10.0},
    )
    # Stand still + upright (unitree).
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=stand_still,
        weight=-1.0,
        params={
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
        },
    )
    # BipedRobot port (stateless terms only; see rewards.py header for what was
    # deliberately not ported). torso: no equivalent existed; swing: complements
    # foot_clearance (speed-weighted) with a pure height-window shaper.
    # Weights are conservative first-pass values, not BipedRobot production
    # (swing 1.5 / torso 0.0): torso 1.0 matches the upright scale, swing 0.5
    # sits below knee_flexion 1.0 so it cannot outrank the anti-hop stack.
    cfg.rewards["biped_torso_centering"] = RewardTermCfg(
        func=biped_torso_centering,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.1,
            "sigma": 1.0,
        },
    )
    cfg.rewards["biped_swing_height"] = RewardTermCfg(
        func=biped_swing_height,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.1,
            "min_height": 0.05,
            "max_height": 0.15,
        },
    )
    cfg.rewards["body_orientation_l2"] = RewardTermCfg(
        func=body_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=(TORSO_BODY,))},
    )

    # Unitree drops foot_swing_height from the active set (its base class term
    # needs per-joint std dicts that we don't carry). Foot shaping is handled by
    # foot_clearance + foot_slip + feet_gait instead.
    cfg.rewards.pop("foot_swing_height", None)

    # Joint posture stds (duck). The base `pose` reward reads std_walking/
    # std_standing from these dicts keyed by joint regex; with no match they
    # resolve to an empty std tensor and crash at compute time. Mirror Unitree
    # G1: looser std on hip/knee/ankle (natural stride), tighter elsewhere.
    duck_pose_std = {
        r".*hip_roll_test.*": 0.5,
        r".*hip_test.*": 0.5,
        r".*knee_test.*": 0.5,
        r".*ankle_test.*": 0.15,
    }
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = duck_pose_std
    cfg.rewards["pose"].params["std_running"] = duck_pose_std

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )
        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def unitree_twoleg_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """TwoLeg flat-terrain velocity config (clone of unitree_g1_flat)."""
    cfg = unitree_twoleg_rough_env_cfg(play=play)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms.pop("height_scan", None)

    cfg.curriculum.pop("terrain_levels", None)

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
        twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
        twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

    return cfg
