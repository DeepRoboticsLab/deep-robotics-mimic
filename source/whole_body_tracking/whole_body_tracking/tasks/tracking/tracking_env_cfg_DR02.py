from __future__ import annotations

from dataclasses import MISSING, dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
import isaaclab.terrains as terrain_gen

##
# Scene definition
##

ZERO_VELOCITY_RANGE = {
    "x": (-0.0, 0.0),
    "y": (-0.0, 0.0),
    "z": (-0.0, 0.0),
    "roll": (-0.0, 0.0),
    "pitch": (-0.0, 0.0),
    "yaw": (-0.0, 0.0),
}


VELOCITY_RANGE = {
    "x": (-1, 1),
    "y": (-1, 1),
    "z": (-1, 1),
    "roll": (-3, 3),
    "pitch": (-3, 3),
    "yaw": (-3, 3),
}

VELOCITY_RANGE_1 = {
    "x": (-1.6, 1.6),
    "y": (-1.6, 1.6),
    "z": (-1, 1),
    "roll": (-4, 4),
    "pitch": (-4, 4),
    "yaw": (-3, 3),
}

VELOCITY_RANGE_HIGH = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.5, 0.5),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
}

VELOCITY_RANGE_LOW = {
    "x": (-0.1, 0.1),
    "y": (-0.1, 0.1),
    "z": (-0.1, 0.1),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2,  0.2),
    "yaw": (-0.3, 0.3),
}

AVERAGE_WEIGHT = [1.0] * 14




ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=5.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.02,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.06), noise_step=0.02, border_width=0.0
        ),
    },
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # Plane terrain (~1.8s per round on RTX 4090)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=ZERO_VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)




@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)

        # --- Reference motion (joint_pos + joint_vel: 29 + 29 = 58 dims) ---
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})

        # --- Robot body observations ---
        base_link_gravity = ObsTerm(func=mdp.body_projected_gravity_b,  # base gravity projection (3 dims)
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))  # root angular velocity (3 dims)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))  # joint pos relative to default (29 dims)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.0, n_max=1.0))  # joint vel relative to default (29 dims)
        actions = ObsTerm(func=mdp.last_action)  # last action (29 dims)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        base_link_gravity = ObsTerm(func=mdp.body_projected_gravity_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])}
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 1024,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "com_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "z": (-0.08, 0.08)},
        },
    )

    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )


    push_robot_low = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1,3), 
        params={"velocity_range": VELOCITY_RANGE_LOW},
    )

# Reward scaling factor for penalty terms
scale = 3

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp_new,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3,
                "weight": [1.0, 1.0, 1.0]},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )

    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3,
                "weight" : [1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]},
    )
    
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4,
                "weight" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0,
                "weight" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
    )
    
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14,
                "weight" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
    )

    motion_feet_z_pos = RewTerm(
        func=mdp.motion_feet_z_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.15},  # tighter std for stricter foot tracking
    )

    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0 ,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-5e-7 * scale 
    )

    joint_power = RewTerm(
        func=mdp.joint_power, weight=-1e-6 * scale
    )

    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-1e-9 * scale #-1e-9*scale
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2 * scale * 2 ) # 4

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_x_link$)(?!right_ankle_x_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.4},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.12,
            "body_names": [
                "left_ankle_x_link",
                "right_ankle_x_link",
            ],
        },
    )



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP (currently disabled)."""

    # All curriculum terms are disabled. Reward std values are fixed.


##
# Environment configuration
##

 
@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=10)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
