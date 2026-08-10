import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.robots.actuator import RandomPDActuatorCfg

DR02_PRO_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ASSET_DIR}/DR02/DR02_pro.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=57.3 * 100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9), 
        joint_pos={
            ".*_ankle_x_joint" : 0. ,
            ".*_ankle_y_joint" : 0. ,
            ".*_elbow_joint" : 0. ,
            ".*_hip_x_joint" : 0. ,
            ".*_hip_y_joint" : 0. ,
            ".*_hip_z_joint" : 0. ,
            ".*_knee_joint" : 0. ,
            ".*_shoulder_x_joint" : 0.,
            ".*_shoulder_y_joint" : 0. ,
            ".*_shoulder_z_joint" : 0. ,
            ".*_wrist_x_joint" : 0. ,
            ".*_wrist_y_joint" : 0. ,
            ".*_wrist_z_joint" : 0. ,
            "waist_z_joint" : 0. ,
            "waist_x_joint" : 0. ,
            "waist_y_joint" : 0. ,
            # "neck_z_joint" : 0. ,
            # "neck_y_joint" : 0. ,
        },
        joint_vel={".*": 0.0},
    ),

    soft_joint_pos_limit_factor=0.95, # 0.9


    # pd
    actuators={
        "legs": RandomPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_y_joint",
                ".*_hip_x_joint",
                ".*_hip_z_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_y_joint": 300,
                ".*_hip_x_joint": 300,
                ".*_hip_z_joint": 135,
                ".*_knee_joint": 300,
            },
            velocity_limit_sim={
                ".*_hip_y_joint": 19.6,
                ".*_hip_x_joint": 19.6,
                ".*_hip_z_joint": 20.4,
                ".*_knee_joint": 19.6,
            },
            stiffness={
                ".*_hip_y_joint": 300,
                ".*_hip_x_joint": 300,
                ".*_hip_z_joint": 300,
                ".*_knee_joint": 300,
            },
            damping={
                ".*_hip_y_joint": 10,
                ".*_hip_x_joint": 10,
                ".*_hip_z_joint": 10,
                ".*_knee_joint": 10,
            },
            armature={
                ".*_hip_y_joint": 0.,
                ".*_hip_x_joint": 0.,
                ".*_hip_z_joint": 0.,
                ".*_knee_joint": 0.,
            },
            motor_strength = (0.8, 1.0),
            PD_random_range = (0.8, 1.2),
            
            friction=0.0,
            min_delay=0,
            max_delay=8,
        ),

        "feet": RandomPDActuatorCfg(
            joint_names_expr=[
                ".*_ankle_y_joint",
                ".*_ankle_x_joint",
            ],
            effort_limit_sim={
                ".*_ankle_y_joint": 135,
                ".*_ankle_x_joint": 48.0,
            },
            velocity_limit_sim={
                ".*_ankle_y_joint": 20.4,
                ".*_ankle_x_joint": 25.7,
            },
            stiffness={
                ".*_ankle_y_joint": 80.0,
                ".*_ankle_x_joint": 30.0,
            },
            damping={
                ".*_ankle_y_joint": 3.0,
                ".*_ankle_x_joint": 1.0,
            },
            armature={
                ".*_ankle_y_joint": 0.01,
                ".*_ankle_x_joint": 0.01,
            },
            motor_strength = (0.8, 1.0),
            PD_random_range = (0.8, 1.2),
            
            friction=0.0,
            min_delay=0,
            max_delay=8,
        ),

        "waist": RandomPDActuatorCfg(
            joint_names_expr=[
                "waist_x_joint",
                "waist_y_joint",
                "waist_z_joint",
            ],
            effort_limit_sim={
                "waist_x_joint": 135.0,
                "waist_y_joint": 300.0,
                "waist_z_joint": 135.0,
            },
            velocity_limit_sim={
                "waist_x_joint": 20.4,
                "waist_y_joint": 19.6,
                "waist_z_joint": 20.4,
            },   
            stiffness={
                "waist_x_joint": 200.0,
                "waist_y_joint": 200.0,
                "waist_z_joint": 200.0,
            },
            damping={
                "waist_x_joint": 3.0,
                "waist_y_joint": 6.0,
                "waist_z_joint": 10.0,
            },     
            armature={
                "waist_x_joint": 0.,
                "waist_y_joint": 0.,
                "waist_z_joint": 0.,
            },   
            motor_strength = (0.8, 1.0),
            PD_random_range = (0.8, 1.2),
            friction=0.0,
            min_delay=0,
            max_delay=8,
        ),


        "arms": RandomPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_y_joint",
                ".*_shoulder_x_joint",
                ".*_shoulder_z_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_y_joint": 135.0,
                ".*_shoulder_x_joint": 135.0,
                ".*_shoulder_z_joint": 135.0,
                ".*_elbow_joint": 135.0,
            },
            velocity_limit_sim={
                ".*_shoulder_y_joint": 20.4,
                ".*_shoulder_x_joint": 20.4,
                ".*_shoulder_z_joint": 20.4,
                ".*_elbow_joint": 20.4,
            },
            stiffness={
                ".*_shoulder_y_joint": 100.0,
                ".*_shoulder_x_joint": 100.0,
                ".*_shoulder_z_joint": 100.0,
                ".*_elbow_joint": 100.0,
            },
            damping={
                ".*_shoulder_y_joint": 5.0,
                ".*_shoulder_x_joint": 5.0,
                ".*_shoulder_z_joint": 5.0,
                ".*_elbow_joint": 5.0,
            },
            armature={
                ".*_shoulder_y_joint": 0.0,
                ".*_shoulder_x_joint": 0.0,
                ".*_shoulder_z_joint": 0.0,
                ".*_elbow_joint": 0.0,
            },
            motor_strength = (0.8, 1.0),
            PD_random_range = (0.8, 1.2),
            friction=0.0,
            min_delay=0,
            max_delay=8,
        ),

        "wrist": RandomPDActuatorCfg(
            joint_names_expr=[
                ".*_wrist_z_joint",
                ".*_wrist_y_joint",
                ".*_wrist_x_joint",

            ],
            effort_limit_sim={
                ".*_wrist_z_joint": 48.0,
                ".*_wrist_y_joint": 48.0,
                ".*_wrist_x_joint": 48.0,
            },
            velocity_limit_sim={
                ".*_wrist_z_joint": 20.4,
                ".*_wrist_y_joint": 20.4,
                ".*_wrist_x_joint": 20.4,

            },
            stiffness={
                ".*_wrist_z_joint": 80.0,
                ".*_wrist_y_joint": 80.0,
                ".*_wrist_x_joint": 80.0,

            },
            damping={
                ".*_wrist_z_joint": 3.0,
                ".*_wrist_y_joint": 3.0,
                ".*_wrist_x_joint": 3.0,

            },
            armature={
                ".*_wrist_z_joint": 0.01,
                ".*_wrist_y_joint": 0.01,
                ".*_wrist_x_joint": 0.01,
            },
            motor_strength = (0.8, 1.0),
            PD_random_range = (0.8, 1.2),
            friction=0.0,
            min_delay=0,
            max_delay=8,
        ),
        
    },

    

)

DR02_PRO_ACTION_SCALE = {}
for a in DR02_PRO_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr

    for n in names:
        DR02_PRO_ACTION_SCALE[n] = 0.5
