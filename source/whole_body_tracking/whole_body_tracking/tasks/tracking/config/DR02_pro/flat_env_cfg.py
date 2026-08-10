from isaaclab.utils import configclass

from whole_body_tracking.robots.DR02_pro import DR02_PRO_ACTION_SCALE, DR02_PRO_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg_DR02 import TrackingEnvCfg



@configclass
class DR02_PRO_FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = DR02_PRO_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = DR02_PRO_ACTION_SCALE
        self.commands.motion.anchor_body_name = "body"
        # Body link names recorded in motion file (14 total)
        self.commands.motion.body_names = [
            "base_link",              # 0
            "body",                   # 1
            "left_shoulder_x_link",   # 2
            "left_elbow_link",        # 3
            "left_wrist_x_link",      # 4
            "right_shoulder_x_link",  # 5
            "right_elbow_link",       # 6
            "right_wrist_x_link",     # 7
            "left_hip_x_link",        # 8
            "left_knee_link",         # 9
            "left_ankle_x_link",      # 10
            "right_hip_x_link",       # 11
            "right_knee_link",        # 12
            "right_ankle_x_link"      # 13
        ]