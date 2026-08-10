from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Robot observations (world frame)
# ---------------------------------------------------------------------------

def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot anchor orientation in world frame as rotation matrix (first 2 columns, 6 dims)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot anchor linear velocity in world frame (3 dims)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_anchor_lin_vel_w.view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot anchor angular velocity in world frame (3 dims)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_anchor_ang_vel_w.view(env.num_envs, -1)


# ---------------------------------------------------------------------------
# Robot observations (anchor body frame)
# ---------------------------------------------------------------------------

def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot body positions relative to anchor frame (3 dims per body)."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)

    # Compute relative transform: target pose w.r.t. anchor frame
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot body orientations relative to anchor frame as rotation matrix (6 dims per body)."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


# ---------------------------------------------------------------------------
# Motion phase and joint tracking
# ---------------------------------------------------------------------------

def motion_phase(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Normalized position in the motion sequence [0, 1].

    Shape: [num_envs, 1]
    """
    command = env.command_manager.get_term("motion")
    time_steps = command.time_steps
    total_steps = command.motion.time_step_total
    return (time_steps.float() / total_steps).unsqueeze(-1)


def joint_pos_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Difference between current and target joint positions.

    Shape: [num_envs, num_joints]
    """
    command = env.command_manager.get_term("motion")
    target_joint_pos = command.joint_pos
    current_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return target_joint_pos - current_joint_pos


# ---------------------------------------------------------------------------
# Motion observations (anchor body frame)
# ---------------------------------------------------------------------------

def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Motion anchor position relative to robot anchor frame (3 dims)."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Motion anchor orientation relative to robot anchor frame as rotation matrix (6 dims)."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)
