from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate if the anchor position error exceeds the threshold."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate if the anchor Z-axis position error exceeds the threshold."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminate if the anchor orientation error (gravity projection) exceeds the threshold."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate if any body position error exceeds the threshold."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate if any body Z-axis position error exceeds the threshold."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)

def bad_physics_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate the episode if any joint or rigid body data becomes invalid (NaN/Inf).
    
    This function performs comprehensive checks for NaN or Inf values in:
    1. All joint positions (dof_pos)
    2. All joint velocities (dof_vel)
    3. All body link positions (body_link_pos_w)
    4. All body link velocities (body_link_lin_vel_w, body_link_ang_vel_w)
    5. All body center of mass positions (body_com_pos_w)
    6. All body center of mass velocities (body_com_lin_vel_w, body_com_ang_vel_w)

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset (robot) to check. Defaults to "robot".
    """
    # Initialize the reset buffer with False for all environments
    reset_buf = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    robot_data = env.scene[asset_cfg.name].data

    # --- Check Joint Data ---
    dof_pos = robot_data.joint_pos
    dof_vel = robot_data.joint_vel

    # Check for NaNs or Infs in joints
    invalid_joint_pos = torch.any(torch.logical_or(torch.isnan(dof_pos), torch.isinf(dof_pos)), dim=-1)
    invalid_joint_vel = torch.any(torch.logical_or(torch.isnan(dof_vel), torch.isinf(dof_vel)), dim=-1)
    
    # Combine joint conditions
    joint_conditions = torch.logical_or(invalid_joint_pos, invalid_joint_vel)
    reset_buf = torch.logical_or(reset_buf, joint_conditions)

    # --- Check ALL Rigid Body Data (Links and Center of Mass) ---
    
    # Body Link Pose (Position)
    body_link_pos_w = robot_data.body_link_pos_w
    invalid_body_link_pos = torch.any(torch.logical_or(torch.isnan(body_link_pos_w), torch.isinf(body_link_pos_w)), dim=[-1, -2]) # Check across all bodies and coords
    reset_buf = torch.logical_or(reset_buf, invalid_body_link_pos)

    # Body Link Velocity (Linear and Angular)
    body_link_lin_vel_w = robot_data.body_link_lin_vel_w
    body_link_ang_vel_w = robot_data.body_link_ang_vel_w
    invalid_body_link_lin_vel = torch.any(torch.logical_or(torch.isnan(body_link_lin_vel_w), torch.isinf(body_link_lin_vel_w)), dim=[-1, -2])
    invalid_body_link_ang_vel = torch.any(torch.logical_or(torch.isnan(body_link_ang_vel_w), torch.isinf(body_link_ang_vel_w)), dim=[-1, -2])

    body_link_vel_conditions = torch.logical_or(invalid_body_link_lin_vel, invalid_body_link_ang_vel)
    reset_buf = torch.logical_or(reset_buf, body_link_vel_conditions)

    # Body Center of Mass Pose (Position)
    body_com_pos_w = robot_data.body_com_pos_w
    invalid_body_com_pos = torch.any(torch.logical_or(torch.isnan(body_com_pos_w), torch.isinf(body_com_pos_w)), dim=[-1, -2])
    reset_buf = torch.logical_or(reset_buf, invalid_body_com_pos)

    # Body Center of Mass Velocity (Linear and Angular)
    body_com_lin_vel_w = robot_data.body_com_lin_vel_w
    body_com_ang_vel_w = robot_data.body_com_ang_vel_w
    invalid_body_com_lin_vel = torch.any(torch.logical_or(torch.isnan(body_com_lin_vel_w), torch.isinf(body_com_lin_vel_w)), dim=[-1, -2])
    invalid_body_com_ang_vel = torch.any(torch.logical_or(torch.isnan(body_com_ang_vel_w), torch.isinf(body_com_ang_vel_w)), dim=[-1, -2])

    body_com_vel_conditions = torch.logical_or(invalid_body_com_lin_vel, invalid_body_com_ang_vel)
    reset_buf = torch.logical_or(reset_buf, body_com_vel_conditions)
    
    return reset_buf


def link_groups_collision(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    link_group_1: list[str] = ["elbow_link"],
    link_group_2: list[str] = ["hip_y_link", "hip_x_link", "hip_z_link"],
    threshold: float = 1.0
) -> torch.Tensor:
    """Detect collision between two groups of links.
    
    Args:
        env: The environment instance.
        asset_cfg: Asset configuration.
        link_group_1: First group of link name patterns.
        link_group_2: Second group of link name patterns.
        threshold: Contact force threshold (N).
    
    Returns:
        Bool tensor: True if collision detected and termination needed.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors["elbow_contact_forces"]
    
    # --- Get indices for both link groups ---
    group_1_indices = asset.find_bodies(link_group_1)
    group_2_indices = asset.find_bodies(link_group_2)
    
    # --- Check collision between link groups ---
    collision_detected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for idx1 in group_1_indices:
        contact_forces = asset.data.contact_force_b[:, idx1]
        for idx2 in group_2_indices:
            contact_force_magnitude = torch.norm(contact_forces[:, idx2], dim=1)
            collision = contact_force_magnitude > threshold
            collision_detected = collision_detected | collision
    
    return collision_detected


def self_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if any of the specified bodies have contact forces exceeding threshold.

    Detects self-collision by checking if the contact force on the bodies specified
    in sensor_cfg.body_names exceeds the given threshold.

    Args:
        env: The environment instance.
        sensor_cfg: The contact sensor configuration. The body_names in sensor_cfg
            specify which links to monitor for collision.
        threshold: The force threshold above which a contact is considered a collision.

    Returns:
        A bool tensor of shape (num_envs,) indicating whether to terminate.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Get the net forces for the specified bodies
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # [num_envs, num_bodies, 3]
    # Compute force magnitude for each body
    force_magnitude = torch.norm(net_forces, dim=-1)  # [num_envs, num_bodies]
    # Terminate if any body has collision force exceeding threshold
    return torch.any(force_magnitude > threshold, dim=-1)