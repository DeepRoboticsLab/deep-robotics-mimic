from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude
from dataclasses import dataclass

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# --- Predefined body indexes ---
Body_Indexes = list(range(14))

Tracking_Weight_init = [1.0] * 14


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    """Get indexes of specific bodies from the motion command's body list."""
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _log_tracking(env: ManagerBasedRLEnv, log_name: str | None, value: torch.Tensor, raw_error: torch.Tensor | None = None) -> None:
    """Write raw tracking quality and error to env.extras['log'].

    Logs two entries when log_name is provided:
      - 'Tracking/{log_name}':       reward = exp(-error/std²), range [0, 1]
      - 'Tracking/error_{log_name}':  sqrt(mean(error²)), independent of std

    The raw error log is used by the curriculum's reward-gated std schedule
    to decide when to shrink std, breaking the std-reward feedback loop.

    Args:
        env: The environment instance.
        log_name: Name for TensorBoard logging. If None, no logging.
        value: The reward value (exp(-error/std²)), mean across envs.
        raw_error: The squared error before exp, mean across envs.
                   If provided, sqrt is logged as the raw error metric.
    """
    if log_name is None:
        return
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"][f"Tracking/{log_name}"] = value.mean().item()
    if raw_error is not None:
        env.extras["log"][f"Tracking/error_{log_name}"] = raw_error.mean().sqrt().item()


def _get_body_indexes_sensor(contact_sensor, body_names: list[str]) -> list[int]:
    """Get indexes of specific bodies from a contact sensor's body list."""
    return [i for i, name in enumerate(contact_sensor.body_names) if (body_names is None) or (name in body_names)]


def _get_joint_indexes_asset(asset: Articulation, joint_names: list[str]) -> list[int]:
    """Get indexes of specific joints from the asset's body list."""
    return [i for i, name in enumerate(asset.body_names) if (joint_names is None) or (name in joint_names)]


def quat_wxyz_to_euler(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz order) to Euler angles (roll, pitch, yaw).

    Args:
        quat_wxyz: Quaternion tensor of shape [num_envs, 1, 4] or [num_envs, 4].

    Returns:
        Euler angles tensor of shape [num_envs, 3] in (roll, pitch, yaw) order.
    """
    # Ensure input shape is [num_envs, 4]
    if quat_wxyz.dim() == 3:
        quat_wxyz = quat_wxyz.squeeze(1)

    # Extract quaternion components (w, x, y, z)
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]

    # Compute Euler angles (roll, pitch, yaw)
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)  # clamp to prevent numerical errors
    pitch = torch.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Combine into [roll, pitch, yaw]
    euler_angles = torch.stack([roll, pitch, yaw], dim=1)

    return euler_angles


# ---------------------------------------------------------------------------
# Anchor position / orientation rewards
# ---------------------------------------------------------------------------

def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """World-frame anchor position error reward."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_position_error_exp_new(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    weight: list[float] = [1.0, 1.0, 1.0],
    log_name: str | None = None,
) -> torch.Tensor:
    """World-frame anchor position error reward with per-axis weighting."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos_diff_square = torch.square(command.anchor_pos_w - command.robot_anchor_pos_w)
    weight_tensor = torch.tensor(weight, dtype=pos_diff_square.dtype, device=env.device)
    error = torch.sum(pos_diff_square * weight_tensor, dim=-1)
    result = torch.exp(-error / std**2)
    _log_tracking(env, log_name, result, error)
    return result


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float, log_name: str | None = None) -> torch.Tensor:
    """World-frame anchor orientation error reward."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    result = torch.exp(-error / std**2)
    _log_tracking(env, log_name, result, error)
    return result


# ---------------------------------------------------------------------------
# Body position / orientation / velocity rewards
# ---------------------------------------------------------------------------

def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None, weight: list = Tracking_Weight_init, log_name: str | None = None
) -> torch.Tensor:
    """World-frame body position error reward (relative to motion reference).

    If log_name is provided, the raw tracking quality is written to env.extras['log']
    under 'Tracking/{log_name}', bypassing the weight=0 suppression in TensorBoard.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if body_names is not None:
        body_indexes = _get_body_indexes(command, body_names)
    else:
        body_indexes = Body_Indexes

    tracking_weight = torch.tensor(weight, device=env.device)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]) * tracking_weight[None, :, None], dim=-1
    )
    result = torch.exp(-error.mean(-1) / std**2)
    _log_tracking(env, log_name, result, error.mean(-1))
    return result


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None, weight: list = Tracking_Weight_init, log_name: str | None = None
) -> torch.Tensor:
    """World-frame body orientation error reward (relative to motion reference).

    If log_name is provided, the raw tracking quality is written to env.extras['log']
    under 'Tracking/{log_name}', bypassing the weight=0 suppression in TensorBoard.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if body_names is not None:
        body_indexes = _get_body_indexes(command, body_names)
    else:
        body_indexes = Body_Indexes
    tracking_weight = torch.tensor(weight, device=env.device)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2 * tracking_weight[None, :]
    )
    result = torch.exp(-error.mean(-1) / std**2)
    _log_tracking(env, log_name, result, error.mean(-1))
    return result


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None, weight: list = Tracking_Weight_init
) -> torch.Tensor:
    """World-frame body linear velocity error reward."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if body_names is not None:
        body_indexes = _get_body_indexes(command, body_names)
    else:
        body_indexes = Body_Indexes
    tracking_weight = torch.tensor(weight, device=env.device)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]) * tracking_weight[None, :, None], dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None, weight: list = Tracking_Weight_init
) -> torch.Tensor:
    """World-frame body angular velocity error reward."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if body_names is not None:
        body_indexes = _get_body_indexes(command, body_names)
    else:
        body_indexes = Body_Indexes
    tracking_weight = torch.tensor(weight, device=env.device)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]) * tracking_weight[None, :, None], dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


# ---------------------------------------------------------------------------
# Joint tracking reward
# ---------------------------------------------------------------------------

def motion_joint_pos_tracking(env: ManagerBasedRLEnv, std: float, n_frames: int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), log_name: str | None = None) -> torch.Tensor:
    """Joint position tracking reward — only active for the first n_frames.

    reward = exp(-sum(joint_pos_error^2) / std^2), returns 0 after n_frames.

    If log_name is provided, the raw tracking quality is written to env.extras['log']
    under 'Tracking/{log_name}', bypassing the weight=0 suppression in TensorBoard.

    Args:
        env: The environment instance.
        std: Standard deviation controlling reward sensitivity.
        n_frames: Number of frames for which the reward is active.
        asset_cfg: Robot asset configuration (contains joint_ids).
        log_name: Optional name for TensorBoard logging of raw tracking quality.
    """
    command = env.command_manager.get_term("motion")
    time_steps = command.time_steps

    target = command.joint_pos
    current = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    error = torch.sum((target - current) ** 2, dim=-1)
    reward = torch.exp(-error / std**2)

    # --- Log unmasked tracking quality (bypasses n_frames mask) ---
    _log_tracking(env, log_name, reward)

    # Only active for the first n_frames, returns 0 afterwards
    mask = (time_steps < n_frames).float()
    return reward * mask


# ---------------------------------------------------------------------------
# Ankle / joint penalty rewards
# ---------------------------------------------------------------------------

def ankle_body_angular_velocity_l2(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Penalize ankle body angular velocity magnitude."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    feet_names = ["left_ankle_x_link", "right_ankle_x_link"]
    feet_indexes = _get_body_indexes(command, feet_names)

    return torch.sum(torch.square(command.robot_body_ang_vel_w[:, feet_indexes]), dim=(1, 2))


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint power (velocity * torque)."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def ankle_joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize ankle joint torques squared."""
    asset: Articulation = env.scene[asset_cfg.name]

    torques = asset.data.applied_torque
    ankle_torques = torques[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.square(ankle_torques), dim=1)

    return penalty


def ankle_joint_torques_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    torque_threshold: float = 50.0,
) -> torch.Tensor:
    """Penalize ankle joint torques that exceed a threshold (L1 penalty on excess)."""
    asset: Articulation = env.scene[asset_cfg.name]

    compute_torques = asset.data.computed_torque
    ankle_torques = compute_torques[:, asset_cfg.joint_ids]

    # Compute excess torque beyond threshold (only penalize the excess)
    excess_torques = torch.abs(ankle_torques) - torque_threshold
    excess_torques = torch.clamp_min(excess_torques, 0.0)

    # L1 penalty: sum of all excess magnitudes
    penalty = torch.sum(excess_torques, dim=1)

    return penalty


# ---------------------------------------------------------------------------
# Feet orientation reward
# ---------------------------------------------------------------------------

def feet_orientation(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "motion") -> torch.Tensor:
    """Reward feet flatness when in contact with the ground.

    Penalizes pitch angle of feet when contact force is detected,
    encouraging flat foot placement during standing/walking.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    feet_names = ["left_ankle_x_link", "right_ankle_x_link"]
    feet_indexes_quat = _get_body_indexes(command, feet_names)

    # Get contact forces and foot quaternions
    feet_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # [num_envs, num_feet, 3]
    feet_quat = command.robot_body_quat_w[:, feet_indexes_quat]  # [num_envs, num_feet, 4(wxyz)]

    num_envs = feet_quat.shape[0]
    num_feet = len(feet_names)

    # Extract pitch angle from each foot quaternion
    feet_pitch = torch.zeros((num_envs, num_feet), device=env.device)
    for i in range(num_feet):
        euler_angles = quat_wxyz_to_euler(feet_quat[:, i, :])  # returns (roll, pitch, yaw)
        feet_pitch[:, i] = euler_angles[:, 1]  # extract pitch angle

    # Contact detection: Z-force > 3.0 N
    contact = feet_contact_forces[:, :, 2] > 3.0  # [num_envs, num_feet]

    # Penalize pitch when in contact (encourage flat feet)
    reward = (contact[:, 0] * torch.square(feet_pitch[:, 0]) +
              contact[:, 1] * torch.square(feet_pitch[:, 1]))

    return reward


def motion_feet_z_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Feet Z-axis position tracking reward — only tracks ankle_x_link Z position.

    Unlike motion_relative_body_position_error_exp, this function only computes
    the Z-axis position error of ankle_x_link to strengthen foot-lift tracking signal
    without dilution from XY errors.

    reward = exp(-sum(feet_z_error^2) / std^2)

    Args:
        env: The environment instance.
        command_name: MotionCommand name.
        std: Standard deviation controlling reward sensitivity.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    feet_names = ["left_ankle_x_link", "right_ankle_x_link"]
    feet_indexes = _get_body_indexes(command, feet_names)
    error = torch.sum(
        torch.square(
            command.body_pos_relative_w[:, feet_indexes, 2]
            - command.robot_body_pos_w[:, feet_indexes, 2]
        ), dim=-1
    )
    return torch.exp(-error / std**2)
