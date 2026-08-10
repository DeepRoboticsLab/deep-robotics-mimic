from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions


class DelayedImplicitActuator(ImplicitActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedImplicitActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # --- Instantiate delay buffers ---
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # --- Determine number of environments ---
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # --- Set random delay for each environment ---
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # --- Reset buffers ---
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # --- Apply delay to all setpoints ---
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # --- Compute actuator model ---
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""


class RandomPDActuator(DelayedPDActuator):
    """PD actuator with domain randomization on stiffness, damping, motor strength, and velocity limits.

    Extends :class:`DelayedPDActuator` by adding per-environment randomization of:
    - PD gains (stiffness/damping multiplied by a random factor)
    - Motor strength (output torque scaling)
    - Saturation effort (based on velocity limit and battery voltage model)
    - Velocity limits (randomized between min and nominal)
    """

    cfg: RandomPDActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: RandomPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # --- Initialize randomization parameters ---
        self.random_pd = math_utils.sample_uniform(
            self.cfg.PD_random_range[0], self.cfg.PD_random_range[1],
            (self._num_envs, self.num_joints), device=self._device,
        )
        self.random_stiffness = self.stiffness * self.random_pd
        self.random_damping = self.damping * self.random_pd
        self.motor_strength = torch.ones((self._num_envs, self.num_joints), device=self._device)

        self._joint_vel = torch.zeros_like(self.computed_effort)
        self.t_n_vel = torch.ones((self._num_envs, 1), device=self._device)
        self._saturation_effort = self.computed_effort.clone()
        self._zeros_effort = torch.zeros_like(self.computed_effort)

        # --- Velocity limit randomization parameters ---
        self.min_velocity_limit = self.velocity_limit[[0]] * 0.8
        self.battary_v = torch.zeros((self._num_envs, 1), device=self._device)
        self.random_velocity_limit = self.velocity_limit.clone()

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # --- Determine number of environments ---
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)

        # --- Randomize PD gains ---
        self.random_pd[env_ids] = math_utils.sample_uniform(
            self.cfg.PD_random_range[0], self.cfg.PD_random_range[1],
            (num_envs, self.num_joints), device=self._device,
        )
        self.random_stiffness[env_ids] = self.stiffness[env_ids] * self.random_pd[env_ids]
        self.random_damping[env_ids] = self.damping[env_ids] * self.random_pd[env_ids]

        # --- Randomize motor strength ---
        self.motor_strength[env_ids] = math_utils.sample_uniform(
            self.cfg.motor_strength[0], self.cfg.motor_strength[1],
            (num_envs, self.num_joints), device=self._device,
        )

        # --- Randomize velocity saturation curve ---
        self.t_n_vel[env_ids] = math_utils.sample_uniform(
            self.cfg.t_n_vel_range[0], self.cfg.t_n_vel_range[1],
            (num_envs, 1), device=self._device,
        )
        self._saturation_effort[env_ids] = self.effort_limit[env_ids] / (1 - self.t_n_vel[env_ids]).clip(0.01, 1)

        # --- Randomize velocity limits (battery voltage model) ---
        self.battary_v[env_ids] = torch.rand((num_envs, 1), device=self._device)
        self.random_velocity_limit[env_ids] = self.battary_v[env_ids] * (
            self.velocity_limit[env_ids] - self.min_velocity_limit) + self.min_velocity_limit

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # --- Apply delay to all setpoints ---
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        self._joint_vel[:] = joint_vel

        # --- Compute PD torque (joint_efforts omitted for efficiency) ---
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        self.computed_effort = self.random_stiffness * error_pos + self.random_damping * error_vel

        # --- Clip and apply motor strength ---
        self.applied_effort = self._clip_effort(self.computed_effort) * self.motor_strength
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Clip torques based on motor saturation curve and effort limits."""
        # --- Max limit (decreasing with velocity) ---
        max_effort = self._saturation_effort * (1.0 - self._joint_vel / self.random_velocity_limit)
        max_effort = torch.clip(max_effort, min=self._zeros_effort, max=self.effort_limit)

        # --- Min limit (increasing with velocity) ---
        min_effort = self._saturation_effort * (-1.0 - self._joint_vel / self.random_velocity_limit)
        min_effort = torch.clip(min_effort, min=-self.effort_limit, max=self._zeros_effort)

        return torch.clip(effort, min=min_effort, max=max_effort)


@configclass
class RandomPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for a randomized PD actuator with delay."""

    class_type: type = RandomPDActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed."""

    max_delay: int = 8
    """Maximum number of physics time-steps with which the actuator command may be delayed."""

    motor_strength: tuple = (0.9, 1.1)
    """Range for motor output strength randomization."""

    PD_random_range: tuple = (0.9, 1.1)
    """Range for PD gain randomization (multiplied on top of nominal stiffness/damping)."""

    pos_bias_range: tuple = (-0.04, 0.04)
    """Range for motor zero-position randomization (currently unused)."""

    t_n_vel_range: tuple = (1 / 3, 2 / 3)
    """Range for velocity saturation curve parameter (T = Tmax - k*n, where n is rotation speed)."""
