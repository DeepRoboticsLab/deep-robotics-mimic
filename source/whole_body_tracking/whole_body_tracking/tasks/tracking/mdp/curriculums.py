from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _linear_scale(current: float, low: float, high: float, floor: float = 0.1) -> float:
    """Linear ramp: `floor` below `low`, 1.0 above `high`, linear in between."""
    if current <= low:
        return floor
    if current >= high:
        return 1.0
    return floor + (1.0 - floor) * (current - low) / (high - low)


class MotionTrackingCurriculum(ManagerTermBase):  # type: ignore[misc]
    """Curriculum driven by mean episode length.

    A single scale in [0.1, 1.0] is computed from the peak mean episode length
    seen so far (monotonically non-decreasing):

        iteration < warmup_iters                             →  scale = 0.1
        peak_ep_len < low_ep_len_frac  * max_episode_length  →  scale = 0.1
        peak_ep_len > high_ep_len_frac * max_episode_length  →  scale = 1.0
        in between                                           →  linear ramp

    The same scale is applied to:
      - Regularization reward weights  (weight = base_weight * scale)
      - Event velocity ranges          (range  = max_range   * scale)
      - Termination thresholds         (loose → tight as scale 0.1 → 1.0)

    Args:
        reg_term_names: Reward term names whose weights are scaled.
        term_thresholds: ``{term_name: (base_threshold, max_threshold)}``.
            scale=0.1 → near max_threshold (loose); scale=1 → base_threshold (tight).
        event_velocity_ranges: ``{event_name: max_vel_range_dict}``.
            Each velocity value is multiplied by scale.
        low_ep_len_frac: Episode-length fraction below which scale = 0.1. Default 0.8.
        high_ep_len_frac: Episode-length fraction above which scale = 1.0. Default 0.9.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        raw = cfg.params.get("reg_term_names", [])
        self._reg_term_names: list[str] = list(raw) if not isinstance(raw, str) else []
        self._base_weights: dict[str, float] = {
            name: env.reward_manager.get_term_cfg(name).weight
            for name in self._reg_term_names
        }
        self._ema_ep_len: float = 0.0   # smoothed mean episode length
        self._peak_ep_len: float = 0.0  # monotonic peak of EMA — never decreases

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reg_term_names: list[str],
        term_thresholds: dict[str, tuple[float, float]] | None = None,
        event_velocity_ranges: dict[str, dict] | None = None,
        low_ep_len_frac: float = 0.8,
        high_ep_len_frac: float = 0.9,
        warmup_iters: int = 100,
        ema_alpha: float = 0.05,
        num_steps_per_env: int = 24,
    ) -> float:
        """Update the curriculum; returns the current scale in [0.1, 1.0]."""
        current_iter = env.common_step_counter / num_steps_per_env

        # During warmup the scale is fixed at 0.1; EMA/peak are not updated.
        if current_iter < warmup_iters:
            scale = 0.1
        else:
            if len(env_ids) > 0:
                batch_mean = float(env.episode_length_buf[env_ids].float().mean().item())
                self._ema_ep_len = ema_alpha * batch_mean + (1.0 - ema_alpha) * self._ema_ep_len
                self._peak_ep_len = max(self._peak_ep_len, self._ema_ep_len)

            low   = low_ep_len_frac  * env.max_episode_length
            high  = high_ep_len_frac * env.max_episode_length
            scale = _linear_scale(self._peak_ep_len, low, high)

        # --- regularization reward weights: base_weight * scale ---
        for name in reg_term_names:
            term_cfg = env.reward_manager.get_term_cfg(name)
            term_cfg.weight = self._base_weights[name] * scale
            env.reward_manager.set_term_cfg(name, term_cfg)

        # --- termination thresholds: loose (max_thr) → tight (base_thr) ---
        if term_thresholds:
            for term_name, (base_thr, max_thr) in term_thresholds.items():
                term_cfg = env.termination_manager.get_term_cfg(term_name)
                term_cfg.params["threshold"] = max_thr - (max_thr - base_thr) * scale
                env.termination_manager.set_term_cfg(term_name, term_cfg)

        # --- event velocity ranges: max_range * scale ---
        if event_velocity_ranges:
            for event_name, max_range in event_velocity_ranges.items():
                scaled_range = {
                    key: (max_range[key][0] * scale, max_range[key][1] * scale)
                    for key in max_range
                }
                event_cfg = env.event_manager.get_term_cfg(event_name)
                event_cfg.params["velocity_range"] = scaled_range
                env.event_manager.set_term_cfg(event_name, event_cfg)

        if "log" not in env.extras:
            env.extras["log"] = {}
        env.extras["log"]["Curriculum/ema_ep_len"]  = self._ema_ep_len
        env.extras["log"]["Curriculum/peak_ep_len"] = self._peak_ep_len
        env.extras["log"]["Curriculum/scale"]       = scale

        return scale
