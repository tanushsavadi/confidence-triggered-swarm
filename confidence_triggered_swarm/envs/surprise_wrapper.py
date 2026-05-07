"""Surprise injection wrapper for distributional-shift experiments.

Wraps a FormationAviary to inject configurable perturbations:
- Wind (random external forces via pybullet)
- Sensor noise (additive Gaussian + optional dropout)
- Actuator weakening (scale one drone's actions toward zero)
- Goal relocation (randomly perturb current waypoint)

Perturbation params are bundled in SurpriseConfig with named severity
presets via SurpriseConfig.from_severity().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium
import numpy as np
import pybullet as p
from numpy.typing import NDArray


@dataclass
class SurpriseConfig:
    """Config bundle for all surprise perturbations.

    Each perturbation has an *_enabled flag so they can be toggled independently.
    """

    # wind
    wind_enabled: bool = False
    wind_force_range: tuple = (0.0, 0.0)  # (min, max) Newtons
    wind_change_freq: int = 50  # steps between wind direction changes

    # sensor noise
    sensor_noise_enabled: bool = False
    sensor_noise_std: float = 0.0
    sensor_dropout_prob: float = 0.0  # prob of zeroing a drone's obs

    # actuator weakening
    actuator_enabled: bool = False
    actuator_drone_idx: int = 0
    actuator_weakness: float = 1.0  # 1.0 = healthy, 0.5 = half thrust

    # goal relocation
    goal_shift_enabled: bool = False
    goal_shift_prob: float = 0.0
    goal_shift_magnitude: float = 0.3  # metres

    @classmethod
    def from_severity(cls, severity: str) -> "SurpriseConfig":
        """Return a preset config for the given severity level."""
        presets: dict[str, SurpriseConfig] = {
            "clean": cls(
                wind_enabled=False,
                sensor_noise_enabled=False,
                actuator_enabled=False,
                goal_shift_enabled=False,
            ),
            "mild": cls(
                wind_enabled=True,
                wind_force_range=(0.0, 0.02),
                wind_change_freq=100,
                sensor_noise_enabled=True,
                sensor_noise_std=0.01,
                sensor_dropout_prob=0.0,
                actuator_enabled=False,
                goal_shift_enabled=False,
            ),
            "moderate": cls(
                wind_enabled=True,
                wind_force_range=(0.0, 0.05),
                wind_change_freq=75,
                sensor_noise_enabled=True,
                sensor_noise_std=0.02,
                sensor_dropout_prob=0.02,
                actuator_enabled=True,
                actuator_weakness=0.85,
                goal_shift_enabled=False,
            ),
            "severe": cls(
                wind_enabled=True,
                wind_force_range=(0.0, 0.1),
                wind_change_freq=50,
                sensor_noise_enabled=True,
                sensor_noise_std=0.05,
                sensor_dropout_prob=0.05,
                actuator_enabled=True,
                actuator_weakness=0.7,
                goal_shift_enabled=True,
                goal_shift_prob=0.001,
                goal_shift_magnitude=0.1,
            ),
        }
        if severity not in presets:
            raise ValueError(
                f"Unknown severity: {severity}. "
                f"Choose from {list(presets.keys())}"
            )
        return presets[severity]

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> "SurpriseConfig":
        """Create from a config dictionary (e.g. from YAML)."""
        wind_range = config_dict.get("wind_range", [0.0, 0.0])
        sensor_noise = config_dict.get("sensor_noise", 0.0)
        actuator_weakness = config_dict.get("actuator_weakness", 1.0)
        goal_shift_prob = config_dict.get("goal_shift_prob", 0.0)

        return cls(
            wind_enabled=wind_range[1] > 0,
            wind_force_range=tuple(wind_range),
            sensor_noise_enabled=sensor_noise > 0,
            sensor_noise_std=sensor_noise,
            actuator_enabled=actuator_weakness < 1.0,
            actuator_weakness=actuator_weakness,
            goal_shift_enabled=goal_shift_prob > 0,
            goal_shift_prob=goal_shift_prob,
        )


class SurpriseWrapper(gymnasium.Wrapper):
    """Gym wrapper that injects distributional shifts into a drone env."""

    def __init__(
        self,
        env: gymnasium.Env,
        config: SurpriseConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        self.config = config if config is not None else SurpriseConfig()
        self._rng = np.random.default_rng(seed)

        self._current_wind: NDArray = np.zeros(3, dtype=np.float64)
        self._wind_step_counter: int = 0

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[NDArray, Dict[str, Any]]:
        """Reset base env and internal surprise state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current_wind = np.zeros(3, dtype=np.float64)
        self._wind_step_counter = 0

        obs, info = self.env.reset(seed=seed, options=options)

        if self.config.sensor_noise_enabled:
            obs = self._add_sensor_noise(obs)

        info["surprise_active"] = self._any_surprise_active()
        return obs, info

    def step(
        self, action: NDArray
    ) -> Tuple[NDArray, float, bool, bool, Dict[str, Any]]:
        """Step with surprise injections.

        Order: wind forces -> weaken actuator -> base step -> sensor noise -> goal shift.
        """
        if self.config.wind_enabled:
            self._apply_wind()

        if self.config.actuator_enabled:
            action = self._weaken_actuator(action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.config.sensor_noise_enabled:
            obs = self._add_sensor_noise(obs)

        if self.config.goal_shift_enabled:
            self._maybe_shift_goal()

        info["surprise_active"] = self._any_surprise_active()
        info["wind_applied"] = (
            self._current_wind.copy() if self.config.wind_enabled else None
        )

        return obs, reward, terminated, truncated, info

    # -- internal helpers --

    def _any_surprise_active(self) -> bool:
        c = self.config
        return (
            c.wind_enabled
            or c.sensor_noise_enabled
            or c.actuator_enabled
            or c.goal_shift_enabled
        )

    def _apply_wind(self) -> None:
        """Apply random external wind force to every drone.

        Wind direction resampled every wind_change_freq steps.
        In VEL mode the PID handles stability so moderate wind
        creates velocity tracking errors rather than instant crashes.
        """
        self._wind_step_counter += 1
        if self._wind_step_counter % self.config.wind_change_freq == 1:
            direction = self._rng.standard_normal(3)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
            lo, hi = self.config.wind_force_range
            magnitude = self._rng.uniform(lo, hi)
            self._current_wind = direction * magnitude

        base = self.env.unwrapped
        for i in range(base.NUM_DRONES):
            p.applyExternalForce(
                objectUniqueId=int(base.DRONE_IDS[i]),
                linkIndex=4,  # center-of-mass link
                forceObj=self._current_wind.tolist(),
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
                physicsClientId=base.CLIENT,
            )

    def _add_sensor_noise(self, obs: NDArray) -> NDArray:
        """Additive Gaussian noise + optional dropout (zero entire drone obs)."""
        obs = obs.copy()

        noise = self._rng.normal(
            loc=0.0, scale=self.config.sensor_noise_std, size=obs.shape
        ).astype(obs.dtype)
        obs = obs + noise

        if self.config.sensor_dropout_prob > 0.0:
            base = self.env.unwrapped
            for i in range(base.NUM_DRONES):
                if self._rng.random() < self.config.sensor_dropout_prob:
                    obs[i, :] = 0.0

        obs = np.clip(
            obs,
            self.observation_space.low,
            self.observation_space.high,
        )
        return obs

    def _weaken_actuator(self, action: NDArray) -> NDArray:
        """Scale designated drone's action toward zero (reduce thrust authority)."""
        action = action.copy()
        idx = self.config.actuator_drone_idx
        base = self.env.unwrapped
        if 0 <= idx < base.NUM_DRONES:
            action[idx] = action[idx] * self.config.actuator_weakness
        return action

    def _maybe_shift_goal(self) -> None:
        """With some probability, perturb the current waypoint."""
        if self._rng.random() >= self.config.goal_shift_prob:
            return

        base = self.env.unwrapped
        if not hasattr(base, "waypoints") or not hasattr(base, "current_waypoint_idx"):
            return

        idx = base.current_waypoint_idx
        if idx >= len(base.waypoints):
            return

        shift = self._rng.standard_normal(3) * self.config.goal_shift_magnitude
        base.waypoints[idx] = base.waypoints[idx] + shift


class DomainRandomizationWrapper(SurpriseWrapper):
    """Sample a fresh surprise configuration at each episode reset.

    This wrapper is meant for robust-training baselines, where the policy sees a
    range of perturbations during training instead of only clean flight.
    """

    DEFAULT_RANGES: Dict[str, Tuple[float, float]] = {
        "wind_force_range": (0.0, 0.08),
        "sensor_noise_std_range": (0.0, 0.04),
        "sensor_dropout_prob_range": (0.0, 0.04),
        "actuator_weakness_range": (0.75, 1.0),
        "goal_shift_prob_range": (0.0, 0.0008),
        "goal_shift_magnitude_range": (0.0, 0.1),
    }

    def __init__(
        self,
        env: gymnasium.Env,
        ranges: Dict[str, Tuple[float, float]] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(env, SurpriseConfig(), seed=seed)
        self.ranges = dict(self.DEFAULT_RANGES)
        if ranges:
            for key, value in ranges.items():
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    self.ranges[key] = (float(value[0]), float(value[1]))

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[NDArray, Dict[str, Any]]:
        """Sample perturbation magnitudes, then reset the wrapped env."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.config = self._sample_config()
        obs, info = super().reset(seed=seed, options=options)
        info["domain_randomized"] = True
        info["sampled_surprise_config"] = {
            "wind_force_range": self.config.wind_force_range,
            "sensor_noise_std": self.config.sensor_noise_std,
            "sensor_dropout_prob": self.config.sensor_dropout_prob,
            "actuator_weakness": self.config.actuator_weakness,
            "goal_shift_prob": self.config.goal_shift_prob,
            "goal_shift_magnitude": self.config.goal_shift_magnitude,
        }
        return obs, info

    def _uniform(self, key: str) -> float:
        lo, hi = self.ranges[key]
        return float(self._rng.uniform(lo, hi))

    def _sample_config(self) -> SurpriseConfig:
        wind_max = self._uniform("wind_force_range")
        sensor_noise = self._uniform("sensor_noise_std_range")
        dropout = self._uniform("sensor_dropout_prob_range")
        actuator = self._uniform("actuator_weakness_range")
        goal_prob = self._uniform("goal_shift_prob_range")
        goal_mag = self._uniform("goal_shift_magnitude_range")

        return SurpriseConfig(
            wind_enabled=wind_max > 0.0,
            wind_force_range=(0.0, wind_max),
            wind_change_freq=75,
            sensor_noise_enabled=sensor_noise > 0.0 or dropout > 0.0,
            sensor_noise_std=sensor_noise,
            sensor_dropout_prob=dropout,
            actuator_enabled=actuator < 0.999,
            actuator_weakness=actuator,
            goal_shift_enabled=goal_prob > 0.0 and goal_mag > 0.0,
            goal_shift_prob=goal_prob,
            goal_shift_magnitude=goal_mag,
        )
