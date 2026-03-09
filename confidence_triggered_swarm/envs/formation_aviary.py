"""FormationAviary — multi-drone formation + waypoint tracking env.

Extends BaseRLAviary from gym-pybullet-drones v2.0. The swarm centroid
visits a sequence of 3D waypoints while drones maintain formation offsets.
"""

from __future__ import annotations

import numpy as np
import pybullet as p
from gymnasium import spaces
from numpy.typing import NDArray

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)


class FormationAviary(BaseRLAviary):
    """Multi-drone formation + waypoint tracking environment.

    Reward combines formation keeping, waypoint tracking, collision penalty,
    and a small alive bonus. Episode ends when all waypoints are reached
    or on timeout/out-of-bounds/excessive tilt.
    """

    _EXTRA_OBS_DIM: int = 3  # [dx, dy, dz] to current waypoint

    def __init__(
        self,
        num_drones: int = 2,
        waypoints: list | NDArray | None = None,
        formation_offsets: list | NDArray | None = None,
        waypoint_threshold: float = 0.2,
        gui: bool = False,
        record: bool = False,
        freq: int = 240,
        ctrl_freq: int = 30,
        episode_len_sec: float = 15.0,
        init_height: float = 0.5,
        tilt_threshold: float = 1.0,
        z_min: float = 0.01,
        speed_limit: float = 0.5,
        **kwargs,
    ) -> None:
        # must set before super().__init__() which calls _observationSpace
        self.EPISODE_LEN_SEC: float = episode_len_sec

        self.init_height = float(init_height)
        self.tilt_threshold = float(tilt_threshold)
        self.z_min = float(z_min)
        self._speed_limit = float(speed_limit)

        # waypoints for the formation centroid
        if waypoints is not None:
            self.waypoints = np.asarray(waypoints, dtype=np.float64)
        else:
            # simple 3-point course at init_height
            # first waypoint offset from start so bonus isn't free
            self.waypoints = np.array(
                [
                    [0.15, 0.15, self.init_height],
                    [0.4, 0.4, self.init_height],
                    [0.6, 0.0, self.init_height + 0.2],
                ],
                dtype=np.float64,
            )

        # formation offsets relative to centroid
        if formation_offsets is not None:
            self.formation_offsets = np.asarray(formation_offsets, dtype=np.float64)
        else:
            self.formation_offsets = np.zeros((num_drones, 3), dtype=np.float64)
            for i in range(num_drones):
                self.formation_offsets[i, 0] = (i - (num_drones - 1) / 2.0) * 0.5

        self.waypoint_threshold = float(waypoint_threshold)
        self.current_waypoint_idx: int = 0

        # initial positions (in formation at init_height)
        initial_xyzs = np.zeros((num_drones, 3), dtype=np.float64)
        for i in range(num_drones):
            initial_xyzs[i, 0] = self.formation_offsets[i, 0]
            initial_xyzs[i, 1] = self.formation_offsets[i, 1]
            initial_xyzs[i, 2] = self.init_height

        drone_model = kwargs.pop("drone_model", DroneModel.CF2X)
        physics = kwargs.pop("physics", Physics.PYB)
        obs = kwargs.pop("obs", ObservationType.KIN)
        act = kwargs.pop("act", ActionType.VEL)

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            initial_xyzs=initial_xyzs,
            physics=physics,
            pyb_freq=freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
            **kwargs,
        )

        # override default SPEED_LIMIT after init
        self.SPEED_LIMIT = self._speed_limit

    def reset(self, seed: int = None, options: dict = None):
        """Reset env and waypoint tracking state."""
        self.current_waypoint_idx = 0
        return super().reset(seed=seed, options=options)

    # -- observation space / computation --

    def _observationSpace(self) -> spaces.Box:
        """Extend base KIN obs with 3 waypoint-relative dims."""
        base_space = super()._observationSpace()
        base_low = base_space.low
        base_high = base_space.high

        extra_low = np.full(
            (self.NUM_DRONES, self._EXTRA_OBS_DIM), -np.inf, dtype=np.float32
        )
        extra_high = np.full(
            (self.NUM_DRONES, self._EXTRA_OBS_DIM), np.inf, dtype=np.float32
        )
        return spaces.Box(
            low=np.hstack([base_low, extra_low]),
            high=np.hstack([base_high, extra_high]),
            dtype=np.float32,
        )

    def _computeObs(self) -> NDArray:
        """Append [dx, dy, dz] to current waypoint for each drone."""
        base_obs = super()._computeObs()
        current_wp = self.waypoints[self.current_waypoint_idx]

        wp_offsets = np.zeros(
            (self.NUM_DRONES, self._EXTRA_OBS_DIM), dtype=np.float32
        )
        for i in range(self.NUM_DRONES):
            wp_offsets[i, :] = current_wp - self.pos[i, :]

        return np.hstack([base_obs, wp_offsets]).astype(np.float32)

    # -- reward --

    def _computeReward(self) -> float:
        """Formation + waypoint tracking reward (single scalar).

        Components per drone (summed over all drones):
        - waypoint tracking: max(0, 2 - 2*dist)
        - formation keeping: max(0, 1 - 3*error)
        - waypoint bonus: +10 when centroid reaches waypoint
        - collision penalty: -5 if any drone within 0.08m
        - alive bonus: +0.1 per step
        - boundary penalty: ramp for low altitude or high tilt
        """
        reward = 0.0
        centroid = np.mean(self.pos[: self.NUM_DRONES], axis=0)
        current_wp = self.waypoints[self.current_waypoint_idx]
        waypoint_dist = float(np.linalg.norm(centroid - current_wp))
        waypoint_reached = waypoint_dist < self.waypoint_threshold

        for i in range(self.NUM_DRONES):
            # formation keeping
            ideal_pos = centroid + self.formation_offsets[i]
            formation_error = float(np.linalg.norm(self.pos[i] - ideal_pos))
            r_formation = max(0.0, 1.0 - formation_error * 3.0)

            # waypoint tracking
            r_waypoint = max(0.0, 2.0 - 2.0 * waypoint_dist)

            # waypoint reached bonus
            r_bonus = 10.0 if waypoint_reached else 0.0

            # collision penalty
            r_collision = 0.0
            for j in range(self.NUM_DRONES):
                if j == i:
                    continue
                dist_ij = float(np.linalg.norm(self.pos[i] - self.pos[j]))
                if dist_ij < 0.08:
                    r_collision = -5.0
                    break

            r_alive = 0.1

            # boundary proximity penalty
            state = self._getDroneStateVector(i)
            z_i = state[2]
            roll_i, pitch_i = abs(state[7]), abs(state[8])
            r_boundary = 0.0
            if z_i < 0.1:
                r_boundary -= 2.0 * (0.1 - z_i) / 0.1
            tilt_margin = 0.6
            if roll_i > tilt_margin or pitch_i > tilt_margin:
                excess = max(roll_i - tilt_margin, pitch_i - tilt_margin)
                r_boundary -= 2.0 * excess

            reward += r_formation + r_waypoint + r_bonus + r_collision + r_alive + r_boundary

        # advance waypoint (once per step, not per drone)
        if waypoint_reached and self.current_waypoint_idx < len(self.waypoints) - 1:
            self.current_waypoint_idx += 1

        return reward

    # -- termination / truncation --

    def _computeTerminated(self) -> bool:
        """True when all waypoints have been reached."""
        if self.current_waypoint_idx >= len(self.waypoints) - 1:
            centroid = np.mean(self.pos[: self.NUM_DRONES], axis=0)
            last_wp = self.waypoints[-1]
            if float(np.linalg.norm(centroid - last_wp)) < self.waypoint_threshold:
                return True
        return False

    def _computeTruncated(self) -> bool:
        """True on timeout, out-of-bounds, or excessive tilt."""
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            x, y, z = state[0], state[1], state[2]
            roll, pitch = state[7], state[8]

            if abs(x) > 3.0 or abs(y) > 3.0 or z > 3.0 or z < self.z_min:
                return True
            if abs(roll) > self.tilt_threshold or abs(pitch) > self.tilt_threshold:
                return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    # -- info --

    def _computeInfo(self) -> dict:
        """Diagnostic info about the current step."""
        centroid = np.mean(self.pos[: self.NUM_DRONES], axis=0)
        current_wp = self.waypoints[self.current_waypoint_idx]

        formation_errors = []
        for i in range(self.NUM_DRONES):
            ideal = centroid + self.formation_offsets[i]
            formation_errors.append(float(np.linalg.norm(self.pos[i] - ideal)))

        return {
            "current_waypoint_idx": self.current_waypoint_idx,
            "total_waypoints": len(self.waypoints),
            "waypoint_error": float(np.linalg.norm(centroid - current_wp)),
            "mean_formation_error": float(np.mean(formation_errors)),
            "drone_positions": self.pos[: self.NUM_DRONES].copy(),
            "centroid": centroid.copy(),
        }
