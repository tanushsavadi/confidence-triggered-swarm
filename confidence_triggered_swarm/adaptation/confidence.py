"""Confidence monitoring via entropy + MC Dropout uncertainty.

Tracks two signals:
1. Policy entropy (primary) — high entropy = low confidence
2. MC Dropout variance (secondary) — high variance = low confidence

Confidence is normalized against a calibrated baseline from clean rollouts,
then smoothed over a rolling window. When avg confidence drops below the
threshold, the monitor signals that adaptation is needed.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from numpy.typing import NDArray

from confidence_triggered_swarm.algorithms.policy import ActorCritic
from confidence_triggered_swarm.utils.seeding import reset_env


class ConfidenceMonitor:
    """Monitors policy confidence using entropy + optional MC Dropout.

    Maintains a rolling window and triggers adaptation when avg confidence
    drops below threshold.
    """

    def __init__(
        self,
        policy: ActorCritic,
        confidence_threshold: float = 0.5,
        window_size: int = 50,
        use_mc_dropout: bool = True,
        mc_samples: int = 10,
        device: str = "cpu",
    ) -> None:
        self.policy = policy
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.use_mc_dropout = use_mc_dropout
        self.mc_samples = mc_samples
        self.device = device

        # rolling windows
        self.confidence_history: deque[float] = deque(maxlen=window_size)
        self.entropy_history: deque[float] = deque(maxlen=window_size)
        self.mc_variance_history: deque[float] = deque(maxlen=window_size)

        # baseline stats (set during calibration)
        self.baseline_entropy: float = 1.0
        self.baseline_entropy_std: float = 1.0
        self.baseline_mc_variance: float = 1.0
        self.baseline_mc_variance_std: float = 1.0

        self.adaptation_count: int = 0
        self._calibrated: bool = False

    # -- calibration --

    def calibrate(
        self,
        env: Any,
        n_episodes: int = 5,
        base_seed: int | None = None,
        seed_stream: int = 0,
    ) -> None:
        """Run policy on clean env to get baseline entropy/variance stats.

        These are used to z-score normalize future readings.
        """
        all_entropies: List[float] = []
        all_variances: List[float] = []

        self.policy.eval()

        for _ep in range(n_episodes):
            obs, _info = reset_env(env, base_seed, _ep, seed_stream)
            done = False
            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)

                # per-drone entropy
                for i in range(obs.shape[0]):
                    entropy = self.policy.get_entropy(obs_tensor[i : i + 1])
                    all_entropies.append(entropy)

                    if self.use_mc_dropout:
                        _, variance = self.policy.mc_predict(
                            obs_tensor[i : i + 1], self.mc_samples
                        )
                        all_variances.append(variance)

                # deterministic actions for stable calibration
                obs_tensor_act = torch.FloatTensor(obs).to(self.device)
                if obs_tensor_act.dim() == 2:  # (num_drones, obs_dim)
                    actions_list = []
                    for d in range(obs_tensor_act.shape[0]):
                        with torch.no_grad():
                            action_d, _, _, _ = self.policy.act(
                                obs_tensor_act[d].unsqueeze(0), deterministic=True
                            )
                        actions_list.append(action_d.cpu().numpy().flatten())
                    action = np.array(actions_list)
                else:
                    with torch.no_grad():
                        action_d, _, _, _ = self.policy.act(
                            obs_tensor_act.unsqueeze(0), deterministic=True
                        )
                    action = action_d.cpu().numpy().flatten()
                obs, _reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)

        self.baseline_entropy = float(np.mean(all_entropies)) if all_entropies else 1.0
        self.baseline_entropy_std = float(np.std(all_entropies)) + 1e-8

        if self.use_mc_dropout and all_variances:
            self.baseline_mc_variance = float(np.mean(all_variances))
            self.baseline_mc_variance_std = float(np.std(all_variances)) + 1e-8

        self._calibrated = True

    # -- confidence computation --

    def compute_confidence(self, obs: NDArray[np.floating]) -> Dict[str, Any]:
        """Compute confidence from a multi-drone observation.

        Returns dict with confidence (0-1), raw entropy, mc_variance, z-score,
        and per-drone confidence values.
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)

        per_drone_entropy: List[float] = []
        per_drone_variance: List[float] = []

        for i in range(obs.shape[0]):
            single_obs = obs_tensor[i : i + 1]
            ent = self.policy.get_entropy(single_obs)
            per_drone_entropy.append(ent)

            if self.use_mc_dropout:
                _, var = self.policy.mc_predict(single_obs, self.mc_samples)
                per_drone_variance.append(var)

        mean_entropy = float(np.mean(per_drone_entropy))
        entropy_zscore = (mean_entropy - self.baseline_entropy) / self.baseline_entropy_std

        # sigmoid mapping of z-score:
        # z=0 (baseline) -> confidence ~0.5
        # z>>0 (high entropy) -> low confidence
        # z<<0 (low entropy) -> high confidence
        entropy_confidence: float = 1.0 / (1.0 + float(np.exp(entropy_zscore)))

        mean_variance: Optional[float] = None
        if self.use_mc_dropout and per_drone_variance:
            mean_variance = float(np.mean(per_drone_variance))
            var_zscore = (
                mean_variance - self.baseline_mc_variance
            ) / self.baseline_mc_variance_std
            mc_confidence = 1.0 / (1.0 + float(np.exp(var_zscore)))
            # blend: 70% entropy, 30% MC variance
            confidence = 0.7 * entropy_confidence + 0.3 * mc_confidence
        else:
            confidence = entropy_confidence

        # per-drone confidence (entropy-only for simplicity)
        per_drone_conf = [
            1.0
            / (
                1.0
                + float(
                    np.exp(
                        (e - self.baseline_entropy) / self.baseline_entropy_std
                    )
                )
            )
            for e in per_drone_entropy
        ]

        return {
            "confidence": float(confidence),
            "entropy": float(mean_entropy),
            "mc_variance": float(mean_variance) if mean_variance is not None else None,
            "entropy_zscore": float(entropy_zscore),
            "per_drone_confidence": per_drone_conf,
        }

    # -- update and trigger --

    def update(self, obs: NDArray[np.floating]) -> Dict[str, Any]:
        """Compute confidence, update rolling histories, return result."""
        result = self.compute_confidence(obs)
        self.confidence_history.append(result["confidence"])
        self.entropy_history.append(result["entropy"])
        if result["mc_variance"] is not None:
            self.mc_variance_history.append(result["mc_variance"])
        return result

    def should_adapt(self) -> bool:
        """Check if adaptation should fire.

        Needs a few readings before triggering (min window_size//3 or 10).
        Returns True when rolling avg confidence < threshold.
        """
        min_readings = min(self.window_size // 3, 10)  # need fewer readings to fire faster
        if len(self.confidence_history) < min_readings:
            return False
        avg_confidence = float(np.mean(list(self.confidence_history)))
        return avg_confidence < self.confidence_threshold

    # -- convenience accessors --

    def get_mean_confidence(self) -> float:
        """Current mean confidence, or 1.0 if no data yet."""
        if not self.confidence_history:
            return 1.0
        return float(np.mean(list(self.confidence_history)))

    def get_stats(self) -> Dict[str, float]:
        """Summary stats for logging."""
        return {
            "avg_confidence": (
                float(np.mean(list(self.confidence_history)))
                if self.confidence_history
                else 1.0
            ),
            "min_confidence": (
                float(min(self.confidence_history))
                if self.confidence_history
                else 1.0
            ),
            "avg_entropy": (
                float(np.mean(list(self.entropy_history)))
                if self.entropy_history
                else 0.0
            ),
            "adaptation_count": self.adaptation_count,
            "window_fill": len(self.confidence_history) / self.window_size,
        }

    def reset(self) -> None:
        """Clear rolling windows (baseline calibration is preserved)."""
        self.confidence_history.clear()
        self.entropy_history.clear()
        self.mc_variance_history.clear()
