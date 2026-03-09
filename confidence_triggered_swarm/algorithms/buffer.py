"""Rollout buffer for on-policy PPO data collection.

Stores per-drone transitions and computes GAE returns/advantages.
For IPPO the buffer holds rollout_steps * num_drones entries since
each env step gives one transition per drone.
"""

from typing import Dict, Iterator

import numpy as np
import torch


class RolloutBuffer:
    """Stores rollout data and computes GAE for PPO updates.

    Pre-allocates numpy arrays, fills them during collection,
    then yields random minibatches as torch tensors.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self._pos = 0

        # pre-allocate storage
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # filled after rollout
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add one transition. Bumps the position pointer."""
        assert self._pos < self.buffer_size, (
            f"Buffer overflow: pos={self._pos}, size={self.buffer_size}. "
            "Call reset() before collecting a new rollout."
        )

        self.observations[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.values[self._pos] = value
        self.log_probs[self._pos] = log_prob
        self.dones[self._pos] = float(done)

        self._pos += 1

    def compute_returns_and_advantages(self, last_value: float) -> None:
        """Compute GAE advantages and discounted returns.

        last_value is V(s_{T+1}) for bootstrapping (0 if episode done).
        """
        size = self._pos  # might be less than buffer_size

        last_gae = 0.0
        for t in reversed(range(size)):
            if t == size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:size] = self.advantages[:size] + self.values[:size]

    def get_batches(self, batch_size: int = 64) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield shuffled minibatches as torch tensors.

        Advantages are normalized (zero mean, unit std) per batch.
        """
        size = self._pos
        indices = np.arange(size)
        np.random.shuffle(indices)

        for start in range(0, size, batch_size):
            end = start + batch_size
            if end > size:
                break  # skip incomplete last batch
            batch_idx = indices[start:end]

            # normalize advantages within the minibatch
            adv = self.advantages[batch_idx].copy()
            adv_std = adv.std()
            if adv_std > 1e-8:
                adv = (adv - adv.mean()) / (adv_std + 1e-8)
            else:
                adv = adv - adv.mean()

            yield {
                "observations": torch.as_tensor(self.observations[batch_idx], dtype=torch.float32),
                "actions": torch.as_tensor(self.actions[batch_idx], dtype=torch.float32),
                "log_probs": torch.as_tensor(self.log_probs[batch_idx], dtype=torch.float32),
                "advantages": torch.as_tensor(adv, dtype=torch.float32),
                "returns": torch.as_tensor(self.returns[batch_idx], dtype=torch.float32),
                "values": torch.as_tensor(self.values[batch_idx], dtype=torch.float32),
            }

    def reset(self) -> None:
        """Reset for a new rollout."""
        self._pos = 0
