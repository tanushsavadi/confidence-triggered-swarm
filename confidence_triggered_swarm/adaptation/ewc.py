"""Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.

Saves a snapshot of the clean-trained policy weights and approximates the
diagonal Fisher Information Matrix. During adaptation, adds a quadratic
penalty that discourages big changes to "important" weights:

    L_ewc = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from confidence_triggered_swarm.algorithms.policy import ActorCritic


class EWCRegularizer:
    """Lightweight EWC for continual RL.

    Stores a param snapshot + diagonal Fisher, then computes a penalty
    during adaptation to keep weights close to the original policy.
    """

    def __init__(
        self,
        policy: ActorCritic,
        ewc_lambda: float = 1000.0,
        device: str = "cpu",
    ) -> None:
        self.policy = policy
        self.ewc_lambda = ewc_lambda
        self.device = device

        # anchor params (clean-trained policy snapshot)
        self.saved_params: Dict[str, torch.Tensor] = {}

        # diagonal Fisher approximation
        self.fisher: Dict[str, torch.Tensor] = {}

        self._has_snapshot: bool = False
        self._has_fisher: bool = False

    def snapshot(self) -> None:
        """Save current params as the anchor point (call after clean training)."""
        self.saved_params = {}
        for name, param in self.policy.named_parameters():
            self.saved_params[name] = param.data.clone().detach()
        self._has_snapshot = True

    def compute_fisher(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> None:
        """Compute diagonal Fisher from recent experience.

        Uses empirical Fisher: F_ii = E[ (d log pi / d theta_i)^2 ]
        Gradients are per-sample, squared, then averaged.
        """
        observations = observations.to(self.device)
        actions = actions.to(self.device)

        # optional subsampling to limit compute
        if n_samples is not None and n_samples < observations.shape[0]:
            indices = torch.randperm(observations.shape[0])[:n_samples]
            observations = observations[indices]
            actions = actions[indices]

        self.fisher = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.policy.named_parameters()
        }

        self.policy.eval()
        n = observations.shape[0]

        for i in range(n):
            self.policy.zero_grad()
            obs_i = observations[i : i + 1]
            act_i = actions[i : i + 1]

            log_prob, _, _ = self.policy.evaluate_actions(obs_i, act_i)
            log_prob.backward()

            for name, param in self.policy.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.data.clone() ** 2

        # average
        for name in self.fisher:
            self.fisher[name] /= max(n, 1)

        self._has_fisher = True

    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty: 0.5 * lambda * sum F_i * (theta - theta*)^2.

        Returns 0 if snapshot or Fisher haven't been computed yet.
        """
        if not self._has_snapshot or not self._has_fisher:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.policy.named_parameters():
            if name in self.saved_params and name in self.fisher:
                diff = param - self.saved_params[name].to(self.device)
                loss = loss + (self.fisher[name].to(self.device) * diff ** 2).sum()

        return 0.5 * self.ewc_lambda * loss

    @property
    def is_ready(self) -> bool:
        """Whether both snapshot and Fisher have been computed."""
        return self._has_snapshot and self._has_fisher

    def reset(self) -> None:
        """Clear stored Fisher and param snapshots."""
        self.saved_params.clear()
        self.fisher.clear()
        self._has_snapshot = False
        self._has_fisher = False
