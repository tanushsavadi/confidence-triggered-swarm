from __future__ import annotations

"""ActorCritic policy with MC Dropout for uncertainty estimation.

Shared-backbone MLP with separate actor/critic heads.
Dropout stays active during inference for confidence estimation.
Orthogonal init for PPO stability.
"""

from typing import List, Tuple

import math

import torch
import torch.nn as nn
from torch.distributions import Normal


# supported activations
ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def _init_weights(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    """Orthogonal init for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ActorCritic(nn.Module):
    """Actor-Critic network with MC Dropout.

    Shared MLP backbone -> actor head (mean + learnable log_std) + critic head.
    Dropout layers stay active at inference for MC uncertainty estimates.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] | None = None,
        activation: str = "tanh",
        mc_dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dims = hidden_dims or [256, 256]
        self.mc_dropout_p = mc_dropout_p

        act_cls = ACTIVATIONS.get(activation, nn.Tanh)

        # shared feature extractor
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_cls())
            layers.append(nn.Dropout(p=mc_dropout_p))
            in_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)

        # orthogonal init on the feature extractor
        for module in self.feature_extractor:
            if isinstance(module, nn.Linear):
                _init_weights(module, gain=math.sqrt(2))

        # actor head
        self.actor_head = nn.Linear(self.hidden_dims[-1], act_dim)
        _init_weights(self.actor_head, gain=0.01)

        # learnable log-std per action dim
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # critic head
        self.critic_head = nn.Linear(self.hidden_dims[-1], 1)
        _init_weights(self.critic_head, gain=1.0)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[Normal, torch.Tensor]:
        """Forward pass -> (action distribution, value estimate)."""
        features = self.feature_extractor(obs)
        action_mean = self.actor_head(features)
        action_std = self.log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        value = self.critic_head(features)
        return dist, value

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick action(s) given obs. Returns (action, log_prob, value, entropy)."""
        dist, value = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, value.squeeze(-1), entropy

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate actions under current policy (for PPO update)."""
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value.squeeze(-1), entropy

    # -- confidence / uncertainty helpers --

    def get_entropy(self, obs: torch.Tensor) -> float:
        """Mean entropy for confidence monitoring."""
        with torch.no_grad():
            dist, _ = self.forward(obs)
            entropy = dist.entropy().sum(dim=-1).mean().item()
        return entropy

    def mc_predict(
        self, obs: torch.Tensor, n_samples: int = 10
    ) -> Tuple[torch.Tensor, float]:
        """MC Dropout: run n stochastic forward passes, return mean action + variance."""
        self.train()  # keep dropout on
        predictions: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(n_samples):
                dist, _ = self.forward(obs)
                predictions.append(dist.mean)
        self.eval()
        preds = torch.stack(predictions)
        mean_action = preds.mean(dim=0)
        variance = preds.var(dim=0).mean().item()
        return mean_action, variance
