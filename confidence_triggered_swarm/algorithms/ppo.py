from __future__ import annotations

"""PPO Agent for Independent PPO (IPPO) multi-drone training.

Each drone shares the same policy but acts independently. The agent
handles rollout collection, GAE computation, and PPO-Clip updates.

IPPO detail: each env step produces num_drones transitions that all
go into one buffer, so the shared policy trains on all drones' experience.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from confidence_triggered_swarm.algorithms.policy import ActorCritic
from confidence_triggered_swarm.algorithms.buffer import RolloutBuffer


class PPOAgent:
    """PPO agent for IPPO multi-drone control.

    Manages the shared policy, rollout buffer, and PPO update loop.
    Supports checkpointing and optional MetricsLogger integration.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 4,
        num_drones: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        rollout_steps: int = 2048,
        hidden_dims: List[int] | None = None,
        activation: str = "tanh",
        mc_dropout_p: float = 0.1,
        device: str = "auto",
    ) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_drones = num_drones
        self.rollout_steps = rollout_steps
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # pick device: CUDA > MPS > CPU
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # shared policy network
        self.policy = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            mc_dropout_p=mc_dropout_p,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # rollout buffer — IPPO: each env step -> num_drones transitions
        buffer_size = rollout_steps * num_drones
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            obs_dim=obs_dim,
            act_dim=act_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    def select_actions(
        self, obs_multi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select actions for all drones.

        Returns (actions, log_probs, values, entropies), all numpy arrays.
        """
        obs_tensor = torch.as_tensor(obs_multi, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            actions, log_probs, values, entropies = self.policy.act(obs_tensor)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
            entropies.cpu().numpy(),
        )

    def collect_rollout(self, env: Any) -> Dict[str, float]:
        """Collect rollout_steps env steps into the buffer.

        Each step produces num_drones transitions (IPPO).
        """
        self.policy.eval()
        self.buffer.reset()

        obs, info = env.reset()
        episode_rewards: list[float] = []
        current_ep_reward = 0.0
        episode_count = 0
        all_entropies: list[float] = []

        for _step in range(self.rollout_steps):
            actions, log_probs, values, ents = self.select_actions(obs)
            all_entropies.extend(ents.tolist())

            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = bool(terminated or truncated)

            # split shared reward across drones
            per_drone_reward = float(reward) / self.num_drones
            for i in range(self.num_drones):
                self.buffer.add(
                    obs=obs[i],
                    action=actions[i],
                    reward=per_drone_reward,
                    value=float(values[i]),
                    log_prob=float(log_probs[i]),
                    done=done,
                )

            current_ep_reward += float(reward)

            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                episode_count += 1
                obs, info = env.reset()
            else:
                obs = next_obs

        # bootstrap value for the last state
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
            _, last_values = self.policy.forward(obs_tensor)
            last_value = last_values.mean().item()  # mean across drones

        self.buffer.compute_returns_and_advantages(last_value)

        return {
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else current_ep_reward,
            "mean_entropy": float(np.mean(all_entropies)) if all_entropies else 0.0,
            "episode_count": episode_count,
            "total_steps": self.rollout_steps * self.num_drones,
        }

    def update(self) -> Dict[str, float]:
        """PPO-Clip update on the collected rollout buffer.

        Runs n_epochs over random minibatches with KL early stopping.
        """
        self.policy.train()

        all_policy_losses: list[float] = []
        all_value_losses: list[float] = []
        all_entropy_losses: list[float] = []
        all_kls: list[float] = []
        all_clip_fracs: list[float] = []

        # target KL for early stopping — keeps training stable
        target_kl: float = 0.015
        early_stopped = False

        for _epoch in range(self.n_epochs):
            if early_stopped:
                break
            for batch in self.buffer.get_batches(self.batch_size):
                obs = batch["observations"].to(self.device)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)
                old_values = batch["values"].to(self.device)

                new_log_probs, new_values, entropy = self.policy.evaluate_actions(obs, actions)

                # clipped surrogate policy loss
                log_ratio = new_log_probs - old_log_probs
                log_ratio = torch.clamp(log_ratio, -20.0, 20.0)  # prevent exp() blowup
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # clipped value loss
                value_pred_clipped = old_values + torch.clamp(
                    new_values - old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss_unclipped = (new_values - returns) ** 2
                value_loss_clipped = (value_pred_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # entropy bonus (negative -> maximize entropy)
                entropy_loss = -entropy.mean()

                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # bail on NaN
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    break

                # gradient step with clipping
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # stats
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.clip_epsilon).float().mean().item()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(entropy_loss.item())
                all_kls.append(approx_kl)
                all_clip_fracs.append(clip_frac)

                # KL early stopping
                if approx_kl > 1.5 * target_kl:
                    early_stopped = True
                    break

        return {
            "policy_loss": float(np.mean(all_policy_losses)) if all_policy_losses else 0.0,
            "value_loss": float(np.mean(all_value_losses)) if all_value_losses else 0.0,
            "entropy_loss": float(np.mean(all_entropy_losses)) if all_entropy_losses else 0.0,
            "approx_kl": float(np.mean(all_kls)) if all_kls else 0.0,
            "clip_fraction": float(np.mean(all_clip_fracs)) if all_clip_fracs else 0.0,
        }

    def train(
        self,
        env: Any,
        total_timesteps: int,
        log_interval: int = 1,
        save_path: Optional[str] = None,
        logger: Optional[Any] = None,
    ) -> List[Dict[str, float]]:
        """Main training loop: collect rollouts -> PPO update -> repeat.

        LR is linearly annealed to 0 over training.
        """
        num_rollouts = max(1, total_timesteps // self.rollout_steps)
        all_stats: list[dict] = []
        best_reward = -float("inf")

        initial_lr = self.optimizer.param_groups[0]["lr"]

        for rollout_idx in range(num_rollouts):
            # linear LR decay
            frac = 1.0 - rollout_idx / num_rollouts
            new_lr = initial_lr * frac
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

            rollout_stats = self.collect_rollout(env)
            update_stats = self.update()

            stats = {**rollout_stats, **update_stats, "rollout": rollout_idx, "lr": new_lr}
            all_stats.append(stats)

            if log_interval and (rollout_idx + 1) % log_interval == 0:
                timestep = (rollout_idx + 1) * self.rollout_steps
                print(
                    f"[Rollout {rollout_idx + 1}/{num_rollouts}] "
                    f"steps={timestep}/{total_timesteps} "
                    f"reward={stats['mean_reward']:.2f} "
                    f"entropy={stats['mean_entropy']:.3f} "
                    f"pi_loss={stats['policy_loss']:.4f} "
                    f"v_loss={stats['value_loss']:.4f} "
                    f"kl={stats['approx_kl']:.4f} "
                    f"lr={new_lr:.2e}"
                )

            if logger is not None:
                global_step = (rollout_idx + 1) * self.rollout_steps
                logger.log_training_step(stats, global_step)

            # save best model
            if save_path and stats["mean_reward"] > best_reward:
                best_reward = stats["mean_reward"]
                self.save(save_path)

        return all_stats

    def save(self, path: str) -> None:
        """Save policy + optimizer checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load policy + optimizer from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
