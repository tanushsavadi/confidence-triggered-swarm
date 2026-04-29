"""Lifelong Trainer — orchestrates confidence-triggered adaptation.

The loop:
1. Run episode with current policy in surprise env
2. Monitor confidence throughout
3. After episode, if confidence was low:
   a. Collect adaptation data from the episode
   b. Reward-weighted regression + KL anchoring + EWC
   c. Mix in clean replay to prevent forgetting
4. Repeat

This is *between-episode* adaptation — no mid-episode updates.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from confidence_triggered_swarm.algorithms.ppo import PPOAgent
from confidence_triggered_swarm.adaptation.confidence import ConfidenceMonitor
from confidence_triggered_swarm.adaptation.ewc import EWCRegularizer


class LifelongTrainer:
    """Orchestrates confidence-triggered between-episode adaptation.

    Runs episodes, monitors confidence, and adapts the policy when
    confidence drops. Uses reward-weighted regression, KL anchoring
    toward the clean policy, EWC penalty, and clean replay mixing.
    """

    def __init__(
        self,
        agent: PPOAgent,
        confidence_monitor: ConfidenceMonitor,
        ewc_regularizer: EWCRegularizer,
        adapt_epochs: int = 5,
        adapt_lr: float = 1e-4,
        adapt_batch_size: int = 32,
        replay_buffer_size: int = 10000,
        min_episode_steps: int = 30,
        min_adapt_reward: float = -5.0,
        clean_replay_ratio: float = 0.2,
        kl_anchor_coef: float = 0.5,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        self.agent = agent
        self.monitor = confidence_monitor
        self.ewc = ewc_regularizer
        self.adapt_epochs = adapt_epochs
        self.adapt_lr = adapt_lr
        self.adapt_batch_size = adapt_batch_size
        self.replay_buffer_size = replay_buffer_size
        self.device = device
        self.config = config or {}

        # read adaptation sub-config (with fallbacks to constructor args)
        adapt_config = self.config.get('adaptation', {})
        self.min_episode_steps = adapt_config.get('min_episode_steps', min_episode_steps)
        self.min_adapt_reward = adapt_config.get('min_adapt_reward', min_adapt_reward)
        self.clean_replay_ratio = adapt_config.get('clean_replay_ratio', clean_replay_ratio)
        self.kl_anchor_coef = adapt_config.get('kl_anchor_coef', kl_anchor_coef)

        # separate optimizer for adaptation (not the main PPO one)
        self.adapt_optimizer = optim.Adam(
            self.agent.policy.parameters(), lr=adapt_lr
        )

        # replay buffer for surprise experience
        self.replay_buffer: Dict[str, List[np.ndarray]] = {
            "observations": [],
            "actions": [],
            "rewards": [],
        }

        # clean replay buffer (filled during setup)
        self.clean_replay: Dict[str, List[np.ndarray]] = {
            "observations": [],
            "actions": [],
            "rewards": [],
        }

        # frozen copy of clean policy for KL anchoring
        self.clean_policy_state: Optional[Dict[str, torch.Tensor]] = None
        self.clean_policy_copy: Optional[nn.Module] = None

        # counters
        self.total_adaptations: int = 0
        self.adaptation_history: List[Dict[str, Any]] = []

    # -- setup (call after baseline training, before deployment) --

    def setup(self, clean_env: Any, n_calibration_episodes: int = 5) -> None:
        """Initialize the lifelong system.

        1. Calibrate confidence monitor on clean env
        2. EWC snapshot of clean policy
        3. Compute Fisher from clean rollouts
        4. Store frozen clean policy for KL anchoring
        5. Populate clean replay buffer
        """
        print("Calibrating confidence monitor...")
        self.monitor.calibrate(clean_env, n_calibration_episodes)

        print("Taking EWC snapshot of clean policy...")
        self.ewc.snapshot()

        # store clean policy for KL anchoring
        self.clean_policy_state = {
            k: v.clone() for k, v in self.agent.policy.state_dict().items()
        }

        # deep copy for KL — avoids autograd version counter issues
        self.clean_policy_copy = copy.deepcopy(self.agent.policy)
        self.clean_policy_copy.eval()
        for p in self.clean_policy_copy.parameters():
            p.requires_grad_(False)

        print("Computing Fisher information...")
        # collect clean rollouts for Fisher + clean replay
        all_obs: List[np.ndarray] = []
        all_acts: List[np.ndarray] = []
        all_rewards: List[float] = []

        for _ep in range(n_calibration_episodes):
            obs, _info = clean_env.reset()
            done = False
            while not done:
                actions, _log_probs, _values, _ents = self.agent.select_actions(obs)
                for i in range(obs.shape[0]):
                    all_obs.append(obs[i].copy())
                    all_acts.append(actions[i].copy())
                next_obs, reward, terminated, truncated, _info = clean_env.step(actions)
                for i in range(obs.shape[0]):
                    all_rewards.append(float(reward) / obs.shape[0])
                done = bool(terminated or truncated)
                obs = next_obs

        if all_obs:
            obs_tensor = torch.as_tensor(
                np.array(all_obs), dtype=torch.float32
            ).to(self.device)
            act_tensor = torch.as_tensor(
                np.array(all_acts), dtype=torch.float32
            ).to(self.device)
            self.ewc.compute_fisher(
                obs_tensor, act_tensor, n_samples=min(2000, len(all_obs))
            )

            # fill clean replay (cap at half the replay buffer)
            max_clean = self.replay_buffer_size // 2
            n_to_store = min(len(all_obs), max_clean)
            for i in range(n_to_store):
                self.clean_replay["observations"].append(all_obs[i])
                self.clean_replay["actions"].append(all_acts[i])
                if i < len(all_rewards):
                    self.clean_replay["rewards"].append(all_rewards[i])

            print(
                f"Lifelong system initialized. "
                f"Fisher from {len(all_obs)} transitions. "
                f"Clean replay: {len(self.clean_replay['observations'])} transitions. "
                f"KL anchor stored."
            )
        else:
            print(
                "WARNING: No transitions collected during setup. "
                "Fisher not computed."
            )

    # -- episode execution --

    def run_episode(self, env: Any) -> Dict[str, Any]:
        """Run one episode with confidence monitoring + potential adaptation."""
        obs, info = env.reset()
        done = False

        episode_data: Dict[str, List[Any]] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "confidences": [],
            "entropies": [],
        }
        total_reward = 0.0
        step_count = 0

        # NOTE: we don't reset the confidence monitor between episodes.
        # confidence accumulates so the rolling window fills even with
        # short episodes (~6-12 steps).

        while not done:
            conf_result = self.monitor.update(obs)

            actions, _log_probs, _values, _entropies = self.agent.select_actions(obs)

            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = bool(terminated or truncated)

            # store per-drone transitions
            for i in range(obs.shape[0]):
                episode_data["observations"].append(obs[i].copy())
                episode_data["actions"].append(actions[i].copy())
                episode_data["rewards"].append(float(reward) / obs.shape[0])

            episode_data["confidences"].append(conf_result["confidence"])
            episode_data["entropies"].append(conf_result["entropy"])

            total_reward += float(reward)
            step_count += 1
            obs = next_obs

        # -- episode quality gate --
        mean_ep_reward = total_reward / max(step_count, 1)
        episode_is_useful = (
            step_count >= self.min_episode_steps
            and mean_ep_reward >= self.min_adapt_reward
        )

        stats: Dict[str, Any] = {}

        if episode_is_useful:
            self._add_to_replay(episode_data)
        else:
            # don't add crash/junk data to replay
            stats['rejected_episodes'] = stats.get('rejected_episodes', 0) + 1

        # check adaptation trigger — rolling window OR this episode's avg
        adapted = False
        adapt_stats: Optional[Dict[str, Any]] = None
        ep_avg_conf = (
            float(np.mean(episode_data["confidences"]))
            if episode_data["confidences"]
            else 1.0
        )
        should_adapt = self.monitor.should_adapt() or (
            ep_avg_conf < self.monitor.confidence_threshold
            and len(self.replay_buffer["observations"]) >= self.adapt_batch_size
        )

        if should_adapt and episode_is_useful:
            adapt_stats = self._adapt()
            adapted = not bool(adapt_stats.get("skipped", False))
        elif should_adapt and not episode_is_useful:
            adapt_stats = {
                "skipped": True,
                "reason": "episode_too_short_or_low_reward",
                "episode_steps": step_count,
                "mean_reward": mean_ep_reward,
            }

        stats.update({
            "total_reward": total_reward,
            "episode_length": step_count,
            "waypoints_reached": info.get("current_waypoint_idx", 0),
            "avg_confidence": (
                float(np.mean(episode_data["confidences"]))
                if episode_data["confidences"]
                else 1.0
            ),
            "min_confidence": (
                float(min(episode_data["confidences"]))
                if episode_data["confidences"]
                else 1.0
            ),
            "avg_entropy": (
                float(np.mean(episode_data["entropies"]))
                if episode_data["entropies"]
                else 0.0
            ),
            "adapted": adapted,
            "adapt_stats": adapt_stats,
            "per_step_confidence": episode_data["confidences"],
        })

        return stats

    # -- replay buffer management --

    def _add_to_replay(self, episode_data: Dict[str, List[Any]]) -> None:
        """Add episode transitions to replay, trimming oldest if full."""
        for key in ("observations", "actions", "rewards"):
            self.replay_buffer[key].extend(episode_data[key])

        # trim to size limit
        current_size = len(self.replay_buffer["observations"])
        if current_size > self.replay_buffer_size:
            overflow = current_size - self.replay_buffer_size
            for key in ("observations", "actions", "rewards"):
                self.replay_buffer[key] = self.replay_buffer[key][overflow:]

    # -- adaptation --

    def _adapt(self) -> Dict[str, Any]:
        """Reward-weighted adaptation with KL anchoring + clean replay.

        1. Reward-weighted regression (normalize rewards, weight log_probs)
        2. Clean replay mixing (clean_replay_ratio fraction per batch)
        3. KL anchoring toward clean policy
        4. EWC penalty
        """
        if len(self.replay_buffer["observations"]) < self.adapt_batch_size:
            return {"skipped": True, "reason": "insufficient_data"}

        self.total_adaptations += 1
        self.monitor.adaptation_count += 1

        obs_array = np.array(self.replay_buffer["observations"])
        act_array = np.array(self.replay_buffer["actions"])
        rew_array = np.array(self.replay_buffer["rewards"])

        # reward weights via softmax normalization
        rew_tensor = torch.FloatTensor(rew_array).to(self.device)
        if rew_tensor.std() > 1e-8:
            rew_weights = (rew_tensor - rew_tensor.mean()) / (rew_tensor.std() + 1e-8)
            rew_weights = torch.clamp(rew_weights, -2.0, 2.0)
            rew_weights = torch.softmax(rew_weights, dim=0) * len(rew_weights)
        else:
            rew_weights = torch.ones_like(rew_tensor)

        obs_tensor = torch.FloatTensor(obs_array).to(self.device)
        act_tensor = torch.FloatTensor(act_array).to(self.device)

        # prep clean replay data
        clean_obs: Optional[torch.Tensor] = None
        clean_act: Optional[torch.Tensor] = None
        if len(self.clean_replay['observations']) > 0:
            clean_obs = torch.FloatTensor(
                np.array(self.clean_replay['observations'])
            ).to(self.device)
            clean_act = torch.FloatTensor(
                np.array(self.clean_replay['actions'])
            ).to(self.device)

        self.agent.policy.train()

        total_policy_loss = 0.0
        total_ewc_loss = 0.0
        total_kl_loss = 0.0
        total_entropy_loss = 0.0
        n_updates = 0

        for _epoch in range(self.adapt_epochs):
            indices = torch.randperm(len(obs_tensor))
            for start in range(0, len(indices), self.adapt_batch_size):
                end = min(start + self.adapt_batch_size, len(indices))
                batch_idx = indices[start:end]

                batch_obs = obs_tensor[batch_idx]
                batch_act = act_tensor[batch_idx]
                batch_w = rew_weights[batch_idx]

                # mix in clean replay
                if clean_obs is not None and len(clean_obs) > 0:
                    n_clean = max(1, int(len(batch_idx) * self.clean_replay_ratio))
                    clean_idx = torch.randint(0, len(clean_obs), (n_clean,))
                    batch_obs = torch.cat([batch_obs, clean_obs[clean_idx]])
                    batch_act = torch.cat([batch_act, clean_act[clean_idx]])
                    # clean replay gets weight 1.0 (neutral)
                    batch_w = torch.cat([batch_w, torch.ones(n_clean, device=self.device)])

                log_probs, _values, entropy = self.agent.policy.evaluate_actions(
                    batch_obs, batch_act
                )

                # reward-weighted policy loss
                policy_loss = -(batch_w * log_probs).mean()
                entropy_loss = -0.01 * entropy.mean()

                ewc_loss = self.ewc.penalty()
                kl_loss = self._compute_kl_anchor(batch_obs)

                total_loss = policy_loss + entropy_loss + ewc_loss + self.kl_anchor_coef * kl_loss

                self.adapt_optimizer.zero_grad()
                total_loss.backward()
                # clip so we don't blow up gradients
                torch.nn.utils.clip_grad_norm_(
                    self.agent.policy.parameters(), 0.5
                )
                self.adapt_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_ewc_loss += ewc_loss.item()
                total_kl_loss += kl_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1

        adapt_stats: Dict[str, Any] = {
            "skipped": False,
            "n_updates": n_updates,
            "mean_policy_loss": total_policy_loss / max(n_updates, 1),
            "mean_ewc_loss": total_ewc_loss / max(n_updates, 1),
            "mean_kl_loss": total_kl_loss / max(n_updates, 1),
            "mean_entropy_loss": total_entropy_loss / max(n_updates, 1),
            "replay_buffer_size": len(self.replay_buffer["observations"]),
            "clean_replay_size": len(self.clean_replay["observations"]),
            "adaptation_number": self.total_adaptations,
        }

        self.adaptation_history.append(adapt_stats)
        print(
            f"  [Adaptation #{self.total_adaptations}] "
            f"updates={n_updates}, "
            f"pi_loss={adapt_stats['mean_policy_loss']:.4f}, "
            f"ewc_loss={adapt_stats['mean_ewc_loss']:.4f}, "
            f"kl_loss={adapt_stats['mean_kl_loss']:.4f}"
        )

        return adapt_stats

    # -- KL anchoring --

    def _compute_kl_anchor(self, obs: torch.Tensor) -> torch.Tensor:
        """KL divergence between current and clean policy.

        Uses a frozen deep-copy to avoid autograd version counter issues.
        """
        if self.clean_policy_copy is None:
            return torch.tensor(0.0, device=self.device)

        # current distribution (needs gradients)
        dist_current, _ = self.agent.policy(obs)

        # clean distribution from frozen copy
        with torch.no_grad():
            dist_clean, _ = self.clean_policy_copy(obs)

        kl = torch.distributions.kl_divergence(dist_current, dist_clean).mean()
        return kl

    # -- multi-episode evaluation --

    def run_evaluation(
        self,
        env: Any,
        n_episodes: int = 50,
        label: str = "",
    ) -> Dict[str, Any]:
        """Run multiple episodes, aggregate stats."""
        all_stats: List[Dict[str, Any]] = []
        total_rejected = 0
        for ep in range(n_episodes):
            stats = self.run_episode(env)
            all_stats.append(stats)
            total_rejected += stats.get('rejected_episodes', 0)

            if (ep + 1) % 10 == 0:
                recent = all_stats[-10:]
                avg_reward = float(np.mean([s["total_reward"] for s in recent]))
                avg_conf = float(np.mean([s["avg_confidence"] for s in recent]))
                n_adapted = sum(1 for s in recent if s["adapted"])
                print(
                    f"  [{label}] Episode {ep + 1}/{n_episodes}: "
                    f"avg_reward={avg_reward:.2f}, "
                    f"avg_conf={avg_conf:.3f}, "
                    f"adaptations={n_adapted}/10"
                )

        rewards = [s["total_reward"] for s in all_stats]
        confidences = [s["avg_confidence"] for s in all_stats]
        waypoints = [s["waypoints_reached"] for s in all_stats]
        lengths = [s["episode_length"] for s in all_stats]
        n_adapted_total = sum(1 for s in all_stats if s["adapted"])

        return {
            "label": label,
            "n_episodes": n_episodes,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_confidence": float(np.mean(confidences)),
            "mean_waypoints_reached": float(np.mean(waypoints)),
            "mean_episode_length": float(np.mean(lengths)),
            "total_adaptations": n_adapted_total,
            "adaptation_rate": n_adapted_total / max(n_episodes, 1),
            "rejected_episodes": total_rejected,
            "all_rewards": rewards,
            "all_confidences": confidences,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Current lifelong training stats."""
        return {
            "adaptation_count": self.total_adaptations,
            "total_adaptations": self.total_adaptations,
            "replay_buffer_size": len(self.replay_buffer["observations"]),
            "clean_replay_size": len(self.clean_replay["observations"]),
            "confidence_stats": self.monitor.get_stats(),
            "adaptation_history": self.adaptation_history,
        }
