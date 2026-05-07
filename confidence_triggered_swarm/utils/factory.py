"""Shared helpers for creating envs, agents, and running eval episodes.

This was copy-pasted across like 5 files so I pulled it out.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)

from confidence_triggered_swarm.algorithms.ppo import PPOAgent
from confidence_triggered_swarm.envs.formation_aviary import FormationAviary
from confidence_triggered_swarm.envs.surprise_wrapper import (
    DomainRandomizationWrapper,
    SurpriseConfig,
    SurpriseWrapper,
)
from confidence_triggered_swarm.utils.seeding import reset_env


def create_env(
    config: dict,
    severity: str = "clean",
    gui: bool = False,
    surprise_config_dict: Optional[dict] = None,
    seed: int | None = None,
    domain_randomized: bool = False,
) -> Any:
    """Create a FormationAviary, optionally wrapped with SurpriseWrapper.

    If severity != 'clean', wraps with SurpriseConfig.from_severity().
    If surprise_config_dict is given, uses SurpriseConfig.from_config_dict() instead.
    """
    env_cfg = config.get("env", {})

    env = FormationAviary(
        num_drones=env_cfg.get("num_drones", 2),
        gui=gui,
        freq=env_cfg.get("freq", 240),
        ctrl_freq=env_cfg.get("ctrl_freq", 30),
        episode_len_sec=env_cfg.get("episode_len_sec", 15.0),
        init_height=env_cfg.get("init_height", 0.5),
        speed_limit=env_cfg.get("speed_limit", 0.5),
        tilt_threshold=env_cfg.get("tilt_threshold", 1.0),
        z_min=env_cfg.get("z_min", 0.02),
        waypoint_threshold=env_cfg.get("waypoint_threshold", 0.2),
        drone_model=DroneModel[env_cfg.get("drone_model", "CF2X")],
        physics=Physics[env_cfg.get("physics", "PYB")],
        obs=ObservationType[env_cfg.get("obs_type", "KIN")],
        act=ActionType[env_cfg.get("action_type", "VEL")],
    )

    # wrap with surprises if needed
    if domain_randomized or severity == "domain_randomized":
        ranges = config.get("domain_randomization", {})
        env = DomainRandomizationWrapper(env, ranges=ranges, seed=seed)
    elif surprise_config_dict is not None:
        sc = SurpriseConfig.from_config_dict(surprise_config_dict)
        env = SurpriseWrapper(env, sc, seed=seed)
    elif severity != "clean":
        env = SurpriseWrapper(env, SurpriseConfig.from_severity(severity), seed=seed)

    return env


def create_agent(
    config: dict,
    obs_dim: int,
    act_dim: int,
    device: str = "cpu",
) -> PPOAgent:
    """Spin up a PPOAgent from a config dict. Weights are NOT loaded."""
    train_cfg = config.get("training", {})
    policy_cfg = config.get("policy", {})
    env_cfg = config.get("env", {})

    return PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_drones=env_cfg.get("num_drones", 2),
        lr=train_cfg.get("lr", 3e-4),
        gamma=train_cfg.get("gamma", 0.99),
        gae_lambda=train_cfg.get("gae_lambda", 0.95),
        clip_epsilon=train_cfg.get("clip_epsilon", 0.2),
        entropy_coef=train_cfg.get("entropy_coef", 0.01),
        value_coef=train_cfg.get("value_coef", 0.5),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
        n_epochs=train_cfg.get("n_epochs", 10),
        batch_size=train_cfg.get("batch_size", 64),
        rollout_steps=train_cfg.get("rollout_steps", 2048),
        hidden_dims=policy_cfg.get("hidden_dims", [256, 256]),
        activation=policy_cfg.get("activation", "tanh"),
        mc_dropout_p=policy_cfg.get("mc_dropout_p", 0.1),
        device=device,
    )


def load_agent(
    model_path: str,
    config: dict,
    obs_dim: int,
    act_dim: int,
    device: str = "cpu",
) -> PPOAgent:
    """Create agent and load checkpoint weights."""
    agent = create_agent(config, obs_dim, act_dim, device)
    agent.load(model_path)
    return agent


def run_frozen_episodes(
    agent: PPOAgent,
    env: Any,
    n_episodes: int,
    device: str = "cpu",
    label: str = "",
    deterministic: bool = False,
    base_seed: int | None = None,
    seed_stream: int = 0,
) -> Dict[str, Any]:
    """Run episodes with frozen policy — no gradient updates.

    Returns dict with mean_reward, std_reward, episode lengths,
    waypoints reached, success rate, and raw reward list.
    """
    agent.policy.eval()
    all_rewards: List[float] = []
    all_lengths: List[int] = []
    all_waypoints: List[int] = []

    for ep in range(n_episodes):
        obs, info = reset_env(env, base_seed, ep, seed_stream)
        done = False
        ep_reward = 0.0
        step_count = 0

        while not done:
            actions, _, _, _ = agent.select_actions(
                obs, deterministic=deterministic
            )
            obs, reward, terminated, truncated, info = env.step(actions)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            step_count += 1

        all_rewards.append(ep_reward)
        all_lengths.append(step_count)
        all_waypoints.append(info.get("current_waypoint_idx", 0))

        if label and (ep + 1) % 10 == 0:
            recent_reward = float(np.mean(all_rewards[-10:]))
            print(
                f"  [{label}] Episode {ep + 1}/{n_episodes}: "
                f"avg_reward(last10)={recent_reward:.2f}"
            )

    total_waypoints = info.get("total_waypoints", 5) if info else 5
    success_count = sum(1 for w in all_waypoints if w >= total_waypoints - 1)

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_episode_length": float(np.mean(all_lengths)),
        "mean_waypoints_reached": float(np.mean(all_waypoints)),
        "success_rate": float(success_count / max(n_episodes, 1)),
        "all_rewards": [float(r) for r in all_rewards],
        "all_waypoints": [int(w) for w in all_waypoints],
        "n_episodes": n_episodes,
    }
