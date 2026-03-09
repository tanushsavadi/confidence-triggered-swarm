#!/usr/bin/env python3
# Quick smoke test for FormationAviary and SurpriseWrapper.
# Just makes sure everything can be created, reset, and stepped without crashing.
#
# Usage:
#     python -m confidence_triggered_swarm.scripts.test_env
#     python -m confidence_triggered_swarm.scripts.test_env --gui --episodes 5

from __future__ import annotations

import argparse
import time

import numpy as np

from gym_pybullet_drones.utils.enums import ActionType

from confidence_triggered_swarm.envs.formation_aviary import FormationAviary
from confidence_triggered_swarm.envs.surprise_wrapper import (
    SurpriseConfig,
    SurpriseWrapper,
)


def _run_episode(env, max_steps: int = 2000):
    """Run one episode with random actions, return per-step rewards."""
    obs, info = env.reset()
    rewards = []
    for t in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    return rewards, info


def test_formation_aviary(num_episodes: int = 3, gui: bool = False) -> None:
    """Basic FormationAviary test with VEL action mode."""
    print("\n--- FormationAviary (VEL mode) ---")

    env = FormationAviary(
        num_drones=2,
        gui=gui,
        episode_len_sec=15.0,
        init_height=0.5,
        speed_limit=0.5,
        tilt_threshold=1.0,
        z_min=0.02,
        waypoint_threshold=0.2,
    )

    # sanity checks
    assert env.ACT_TYPE == ActionType.VEL, f"Expected VEL, got {env.ACT_TYPE}"
    print(f"  act_type={env.ACT_TYPE}  speed_limit={env.SPEED_LIMIT}")
    print(f"  obs_space={env.observation_space}  act_space={env.action_space}")

    obs, info = env.reset()
    obs_dim = obs.shape[-1] if obs.ndim > 1 else obs.shape[0]
    act_dim = env.action_space.shape[-1]
    print(f"  obs shape={obs.shape}  obs_dim={obs_dim}  act_dim={act_dim}")
    assert act_dim == 4, f"Expected act_dim=4 for VEL, got {act_dim}"

    all_ep_rewards = []
    for ep in range(num_episodes):
        t0 = time.time()
        rewards, info = _run_episode(env)
        elapsed = time.time() - t0
        total_r = sum(rewards)
        all_ep_rewards.append(total_r)
        print(f"  ep {ep+1}/{num_episodes}: steps={len(rewards):4d}  "
              f"reward={total_r:+8.2f}  "
              f"wp={info.get('current_waypoint_idx', '?')}/{info.get('total_waypoints', '?')}  "
              f"time={elapsed:.2f}s")

    env.close()
    print(f"  mean reward: {float(np.mean(all_ep_rewards)):+.2f}")
    print("  OK!")


def test_surprise_wrapper(severity: str = "mild", gui: bool = False) -> None:
    """Test SurpriseWrapper with a given severity preset."""
    print(f"\n--- SurpriseWrapper (severity='{severity}') ---")

    base_env = FormationAviary(
        num_drones=2, gui=gui, episode_len_sec=15.0,
        init_height=0.5, speed_limit=0.5, tilt_threshold=1.0,
        z_min=0.02, waypoint_threshold=0.2,
    )
    config = SurpriseConfig.from_severity(severity)
    env = SurpriseWrapper(base_env, config=config, seed=42)

    print(f"  obs_space={env.observation_space}  act_space={env.action_space}")
    print(f"  config={config}")

    obs, info = env.reset()
    print(f"  obs shape={obs.shape}  surprise_active={info.get('surprise_active')}")

    t0 = time.time()
    rewards, info = _run_episode(env)
    elapsed = time.time() - t0
    print(f"  steps={len(rewards):4d}  reward={sum(rewards):+8.2f}  time={elapsed:.2f}s")
    if info.get("wind_applied") is not None:
        print(f"  last wind: {info['wind_applied']}")

    env.close()
    print("  OK!")


def test_severity_presets(gui: bool = False) -> None:
    """Check all severity presets can be constructed and stepped."""
    print("\n--- SurpriseConfig presets ---")
    for level in ("clean", "mild", "moderate", "severe"):
        cfg = SurpriseConfig.from_severity(level)
        print(f"  {level:>10s}: wind={cfg.wind_enabled}, noise={cfg.sensor_noise_std}, "
              f"actuator={cfg.actuator_weakness}, goal_shift={cfg.goal_shift_prob}")

        base = FormationAviary(num_drones=2, gui=gui)
        wrapped = SurpriseWrapper(base, config=cfg, seed=0)
        obs, info = wrapped.reset()
        obs, reward, terminated, truncated, info = wrapped.step(wrapped.action_space.sample())
        wrapped.close()
        print(f"             step OK, obs shape={obs.shape}, reward={reward:.2f}")
    print("  All presets valid!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test env and surprise wrapper")
    parser.add_argument("--gui", action="store_true", help="Open PyBullet GUI")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Random-action episodes for base env test")
    args = parser.parse_args()

    print("\nEnvironment Smoke Test")
    print("=" * 50)

    test_formation_aviary(num_episodes=args.episodes, gui=args.gui)
    test_severity_presets(gui=args.gui)
    test_surprise_wrapper(severity="mild", gui=args.gui)

    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == "__main__":
    main()
