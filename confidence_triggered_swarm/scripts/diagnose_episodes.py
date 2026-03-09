#!/usr/bin/env python3
# Diagnose short episodes in the trained baseline.
# Loads model, runs episodes, logs per-step state to figure out
# which truncation condition fires and why episodes end early.
#
# Usage:
#     python -m confidence_triggered_swarm.scripts.diagnose_episodes
#     python -m confidence_triggered_swarm.scripts.diagnose_episodes --model runs/baseline/best_model.pt --episodes 20

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from confidence_triggered_swarm.configs import load_config
from confidence_triggered_swarm.envs.formation_aviary import FormationAviary
from confidence_triggered_swarm.utils.factory import create_env, load_agent


def diagnose_truncation(env: FormationAviary) -> str:
    """Check which truncation condition would fire right now."""
    for i in range(env.NUM_DRONES):
        state = env._getDroneStateVector(i)
        x, y, z = state[0], state[1], state[2]
        roll, pitch = state[7], state[8]

        if z < env.z_min:
            return f"z_min (drone {i}: z={z:.4f} < {env.z_min})"
        if z > 3.0:
            return f"z_max (drone {i}: z={z:.4f} > 3.0)"
        if abs(x) > 3.0:
            return f"x_oob (drone {i}: x={x:.4f})"
        if abs(y) > 3.0:
            return f"y_oob (drone {i}: y={y:.4f})"
        if abs(roll) > env.tilt_threshold:
            return f"roll (drone {i}: roll={roll:.4f} > {env.tilt_threshold})"
        if abs(pitch) > env.tilt_threshold:
            return f"pitch (drone {i}: pitch={pitch:.4f} > {env.tilt_threshold})"

    if env.step_counter / env.PYB_FREQ > env.EPISODE_LEN_SEC:
        return "timeout"

    return "none"


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose episode quality")
    parser.add_argument("--model", type=str, default="runs/baseline/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of diagnostic episodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    # use factory to create env and agent
    env = create_env(config, severity="clean", gui=False)

    obs_sample, _ = env.reset()
    obs_dim = obs_sample.shape[-1] if obs_sample.ndim > 1 else obs_sample.shape[0]
    act_dim = env.action_space.shape[-1]

    agent = load_agent(args.model, config, obs_dim, act_dim)
    agent.policy.eval()

    # need the unwrapped env for truncation diagnosis
    # (create_env returns FormationAviary directly for severity="clean")
    raw_env = env

    print(f"Episode diagnostic | model={args.model}")
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  drones={raw_env.NUM_DRONES}")
    print(f"  z_min={raw_env.z_min}  tilt_thresh={raw_env.tilt_threshold}")
    print(f"  budget={raw_env.EPISODE_LEN_SEC}s = {int(raw_env.EPISODE_LEN_SEC * raw_env.CTRL_FREQ)} steps")
    print(f"  waypoints={raw_env.waypoints.tolist()}")

    n_episodes = args.episodes
    all_lengths = []
    all_rewards = []
    all_waypoints = []
    all_reasons = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        max_wp = 0

        print(f"\n--- Episode {ep + 1}/{n_episodes} ---")

        while not done:
            actions, _, _, _ = agent.select_actions(obs)
            obs, reward, terminated, truncated, info = env.step(actions)
            done = bool(terminated or truncated)
            ep_reward += float(reward)
            step += 1
            wp_idx = info.get("current_waypoint_idx", 0)
            max_wp = max(max_wp, wp_idx)

            # verbose logging: every step for first 3 eps, every 10 after
            log_this = (ep < 3) or (step % 10 == 0) or done
            if log_this:
                positions = []
                orientations = []
                for i in range(raw_env.NUM_DRONES):
                    state = raw_env._getDroneStateVector(i)
                    positions.append((state[0], state[1], state[2]))
                    orientations.append((state[7], state[8], state[9]))

                pos_str = "  ".join(
                    [f"d{i}=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})" for i, p in enumerate(positions)]
                )
                rpy_str = "  ".join(
                    [f"d{i}=({o[0]:+.3f},{o[1]:+.3f},{o[2]:+.3f})" for i, o in enumerate(orientations)]
                )

                trunc_reason = diagnose_truncation(raw_env)
                print(f"  step={step:4d}  rew={reward:+7.2f}  wp={wp_idx}  "
                      f"pos=[{pos_str}]  rpy=[{rpy_str}]  "
                      f"term={terminated}  trunc={truncated}  reason={trunc_reason}")

        reason = diagnose_truncation(raw_env)
        if terminated:
            reason = "all_waypoints_reached"

        all_lengths.append(step)
        all_rewards.append(ep_reward)
        all_waypoints.append(max_wp)
        all_reasons.append(reason)

        print(f"  >> END: steps={step}, reward={ep_reward:.2f}, wp={max_wp}, reason={reason}")
        for i in range(raw_env.NUM_DRONES):
            state = raw_env._getDroneStateVector(i)
            print(f"     Drone {i}: pos=({state[0]:.4f}, {state[1]:.4f}, {state[2]:.4f}) "
                  f"rpy=({state[7]:.4f}, {state[8]:.4f}, {state[9]:.4f}) "
                  f"vel=({state[10]:.4f}, {state[11]:.4f}, {state[12]:.4f})")

    env.close()

    # summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  length: mean={np.mean(all_lengths):.1f} ({np.mean(all_lengths)/raw_env.CTRL_FREQ:.2f}s) "
          f"std={np.std(all_lengths):.1f} min={np.min(all_lengths)} max={np.max(all_lengths)}")
    print(f"  reward: mean={np.mean(all_rewards):.2f} std={np.std(all_rewards):.2f} "
          f"per_step={np.mean(all_rewards)/max(np.mean(all_lengths), 1):.2f}")
    print(f"  waypoints: mean={np.mean(all_waypoints):.2f} max={np.max(all_waypoints)}")

    print(f"\n  Truncation reasons:")
    for reason, count in Counter(all_reasons).most_common():
        print(f"    {reason}: {count}/{n_episodes} ({count/n_episodes*100:.0f}%)")


if __name__ == "__main__":
    main()
