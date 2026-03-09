# Visualize a trained policy with 3D PyBullet rendering.
# Opens a GUI window showing drones flying in formation.
#
# Usage:
#     python -m confidence_triggered_swarm.scripts.visualize --model-path runs/baseline/best_model.pt
#     python -m confidence_triggered_swarm.scripts.visualize --model-path runs/baseline/best_model.pt --surprise moderate

import argparse
import time

import numpy as np
import torch

from confidence_triggered_swarm.envs.formation_aviary import FormationAviary
from confidence_triggered_swarm.envs.surprise_wrapper import SurpriseWrapper, SurpriseConfig
from confidence_triggered_swarm.algorithms.ppo import PPOAgent
from confidence_triggered_swarm.configs import load_config

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


def main():
    parser = argparse.ArgumentParser(description="Visualize trained drone swarm policy")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model (.pt file)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    parser.add_argument("--surprise", type=str, default="clean",
                        choices=["clean", "mild", "moderate", "severe"],
                        help="Surprise severity (default: clean)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to visualize")
    parser.add_argument("--slow", action="store_true",
                        help="Slow down for better viewing")
    args = parser.parse_args()

    config = load_config(args.config)
    env_cfg = config["env"]

    print(f"\nVisualization | model={args.model_path} | surprise={args.surprise}")
    print(f"  episodes={args.episodes}  GUI=ON")
    print("\n  Controls: left-drag=rotate, scroll=zoom, middle-drag=pan\n")

    # create env with GUI enabled
    env = FormationAviary(
        num_drones=env_cfg.get("num_drones", 2),
        freq=env_cfg.get("freq", 240),
        ctrl_freq=env_cfg.get("ctrl_freq", 30),
        gui=True,
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

    if args.surprise != "clean":
        surprise_config = SurpriseConfig.from_severity(args.surprise)
        env = SurpriseWrapper(env, surprise_config)
        print(f"  Surprise wrapper active: {args.surprise}")

    obs_sample, _ = env.reset()
    obs_dim = obs_sample.shape[-1]
    act_dim = env.action_space.shape[-1]

    # use CPU for viz — more stable with GUI
    policy_cfg = config["policy"]
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_drones=env_cfg["num_drones"],
        hidden_dims=policy_cfg["hidden_dims"],
        activation=policy_cfg["activation"],
        mc_dropout_p=policy_cfg["mc_dropout_p"],
        device="cpu",
    )
    agent.load(args.model_path)
    agent.policy.eval()
    print(f"  Model loaded, params={sum(p.numel() for p in agent.policy.parameters()):,}")

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        print(f"\n  Episode {ep + 1}/{args.episodes}")
        print(f"  {'Step':>6} | {'Reward':>8} | {'WP':>4} | {'Form Err':>10}")
        print(f"  {'-'*40}")

        while not done:
            actions, _, _, _ = agent.select_actions(obs)
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            total_reward += reward
            step_count += 1

            if step_count % 10 == 0:
                wp_idx = info.get("current_waypoint_idx", 0)
                form_err = info.get("mean_formation_error", 0.0)
                # TODO: maybe show total_waypoints too
                n_wp = len(env.unwrapped.waypoints) if hasattr(env, "unwrapped") else "?"
                print(f"  {step_count:>6} | {total_reward:>+8.2f} | {wp_idx:>2}/{n_wp} | {form_err:>10.4f}")

            if args.slow:
                time.sleep(0.05)
            else:
                time.sleep(1.0 / env_cfg["ctrl_freq"])

        print(f"  {'-'*40}")
        print(f"  Done: {step_count} steps, reward={total_reward:.2f}, "
              f"wp={info.get('current_waypoint_idx', 0)}")

        if ep < args.episodes - 1:
            print("  Next episode in 2s...")
            time.sleep(2.0)

    env.close()
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
