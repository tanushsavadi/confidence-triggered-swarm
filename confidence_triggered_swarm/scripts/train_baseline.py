# Train IPPO baseline on clean FormationAviary (no surprises).
#
# Usage:
#     python -m confidence_triggered_swarm.scripts.train_baseline
#     python -m confidence_triggered_swarm.scripts.train_baseline --timesteps 500000
#     python -m confidence_triggered_swarm.scripts.train_baseline --seed 123 --save-dir runs/exp1

import argparse
import datetime
from pathlib import Path

import numpy as np
import torch

from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)

from confidence_triggered_swarm.algorithms.ppo import PPOAgent
from confidence_triggered_swarm.configs import load_config
from confidence_triggered_swarm.envs.formation_aviary import FormationAviary
from confidence_triggered_swarm.utils.logger import MetricsLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train IPPO baseline on clean FormationAviary"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps (overrides config)")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Enable PyBullet GUI")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory for checkpoints/logs")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    env_cfg = config["env"]
    train_cfg = config["training"]
    policy_cfg = config["policy"]

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    total_timesteps = args.timesteps if args.timesteps is not None else train_cfg["total_timesteps"]
    gui = args.gui or env_cfg.get("gui", False)

    # save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("runs") / f"baseline_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Training IPPO baseline | seed={seed} | timesteps={total_timesteps:,}")
    print(f"  drones={env_cfg['num_drones']}  rollout={train_cfg['rollout_steps']}  "
          f"batch={train_cfg['batch_size']}  epochs={train_cfg['n_epochs']}")
    print(f"  hidden={policy_cfg['hidden_dims']}  save_dir={save_dir}  gui={gui}")

    # create env — we build it directly here since training uses specific setup
    env = FormationAviary(
        num_drones=env_cfg.get("num_drones", 2),
        freq=env_cfg.get("freq", 240),
        ctrl_freq=env_cfg.get("ctrl_freq", 30),
        gui=gui,
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

    # figure out obs/act dims from the env
    obs_sample, _ = env.reset()
    obs_dim = obs_sample.shape[-1] if obs_sample.ndim > 1 else obs_sample.shape[0]
    act_dim = env.action_space.shape[-1]  # 4 for VEL mode

    print(f"  act_type={env.ACT_TYPE}  obs_dim={obs_dim}  act_dim={act_dim}")

    logger = MetricsLogger(
        log_dir=str(save_dir / "logs"),
        experiment_name=f"baseline_seed{seed}",
    )

    # build agent from scratch for training
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_drones=env_cfg["num_drones"],
        lr=train_cfg["lr"],
        gamma=train_cfg["gamma"],
        gae_lambda=train_cfg["gae_lambda"],
        clip_epsilon=train_cfg["clip_epsilon"],
        entropy_coef=train_cfg["entropy_coef"],
        value_coef=train_cfg["value_coef"],
        max_grad_norm=train_cfg["max_grad_norm"],
        n_epochs=train_cfg["n_epochs"],
        batch_size=train_cfg["batch_size"],
        rollout_steps=train_cfg["rollout_steps"],
        hidden_dims=policy_cfg["hidden_dims"],
        activation=policy_cfg["activation"],
        mc_dropout_p=policy_cfg["mc_dropout_p"],
        device=train_cfg.get("device", "auto"),
    )

    print(f"  device={agent.device}  params={sum(p.numel() for p in agent.policy.parameters()):,}")

    # train
    best_model_path = str(save_dir / "best_model.pt")
    all_stats = agent.train(
        env=env,
        total_timesteps=total_timesteps,
        log_interval=1,
        save_path=best_model_path,
        logger=logger,
    )

    # save final checkpoint
    final_path = str(save_dir / f"final_model_seed{seed}.pt")
    agent.save(final_path)
    logger.save_csv("training_metrics.csv")

    # quick summary
    if all_stats:
        rewards = [s["mean_reward"] for s in all_stats]
        print(f"\nDone! {len(all_stats)} rollouts, best={max(rewards):.2f}, "
              f"final={rewards[-1]:.2f}, mean={np.mean(rewards):.2f}")
    print(f"  best_model: {best_model_path}")
    print(f"  final_model: {final_path}")

    logger.close()
    env.close()


if __name__ == "__main__":
    main()
