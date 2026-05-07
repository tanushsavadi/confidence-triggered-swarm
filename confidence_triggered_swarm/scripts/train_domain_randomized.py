"""Train an IPPO robust baseline with per-episode domain randomization."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from confidence_triggered_swarm.algorithms.ppo import PPOAgent
from confidence_triggered_swarm.configs import load_config
from confidence_triggered_swarm.utils.factory import create_env
from confidence_triggered_swarm.utils.logger import MetricsLogger
from confidence_triggered_swarm.utils.seeding import reset_env, set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train IPPO with per-episode domain randomization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="confidence_triggered_swarm/configs/domain_randomized.yaml",
        help="Path to domain-randomization config override.",
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    env_cfg = config["env"]
    train_cfg = config["training"]
    policy_cfg = config["policy"]

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    total_timesteps = (
        args.timesteps if args.timesteps is not None else train_cfg["total_timesteps"]
    )
    set_global_seeds(seed)

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("runs") / f"domain_randomized_seed{seed}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    env = create_env(
        config,
        severity="domain_randomized",
        gui=args.gui or env_cfg.get("gui", False),
        seed=seed,
        domain_randomized=True,
    )
    obs_sample, _ = reset_env(env, seed, 0, 1)
    obs_dim = obs_sample.shape[-1] if obs_sample.ndim > 1 else obs_sample.shape[0]
    act_dim = env.action_space.shape[-1]

    print(
        f"Training domain-randomized IPPO | seed={seed} | "
        f"timesteps={total_timesteps:,} | save_dir={save_dir}"
    )
    print(f"  obs_dim={obs_dim} act_dim={act_dim}")

    logger = MetricsLogger(
        log_dir=str(save_dir / "logs"),
        experiment_name=f"domain_randomized_seed{seed}",
    )

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

    best_model_path = str(save_dir / "best_model.pt")
    all_stats = agent.train(
        env=env,
        total_timesteps=total_timesteps,
        log_interval=1,
        save_path=best_model_path,
        logger=logger,
    )
    final_path = str(save_dir / f"final_model_seed{seed}.pt")
    agent.save(final_path)
    logger.save_csv("training_metrics.csv")

    metadata = {
        "seed": seed,
        "timesteps": total_timesteps,
        "best_model": best_model_path,
        "final_model": final_path,
        "config_path": args.config,
        "config": config,
        "rollout_count": len(all_stats),
        "best_rollout_reward": max((s["mean_reward"] for s in all_stats), default=None),
        "final_rollout_reward": all_stats[-1]["mean_reward"] if all_stats else None,
        "mean_rollout_reward": float(np.mean([s["mean_reward"] for s in all_stats]))
        if all_stats
        else None,
    }
    (save_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"\nDone. best_model={best_model_path}")
    print(f"      final_model={final_path}")
    logger.close()
    env.close()


if __name__ == "__main__":
    main()
