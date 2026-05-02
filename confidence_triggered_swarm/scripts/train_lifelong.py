# Train baseline then evaluate with lifelong adaptation under surprise.
#
# Steps: train (or load) baseline -> eval frozen on each severity ->
#         eval with lifelong adaptation -> compare & plot
#
# For sequential continual learning (clean→…→severe) with retroactive eval on
# all severities after each phase, see `train_continual.py`.
#
# Usage:
#     python -m confidence_triggered_swarm.scripts.train_lifelong
#     python -m confidence_triggered_swarm.scripts.train_lifelong --baseline-path runs/baseline/best_model.pt
#     python -m confidence_triggered_swarm.scripts.train_lifelong --eval-episodes 30 --save-dir runs/ll_test

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from confidence_triggered_swarm.adaptation.confidence import ConfidenceMonitor
from confidence_triggered_swarm.adaptation.ewc import EWCRegularizer
from confidence_triggered_swarm.adaptation.lifelong_trainer import LifelongTrainer
from confidence_triggered_swarm.configs import load_config
from confidence_triggered_swarm.utils.factory import (
    create_env,
    create_agent,
    run_frozen_episodes,
)
from confidence_triggered_swarm.utils.logger import MetricsLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train IPPO baseline + lifelong adaptation evaluation"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    parser.add_argument("--baseline-path", type=str, default=None,
                        help="Pre-trained baseline (skip training if given)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Training timesteps (overrides config)")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="Eval episodes per severity level")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Enable PyBullet GUI")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    env_cfg = config["env"]
    train_cfg = config["training"]
    policy_cfg = config["policy"]
    adapt_cfg = config.get("adaptation", {})
    eval_cfg = config.get("evaluation", {})

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    total_timesteps = args.timesteps if args.timesteps is not None else train_cfg["total_timesteps"]
    n_eval_episodes = (
        args.eval_episodes
        if args.eval_episodes is not None
        else eval_cfg.get("n_eval_episodes", 50)
    )
    gui = args.gui or env_cfg.get("gui", False)

    # pick device
    device_str = train_cfg.get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("runs") / f"lifelong_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Lifelong adaptation | seed={seed} | device={device_str}")
    print(f"  timesteps={total_timesteps:,}  eval_eps={n_eval_episodes}  "
          f"drones={env_cfg['num_drones']}  save_dir={save_dir}")
    print(f"  baseline: {args.baseline_path or '(will train)'}")

    # -- get obs/act dims from a quick env --
    clean_env = create_env(config, severity="clean", gui=gui)
    obs_sample, _ = clean_env.reset()
    obs_dim = obs_sample.shape[-1] if obs_sample.ndim > 1 else obs_sample.shape[0]
    act_dim = clean_env.action_space.shape[-1]
    clean_env.close()

    print(f"  obs_dim={obs_dim}  act_dim={act_dim}")

    # -- train or load baseline --
    baseline_path = str(save_dir / "baseline_model.pt")

    if args.baseline_path is not None:
        print(f"\n  Loading baseline from {args.baseline_path}")
        baseline_path = args.baseline_path
    else:
        print(f"\n  Training baseline for {total_timesteps:,} steps...")
        train_env = create_env(config, severity="clean", gui=gui)

        logger = MetricsLogger(
            log_dir=str(save_dir / "logs"),
            experiment_name=f"baseline_seed{seed}",
        )

        agent = create_agent(config, obs_dim, act_dim, device=device_str)
        print(f"  params={sum(p.numel() for p in agent.policy.parameters()):,}")

        all_stats = agent.train(
            env=train_env,
            total_timesteps=total_timesteps,
            log_interval=1,
            save_path=str(save_dir / "best_baseline.pt"),
            logger=logger,
        )

        agent.save(baseline_path)
        logger.save_csv("baseline_training.csv")
        logger.close()
        train_env.close()

        if all_stats:
            rewards = [s["mean_reward"] for s in all_stats]
            print(f"  Baseline done! best={max(rewards):.2f}, final={rewards[-1]:.2f}")
        print(f"  Saved to: {baseline_path}")

    # -- evaluate across surprise levels --
    severity_levels = ["clean", "mild", "moderate", "severe"]
    frozen_results: Dict[str, Dict[str, Any]] = {}
    lifelong_results: Dict[str, Dict[str, Any]] = {}
    deterministic_eval = config.get("evaluation", {}).get("deterministic_actions", True)

    # phase 1: frozen baseline on each severity
    print("\n--- PHASE 1: Frozen Baseline ---")

    for severity in severity_levels:
        print(f"\n  Frozen baseline on '{severity}'...")
        eval_env = create_env(config, severity=severity, gui=False)
        frozen_agent = create_agent(config, obs_dim, act_dim, device=device_str)
        frozen_agent.load(baseline_path)

        results = run_frozen_episodes(
            frozen_agent, eval_env, n_eval_episodes,
            device=device_str, label=f"frozen/{severity}",
            deterministic=deterministic_eval,
        )
        results["severity"] = severity
        results["mode"] = "frozen"
        frozen_results[severity] = results
        eval_env.close()

    # phase 2: lifelong adaptation on each severity
    print("\n--- PHASE 2: Lifelong Adaptation ---")

    for severity in severity_levels:
        print(f"\n  Lifelong on '{severity}'...")

        # fresh agent from baseline each time
        ll_agent = create_agent(config, obs_dim, act_dim, device=device_str)
        ll_agent.load(baseline_path)

        cal_env = create_env(config, severity="clean", gui=False)
        surprise_env = create_env(config, severity=severity, gui=False)

        # set up adaptation components
        monitor = ConfidenceMonitor(
            policy=ll_agent.policy,
            confidence_threshold=adapt_cfg.get("confidence_threshold", 0.5),
            window_size=adapt_cfg.get("confidence_window", 50),
            use_mc_dropout=True,
            mc_samples=policy_cfg.get("mc_samples", 10),
            device=device_str,
        )
        ewc = EWCRegularizer(
            policy=ll_agent.policy,
            ewc_lambda=adapt_cfg.get("ewc_lambda", 1000),
            device=device_str,
        )
        trainer = LifelongTrainer(
            agent=ll_agent,
            confidence_monitor=monitor,
            ewc_regularizer=ewc,
            adapt_epochs=adapt_cfg.get("adapt_epochs", 5),
            adapt_lr=adapt_cfg.get("adapt_lr", 1e-4),
            adapt_batch_size=adapt_cfg.get("adapt_batch_size", 32),
            replay_buffer_size=adapt_cfg.get("replay_buffer_size", 10000),
            config=config,
            device=device_str,
        )

        # calibrate + EWC snapshot
        trainer.setup(cal_env, n_calibration_episodes=5)

        results = trainer.run_evaluation(
            surprise_env, n_episodes=n_eval_episodes,
            label=f"lifelong/{severity}",
        )
        results["severity"] = severity
        results["mode"] = "lifelong"
        lifelong_results[severity] = results

        cal_env.close()
        surprise_env.close()

    # -- comparison table --
    print("\n" + "=" * 80)
    print("  COMPARISON: Frozen vs Lifelong")
    print("=" * 80)

    header = (
        f"{'Severity':<12} | "
        f"{'Frozen Rew':>10} | "
        f"{'LL Rew':>10} | "
        f"{'Delta':>8} | "
        f"{'Frozen WP':>9} | "
        f"{'LL WP':>6} | "
        f"{'Adapt Rate':>10}"
    )
    print(header)
    print("-" * 80)

    for severity in severity_levels:
        fr = frozen_results[severity]
        lr = lifelong_results[severity]
        delta = lr["mean_reward"] - fr["mean_reward"]
        print(
            f"{severity:<12} | "
            f"{fr['mean_reward']:>10.2f} | "
            f"{lr['mean_reward']:>10.2f} | "
            f"{delta:>+8.2f} | "
            f"{fr['mean_waypoints_reached']:>9.2f} | "
            f"{lr['mean_waypoints_reached']:>6.2f} | "
            f"{lr.get('adaptation_rate', 0.0):>10.2f}"
        )

    print("-" * 80)

    # -- save results JSON --
    def _to_native(obj: Any) -> Any:
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(x) for x in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    all_results = {
        "frozen_baseline": {},
        "lifelong": {},
        "config": {
            "seed": seed,
            "total_timesteps": total_timesteps,
            "n_eval_episodes": n_eval_episodes,
            "baseline_path": baseline_path,
            "device": device_str,
        },
    }
    for severity in severity_levels:
        all_results["frozen_baseline"][severity] = _to_native(frozen_results[severity])
        all_results["lifelong"][severity] = _to_native(lifelong_results[severity])

    results_path = save_dir / "lifelong_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # -- quick inline plots --
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Frozen Baseline vs Lifelong Adaptation", fontsize=14, fontweight="bold")

        x = np.arange(len(severity_levels))
        width = 0.35

        # reward comparison
        ax = axes[0, 0]
        bl_rew = [frozen_results[s]["mean_reward"] for s in severity_levels]
        ll_rew = [lifelong_results[s]["mean_reward"] for s in severity_levels]
        ax.bar(x - width / 2, bl_rew, width, label="Frozen", color="#4C72B0")
        ax.bar(x + width / 2, ll_rew, width, label="Lifelong", color="#DD8452")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Mean Reward by Severity")
        ax.set_xticks(x)
        ax.set_xticklabels(severity_levels)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # waypoints
        ax = axes[0, 1]
        bl_wp = [frozen_results[s]["mean_waypoints_reached"] for s in severity_levels]
        ll_wp = [lifelong_results[s]["mean_waypoints_reached"] for s in severity_levels]
        ax.bar(x - width / 2, bl_wp, width, label="Frozen", color="#4C72B0")
        ax.bar(x + width / 2, ll_wp, width, label="Lifelong", color="#DD8452")
        ax.set_ylabel("Mean Waypoints")
        ax.set_title("Waypoints Reached by Severity")
        ax.set_xticks(x)
        ax.set_xticklabels(severity_levels)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # confidence over episodes
        ax = axes[1, 0]
        colors = ["#55A868", "#C44E52", "#8172B3", "#CCB974"]
        for idx, sev in enumerate(severity_levels):
            confs = lifelong_results[sev].get("all_confidences", [])
            if confs:
                ax.plot(range(1, len(confs) + 1), confs,
                        label=sev, color=colors[idx], alpha=0.8)
        thresh = adapt_cfg.get("confidence_threshold", 0.5)
        ax.axhline(y=thresh, color="red", linestyle="--", alpha=0.5, label="Threshold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Confidence")
        ax.set_title("Confidence Over Episodes")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # adaptation rate
        ax = axes[1, 1]
        adapt_rates = [lifelong_results[s].get("adaptation_rate", 0.0) for s in severity_levels]
        ax.bar(x, adapt_rates, width * 1.5,
               color=[colors[i] for i in range(len(severity_levels))])
        ax.set_ylabel("Adaptation Rate")
        ax.set_title("Adaptation Rate by Severity")
        ax.set_xticks(x)
        ax.set_xticklabels(severity_levels)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plot_path = save_dir / "lifelong_plots.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plots saved to {plot_path}")

    except ImportError:
        print("[WARNING] matplotlib not available, skipping plots")

    print("\nLifelong evaluation complete!")


if __name__ == "__main__":
    main()
