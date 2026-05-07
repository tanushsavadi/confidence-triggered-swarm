# Evaluate trained models on the surprise severity suite.
# Thin wrapper around the Evaluator class — just parses args and runs it.
#
# Usage:
#     python -m confidence_triggered_swarm.scripts.evaluate --baseline-path runs/baseline/best_model.pt
#     python -m confidence_triggered_swarm.scripts.evaluate --baseline-path runs/baseline/best_model.pt --mode frozen
#     python -m confidence_triggered_swarm.scripts.evaluate --baseline-path runs/baseline/best_model.pt --episodes 30

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from confidence_triggered_swarm.configs import load_config
from confidence_triggered_swarm.evaluation.evaluator import Evaluator
from confidence_triggered_swarm.utils.seeding import set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on surprise severity suite"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    parser.add_argument("--baseline-path", type=str, required=True,
                        help="Path to trained baseline .pt file")
    parser.add_argument("--severity", type=str, default="clean,mild,moderate,severe",
                        help="Comma-separated severity levels")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Eval episodes per severity (overrides config)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["frozen", "lifelong", "both"],
                        help="What to evaluate (default: both)")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Enable PyBullet GUI")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    train_cfg = config.get("training", {})

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    severity_levels = [s.strip() for s in args.severity.split(",")]

    if args.episodes is not None:
        config.setdefault("evaluation", {})["n_eval_episodes"] = args.episodes

    device_str = train_cfg.get("device", "auto")

    if args.save_dir:
        save_dir = args.save_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = str(Path("results") / f"eval_{timestamp}")

    set_global_seeds(seed)

    # check baseline exists
    baseline_path = Path(args.baseline_path)
    if not baseline_path.exists():
        print(f"ERROR: Model not found at {baseline_path}")
        print("Train one first: python -m confidence_triggered_swarm.scripts.train_baseline")
        return

    print(f"Evaluation | model={args.baseline_path} | mode={args.mode}")
    print(f"  severity={severity_levels}  "
          f"episodes={config.get('evaluation', {}).get('n_eval_episodes', 50)}  "
          f"seed={seed}  device={device_str}")
    print(f"  save_dir={save_dir}")

    evaluator = Evaluator(
        config=config,
        baseline_model_path=str(baseline_path),
        save_dir=save_dir,
        device=device_str,
        seed=seed,
        config_path=args.config,
    )

    if args.mode == "both":
        print("\nRunning full suite (frozen + lifelong + forgetting)...")
        results = evaluator.run_full_evaluation()

    elif args.mode == "frozen":
        print("\nRunning frozen baseline evaluation...")
        frozen_results = evaluator.evaluate_frozen_baseline(severity_levels)
        results = {
            "baseline": frozen_results,
            "summary": {
                "mode": args.mode,
                "seed": seed,
                "n_eval_episodes": config.get("evaluation", {}).get("n_eval_episodes", 50),
                "severity_levels": severity_levels,
                "baseline_model": str(baseline_path),
                "device": str(evaluator.device),
                "config_path": args.config,
                "config": config,
            },
        }
        evaluator.save_results(results, "frozen_results.json")

        # print table
        print(f"\n{'Severity':<12} | {'Reward':>10} | {'Std':>8} | {'WP':>6} | {'Success':>8}")
        print("-" * 55)
        for sev in severity_levels:
            if sev in frozen_results:
                r = frozen_results[sev]
                print(f"{sev:<12} | {r['mean_reward']:>10.2f} | {r['std_reward']:>8.2f} | "
                      f"{r['mean_waypoints_reached']:>6.2f} | {r['success_rate']:>7.1%}")

    elif args.mode == "lifelong":
        print("\nRunning lifelong adaptation evaluation...")
        lifelong_results = evaluator.evaluate_lifelong(severity_levels)
        results = {
            "lifelong": lifelong_results,
            "summary": {
                "mode": args.mode,
                "seed": seed,
                "n_eval_episodes": config.get("evaluation", {}).get("n_eval_episodes", 50),
                "severity_levels": severity_levels,
                "baseline_model": str(baseline_path),
                "device": str(evaluator.device),
                "config_path": args.config,
                "config": config,
            },
        }
        evaluator.save_results(results, "lifelong_results.json")

        print(f"\n{'Severity':<12} | {'Reward':>10} | {'Conf':>8} | {'WP':>6} | "
              f"{'Adapt Rate':>10} | {'Adaptations':>11}")
        print("-" * 70)
        for sev in severity_levels:
            if sev in lifelong_results:
                r = lifelong_results[sev]
                print(f"{sev:<12} | {r['mean_reward']:>10.2f} | "
                      f"{r.get('mean_confidence', 0):>8.3f} | "
                      f"{r['mean_waypoints_reached']:>6.2f} | "
                      f"{r.get('adaptation_rate', 0):>10.2f} | "
                      f"{r.get('total_adaptations', 0):>11d}")

    print(f"\nEvaluation complete! Results in {save_dir}/")


if __name__ == "__main__":
    main()
