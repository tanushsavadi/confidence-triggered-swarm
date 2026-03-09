# Ablation studies on the severe surprise level.
#
# Tests the full lifelong method against three ablated configs:
#   1. No KL anchoring (kl_anchor_coef = 0)
#   2. No clean replay (clean_replay_ratio = 0)
#   3. No EWC (ewc_lambda = 0)
#
# Results go to runs/ablations/ablation_results.json
#
# Usage:
#     python -m confidence_triggered_swarm.scripts.run_ablations

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from confidence_triggered_swarm.configs import load_config
from confidence_triggered_swarm.adaptation.confidence import ConfidenceMonitor
from confidence_triggered_swarm.adaptation.ewc import EWCRegularizer
from confidence_triggered_swarm.adaptation.lifelong_trainer import LifelongTrainer
from confidence_triggered_swarm.utils.factory import (
    create_env,
    create_agent,
    run_frozen_episodes,
)

# paths and settings
BASELINE_MODEL = "runs/baseline/best_model.pt"
SAVE_DIR = Path("runs/ablations")
N_EPISODES = 15          # per ablation (~10 min each on MPS)
SEVERITY = "severe"
SEED = 42

# what we're ablating
ABLATIONS: Dict[str, Dict[str, Any]] = {
    "no_kl_anchor": {
        "label": "No KL Anchoring",
        "overrides": {"adaptation.kl_anchor_coef": 0.0},
    },
    "no_clean_replay": {
        "label": "No Clean Replay",
        "overrides": {"adaptation.clean_replay_ratio": 0.0},
    },
    "no_ewc": {
        "label": "No EWC",
        "overrides": {"adaptation.ewc_lambda": 0.0},
    },
}


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _apply_overrides(config: dict, overrides: Dict[str, Any]) -> dict:
    """Deep-copy config and apply dot-path overrides like 'adaptation.ewc_lambda'."""
    cfg = copy.deepcopy(config)
    for dotpath, value in overrides.items():
        parts = dotpath.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return cfg


def _run_frozen(config: dict, obs_dim: int, act_dim: int, device: str) -> Dict[str, Any]:
    """Frozen baseline on severe — no adaptation at all."""
    agent = create_agent(config, obs_dim, act_dim, device)
    agent.load(BASELINE_MODEL)

    env = create_env(config, severity=SEVERITY, gui=False)
    results = run_frozen_episodes(agent, env, N_EPISODES, device=device, label="frozen/severe")
    env.close()

    # add fields that lifelong results have, for consistent table formatting
    results["mean_confidence"] = None
    results["total_adaptations"] = 0
    return results


def _run_lifelong(
    config: dict, obs_dim: int, act_dim: int, device: str, label: str
) -> Dict[str, Any]:
    """Run lifelong adaptation on severe with the given config."""
    adapt_cfg = config.get("adaptation", {})
    policy_cfg = config.get("policy", {})

    agent = create_agent(config, obs_dim, act_dim, device)
    agent.load(BASELINE_MODEL)

    clean_env = create_env(config, severity="clean", gui=False)
    surprise_env = create_env(config, severity=SEVERITY, gui=False)

    monitor = ConfidenceMonitor(
        policy=agent.policy,
        confidence_threshold=adapt_cfg.get("confidence_threshold", 0.5),
        window_size=adapt_cfg.get("confidence_window", 50),
        use_mc_dropout=True,
        mc_samples=policy_cfg.get("mc_samples", 10),
        device=device,
    )
    ewc = EWCRegularizer(
        policy=agent.policy,
        ewc_lambda=adapt_cfg.get("ewc_lambda", 1000),
        device=device,
    )
    trainer = LifelongTrainer(
        agent=agent,
        confidence_monitor=monitor,
        ewc_regularizer=ewc,
        adapt_epochs=adapt_cfg.get("adapt_epochs", 5),
        adapt_lr=adapt_cfg.get("adapt_lr", 1e-4),
        adapt_batch_size=adapt_cfg.get("adapt_batch_size", 32),
        replay_buffer_size=adapt_cfg.get("replay_buffer_size", 10000),
        config=config,
        device=device,
    )

    trainer.setup(clean_env, n_calibration_episodes=5)

    results = trainer.run_evaluation(
        surprise_env, n_episodes=N_EPISODES, label=label
    )

    clean_env.close()
    surprise_env.close()

    return {
        "mean_reward": float(results["mean_reward"]),
        "std_reward": float(results["std_reward"]),
        "mean_waypoints_reached": float(results["mean_waypoints_reached"]),
        "mean_episode_length": float(results["mean_episode_length"]),
        "mean_confidence": float(results["mean_confidence"]),
        "total_adaptations": int(results["total_adaptations"]),
        "adaptation_rate": float(results["adaptation_rate"]),
        "n_episodes": N_EPISODES,
        "all_rewards": [float(r) for r in results.get("all_rewards", [])],
    }


def _print_table(results: Dict[str, Any], total_elapsed: float) -> None:
    full_reward = results["full_method"]["mean_reward"]

    print(f"\n{'='*90}")
    print("  ABLATION RESULTS — Severe Surprise")
    print(f"{'='*90}")

    print(f"{'Variant':<20} | {'Reward':>10} | {'Std':>8} | {'WP':>6} | "
          f"{'Conf':>8} | {'Adaptations':>11} | {'vs Full':>8}")
    print("-" * 90)

    order = ["frozen_severe", "full_method", "no_kl_anchor", "no_clean_replay", "no_ewc"]
    labels = {
        "frozen_severe": "Frozen (no adapt)",
        "full_method": "Full Method",
        "no_kl_anchor": "No KL Anchoring",
        "no_clean_replay": "No Clean Replay",
        "no_ewc": "No EWC",
    }

    for key in order:
        r = results[key]
        mr = r["mean_reward"]
        conf = r.get("mean_confidence")
        conf_str = f"{conf:.3f}" if conf is not None else "n/a"
        adapt = r.get("total_adaptations", 0)

        if key == "full_method":
            delta_str = "baseline"
        elif abs(full_reward) > 1e-8:
            delta_str = f"{(mr - full_reward) / abs(full_reward) * 100:+.1f}%"
        else:
            delta_str = "n/a"

        print(f"{labels[key]:<20} | {mr:>10.2f} | {r['std_reward']:>8.2f} | "
              f"{r['mean_waypoints_reached']:>6.2f} | {conf_str:>8} | "
              f"{adapt:>11} | {delta_str:>8}")

    print("-" * 90)
    mins = int(total_elapsed // 60)
    secs = total_elapsed % 60
    print(f"  Total time: {mins}m {secs:.1f}s")


def main() -> None:
    start_time = time.time()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    base_config = load_config()
    base_config.setdefault("evaluation", {})["n_eval_episodes"] = N_EPISODES

    device = _resolve_device()

    # get obs/act dims
    tmp_env = create_env(base_config, severity="clean", gui=False)
    obs_sample, _ = tmp_env.reset()
    obs_dim = obs_sample.shape[-1] if obs_sample.ndim > 1 else obs_sample.shape[0]
    act_dim = tmp_env.action_space.shape[-1]
    tmp_env.close()

    print(f"Ablation studies | model={BASELINE_MODEL} | severity={SEVERITY}")
    print(f"  episodes={N_EPISODES}  device={device}")

    all_results: Dict[str, Any] = {}

    # 1. frozen baseline
    print(f"\n  [1/5] Frozen baseline...")
    t0 = time.time()
    all_results["frozen_severe"] = _run_frozen(base_config, obs_dim, act_dim, device)
    print(f"  -> reward={all_results['frozen_severe']['mean_reward']:.2f} ({time.time()-t0:.1f}s)")

    # 2. full method
    print(f"\n  [2/5] Full method...")
    t0 = time.time()
    all_results["full_method"] = _run_lifelong(
        base_config, obs_dim, act_dim, device, label="full_method/severe"
    )
    print(f"  -> reward={all_results['full_method']['mean_reward']:.2f} ({time.time()-t0:.1f}s)")

    # 3-5. ablations
    for idx, (key, ablation) in enumerate(ABLATIONS.items(), start=3):
        print(f"\n  [{idx}/5] {ablation['label']}...")
        ablated_config = _apply_overrides(base_config, ablation["overrides"])
        for dotpath, val in ablation["overrides"].items():
            print(f"    override: {dotpath} = {val}")

        t0 = time.time()
        all_results[key] = _run_lifelong(
            ablated_config, obs_dim, act_dim, device, label=f"{key}/severe"
        )
        print(f"  -> reward={all_results[key]['mean_reward']:.2f} ({time.time()-t0:.1f}s)")

    # save
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    results_path = SAVE_DIR / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    _print_table(all_results, time.time() - start_time)


if __name__ == "__main__":
    main()
