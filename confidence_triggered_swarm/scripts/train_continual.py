#!/usr/bin/env python3
"""Sequential continual learning: adapt through clean→mild→moderate→severe once.

After each phase, re-evaluates on *all* severities (retroactive matrix R[i,j]).
Complements `train_lifelong.py`, which resets from baseline per severity.

Usage:
    python -m confidence_triggered_swarm.scripts.train_continual \\
        --baseline-path runs/baseline/best_model.pt --save-dir runs/continual_run
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from confidence_triggered_swarm.adaptation.confidence import ConfidenceMonitor
from confidence_triggered_swarm.adaptation.ewc import EWCRegularizer
from confidence_triggered_swarm.adaptation.lifelong_trainer import LifelongTrainer
from confidence_triggered_swarm.configs import load_config
from confidence_triggered_swarm.evaluation.continual_metrics import compute_all_metrics
from confidence_triggered_swarm.utils.factory import (
    create_env,
    create_agent,
    run_frozen_episodes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sequential continual adaptation + R-matrix eval (van de Ven / GEM-style)"
    )
    p.add_argument("--config", type=str, default=None, help="Path to config YAML")
    p.add_argument("--baseline-path", type=str, required=True, help="Pre-trained baseline")
    p.add_argument(
        "--n-adapt-episodes",
        type=int,
        default=None,
        help="Episodes per phase for adaptation (lifelong updates); default from config eval",
    )
    p.add_argument(
        "--n-eval-episodes",
        type=int,
        default=None,
        help="Episodes per eval cell in R matrix; default from config eval",
    )
    p.add_argument("--save-dir", type=str, default=None, help="Output directory")
    p.add_argument("--gui", action="store_true", default=False, help="PyBullet GUI")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    return p.parse_args()


def _to_native(obj: Any) -> Any:
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    env_cfg = config["env"]
    train_cfg = config["training"]
    policy_cfg = config["policy"]
    adapt_cfg = config.get("adaptation", {})
    eval_cfg = config.get("evaluation", {})

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    n_adapt = (
        args.n_adapt_episodes
        if args.n_adapt_episodes is not None
        else eval_cfg.get("n_eval_episodes", 50)
    )
    n_eval = (
        args.n_eval_episodes
        if args.n_eval_episodes is not None
        else eval_cfg.get("n_eval_episodes", 50)
    )
    gui = args.gui or env_cfg.get("gui", False)

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
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("runs") / f"continual_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    phases = ["clean", "mild", "moderate", "severe"]
    baseline_path = args.baseline_path

    print(f"Continual sequential adaptation | seed={seed} | device={device_str}")
    print(f"  phases={phases}  n_adapt={n_adapt}  n_eval={n_eval}  save_dir={save_dir}")
    print(f"  baseline: {baseline_path}")

    clean_probe = create_env(config, severity="clean", gui=gui)
    obs_sample, _ = clean_probe.reset()
    obs_dim = obs_sample.shape[-1] if obs_sample.ndim > 1 else obs_sample.shape[0]
    act_dim = clean_probe.action_space.shape[-1]
    clean_probe.close()

    # --- Lifelong agent: single trajectory through all phases ---
    ll_agent = create_agent(config, obs_dim, act_dim, device=device_str)
    ll_agent.load(baseline_path)

    frozen_agent = create_agent(config, obs_dim, act_dim, device=device_str)
    frozen_agent.load(baseline_path)

    cal_env = create_env(config, severity="clean", gui=False)
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
    trainer.setup(cal_env, n_calibration_episodes=5)
    cal_env.close()

    T = len(phases)
    R_ll = np.zeros((T, T), dtype=float)
    R_fr = np.zeros((T, T), dtype=float)

    per_episode: Dict[str, List[float]] = {"lifelong": [], "frozen": []}
    phase_boundaries: List[int] = []

    env_cache: Dict[str, Any] = {}  # lazy-created envs by severity

    def get_env(severity: str) -> Any:
        if severity not in env_cache:
            env_cache[severity] = create_env(config, severity=severity, gui=False)
        return env_cache[severity]

    for i, phase_sev in enumerate(phases):
        print(f"\n=== PHASE {i}: adapt on '{phase_sev}' ({n_adapt} episodes) ===")
        surprise_env = get_env(phase_sev)

        ll_stats = trainer.run_evaluation(
            surprise_env,
            n_episodes=n_adapt,
            label=f"continual/lifelong/{phase_sev}",
        )
        per_episode["lifelong"].extend(ll_stats.get("all_rewards", []))

        fr_stats = run_frozen_episodes(
            frozen_agent,
            surprise_env,
            n_adapt,
            device=device_str,
            label=f"continual/frozen/{phase_sev}",
        )
        per_episode["frozen"].extend(fr_stats.get("all_rewards", []))

        phase_boundaries.append(len(per_episode["lifelong"]))

        print(f"\n--- Retroactive eval after phase {i} (n_eval={n_eval}) ---")
        for j, eval_sev in enumerate(phases):
            eval_env = get_env(eval_sev)
            ll_r = run_frozen_episodes(
                ll_agent,
                eval_env,
                n_eval,
                device=device_str,
                label=f"R_ll[{i},{j}] {eval_sev}",
            )
            fr_r = run_frozen_episodes(
                frozen_agent,
                eval_env,
                n_eval,
                device=device_str,
                label=f"R_fr[{i},{j}] {eval_sev}",
            )
            R_ll[i, j] = ll_r["mean_reward"]
            R_fr[i, j] = fr_r["mean_reward"]
            print(
                f"  R_lifelong[{i},{j}] ({eval_sev})={R_ll[i, j]:.2f}  "
                f"R_frozen[{i},{j}]={R_fr[i, j]:.2f}"
            )

    for e in env_cache.values():
        e.close()

    frozen_baseline_row = np.array([R_fr[0, j] for j in range(T)], dtype=float)
    baseline_mag = float(np.mean(np.abs(np.diag(R_ll)))) if T else 1.0

    metrics_ll = compute_all_metrics(R_ll, frozen_baseline_row, baseline_magnitude=baseline_mag)
    # Frozen policy does not change across phases: use row 0 for evals; BWT/FWT are 0 by definition.
    clean_0 = float(R_fr[0, 0])
    metrics_fr = {
        "average_reward": float(np.mean(R_fr[-1])),
        "backward_transfer": 0.0,
        "forward_transfer": 0.0,
        "remembering": 1.0,
        "clean_retention": [clean_0] * T,
    }

    out: Dict[str, Any] = {
        "phases": phases,
        "R_lifelong": R_ll.tolist(),
        "R_frozen": R_fr.tolist(),
        "per_episode_rewards": per_episode,
        "phase_boundaries": phase_boundaries,
        "frozen_baseline_per_task": frozen_baseline_row.tolist(),
        "metrics": {
            "lifelong": metrics_ll,
            "frozen": metrics_fr,
        },
        "config": {
            "seed": seed,
            "n_adapt_episodes": n_adapt,
            "n_eval_episodes": n_eval,
            "baseline_path": baseline_path,
            "device": device_str,
        },
    }

    path = save_dir / "continual_results.json"
    with open(path, "w") as f:
        json.dump(_to_native(out), f, indent=2)
    print(f"\nSaved {path}")

    print("\n=== Metrics (lifelong) ===")
    for k, v in metrics_ll.items():
        print(f"  {k}: {v}")
    print("\n=== Metrics (frozen reference matrix) ===")
    for k, v in metrics_fr.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
