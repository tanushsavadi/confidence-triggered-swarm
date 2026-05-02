# Adaptation Tuning Notes

These runs test whether a more permissive confidence trigger and quality gate improve the saved results. They are exploratory and should **not** replace the canonical results unless followed by a full 50-episode evaluation and continual-learning rerun.

## Code Changes From Tuning

- `load_config()` now supports partial YAML overrides merged into `default.yaml`.
- `LifelongTrainer` no longer counts skipped adaptation attempts as real adaptations.

## Sweep Configs

| Config | Intent |
|--------|--------|
| `confidence_triggered_swarm/configs/sweep_conservative_plus.yaml` | Slightly more permissive trigger and gate. |
| `confidence_triggered_swarm/configs/sweep_balanced.yaml` | More aggressive trigger with reduced KL/EWC. |
| `confidence_triggered_swarm/configs/sweep_aggressive.yaml` | Boundary test for strong triggering. |
| `confidence_triggered_swarm/configs/sweep_rescue.yaml` | Allows very short episodes into replay, with stronger clean replay/KL/EWC. |

## Results Summary

All runs use `runs/baseline/best_model.pt`, seed 42, and lifelong-only evaluation.

| Run | Episodes | Clean | Mild | Moderate | Severe | Adapt rates | Rejected |
|---|---:|---:|---:|---:|---:|---|---|
| `tuning_default_fixed_30ep` | 30 | 1423.9 | 136.4 | 40.6 | 34.3 | clean:0.03, mild:0.03, moderate:0.03, severe:0.03 | clean:0, mild:19, moderate:28, severe:28 |
| `tuning_conservative_plus_30ep` | 30 | 1321.8 | 124.5 | 51.2 | 37.2 | clean:0.03, mild:0.03, moderate:0.03, severe:0.03 | clean:0, mild:20, moderate:27, severe:27 |
| `tuning_rescue_30ep` | 30 | 1320.6 | 123.1 | 47.3 | 20.1 | clean:0.03, mild:0.07, moderate:0.13, severe:0.10 | clean:0, mild:0, moderate:0, severe:0 |
| `tuning_conservative_plus_10ep` | 10 | n/a | 130.9 | 47.2 | 28.5 | mild:0.10, moderate:0.10, severe:0.10 | mild:7, moderate:9, severe:9 |
| `tuning_balanced_10ep` | 10 | n/a | 36.6 | 58.1 | 13.3 | mild:0.10, moderate:0.10, severe:0.00 | mild:9, moderate:9, severe:10 |
| `tuning_aggressive_10ep` | 10 | n/a | 264.8 | 24.1 | 21.0 | mild:0.20, moderate:0.10, severe:0.00 | mild:3, moderate:9, severe:10 |
| `tuning_rescue_fixed_10ep` | 10 | n/a | 135.3 | 35.8 | 22.0 | mild:0.10, moderate:0.10, severe:0.20 | mild:0, moderate:0, severe:0 |

## Interpretation

The current/default behavior remains the best overall 30-episode result. `sweep_conservative_plus` modestly improves moderate and severe in the matched 30-episode run, but it lowers clean and mild. `sweep_rescue` proves that the quality gate is the main reason severe rarely adapts, but allowing all short episodes into replay hurts severe performance over 30 episodes.

The best presentation choice is to keep the canonical results and mention that early tuning suggests the confidence gate is a real bottleneck. A stronger next step would be a better adaptation objective, not just a looser trigger: the current reward-weighted regression can imitate actions from short failure episodes, which is risky under severe surprise.

## Improved Adaptation Pass, April 29 2026

This pass added deterministic evaluation/deployment, a PPO-style episode adaptation option, advantage-based replay filtering, and a stronger sweep runner. Results are stored under `runs/improved_*` so they do not overwrite `runs/professor_ready/`.

### Short Sweep

Command:

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.sweep_improved_adaptation \
  --episodes 10 \
  --baseline-path runs/baseline/best_model.pt \
  --save-root runs/improved_sweep_10ep
```

Summary table: `runs/improved_sweep_10ep/summary.md`

| Config | Clean | Mild | Moderate | Severe | Surprise mean | Adaptations | Skipped | Rejected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `current_default` | 1913.67 | 51.00 | 24.05 | 17.74 | 30.93 | 1 | 0 | 30 |
| `deterministic_only` | 1913.67 | 406.05 | 115.53 | 83.97 | 201.85 | 4 | 0 | 18 |
| `ppo_episode` | 1995.75 | 107.32 | 26.93 | 21.56 | 51.94 | 2 | 0 | 27 |
| `ppo_conservative` | 2092.81 | 127.03 | 99.42 | 12.61 | 79.69 | 3 | 0 | 22 |
| `ppo_moderate` | 2133.50 | 315.64 | 106.44 | 13.83 | 145.30 | 3 | 0 | 21 |
| `reward_weighted_rescue` | 1979.97 | 344.60 | 117.21 | 48.06 | 169.96 | 4 | 0 | 19 |

The short sweep shows a clear ablation result: deterministic deployment/evaluation is the strongest improvement. PPO-style episode adaptation is implemented and works, but these first PPO configs did not beat the deterministic reward-weighted baseline. The early-rescue reward-weighted config adapted earlier but did not outperform deterministic-only in the 10-episode sweep.

### 50-Episode Validation

Commands:

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.evaluate \
  --config confidence_triggered_swarm/configs/improved_deterministic.yaml \
  --baseline-path runs/baseline/best_model.pt \
  --mode both \
  --severity clean,mild,moderate,severe \
  --episodes 50 \
  --save-dir runs/improved_validation/deterministic_50ep
```

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.evaluate \
  --config confidence_triggered_swarm/configs/improved_reward_weighted_rescue.yaml \
  --baseline-path runs/baseline/best_model.pt \
  --mode both \
  --severity clean,mild,moderate,severe \
  --episodes 50 \
  --save-dir runs/improved_validation/reward_weighted_rescue_50ep
```

| Validation | Clean Δ | Mild Δ | Moderate Δ | Severe Δ | Forgetting |
|---|---:|---:|---:|---:|---|
| `deterministic_50ep` | +83.60 (+4.3%) | -44.74 (-16.1%) | +39.77 (+47.7%) | -1.23 (-4.0%) | No |
| `reward_weighted_rescue_50ep` | -27.82 (-1.4%) | -40.95 (-16.1%) | -1.66 (-1.6%) | -2.96 (-8.0%) | No |

### Decision

Do not replace the canonical professor-ready results with these improved runs. The deterministic run is useful evidence because it improves moderate robustness and clean retention, but it fails the acceptance bar that mild and severe must improve over frozen. The rescue run confirms that earlier adaptation is not sufficient by itself. For presentation, this should be framed as an honest tuning appendix: deterministic deployment is a real methodological improvement, PPO-style adaptation infrastructure is now available, and the remaining bottleneck is adaptation data quality/objective strength rather than just trigger frequency.

## Commands

Example matched 30-episode evaluation:

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.evaluate \
  --config confidence_triggered_swarm/configs/sweep_conservative_plus.yaml \
  --baseline-path runs/baseline/best_model.pt \
  --mode lifelong \
  --severity clean,mild,moderate,severe \
  --episodes 30 \
  --save-dir runs/tuning_conservative_plus_30ep
```
