# `confidence_triggered_swarm`

Main Python package for the project. This is where the trainable system lives:
the drone environment, PPO baseline, confidence monitor, lifelong adaptation
loop, evaluation harness, plotting scripts, and config files.

If you are new to the repository, read the root `README.md` first. This file is
the technical map for the code package.

## Clean Training vs Surprise Evaluation

The baseline policy is trained only on the clean `FormationAviary` environment.
Surprises are not used during baseline training. They are injected later by
`envs/surprise_wrapper.py` during frozen evaluation, lifelong adaptation
evaluation, and sequential continual-learning evaluation.

That split matters:

- `train_baseline.py` trains clean IPPO.
- `evaluate.py` loads the clean baseline and tests frozen/lifelong variants
  under clean, mild, moderate, and severe surprise presets.
- `train_continual.py` loads the clean baseline once, adapts through the
  severity sequence, and re-evaluates all severities after each phase.

## Module Overview

### `algorithms/`

PPO agent, actor-critic network, and rollout buffer. We use IPPO: each drone runs the same policy with shared weights, but collects independent rollouts. The actor-critic MLP has MC dropout built in so we can estimate uncertainty at inference time without any extra models.

- [`policy.py`](algorithms/policy.py) - Actor-critic network. Two-headed MLP (shared backbone to policy head and value head). MC dropout is always-on so we can sample variance during deployment.
- [`ppo.py`](algorithms/ppo.py) - PPO trainer with clipped surrogate objective, GAE, value function clipping, and entropy bonus. Handles both baseline training and adaptation updates.
- [`buffer.py`](algorithms/buffer.py) - Rollout buffer with GAE computation. Stores transitions for all drones, computes advantages, and supports mini-batch sampling.

### `adaptation/`

The lifelong learning pieces. Confidence monitoring tells us when the policy is struggling, and the adaptation components fix it without forgetting.

- [`confidence.py`](adaptation/confidence.py) - Dual-signal confidence monitor. Combines policy entropy (how uncertain the action distribution is) and MC dropout variance (how much the network's predictions jitter across stochastic forward passes). Calibrated on clean episodes so we know what "normal" looks like.
- [`ewc.py`](adaptation/ewc.py) - Elastic Weight Consolidation. Computes the Fisher information matrix on clean data, then penalizes changes to important weights during adaptation. This is our main defense against catastrophic forgetting.
- [`lifelong_trainer.py`](adaptation/lifelong_trainer.py) - Orchestrates the whole between-episode adaptation loop. Checks confidence after each episode, decides whether to adapt, runs reward-weighted updates with KL anchoring and clean replay, and manages the EWC penalty.

### `envs/`

Custom Gym environments built on top of gym-pybullet-drones.

- [`formation_aviary.py`](envs/formation_aviary.py) - `FormationAviary` environment. Two drones learn to fly in formation at a target height with a target spacing. Reward penalizes distance from formation targets, height error, and excessive tilting. Extends `BaseAviary` from gym-pybullet-drones.
- [`surprise_wrapper.py`](envs/surprise_wrapper.py) - `SurpriseWrapper` that injects distributional shifts: wind forces, sensor noise, actuator weakness, and goal position shifts. Has severity presets (mild/moderate/severe) so we can test degradation systematically.

### `evaluation/`

- [`evaluator.py`](evaluation/evaluator.py) - `Evaluator` class that runs systematic comparisons. Evaluates frozen baseline vs adapted policy across all severity levels, collects per-episode stats, and outputs JSON results.
- [`continual_metrics.py`](evaluation/continual_metrics.py) - GEM-style metrics from a reward matrix `R[i,j]`: average reward, backward/forward transfer, remembering, clean retention curve.

### `configs/`

- [`default.yaml`](configs/default.yaml) - Single source of truth for all hyperparameters. Scripts load this by default; command-line args override specific values like `--total-timesteps`, `--seed`, etc.

### `scripts/`

Entry points. All runnable as `python -m confidence_triggered_swarm.scripts.<name>`.

### `utils/`

- [`logger.py`](utils/logger.py) - Dual logger: writes TensorBoard events and CSV files. Used during both training and evaluation.
- [`factory.py`](utils/factory.py) - Factory functions for creating environments, policies, and trainers from config.

## Key Hyperparameters

From [`configs/default.yaml`](configs/default.yaml):

| Parameter | Value | What it controls |
|-----------|-------|-----------------|
| `num_drones` | 2 | Agents in the swarm |
| `hidden_dims` | [256, 256] | Actor-critic MLP layers |
| `lr` | 1e-4 | PPO learning rate |
| `gamma` | 0.99 | Discount factor |
| `clip_epsilon` | 0.2 | PPO clipping |
| `batch_size` | 64 | Mini-batch size |
| `mc_dropout_p` | 0.1 | Dropout rate for uncertainty |
| `ewc_lambda` | 1000 | EWC regularization strength |
| `confidence_threshold` | 0.5 | Confidence trigger point |
| `kl_anchor_coef` | 0.5 | KL penalty weight during adaptation |
| `clean_replay_ratio` | 0.2 | Fraction of clean data mixed into adaptation batches |

## Scripts Reference

| Script | What it does |
|--------|-------------|
| [`train_baseline.py`](scripts/train_baseline.py) | Train IPPO on clean formation flight |
| [`train_domain_randomized.py`](scripts/train_domain_randomized.py) | Train a robust baseline with per-episode domain randomization |
| [`train_lifelong.py`](scripts/train_lifelong.py) | Full pipeline: train baseline, evaluate frozen, adapt, evaluate adapted (per severity, reset from baseline) |
| [`train_continual.py`](scripts/train_continual.py) | Sequential continual learning: one agent adapts clean to mild to moderate to severe; retroactive `R[i,j]` matrix + CL metrics |
| [`evaluate.py`](scripts/evaluate.py) | Standalone evaluation using the Evaluator class |
| [`run_ablations.py`](scripts/run_ablations.py) | Ablation study; disables components one at a time |
| [`run_extension_experiments.py`](scripts/run_extension_experiments.py) | Extension runner for seeded screening, final validation, continual checks, and aggregate summaries |
| [`generate_plots.py`](scripts/generate_plots.py) | Figures from JSON: `--evaluation-results`, `--ablation-results`, optional `--continual-results` to `runs/professor_ready/` fig1 through fig8 |
| [`test_env.py`](scripts/test_env.py) | Sanity check that the environment initializes and steps |
| [`diagnose_episodes.py`](scripts/diagnose_episodes.py) | Debug tool; logs per-step states for a few episodes |
| [`visualize.py`](scripts/visualize.py) | 3D PyBullet visualization of trained policy |

## Config

[`configs/default.yaml`](configs/default.yaml) is the single config file. All scripts read it on startup. You can override specific values from the command line:

```bash
python -m confidence_triggered_swarm.scripts.train_baseline \
    --total-timesteps 500000 \
    --seed 42 \
    --output-dir runs/baseline
```

If you want to change something deeper (network size, EWC lambda, etc.), edit the YAML directly.
