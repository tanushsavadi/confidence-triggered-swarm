# Confidence-Triggered Swarm Adaptation

Lifelong adaptation for a clean-trained drone-swarm policy under post-training
surprises.

This repository is the CS 590NN/690NN final project codebase. It contains the
environment, PPO training code, confidence-triggered adaptation code, saved
experiment results, professor-ready figures, and final report/presentation
handoff material.

## Start Here

The shortest accurate description is:

> We train a two-drone PPO policy only on clean formation flight, then test and
> adapt it after training under surprises such as wind, sensor noise, actuator
> weakness, and shifted goals.

This is not domain randomization. The baseline checkpoint was not trained with
surprises. The scientific question is whether a clean-trained policy can detect
post-training shift, adapt between episodes, and avoid forgetting clean flight.

## What To Read First

| Goal | File |
|---|---|
| Understand the project in 10 minutes | This README |
| See the main results and professor-facing interpretation | `docs/professor_brief.md` |
| Build the final report and slides | `final_submission/README.md` |
| Grab slide/report figures and know what each means | `docs/artifact_guide.md` |
| Inspect module-level code structure | `confidence_triggered_swarm/README.md` |
| Read the older long-form writeup | `REPORT.md` |
| Check final submission readiness | `docs/final_submission_audit.md` |
| Do the final repository push review | `docs/final_push_checklist.md` |

## Main Claim

The final report uses the extension validation as the main empirical result:
three controlled evaluation seeds, 75 episodes per severity, and the same
clean-trained checkpoint for every method. The saved results support a careful
claim:

- A clean-trained drone policy is brittle under post-training surprises.
- Confidence-triggered adaptation partially recovers reward on mild and severe
  surprises while mostly preserving clean reward.
- Moderate surprise remains a clear failure case for every adaptive variant.
- Clean performance is retained in the forgetting check and in the sequential
  continual-learning matrix, but those diagnostics are still seed-42 support
  runs.
- Success rates remain near zero, so the method improves shaped reward and
  waypoint progress rather than reliably solving the surprised task.

Use this phrasing in the report and presentation. Do not claim that lifelong
adaptation beats frozen evaluation on every metric.

## Training Setup

The baseline policy is trained in the clean `FormationAviary` environment.

| Setting | Value |
|---|---:|
| Simulator | `gym-pybullet-drones` with PyBullet |
| Environment | `FormationAviary` |
| Drones | 2 |
| Drone model | `CF2X` |
| Physics | `PYB` |
| Action type | `VEL` |
| Observation type | `KIN` plus waypoint offset |
| Physics frequency | 240 Hz |
| Control frequency | 30 Hz |
| Episode length | 15 s |
| Initial height | 0.5 m |
| Formation spacing | 0.5 m |
| Speed limit | 0.5 m/s |
| Waypoint threshold | 0.2 m |
| Tilt truncation | 1.0 rad |
| Minimum altitude | 0.01 m |

Default waypoints:

```text
[0.15, 0.15, 0.5]
[0.40, 0.40, 0.5]
[0.60, 0.00, 0.7]
```

Baseline PPO/IPPO values:

| Setting | Value |
|---|---:|
| Total timesteps | 1,000,000 |
| Seed | 42 |
| Learning rate | 1e-4 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| PPO clip epsilon | 0.2 |
| Entropy coefficient | 0.005 |
| Value coefficient | 0.5 |
| Max grad norm | 0.5 |
| PPO epochs | 10 |
| Batch size | 64 |
| Rollout steps | 2048 |
| Hidden layers | `[256, 256]` |
| Activation | `tanh` |
| MC dropout | 0.1 |

## Surprise Presets

Surprises are applied after baseline training through `SurpriseWrapper`.

| Severity | Wind max | Sensor noise | Dropout | Actuator scale | Goal shift |
|---|---:|---:|---:|---:|---|
| clean | 0.00 N | 0.00 | 0.00 | 1.00 | none |
| mild | 0.02 N | 0.01 | 0.00 | 1.00 | none |
| moderate | 0.05 N | 0.02 | 0.02 | 0.85 | none |
| severe | 0.10 N | 0.05 | 0.05 | 0.70 | `p=0.001`, 0.1 m |

## Adaptation Setup

The lifelong policy starts from the same clean-trained baseline checkpoint. It
calibrates confidence on clean rollouts and adapts only between episodes.

| Setting | Value |
|---|---:|
| Method | `reward_weighted_bc` |
| Confidence signals | policy entropy plus MC-dropout variance |
| MC samples | 10 |
| Confidence threshold | 0.5 |
| Confidence window | 30 |
| Adaptation epochs | 5 |
| Adaptation learning rate | 1e-4 |
| EWC lambda | 1000 |
| Clean replay ratio | 0.2 |
| KL anchor coefficient | 0.5 |
| Replay buffer size | 10000 |
| Minimum episode steps for replay | 30 |
| Minimum per-step reward for replay | -5.0 |
| Evaluation actions | deterministic |

The anti-forgetting safeguards are:

- KL anchoring to the clean policy.
- Clean replay from calibration episodes.
- Elastic Weight Consolidation on important clean-task parameters.

## Canonical Results

The main results are already saved. You do not need to rerun expensive
experiments to prepare slides or the report.

| Artifact | Meaning |
|---|---|
| `runs/baseline/best_model.pt` | Clean-trained baseline checkpoint |
| `runs/extension/validation/aggregate_summary.json` | Final three-seed validation used by the report |
| `runs/extension/validation/diagnostic_reward_retention.pdf` | Final report figure for reward recovery vs clean retention |
| `runs/full_eval/evaluation_results.json` | Seed-42 frozen vs lifelong evaluation and forgetting check |
| `runs/ablations/ablation_results.json` | Severe-surprise ablations for KL, clean replay, and EWC |
| `runs/continual_run/continual_results.json` | Sequential clean, mild, moderate, severe continual-learning matrix |
| `runs/professor_ready/` | PNG/PDF figures and a local README generated from the canonical JSONs |

Final validation results from
`runs/extension/validation/aggregate_summary.json`:

| Severity | Frozen | Current | Always-adapt | Improved PPO | Reward rescue |
|---|---:|---:|---:|---:|---:|
| clean | 2011.6 +/- 0.0 | 1972.4 +/- 10.9 | 1971.9 +/- 3.2 | 1988.0 +/- 19.9 | 2004.9 +/- 23.6 |
| mild | 199.8 +/- 16.9 | 235.7 +/- 24.8 | 238.2 +/- 26.4 | 241.5 +/- 34.5 | 233.1 +/- 34.7 |
| moderate | 119.3 +/- 13.3 | 77.4 +/- 10.6 | 77.3 +/- 9.7 | 78.3 +/- 11.0 | 77.7 +/- 9.9 |
| severe | 30.5 +/- 5.6 | 38.6 +/- 3.9 | 38.1 +/- 3.8 | 38.2 +/- 4.0 | 38.3 +/- 4.0 |

The report uses paired seed-level percent changes for method deltas. Under that
definition, reward-weighted rescue preserves clean reward (-0.33%) and improves
mild (+17.8%) and severe (+39.2%) reward, but drops on moderate (-31.5%).

Sequential continual-learning clean retention from
`runs/continual_run/continual_results.json`:

| Phase completed | Clean reward |
|---|---:|
| after clean | 1316.0 |
| after mild | 1261.3 |
| after moderate | 1406.2 |
| after severe | 1322.9 |

This matrix directly answers the forgetting question: clean was re-evaluated
after adapting to later surprise phases.

## Slide And Report Artifacts

Use these files for the slide deck:

| Figure | File | Use |
|---|---|---|
| Fig 1 | `runs/professor_ready/fig1_frozen_vs_lifelong.png` | Main frozen vs lifelong result |
| Fig 2 | `runs/professor_ready/fig2_degradation.png` | Shows frozen degradation under surprise |
| Fig 3 | `runs/professor_ready/fig3_ablations.png` | Safeguard ablations and tuning discussion |
| Fig 4 | `runs/professor_ready/fig4_forgetting.png` | Clean-before vs clean-after adaptation |
| Fig 5 | `runs/professor_ready/fig5_training_over_time.png` | Sequential reward over time |
| Fig 6 | `runs/professor_ready/fig6_continual_matrix.png` | Most important continual-learning slide |
| Fig 7 | `runs/professor_ready/fig7_clean_retention.png` | Clean retention across phases |
| Fig 8 | `runs/professor_ready/fig8_cl_metrics.png` | BWT, FWT, remembering, average reward |

The final report's main validation figure is separate:

| Figure | File | Use |
|---|---|---|
| Validation | `runs/extension/validation/diagnostic_reward_retention.pdf` | Reward recovery vs clean retention across final methods |

An editable 10-slide PPTX for the May 4, 2026, 10-12 minute presentation has
already been generated at:

```text
final_submission/confidence_triggered_swarm_premium.pptx
```

Detailed slide timing is in `final_submission/slides_10_12min_outline.md`.
The complete rehearsal script is in `final_submission/slide_speaker_notes.md`.
Artifact-by-artifact guidance is in `docs/artifact_guide.md`.

Regenerate all figures from saved JSON outputs:

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.generate_plots \
  --evaluation-results runs/full_eval/evaluation_results.json \
  --ablation-results runs/ablations/ablation_results.json \
  --continual-results runs/continual_run/continual_results.json \
  --output-dir runs/professor_ready
```

## Repository Layout

```text
confidence_triggered_swarm/
  algorithms/      PPO agent, actor-critic policy, rollout buffer
  adaptation/      confidence monitor, EWC, lifelong trainer
  envs/            FormationAviary and SurpriseWrapper
  evaluation/      evaluator and continual-learning metrics
  configs/         YAML hyperparameters
  scripts/         training, evaluation, plotting, diagnostics
  utils/           factories and logging

runs/              saved checkpoints, JSON results, and generated figures
docs/              professor brief, demo notes, artifact guide, audit docs
final_submission/  Compiled final report, report source, and 10-12 minute slide outline
_research/         reference MAPPO/gym-pybullet-drones code, not core project code
gym-pybullet-drones-install/
                   local editable copy of the simulator dependency
```

## Setup

In this checked-out workspace, `.venv310` already exists. In a fresh clone,
create an environment with Python 3.10 or newer:

```bash
conda create -n drones python=3.10 -y
conda activate drones
```

Install the simulator and project dependencies:

```bash
cd gym-pybullet-drones-install
pip install -e .
cd ..
pip install -r confidence_triggered_swarm/requirements.txt
```

## Common Commands

Run a quick environment smoke test:

```bash
python -m confidence_triggered_swarm.scripts.test_env
```

Train the clean baseline:

```bash
python -m confidence_triggered_swarm.scripts.train_baseline \
  --timesteps 1000000 \
  --seed 42 \
  --save-dir runs/baseline
```

Evaluate frozen and lifelong policies:

```bash
python -m confidence_triggered_swarm.scripts.evaluate \
  --baseline-path runs/baseline/best_model.pt \
  --mode both \
  --episodes 50 \
  --save-dir runs/full_eval
```

Run the final extension validation summary from saved outputs:

```bash
MPLCONFIGDIR=/private/tmp/mplcache ./.venv310/bin/python -m confidence_triggered_swarm.scripts.run_extension_experiments \
  summarize \
  --save-root runs/extension
```

Run the sequential continual-learning evaluation:

```bash
python -m confidence_triggered_swarm.scripts.train_continual \
  --baseline-path runs/baseline/best_model.pt \
  --save-dir runs/continual_run
```

Check final submission readiness:

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.check_final_readiness
```

## Final Submission Status

The final handoff folder is `final_submission/`.

Current state:

- `final_submission/final_report.pdf` is compiled.
- `final_submission/final_report.tex` includes author contributions and the
  public GitHub repository link.
- `final_submission/neurips_2026.sty` is present.
- `check_final_readiness` passes all automated checks except the local
  `pdfinfo` page-count check, because `pdfinfo` is not installed. The LaTeX log
  reports an 8-page PDF.

## Future Work

- Independent training seeds, not only controlled evaluation seeds from one
  trained checkpoint.
- Larger swarms beyond two drones.
- Stronger adaptation objectives and better confidence triggers.
- Peer-help or communication between drones.
- Richer surprise types, including communication loss and progressive sensor degradation.
- Online within-episode adaptation.
- Real-world transfer to physical quadrotors.
