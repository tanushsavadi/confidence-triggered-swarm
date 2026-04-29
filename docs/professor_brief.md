# Professor Brief

## One-Sentence Summary

We trained a clean two-drone PPO formation policy, evaluated how it breaks under post-training surprises, then added confidence-triggered lifelong adaptation with anti-forgetting safeguards and a sequential continual-learning evaluation to check for catastrophic forgetting.

## What Is Implemented

- Clean-trained shared-policy IPPO baseline for two drones.
- Surprise suite with clean, mild, moderate, and severe conditions.
- Confidence monitor using policy entropy plus MC-dropout variance.
- Between-episode lifelong adaptation.
- Anti-forgetting safeguards: KL anchoring, clean replay, and EWC.
- Frozen vs lifelong evaluation across severities.
- Forgetting check on clean after severe adaptation.
- Ablation study for KL, clean replay, and EWC.
- Sequential continual-learning run: clean -> mild -> moderate -> severe, with re-evaluation on every severity after every phase.

## Main Per-Severity Results

Source: `runs/full_eval/evaluation_results.json`, 50 episodes per condition, seed 42.

| Severity | Frozen reward | Lifelong reward | Change |
|----------|--------------:|----------------:|-------:|
| Clean | 1305.2 | 1358.9 | +4.1% |
| Mild | 105.9 | 159.7 | +50.8% |
| Moderate | 49.4 | 45.2 | -8.5% |
| Severe | 27.3 | 42.2 | +54.7% |

Interpretation: the clean-trained policy is brittle under surprise. Lifelong adaptation helps on mild and severe in this seed, but moderate remains mixed.

## Forgetting Check

Source: `runs/full_eval/evaluation_results.json`.

| Condition | Clean reward | Waypoints |
|-----------|-------------:|----------:|
| Baseline on clean | 1308.9 | 0.90 |
| After severe adaptation, evaluated on clean | 1386.1 | 0.86 |

The evaluator reports `forgetting_detected: false`. Clean reward is maintained after severe adaptation, although this is still a single-seed result.

## Professor Feedback Response

The feedback asked whether we trained/adapted on harder conditions and then went back to test clean performance. That is now answered by `train_continual.py` and `runs/continual_run/continual_results.json`.

Sequential run: clean -> mild -> moderate -> severe. After each phase, the same lifelong policy is evaluated on all severities.

| Phase completed | Clean eval | Mild eval | Moderate eval | Severe eval |
|-----------------|-----------:|----------:|--------------:|------------:|
| After clean | 1316.0 | 146.9 | 48.1 | 38.5 |
| After mild | 1261.3 | 143.5 | 94.9 | 21.5 |
| After moderate | 1406.2 | 159.2 | 40.3 | 31.8 |
| After severe | 1322.9 | 131.2 | 92.0 | 26.1 |

Clean retention remains in the same range across the sequence: 1316.0 -> 1261.3 -> 1406.2 -> 1322.9. This supports the claim that this run does not show catastrophic forgetting on the clean task.

## Continual-Learning Metrics

Source: `runs/continual_run/continual_results.json`.

| Metric | Lifelong | Frozen reference |
|--------|---------:|-----------------:|
| Average reward after final phase | 393.1 | 412.9 |
| BWT | +15.4 | 0.0 |
| FWT | +17.2 | 0.0 |
| Remembering | 1.0 | 1.0 |

Best phrasing: the continual run shows clean retention and positive transfer metrics, but not a universal performance win over frozen.

## Figures To Show

- `runs/professor_ready/fig1_frozen_vs_lifelong.png`: frozen vs lifelong reward by severity.
- `runs/professor_ready/fig2_degradation.png`: degradation curve as surprise severity increases.
- `runs/professor_ready/fig4_forgetting.png`: clean performance before and after severe adaptation.
- `runs/professor_ready/fig5_training_over_time.png`: sequential reward over time.
- `runs/professor_ready/fig6_continual_matrix.png`: reward matrix, most direct answer to the feedback.
- `runs/professor_ready/fig7_clean_retention.png`: clean reward after each phase.
- `runs/professor_ready/fig8_cl_metrics.png`: average reward, BWT, FWT, remembering.

## Honest Limitations

- All canonical results are seed 42 only.
- The policy is tested with two drones, not a large swarm.
- Severe absolute reward is still low.
- The confidence trigger is conservative: 2% adaptation rate in the main per-severity evaluation.
- The final continual average reward is slightly lower than the frozen reference, so the result should be framed as retention plus partial robustness, not a complete lifelong-learning win.
