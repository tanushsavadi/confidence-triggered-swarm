# Professor Brief

## One-Sentence Summary

We trained a clean two-drone PPO formation policy, evaluated how it breaks under
post-training surprises, then added confidence-triggered lifelong adaptation
with anti-forgetting safeguards and a sequential continual-learning diagnostic
to check for catastrophic forgetting.

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
- Extension validation over three controlled evaluation seeds and 75 episodes
  per severity, comparing frozen, current lifelong, always-adapt, improved PPO,
  and reward-weighted rescue.

## Final Validation Results

Source: `runs/extension/validation/aggregate_summary.json`, three controlled
evaluation seeds, 75 episodes per severity. Entries are reward mean +/- standard
error across seeds.

| Severity | Frozen | Current | Always-adapt | Improved PPO | Reward rescue |
|----------|-------:|--------:|-------------:|-------------:|--------------:|
| Clean | 2011.6 +/- 0.0 | 1972.4 +/- 10.9 | 1971.9 +/- 3.2 | 1988.0 +/- 19.9 | 2004.9 +/- 23.6 |
| Mild | 199.8 +/- 16.9 | 235.7 +/- 24.8 | 238.2 +/- 26.4 | 241.5 +/- 34.5 | 233.1 +/- 34.7 |
| Moderate | 119.3 +/- 13.3 | 77.4 +/- 10.6 | 77.3 +/- 9.7 | 78.3 +/- 11.0 | 77.7 +/- 9.9 |
| Severe | 30.5 +/- 5.6 | 38.6 +/- 3.9 | 38.1 +/- 3.8 | 38.2 +/- 4.0 | 38.3 +/- 4.0 |

Interpretation: the clean-trained policy is brittle under surprise. Adaptive
variants improve mild and severe reward and mostly retain clean reward, but all
adaptive variants are worse than frozen on moderate. Success rates are near
zero, so this is partial reward recovery, not reliable task completion.

Paired seed-level changes used in the final report:

| Method | Clean | Mild | Moderate | Severe |
|--------|------:|-----:|---------:|-------:|
| Current | -1.95% | +19.35% | -31.59% | +40.14% |
| Always-adapt | -1.98% | +20.59% | -31.99% | +38.93% |
| Improved PPO | -1.17% | +22.09% | -30.83% | +39.04% |
| Reward rescue | -0.33% | +17.79% | -31.54% | +39.16% |

## Supporting Forgetting Check

Source: `runs/full_eval/evaluation_results.json`, seed 42.

| Condition | Clean reward | Waypoints |
|-----------|-------------:|----------:|
| Baseline on clean | 1308.9 | 0.90 |
| After severe adaptation, evaluated on clean | 1386.1 | 0.86 |

The evaluator reports `forgetting_detected: false`. Clean reward is maintained
after severe adaptation, although this is still a single-seed diagnostic.

## Professor Feedback Response

The feedback asked whether we trained/adapted on harder conditions and then went back to test clean performance. That is now answered by `train_continual.py` and `runs/continual_run/continual_results.json`.

Sequential run: clean -> mild -> moderate -> severe. After each phase, the same
lifelong policy is evaluated on all severities. This is supporting evidence for
the neuroscience/course framing around stable memory and continual learning.

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

Best phrasing: the continual run shows clean retention and positive transfer
metrics in one seed, but not a universal performance win over frozen.

## Figures To Show

- `runs/extension/validation/diagnostic_reward_retention.pdf`: final validation
  reward recovery versus clean retention.
- `runs/professor_ready/fig1_frozen_vs_lifelong.png`: seed-42 frozen vs lifelong reward by severity.
- `runs/professor_ready/fig2_degradation.png`: degradation curve as surprise severity increases.
- `runs/professor_ready/fig4_forgetting.png`: clean performance before and after severe adaptation.
- `runs/professor_ready/fig5_training_over_time.png`: sequential reward over time.
- `runs/professor_ready/fig6_continual_matrix.png`: reward matrix, most direct answer to the feedback.
- `runs/professor_ready/fig7_clean_retention.png`: clean reward after each phase.
- `runs/professor_ready/fig8_cl_metrics.png`: average reward, BWT, FWT, remembering.

## Honest Limitations

- Final validation uses three controlled evaluation seeds, but still only one
  trained baseline checkpoint.
- The policy is tested with two drones, not a large swarm.
- Severe absolute reward is still low.
- Success rates remain near zero.
- Moderate surprise remains a failure case for every adaptive method.
- The confidence trigger is conservative in the final validation.
- The final continual average reward is slightly lower than the frozen
  reference, so the result should be framed as retention plus partial
  robustness, not a complete lifelong-learning win.
