# Continual-Learning Presentation Outline

## Slide 1: Problem

- A policy trained in clean simulation can fail after deployment shifts.
- Surprise conditions: wind, sensor noise, actuator weakness, goal shift.
- The question is not just adaptation; it is whether adaptation destroys clean-task skill.

## Slide 2: Method

- Train shared-policy IPPO on clean two-drone formation flight.
- Deploy to surprise environments.
- Trigger adaptation with confidence monitoring:
  - policy entropy
  - MC-dropout variance
- Protect clean behavior with:
  - KL anchoring
  - clean replay
  - EWC

## Slide 3: Frozen Policy Degradation

Use `runs/professor_ready/fig2_degradation.png`.

- Clean reward: 1305.2.
- Mild reward: 105.9.
- Severe reward: 27.3.
- Takeaway: clean-only training is brittle under post-training surprise.

## Slide 4: Frozen vs Lifelong

Use `runs/professor_ready/fig1_frozen_vs_lifelong.png`.

| Severity | Frozen | Lifelong | Change |
|----------|-------:|---------:|-------:|
| Clean | 1305.2 | 1358.9 | +4.1% |
| Mild | 105.9 | 159.7 | +50.8% |
| Moderate | 49.4 | 45.2 | -8.5% |
| Severe | 27.3 | 42.2 | +54.7% |

Takeaway: adaptation helps in mild and severe, but not every condition.

## Slide 5: Forgetting Check

Use `runs/professor_ready/fig4_forgetting.png`.

- Baseline clean reward: 1308.9.
- Clean reward after severe adaptation: 1386.1.
- `forgetting_detected: false`.
- Takeaway: no catastrophic forgetting in this seed.

## Slide 6: Sequential Continual Evaluation

Use `runs/professor_ready/fig6_continual_matrix.png`.

Explain the matrix:

- Rows: training phase completed.
- Columns: evaluation task.
- This directly answers whether clean was re-evaluated after mild/moderate/severe.

Clean-retention values:

| Phase completed | Clean reward |
|-----------------|-------------:|
| After clean | 1316.0 |
| After mild | 1261.3 |
| After moderate | 1406.2 |
| After severe | 1322.9 |

Takeaway: clean remains stable across the sequential run.

## Slide 7: Continual Metrics

Use `runs/professor_ready/fig8_cl_metrics.png`.

| Metric | Lifelong |
|--------|---------:|
| Average reward | 393.1 |
| BWT | +15.4 |
| FWT | +17.2 |
| Remembering | 1.0 |

Takeaway: positive BWT/FWT and full remembering support the no-forgetting claim, but final average reward is slightly below the frozen reference.

## Slide 8: Limitations

- Single seed only.
- Two drones only.
- Severe reward remains low.
- Confidence trigger is conservative.
- Multi-seed validation and stronger adaptation triggers are the next steps.

## Final Claim

This project demonstrates a complete pipeline for evaluating confidence-triggered lifelong adaptation in drone swarms, including a continual-learning matrix that re-tests clean performance after later adaptation phases. The saved results show clean retention and partial robustness under surprise, with clear limitations for future work.
