# 10-12 Minute Presentation Outline

Presentation date: May 4, 2026.

Target duration: about 11 minutes, with a hard cap of 12 minutes. The deck has
12 slides so each slide should average 50-60 seconds. If time is running long,
compress slides 8-10 and keep the final claim intact.

| Time | Slide | Visual | Message |
|---:|---|---|---|
| 0:00-0:45 | Title | Thesis panel | Clean-trained PPO is tested after training under surprise, then adapted with a confidence trigger. |
| 0:45-1:40 | Project arc | Pipeline and setup values | The project built the full clean-train, surprise-test, adapt-and-audit loop. |
| 1:40-2:35 | Frozen brittleness | `fig2_degradation.png` | Clean training alone collapses under mild, moderate, and severe deployment shift. |
| 2:35-3:35 | Adaptation loop | Four-step method diagram | Confidence uses entropy plus MC-dropout variance; updates happen between episodes with KL, clean replay, and EWC. |
| 3:35-4:35 | Main result | `fig1_frozen_vs_lifelong.png` | Lifelong improves clean, mild, and severe, while moderate is worse; this is partial robustness. |
| 4:35-5:20 | Forgetting check | `fig4_forgetting.png` | After severe adaptation, clean reward is retained and `forgetting_detected=false`. |
| 5:20-6:25 | Continual matrix | `fig6_continual_matrix.png` | Clean was re-tested after each later phase, directly answering the clean-after-adaptation question. |
| 6:25-7:20 | Ablations | `fig3_ablations.png` | Safeguards affect the severe-surprise tradeoff; the trigger/objective remain bottlenecks. |
| 7:20-8:10 | Training over time | `fig5_training_over_time.png` | Sequential traces show recovery is noisy and severe remains hard. |
| 8:10-9:05 | CL metrics | `fig8_cl_metrics.png` | Backward/forward transfer are positive, remembering is 1.0, but final average reward trails frozen. |
| 9:05-10:05 | Work completed | Artifact map | Summarize implementation, evaluation, generated figures, report draft, deck, and readiness checker. |
| 10:05-11:10 | Closing claim | Final metrics and future work | The contribution is clean retention plus partial robustness, with multi-seed and stronger objectives as future work. |

## Required Phrasing

- Novelty: the integrated post-training surprise benchmark plus
  confidence-triggered lifelong adaptation and explicit clean-after-surprise
  forgetting audits.
- Training: baseline was trained clean only; surprises were not used during
  baseline training.
- Results: quote the main changes exactly: clean +4.1%, mild +50.8%, moderate
  -8.5%, severe +54.7%.
- Limitation: all canonical results are seed 42 only.
- Team contributions: add one spoken sentence matching the final report
  contribution section after the team fills it in.

## Timing Guardrails

- Do not spend more than 2 minutes total on environment values and defaults.
- Do not read every number on the plots; state the one-sentence claim for each
  plot and point to the key number.
- If the talk reaches slide 8 after 7 minutes, skip detailed ablation values and
  say the takeaway: safeguards matter, but trigger/objective quality is the
  remaining bottleneck.
- Keep slide 12 under 65 seconds so the talk ends before 12 minutes.
