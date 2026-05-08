# 10-12 Minute Premium Deck Outline

Presentation date: May 4, 2026.

Target duration: 10:30-11:30, with a hard cap of 12 minutes. This outline
matches `confidence_triggered_swarm_premium.pptx`, the current 10-slide deck.

| Time | Slide | Message |
|---:|---|---|
| 0:00-0:55 | 1. Title | Clean-trained PPO is evaluated after training under surprise, then adapted only when confidence drops. |
| 0:55-1:55 | 2. Motivation | Simulator-trained drone policies can fail when wind, sensor noise, actuator weakness, or mission goals shift. |
| 1:55-3:05 | 3. Task and environment | Two CF2X drones, clean `FormationAviary`, 450 control steps, 75 observations per drone, surprises injected only at evaluation/adaptation time. |
| 3:05-4:30 | 4. Method | Entropy plus MC-dropout confidence triggers between-episode reward-weighted updates, protected by KL anchoring, clean replay, and EWC. |
| 4:30-5:45 | 5. Main results | The method gives partial robustness: clean, mild, and severe improve; moderate regresses. |
| 5:45-6:45 | 6. Forgetting analysis | After severe adaptation, clean reward is retained and the evaluator reports `forgetting_detected: false`. |
| 6:45-8:00 | 7. Continual learning | The sequential matrix re-tests clean after every phase; the clean column stays stable. |
| 8:00-9:05 | 8. Ablations | Safeguards reduce variance but can cost peak severe reward; the trigger and data-quality gate are still bottlenecks. |
| 9:05-9:45 | 9. Author contributions | Keep this brief and factual; clarify implementation/experiments versus proposal/report support. |
| 9:45-11:10 | 10. Conclusion | Summarize the full pipeline, honest limitations, and future work. |

## Must-Say Lines

- "The baseline was trained on clean episodes only; surprises were introduced
  after training."
- "The result is partial robustness, not a universal win."
- "The slide figures are seed-42 diagnostics; the final written report adds
  three controlled evaluation seeds for the main validation table."
- "The clean-after-adaptation checks are the key evidence against catastrophic
  forgetting in this setting."

## Timing Guardrails

- Do not spend more than 70 seconds on slide 3.
- Do not read every table cell on slides 5 and 7; state the pattern and point to
  the one or two values that matter.
- If running late at slide 8, compress ablations to one sentence: safeguards
  trade peak reward for stability, and the remaining bottleneck is trigger/data
  quality.
- Keep slide 9 under 40 seconds.
