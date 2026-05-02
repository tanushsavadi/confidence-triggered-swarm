# Slide Speaker Notes

These notes match `confidence_triggered_swarm_final_presentation.pptx`, target
an 11-minute presentation for May 4, 2026, and use only canonical artifacts
from `runs/full_eval`, `runs/ablations`, `runs/continual_run`, and
`runs/professor_ready`.

## 1. Title, 0:00-0:45

We trained a two-drone shared PPO policy on clean formation flight only. The
project asks what happens after training when deployment conditions shift, and
whether a confidence-triggered adaptation layer can recover without forgetting
clean flight. The thesis is careful: retention plus partial robustness, not a
claim that lifelong adaptation wins everywhere.

## 2. Project Arc, 0:45-1:40

The baseline training environment is clean `FormationAviary`: two CF2X drones,
PyBullet physics, velocity control, kinematic observations plus waypoint offset,
240 Hz physics and 30 Hz control. Surprises are introduced only after training.
Across the process we built the environment hooks, the clean PPO baseline, the
surprise wrapper, the confidence monitor, the between-episode adaptation loop,
and the evaluation/plotting package.

## 3. Frozen Degradation, 1:40-2:35

The clean-trained frozen policy is brittle. Clean reward is 1305.2, but mild
surprise drops frozen reward to 105.9 and severe drops it to 27.3. This is the
motivation for post-training adaptation.

## 4. Adaptation Loop, 2:35-3:35

Confidence is computed from policy entropy and MC-dropout variance calibrated on
clean episodes. Adaptation happens between episodes only. Usable episodes pass a
quality gate, then reward-weighted updates are constrained by KL anchoring,
clean replay, and EWC. The key default values are threshold 0.5, window 30, 5
adaptation epochs, adaptation LR 1e-4, EWC lambda 1000, clean replay ratio 0.2,
and KL anchor 0.5.

## 5. Main Result, 3:35-4:35

Lifelong adaptation improves clean by 4.1%, mild by 50.8%, and severe by 54.7%.
Moderate is worse by 8.5%, so the honest claim is partial robustness, not a
universal win.

## 6. Forgetting Check, 4:35-5:20

After severe adaptation, clean reward is 1386.1 compared with 1308.9 before
adaptation. The evaluator reports `forgetting_detected: false`. Waypoints are
slightly lower, so phrase this as retention rather than a perfect improvement.

## 7. Continual Matrix, 5:20-6:25

This is the key slide for the professor feedback. The same lifelong policy is
adapted through clean, mild, moderate, and severe. After each phase, every
severity is re-evaluated. The clean column stays in the same range, so clean was
explicitly tested after adapting to harder conditions.

## 8. Ablation Evidence, 6:25-7:20

The ablation run was done on severe surprise to stress the anti-forgetting
design. Removing a safeguard can raise short-run severe reward, but it also
relaxes the clean-skill guardrails. This is why the final conclusion is not
"delete safeguards"; it is that confidence triggering and adaptation-data
quality need improvement.

## 9. Training Over Time, 7:20-8:10

The sequential training trace shows that this was not only a final aggregate
comparison. We tracked per-episode reward as the lifelong policy moved through
clean, mild, moderate, and severe phases. The high clean rewards and noisy
severe behavior make the core tradeoff visible: adaptation helps some shifts
but severe deployment remains difficult.

## 10. Continual-Learning Metrics, 8:10-9:05

The continual-learning summary is mixed in an informative way. Backward transfer
is +15.4, forward transfer is +17.2, and remembering is 1.0, which supports the
retention story. But final average reward is 393.1, below the frozen reference
of 412.9, so the method is promising but not finished.

## 11. Work Completed, 9:05-10:05

This slide covers the full project process: policy/environment implementation,
surprise definitions, confidence-triggered adaptation, forgetting safeguards,
frozen and lifelong evaluations, ablations, continual-learning metrics,
generated figures, report draft, editable deck, speaker notes, and readiness
checker. Add one sentence naming who worked on which parts once the verified
team contribution text is finalized.

## 12. Closing Claim, 10:05-11:10

The result is clean retention plus partial robustness under post-training
surprise. Limitations are important: one seed, two drones, low severe reward,
and a conservative adaptation trigger. Future work should add multi-seed
validation, larger swarms, better triggers/objectives, and peer-help mechanisms.
