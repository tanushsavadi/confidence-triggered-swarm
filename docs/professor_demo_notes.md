# Professor Demo Notes

## Recommended Demo Flow

1. Start with the task: two drones must follow waypoints while maintaining formation.
2. Show that the baseline is trained only on clean conditions.
3. Show the surprise suite: wind, sensor noise, actuator weakness, and goal shift.
4. Show final validation with `runs/extension/validation/diagnostic_reward_retention.pdf`, or frozen degradation with `fig2_degradation.png` if presenting from the original deck.
5. Explain the lifelong system: confidence trigger, reward-weighted update, KL anchor, clean replay, EWC.
6. Show frozen vs lifelong with `fig1_frozen_vs_lifelong.png`.
7. Show the forgetting check with `fig4_forgetting.png`.
8. Address the professor feedback directly with `fig6_continual_matrix.png` and `fig7_clean_retention.png`.
9. Close with limitations and future work.

## Main Talking Points

- The clean-trained policy performs well in the clean environment but collapses under even mild distribution shift.
- The adaptation system is deliberately conservative: it adapts between episodes and uses safeguards to avoid overwriting clean-task behavior.
- The final validation improves mild and severe reward for adaptive variants, but every adaptive method is worse than frozen on moderate.
- The key addition after feedback is the sequential continual-learning run.
- In that run, clean is evaluated after every later phase, so we directly check whether adaptation causes catastrophic forgetting.
- Clean performance remains stable across the sequence, and BWT/FWT are positive.

## Figures In Order

1. `../runs/professor_ready/fig2_degradation.png`
2. `../runs/professor_ready/fig1_frozen_vs_lifelong.png`
3. `../runs/professor_ready/fig4_forgetting.png`
4. `../runs/professor_ready/fig5_training_over_time.png`
5. `../runs/professor_ready/fig6_continual_matrix.png`
6. `../runs/professor_ready/fig7_clean_retention.png`
7. `../runs/professor_ready/fig8_cl_metrics.png`

## Short Script

"We trained a shared PPO policy for two drones in a clean formation-flight environment. Then we tested it under post-training surprises: wind, sensor noise, actuator weakness, and shifted goals. In the final validation, the frozen clean policy is brittle, dropping from about 2012 reward on clean to about 200 on mild and 30 on severe.

To respond, we built a confidence-triggered lifelong adaptation loop. The policy monitors entropy and MC-dropout variance; when confidence is low and the episode is usable, it performs between-episode fine-tuning. To reduce forgetting, we anchor the policy with a KL penalty, mix in clean replay, and add EWC.

In the final validation, adaptive variants improve mild and severe reward while mostly retaining clean reward, but moderate is a clear failure case. We also check clean after severe adaptation and do not observe catastrophic forgetting in the supporting seed-42 diagnostic.

The feedback asked for a true continual-learning view, so we added a sequential run: clean to mild to moderate to severe. After each phase, we re-evaluate every condition. This matrix is the key result: clean is explicitly retested after mild, moderate, and severe. Clean reward stays in the same range, and the continual metrics show positive BWT and FWT in this seed. The honest conclusion is that we show clean retention and partial robustness, but not reliable task success."

## Commands For Reproduction

Regenerate presentation figures from saved JSON:

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.generate_plots \
  --evaluation-results runs/full_eval/evaluation_results.json \
  --ablation-results runs/ablations/ablation_results.json \
  --continual-results runs/continual_run/continual_results.json \
  --output-dir runs/professor_ready
```

Run the sequential continual experiment again, if needed:

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.train_continual \
  --baseline-path runs/baseline/best_model.pt \
  --save-dir runs/continual_run
```

The second command is expensive compared with figure generation. The saved canonical results are already present.

Regenerate the final validation summary and diagnostic figure from saved outputs:

```bash
MPLCONFIGDIR=/private/tmp/mplcache ./.venv310/bin/python -m confidence_triggered_swarm.scripts.run_extension_experiments \
  summarize \
  --save-root runs/extension
```
