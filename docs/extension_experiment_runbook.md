# Extension Experiment Runbook

This runbook records the extension pass that strengthened the evidence with
controlled seeds, stronger adaptation comparisons, and aggregate tables. The
final report now uses `runs/extension/validation/aggregate_summary.json` as its
main empirical source.

## Quick Smoke Test

Run this before launching long jobs:

```bash
MPLCONFIGDIR=/private/tmp/mplcache ./.venv310/bin/python -m confidence_triggered_swarm.scripts.run_extension_experiments \
  screening \
  --episodes 1 \
  --seeds 42 \
  --methods frozen,current_default \
  --save-root runs/extension_smoke \
  --baseline-path runs/baseline/best_model.pt \
  --skip-existing
```

Expected outputs:

- `runs/extension_smoke/screening/aggregate_summary.json`
- `runs/extension_smoke/screening/aggregate_summary.md`
- `runs/extension_smoke/screening/diagnostic_reward_retention.png`

## Stage 1-2: Method Screening

This runs the short 25-episode screen across the planned seeds and methods:

```bash
MPLCONFIGDIR=/private/tmp/mplcache ./.venv310/bin/python -m confidence_triggered_swarm.scripts.run_extension_experiments \
  screening \
  --screening-episodes 25 \
  --seeds 42,123,456 \
  --save-root runs/extension \
  --baseline-path runs/baseline/best_model.pt \
  --skip-existing
```

The runner writes `runs/extension/screening/selection.json`. The tuned method is
selected by the fixed rule from the plan: highest mean surprise reward among
candidates with less than 5% clean reward drop and severe variance no worse than
`always_adapt`. If nothing satisfies the rule, the summary says that explicitly.

## Stage 3: Final Validation

After screening finishes, run the final five-method validation:

```bash
MPLCONFIGDIR=/private/tmp/mplcache ./.venv310/bin/python -m confidence_triggered_swarm.scripts.run_extension_experiments \
  validation \
  --validation-episodes 75 \
  --seeds 42,123,456 \
  --methods frozen,current_default,always_adapt,improved_ppo,reward_weighted_rescue \
  --save-root runs/extension \
  --baseline-path runs/baseline/best_model.pt \
  --skip-existing
```

This validates `frozen`, `current_default`, `always_adapt`, `improved_ppo`, and
`reward_weighted_rescue` under the same protocol. The screening rule selected
`improved_ppo`, but the final report also includes `reward_weighted_rescue`
because it is the clean-retaining tuned variant.

## Stage 4: Continual Validation

Run the sequential clean -> mild -> moderate -> severe matrix for the current
method and the tuned method:

```bash
MPLCONFIGDIR=/private/tmp/mplcache ./.venv310/bin/python -m confidence_triggered_swarm.scripts.run_extension_experiments \
  continual \
  --seeds 42 \
  --continual-adapt-episodes 50 \
  --continual-eval-episodes 30 \
  --tuned-method auto \
  --save-root runs/extension \
  --baseline-path runs/baseline/best_model.pt \
  --skip-existing
```

Use three continual seeds only if there is enough time. The final report can
still use one continual matrix as a detailed diagnostic if the main validation
table has three controlled seeds.

## Stage 5: Domain-Randomized Baseline

Train and evaluate one robust-training baseline:

```bash
MPLCONFIGDIR=/private/tmp/mplcache ./.venv310/bin/python -m confidence_triggered_swarm.scripts.run_extension_experiments \
  domain-randomized \
  --domain-timesteps 1000000 \
  --validation-episodes 75 \
  --save-root runs/extension \
  --skip-existing
```

If training is interrupted but a checkpoint exists, pass it back in:

```bash
MPLCONFIGDIR=/private/tmp/mplcache ./.venv310/bin/python -m confidence_triggered_swarm.scripts.run_extension_experiments \
  domain-randomized \
  --domain-baseline-path runs/extension/domain_randomized/seed_42/best_model.pt \
  --validation-episodes 75 \
  --save-root runs/extension \
  --skip-existing
```

## Summaries

Regenerate summaries without rerunning experiments:

```bash
MPLCONFIGDIR=/private/tmp/mplcache ./.venv310/bin/python -m confidence_triggered_swarm.scripts.run_extension_experiments \
  summarize \
  --save-root runs/extension
```

Use `aggregate_summary.json` as the source of truth for report tables. Do not
hand-enter new numbers into the paper from terminal output.
