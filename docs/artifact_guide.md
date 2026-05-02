# Artifact Guide

This guide is for preparing the final slide deck, report, and demo without
having to reverse-engineer the experiment folders.

## Canonical Story

Use the canonical artifacts unless you intentionally decide to present the
newer tuning runs as an appendix.

The baseline policy was trained on clean formation flight only. Surprises were
introduced after training for evaluation and adaptation.

Recommended final claim:

> Confidence-triggered adaptation shows clean retention and partial robustness
> under post-training surprise. It improves mild and severe surprise in the
> canonical seed, but moderate remains mixed, so the method is not a universal
> performance win.

Avoid saying:

> Lifelong adaptation beats frozen PPO everywhere.

That is not supported by the saved results.

## Must-Have Files

| File or directory | What it contains | Use |
|---|---|---|
| `runs/baseline/best_model.pt` | Clean-trained baseline checkpoint | Reproduction and demos |
| `runs/full_eval/evaluation_results.json` | Frozen/lifelong per-severity results plus forgetting check | Main report tables and fig1, fig2, fig4 |
| `runs/ablations/ablation_results.json` | Severe-surprise safeguard ablations | Fig3 and backup discussion |
| `runs/continual_run/continual_results.json` | Sequential clean-to-severe reward matrix | Fig5 to fig8 |
| `runs/professor_ready/` | Generated PNG/PDF figures plus local README | Slides and report |
| `final_submission/final_report.tex` | NeurIPS-style report draft | Final report |
| `final_submission/confidence_triggered_swarm_final_presentation.pptx` | Editable 12-slide generated deck | Presentation |
| `final_submission/slides_10_12min_outline.md` | Timed 10-12 minute presentation plan | Slide deck build guide |
| `final_submission/slide_speaker_notes.md` | Speaker notes for generated deck | Presentation rehearsal |

## Figure Map

| Figure | File | Source JSON | Best slide use |
|---|---|---|---|
| Fig 1 | `runs/professor_ready/fig1_frozen_vs_lifelong.png` | `runs/full_eval/evaluation_results.json` | Main result: frozen vs lifelong reward |
| Fig 2 | `runs/professor_ready/fig2_degradation.png` | `runs/full_eval/evaluation_results.json` | Problem setup: clean policy degrades under surprise |
| Fig 3 | `runs/professor_ready/fig3_ablations.png` | `runs/ablations/ablation_results.json` | Backup: stability/performance tradeoff in safeguards |
| Fig 4 | `runs/professor_ready/fig4_forgetting.png` | `runs/full_eval/evaluation_results.json` | Forgetting check after severe adaptation |
| Fig 5 | `runs/professor_ready/fig5_training_over_time.png` | `runs/continual_run/continual_results.json` | Sequential reward trajectory |
| Fig 6 | `runs/professor_ready/fig6_continual_matrix.png` | `runs/continual_run/continual_results.json` | Most direct answer to clean-after-adaptation question |
| Fig 7 | `runs/professor_ready/fig7_clean_retention.png` | `runs/continual_run/continual_results.json` | Clean retention over phases |
| Fig 8 | `runs/professor_ready/fig8_cl_metrics.png` | `runs/continual_run/continual_results.json` | Continual-learning metrics summary |

Use PNGs in slides. Use PDFs in the LaTeX report when possible.

## Main Numbers To Quote

From `runs/full_eval/evaluation_results.json`, 50 episodes per condition,
seed 42:

| Severity | Frozen reward | Lifelong reward | Change |
|---|---:|---:|---:|
| clean | 1305.2 | 1358.9 | +4.1% |
| mild | 105.9 | 159.7 | +50.8% |
| moderate | 49.4 | 45.2 | -8.5% |
| severe | 27.3 | 42.2 | +54.7% |

Forgetting check:

| Condition | Clean reward | Waypoints |
|---|---:|---:|
| Baseline on clean | 1308.9 | 0.90 |
| After severe adaptation, evaluated on clean | 1386.1 | 0.86 |

Sequential clean retention:

| Phase completed | Clean reward |
|---|---:|
| after clean | 1316.0 |
| after mild | 1261.3 |
| after moderate | 1406.2 |
| after severe | 1322.9 |

Continual metrics:

| Metric | Lifelong | Frozen reference |
|---|---:|---:|
| Average reward after final phase | 393.1 | 412.9 |
| Backward transfer | +15.4 | 0.0 |
| Forward transfer | +17.2 | 0.0 |
| Remembering | 1.0 | 1.0 |

## Recommended 10-12 Minute Slide Deck

The generated deck is:

`final_submission/confidence_triggered_swarm_final_presentation.pptx`

It is designed for the May 4, 2026, presentation window and targets about
11 minutes.

1. Title and thesis: clean-trained PPO under post-training surprise.
2. Project arc: full clean-train, surprise-test, adapt-and-audit loop.
3. Frozen brittleness: show `fig2_degradation.png`.
4. Adaptation loop: confidence trigger, quality gate, reward-weighted update,
   KL anchor, clean replay, and EWC.
5. Main result: show `fig1_frozen_vs_lifelong.png`.
6. Forgetting check: show `fig4_forgetting.png`.
7. Sequential continual matrix: show `fig6_continual_matrix.png`.
8. Ablations: show `fig3_ablations.png`.
9. Training over time: show `fig5_training_over_time.png`.
10. Continual-learning metrics: show `fig8_cl_metrics.png`.
11. Work completed: implementation, experiments, figures, report, deck, and
    readiness checker.
12. Closing claim: clean retention plus partial robustness; future work.

The timed version is in `final_submission/slides_10_12min_outline.md`.
Speaker notes are in `final_submission/slide_speaker_notes.md`.

## Regenerate Figures

From the repository root:

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.generate_plots \
  --evaluation-results runs/full_eval/evaluation_results.json \
  --ablation-results runs/ablations/ablation_results.json \
  --continual-results runs/continual_run/continual_results.json \
  --output-dir runs/professor_ready
```

## Validate Handoff State

From the repository root:

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.check_final_readiness
```

Expected remaining warnings before final submission:

- `neurips_2026.sty` must be added before compiling the report.
- Author contributions must be filled in.
- The compiled PDF page count must be checked.

## Noncanonical Runs

The `runs/improved_*` folders and improved config files are useful tuning
evidence, but they should not replace the canonical story unless the team
explicitly decides to rerun and update all report numbers. Current notes in
`docs/tuning_notes.md` recommend treating them as appendix context.
