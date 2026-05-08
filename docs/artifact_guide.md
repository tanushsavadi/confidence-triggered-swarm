# Artifact Guide

This guide is for preparing the final slide deck, report, and demo without
having to reverse-engineer the experiment folders.

## Canonical Story

The baseline policy was trained on clean formation flight only. Surprises were
introduced after training for evaluation and adaptation.

The final written report uses `runs/extension/validation/aggregate_summary.json`
as the main empirical source. The older seed-42 professor-ready plots remain
useful presentation diagnostics, but they are no longer the primary report
table.

Recommended final claim:

> Confidence-triggered adaptation shows partial reward recovery under
> post-training surprise. In the final three-seed validation, adaptive variants
> improve mild and severe reward and mostly retain clean reward, but all of
> them fail on moderate surprise and success rates stay near zero.

Avoid saying:

> Lifelong adaptation beats frozen PPO everywhere.

That is not supported by the saved results.

## Must-Have Files

| File or directory | What it contains | Use |
|---|---|---|
| `runs/baseline/best_model.pt` | Clean-trained baseline checkpoint | Reproduction and demos |
| `runs/extension/validation/aggregate_summary.json` | Final three-seed validation across frozen/current/always-adapt/improved PPO/reward rescue | Main report table |
| `runs/extension/validation/diagnostic_reward_retention.pdf` | Reward recovery versus clean retention plot | Main report validation figure |
| `runs/full_eval/evaluation_results.json` | Seed-42 frozen/lifelong per-severity results plus forgetting check | Supporting fig1, fig2, fig4 |
| `runs/ablations/ablation_results.json` | Severe-surprise safeguard ablations | Fig3 and backup discussion |
| `runs/continual_run/continual_results.json` | Sequential clean-to-severe reward matrix | Fig5 to fig8 |
| `runs/professor_ready/` | Generated PNG/PDF figures plus local README | Slides and report |
| `final_submission/final_report.tex` | NeurIPS-style report source | Final report |
| `final_submission/final_report.pdf` | Compiled 8-page report | Submission PDF |
| `final_submission/confidence_triggered_swarm_premium.pptx` | Editable 10-slide premium deck | Presentation |
| `final_submission/slides_10_12min_outline.md` | Timed 10-12 minute presentation plan | Slide deck build guide |
| `final_submission/slide_speaker_notes.md` | Complete rehearsal script for premium deck | Presentation rehearsal |

## Figure Map

| Figure | File | Source JSON | Best slide use |
|---|---|---|---|
| Validation | `runs/extension/validation/diagnostic_reward_retention.pdf` | `runs/extension/validation/aggregate_summary.json` | Final report: reward recovery vs clean retention |
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

From `runs/extension/validation/aggregate_summary.json`, three controlled
evaluation seeds and 75 episodes per severity:

| Severity | Frozen | Current | Always-adapt | Improved PPO | Reward rescue |
|---|---:|---:|---:|---:|---:|
| clean | 2011.6 +/- 0.0 | 1972.4 +/- 10.9 | 1971.9 +/- 3.2 | 1988.0 +/- 19.9 | 2004.9 +/- 23.6 |
| mild | 199.8 +/- 16.9 | 235.7 +/- 24.8 | 238.2 +/- 26.4 | 241.5 +/- 34.5 | 233.1 +/- 34.7 |
| moderate | 119.3 +/- 13.3 | 77.4 +/- 10.6 | 77.3 +/- 9.7 | 78.3 +/- 11.0 | 77.7 +/- 9.9 |
| severe | 30.5 +/- 5.6 | 38.6 +/- 3.9 | 38.1 +/- 3.8 | 38.2 +/- 4.0 | 38.3 +/- 4.0 |

Paired seed-level percent changes used in the report:

| Method | Clean | Mild | Moderate | Severe |
|---|---:|---:|---:|---:|
| Current | -1.95% | +19.35% | -31.59% | +40.14% |
| Always-adapt | -1.98% | +20.59% | -31.99% | +38.93% |
| Improved PPO | -1.17% | +22.09% | -30.83% | +39.04% |
| Reward rescue | -0.33% | +17.79% | -31.54% | +39.16% |

Supporting seed-42 diagnostics:

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

`final_submission/confidence_triggered_swarm_premium.pptx`

It is designed for the May 4, 2026, presentation window and targets about
11 minutes.

1. Title and thesis: clean-trained PPO under post-training surprise.
2. Motivation: deployment gap and why unconstrained adaptation is risky.
3. Task and environment: clean `FormationAviary`, two drones, and surprise suite.
4. Adaptation loop: confidence trigger, quality gate, reward-weighted update,
   KL anchor, clean replay, and EWC.
5. Main result: show `fig1_frozen_vs_lifelong.png`.
6. Forgetting check: show `fig4_forgetting.png`.
7. Sequential continual matrix: show `fig6_continual_matrix.png`.
8. Ablations: show `fig3_ablations.png`.
9. Author contributions: factual split of implementation, experiments, and
   proposal/report support.
10. Closing claim: clean retention plus partial robustness; future work.

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

Expected state:

- Submission docs, style file, author contributions, result artifacts, figures,
  and deck are present.
- The only known warning is that the local checker cannot verify PDF page count
  because `pdfinfo` is not installed. The LaTeX log reports an 8-page PDF.

## Noncanonical Runs

The `runs/improved_*` folders and early tuning notes are useful development
context. For final claims, use `runs/extension/validation/aggregate_summary.json`
and the compiled report.
