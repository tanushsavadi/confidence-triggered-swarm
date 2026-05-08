# Docs

This folder contains explanation and handoff material. It is meant to help a
reader understand the project without reading every source file.

## Recommended Reading Order

1. `../README.md` for the full first-time project overview.
2. `artifact_guide.md` for final report assets, slide assets, and exact numbers
   to quote.
3. `professor_brief.md` for the concise scientific interpretation.
4. `professor_demo_notes.md` for a presentation script and commands.
5. `final_submission_audit.md` for the final-submission readiness scan.
6. `final_push_checklist.md` for the final repo/submission push state.

## File Index

| File | Purpose |
|------|---------|
| `artifact_guide.md` | Final validation JSONs, figure map, slide-deck asset guide, and exact numbers to quote. |
| `professor_brief.md` | Short status summary with final validation, professor feedback response, and limitations. |
| `professor_demo_notes.md` | Demo flow, talk script, and reproduction commands. |
| `continual_presentation_outline.md` | Slide-ready continual-learning outline focused on `train_lifelong` vs `train_continual`. |
| `final_submission_audit.md` | Course-rubric readiness audit and final known caveats. |
| `final_push_checklist.md` | Final repository push/submission checklist and presentation timing guardrails. |
| `tuning_notes.md` | Exploratory adaptation-trigger sweep results and interpretation. |
| `2403.05175v1.pdf` | van de Ven et al., *Continual Learning and Catastrophic Forgetting* reference PDF. |

## Canonical Artifact Rule

Use these for final report and slides:

- `runs/extension/validation/aggregate_summary.json` for the final report's
  main validation table.
- `runs/extension/validation/diagnostic_reward_retention.pdf` for the final
  report's reward-recovery/clean-retention figure.
- `runs/full_eval/evaluation_results.json` for supporting seed-42 fig1, fig2,
  and fig4.
- `runs/ablations/ablation_results.json` for fig3.
- `runs/continual_run/continual_results.json` for fig5 through fig8.
- `runs/professor_ready/` for generated PNG/PDF figures.

Treat `runs/professor_ready/` as presentation support and
`runs/extension/validation/` as the final report's main empirical source.

Final report and timed presentation prep live in `../final_submission/`.
