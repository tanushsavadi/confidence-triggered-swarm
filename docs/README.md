# Docs

This folder contains explanation and handoff material. It is meant to help a
reader understand the project without reading every source file.

## Recommended Reading Order

1. `../README.md` for the full first-time project overview.
2. `artifact_guide.md` for slide/report assets and exact numbers to quote.
3. `professor_brief.md` for the concise scientific interpretation.
4. `professor_demo_notes.md` for a presentation script and commands.
5. `final_submission_audit.md` for remaining final-submission tasks.
6. `final_push_checklist.md` before staging or pushing the final repo.

## File Index

| File | Purpose |
|------|---------|
| `artifact_guide.md` | Canonical JSONs, figure map, slide-deck asset guide, and exact numbers to quote. |
| `professor_brief.md` | Short status summary with main results, professor feedback response, and limitations. |
| `professor_demo_notes.md` | Demo flow, talk script, and reproduction commands. |
| `continual_presentation_outline.md` | Slide-ready continual-learning outline focused on `train_lifelong` vs `train_continual`. |
| `final_submission_audit.md` | Course-rubric readiness audit and remaining final-submission TODOs. |
| `final_push_checklist.md` | Final repository push/submission checklist and presentation timing guardrails. |
| `tuning_notes.md` | Exploratory adaptation-trigger sweep results and interpretation. |
| `2403.05175v1.pdf` | van de Ven et al., *Continual Learning and Catastrophic Forgetting* reference PDF. |

## Canonical Artifact Rule

Use these for final report and slides:

- `runs/full_eval/evaluation_results.json` for fig1, fig2, and fig4.
- `runs/ablations/ablation_results.json` for fig3.
- `runs/continual_run/continual_results.json` for fig5 through fig8.
- `runs/professor_ready/` for generated PNG/PDF figures.

Treat `runs/improved_*` and other tuning runs as appendix context unless the
team intentionally updates the final story and regenerates all figures.

Final report and timed presentation prep live in `../final_submission/`.
