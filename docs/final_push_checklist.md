# Final Push Checklist

Use this before the final repository push/submission.

## Ready Artifacts

- Root `README.md` explains the project for a first-time reader.
- `docs/artifact_guide.md` maps canonical JSONs to every report/deck figure.
- `runs/extension/validation/aggregate_summary.json` is the source of truth for
  the final report validation table.
- `runs/extension/validation/diagnostic_reward_retention.pdf` is the main final
  report validation figure.
- `runs/professor_ready/` contains the eight generated figures in PNG and PDF.
- `final_submission/confidence_triggered_swarm_premium.pptx` is the editable
  10-slide deck for a 10-12 minute talk.
- `final_submission/slides_10_12min_outline.md` gives the timed talk plan.
- `final_submission/slide_speaker_notes.md` gives the complete rehearsal script.
- `final_submission/final_report.tex` is the NeurIPS-style report source.
- `final_submission/final_report.pdf` is compiled and ready for submission.
- `confidence_triggered_swarm/scripts/check_final_readiness.py` checks the
  canonical final-submission files.

## Final Manual Items

1. Re-run:

   ```bash
   ./.venv310/bin/python -m confidence_triggered_swarm.scripts.check_final_readiness
   ```

2. Expected warning: `pdfinfo` is not installed, so the checker cannot verify
   PDF page count automatically. The LaTeX log reports an 8-page PDF.
3. Review `git status --short` before staging. The only expected untracked
   local folder is `.codex/`.
4. Final report claims should use `runs/extension/validation`; the
   `runs/professor_ready` figures are supporting seed-42 presentation
   diagnostics.

## Presentation Timing

The deck is designed to land around 11 minutes:

- Slides 1-3: thesis, motivation, and environment.
- Slides 4-7: adaptation method, main result, forgetting check, and continual
  matrix.
- Slides 8-10: ablations, contributions, conclusion, limitations, and future
  work.

Keep ablations and contributions concise if the talk runs long; do not cut the
forgetting and continual-matrix slides.
