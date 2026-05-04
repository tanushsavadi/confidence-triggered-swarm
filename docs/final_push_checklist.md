# Final Push Checklist

Use this before the final repository push/submission for the May 4, 2026
presentation.

## Ready Artifacts

- Root `README.md` explains the project for a first-time reader.
- `docs/artifact_guide.md` maps canonical JSONs to every report/deck figure.
- `runs/professor_ready/` contains the eight generated figures in PNG and PDF.
- `final_submission/confidence_triggered_swarm_premium.pptx` is the editable
  10-slide deck for a 10-12 minute talk.
- `final_submission/slides_10_12min_outline.md` gives the timed talk plan.
- `final_submission/slide_speaker_notes.md` gives the complete rehearsal script.
- `final_submission/final_report.tex` is the NeurIPS-style report draft.
- `confidence_triggered_swarm/scripts/check_final_readiness.py` checks the
  canonical final-submission files.

## Final Manual Items

1. Replace `TODO_AUTHOR_CONTRIBUTIONS` in
   `final_submission/final_report.tex` with the verified division of work.
2. Place the official `neurips_2026.sty` file in `final_submission/` or on the
   TeX path.
3. Compile `final_submission/final_report.tex` and confirm the report is 5-9
   content pages, excluding references.
4. Re-run:

   ```bash
   ./.venv310/bin/python -m confidence_triggered_swarm.scripts.check_final_readiness
   ```

5. Review `git status --short` before staging. The canonical presentation story
   uses `runs/full_eval`, `runs/ablations`, `runs/continual_run`, and
   `runs/professor_ready`; `runs/improved_*` should be included only if the team
   wants to archive the full tuning process in the final push.

## Presentation Timing

The deck is designed to land around 11 minutes:

- Slides 1-3: thesis, motivation, and environment.
- Slides 4-7: adaptation method, main result, forgetting check, and continual
  matrix.
- Slides 8-10: ablations, contributions, conclusion, limitations, and future
  work.

Keep ablations and contributions concise if the talk runs long; do not cut the
forgetting and continual-matrix slides.
