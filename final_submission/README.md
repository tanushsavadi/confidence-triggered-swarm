# Final Submission Package

This folder is the handoff area for the CS 590NN/690NN final report and
presentation. It intentionally points to saved experiment artifacts instead of
duplicating results, so the report, slides, and figures stay aligned.

For a broader artifact map, read `../docs/artifact_guide.md`.

## Files

| File | Purpose |
|---|---|
| `confidence_triggered_swarm_premium.pptx` | Editable 10-slide PowerPoint deck for the final 10-12 minute presentation. |
| `final_report.tex` | NeurIPS-style report draft tied to canonical JSON outputs and figures. |
| `references.bib` | BibTeX entries used by the report. |
| `slides_10_12min_outline.md` | Timed 10-12 minute presentation plan with figure mapping. |
| `slide_speaker_notes.md` | Complete rehearsal script for the premium 10-slide deck. |

## What Still Needs Human Input

1. Add the official `neurips_2026.sty` file to this directory or the TeX path.
2. Replace `TODO_AUTHOR_CONTRIBUTIONS` in `final_report.tex` with the verified
   division of work.
3. Compile the report and confirm it is 5-9 pages including figures, excluding
   references.
4. Rehearse the premium 10-slide deck against the 10-12 minute timing plan.

## Build the Report

The report source uses `\usepackage[preprint]{neurips_2026}`. The official
NeurIPS 2026 instructions say the supported style file is `neurips_2026.sty`;
place that file in this directory or somewhere on the TeX path before compiling.

```bash
cd final_submission
latexmk -pdf final_report.tex
```

The course requires 5-9 pages including figures, excluding references. After
compilation, check the PDF page count manually or with a PDF metadata tool.

## Refresh Figures

From the repository root:

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.generate_plots \
  --evaluation-results runs/full_eval/evaluation_results.json \
  --ablation-results runs/ablations/ablation_results.json \
  --continual-results runs/continual_run/continual_results.json \
  --output-dir runs/professor_ready
```

## Slide Assets

The editable deck is already generated:

```text
final_submission/confidence_triggered_swarm_premium.pptx
```

It is designed for the May 4, 2026 presentation window and should be presented
in about 11 minutes, with a hard cap of 12 minutes. It uses PNG files from
`../runs/professor_ready/` as proof objects:

- `fig1_frozen_vs_lifelong.png` for the main result.
- `fig4_forgetting.png` for clean retention after severe adaptation.
- `fig6_continual_matrix.png` for the key continual-learning answer.
- `fig3_ablations.png` for safeguard/tuning discussion.

Use `fig3_ablations.png`, `fig5_training_over_time.png`, and
`fig7_clean_retention.png` as backup or appendix slides.

Speaker notes are in `slide_speaker_notes.md`. Timing is in
`slides_10_12min_outline.md`.

## Readiness Check

From the repository root:

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.check_final_readiness
```

Known item that still needs team input: replace the `TODO_AUTHOR_CONTRIBUTIONS`
block in `final_report.tex` with the verified division of work.
