# Final Submission Audit

Last scan: 2026-05-02.

## Repository Scan

- Main package: `confidence_triggered_swarm/` with environments, PPO, adaptation, evaluation, plotting, and scripts.
- Canonical result artifacts: `runs/full_eval/evaluation_results.json`, `runs/continual_run/continual_results.json`, `runs/ablations/ablation_results.json`.
- Presentation figures: `runs/professor_ready/fig1` through `fig8` in both PNG and PDF.
- Presentation deck: `final_submission/confidence_triggered_swarm_premium.pptx` is a 10-slide, 10-12 minute deck for May 4, 2026.
- Existing writeups: `REPORT.md`, `docs/artifact_guide.md`, `docs/professor_brief.md`, `docs/professor_demo_notes.md`, `docs/continual_presentation_outline.md`.
- Reference code: `_research/` and `gym-pybullet-drones-install/`; these are support/reference folders, not the core project implementation.
- Current git state was already dirty before this prep pass, mainly from improved adaptation/tuning changes and generated `runs/improved_*` artifacts.

## MCP/External Checks Used

- DuckDuckGo MCP and web verification found the NeurIPS 2026 formatting instructions and downloads page.
- The current NeurIPS instructions state a nine-page content limit including figures, with references outside the content page count, and identify `neurips_2026.sty` as the supported style file.
- Context7 MCP was used for PyTorch documentation checks on reproducibility and checkpointing; the repo already uses state-dict checkpoints and seeds NumPy/PyTorch in training/evaluation scripts.

## Course Requirement Mapping

| Requirement | Current support | Status |
|---|---|---|
| Choice of project | README, report intro, final report draft | Ready |
| Previous work description | `REPORT.md`, `final_submission/references.bib`, related work draft | Ready |
| Novelty | Final report intro and slides identify the integrated surprise/adaptation/continual matrix contribution | Ready |
| Results | Canonical JSONs and `runs/professor_ready/` figures | Ready |
| Conclusions and future work | `REPORT.md`, final report discussion, slides limitations | Ready |
| Clarity in writing/presenting | Root README, artifact guide, professor brief, demo notes, 10-12 minute outline | Ready |
| Who worked on what | `final_submission/final_report.tex` has a required placeholder | Needs team input |
| NeurIPS format | LaTeX draft uses `neurips_2026`; official style file must be supplied before compile | Needs style file |
| 5-9 pages excluding references | Must be checked after compiling PDF | Needs final check |

## Recommended Final Claims

- Strong claim: the clean-trained policy is brittle under post-training surprise.
- Supported claim: confidence-triggered lifelong adaptation improves mild and severe reward in the canonical seed and maintains clean performance in forgetting probes.
- Careful claim: the sequential matrix shows clean retention and positive BWT/FWT for seed 42.
- Avoid claiming: the lifelong policy universally beats frozen evaluation, because moderate is worse in the main run and final continual average reward is slightly below the frozen reference.

## Remaining Submission TODOs

1. Fill in `TODO_AUTHOR_CONTRIBUTIONS` in `final_submission/final_report.tex`.
2. Place the official `neurips_2026.sty` in `final_submission/` or on the TeX path.
3. Compile `final_submission/final_report.tex` and check the content page count is between 5 and 9.
4. Regenerate figures if any JSON results change.
5. Decide whether improved adaptation runs are only appendix/talk context; current docs recommend not replacing canonical professor-ready results.
