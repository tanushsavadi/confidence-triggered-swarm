# Professor-Ready Figures

This folder contains generated figures for the final report and slide deck.
Each figure is saved as both PNG and PDF.

Use PNGs for slides. Use PDFs for the NeurIPS-style LaTeX report.

## Figure Map

| Figure | Files | Source data | Use |
|---|---|---|---|
| Fig 1 | `fig1_frozen_vs_lifelong.*` | `runs/full_eval/evaluation_results.json` | Main frozen vs lifelong result |
| Fig 2 | `fig2_degradation.*` | `runs/full_eval/evaluation_results.json` | Clean-trained policy degradation under surprise |
| Fig 3 | `fig3_ablations.*` | `runs/ablations/ablation_results.json` | Ablation backup slide |
| Fig 4 | `fig4_forgetting.*` | `runs/full_eval/evaluation_results.json` | Clean forgetting check |
| Fig 5 | `fig5_training_over_time.*` | `runs/continual_run/continual_results.json` | Sequential reward over time |
| Fig 6 | `fig6_continual_matrix.*` | `runs/continual_run/continual_results.json` | Key continual-learning matrix |
| Fig 7 | `fig7_clean_retention.*` | `runs/continual_run/continual_results.json` | Clean retention after each phase |
| Fig 8 | `fig8_cl_metrics.*` | `runs/continual_run/continual_results.json` | Continual-learning metrics |

## Regenerate

From the repository root:

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.generate_plots \
  --evaluation-results runs/full_eval/evaluation_results.json \
  --ablation-results runs/ablations/ablation_results.json \
  --continual-results runs/continual_run/continual_results.json \
  --output-dir runs/professor_ready
```

Do not mix these figures with older draft figures in `runs/final_plots/` unless
the labels and report numbers are updated.
