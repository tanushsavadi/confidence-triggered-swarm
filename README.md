# Confidence-Triggered Swarm Adaptation

Lifelong learning for drone swarms under distributional shift.

## What This Is

CS 690NN (Neural Networks) final project. We train a pair of drones to fly in formation using IPPO, then throw surprises at them after training — wind gusts, sensor noise, actuator failures, shifted goals. The system monitors its own confidence and adapts between episodes when things go wrong, without forgetting how to fly clean.

## Current Repo Status

This repository already contains:

- a trained baseline checkpoint and rollout logs in `runs/baseline/`
- saved frozen/lifelong evaluation artifacts in `runs/full_eval/` and `runs/lifelong_eval/`
- saved ablation results in `runs/ablations/`
- a professor-facing summary in `docs/professor_brief.md`
- demo/talk notes in `docs/professor_demo_notes.md`
- **canonical** sequential continual-learning results in **`runs/continual_run/continual_results.json`** (50 adapt / 50 eval per phase, seed 42; fig5–8). Older exploratory run: `runs/continual_20260417/`.

**`runs/professor_ready/`** holds PNG/PDF figures generated from the JSONs below (regenerate after changing eval data). Older draft visuals may live under `runs/final_plots/` and should not be mixed with current numbers without relabeling.

Regenerate all eight figures from saved JSON outputs with:

```bash
./.venv310/bin/python -m confidence_triggered_swarm.scripts.generate_plots \
    --evaluation-results runs/full_eval/evaluation_results.json \
    --ablation-results runs/ablations/ablation_results.json \
    --continual-results runs/continual_run/continual_results.json \
    --output-dir runs/professor_ready
```

Omit `--continual-results` if you only want fig1–fig4.

### Artifact lineage (what matches which figures)

| Outputs | Role |
|---------|------|
| `runs/full_eval/evaluation_results.json` | **fig1, fig2, fig4** — `Evaluator` suite (frozen + lifelong per severity + forgetting); refresh with `python -m confidence_triggered_swarm.scripts.evaluate --baseline-path runs/baseline/best_model.pt --save-dir runs/full_eval` |
| `runs/ablations/ablation_results.json` | **fig3** |
| **`runs/continual_run/continual_results.json`** | **fig5–fig8** — sequential continual matrix + metrics |
| `runs/professor_ready/` | Generated PNG/PDF from the command above |

Slide-ready bullets and figure mapping: [`docs/continual_presentation_outline.md`](docs/continual_presentation_outline.md).

## How It Works

1. Train IPPO (shared-weight PPO) on clean formation flight (2 drones, PyBullet)
2. Deploy to surprise environments (wind, noise, weak motors, moved goals)
3. Monitor confidence via entropy + MC dropout variance
4. When confidence drops below threshold, adapt between episodes using:
   - **Reward-weighted policy updates** — learn more from good episodes
   - **KL anchoring** — don't drift too far from the original policy
   - **Clean data replay** — mix in clean experiences so we don't forget
   - **EWC regularization** — protect important weights

## Project Structure

```
confidence_triggered_swarm/
├── algorithms/      # PPO, actor-critic, replay buffer
├── adaptation/      # confidence monitor, EWC, lifelong trainer
├── envs/            # formation aviary + surprise wrapper
├── evaluation/      # systematic eval harness + continual_metrics (BWT/FWT helpers)
├── configs/         # YAML hyperparameters
├── scripts/         # training, eval, plotting, diagnostics
└── utils/           # logging, shared helpers

runs/                # saved models + experiment results
docs/                # design notes, professor brief, presentation outline — see docs/README.md
_research/           # reference MAPPO codebase (not used in our code)
gym-pybullet-drones-install/  # local copy of PyBullet drones lib
```

See [`confidence_triggered_swarm/README.md`](confidence_triggered_swarm/README.md) for detailed module docs. **`REPORT.md`** is the long-form write-up; its opening note explains how to align claims with **`runs/full_eval/`** and **`runs/continual_run/`** JSON.

## Setup & Reproduction

**1. Clone the repo**

```bash
git clone <repo-url>
cd "Final Project"
```

**2. Create environment** (Python 3.10+)

```bash
conda create -n drones python=3.10 -y
conda activate drones
```

**3. Install gym-pybullet-drones**

```bash
cd gym-pybullet-drones-install && pip install -e . && cd ..
```

**4. Install project dependencies**

```bash
pip install -r confidence_triggered_swarm/requirements.txt
```

**5. Run experiments**

```bash
# Train baseline (~20 min on CPU, M-series Mac)
python -m confidence_triggered_swarm.scripts.train_baseline \
    --timesteps 500000 --seed 42 --save-dir runs/baseline

# Evaluate frozen vs adapted across severity levels
python -m confidence_triggered_swarm.scripts.train_lifelong \
    --baseline-path runs/baseline/best_model.pt --save-dir runs/lifelong_eval

# Sequential continual learning: adapt clean→mild→moderate→severe once, re-eval all severities after each phase
python -m confidence_triggered_swarm.scripts.train_continual \
    --baseline-path runs/baseline/best_model.pt --save-dir runs/continual_run

# Generate professor-ready plots from saved JSON outputs
./.venv310/bin/python -m confidence_triggered_swarm.scripts.generate_plots \
    --evaluation-results runs/full_eval/evaluation_results.json \
    --ablation-results runs/ablations/ablation_results.json \
    --continual-results runs/continual_run/continual_results.json \
    --output-dir runs/professor_ready
```

We used seed=42 for all experiments. Training takes ~20 min on CPU (MacBook Pro M-series). In this checked-out workspace, the project virtualenv is `.venv310`; in a fresh environment, replace `./.venv310/bin/python` with the active environment's `python`.

## Figures

Artifact-based figures can be generated into `runs/professor_ready/` (PNG + PDF):

- **fig1** — Frozen vs lifelong reward by severity from saved evaluation JSON
- **fig2** — Degradation curve from saved evaluation JSON
- **fig3** — Ablation study from saved ablation JSON
- **fig4** — Forgetting check from saved evaluation JSON

### Continual learning evaluation (professor feedback)

This follows the diagnostic view in van de Ven, Soures & Kudithipudi, *Continual Learning and Catastrophic Forgetting* ([arXiv:2403.05175](https://arxiv.org/abs/2403.05175), also in `docs/2403.05175v1.pdf`): periodic evaluation, backward/forward transfer, and “training over time” style plots.

`train_lifelong.py` evaluates each severity **independently** from the baseline checkpoint. **`train_continual.py`** runs one **sequential** run (clean → mild → moderate → severe): after each phase it measures mean episode reward on **every** severity, building a matrix `R[i,j]` (performance on task `j` after training through phase `i`). That answers “how is **clean** after adapting on **mild**?” directly. Metrics (GEM-style; Lopez-Paz & Ranzato 2017): **average reward** (last row mean), **backward transfer (BWT)**, **forward transfer (FWT)**, **remembering** — see `confidence_triggered_swarm/evaluation/continual_metrics.py`.

- **fig5** — Per-episode reward across sequential phases (raw + rolling mean); frozen vs lifelong
- **fig6** — Heatmaps of `R_lifelong` and `R_frozen`
- **fig7** — Clean-task reward vs phase (forgetting / retention curve)
- **fig8** — Bar chart of CL metrics (lifelong vs frozen reference; frozen BWT/FWT are 0 by construction)

## Phase 2 Roadmap

Everything below is future work — not implemented yet. These are directions we'd pursue with more time.

**Phase 2A: Multi-seed validation** — Everything so far is seed=42 only. We need 3–5 seeds to get confidence intervals and make sure results aren't a lucky draw.

**Phase 2B: Scale to 4 drones** — We currently run 2 drones. The architecture should handle more, but we haven't tested it. Coordination gets harder with more agents.

**Phase 2C: Confidence-guided peer help** — The interesting extension from our proposal. When one drone's confidence drops, have nearby drones share their experience. This is the main idea we want to explore next.

**Phase 2D: Selective response filtering** — Not all peer advice is useful. If a drone is struggling with wind and another drone shares experience from a calm region, that won't help. We'd add a relevance filter based on state similarity.

**Phase 2E: Compare anti-forgetting methods** — We use EWC + clean replay. Would be good to benchmark against PackNet, progressive nets, or even plain L2 regularization to see what actually matters.

**Phase 2F: Richer surprise suite** — More distribution shifts: partial observability, communication delays, drones dropping out mid-episode. The current surprise wrapper makes this easy to add.

**Phase 2G: Online adaptation** — Right now we only adapt between episodes. Could also do lightweight updates within an episode, though that's trickier to get right without destabilizing the policy.

## Class Context

Built for CS 690NN (Neural Networks). Core themes: confidence evaluation under distribution shift, selective experience filtering, replay-based anti-forgetting, and multi-agent coordination for simulated robotics.
