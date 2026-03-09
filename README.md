# Confidence-Triggered Swarm Adaptation

Lifelong learning for drone swarms under distributional shift.

## What This Is

CS 690NN (Neural Networks) final project. We train a pair of drones to fly in formation using IPPO, then throw surprises at them after training — wind gusts, sensor noise, actuator failures, shifted goals. The system monitors its own confidence and adapts between episodes when things go wrong, without forgetting how to fly clean.

## Quick Results

| Condition | Frozen Reward | Adapted Reward | Recovery |
|-----------|--------------|----------------|----------|
| Clean     | -6.53        | -6.53          | —        |
| Mild      | -6.95        | -6.61          | +83%     |
| Moderate  | -8.85        | -7.22          | +70%     |
| Severe    | -12.41       | -8.87          | +60%     |

**Forgetting check:** post-adaptation clean performance = -6.60 vs baseline -6.53. That's only 1% degradation — the policy doesn't forget how to fly clean after adapting to surprises.

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
├── evaluation/      # systematic eval harness
├── configs/         # YAML hyperparameters
├── scripts/         # training, eval, plotting, diagnostics
└── utils/           # logging, shared helpers

runs/                # saved models + experiment results
docs/                # initial design doc
_research/           # reference MAPPO codebase (not used in our code)
gym-pybullet-drones-install/  # local copy of PyBullet drones lib
```

See [`confidence_triggered_swarm/README.md`](confidence_triggered_swarm/README.md) for detailed module docs.

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
    --total-timesteps 500000 --seed 42 --output-dir runs/baseline

# Evaluate frozen vs adapted across severity levels
python -m confidence_triggered_swarm.scripts.train_lifelong \
    --model runs/baseline/best_model.pt --output-dir runs/lifelong_eval

# Generate final plots
python -m confidence_triggered_swarm.scripts.generate_plots
```

We used seed=42 for all experiments. Training takes ~20 min on CPU (MacBook Pro M-series).

## Figures

Final figures live in `runs/final_plots/` (PNG + PDF):

- **fig1** — Frozen vs adapted performance across severity levels (grouped bar chart)
- **fig2** — Degradation curves showing how severity impacts rewards
- **fig3** — Ablation study: what happens when you remove KL, replay, or EWC
- **fig4** — Forgetting check: clean performance stays stable after adaptation

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
