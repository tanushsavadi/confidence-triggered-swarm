> **Note**: This was our initial design document written before implementation began.
> Some details changed during development — see the main README for the final state.

# Experiment Plan: Confidence-Triggered Lifelong Adaptation for Drone Swarms

## 1. Architecture Decisions and Rationale

### 1.1 Why Custom IPPO (not SB3 PPO or MAPPO)

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **SB3 PPO** | Mature, well-tested | Treats multi-drone as single agent; no per-drone confidence hooks | ❌ Rejected |
| **MAPPO** | True multi-agent; shared critic | Complex integration with gym-pybullet-drones; heavy codebase | ❌ Future work |
| **Custom IPPO** | Full control over forward pass; easy MC dropout insertion; per-drone confidence | Requires implementing PPO from scratch | ✅ Selected |

**Rationale:** We need to insert Monte Carlo dropout into the policy's forward pass and extract per-drone epistemic uncertainty at every step. SB3's PPO abstracts away the forward pass, making confidence hooks difficult. MAPPO's codebase is too heavy for our scope. A lightweight custom IPPO (~300 lines) gives us full control.

### 1.2 Confidence Estimation via MC Dropout

- **Method:** Enable dropout at inference time, run `mc_samples=10` forward passes, compute variance of action means
- **Metric:** `confidence = 1 / (1 + mean_variance)` ∈ (0, 1]
- **Threshold:** When confidence < 0.5, trigger adaptation
- **Why MC Dropout:** Simple to implement, no architectural changes needed, well-established in uncertainty estimation literature

### 1.3 Lifelong Adaptation with EWC

- **Problem:** Catastrophic forgetting when adapting to new conditions
- **Solution:** Elastic Weight Consolidation (EWC) penalizes changes to weights important for previous tasks
- **Fisher Information:** Computed from rollout data after each training phase
- **Lambda:** 1000 (high to strongly protect learned skills)

### 1.4 Environment Design

- **Base:** `BaseRLAviary` from gym-pybullet-drones v2.0
- **Custom `FormationAviary`:** Extends BaseRLAviary with formation reward (inter-drone distance maintenance + hover accuracy)
- **`SurpriseWrapper`:** Gymnasium wrapper that injects distributional shifts:
  - Wind forces via `p.applyExternalForce()`
  - Sensor noise via observation perturbation
  - Actuator degradation via KF scaling
  - Goal position shifts

### 1.5 Observation and Action Spaces

- **Observation per drone:** 12D (pos[3], rpy[3], vel[3], ang_vel[3]) + action buffer
- **Action per drone:** 4D RPM commands, range [-1, 1], mapped to `HOVER_RPM * (1 + 0.05 * action)`
- **IPPO treats each drone independently:** Same policy network, separate observations
- **Physics:** `PYB_GND_DRAG_DW` for realistic ground effect and downwash

---

## 2. Exact File Structure

```
confidence_triggered_swarm/
├── __init__.py                   # Package root, version info
├── configs/
│   ├── __init__.py               # Config loading utilities
│   └── default.yaml              # All hyperparameters
├── envs/
│   ├── __init__.py               # Expose FormationAviary, SurpriseWrapper
│   ├── formation_aviary.py       # FormationAviary(BaseRLAviary) - custom reward
│   └── surprise_wrapper.py       # SurpriseWrapper(gymnasium.Wrapper) - perturbations
├── algorithms/
│   ├── __init__.py               # Expose PPOAgent, ActorCritic, RolloutBuffer
│   ├── ppo.py                    # PPOAgent: collect rollouts, compute GAE, update
│   ├── policy.py                 # ActorCritic: MLP + MC dropout + value head
│   └── buffer.py                 # RolloutBuffer: store transitions, compute returns
├── adaptation/
│   ├── __init__.py               # Expose ConfidenceMonitor, EWCRegularizer, LifelongTrainer
│   ├── confidence.py             # ConfidenceMonitor: MC dropout uncertainty estimation
│   ├── ewc.py                    # EWCRegularizer: Fisher information + penalty
│   └── lifelong_trainer.py       # LifelongTrainer: orchestrates adaptation loop
├── evaluation/
│   ├── __init__.py               # Expose Evaluator
│   └── evaluator.py              # Evaluator: run episodes, collect metrics, compare
├── utils/
│   ├── __init__.py               # Expose MetricsLogger
│   └── logger.py                 # MetricsLogger: TensorBoard + CSV logging
├── scripts/
│   ├── train_baseline.py         # Train IPPO baseline (no adaptation)
│   ├── train_lifelong.py         # Train with confidence-triggered adaptation
│   ├── evaluate.py               # Full evaluation across surprise levels
│   └── test_env.py               # Minimal environment smoke test
├── README.md                     # Setup, usage, reproduction guide
└── requirements.txt              # Python dependencies
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `configs/` | YAML loading, hyperparameter management |
| `envs/` | Gymnasium environments, reward shaping, perturbation injection |
| `algorithms/` | Core IPPO implementation (PPO update, policy network, replay buffer) |
| `adaptation/` | Confidence monitoring, EWC regularization, lifelong training loop |
| `evaluation/` | Standardized evaluation, metrics collection, comparison |
| `utils/` | Logging, visualization helpers |
| `scripts/` | Entry points for training, evaluation, testing |

---

## 3. Implementation Phases

### Phase 1: Foundation (Days 1–2)
- [x] Set up project skeleton with all stubs
- [ ] Implement `FormationAviary` with formation + hover reward
- [ ] Implement `SurpriseWrapper` with wind injection
- [ ] Implement `test_env.py` and verify env runs

**Deliverable:** Environment that resets, steps, renders, and applies wind forces.

### Phase 2: IPPO Baseline (Days 3–5)
- [ ] Implement `ActorCritic` network (MLP with tanh, MC dropout layers)
- [ ] Implement `RolloutBuffer` (store obs, actions, rewards, values, log_probs)
- [ ] Implement `PPOAgent` (rollout collection, GAE computation, PPO update)
- [ ] Implement `MetricsLogger` (TensorBoard integration)
- [ ] Train baseline on clean FormationAviary (2 drones, hover + formation)

**Deliverable:** Trained baseline that achieves stable 2-drone formation hover.

### Phase 3: Confidence Monitoring (Days 6–7)
- [ ] Implement MC dropout forward pass in `ActorCritic`
- [ ] Implement `ConfidenceMonitor` (track per-drone confidence over time)
- [ ] Verify confidence drops when surprise is injected
- [ ] Tune `mc_samples` and `confidence_threshold`

**Deliverable:** Working confidence estimation that reliably detects distribution shifts.

### Phase 4: Lifelong Adaptation (Days 8–10)
- [ ] Implement `EWCRegularizer` (Fisher diagonal computation, penalty term)
- [ ] Implement `LifelongTrainer` (confidence trigger → adaptation episodes → EWC update)
- [ ] Implement experience replay buffer for adaptation
- [ ] Train lifelong system across progressive surprise levels

**Deliverable:** System that adapts to new conditions while preserving old skills.

### Phase 5: Evaluation & Analysis (Days 11–13)
- [ ] Implement `Evaluator` (multi-condition evaluation, metrics aggregation)
- [ ] Run full experiment suite (see Section 4)
- [ ] Generate comparison plots (baseline vs. lifelong)
- [ ] Statistical analysis (mean ± std over seeds)

**Deliverable:** Complete experimental results with plots and analysis.

### Phase 6: Documentation & Polish (Day 14)
- [ ] Write final README with reproduction instructions
- [ ] Clean up code, add final docstrings
- [ ] Package results for submission

---

## 4. Experiments

### 4.1 Experiment 1: Baseline IPPO Training
- **Goal:** Establish baseline performance on clean environment
- **Setup:** 2 drones, FormationAviary, no perturbations
- **Training:** 1M timesteps, 3 random seeds
- **Metrics:** Episode reward, formation error (mean inter-drone distance deviation), hover error (mean altitude deviation)
- **Expected:** Converges to stable formation within 500K steps

### 4.2 Experiment 2: Confidence Calibration
- **Goal:** Verify confidence metric correlates with surprise
- **Setup:** Load trained baseline, evaluate on 4 surprise levels
- **Surprise levels:**
  - `clean`: No perturbations
  - `mild`: wind=[0.0, 0.02], noise_std=0.01
  - `moderate`: wind=[0.0, 0.05], noise_std=0.05, actuator=0.9
  - `severe`: wind=[0.0, 0.1], noise_std=0.1, actuator=0.7, goal_shift=0.3
- **Metrics:** Mean confidence per surprise level, confidence distribution plots
- **Expected:** Monotonically decreasing confidence with increasing surprise

### 4.3 Experiment 3: Adaptation Without EWC (Ablation)
- **Goal:** Show catastrophic forgetting when adapting without EWC
- **Setup:** Train baseline → adapt to `moderate` surprise → evaluate on `clean`
- **Metrics:** Pre/post adaptation performance on both clean and moderate
- **Expected:** Performance on `moderate` improves, but `clean` performance degrades significantly

### 4.4 Experiment 4: Full Lifelong Adaptation (Main Result)
- **Goal:** Demonstrate confidence-triggered adaptation with EWC
- **Setup:** Sequential phases:
  1. Train on `clean` (1M steps)
  2. Expose to `mild`, adapt when confidence drops (confidence threshold=0.5)
  3. Expose to `moderate`, adapt again
  4. Evaluate on all conditions
- **Metrics:** Performance retention on old conditions + adaptation to new ones
- **Expected:** Maintains >80% of clean performance while adapting to new conditions

### 4.5 Experiment 5: Ablation Studies
- **Goal:** Understand contribution of each component
- **Ablations:**
  - No confidence trigger (continuous adaptation) vs. triggered
  - EWC lambda sweep: [0, 100, 1000, 10000]
  - MC dropout samples: [1, 5, 10, 20]
  - Confidence threshold: [0.3, 0.5, 0.7]
- **Metrics:** Same as Experiment 4 for each ablation
- **Expected:** Moderate EWC lambda and threshold provide best trade-off

### Experiment Execution Order
1. **Exp 1** → validates environment and training pipeline
2. **Exp 2** → validates confidence estimation
3. **Exp 3** → motivates EWC (shows the problem)
4. **Exp 4** → main result
5. **Exp 5** → ablations (if time permits)

---

## 5. Expected Timeline and Effort Estimates

| Phase | Days | Hours | Dependencies |
|-------|------|-------|-------------|
| Phase 1: Foundation | 1–2 | 8–10h | None |
| Phase 2: IPPO Baseline | 3–5 | 12–15h | Phase 1 |
| Phase 3: Confidence | 6–7 | 6–8h | Phase 2 |
| Phase 4: Adaptation | 8–10 | 10–12h | Phase 3 |
| Phase 5: Evaluation | 11–13 | 10–12h | Phase 4 |
| Phase 6: Documentation | 14 | 4–6h | Phase 5 |
| **Total** | **14 days** | **50–63h** | |

### Compute Requirements
- **Training:** GPU recommended (CUDA), CPU feasible but slower
- **Per training run:** ~30 min (1M steps on GPU), ~2h (CPU)
- **Full experiment suite:** ~6h GPU / ~24h CPU
- **Storage:** ~500MB for checkpoints + logs per experiment

### Risk Mitigation
| Risk | Mitigation |
|------|------------|
| Formation reward doesn't converge | Fall back to hover-only (MultiHoverAviary) |
| MC dropout confidence not discriminative | Try ensemble disagreement instead |
| EWC lambda too sensitive | Grid search over [100, 500, 1000, 5000] |
| Training too slow on CPU | Reduce episode_len_sec to 4, num_drones to 2 |

---

## 6. Key Design Decisions Summary

1. **IPPO over centralized PPO:** Per-drone confidence requires per-drone policy evaluation
2. **MC Dropout over ensembles:** Simpler, single network, no extra memory
3. **EWC over replay-only:** Provides principled forgetting prevention with theoretical backing
4. **Gymnasium wrapper for surprise:** Clean separation of concerns; env doesn't know about perturbations
5. **YAML config:** Single source of truth for all hyperparameters; easy to sweep
6. **Phased training:** Progressive difficulty mimics real deployment scenario
