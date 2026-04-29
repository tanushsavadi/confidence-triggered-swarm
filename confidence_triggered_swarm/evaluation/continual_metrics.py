"""Continual learning diagnostic metrics from reward matrices.

Implements metrics aligned with Lopez-Paz & Ranzato (GEM, NeurIPS 2017) and
Díaz-Rodríguez et al. (NeurIPS Workshop 2018), as summarized in van de Ven et al.
(Continual Learning and Catastrophic Forgetting, arXiv:2403.05175).

Let R[i, j] be mean episode reward on eval task j after completing training phase i
(phases and tasks share the same ordering, e.g. clean, mild, moderate, severe).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def average_reward(R: np.ndarray) -> float:
    """Mean reward across all tasks after the final training phase.

    Uses the last row of R: performance on every task after learning all phases.
    """
    R = np.asarray(R, dtype=float)
    if R.size == 0:
        return 0.0
    return float(np.mean(R[-1]))


def backward_transfer(R: np.ndarray) -> float:
    """Backward transfer (BWT): influence of later learning on earlier tasks.

    BWT = (1 / (T-1)) * sum_{j=0}^{T-2} ( R[T-1, j] - R[j, j] )

    Negative BWT indicates catastrophic forgetting on earlier tasks.
    """
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    if T < 2:
        return 0.0
    terms: List[float] = []
    for j in range(T - 1):
        terms.append(float(R[T - 1, j] - R[j, j]))
    return float(np.mean(terms)) if terms else 0.0


def forward_transfer(
    R: np.ndarray,
    frozen_baseline: Sequence[float],
) -> float:
    """Forward transfer (FWT): benefit from prior phases before training on task j.

    FWT = mean_{j=1}^{T-1} ( R[j-1, j] - frozen_baseline[j] )

    Compares performance on task j after completing phase j-1 only, vs the
    frozen policy that never adapted (same baseline used for all methods).
    """
    R = np.asarray(R, dtype=float)
    base = np.asarray(frozen_baseline, dtype=float)
    T = R.shape[0]
    if T < 2:
        return 0.0
    terms: List[float] = []
    for j in range(1, T):
        if j < len(base):
            terms.append(float(R[j - 1, j] - base[j]))
    return float(np.mean(terms)) if terms else 0.0


def remembering(
    R: np.ndarray,
    baseline_magnitude: float,
) -> float:
    """Normalized remembering score from backward transfer (higher is better).

    remembering = 1 - |min(BWT, 0)| / (|baseline_magnitude| + eps)

    Clipped to [0, 1]. Use a scale for baseline_magnitude comparable to rewards
    (e.g. mean absolute diagonal of R at phase 0, or mean frozen baseline).
    """
    bwt = backward_transfer(R)
    eps = 1e-8
    denom = abs(float(baseline_magnitude)) + eps
    if bwt >= 0:
        return 1.0
    return float(np.clip(1.0 - abs(bwt) / denom, 0.0, 1.0))


def clean_retention_curve(R: np.ndarray, clean_idx: int = 0) -> np.ndarray:
    """Reward on the clean task after each training phase (column clean_idx).

    Returns shape (T,) with R[i, clean_idx] for i = 0..T-1.
    """
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    if T == 0:
        return np.array([], dtype=float)
    return np.array([float(R[i, clean_idx]) for i in range(T)], dtype=float)


def compute_all_metrics(
    R: np.ndarray,
    frozen_baseline: Sequence[float],
    baseline_magnitude: float | None = None,
) -> dict:
    """Bundle of CL metrics for JSON export."""
    R = np.asarray(R, dtype=float)
    if baseline_magnitude is None:
        baseline_magnitude = float(
            np.mean(np.abs(np.diag(R))) if R.size else 1.0
        )
    return {
        "average_reward": average_reward(R),
        "backward_transfer": backward_transfer(R),
        "forward_transfer": forward_transfer(R, frozen_baseline),
        "remembering": remembering(R, baseline_magnitude),
        "clean_retention": clean_retention_curve(R, clean_idx=0).tolist(),
    }
