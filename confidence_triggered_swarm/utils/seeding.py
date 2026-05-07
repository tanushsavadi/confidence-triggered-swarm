"""Shared seeding helpers for reproducible experiment runs."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_global_seeds(seed: int | None) -> None:
    """Seed Python, NumPy, and Torch when a seed is provided."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def episode_seed(
    base_seed: int | None,
    episode_idx: int,
    stream: int = 0,
) -> int | None:
    """Derive a stable per-episode seed from a run seed and stream id."""
    if base_seed is None:
        return None
    return int(base_seed + stream * 100_000 + episode_idx)


def reset_env(
    env: Any,
    base_seed: int | None,
    episode_idx: int,
    stream: int = 0,
) -> tuple[Any, dict]:
    """Reset an environment with a derived per-episode seed."""
    seed = episode_seed(base_seed, episode_idx, stream)
    if seed is None:
        return env.reset()
    return env.reset(seed=seed)
