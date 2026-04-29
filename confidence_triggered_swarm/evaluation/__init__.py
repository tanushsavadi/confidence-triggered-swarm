"""Evaluation modules for benchmarking drone swarm performance.

`Evaluator` is imported lazily so lightweight helpers (e.g. continual_metrics) can be
used without pulling in PyBullet / gym_pybullet_drones.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Evaluator"]


def __getattr__(name: str) -> Any:
    if name == "Evaluator":
        from confidence_triggered_swarm.evaluation.evaluator import Evaluator

        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
