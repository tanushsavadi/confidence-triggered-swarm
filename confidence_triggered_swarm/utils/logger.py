from __future__ import annotations

"""Metrics logging — TensorBoard + CSV.

Unified interface for training/eval metrics with TB scalars and CSV export.
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class MetricsLogger:
    """Logger with TensorBoard and CSV support.

    Logs scalars to TB for real-time monitoring and accumulates rows for CSV.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "default",
        use_tensorboard: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_tensorboard = use_tensorboard

        # TB writer
        self._writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("[WARNING] TensorBoard not available, disabling TB logging")
                self.use_tensorboard = False

        # CSV stuff
        self._csv_path = self.log_dir / "metrics.csv"
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields: List[str] = []

        # in-memory history
        self._history: List[Dict[str, Any]] = []

    # -- TB helpers --

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar to TensorBoard."""
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def log_scalars(
        self, metrics: Dict[str, float], step: int, prefix: str = ""
    ) -> None:
        """Log multiple scalars at once."""
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.log_scalar(tag, value, step)

    # -- CSV helpers --

    def _ensure_csv(self, row: Dict[str, Any]) -> None:
        """Lazily create the CSV file on the first row."""
        if self._csv_writer is None:
            self._csv_fields = list(row.keys())
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_fields, extrasaction="ignore"
            )
            self._csv_writer.writeheader()

    def log_to_csv(self, row: Dict[str, Any]) -> None:
        """Append a row to the CSV log."""
        self._ensure_csv(row)
        self._csv_writer.writerow(row)  # type: ignore[union-attr]
        self._csv_file.flush()  # type: ignore[union-attr]

    # -- high-level logging --

    def log_training_step(self, stats: Dict[str, Any], global_step: int) -> None:
        """Log training metrics to TB and CSV."""
        numeric_stats = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
        self.log_scalars(numeric_stats, global_step, prefix="train")

        row = {"global_step": global_step, **numeric_stats}
        self.log_to_csv(row)
        self._history.append(row)

    def log_evaluation(self, eval_results: Dict[str, Any], global_step: int) -> None:
        """Log eval results to TB."""
        numeric = {k: v for k, v in eval_results.items() if isinstance(v, (int, float))}
        self.log_scalars(numeric, global_step, prefix="eval")

    def save_csv(self, filename: str = "metrics.csv") -> None:
        """Save accumulated history to a new CSV file."""
        if not self._history:
            return
        path = self.log_dir / filename
        fields = list(self._history[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._history)

    # -- lifecycle --

    def close(self) -> None:
        """Close file handles."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
