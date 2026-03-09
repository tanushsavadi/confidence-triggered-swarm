"""Adaptation modules for lifelong learning."""

from confidence_triggered_swarm.adaptation.confidence import ConfidenceMonitor
from confidence_triggered_swarm.adaptation.ewc import EWCRegularizer
from confidence_triggered_swarm.adaptation.lifelong_trainer import LifelongTrainer

__all__ = ["ConfidenceMonitor", "EWCRegularizer", "LifelongTrainer"]
