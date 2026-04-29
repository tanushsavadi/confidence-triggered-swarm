"""Configuration loading utilities."""

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overrides into base and return the merged dict."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path : str or None
        Path to the YAML config file. If None, loads the default config.

    Returns
    -------
    dict
        Configuration dictionary with all hyperparameters.
    """
    default_path = Path(__file__).parent / "default.yaml"
    with open(default_path, "r") as f:
        config = yaml.safe_load(f)

    if path is None:
        return config

    with open(Path(path), "r") as f:
        overrides = yaml.safe_load(f) or {}

    config = _deep_update(config, overrides)
    return config
