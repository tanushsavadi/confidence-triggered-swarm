"""Configuration loading utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml


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
    if path is None:
        path = Path(__file__).parent / "default.yaml"
    else:
        path = Path(path)

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
