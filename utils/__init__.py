# utils/config_loader.py

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix not in {".yaml", ".yml"}:
        raise ValueError("Config file must be YAML.")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    return config