"""
config_loader.py
Loads and validates config.yaml into a clean, dot-accessible object.
"""

import yaml
from pathlib import Path
from typing import Any


class Config:
    """Wraps a nested dict so attributes are dot-accessible: cfg.viewer.target_fps"""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def set(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __repr__(self):
        return f"Config({self.__dict__})"


def load_config(path: str = "config/config.yaml") -> Config:
    """
    Load YAML config from *path* and return a dot-accessible Config object.

    Args:
        path: Path to the YAML config file.

    Returns:
        Config object with dot-accessible attributes.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            "Create config/config.yaml before running."
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path.resolve()}")

    return Config(raw)