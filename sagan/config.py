"""Global configuration for the Sagan XAI library.

The singleton :data:`config` object is created at import time and shared
across all modules.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SaganConfig:
    """Centralised configuration for Sagan XAI.

    Includes model hyper-parameters and storage directories.
    """

    home_dir: Path = field(default_factory=lambda: Path.home() / ".sagan")
    models_dir: Path = field(default_factory=lambda: Path.home() / ".sagan" / "xai_models")

    # Model architecture
    default_window: int = 10
    default_horizon: int = 3
    default_epochs: int = 30
    default_head_dim: int = 32
    default_num_heads: int = 4
    default_ff_dim: int = 64
    default_dropout: float = 0.1
    default_threshold: float = 0.01

    # PINN / XAI
    pinn_lambda: float = 0.01
    xai_confidence_threshold: float = 0.6
    regime_change_threshold: float = 0.3

    def __post_init__(self) -> None:
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "SaganConfig":
        """Create a :class:`SaganConfig` from environment variables."""
        kwargs: dict = {}
        _env_map = {
            "SAGAN_HOME_DIR": ("home_dir", Path),
            "SAGAN_MODELS_DIR": ("models_dir", Path),
            "SAGAN_DEFAULT_WINDOW": ("default_window", int),
            "SAGAN_DEFAULT_HORIZON": ("default_horizon", int),
            "SAGAN_DEFAULT_EPOCHS": ("default_epochs", int),
        }
        for env_key, (field_name, cast) in _env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                kwargs[field_name] = cast(val)
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, overrides: dict) -> "SaganConfig":
        """Create a :class:`SaganConfig` with selective field overrides."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in overrides.items() if k in valid_fields}
        return cls(**filtered)


# Singleton
config = SaganConfig()

