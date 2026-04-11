"""Global configuration for the Sagan XAI library.

The singleton :data:`config` object is created at import time and shared
across all modules. Override individual fields for a session::

    from sagan import config
    config.default_epochs = 50
    config.pinn_lambda = 0.05

Or create a fresh instance from environment variables::

    from sagan.config import SaganConfig
    cfg = SaganConfig.from_env()

Or from a dictionary (useful in notebooks/testing)::

    cfg = SaganConfig.from_dict({"default_window": 20, "pinn_lambda": 0.001})
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SaganConfig:
    """Centralised runtime configuration for Sagan XAI.

    Attributes:
        home_dir: Root directory for Sagan data. Defaults to ``~/.sagan``.
        models_dir: Where trained models are stored.
            Defaults to ``~/.sagan/xai_models``.
        default_window: Look-back window in trading days.
        default_horizon: Forward horizon used for label generation.
        default_epochs: Maximum training epochs (early stopping may stop earlier).
        default_head_dim: Key/value dimension per attention head in the TFT.
        default_num_heads: Number of parallel attention heads.
        default_ff_dim: Hidden units in the feed-forward layer.
        default_dropout: Dropout probability applied inside the TFT.
        default_threshold: Return threshold (fraction) for buy/sell/hold labelling.
        pinn_lambda: Weight of the Ornstein–Uhlenbeck regularisation term.
        xai_confidence_threshold: Minimum softmax probability to suppress the
            XAI override flag.
        regime_change_threshold: Threshold for flagging regime transitions in
            the XAI layer.
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

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "SaganConfig":
        """Create a :class:`SaganConfig` from environment variables.

        Environment variables are prefixed with ``SAGAN_`` and use the
        uppercase field name. For example:

        .. code-block:: bash

            export SAGAN_DEFAULT_WINDOW=20
            export SAGAN_PINN_LAMBDA=0.005
            export SAGAN_HOME_DIR=/data/sagan

        Returns:
            A new :class:`SaganConfig` instance populated from the environment.

        Example:
            >>> import os
            >>> os.environ["SAGAN_DEFAULT_EPOCHS"] = "50"
            >>> from sagan.config import SaganConfig
            >>> cfg = SaganConfig.from_env()
            >>> cfg.default_epochs
            50
        """
        kwargs: dict = {}
        _env_map = {
            "SAGAN_HOME_DIR": ("home_dir", Path),
            "SAGAN_MODELS_DIR": ("models_dir", Path),
            "SAGAN_DEFAULT_WINDOW": ("default_window", int),
            "SAGAN_DEFAULT_HORIZON": ("default_horizon", int),
            "SAGAN_DEFAULT_EPOCHS": ("default_epochs", int),
            "SAGAN_DEFAULT_HEAD_DIM": ("default_head_dim", int),
            "SAGAN_DEFAULT_NUM_HEADS": ("default_num_heads", int),
            "SAGAN_DEFAULT_FF_DIM": ("default_ff_dim", int),
            "SAGAN_DEFAULT_DROPOUT": ("default_dropout", float),
            "SAGAN_DEFAULT_THRESHOLD": ("default_threshold", float),
            "SAGAN_PINN_LAMBDA": ("pinn_lambda", float),
            "SAGAN_XAI_CONFIDENCE_THRESHOLD": ("xai_confidence_threshold", float),
            "SAGAN_REGIME_CHANGE_THRESHOLD": ("regime_change_threshold", float),
        }
        for env_key, (field_name, cast) in _env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                kwargs[field_name] = cast(val)
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, overrides: dict) -> "SaganConfig":
        """Create a :class:`SaganConfig` with selective field overrides.

        Args:
            overrides: Mapping of field names to new values. Unknown keys are
                silently ignored.

        Returns:
            A new :class:`SaganConfig` instance.

        Example:
            >>> from sagan.config import SaganConfig
            >>> cfg = SaganConfig.from_dict({"default_window": 20, "pinn_lambda": 0.001})
            >>> cfg.default_window
            20
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in overrides.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Module-level singleton (shared across all sagan modules)
# ---------------------------------------------------------------------------
config = SaganConfig()
