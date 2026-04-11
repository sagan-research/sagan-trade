"""Sagan XAI – Explainable Probabilistic Ensemble for Trading.

Sagan combines Physics-Informed Neural Networks (PINN), a Temporal Fusion
Transformer (TFT), and an XAI-RL override layer to produce calibrated
LONG / SHORT / NEUTRAL signals with built-in explainability.

Quick start::

    import sagan

    # Train a new ensemble on a basket of tickers
    model_id = sagan.train(["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])

    # Get a trading signal
    result = sagan.predict()
    print(result["signal"], result["confidence"])

    # Parallel training (one model per ticker)
    results = sagan.train_parallel_from_fetch(["AAPL", "MSFT"], num_processes=4)

Documentation:
    https://sagan-docs.vercel.app
"""

from __future__ import annotations

try:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("sagan")
    except PackageNotFoundError:
        __version__ = "0.1.0"  # fallback for editable/dev installs
except ImportError:
    __version__ = "0.1.0"

from sagan.config import config, SaganConfig
from sagan.ensemble import ExplainableEnsemble, train
from sagan.exceptions import (
    SaganError,
    ModelNotFoundError,
    InsufficientDataError,
    FetchError,
    ConfigurationError,
    RegistryCorruptedError,
)
from sagan.logging_config import setup_logging
from sagan.parallel import train_parallel, train_parallel_from_fetch
from sagan.predict import predict, batch_predict
from sagan.registry import list_models, delete_model, export_model, get_model
from sagan.utils import (
    sharpe_ratio,
    max_drawdown,
    annualised_return,
    calmar_ratio,
    win_rate,
    profit_factor,
)

__all__ = [
    # Core API
    "train",
    "predict",
    "batch_predict",
    # Ensemble class
    "ExplainableEnsemble",
    # Parallel training
    "train_parallel",
    "train_parallel_from_fetch",
    # Registry
    "list_models",
    "delete_model",
    "export_model",
    "get_model",
    # Config
    "config",
    "SaganConfig",
    # Logging
    "setup_logging",
    # Metrics
    "sharpe_ratio",
    "max_drawdown",
    "annualised_return",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    # Exceptions
    "SaganError",
    "ModelNotFoundError",
    "InsufficientDataError",
    "FetchError",
    "ConfigurationError",
    "RegistryCorruptedError",
    # Version
    "__version__",
]
