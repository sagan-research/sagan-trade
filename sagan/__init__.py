"""Sagan – Symbolic Mathematical Engine for Trading.

Sagan replaces black-box neural networks with transparent, human-readable 
mathematations discovered via FunctionGemma (via Ollama). 
Targeting R2 > 0.95 for all variables to ensure precision.

Quick start::

    import sagan

    # Train a new symbolic model
    model_id = sagan.train(["AAPL"], signals=["Close", "RSI", "Volume"], target_r2=0.95)

    # Get a trading signal
    result = sagan.predict()
    print(result["signal"], result["formula"])
"""

from __future__ import annotations

try:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("sagan")
    except PackageNotFoundError:
        __version__ = "0.4.0"  # fallback for editable/dev installs
except ImportError:
    __version__ = "0.4.0"

from sagan.config import config, SaganConfig
from sagan.ensemble import SymbolicRegressor, train
from sagan.exceptions import (
    SaganError,
    ModelNotFoundError,
    InsufficientDataError,
    FetchError,
    ConfigurationError,
    RegistryCorruptedError,
)
from sagan.logging_config import setup_logging
from sagan.predict import predict, batch_predict
from sagan.registry import list_models, delete_model, export_model, get_model
from sagan.research import BacktestEngine
from sagan.autonomous import AutonomousResearcher
from sagan.nlp import SaganInterpreter, CopilotOrchestrator
from sagan.portfolio import PortfolioRebalancer, import_portfolio, get_snaptrade_holdings
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
    # Symbolic Engine
    "SymbolicRegressor",
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
    # Research & Autonomy
    "BacktestEngine",
    "AutonomousResearcher",
    "SaganInterpreter",
    "CopilotOrchestrator",
    "PortfolioRebalancer",
    "import_portfolio",
    "get_snaptrade_holdings",
    # Version
    "__version__",
]
