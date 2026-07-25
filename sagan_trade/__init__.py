"""
Sagan Trade - SOTA Quantitative Finance Library

A cutting-edge quantitative finance library featuring:
- Symbolic Regression for interpretable alpha discovery
- Temporal Fusion Transformers (TFT) for multi-horizon forecasting
- Physics-Informed Neural Networks (PINNs) for option pricing/volatility
- Advanced Portfolio Optimization (HRP, Risk Parity, Black-Litterman)
- Institutional-Grade Backtesting (Walk-Forward, Combinatorial Purged K-Fold)
- Market Microstructure Modeling (Hawkes, Bates Jump-Diffusion)
- Optimal Execution (Almgren-Chriss)
"""

from .asymmetric_risk_engine import AsymmetricRiskEngine
from .backtest_engine import BacktestEngine
from .backtest_engine import BacktestResult as BacktestResultSimple
from .backtesting_advanced import (
    AggregatedBacktestResult,
    BacktestConfig,
    BacktestResult,
    CombinatorialPurgedCVBacktester,
    MonteCarloBacktester,
    PurgedKFoldBacktester,
    WalkForwardBacktester,
    compute_performance_metrics,
    run_backtest,
)
from .data import DataConfig, DataLoader
from .execution import (
    AlmgrenChrissModel,
    BertsimasLoModel,
    ExecutionConfig,
    ExecutionModel,
    ExecutionResult,
    GatheralSchiedModel,
    ImplementationShortfallModel,
    ObizhaevaWangModel,
    POVModel,
    TWAPModel,
    VWAPModel,
    compare_execution_models,
    create_execution_model,
    optimize_execution,
)
from .feature_engineering import (
    CrossSectionalFeatures,
    FeatureConfig,
    FeatureEngine,
    MicrostructureFeatures,
    TechnicalIndicators,
    create_feature_engine,
)

try:
    from .firestore_client import SaganFirestore
except ImportError:
    SaganFirestore = None  # type: ignore[misc,assignment]
from sagan.simulator import HawkesLOBSimulator

from .market_microstructure import (
    analyze_portfolio,
    simulate_price_range,
    simulate_price_range_gbm,
    simulate_price_range_merton,
    visualize_stock_insights,
)
from .pinn_models import (
    BlackScholesPINN,
    HestonPINN,
    PINNConfig,
    PINNTrainer,
    create_bs_pinn,
    create_heston_pinn,
)
from .portfolio_optimization import (
    BlackLittermanOptimizer,
    HierarchicalRiskParity,
    MaximumDiversification,
    MeanVarianceOptimizer,
    MinimumVariance,
    OptimizationConfig,
    OptimizationResult,
    RiskParityOptimizer,
    create_optimizer,
    efficient_frontier,
    optimize_portfolio,
)
from .symbolic_regressor import SymbolicRegressor
from .tft_model import TemporalFusionTransformer, TFTConfig, create_tft_model
from .volatility_regime_filter import VolatilityRegimeFilter

__all__ = [
    # Core
    "SymbolicRegressor",
    "AsymmetricRiskEngine",
    "BacktestEngine",
    "BacktestResultSimple",
    "VolatilityRegimeFilter",
    "SaganFirestore",
    # Market Microstructure
    "simulate_price_range",
    "simulate_price_range_gbm",
    "simulate_price_range_merton",
    "analyze_portfolio",
    "visualize_stock_insights",
    "HawkesLOBSimulator",
    # Portfolio Optimization
    "HierarchicalRiskParity",
    "RiskParityOptimizer",
    "BlackLittermanOptimizer",
    "MeanVarianceOptimizer",
    "MaximumDiversification",
    "MinimumVariance",
    "OptimizationConfig",
    "OptimizationResult",
    "create_optimizer",
    "optimize_portfolio",
    "efficient_frontier",
    # Deep Learning
    "TemporalFusionTransformer",
    "TFTConfig",
    "create_tft_model",
    "PINNConfig",
    "BlackScholesPINN",
    "HestonPINN",
    "PINNTrainer",
    "create_bs_pinn",
    "create_heston_pinn",
    # Execution
    "ExecutionConfig",
    "ExecutionResult",
    "ExecutionModel",
    "AlmgrenChrissModel",
    "BertsimasLoModel",
    "ObizhaevaWangModel",
    "GatheralSchiedModel",
    "TWAPModel",
    "VWAPModel",
    "POVModel",
    "ImplementationShortfallModel",
    "create_execution_model",
    "optimize_execution",
    "compare_execution_models",
    # Advanced Backtesting
    "BacktestConfig",
    "BacktestResult",
    "AggregatedBacktestResult",
    "WalkForwardBacktester",
    "PurgedKFoldBacktester",
    "CombinatorialPurgedCVBacktester",
    "MonteCarloBacktester",
    "compute_performance_metrics",
    "run_backtest",
    # Feature Engineering
    "FeatureConfig",
    "FeatureEngine",
    "TechnicalIndicators",
    "MicrostructureFeatures",
    "CrossSectionalFeatures",
    "create_feature_engine",
    # Data
    "DataLoader",
    "DataConfig",
]

__version__ = "2.0.0"
