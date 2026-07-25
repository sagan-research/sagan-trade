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

from .volatility_regime_filter import VolatilityRegimeFilter
from .symbolic_regressor import SymbolicRegressor
from .asymmetric_risk_engine import AsymmetricRiskEngine
from .backtest_engine import BacktestEngine, BacktestResult
from .market_microstructure import (
    simulate_price_range,
    simulate_price_range_gbm,
    simulate_price_range_merton,
    analyze_portfolio,
    visualize_stock_insights,
    HawkesLOBSimulator,
)
from .firestore_client import SaganFirestore
from .portfolio_optimization import (
    HierarchicalRiskParity,
    RiskParityOptimizer,
    BlackLittermanOptimizer,
    MeanVarianceOptimizer,
    MaximumDiversification,
    MinimumVariance,
    OptimizationConfig,
    OptimizationResult,
    create_optimizer,
    optimize_portfolio,
    efficient_frontier,
)
from .tft_model import TemporalFusionTransformer, TFTConfig, create_tft_model
from .pinn_models import (
    PINNConfig,
    BlackScholesPINN,
    HestonPINN,
    PINNTrainer,
    create_bs_pinn,
    create_heston_pinn,
)
from .execution import (
    ExecutionConfig,
    ExecutionResult,
    ExecutionModel,
    AlmgrenChrissModel,
    BertsimasLoModel,
    ObizhaevaWangModel,
    GatheralSchiedModel,
    TWAPModel,
    VWAPModel,
    POVModel,
    ImplementationShortfallModel,
    create_execution_model,
    optimize_execution,
    compare_execution_models,
)
from .backtesting_advanced import (
    BacktestConfig,
    BacktestResult,
    AggregatedBacktestResult,
    WalkForwardBacktester,
    PurgedKFoldBacktester,
    CombinatorialPurgedCVBacktester,
    MonteCarloBacktester,
    compute_performance_metrics,
    run_backtest,
)
from .feature_engineering import (
    FeatureConfig,
    FeatureEngine,
    TechnicalIndicators,
    MicrostructureFeatures,
    CrossSectionalFeatures,
    create_feature_engine,
)
from .data import DataLoader, DataConfig

__all__ = [
    # Core
    "SymbolicRegressor",
    "AsymmetricRiskEngine",
    "BacktestEngine",
    "BacktestResult",
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