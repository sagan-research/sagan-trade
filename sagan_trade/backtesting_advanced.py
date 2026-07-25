"""
Advanced Backtesting Module.

Implements institutional-grade backtesting methodologies:
- Walk-Forward Analysis (WFA) / Rolling Window
- Combinatorial Purged K-Fold Cross-Validation (CPCV) - Lopez de Prado
- Monte Carlo Simulation for strategy robustness
- Probabilistic Sharpe Ratio (PSR)
- Deflated Sharpe Ratio (DSR)
- Performance Attribution

References:
- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning." Ch. 12.
- Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio."
- Harvey, C. & Liu, Y. (2015). "Backtesting."
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any, Callable, Iterator, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from joblib import Parallel, delayed
import itertools


class BacktestMethod(Enum):
    """Backtesting methodology types."""
    WALK_FORWARD = "walk_forward"
    PURGED_KFOLD = "purged_kfold"
    CPCV = "cp_cv"  # Combinatorial Purged Cross-Validation
    MONTE_CARLO = "monte_carlo"
    EXPANDING_WINDOW = "expanding_window"
    ROLLING_WINDOW = "rolling_window"


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Walk-forward parameters
    train_window: int = 252       # Training window (e.g., 1 year daily)
    test_window: int = 63         # Test window (e.g., 1 quarter)
    step_size: int = 21           # Step between folds (e.g., 1 month)
    min_train_size: int = 126     # Minimum training observations
    
    # Purged K-Fold parameters
    n_splits: int = 5
    purge_gap: int = 5            # Gap to avoid lookahead bias
    embargo_pct: float = 0.01     # Embargo percentage
    
    # CPCV parameters
    n_combinations: int = 10      # Number of combinations to test
    
    # Monte Carlo parameters
    n_simulations: int = 1000
    block_size: int = 21          # Block bootstrap size
    
    # Performance metrics
    risk_free_rate: float = 0.02  # Annual risk-free rate
    confidence_level: float = 0.95
    
    # Transaction costs
    commission: float = 0.0003    # 3 bps
    slippage: float = 0.0001      # 1 bp
    
    # Parallel processing
    n_jobs: int = -1              # Use all cores


@dataclass
class BacktestResult:
    """Results from a single backtest fold/simulation."""
    
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # Risk metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Trade metrics
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade: float = 0.0
    
    # Probabilistic metrics
    probabilistic_sharpe_ratio: float = 0.0
    deflated_sharpe_ratio: float = 0.0
    
    # Equity curve
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    positions: pd.Series = field(default_factory=pd.Series)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding series)."""
        return {k: v for k, v in self.__dict__.items() 
                if not isinstance(v, pd.Series)}


@dataclass
class AggregatedBacktestResult:
    """Aggregated results across all folds/simulations."""
    
    method: str
    n_folds: int
    individual_results: List[BacktestResult]
    
    # Aggregate metrics (mean ± std)
    mean_return: float = 0.0
    std_return: float = 0.0
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    mean_max_dd: float = 0.0
    
    # Statistical significance
    psr: float = 0.0              # Probabilistic Sharpe Ratio
    dsr: float = 0.0              # Deflated Sharpe Ratio
    p_value: float = 1.0
    
    # Aggregate equity curve
    aggregate_equity: pd.Series = field(default_factory=pd.Series)
    aggregate_returns: pd.Series = field(default_factory=pd.Series)
    
    def summary(self) -> pd.DataFrame:
        """Return summary statistics as DataFrame."""
        return pd.DataFrame({
            "Metric": [
                "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
                "Sortino Ratio", "Calmar Ratio", "Max Drawdown",
                "Win Rate", "Profit Factor", "Probabilistic Sharpe",
                "Deflated Sharpe", "p-value"
            ],
            "Value": [
                f"{self.mean_return:.2%}", f"{self.std_return:.2%}",
                f"{self.mean_sharpe:.3f}", f"{self.individual_results[0].sortino_ratio:.3f}" if self.individual_results else "N/A",
                f"{self.individual_results[0].calmar_ratio:.3f}" if self.individual_results else "N/A",
                f"{self.mean_max_dd:.2%}",
                f"{np.mean([r.win_rate for r in self.individual_results]):.2%}" if self.individual_results else "N/A",
                f"{np.mean([r.profit_factor for r in self.individual_results]):.3f}" if self.individual_results else "N/A",
                f"{self.psr:.3f}", f"{self.dsr:.3f}", f"{self.p_value:.4f}"
            ]
        })


class BaseBacktester(ABC):
    """Abstract base class for backtesters."""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        
    @abstractmethod
    def generate_splits(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index splits."""
        pass
    
    @abstractmethod
    def run(self, 
            strategy: Callable,
            prices: pd.DataFrame,
            signals: Optional[pd.DataFrame] = None,
            **kwargs) -> AggregatedBacktestResult:
        """Run backtest."""
        pass


class WalkForwardBacktester(BaseBacktester):
    """
    Walk-Forward Analysis (Rolling/Expanding Window).
    
    Simulates real-time trading by training on a rolling window and 
    testing on the subsequent period. Most realistic for production.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None, expanding: bool = False):
        super().__init__(config)
        self.expanding = expanding
        
    def generate_splits(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits."""
        cfg = self.config
        indices = np.arange(n_samples)
        
        start = 0
        while True:
            train_end = start + cfg.train_window
            test_end = train_end + cfg.test_window
            
            if test_end > n_samples:
                break
                
            if self.expanding:
                train_idx = indices[:train_end]
            else:
                train_idx = indices[start:train_end]
                
            test_idx = indices[train_end:test_end]
            
            if len(train_idx) >= cfg.min_train_size:
                yield train_idx, test_idx
                
            start += cfg.step_size
            
    def run(self,
            strategy: Callable,
            prices: pd.DataFrame,
            signals: Optional[pd.DataFrame] = None,
            **kwargs) -> AggregatedBacktestResult:
        """
        Run walk-forward backtest.
        
        Args:
            strategy: Function(train_prices, train_signals) -> test_positions
            prices: Price DataFrame with DatetimeIndex
            signals: Optional signal DataFrame
            **kwargs: Additional arguments passed to strategy
            
        Returns:
            AggregatedBacktestResult
        """
        n_samples = len(prices)
        results = []
        
        for fold_id, (train_idx, test_idx) in enumerate(self.generate_splits(n_samples)):
            train_prices = prices.iloc[train_idx]
            test_prices = prices.iloc[test_idx]
            
            train_signals = signals.iloc[train_idx] if signals is not None else None
            test_signals = signals.iloc[test_idx] if signals is not None else None
            
            # Train strategy
            positions = strategy(
                train_prices, train_signals, 
                test_prices, test_signals, **kwargs
            )
            
            # Evaluate on test period
            result = self._evaluate_fold(
                fold_id, train_idx, test_idx, prices.index,
                test_prices, positions
            )
            results.append(result)
            
        return self._aggregate_results(results, "WalkForward")
    
    def _evaluate_fold(self, fold_id: int, train_idx: np.ndarray, 
                       test_idx: np.ndarray, index: pd.DatetimeIndex,
                       test_prices: pd.DataFrame, positions: pd.DataFrame) -> BacktestResult:
        """Evaluate a single fold."""
        returns = test_prices.pct_change().dropna()
        
        # Align positions with returns
        common_idx = returns.index.intersection(positions.index)
        returns = returns.loc[common_idx]
        positions = positions.loc[common_idx]
        
        # Portfolio returns
        if isinstance(positions, pd.DataFrame):
            port_returns = (returns * positions.shift(1)).sum(axis=1)
        else:
            port_returns = returns * positions.shift(1)
            
        # Apply transaction costs
        if isinstance(positions, pd.DataFrame):
            turnover = positions.diff().abs().sum(axis=1)
        else:
            turnover = positions.diff().abs()
        costs = turnover * (self.config.commission + self.config.slippage)
        net_returns = port_returns - costs
        
        # Compute metrics
        equity = (1 + net_returns).cumprod()
        
        result = BacktestResult(
            fold_id=fold_id,
            train_start=index[train_idx[0]],
            train_end=index[train_idx[-1]],
            test_start=index[test_idx[0]],
            test_end=index[test_idx[-1]],
            equity_curve=equity,
            returns=net_returns,
            positions=positions.loc[common_idx],
        )
        
        self._compute_metrics(result, net_returns)
        return result
    
    def _compute_metrics(self, result: BacktestResult, returns: pd.Series):
        """Compute performance metrics."""
        if len(returns) == 0:
            return
            
        ann_factor = 252  # Daily data
        n_periods = len(returns)
        
        # Returns
        total_ret = (1 + returns).prod() - 1
        ann_ret = (1 + total_ret) ** (ann_factor / n_periods) - 1
        ann_vol = returns.std() * np.sqrt(ann_factor)
        
        # Sharpe
        excess = returns - self.config.risk_free_rate / ann_factor
        sharpe = excess.mean() / returns.std() * np.sqrt(ann_factor) if returns.std() > 0 else 0
        
        # Sortino
        downside = returns[returns < 0]
        sortino = (excess.mean() / downside.std() * np.sqrt(ann_factor)) if len(downside) > 0 and downside.std() > 0 else 0
        
        # Drawdown
        equity = result.equity_curve
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()
        
        # Calmar
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        
        # VaR / CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Probabilistic Sharpe Ratio
        psr = self._probabilistic_sharpe(returns, sharpe)
        
        # Deflated Sharpe Ratio (simplified)
        dsr = self._deflated_sharpe(returns, sharpe, len(self.generate_splits(len(returns))))
        
        result.total_return = total_ret
        result.annualized_return = ann_ret
        result.annualized_volatility = ann_vol
        result.sharpe_ratio = sharpe
        result.sortino_ratio = sortino
        result.calmar_ratio = calmar
        result.max_drawdown = max_dd
        result.var_95 = var_95
        result.cvar_95 = cvar_95
        result.probabilistic_sharpe_ratio = psr
        result.deflated_sharpe_ratio = dsr
        result.skewness = stats.skew(returns)
        result.kurtosis = stats.kurtosis(returns)
    
    def _probabilistic_sharpe(self, returns: pd.Series, sharpe: float) -> float:
        """Compute Probabilistic Sharpe Ratio (Bailey & Lopez de Prado)."""
        n = len(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        # PSR = Φ((Sharpe - Sharpe_bench) * sqrt(n-1) / sqrt(1 - skew*Sharpe + (kurt-1)/4 * Sharpe^2))
        sr_bench = 0  # Benchmark Sharpe
        denom = np.sqrt(1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2)
        if denom > 0:
            z = (sharpe - sr_bench) * np.sqrt(n - 1) / denom
            return stats.norm.cdf(z)
        return 0.5
    
    def _deflated_sharpe(self, returns: pd.Series, sharpe: float, n_trials: int) -> float:
        """Compute Deflated Sharpe Ratio."""
        # Simplified DSR - assumes independent trials
        # Full implementation requires correlation structure
        n = len(returns)
        expected_max = stats.norm.ppf(1 - 1/n_trials) if n_trials > 1 else 0
        return sharpe / max(expected_max, 1e-6) if expected_max > 0 else sharpe
    
    def _aggregate_results(self, results: List[BacktestResult], method: str) -> AggregatedBacktestResult:
        """Aggregate results across folds."""
        if not results:
            return AggregatedBacktestResult(method=method, n_folds=0, individual_results=[])
            
        # Mean metrics
        returns_list = [r.annualized_return for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        max_dds = [r.max_drawdown for r in results]
        
        # Combine equity curves (chain them)
        combined_equity = pd.concat([r.equity_curve for r in results])
        combined_returns = pd.concat([r.returns for r in results])
        
        # Aggregate PSR/DSR
        psr = np.mean([r.probabilistic_sharpe_ratio for r in results])
        dsr = np.mean([r.deflated_sharpe_ratio for r in results])
        
        # P-value from t-test on returns
        _, p_val = stats.ttest_1samp(returns_list, 0)
        
        return AggregatedBacktestResult(
            method=method,
            n_folds=len(results),
            individual_results=results,
            mean_return=np.mean(returns_list),
            std_return=np.std(returns_list),
            mean_sharpe=np.mean(sharpes),
            std_sharpe=np.std(sharpes),
            mean_max_dd=np.mean(max_dds),
            psr=psr,
            dsr=dsr,
            p_value=p_val,
            aggregate_equity=combined_equity,
            aggregate_returns=combined_returns,
        )


class PurgedKFoldBacktester(BaseBacktester):
    """
    Purged K-Fold Cross-Validation for Financial Time Series.
    
    Implements the purging and embargoing methodology from Lopez de Prado
    to prevent lookahead bias in time series cross-validation.
    """
    
    def generate_splits(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged K-Fold splits with embargo."""
        cfg = self.config
        indices = np.arange(n_samples)
        
        # Standard K-Fold splits
        fold_size = n_samples // cfg.n_splits
        
        for i in range(cfg.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            test_idx = indices[test_start:test_end]
            
            # Purge: remove observations around test set
            purge_size = int(cfg.purge_gap)
            embargo_size = int(cfg.embargo_pct * n_samples)
            
            # Training indices: before test (with purge) and after test (with embargo)
            train_before = indices[:max(0, test_start - purge_size)]
            train_after = indices[min(n_samples, test_end + embargo_size):]
            
            train_idx = np.concatenate([train_before, train_after])
            
            if len(train_idx) >= cfg.min_train_size:
                yield train_idx, test_idx
                
    def run(self,
            strategy: Callable,
            prices: pd.DataFrame,
            signals: Optional[pd.DataFrame] = None,
            **kwargs) -> AggregatedBacktestResult:
        """Run purged K-Fold backtest."""
        n_samples = len(prices)
        results = []
        
        for fold_id, (train_idx, test_idx) in enumerate(self.generate_splits(n_samples)):
            train_prices = prices.iloc[train_idx]
            test_prices = prices.iloc[test_idx]
            
            train_signals = signals.iloc[train_idx] if signals is not None else None
            test_signals = signals.iloc[test_idx] if signals is not None else None
            
            # Train strategy
            positions = strategy(
                train_prices, train_signals,
                test_prices, test_signals, **kwargs
            )
            
            result = self._evaluate_fold(fold_id, train_idx, test_idx, prices.index, test_prices, positions)
            results.append(result)
            
        return self._aggregate_results(results, "PurgedKFold")


class CombinatorialPurgedCVBacktester(BaseBacktester):
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    Tests all combinations of train/test splits to get distribution of 
    performance metrics. Most robust but computationally intensive.
    """
    
    def generate_splits(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate CPCV splits."""
        cfg = self.config
        
        # Get base purged K-Fold splits
        base_splitter = PurgedKFoldBacktester(cfg)
        base_splits = list(base_splitter.generate_splits(n_samples))
        
        if not base_splits:
            return
            
        n_base = len(base_splits)
        n_test_folds = cfg.n_splits
        
        # Generate combinations of test folds
        for combo in itertools.combinations(range(n_base), n_test_folds):
            # Combine selected test folds
            test_indices = np.concatenate([base_splits[i][1] for i in combo])
            test_indices = np.unique(test_indices)
            
            # Training is all other data (with purging)
            train_indices = []
            for i in range(n_base):
                if i not in combo:
                    train_indices.append(base_splits[i][0])
            train_indices = np.concatenate(train_indices)
            train_indices = np.unique(train_indices)
            
            # Apply purge/embargo around combined test set
            if len(test_indices) > 0:
                test_min, test_max = test_indices.min(), test_indices.max()
                purge = cfg.purge_gap
                embargo = int(cfg.embargo_pct * n_samples)
                
                train_indices = train_indices[
                    (train_indices < test_min - purge) | 
                    (train_indices > test_max + embargo)
                ]
                
            if len(train_indices) >= cfg.min_train_size and len(test_indices) > 0:
                yield train_indices, test_indices
                
    def run(self,
            strategy: Callable,
            prices: pd.DataFrame,
            signals: Optional[pd.DataFrame] = None,
            **kwargs) -> AggregatedBacktestResult:
        """Run CPCV backtest."""
        n_samples = len(prices)
        results = []
        
        # Limit combinations for performance
        max_combos = self.config.n_combinations
        combo_count = 0
        
        for fold_id, (train_idx, test_idx) in enumerate(self.generate_splits(n_samples)):
            if combo_count >= max_combos:
                break
                
            train_prices = prices.iloc[train_idx]
            test_prices = prices.iloc[test_idx]
            
            train_signals = signals.iloc[train_idx] if signals is not None else None
            test_signals = signals.iloc[test_idx] if signals is not None else None
            
            positions = strategy(
                train_prices, train_signals,
                test_prices, test_signals, **kwargs
            )
            
            result = self._evaluate_fold(fold_id, train_idx, test_idx, prices.index, test_prices, positions)
            results.append(result)
            combo_count += 1
            
        return self._aggregate_results(results, "CPCV")


class MonteCarloBacktester(BaseBacktester):
    """
    Monte Carlo Backtesting with Block Bootstrap.
    
    Generates synthetic return paths by resampling blocks of returns
    to preserve autocorrelation structure.
    """
    
    def generate_splits(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Not used for Monte Carlo - uses simulation instead."""
        return iter([])
        
    def run(self,
            strategy: Callable,
            prices: pd.DataFrame,
            signals: Optional[pd.DataFrame] = None,
            **kwargs) -> AggregatedBacktestResult:
        """Run Monte Carlo simulation."""
        cfg = self.config
        returns = prices.pct_change().dropna()
        n_samples = len(returns)
        
        # Fit strategy on full data to get base positions
        base_positions = strategy(prices, signals, prices, signals, **kwargs)
        base_returns = self._compute_portfolio_returns(returns, base_positions)
        
        # Block bootstrap simulations
        results = Parallel(n_jobs=cfg.n_jobs)(
            delayed(self._run_simulation)(sim_id, returns, base_positions, strategy, signals, prices, kwargs)
            for sim_id in range(cfg.n_simulations)
        )
        
        return self._aggregate_results(results, "MonteCarlo")
    
    def _run_simulation(self, sim_id: int, returns: pd.DataFrame, 
                        base_positions: pd.DataFrame,
                        strategy: Callable, signals: pd.DataFrame, 
                        prices: pd.DataFrame, kwargs: Dict) -> BacktestResult:
        """Run single Monte Carlo simulation."""
        cfg = self.config
        n = len(returns)
        
        # Block bootstrap
        block_size = cfg.block_size
        n_blocks = int(np.ceil(n / block_size))
        
        # Sample blocks with replacement
        block_starts = np.random.randint(0, n - block_size + 1, n_blocks)
        sim_returns = []
        for start in block_starts:
            end = min(start + block_size, n)
            sim_returns.append(returns.iloc[start:end])
        sim_returns = pd.concat(sim_returns).iloc[:n]
        sim_returns.index = returns.index
        
        # Simulate prices from returns
        sim_prices = (1 + sim_returns).cumprod() * prices.iloc[0]
        
        # Evaluate strategy on simulated data
        positions = strategy(prices, signals, sim_prices, signals, **kwargs)
        sim_portfolio_returns = self._compute_portfolio_returns(sim_returns, positions)
        
        # Metrics
        result = BacktestResult(
            fold_id=sim_id,
            train_start=returns.index[0],
            train_end=returns.index[-1],
            test_start=returns.index[0],
            test_end=returns.index[-1],
        )
        
        self._compute_metrics(result, sim_portfolio_returns)
        return result
    
    def _compute_portfolio_returns(self, returns: pd.DataFrame, 
                                   positions: pd.DataFrame) -> pd.Series:
        """Compute portfolio returns from asset returns and positions."""
        common_idx = returns.index.intersection(positions.index)
        returns = returns.loc[common_idx]
        positions = positions.loc[common_idx].shift(1)
        
        port_returns = (returns * positions).sum(axis=1)
        
        # Costs
        turnover = positions.diff().abs().sum(axis=1)
        costs = turnover * (self.config.commission + self.config.slippage)
        
        return port_returns - costs


# Utility functions
def compute_performance_metrics(returns: pd.Series, 
                                 risk_free: float = 0.02,
                                 periods_per_year: int = 252) -> Dict[str, float]:
    """Compute comprehensive performance metrics."""
    if len(returns) == 0:
        return {}
        
    ann_factor = periods_per_year
    n = len(returns)
    
    total_ret = (1 + returns).prod() - 1
    ann_ret = (1 + total_ret) ** (ann_factor / n) - 1
    ann_vol = returns.std() * np.sqrt(ann_factor)
    
    excess = returns - risk_free / ann_factor
    sharpe = excess.mean() / returns.std() * np.sqrt(ann_factor) if returns.std() > 0 else 0
    
    downside = returns[returns < 0]
    sortino = excess.mean() / downside.std() * np.sqrt(ann_factor) if len(downside) > 0 and downside.std() > 0 else 0
    
    equity = (1 + returns).cumprod()
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()
    
    # PSR
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    denom = np.sqrt(1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2)
    psr = stats.norm.cdf((sharpe - 0) * np.sqrt(n - 1) / denom) if denom > 0 else 0.5
    
    return {
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skewness": skew,
        "kurtosis": kurt,
        "probabilistic_sharpe_ratio": psr,
        "win_rate": (returns > 0).mean(),
        "profit_factor": returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if (returns < 0).any() else np.inf,
    }


def run_backtest(
    method: str,
    strategy: Callable,
    prices: pd.DataFrame,
    signals: Optional[pd.DataFrame] = None,
    config: Optional[BacktestConfig] = None,
    **kwargs
) -> AggregatedBacktestResult:
    """
    High-level backtest runner.
    
    Args:
        method: Backtest method name
        strategy: Strategy function
        prices: Price DataFrame
        signals: Optional signals DataFrame
        config: Backtest configuration
        **kwargs: Strategy arguments
        
    Returns:
        AggregatedBacktestResult
    """
    method = method.lower()
    
    if method in ["walk_forward", "wf", "rolling"]:
        backtester = WalkForwardBacktester(config, expanding=False)
    elif method in ["expanding_window", "expanding"]:
        backtester = WalkForwardBacktester(config, expanding=True)
    elif method in ["purged_kfold", "pkf", "kfold"]:
        backtester = PurgedKFoldBacktester(config)
    elif method in ["cpcv", "combinatorial_purged_cv"]:
        backtester = CombinatorialPurgedCVBacktester(config)
    elif method in ["monte_carlo", "mc", "bootstrap"]:
        backtester = MonteCarloBacktester(config)
    else:
        raise ValueError(f"Unknown backtest method: {method}")
        
    return backtester.run(strategy, prices, signals, **kwargs)