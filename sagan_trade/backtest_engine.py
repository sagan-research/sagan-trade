import numpy as np
import pandas as pd

from typing import Optional, Dict, Any

class BacktestResult:
    """
    Container class for portfolio backtest results and performance metrics.

    This class encapsulates the full output of a simulation run, providing
    access to top-level metrics as well as the underlying equity curves.

    Args:
        sharpe_ratio (float): The annualized Sharpe ratio.
        max_drawdown (float): The maximum percentage drawdown.
        total_return (float): The total percentage return of the strategy.
        portfolio_values (pd.Series): Time series of portfolio cash values.
        cumulative_returns (pd.Series): Time series of cumulative returns.
        metrics (Dict[str, Any]): A dictionary of extended performance metrics.
    """
    def __init__(self, sharpe_ratio: float, max_drawdown: float, total_return: float, 
                 portfolio_values: pd.Series, cumulative_returns: pd.Series, metrics: Dict[str, Any]) -> None:
        self.sharpe_ratio: float = sharpe_ratio
        self.max_drawdown: float = max_drawdown
        self.total_return: float = total_return
        self.portfolio_values: pd.Series = portfolio_values
        self.cumulative_returns: pd.Series = cumulative_returns
        self.metrics: Dict[str, Any] = metrics

class BacktestEngine:
    """
    High-Fidelity Daily Asset Backtesting Engine.

    Implements realistic fee accounting, dynamic allocation overlays, and 
    strict lookahead bias prevention by shifting signals by one period.

    Args:
        initial_capital (float, optional): Starting portfolio balance. Defaults to 1000000.0.
        maker_fee (float, optional): Passive trading fee fraction. Defaults to 0.0001 (1 bps).
        taker_fee (float, optional): Aggressive trading fee fraction. Defaults to 0.0003 (3 bps).
    """
    def __init__(self, initial_capital: float = 1000000.0, maker_fee: float = 0.0001, taker_fee: float = 0.0003) -> None:
        self.initial_capital: float = initial_capital
        # Handle negative fee inputs (like rebates) by taking absolute value or keeping rebates
        self.maker_fee: float = abs(maker_fee)
        self.taker_fee: float = abs(taker_fee)

    def run(self, prices: pd.Series, alpha_signals: pd.Series, regime_filter: Optional[pd.Series] = None, risk_model: Any = None) -> BacktestResult:
        """
        Run portfolio backtest over prices utilizing given signals and filters.

        This engine calculates target allocations by blending the raw alpha signals
        with the regime filter and risk model outputs. It strictly enforces a 1-period 
        signal shift to prevent lookahead bias. Turnover and transaction costs are computed.

        Args:
            prices (pd.Series): Time series of asset closing prices.
            alpha_signals (pd.Series): Time series of raw alpha scores bounded roughly in [-1, 1].
            regime_filter (Optional[pd.Series], optional): A series indicating market regime scaling (0.0 to 1.0). Defaults to None.
            risk_model (Any, optional): An instantiated risk engine providing `get_risk_multiplier(prices)`. Defaults to None.

        Returns:
            BacktestResult: An object encapsulating the simulation output and equity curve.

        Examples:
            >>> import pandas as pd
            >>> from sagan_trade.backtest_engine import BacktestEngine
            >>> engine = BacktestEngine(initial_capital=100000)
            >>> prices = pd.Series([100, 101, 102])
            >>> signals = pd.Series([1.0, 1.0, 1.0])
            >>> result = engine.run(prices, signals)
            >>> result.metrics['total_return_pct'] >= 0.0
            True
        """
        # Daily simple returns of the underlying asset
        asset_returns = prices.pct_change().fillna(0.0)
        
        # Default filters to 1.0 if not provided
        if regime_filter is None:
            regime_filter = pd.Series(1.0, index=prices.index)
            
        if risk_model is not None:
            risk_multiplier = risk_model.get_risk_multiplier(prices)
        else:
            risk_multiplier = pd.Series(1.0, index=prices.index)
            
        # Target position calculation (bounded to [-1.0, 1.0])
        raw_weights = alpha_signals * regime_filter * risk_multiplier
        raw_weights = np.clip(raw_weights, -1.0, 1.0).fillna(0.0)
        
        # Shift weights to prevent lookahead bias
        position_weights = raw_weights.shift(1).fillna(0.0)
        
        # Portfolio turnover calculation
        turnover = np.abs(position_weights - position_weights.shift(1).fillna(0.0))
        
        # Transaction costs: using taker fee on weight adjustments
        tx_costs = turnover * self.taker_fee
        
        # Gross and Net portfolio returns
        gross_returns = position_weights * asset_returns
        net_returns = gross_returns - tx_costs
        
        # Cumulative returns and equity curve
        cumulative_returns = (1 + net_returns).cumprod()
        portfolio_values = self.initial_capital * cumulative_returns
        
        # Total returns
        total_ret = cumulative_returns.iloc[-1] - 1
        
        # Sharpe Ratio (annualized, 252 trading days)
        mean_ret = net_returns.mean()
        std_ret = net_returns.std()
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(252)
        else:
            sharpe = 0.0
            
        # Max Drawdown
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        metrics = {
            "total_return_pct": round(total_ret * 100, 4),
            "annualized_return_pct": round(((1 + total_ret) ** (252 / len(prices)) - 1) * 100, 4) if len(prices) > 0 else 0.0,
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 3),
            "total_turnover": round(turnover.sum(), 4)
        }
        
        return BacktestResult(
            sharpe_ratio=round(sharpe, 3),
            max_drawdown=round(max_dd * 100, 3),  # Percentage value for readability
            total_return=round(total_ret * 100, 4),
            portfolio_values=portfolio_values,
            cumulative_returns=cumulative_returns,
            metrics=metrics
        )
