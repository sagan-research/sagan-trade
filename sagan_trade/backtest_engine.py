import numpy as np
import pandas as pd

class BacktestResult:
    """
    Container class for portfolio backtest results and performance metrics.
    """
    def __init__(self, sharpe_ratio, max_drawdown, total_return, portfolio_values, cumulative_returns, metrics):
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.total_return = total_return
        self.portfolio_values = portfolio_values
        self.cumulative_returns = cumulative_returns
        self.metrics = metrics

class BacktestEngine:
    """
    High-Fidelity Daily Asset Backtesting Engine implementing fee accounting
    and dynamic allocation from alpha signal overlays and risk models.
    """
    def __init__(self, initial_capital: float = 1000000.0, maker_fee: float = 0.0001, taker_fee: float = 0.0003):
        self.initial_capital = initial_capital
        # Handle negative fee inputs (like rebates) by taking absolute value or keeping rebates
        self.maker_fee = abs(maker_fee)
        self.taker_fee = abs(taker_fee)

    def run(self, prices: pd.Series, alpha_signals: pd.Series, regime_filter: pd.Series = None, risk_model = None) -> BacktestResult:
        """
        Runs portfolio backtest over prices utilizing alpha signals, regime filters, and risk scaling.
        Enforces a 1-period signal shift to strictly prevent lookahead bias.
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
