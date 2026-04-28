"""Reusable financial performance metric helpers.
"""

from __future__ import annotations
import numpy as np

def sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    excess = returns - risk_free_rate / periods_per_year
    std = float(np.std(excess, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(excess) / std)

def max_drawdown(returns: np.ndarray) -> float:
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    return float(np.min(drawdowns))

def annualised_return(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    total = float(np.prod(1 + returns))
    return float(total ** (periods_per_year / n) - 1)

def calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    ann_ret = annualised_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))
    if mdd == 0.0:
        return 0.0
    return float(ann_ret / mdd)

def win_rate(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))

def profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf")
    return float(gains / losses)
