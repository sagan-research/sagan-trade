"""Reusable financial performance metric helpers.

All functions accept a 1-D :class:`numpy.ndarray` of **period returns**
(e.g. daily returns expressed as fractions, not percentages).

Example:
    >>> import numpy as np
    >>> from sagan.utils import sharpe_ratio, max_drawdown
    >>> rets = np.random.normal(0.0005, 0.01, 252)
    >>> print(f"Sharpe: {sharpe_ratio(rets):.2f}")
    >>> print(f"MaxDD:  {max_drawdown(rets):.2%}")
"""

from __future__ import annotations

import numpy as np


def sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sharpe ratio.

    .. math::

        \\text{Sharpe} = \\frac{\\bar{r} - r_f}{\\sigma_r} \\times \\sqrt{N}

    Args:
        returns: 1-D array of period returns.
        periods_per_year: Number of periods per year (252 for daily, 52 for
            weekly, 12 for monthly).
        risk_free_rate: Annualised risk-free rate (default 0.0).

    Returns:
        Annualised Sharpe ratio. Returns 0.0 if standard deviation is zero.

    Example:
        >>> sharpe_ratio(np.array([0.001, -0.002, 0.003, 0.0005]))
        5.17...
    """
    excess = returns - risk_free_rate / periods_per_year
    std = float(np.std(excess, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(excess) / std)


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of a return series.

    .. math::

        \\text{MDD} = \\min_t \\left( \\frac{V_t}{\\max_{\\tau \\le t} V_\\tau} - 1 \\right)

    Args:
        returns: 1-D array of period returns.

    Returns:
        Maximum drawdown as a *negative* fraction (e.g. ``-0.25`` means −25 %).

    Example:
        >>> max_drawdown(np.array([0.01, -0.05, 0.02, -0.10, 0.03]))
        -0.1445...
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    return float(np.min(drawdowns))


def annualised_return(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Geometrically compounded annualised return.

    .. math::

        \\text{Ann. Return} = \\left( \\prod_t (1 + r_t) \\right)^{N / T} - 1

    Args:
        returns: 1-D array of period returns.
        periods_per_year: Number of periods per year.

    Returns:
        Annualised return as a fraction.

    Example:
        >>> annualised_return(np.full(252, 0.001))
        0.2879...
    """
    n = len(returns)
    if n == 0:
        return 0.0
    total = float(np.prod(1 + returns))
    return float(total ** (periods_per_year / n) - 1)


def calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio: annualised return divided by absolute max drawdown.

    Args:
        returns: 1-D array of period returns.
        periods_per_year: Number of periods per year.

    Returns:
        Calmar ratio. Returns 0.0 if drawdown is zero.

    Example:
        >>> calmar_ratio(np.array([0.001] * 252))
        ...
    """
    ann_ret = annualised_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))
    if mdd == 0.0:
        return 0.0
    return float(ann_ret / mdd)


def win_rate(returns: np.ndarray) -> float:
    """Fraction of periods with a positive return.

    Args:
        returns: 1-D array of period returns.

    Returns:
        Win rate in [0, 1].
    """
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def profit_factor(returns: np.ndarray) -> float:
    """Gross profit divided by gross loss.

    Args:
        returns: 1-D array of period returns.

    Returns:
        Profit factor. Returns ``float('inf')`` if there are no losing periods.
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf")
    return float(gains / losses)
