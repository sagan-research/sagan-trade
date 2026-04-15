"""Data fetching and preparation for Sagan XAI.

This module is responsible for:

1. Downloading OHLCV data from Yahoo Finance via :func:`fetch_prices`.
2. Building sliding-window supervised datasets via
   :func:`prepare_probabilistic_data`.
3. Validating ticker symbols via :func:`validate_tickers`.

All public functions raise :class:`~sagan.exceptions.FetchError` or
:class:`~sagan.exceptions.InsufficientDataError` on failure rather than
allowing raw exceptions to propagate.
"""

from __future__ import annotations

import logging
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from sagan.exceptions import FetchError, InsufficientDataError

logger = logging.getLogger("sagan")

__all__ = [
    "fetch_prices",
    "prepare_probabilistic_data",
    "validate_tickers",
]

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds


def fetch_prices(
    tickers: List[str],
    years: int = 5,
    retries: int = _MAX_RETRIES,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.

    Automatically retries on transient failures with exponential back-off and
    falls back to the bare ticker symbol when a ``.NS`` suffix is not found.

    Args:
        tickers: List of Yahoo Finance ticker symbols
            (e.g. ``["RELIANCE.NS", "TCS.NS"]``).
        years: Number of years of historical data to fetch. Defaults to 5.
        retries: Maximum number of download attempts before raising
            :class:`~sagan.exceptions.FetchError`. Defaults to 3.

    Returns:
        A :class:`~pandas.DataFrame` with columns = tickers and a
        :class:`~pandas.DatetimeIndex`, forward-filled and stripped of NaN
        rows.

    Raises:
        FetchError: If all retry attempts fail or the result is empty.
        InsufficientDataError: If fewer than 30 trading days are available
            after cleaning.

    Example:
        >>> from sagan.data import fetch_prices
        >>> prices = fetch_prices(["AAPL", "MSFT"], years=2)
        >>> prices.shape
        (502, 2)
    """
    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=years)
    logger.info("Fetching %d year(s) for %d tickers: %s", years, len(tickers), tickers)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                tickers,
                period=f"{years}y",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            break
        except Exception as exc:
            last_exc = exc
            delay = _RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "yfinance download failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt, retries, exc, delay,
            )
            time.sleep(delay)
    else:
        raise FetchError(tickers, last_exc)

    prices = pd.DataFrame()
    for t in tickers:
        logger.info("Extracting data for '%s'...", t)
        try:
            # 1. Try MultiIndex (t, 'Close')
            if (t, "Close") in data.columns:
                prices[t] = data[(t, "Close")]
            # 2. Try simple 'Close' (if single ticker or flattened)
            elif "Close" in data.columns and (len(tickers) == 1 or t not in data.columns):
                prices[t] = data["Close"]
            # 3. Try level 0 access
            elif t in data.columns:
                temp = data[t]
                if isinstance(temp, pd.DataFrame) and "Close" in temp.columns:
                    prices[t] = temp["Close"]
                elif isinstance(temp, pd.Series) and temp.name == "Close":
                    prices[t] = temp
            # 4. Try suffix fallback
            else:
                alt = t.replace(".NS", "")
                if (alt, "Close") in data.columns:
                    prices[t] = data[(alt, "Close")]
        except Exception as exc:
            logger.warning("Could not extract prices for '%s': %s", t, exc)

    if prices.empty:
        raise FetchError(tickers)

    prices = prices.ffill().dropna()

    if len(prices) < 30:
        raise InsufficientDataError(available=len(prices), required=30)

    logger.info("Fetched %d trading days × %d tickers.", len(prices), len(prices.columns))
    return prices


def validate_tickers(tickers: List[str], years: int = 1) -> List[str]:
    """Check which tickers in *tickers* are fetchable from Yahoo Finance.

    This is a lightweight probe that downloads only 1 year of data to verify
    that each ticker resolves. Useful for pre-flight checks before longer
    training runs.

    Args:
        tickers: Ticker symbols to validate.
        years: How many years of data to attempt for validation. Defaults to 1.

    Returns:
        List of valid ticker symbols (those for which data was found).

    Example:
        >>> from sagan.data import validate_tickers
        >>> validate_tickers(["AAPL", "BADTICKER123"])
        ['AAPL']
    """
    valid: list[str] = []
    for t in tickers:
        try:
            df = fetch_prices([t], years=years, retries=1)
            if not df.empty:
                valid.append(t)
        except (FetchError, InsufficientDataError):
            logger.warning("Ticker '%s' could not be validated — skipping.", t)
    return valid


def prepare_probabilistic_data(
    prices: pd.DataFrame,
    window: int,
    horizon: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], int]:
    """Build a supervised sliding-window dataset from price data.

    Each sample ``X[i]`` is a ``(window, n_stocks)`` array of daily percentage
    returns. The label ``y_probs[i]`` is a one-hot vector indicating whether
    the average forward return over *horizon* days exceeded *threshold* (buy),
    fell below *-threshold* (sell), or stayed within bounds (hold).

    Args:
        prices: DataFrame of adjusted close prices. Each column is a ticker.
        window: Number of historical return periods per input sample.
        horizon: Number of future periods used to compute the forward return.
        threshold: Absolute return threshold (fraction) for buy/sell labelling.

    Returns:
        A tuple ``(X, y_probs, y_ret, symbols, n_stocks)`` where:

        - **X** – ``float32`` array of shape ``(N, window, n_stocks)``
        - **y_probs** – ``float32`` one-hot label array of shape ``(N, 3)``
          with columns [buy, sell, hold]
        - **y_ret** – ``float32`` array of shape ``(N,)`` with raw forward returns
        - **symbols** – list of ticker column names
        - **n_stocks** – number of tickers

    Raises:
        InsufficientDataError: If *prices* does not contain enough rows for at
            least one sample given *window* and *horizon*.

    Example:
        >>> import pandas as pd, numpy as np
        >>> prices = pd.DataFrame(np.random.randn(200, 3).cumsum(0) + 100)
        >>> X, y_prob, y_ret, syms, n = prepare_probabilistic_data(prices, 10, 3, 0.01)
        >>> X.shape
        (186, 10, 3)
    """
    returns = prices.pct_change().dropna()
    n_stocks = returns.shape[1]
    symbols = list(returns.columns)

    min_rows = window + horizon + 1
    if len(returns) < min_rows:
        raise InsufficientDataError(available=len(returns), required=min_rows)

    X, y_probs, y_ret = [], [], []
    for i in range(len(returns) - window - horizon):
        X.append(returns.iloc[i : i + window].values)
        future_ret = float(
            returns.iloc[i + window : i + window + horizon].mean().mean()
        )
        y_ret.append(future_ret)
        if future_ret > threshold:
            y_probs.append([1.0, 0.0, 0.0])
        elif future_ret < -threshold:
            y_probs.append([0.0, 1.0, 0.0])
        else:
            y_probs.append([0.0, 0.0, 1.0])

    return (
        np.array(X, dtype=np.float32),
        np.array(y_probs, dtype=np.float32),
        np.array(y_ret, dtype=np.float32),
        symbols,
        n_stocks,
    )
