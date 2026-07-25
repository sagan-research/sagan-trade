"""
Data Loading and Management Module.

Provides utilities for loading financial data from various sources:
- Yahoo Finance (yfinance)
- CSV files
- Databases
- API endpoints
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


@dataclass
class DataConfig:
    """Configuration for data loading."""

    # Data source
    source: Literal["yfinance", "csv", "database", "api"] = "yfinance"

    # YFinance settings
    period: str = "2y"
    interval: str = "1d"
    auto_adjust: bool = True

    # CSV settings
    csv_path: str | None = None
    csv_index_col: int | None = 0
    csv_parse_dates: bool = True

    # Preprocessing
    fill_method: Literal["ffill", "bfill", "interpolate", "drop"] = "ffill"
    normalize: bool = False

    # Column mapping
    column_map: dict[str, str] = field(
        default_factory=lambda: {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )


class DataLoader:
    """
    Unified data loader for financial time series.

    Supports multiple data sources and preprocessing options.
    """

    def __init__(self, config: DataConfig | None = None):
        self.config = config or DataConfig()
        self._cache: dict[str, pd.DataFrame] = {}

    def load(self, tickers: str | list[str], **kwargs) -> pd.DataFrame:
        """
        Load data for one or more tickers.

        Args:
            tickers: Ticker symbol(s) or file path
            **kwargs: Override config settings

        Returns:
            DataFrame with OHLCV data
        """
        # Merge kwargs with config
        config = DataConfig(**{**self.config.__dict__, **kwargs})

        if isinstance(tickers, str):
            tickers = [tickers]

        # Check cache
        cache_key = "_".join(tickers) + config.period + config.interval
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        if config.source == "yfinance":
            data = self._load_yfinance(tickers, config)
        elif config.source == "csv":
            data = self._load_csv(tickers, config)
        else:
            raise ValueError(f"Unsupported data source: {config.source}")

        # Preprocessing
        data = self._preprocess(data, config)

        # Cache
        self._cache[cache_key] = data

        return data

    def _load_yfinance(self, tickers: list[str], config: DataConfig) -> pd.DataFrame:
        """Load data from Yahoo Finance."""
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance is required for Yahoo Finance data. Install with: pip install yfinance"
            )

        all_data = []

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(
                    period=config.period,
                    interval=config.interval,
                    auto_adjust=config.auto_adjust,
                )

                if df.empty:
                    print(f"Warning: No data found for ticker {ticker}")
                    continue

                # Rename columns
                df = df.rename(columns=config.column_map)

                # Select only OHLCV
                cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                df = df[cols]

                # Add ticker column if multiple
                if len(tickers) > 1:
                    df["ticker"] = ticker

                all_data.append(df)

            except Exception as e:
                print(f"Error loading {ticker}: {e}")
                continue

        if not all_data:
            raise ValueError("No data loaded for any ticker")

        if len(all_data) == 1:
            return all_data[0]
        else:
            # Multi-ticker: return with ticker level
            return pd.concat(all_data, keys=list(tickers[: len(all_data)]), axis=1)

    def _load_csv(self, tickers: list[str], config: DataConfig) -> pd.DataFrame:
        """Load data from CSV file(s)."""
        if config.csv_path is None:
            raise ValueError("csv_path must be specified for CSV source")

        if len(tickers) == 1 and pathlib.Path(config.csv_path).is_file():
            df = pd.read_csv(
                config.csv_path,
                index_col=config.csv_index_col,
                parse_dates=config.csv_parse_dates,
            )
            df = df.rename(columns=config.column_map)
            return df

        # Try loading multiple CSV files
        all_data = []
        for ticker in tickers:
            path = config.csv_path.format(ticker=ticker)
            if pathlib.Path(path).exists():
                df = pd.read_csv(
                    path, index_col=config.csv_index_col, parse_dates=config.csv_parse_dates
                )
                df = df.rename(columns=config.column_map)
                df["ticker"] = ticker
                all_data.append(df)

        if not all_data:
            raise ValueError(f"No CSV files found at {config.csv_path}")

        return pd.concat(all_data)

    def _preprocess(self, data: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        """Apply preprocessing to data."""
        # Fill missing values
        if config.fill_method == "ffill":
            data = data.ffill()
        elif config.fill_method == "bfill":
            data = data.bfill()
        elif config.fill_method == "interpolate":
            data = data.interpolate()
        elif config.fill_method == "drop":
            data = data.dropna()

        # Normalize
        if config.normalize:
            for col in data.select_dtypes(include=[np.number]).columns:
                if col not in ["ticker", "volume"]:
                    data[col] = data[col] / data[col].iloc[0]

        return data

    def get_returns(
        self, tickers: str | list[str], method: Literal["simple", "log"] = "simple", **kwargs
    ) -> pd.DataFrame:
        """Get return series for tickers."""
        data = self.load(tickers, **kwargs)

        close_cols = []
        if isinstance(data.columns, pd.MultiIndex):
            close_cols = [c for c in data.columns if c[1] == "close"]
        elif "close" in data.columns:
            close_cols = ["close"]

        if not close_cols:
            raise ValueError("No 'close' column found in data")

        closes = data[close_cols]

        if method == "simple":
            returns = closes.pct_change()
        elif method == "log":
            returns = np.log(closes / closes.shift(1))
        else:
            raise ValueError(f"Unknown method: {method}")

        return returns.dropna()

    def get_prices(self, tickers: str | list[str], column: str = "close", **kwargs) -> pd.DataFrame:
        """Get price series for tickers."""
        data = self.load(tickers, **kwargs)

        if isinstance(data.columns, pd.MultiIndex):
            price_cols = [c for c in data.columns if c[1] == column]
            return data[price_cols]
        elif column in data.columns:
            return data[column]
        else:
            raise ValueError(f"Column '{column}' not found in data")

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()


def load_data(
    tickers: str | list[str],
    period: str = "2y",
    interval: str = "1d",
    source: str = "yfinance",
    **kwargs,
) -> pd.DataFrame:
    """Convenience function to load data."""
    config = DataConfig(source=source, period=period, interval=interval, **kwargs)
    loader = DataLoader(config)
    return loader.load(tickers)


def load_returns(tickers: str | list[str], method: str = "simple", **kwargs) -> pd.DataFrame:
    """Convenience function to load returns."""
    config = DataConfig(**kwargs)
    loader = DataLoader(config)
    return loader.get_returns(tickers, method=method)
