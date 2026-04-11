"""Unit tests for sagan.data"""

import numpy as np
import pandas as pd
import pytest

from sagan.data import prepare_probabilistic_data


def _make_prices(n_days: int = 100, n_stocks: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, size=(n_days, n_stocks)), axis=0)
    return pd.DataFrame(prices, columns=[f"STOCK_{i}" for i in range(n_stocks)])


class TestPrepareProbabilisticData:
    def test_output_shapes(self):
        prices = _make_prices(n_days=100, n_stocks=3)
        window, horizon, threshold = 10, 3, 0.01
        X, y_probs, y_ret, symbols, n_stocks = prepare_probabilistic_data(
            prices, window, horizon, threshold
        )
        expected_samples = len(prices) - 1 - window - horizon  # pct_change drops 1 row
        assert X.shape == (expected_samples, window, n_stocks)
        assert y_probs.shape == (expected_samples, 3)
        assert y_ret.shape == (expected_samples,)
        assert len(symbols) == n_stocks
        assert n_stocks == 3

    def test_label_sum_to_one(self):
        prices = _make_prices()
        _, y_probs, *_ = prepare_probabilistic_data(prices, 10, 3, 0.01)
        np.testing.assert_allclose(y_probs.sum(axis=1), 1.0, atol=1e-6)

    def test_dtype_float32(self):
        prices = _make_prices()
        X, y_probs, y_ret, *_ = prepare_probabilistic_data(prices, 10, 3, 0.01)
        assert X.dtype == np.float32
        assert y_probs.dtype == np.float32
        assert y_ret.dtype == np.float32

    def test_symbols_match_columns(self):
        prices = _make_prices(n_stocks=4)
        _, _, _, symbols, n = prepare_probabilistic_data(prices, 5, 2, 0.005)
        assert symbols == list(prices.columns)
        assert n == 4

    def test_window_horizon_respected(self):
        prices = _make_prices(n_days=50, n_stocks=2)
        X, y_probs, y_ret, _, _ = prepare_probabilistic_data(prices, 7, 5, 0.01)
        assert X.shape[1] == 7  # window dimension
