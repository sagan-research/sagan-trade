"""Unit tests for sagan.utils financial metrics."""

import numpy as np
import pytest

from sagan.utils import (
    annualised_return,
    calmar_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)


class TestSharpeRatio:
    def test_positive_returns(self):
        rets = np.full(252, 0.001)
        s = sharpe_ratio(rets)
        assert s > 0

    def test_zero_std_returns_zero(self):
        rets = np.zeros(100)
        assert sharpe_ratio(rets) == 0.0

    def test_negative_returns_negative_sharpe(self):
        rets = np.full(252, -0.002)
        assert sharpe_ratio(rets) < 0


class TestMaxDrawdown:
    def test_always_negative_or_zero(self):
        rets = np.random.default_rng(0).normal(0.001, 0.01, 200)
        assert max_drawdown(rets) <= 0

    def test_monotone_positive_returns_zero_dd(self):
        rets = np.full(100, 0.01)
        assert max_drawdown(rets) == pytest.approx(0.0, abs=1e-6)


class TestAnnualisedReturn:
    def test_empty_array(self):
        assert annualised_return(np.array([])) == 0.0

    def test_constant_positive(self):
        rets = np.full(252, 0.001)
        ar = annualised_return(rets)
        # (1.001)^252 - 1 ≈ 0.288
        assert 0.25 < ar < 0.35


class TestCalmarRatio:
    def test_zero_drawdown_returns_zero(self):
        rets = np.full(100, 0.01)
        assert calmar_ratio(rets) == 0.0

    def test_positive_for_positive_strategy(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.005, 252)
        c = calmar_ratio(rets)
        # With positive mean, Calmar should be positive
        assert c > 0


class TestWinRate:
    def test_empty_array(self):
        assert win_rate(np.array([])) == 0.0

    def test_all_positive(self):
        assert win_rate(np.ones(10)) == 1.0

    def test_half_positive(self):
        rets = np.array([1, -1, 1, -1], dtype=float)
        assert win_rate(rets) == pytest.approx(0.5)


class TestProfitFactor:
    def test_no_losses_returns_inf(self):
        rets = np.ones(10)
        assert profit_factor(rets) == float("inf")

    def test_balanced_gains_losses(self):
        rets = np.array([1.0, -1.0])
        assert profit_factor(rets) == pytest.approx(1.0)
