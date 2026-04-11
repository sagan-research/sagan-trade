"""Unit tests for sagan.ensemble (offline, no network)"""

import numpy as np
import pandas as pd
import pytest

from unittest.mock import patch, MagicMock
from sagan.ensemble import ExplainableEnsemble


def _fake_prices(n_days=200, n_stocks=2, seed=0):
    rng = np.random.default_rng(seed)
    arr = 100 + np.cumsum(rng.normal(0, 0.5, (n_days, n_stocks)), axis=0)
    return pd.DataFrame(arr, columns=[f"S{i}" for i in range(n_stocks)])


class TestExplainableEnsemble:
    @patch("sagan.ensemble.fetch_prices")
    @patch("sagan.ensemble.save_model", return_value="sagan_test_abc123")
    def test_train_returns_metadata(self, mock_save, mock_fetch):
        mock_fetch.return_value = _fake_prices()
        ens = ExplainableEnsemble(
            tickers=["S0", "S1"],
            years=1,
            window=5,
            horizon=2,
            epochs=2,
            verbose=False,
        )
        meta = ens.train()
        assert "val_sharpe" in meta
        assert "tickers" in meta
        assert meta["window"] == 5

    @patch("sagan.ensemble.fetch_prices")
    @patch("sagan.ensemble.save_model", return_value="sagan_test_xyz789")
    def test_save_returns_model_id(self, mock_save, mock_fetch):
        mock_fetch.return_value = _fake_prices()
        ens = ExplainableEnsemble(tickers=["S0", "S1"], years=1, window=5, horizon=2, epochs=2, verbose=False)
        ens.train()
        mid = ens.save()
        assert mid == "sagan_test_xyz789"
        mock_save.assert_called_once()

    @patch("sagan.ensemble.fetch_prices")
    @patch("sagan.ensemble.save_model", return_value="mock_id")
    def test_three_models_built(self, mock_save, mock_fetch):
        mock_fetch.return_value = _fake_prices()
        ens = ExplainableEnsemble(tickers=["S0", "S1"], years=1, window=5, horizon=2, epochs=1, verbose=False)
        ens.train()
        assert ens.model_buy is not None
        assert ens.model_sell is not None
        assert ens.model_hold is not None
        assert ens.scaler is not None
