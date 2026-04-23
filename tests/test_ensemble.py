"""Unit tests for sagan.ensemble (offline, no network)"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from sagan.ensemble import SymbolicRegressor

def _fake_data(n_days=200, signals=None):
    if signals is None:
        signals = ["Close", "Volume"]
    rng = np.random.default_rng(0)
    data = {}
    for s in signals:
        data[s] = 100 + np.cumsum(rng.normal(0, 0.5, n_days))
    return pd.DataFrame(data)

class TestSymbolicRegressor:
    @patch("sagan.ensemble.fetch_signal_data")
    @patch("sagan.ensemble.save_model", return_value="sagan_test_abc123")
    @patch("sagan.ensemble.ResourceManager")
    @patch("sagan.ensemble.FunctionGemmaBridge")
    def test_train_returns_metadata(self, mock_llm, mock_rm, mock_save, mock_fetch):
        # Setup mocks
        mock_fetch.return_value = _fake_data()
        mock_rm.return_value.get_worker_count.return_value = 1
        mock_llm.return_value.suggest_composite_function.return_value = "(Close * 0.5)"
        
        # We need to mock fit_signal_worker if it's called via executor, 
        # but here we can just mock the whole train method if it's too complex,
        # or mock the executor.
        
        reg = SymbolicRegressor(tickers=["AAPL"], signals=["Close", "Volume"])
        
        # To avoid actual multiprocessing in tests, we patch the executor
        with patch("sagan.ensemble.ProcessPoolExecutor") as mock_executor:
            mock_exe_inst = mock_executor.return_value.__enter__.return_value
            # Mock the future results
            mock_future_close = MagicMock()
            mock_future_close.result.return_value = ("Close", {"r2": 0.96, "func": "poly"})
            mock_future_vol = MagicMock()
            mock_future_vol.result.return_value = ("Volume", {"r2": 0.94, "func": "fourier"})
            mock_exe_inst.submit.side_effect = [mock_future_close, mock_future_vol]
            
            meta = reg.train()
            
        assert "val_r2" in meta
        assert meta["composite_formula"] == "(Close * 0.5)"
        assert len(meta["fitted_signals"]) == 2

    @patch("sagan.ensemble.fetch_signal_data")
    @patch("sagan.ensemble.save_model", return_value="sagan_test_xyz789")
    def test_save_returns_model_id(self, mock_save, mock_fetch):
        reg = SymbolicRegressor(tickers=["AAPL"])
        reg.meta = {"some": "data"}
        mid = reg.save()
        assert mid == "sagan_test_xyz789"
        mock_save.assert_called_once()
