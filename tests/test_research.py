import pytest
from sagan.research import BacktestEngine
import numpy as np

def test_backtest_engine_basic():
    # Use a real ticker but short period for speed
    engine = BacktestEngine("AAPL", "(Close / RSI) * Volume", period="1y")
    results = engine.run()
    
    assert results["status"] == "success"
    assert "total_return" in results
    assert "sharpe" in results
    assert len(results["equity_curve"]) > 0
    assert len(results["dates"]) == len(results["equity_curve"])

def test_backtest_engine_invalid_formula():
    engine = BacktestEngine("AAPL", "Invalid + Formula + 123", period="1y")
    results = engine.run()
    
    assert results["status"] == "error"
    assert "message" in results

def test_backtest_engine_complex_formula():
    # Test with math functions
    engine = BacktestEngine("TSLA", "np.sin(Close) * np.log(abs(Volume) + 1)", period="1y")
    results = engine.run()
    
    assert results["status"] == "success"
    assert results["ticker"] == "TSLA"
