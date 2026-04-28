import sys
from unittest.mock import MagicMock

# Mock TensorFlow
mock_tf = MagicMock()
sys.modules["tensorflow"] = mock_tf
sys.modules["tensorflow.keras"] = mock_tf

import numpy as np
import pandas as pd
import yfinance as yf
from sagan.models.math_engine import MathematicalEngine
from sagan.desk import AlphaDesk, run_research_backtest
from sagan.ensemble import SymbolicRegressor
import os

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "META", "AMD", "GS", "JPM", "XOM"]

def debug_benchmark():
    print(f"--- Debugging Large-Scale Benchmark ---")
    
    model_ids = []
    for t in TICKERS:
        print(f"Training {t}...")
        # We use returns for fitting to be more predictive
        reg = SymbolicRegressor([t], signals=["Close", "Volume"], period="2y", profile="eco")
        reg.train()
        mid = reg.save()
        model_ids.append(mid)
        
    print("\nInitializing AlphaDesk...")
    desk = AlphaDesk(model_ids, mode="coordinated")
    
    # Check thresholds
    print("\nThresholds:")
    for t, thresh in desk.thresholds.items():
        print(f"  {t}: Buy > {thresh['buy']:.4f}, Sell < {thresh['sell']:.4f}")
        
    # Run backtest
    print("\nRunning Backtest...")
    stats = run_research_backtest(TICKERS, model_ids, years=2)
    
    if stats:
        print("\n--- RESULTS ---")
        print(f"Strategy Ann. Return: {stats['strategy']['annual_return']*100:.2f}%")
        print(f"Benchmark Ann. Return: {stats['benchmark']['annual_return']*100:.2f}%")
        print(f"Sharpe: {stats['strategy']['sharpe']:.2f}")

if __name__ == "__main__":
    debug_benchmark()
