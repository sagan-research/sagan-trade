import sys
from unittest.mock import MagicMock

# Mock TensorFlow to save memory
mock_tf = MagicMock()
sys.modules["tensorflow"] = mock_tf
sys.modules["tensorflow.keras"] = mock_tf
sys.modules["tensorflow.keras.layers"] = mock_tf
sys.modules["tensorflow.keras.models"] = mock_tf

import time
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sagan.models.math_engine import MathematicalEngine
from sagan.desk import AlphaDesk, run_research_backtest
from sagan.ensemble import SymbolicRegressor
import os

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "META", "AMD", "GS", "JPM", "XOM"]

def benchmark_basket():
    print(f"--- Large-Scale Benchmark (Memory Optimized) ---")
    
    model_ids = []
    for t in TICKERS:
        try:
            print(f"Training {t}...")
            reg = SymbolicRegressor([t], period="2y", profile="eco")
            meta = reg.train()
            mid = reg.save()
            model_ids.append(mid)
        except Exception as e:
            print(f"Failed to train {t}: {e}")
    
    if not model_ids:
        print("No models trained. Exiting.")
        return

    print("\nRunning Backtest...")
    stats = run_research_backtest(TICKERS, model_ids, years=2)
    
    if stats:
        print("\n--- RESULTS ---")
        print(f"Strategy Sharpe: {stats['strategy']['sharpe']:.2f}")
        print(f"Benchmark Sharpe: {stats['benchmark']['sharpe']:.2f}")
        
        with open("Basket_Benchmark_Report.md", "w") as f:
            f.write("# Large-Scale Portfolio Benchmark Report (v0.5.0)\n\n")
            f.write(f"**Tickers:** {', '.join(TICKERS)}\n\n")
            f.write("| Metric | Symbolic (v0.5.0) | Buy & Hold |\n")
            f.write("|:---|:---|:---|\n")
            f.write(f"| **Annual Return** | **{stats['strategy']['annual_return']*100:.2f}%** | {stats['benchmark']['annual_return']*100:.2f}% |\n")
            f.write(f"| **Sharpe Ratio** | **{stats['strategy']['sharpe']:.2f}** | {stats['benchmark']['sharpe']:.2f} |\n")
            f.write(f"| **Max Drawdown** | **{stats['strategy']['mdd']*100:.2f}%** | {stats['benchmark']['mdd']*100:.2f}% |\n")

if __name__ == "__main__":
    benchmark_basket()
