import pandas as pd
import numpy as np
from sagan.research import BacktestEngine
from sagan.signals import fetch_signal_data
from sagan.autonomous import AutonomousResearcher

def run_hmean_comparison():
    tickers = ["AAPL", "MSFT"]
    period = "2y"
    results = []
    
    from sagan.models.manager import ResourceManager
    mgr = ResourceManager("eco")
    mgr.apply_optimizations()
    
    researcher = AutonomousResearcher()
    
    for ticker in tickers:
        print(f"\n--- Testing {ticker} ---")
        
        # 1. Get Symbolic Results (Optimized)
        res_auto = researcher.run_full_pipeline(ticker, period=period)
        sym_return = res_auto['backtest']['total_return']
        sym_sharpe = res_auto['backtest']['sharpe']
        sym_formula = res_auto['formula']
        
        # 2. Get HMean_50 Baseline
        # We manually run a backtest with a simple HMean formula: "(Adj Close > HMean_50) * 2 - 1"
        # This converts a boolean to 1 (Long) or -1 (Short)
        hmean_formula = "(Adj_Close - HMean_50) / HMean_50"
        
        # Fetch data with HMean_50
        data = fetch_signal_data(ticker, ["Adj Close", "HMean_50"], period=period)
        
        engine_hmean = BacktestEngine(ticker, hmean_formula, period=period)
        res_hmean = engine_hmean.run()
        
        h_return = res_hmean['total_return']
        h_sharpe = res_hmean['sharpe']
        
        results.append({
            "Ticker": ticker,
            "Sym Return": sym_return,
            "Sym Sharpe": sym_sharpe,
            "HMean Return": h_return,
            "HMean Sharpe": h_sharpe,
            "Better": "Symbolic" if sym_return > h_return else "HMean"
        })
        
        print(f"Symbolic: {sym_return:.2%} (Sharpe: {sym_sharpe:.2f})")
        print(f"HMean:    {h_return:.2%} (Sharpe: {h_sharpe:.2f})")

    df_results = pd.DataFrame(results)
    print("\n--- Summary Results ---")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    run_hmean_comparison()
