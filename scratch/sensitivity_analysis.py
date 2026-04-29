import os
import sys
import pandas as pd
from sagan.parallel import train_parallel
from sagan.desk import run_research_backtest

def main():
    tickers = ["RELIANCE.NS", "HDFC.NS", "INFY.NS", "TCS.NS", "ITC.NS"]
    print(f"Starting Sensitivity Analysis for {tickers}...")
    
    print("1. Training models...")
    model_ids = train_parallel(tickers, profile="turbo")
    m_list = list(model_ids.values())
    
    commissions = {
        "5 bps": 0.0005,
        "10 bps": 0.0010,
        "20 bps": 0.0020
    }
    
    results = {}
    for name, comm in commissions.items():
        print(f"\n--- Running backtest with {name} trading cost ({comm}) ---")
        res = run_research_backtest(tickers, m_list, years=1, commission=comm)
        if res:
            results[name] = resimpl
        else:
            print(f"Backtest failed for {name}.")
            
    print("\n=======================================================")
    print("            SENSITIVITY ANALYSIS SUMMARY               ")
    print("=======================================================")
    print(f"{'Cost':<10} | {'Annual Return':<15} | {'Alpha':<10} | {'Sharpe':<10}")
    print("-" * 55)
    
    for name in commissions.keys():
        if name in results:
            res = results[name]
            ret = f"{res['strategy']['annual_return']:.2%}"
            alpha = f"{res['stats']['alpha']:.2%}"
            sharpe = f"{res['strategy']['sharpe']:.2f}"
            print(f"{name:<10} | {ret:<15} | {alpha:<10} | {sharpe:<10}")
            
    # Calculate range
    if len(results) >= 2:
        returns = [res['strategy']['annual_return'] for res in results.values()]
        print("-" * 55)
        print(f"Return Range: {min(returns):.2%} to {max(returns):.2%}")
        print(f"Sensitivity (Max-Min): {max(returns) - min(returns):.2%}")

if __name__ == "__main__":
    main()