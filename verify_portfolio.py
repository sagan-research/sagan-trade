import sagan
import pandas as pd
import numpy as np
from sagan.parallel import train_parallel
from sagan.desk import AlphaDesk, run_research_backtest

def verify_portfolio_stack():
    tickers = ["MSFT", "GOOGL"]
    print(f"Starting Portfolio Stack Verification for {tickers}...")
    
    # 1. Parallel Training
    model_ids = train_parallel(tickers, profile="turbo")
    print(f"Models trained: {model_ids}")
    
    # 2. Backtest with Vectorized Desk
    m_list = list(model_ids.values())
    print(f"Running backtest with models: {m_list}")
    results = run_research_backtest(tickers, m_list, years=1)
    
    if results:
        print("\n--- Portfolio Results ---")
        print(f"Strategy Annual Return: {results['strategy']['annual_return']:.2%}")
        print(f"Benchmark Annual Return: {results['benchmark']['annual_return']:.2%}")
        print(f"Alpha: {results['stats']['alpha']:.2%}")
        print(f"T-Stat: {results['stats']['t_stat']:.4f} (P-value: {results['stats']['p_value']:.4f})")
    else:
        print("Backtest failed or returned no results.")

if __name__ == "__main__":
    verify_portfolio_stack()
