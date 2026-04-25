import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import logging

# Ensure sagan is in path
sys.path.append(os.getcwd())

import sagan
from sagan.ensemble import SymbolicRegressor
from sagan.models.math_engine import MathematicalEngine
from sagan.utils import sharpe_ratio, max_drawdown, annualised_return, win_rate

def run_backtest(ticker: str = "AAPL", years: int = 2):
    print(f"=== Starting Backtest for {ticker} ===")
    
    # 1. Fetch Data
    print(f"Fetching {years} years of data for {ticker}...")
    data = yf.download(ticker, period=f"{years}y", progress=False, auto_adjust=True)
    if data.empty:
        print("Error: No data fetched.")
        return
    
    # Handle both MultiIndex and SingleIndex
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten MultiIndex if necessary or just access the Close column
        # If single ticker, it might be (ticker, 'Close')
        if ticker in data.columns.levels[0]:
            data = data[ticker]
    
    if 'Close' not in data.columns:
        print(f"Error: Could not find 'Close' column. Available: {data.columns}")
        return

    # Pre-calculate returns
    data['Returns'] = data['Close'].pct_change()
    data = data.dropna()
    
    # Split data: 50% train, 50% test
    split_idx = len(data) // 2
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"Training set: {len(train_data)} days")
    print(f"Testing set: {len(test_data)} days")
    
    # 2. Train Symbolic Model
    print("Training Symbolic Regressor...")
    # We use SymbolicRegressor directly to get the meta
    reg = SymbolicRegressor(tickers=[ticker], signals=["Close", "Volume"], target_r2=0.92)
    meta = reg.train()
    
    formula = meta["composite_formula"]
    fitted = meta["fitted_signals"]
    print(f"Discovered Formula: {formula}")
    
    # 3. Simulate Trading on Test Data
    print("Running simulation...")
    
    # For simplicity, we evaluate the formula at each step of the test data
    # Note: In a real scenario, we'd use a sliding window of features.
    # Here we simulate a simplified daily signal based on the fitted functions.
    
    test_returns = []
    signals = []
    
    # Time indices for evaluation (relative to the training period or just step-by-step)
    # The fitted model uses 't' as input. We'll use the index of the test data.
    t_test = np.arange(split_idx, split_idx + len(test_data))
    
    # Pre-evaluate signal values
    signal_values_history = {}
    for s_name, f_meta in fitted.items():
        signal_values_history[s_name] = MathematicalEngine.evaluate(f_meta["func"], t_test, f_meta["params"])
    
    for i in range(len(test_data)):
        # Construct evaluation context
        eval_context = {s: signal_values_history[s][i] for s in fitted}
        eval_context.update({"np": np, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos})
        
        try:
            clean_formula = formula.replace("^", "**")
            val = eval(clean_formula, {"__builtins__": {}}, eval_context)
            signal = 1 if val > 0 else -1 # 1 for LONG, -1 for SHORT
        except:
            signal = 0
            
        signals.append(signal)
        
        # Calculate strategy return
        # Signal acts on next day's return
        if i < len(test_data) - 1:
            daily_ret = signal * test_data['Returns'].iloc[i+1]
            test_returns.append(daily_ret)
        else:
            test_returns.append(0) # Last day
            
    test_data['Strategy_Returns'] = test_returns
    test_data['Signal'] = signals
    
    # 4. Performance Reporting
    strategy_rets = np.array(test_returns)
    benchmark_rets = test_data['Returns'].values
    
    s_sharpe = sharpe_ratio(strategy_rets)
    b_sharpe = sharpe_ratio(benchmark_rets)
    
    s_mdd = max_drawdown(strategy_rets)
    b_mdd = max_drawdown(benchmark_rets)
    
    s_ann = annualised_return(strategy_rets)
    b_ann = annualised_return(benchmark_rets)
    
    s_win = win_rate(strategy_rets)
    
    # 5. Statistical Tests
    # T-test for mean return significantly > 0
    t_stat, p_val = stats.ttest_1samp(strategy_rets, 0)
    
    # Information Ratio (Strategy vs Benchmark)
    active_returns = strategy_rets - benchmark_rets
    ir = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252) if np.std(active_returns) != 0 else 0
    
    print("\n" + "="*40)
    print(f"BACKTEST RESULTS: {ticker}")
    print("="*40)
    print(f"{'Metric':<20} | {'Strategy':<10} | {'Benchmark':<10}")
    print("-" * 45)
    print(f"{'Annualised Return':<20} | {s_ann:>10.2%} | {b_ann:>10.2%}")
    print(f"{'Sharpe Ratio':<20} | {s_sharpe:>10.2f} | {b_sharpe:>10.2f}")
    print(f"{'Max Drawdown':<20} | {s_mdd:>10.2%} | {b_mdd:>10.2%}")
    print(f"{'Win Rate':<20} | {s_win:>10.2%} | {'N/A':>10}")
    print("-" * 45)
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value:     {p_val:.4f} ({'Significant' if p_val < 0.05 else 'Not Significant'} at 5%)")
    print(f"Information Ratio: {ir:.4f}")
    print("="*40)

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run_backtest(ticker)
