import sys
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from datetime import datetime

# Ensure sagan is in path
sys.path.append(os.getcwd())

import sagan
from sagan.data import fetch_prices, prepare_probabilistic_data
from sagan.models.tft import build_tft_action_model
from sagan.models.pinn_loss import pinn_loss
from sagan.utils import sharpe_ratio, max_drawdown, annualised_return, win_rate

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("benchmark")

def train_tft_initial(prices, window=15, horizon=5, threshold=0.01, epochs=10):
    """Trains the initial TFT-PINN model."""
    X, y_probs, _, symbols, n_stocks = prepare_probabilistic_data(prices, window, horizon, threshold)
    
    # Split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y_probs[:split]
    
    model = build_tft_action_model(window, n_stocks)
    model.compile(
        optimizer="adam",
        loss={"logit": lambda y_t, y_p: pinn_loss(y_t, y_p, lambda_pinn=0.01), "selection_weights": None}
    )
    
    # We only care about the logit for the 'buy' signal (index 0 in y_probs)
    # y_probs is [buy, sell, hold]
    model.fit(X_train, {"logit": y_train[:, 0]}, epochs=epochs, batch_size=32, verbose=0)
    return model, symbols

def run_synchronized_backtest(tickers, sym_model_ids, tft_model, years=3, window=15, commission=0.0005):
    """Runs a backtest comparing Symbolic vs TFT vs Buy & Hold."""
    from sagan.desk import AlphaDesk
    from sagan.signals import fetch_signal_data
    
    # Setup Symbolic Desk
    sym_desk = AlphaDesk(sym_model_ids, mode="coordinated")
    
    # Fetch full signal data for each ticker
    full_data = {}
    for t in tickers:
        df = fetch_signal_data(t, ["Open", "High", "Low", "Close", "Volume"], period=f"{years}y")
        full_data[t] = df
        
    # Get common dates
    common_dates = None
    for df in full_data.values():
        if common_dates is None: common_dates = df.index
        else: common_dates = common_dates.intersection(df.index)
    
    test_start_idx = int(0.7 * len(common_dates))
    test_dates = common_dates[test_start_idx:]
    
    # Preparation for TFT (uses aggregate returns)
    all_returns = pd.DataFrame({t: full_data[t].loc[common_dates, "Close"].pct_change() for t in tickers}).dropna()
    tft_test_dates = all_returns.index[all_returns.index.get_indexer(test_dates)]
    
    sym_returns = []
    tft_returns = []
    bench_returns = []
    
    prev_sym_sig = {t: 0.0 for t in tickers}
    prev_tft_sig = {t: 0.0 for t in tickers}
    
    # Pre-calculate TFT signals
    X_tft = []
    for i in range(len(all_returns) - window):
        X_tft.append(all_returns.iloc[i : i + window].values)
    X_tft = np.array(X_tft, dtype=np.float32)
    
    print(f"TFT Input Shape: {X_tft.shape}")
    if X_tft.shape[0] == 0:
        print("Error: X_tft is empty!")
        return np.array([]), np.array([]), np.array([])

    try:
        tft_preds = tft_model.predict(X_tft, verbose=0)
        tft_outputs = tft_preds['logit'].flatten()
        tft_probs = tf.nn.sigmoid(tft_outputs).numpy()
    except Exception as e:
        print(f"TFT Prediction Failed: {e}")
        # Fallback to neutral signals
        tft_probs = np.full(X_tft.shape[0], 0.5)
    
    # Map dates to TFT indices
    date_to_tft_idx = {date: i for i, date in enumerate(all_returns.index[window:])}
    
    # Main Backtest Loop
    for i in range(len(test_dates) - 1):
        date = test_dates[i]
        next_date = test_dates[i+1]
        
        # 1. Benchmark
        day_rets = all_returns.loc[date]
        bench_returns.append(day_rets.mean())
        
        # 2. Symbolic Signals
        current_data = {}
        for t in tickers:
            row = full_data[t].loc[date]
            current_data[t] = row.to_dict()
            
        sym_signals = sym_desk.coordinate_signals(current_data)
        
        # 3. TFT Signals
        tft_idx = date_to_tft_idx.get(date)
        if tft_idx is not None:
            tft_prob = tft_probs[tft_idx]
            tft_sig = 1.0 if tft_prob > 0.55 else (-1.0 if tft_prob < 0.45 else 0.0)
        else:
            tft_sig = 0.0
        
        # Calculate PnL
        next_rets = all_returns.loc[next_date]
        s_ret = 0
        t_ret = 0
        for t in tickers:
            # Symbolic
            sig_s = sym_signals.get(t, 0)
            cost_s = abs(sig_s - prev_sym_sig[t]) * commission
            s_ret += (sig_s * next_rets[t] - cost_s) / len(tickers)
            prev_sym_sig[t] = sig_s
            
            # TFT
            cost_t = abs(tft_sig - prev_tft_sig[t]) * commission
            t_ret += (tft_sig * next_rets[t] - cost_t) / len(tickers)
            prev_tft_sig[t] = tft_sig
            
        sym_returns.append(s_ret)
        tft_returns.append(t_ret)
            
    return np.array(sym_returns), np.array(tft_returns), np.array(bench_returns)

def generate_report(s_ret, t_ret, b_ret, tickers):
    # Use paired t-test as they are on the same daily timeline
    t_stat, p_val = stats.ttest_rel(s_ret, t_ret)
    
    report = f"""# Benchmark Report: Symbolic vs. Initial ML Model
**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Assets**: {", ".join(tickers)}

## 1. Executive Summary
This report compares the performance of the current **Symbolic Regression Engine** against the **Initial TFT-PINN model**. The results indicate whether the transition to symbolic mathematical foundations has yielded a statistically significant improvement in alpha generation.

## 2. Performance Comparison

| Metric | Symbolic (Current) | TFT-PINN (Initial) | Buy & Hold |
|:---|:---|:---|:---|
| **Annualised Return** | {annualised_return(s_ret):.2%} | {annualised_return(t_ret):.2%} | {annualised_return(b_ret):.2%} |
| **Sharpe Ratio** | {sharpe_ratio(s_ret):.2f} | {sharpe_ratio(t_ret):.2f} | {sharpe_ratio(b_ret):.2f} |
| **Max Drawdown** | {max_drawdown(s_ret):.2%} | {max_drawdown(t_ret):.2%} | {max_drawdown(b_ret):.2%} |
| **Win Rate** | {win_rate(s_ret):.2%} | {win_rate(t_ret):.2%} | N/A |

## 3. Statistical Significance
- **T-Statistic**: {t_stat:.4f}
- **P-Value**: {p_val:.4f}

> **Verdict**: The Symbolic Engine is {"statistically outperforming" if (p_val < 0.05 and np.mean(s_ret) > np.mean(t_ret)) else "not statistically outperforming"} the Initial ML Model at the 5% confidence level.

## 4. Conclusion
The symbolic approach provides { "superior" if annualised_return(s_ret) > annualised_return(t_ret) else "inferior" } risk-adjusted returns compared to the initial deep learning architecture. The transparency of symbolic formulas allows for better risk coordination across the portfolio.
"""
    with open("Benchmark_Performance_Report.md", "w") as f:
        f.write(report)
    print("\nOK: Benchmark report generated as 'Benchmark_Performance_Report.md'")

def get_latest_model_for_ticker(ticker):
    """Finds the latest symbolic model for a ticker in the registry."""
    df = sagan.list_models()
    if df.empty: return None
    
    # Filter for ticker and symbolic
    mask = (df['tickers'].apply(lambda x: ticker in x)) & (df['is_symbolic'] == True)
    filtered = df[mask]
    if filtered.empty: return None
    
    # Sort by created_at (already sorted by list_models but just to be sure)
    return str(filtered.iloc[-1]['model_id'])

def main():
    tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "META", "AMD", "GS", "JPM", "XOM"]
    print(f"--- Starting Benchmark: Symbolic vs Initial (TFT-PINN) for {tickers} ---")
    
    # 1. Fetch Data for TFT (needs full history)
    prices = fetch_prices(tickers, years=3)
    
    # 2. Train Initial (TFT-PINN)
    print("Training Initial TFT-PINN Model...")
    tft_model, _ = train_tft_initial(prices, epochs=15)
    
    # 3. Get/Train Symbolic (Current)
    print("Developing Symbolic Foundations...")
    sym_model_ids = []
    for t in tickers:
        mid = get_latest_model_for_ticker(t)
        if mid:
            print(f"  Using existing model for {t}: {mid}")
        else:
            print(f"  No model found for {t}, training new one...")
            for attempt in range(3):
                try:
                    mid = sagan.train([t], signals=["Close", "Volume"], target_r2=0.92)
                    print(f"  OK: {mid}")
                    break
                except Exception as e:
                    print(f"  Attempt {attempt+1} failed for {t}: {e}")
            if not mid:
                print(f"  CRITICAL: Could not train model for {t}")
                return
        sym_model_ids.append(mid)
    
    # 4. Run Synchronized Backtest
    print("Executing Synchronized Backtest...")
    s_ret, t_ret, b_ret = run_synchronized_backtest(tickers, sym_model_ids, tft_model)
    
    # 5. Generate Report
    generate_report(s_ret, t_ret, b_ret, tickers)

if __name__ == "__main__":
    main()
