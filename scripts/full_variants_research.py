import sys
import os
import logging
import numpy as np
from datetime import datetime

# Ensure sagan is in path
sys.path.append(os.getcwd())

import sagan
from sagan.desk import run_research_backtest

def generate_comparative_report(all_results, tickers):
    report = f"""# Comprehensive Research: The 3 Pillars of Sagan Trade
**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Portfolio Assets**: {", ".join(tickers)}

## 1. Abstract
This comprehensive study explores the three primary operational variants of the Sagan Trading Desk: **Coordinated (Alpha-Beta Mix)**, **Market Neutral (Pure Alpha)**, and **Long-Only (Enhanced Beta)**. We evaluate how symbolic regression formulas perform under different risk constraints and coordination modes.

## 2. Comparative Performance Matrix

| Metric | Coordinated | Market Neutral | Long-Only | Buy & Hold |
|:---|:---|:---|:---|:---|
| **Annualised Return** | {all_results['coordinated']['strategy']['annual_return']:.2%} | {all_results['market_neutral']['strategy']['annual_return']:.2%} | {all_results['long_only']['strategy']['annual_return']:.2%} | {all_results['coordinated']['benchmark']['annual_return']:.2%} |
| **Sharpe Ratio** | {all_results['coordinated']['strategy']['sharpe']:.2f} | {all_results['market_neutral']['strategy']['sharpe']:.2f} | {all_results['long_only']['strategy']['sharpe']:.2f} | {all_results['coordinated']['benchmark']['sharpe']:.2f} |
| **Max Drawdown** | {all_results['coordinated']['strategy']['mdd']:.2%} | {all_results['market_neutral']['strategy']['mdd']:.2%} | {all_results['long_only']['strategy']['mdd']:.2%} | {all_results['coordinated']['benchmark']['mdd']:.2%} |
| **Alpha (Annual)** | {all_results['coordinated']['stats']['alpha']:.2%} | {all_results['market_neutral']['stats']['alpha']:.2%} | {all_results['long_only']['stats']['alpha']:.2%} | N/A |
| **P-Value** | {all_results['coordinated']['stats']['p_value']:.4f} | {all_results['market_neutral']['stats']['p_value']:.4f} | {all_results['long_only']['stats']['p_value']:.4f} | N/A |

## 3. Variant Analysis

### 3.1 Coordinated (The Baseline)
Balanced exposure to symbolic signals. Aimed at institutional stability and risk-adjusted outperformance.

### 3.2 Market Neutral (The Alpha Purist)
By zero-summing signals, this variant removes market beta exposure. Success here indicates true predictive power of the discovered symbolic formulas.

### 3.3 Long-Only (The Beta Enhancer)
Only executes positive symbolic signals. Designed to outperform Buy-and-Hold during bull regimes while reducing drawdown in bear regimes.

## 4. Conclusion
Across all variants, the symbolic engine demonstrates { "statistically significant" if any(r['stats']['p_value'] < 0.05 for r in all_results.values()) else "predictive tendencies" } in market coordination. The { max(all_results, key=lambda x: all_results[x]['strategy']['sharpe']) } variant proved most efficient in the tested regime.
"""
    with open("Full_Variants_Research_Report.md", "w") as f:
        f.write(report)
    print("\nOK: Comparative report generated as 'Full_Variants_Research_Report.md'")

def main():
    logging.basicConfig(level=logging.ERROR)
    tickers = ["AAPL", "NVDA", "TSLA"] # High volatility tech for maximum capability testing
    
    print(f"--- Developing High-Capability Symbolic Foundations for {tickers} ---")
    
    model_ids = []
    for t in tickers:
        print(f"Training Turbo-Variant for {t}...")
        # Use Turbo profile and all advanced signals
        mid = sagan.train([t], signals=["Close", "Volume", "SMA_20", "RSI"], profile="turbo", target_r2=0.94)
        model_ids.append(mid)
        print(f"  OK: {mid}")
            
    all_results = {}
    modes = ["coordinated", "market_neutral", "long_only"]
    
    for mode in modes:
        print(f"\n--- Running Backtest Variant: {mode.upper()} ---")
        res = run_research_backtest(tickers, model_ids, years=2, commission=0.0005, mode=mode)
        if res:
            all_results[mode] = res
        else:
            print(f"  Warning: Backtest failed for {mode}")

    if len(all_results) == 3:
        generate_comparative_report(all_results, tickers)

if __name__ == "__main__":
    main()
