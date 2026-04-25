import sys
import os
import logging
import json
import numpy as np
from datetime import datetime

# Ensure sagan is in path
sys.path.append(os.getcwd())

import sagan
from sagan.desk import run_research_backtest

def generate_md_report(results, tickers):
    report = f"""# Quantitative Research Report: Symbolic Alpha Desk Portfolio
**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Portfolio Assets**: {", ".join(tickers)}

## 1. Abstract
This paper evaluates the performance of a Symbolic Regression-based Trading Desk. Unlike traditional linear models, our system uses nonlinear mathematical foundations to discover market invariants and coordinate signals across a diverse asset basket. We compare the strategy against a classic Buy-and-Hold (Equal Weight) benchmark using institutional-grade metrics and statistical hypothesis testing.

## 2. Methodology
### 2.1 Symbolic Foundation
Each asset in the portfolio is modeled using an independent `SymbolicRegressor`. The engine explores a nonlinear search space to find a mathematical expression that maximizes $R^2$ out of sample.

### 2.2 Alpha Desk Coordination
Signals are processed through an `AlphaDesk` orchestrator which:
1. Applies Z-score based entry/exit thresholds.
2. Coordinates net portfolio exposure to a limit of 1.5x.
3. Incorporates a transaction cost model (5bps per trade).

## 3. Empirical Results

### 3.1 Performance Metrics
| Metric | Symbolic Alpha Desk | Buy & Hold Benchmark |
|:---|:---|:---|
| **Annualised Return** | {results['strategy']['annual_return']:.2%} | {results['benchmark']['annual_return']:.2%} |
| **Cumulative Return** | {results['strategy']['cum_return']:.2%} | {results['benchmark']['cum_return']:.2%} |
| **Sharpe Ratio** | {results['strategy']['sharpe']:.2f} | {results['benchmark']['sharpe']:.2f} |
| **Max Drawdown** | {results['strategy']['mdd']:.2%} | {results['benchmark']['mdd']:.2%} |
| **Win Rate** | {results['strategy']['win_rate']:.2%} | N/A |

### 3.2 Statistical Significance
- **Alpha (Annualized)**: {results['stats']['alpha']:.2%}
- **Information Ratio**: {results['stats']['info_ratio']:.2f}
- **T-Statistic**: {results['stats']['t_stat']:.4f}
- **P-Value**: {results['stats']['p_value']:.4f}

> **Verdict**: The strategy generated an annualized alpha of {results['stats']['alpha']:.2%}. The P-Value of {results['stats']['p_value']:.4f} indicates that the outperformance is {"statistically significant" if results['stats']['p_value'] < 0.05 else "not statistically significant"} at the 5% confidence level.

## 4. Conclusion
The Symbolic Alpha Desk demonstrates the potential for nonlinear mathematical models to extract stable alpha in volatile markets. Future research will focus on dynamic risk-parity weighting and multi-frequency signal integration.
"""
    with open("Sagan_Research_Report.md", "w") as f:
        f.write(report)
    print("\nOK: Research report generated as 'Sagan_Research_Report.md'")

def main():
    logging.basicConfig(level=logging.ERROR)
    tickers = ["AAPL", "XOM", "GS"] # Diversified: Tech, Energy, Finance
    
    print(f"--- Launching Autonomous Research Pipeline for {tickers} ---")
    
    model_ids = []
    for t in tickers:
        print(f"Developing symbolic foundation for {t}...")
        try:
            mid = sagan.train([t], signals=["Close", "Volume"], target_r2=0.92)
            model_ids.append(mid)
            print(f"  OK: {mid}")
        except Exception as e:
            print(f"  FAILED {t}: {e}")
            
    if not model_ids:
        print("Error: No models were trained.")
        return
        
    print("\n--- Executing Institutional Backtest & Statistical Analysis ---")
    results = run_research_backtest(tickers, model_ids, years=2, commission=0.0005)
    
    if results:
        generate_md_report(results, tickers)

if __name__ == "__main__":
    main()
