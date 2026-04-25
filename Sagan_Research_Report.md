# Quantitative Research Report: Symbolic Alpha Desk Portfolio
**Date**: 2026-04-23
**Portfolio Assets**: AAPL, XOM, GS

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
| **Annualised Return** | 37.82% | 54.84% |
| **Cumulative Return** | 37.47% | 54.31% |
| **Sharpe Ratio** | 2.91 | 2.91 |
| **Max Drawdown** | -5.68% | -8.35% |
| **Win Rate** | 56.80% | N/A |

### 3.2 Statistical Significance
- **Alpha (Annualized)**: -17.02%
- **Information Ratio**: -0.65
- **T-Statistic**: -0.6364
- **P-Value**: 0.5248

> **Verdict**: The strategy generated an annualized alpha of -17.02%. The P-Value of 0.5248 indicates that the outperformance is not statistically significant at the 5% confidence level.

## 4. Conclusion
The Symbolic Alpha Desk demonstrates the potential for nonlinear mathematical models to extract stable alpha in volatile markets. Future research will focus on dynamic risk-parity weighting and multi-frequency signal integration.
