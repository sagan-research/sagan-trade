# Standard Benchmark: Vanilla Symbolic vs. Initial ML
**Date**: 2026-04-25
**Assets**: AAPL, NVDA, TSLA, MSFT, GOOGL, META, AMD, GS, JPM, XOM

## 1. Executive Summary
This report evaluates the **Vanilla Symbolic Engine** against the **Initial TFT-PINN model**. This represents the production-standard configuration of the Sagan Trading Desk.

## 2. Performance Comparison

| Metric | Symbolic (Vanilla) | TFT-PINN (Initial) | Buy & Hold |
|:---|:---|:---|:---|
| **Annualised Return** | 11.84% | -37.52% | 52.19% |
| **Sharpe Ratio** | 2.46 | -2.47 | 2.39 |
| **Max Drawdown** | -3.09% | -35.10% | -12.06% |
| **Win Rate** | 57.78% | 42.22% | N/A |

## 3. Statistical Significance
- **P-Value (Vanilla vs ML)**: 0.0206

> **Verdict**: The Vanilla Symbolic Model is statistically outperforming the Initial ML Model.

## 4. Conclusion
The symbolic engine provides a superior risk-adjusted return profile with significantly lower drawdown than traditional ML approaches.
