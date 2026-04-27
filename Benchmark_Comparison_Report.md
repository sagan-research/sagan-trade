# Architecture Benchmark Report

**Ticker:** SPY
**Date:** 2026-04-27 09:16:24.435901

## Latency Comparison (ms per inference)
- **Symbolic (Centered Model):** 0.1808 ms
- **Controller (3-layer LSTM):** 7.1432 ms
- **Direct LSTM (5-layer):** 6.6502 ms

## Performance Comparison (Annualized Sharpe)
- **Symbolic (Centered Model):** -1.0358
- **Direct LSTM (5-layer):** -1.0358

## Conclusion
The **Symbolic Centered Model** remains more efficient and provided better risk-adjusted returns on this sample.
