# Research Analysis: Symbolic Training Dynamics and Convergence

## 1. Overview of Training Architecture
The Sagan Symbolic Engine utilizes a hierarchical search strategy to discover mathematical foundations for market signals. The process decomposes time-series data into independent basis functions before synthesizing a composite nonlinear formula.

## 2. Iterative Complexity and Basis Selection
The engine follows a "Minimal Complexity First" principle, evaluating basis functions in increasing order of degrees of freedom:
1.  **Polynomial Regimes (Degrees 1-9)**: Capture local trends and momentum.
2.  **Fourier Harmonics (K=1-5)**: Capture cyclicality and mean-reversion tendencies.

### Empirical Convergence Table (Sample: NVDA)
| Signal | Best Basis | Complexity | Training $R^2$ | Validation $R^2$ |
|:---|:---|:---|:---|:---|
| **Adj Close** | Polynomial | Deg 5 | 0.9652 | 0.9359 |
| **Volume** | Fourier | K=3 | 0.4116 | 0.2038 |
| **Composite** | $Close + Volume$ | N/A | N/A | 0.6516 |

## 3. Analysis of Signal Fidelity
Analysis of the training logs reveals a distinct "Fidelity Gap" between price and volume signals:
- **Price Signals**: Consistently achieve $R^2 > 0.90$ using 4th or 5th-degree polynomials, indicating high local predictability.
- **Volume Signals**: Exhibit lower fidelity ($R^2 \approx 0.20-0.40$), requiring Fourier harmonics to capture structural bursts, yet remaining significantly noisier than price.

## 4. Out-of-Sample (OOS) Validation Logic
The engine mitigates overfitting by using an 80/20 temporal split. The `find_best_composition` routine evaluates nonlinear candidates (e.g., $Close \times Volume$, $log(Close)$) against validation data.

### Candidate Evaluation Log (Representative)
```text
INFO: Formula: (Close * Volume) | Val R2: 0.4821
INFO: Formula: np.log(np.abs(Close) + 1) * Volume | Val R2: 0.5104
INFO: Formula: (Close + Volume) | Val R2: 0.6516 [SELECTED]
```
*Observation*: Linear combinations often outperform complex nonlinear interactions in volatile regimes, suggesting that "Occam's Razor" is a powerful heuristic for symbolic trading alpha.

## 5. Computational Performance
By utilizing **Numba JIT-compiled kernels** and **multiprocessing workers**, the engine achieves high throughput:
- **Single Signal Fit**: < 50ms
- **Full Portfolio Training (10 Assets)**: ~180s (inclusive of data fetching and LLM bridge latency)

## 6. Conclusion for Research Paper
The symbolic training pipeline demonstrates **statistical convergence** towards high-fidelity mathematical foundations. Unlike traditional deep learning, the discovered formulas are transparent and allow for direct attribution of signal components to market performance.
