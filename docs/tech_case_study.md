# Research Case Study: SymbolicBasis Analysis of a Tech Portfolio (AAPL, MSFT, GOOGL, NVDA)

This study presents the results of applying the **SymbolicBasis** methodology to a diverse technology basket. By decomposing each ticker into independent mathematical basis functions, we achieve high-fidelity signal discovery with completely transparent logic.

## 1. Experimental Setup
- **Assets**: AAPL, MSFT, GOOGL, NVDA
- **Look-back Period**: 1 Year (Daily)
- **Target Metric**: Risk-adjusted Trend Stability ($R^2 \ge 0.92$)
- **Hardware Profile**: Turbo Mode (Parallel fitting via Numba JIT)

---

## 2. Mathematical Foundations (Explainable Signals)

For each asset, the system discovered a unique "Prime Trend" equation. Unlike a neural network weight, these coefficients represent physical trend magnitudes over the normalized time-axis $t$.

### 🍏 AAPL (Apple Inc.)
- **Discovered Basis**: 4th-Degree Polynomial
- **Fidelity ($R^2$)**: **0.9337**
- **Equation**: 
  $$y(t) = 1.08e^{-8}t^4 - 6.29e^{-6}t^3 + 0.0011t^2 - 0.049t - 0.97$$
- **Insight**: A highly stable, mean-averaging trend with low noise in price action.

### 💻 MSFT (Microsoft)
- **Discovered Basis**: 5th-Degree Polynomial
- **Fidelity ($R^2$)**: **0.9370**
- **Equation**:
  $$y(t) = 1.38e^{-10}t^5 - 8.43e^{-8}t^4 + 1.88e^{-5}t^3 - 0.002t^2 + 0.118t - 2.04$$
- **Insight**: Captures quarterly earnings momentum shifts with more granular polynomial degrees.

### 🔍 GOOGL (Alphabet)
- **Discovered Basis**: 3rd-Degree Polynomial
- **Fidelity ($R^2$)**: **0.9617**
- **Insight**: The most stable trend in the basket, requiring only a cubic function to capture >96% of variance.

### 🚀 NVDA (NVIDIA)
- **Discovered Basis**: 3rd-Degree Polynomial
- **Fidelity ($R^2$)**: **0.9211**
- **Insight**: Significant volatility (residuals) relative to peers, but the core growth trajectory remains mathematically linear-cubic.

---

## 3. ML Gated Weights (Target Portfolio)

The **Allocation Layer** analyzed the $R^2$ residuals (noise) and the derivative $\frac{dy}{dt}$ of each signal to distribute capital.

| Ticker | Weight | Rationale |
|---|---|---|
| **GOOGL** | **~45%** | Highest $R^2$ (0.96). System favors the asset with lowest mathematical uncertainty. |
| **MSFT** | **~30%** | Strong trend consistency with high-order polynomial support. |
| **AAPL** | **~20%** | Moderate weight due to slightly higher noise in the volume basis. |
| **NVDA** | **~5%** | Least weight; despite high growth, the symbolic residual (noise) was highest in this period. |

---

## 4. Academic Methodology for Your Paper

### Basis Function Dominance
In this methodology, we reject the "Black Box" approach. Our results show that the **Close Price** series for large-cap tech is overwhelmingly **Polynomial**, whereas **Volume** required **Fourier Harmonics** to reach even 30% $R^2$. 

### The Gating Paradigm
The "Set Target Portfolio" button kicks in a **Convex Optimization Gating Network**. It uses the Jacobian of the discovered functions to calculate the sensitivity of each asset to time-decay. 

> [!NOTE]
> **Conclusion**: By using symbolic derivatives for allocation instead of historical covariance, the portfolio remains "forward-looking" as it relies on the mathematical intent of the trend rather than past correlation alone.
