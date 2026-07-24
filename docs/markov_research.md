# Markov Chain Research for Quantitative Finance

This document synthesizes recent ArXiv research into Markov Chain modeling for quantitative trading and financial modeling, explicitly tailored for the `sagan-trade` engine.

## 1. Hybrid Hidden Markov Model for Modeling Equity Excess Growth Rate Dynamics
*Abdulrahman Alswaidan, Jeffrey D. Varner*

Existing regime-switching models struggle to capture heavy-tailed distributions, negligible linear autocorrelation, and persistent volatility clustering simultaneously. 
**Key Innovations**:
- **Discrete-State Approach**: Discretizes excess growth rates into Laplace quantile-defined states.
- **Jump-Diffusion Augmentation**: A Poisson jump-duration mechanism enforces realistic tail-state dwell times.
- **Direct Transition Counting**: Parameters are estimated directly rather than using the Baum-Welch EM algorithm, allowing it to scale efficiently.

In `sagan-trade`, this inspires the `HybridHiddenMarkovModel`, focusing on discretizing asset returns and using direct transition counts for rapid fitting across multiple assets.

## 2. Multi-Regime Markov-Switching Models with Time-Varying Transition Probabilities
*Samuel Modée, Yushu Li, Sjur Westgaard, Stein Andreas Bethuelsen*

Standard MS models assume constant transition probabilities. This research extends MS models by incorporating time-varying transition probabilities (TVTP), especially via Generalized Autoregressive Score (GAS) models.
**Key Takeaways**:
- **Regime Identification**: While regime means and variances are reliably recovered, TVTP driving coefficients are harder to identify.
- **Forecasting Robustness**: One-step point forecasts are robust to TVTP misspecifications, but filtered regime probabilities are highly sensitive.
- **Application**: Useful for defining market regimes (e.g., U.S. Treasury yields) where the transition probability is driven by exogenous lagged variables.

In `sagan-trade`, this supports the `MarkovRegimeSwitcher` which can incorporate lagged market indicators (like yield changes) as exogenous variables to predict the probability of transitioning from a bull to a bear regime.

## Implementation in Sagan-Trade

We provide two core models in `sagan.markov_models`:
1. `HybridHiddenMarkovModel`: A fast, discrete-state HMM that bypasses Baum-Welch by using quantile-based discretizations and direct transition matrix counting.
2. `MarkovRegimeSwitcher`: A predictive model that identifies current market regimes (e.g., High Volatility vs Low Volatility) based on observable financial sequences.

These models are built to easily integrate into the broader `sagan` ensemble for predictive backtesting and live trading.
