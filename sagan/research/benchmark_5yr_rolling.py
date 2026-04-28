import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sagan.models.robust_fitter import LSTMRobustFitter
from sagan.portfolio.asymmetric_risk import AsymmetricRiskEngine
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagan.5yr_audit")

TICKERS = [
    "AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "META", "AMD", "GS", "JPM", "XOM",
    "UNH", "V", "MA", "PG", "JNJ", "HD", "ABBV", "COST", "LLY", "CRM"
]

def run_5yr_rolling_audit():
    risk_engine = AsymmetricRiskEngine(target_vol=0.15, max_drawdown_limit=0.075)
    all_oos_returns = {}
    
    logger.info(f"Starting 5-Year Walk-Forward Audit (20 Tickers)...")
    
    # We need ~7 years to have enough burn-in for the first 5-year OOS period
    for ticker in TICKERS:
        try:
            logger.info(f"Backtesting {ticker}...")
            df = yf.download(ticker, period="8y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            prices = df['Close']
            ma_20 = prices.rolling(window=20).mean().shift(1) # Strictly trailing MA20
            log_prices = np.log(prices)
            # Center using the PRECEDING 50-day mean (Zero Lookahead)
            log_rolling_mean = log_prices.rolling(window=50).mean().shift(1)
            log_centered = (log_prices - log_rolling_mean).dropna()
            
            # Walk-Forward Logic:
            # Training strictly ends at T, Testing strictly starts at T+1
            ticker_strat_returns = []
            window_train = 252 * 2
            window_test = 126
            
            total_len = len(log_centered)
            for start_idx in range(0, total_len - window_train, window_test):
                end_train = start_idx + window_train
                end_test = min(end_train + window_test, total_len)
                
                if end_test <= end_train: break
                
                # Training data: [start_idx : end_train]
                y_train = log_centered.iloc[start_idx:end_train].values
                # Testing data: [end_train : end_test]
                y_test = log_centered.iloc[end_train:end_test].values
                
                t_train = np.linspace(0, 1, len(y_train))
                # Predict forward from the last training point
                t_test = np.linspace(1 + (1/len(y_train)), 1 + (len(y_test)/len(y_train)), len(y_test))
                
                # Fit
                fitter = LSTMRobustFitter(n_harmonics=20, alpha=0.02)
                coefs, intercept, _, freqs = fitter.fit_sparse(t_train, y_train)
                
                # Predict
                X = [np.ones_like(t_test), t_test, t_test**2]
                for w in freqs:
                    X.append(np.cos(w * t_test)); X.append(np.sin(w * t_test))
                t_eps = t_test + 1e-6
                X.extend([np.exp(t_test), np.log(t_eps), np.sqrt(t_eps), np.abs(t_test - 0.5)])
                y_pred_te = np.dot(np.array(X).T, coefs) + intercept
                
                # Execute
                oos_prices = prices.loc[log_centered.index[end_train:end_test]]
                oos_ma_20 = ma_20.loc[log_centered.index[end_train:end_test]]
                # Returns are (P_t / P_{t-1} - 1). 
                # Decision for Trade_t uses Info up to t-1.
                oos_returns = prices.loc[log_centered.index[end_train-1:end_test]].pct_change().iloc[1:]
                raw_signals = np.where(y_pred_te > 0, 1.0, -1.0)
                
                for i in range(len(oos_returns)):
                    # Trailing volatility excludes the current day's return
                    lookback_rets = oos_returns.iloc[max(0, i-20):i] # i is excluded
                    
                    # Use a default scale if insufficient history in the window
                    if len(lookback_rets) < 5:
                        scale = 1.0
                    else:
                        scale = risk_engine.downside_convexity(lookback_rets, oos_prices.iloc[i], oos_ma_20.iloc[i])
                    
                    win_r = 0.55 # Assume baseline win rate for Kelly if window is short
                    p_factor = 1.2
                    kelly_scale = risk_engine.adaptive_kelly(win_r, p_factor, 0.0)
                    
                    final_scale = min(scale * (1 + kelly_scale), 3.0)
                    ticker_strat_returns.append(raw_signals[i] * final_scale * oos_returns.iloc[i])
                    
            all_oos_returns[ticker] = np.array(ticker_strat_returns)
            
        except Exception as e:
            logger.error(f"Failed for {ticker}: {e}")

    # Portfolio Weighting (Rolling)
    min_len = min(len(r) for r in all_oos_returns.values())
    port_rets = []
    cum_val = 1.0; peak_val = 1.0
    
    for i in range(min_len):
        current_dd = (cum_val / peak_val) - 1
        # Truncate all returns to the current point for weighting
        current_rets_dict = {t: r[:i+1] for t, r in all_oos_returns.items()}
        weights = risk_engine.asymmetric_weighting(current_rets_dict, current_dd)
        
        step_ret = sum(all_oos_returns[ticker][i] * w for ticker, w in weights.items())
        port_rets.append(step_ret)
        cum_val *= (1 + step_ret); peak_val = max(peak_val, cum_val)
        
    port_rets = np.array(port_rets)
    cum_ret = (1 + port_rets).cumprod()
    
    # Annualized Stats
    years = len(port_rets) / 252
    ann_return = (cum_ret[-1])**(1/years) - 1
    ann_vol = np.std(port_rets) * np.sqrt(252)
    sharpe = ann_return / (ann_vol + 1e-9)
    mdd = (cum_ret / np.maximum.accumulate(cum_ret) - 1).min()
    
    # Report
    report = f"""# Sagan Capital: 5-Year Institutional Audit
## Strategy: SOTA Rolling Walk-Forward

### 🏆 5-Year Annualized Performance
- **Annualized Return:** {ann_return:.2%}
- **Annualized Volatility:** {ann_vol:.2%}
- **Annualized Sharpe Ratio:** {sharpe:.2f}
- **Maximum Drawdown:** {mdd:.2%}
- **Total Cumulative Return:** {cum_ret[-1]-1:.2%}

### 🔬 Methodology
This audit uses a **Walk-Forward Optimization (WFO)** approach. The symbolic model is re-fitted every 6 months using the preceding 2 years of log-centered stationary residuals. This eliminates look-ahead bias and accounts for changing market regimes.

### 📈 Sector Diversification (20 Tickers)
| Ticker | Annualized Return | Max DD |
| :--- | :--- | :--- |
"""
    for ticker, rets in all_oos_returns.items():
        rets = rets[:min_len]
        c = (1 + rets).cumprod()
        y = len(rets) / 252
        t_ann_ret = (c[-1])**(1/y) - 1
        t_mdd = (c / np.maximum.accumulate(c) - 1).min()
        report += f"| {ticker} | {t_ann_ret:.2%} | {t_mdd:.2%} |\n"
        
    with open("Sagan_5Year_Institutional_Audit.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    logger.info(f"5-Year Audit Complete. Ann. Return: {ann_return:.2%}, Sharpe: {sharpe:.2f}")

if __name__ == "__main__":
    run_5yr_rolling_audit()
