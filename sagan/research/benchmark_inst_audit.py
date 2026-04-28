import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sagan.models.robust_fitter import LSTMRobustFitter
from sagan.portfolio.asymmetric_risk import AsymmetricRiskEngine
import logging
import os
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagan.inst_audit")

TICKERS = [
    "AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "META", "AMD", "GS", "JPM", "XOM",
    "UNH", "V", "MA", "PG", "JNJ", "HD", "ABBV", "COST", "LLY", "CRM"
]

def run_institutional_audit(starting_cap=10_000_000, trade_cost_bps=5):
    risk_engine = AsymmetricRiskEngine(target_vol=0.15, max_drawdown_limit=0.075)
    cost_per_trade = trade_cost_bps / 10000
    all_oos_returns = {}
    ticker_stats = {}
    
    logger.info(f"Starting Institutional Audit (5-Year Rolling, 5bps Cost, ${starting_cap:,.0f} Cap)...")
    
    for ticker in TICKERS:
        try:
            logger.info(f"Auditing {ticker}...")
            df = yf.download(ticker, period="8y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            prices = df['Close']
            ma_20 = prices.rolling(window=20).mean().shift(1)
            log_prices = np.log(prices)
            log_rolling_mean = log_prices.rolling(window=50).mean().shift(1)
            log_centered = (log_prices - log_rolling_mean).dropna()
            
            ticker_strat_returns = []
            trades = []
            prev_exposure = 0.0
            
            window_train = 252 * 2
            window_test = 126
            total_len = len(log_centered)
            
            for start_idx in range(0, total_len - window_train, window_test):
                end_train = start_idx + window_train
                end_test = min(end_train + window_test, total_len)
                if end_test <= end_train: break
                
                y_train = log_centered.iloc[start_idx:end_train].values
                y_test = log_centered.iloc[end_train:end_test].values
                t_train = np.linspace(0, 1, len(y_train))
                t_test = np.linspace(1 + (1/len(y_train)), 1 + (len(y_test)/len(y_train)), len(y_test))
                
                fitter = LSTMRobustFitter(n_harmonics=20, alpha=0.02)
                coefs, intercept, _, freqs = fitter.fit_sparse(t_train, y_train)
                
                X = [np.ones_like(t_test), t_test, t_test**2]
                for w in freqs:
                    X.append(np.cos(w * t_test)); X.append(np.sin(w * t_test))
                t_eps = t_test + 1e-6
                X.extend([np.exp(t_test), np.log(t_eps), np.sqrt(t_eps), np.abs(t_test - 0.5)])
                y_pred_te = np.dot(np.array(X).T, coefs) + intercept
                
                oos_prices = prices.loc[log_centered.index[end_train:end_test]]
                oos_ma_20 = ma_20.loc[log_centered.index[end_train:end_test]]
                oos_returns = prices.loc[log_centered.index[end_train-1:end_test]].pct_change().iloc[1:]
                raw_signals = np.where(y_pred_te > 0, 1.0, -1.0)
                
                for i in range(len(oos_returns)):
                    lookback_rets = oos_returns.iloc[max(0, i-20):i]
                    if len(lookback_rets) < 5:
                        scale = 1.0
                    else:
                        scale = risk_engine.downside_convexity(lookback_rets, oos_prices.iloc[i], oos_ma_20.iloc[i])
                    
                    exposure = raw_signals[i] * min(scale * 1.15, 3.0) # Adaptive Kelly proxy
                    
                    # Apply Trading Costs on change in exposure
                    turnover = abs(exposure - prev_exposure)
                    cost = turnover * cost_per_trade
                    
                    trade_ret = (exposure * oos_returns.iloc[i]) - cost
                    ticker_strat_returns.append(trade_ret)
                    
                    if abs(exposure - prev_exposure) > 0.01:
                        trades.append({
                            "Type": "BUY" if exposure > prev_exposure else "SELL",
                            "Size_USD": abs(exposure - prev_exposure) * (starting_cap / len(TICKERS)),
                            "Price": oos_prices.iloc[i],
                            "Return": trade_ret
                        })
                    
                    prev_exposure = exposure
                    
            all_oos_returns[ticker] = np.array(ticker_strat_returns)
            
            # Ticker Metrics
            if trades:
                wins = sum(1 for t in trades if t["Return"] > 0)
                ticker_stats[ticker] = {
                    "num_trades": len(trades),
                    "win_rate": wins / len(trades),
                    "avg_buy_usd": np.mean([t["Size_USD"] for t in trades if t["Type"] == "BUY"]) if any(t["Type"] == "BUY" for t in trades) else 0,
                    "avg_sell_usd": np.mean([t["Size_USD"] for t in trades if t["Type"] == "SELL"]) if any(t["Type"] == "SELL" for t in trades) else 0,
                }
            
        except Exception as e:
            logger.error(f"Failed for {ticker}: {e}")

    # Portfolio Stats
    min_len = min(len(r) for r in all_oos_returns.values())
    port_rets = []
    cum_val = 1.0; peak_val = 1.0
    for i in range(min_len):
        current_dd = (cum_val / peak_val) - 1
        current_rets_dict = {t: r[:i+1] for t, r in all_oos_returns.items()}
        weights = risk_engine.asymmetric_weighting(current_rets_dict, current_dd)
        step_ret = sum(all_oos_returns[ticker][i] * w for ticker, w in weights.items())
        port_rets.append(step_ret)
        cum_val *= (1 + step_ret); peak_val = max(peak_val, cum_val)
        
    port_rets = np.array(port_rets)
    cum_ret = (1 + port_rets).cumprod()
    years = len(port_rets) / 252
    ann_return = (cum_ret[-1])**(1/years) - 1
    ann_vol = np.std(port_rets) * np.sqrt(252)
    sharpe = ann_return / (ann_vol + 1e-9)
    mdd = (cum_ret / np.maximum.accumulate(cum_ret) - 1).min()
    
    # Report
    report = f"""# Institutional Audit Report: Sagan Alpha Engine
## 5-Year Rolling Walk-Forward | **5bps Cost per Trade**

### 💎 Executive Summary (Portfolio Level)
- **Starting Capital:** ${starting_cap:,.0f}
- **Annualized Return (Net of Fees):** {ann_return:.2%}
- **Annualized Sharpe Ratio:** {sharpe:.2f}
- **Maximum Drawdown:** {mdd:.2%}
- **5-Year Cumulative Return:** {cum_ret[-1]-1:.2%}

### 📊 Trading Statistics (20 Tickers)
| Ticker | Trades | Win Rate | Avg Buy Size | Avg Sell Size |
| :--- | :--- | :--- | :--- | :--- |
"""
    for ticker, stats in ticker_stats.items():
        report += f"| {ticker} | {stats['num_trades']} | {stats['win_rate']:.2%} | ${stats['avg_buy_usd']:,.0f} | ${stats['avg_sell_usd']:,.0f} |\n"
        
    report += "\n\n**Note:** All sizes assume a portfolio allocation of $500k per ticker initially, scaled by the dynamic convexity engine."
    
    with open("Institutional_Audit_Fees_Report.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    logger.info(f"Institutional Audit Complete. Net Sharpe: {sharpe:.2f}")

if __name__ == "__main__":
    run_institutional_audit()
