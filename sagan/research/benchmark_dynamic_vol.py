import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sagan.models.robust_fitter import LSTMRobustFitter
from sagan.portfolio.risk_engine import RiskEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagan.benchmark_dynamic")

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "META", "AMD", "GS", "JPM", "XOM"]

def run_dynamic_vol_benchmark():
    risk_engine = RiskEngine(target_vol=0.12, max_drawdown_limit=0.075)
    
    # 1. Fetch Market Proxy (SPY) for regime detection
    spy = yf.download("SPY", period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    spy_returns = spy['Close'].pct_change().fillna(0)
    spy_vol = spy_returns.rolling(window=20).std() * np.sqrt(252)
    
    all_oos_returns = {}
    
    logger.info("Starting Dynamic Volatility Management Benchmark...")
    
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            prices = df['Close']
            log_prices = np.log(prices)
            log_centered = (log_prices - log_prices.rolling(window=50).mean()).dropna()
            
            split = int(len(log_centered) * 0.75)
            y_train = log_centered.iloc[:split].values
            y_test = log_centered.iloc[split:].values
            t_train = np.linspace(0, 1, len(y_train))
            t_test = np.linspace(1, 1.33, len(y_test))
            
            fitter = LSTMRobustFitter(n_harmonics=20, alpha=0.01)
            coefs, intercept, _, freqs = fitter.fit_sparse(t_train, y_train)
            
            # Predict OOS
            X = [np.ones_like(t_test), t_test, t_test**2]
            for w in freqs:
                X.append(np.cos(w * t_test)); X.append(np.sin(w * t_test))
            t_eps = t_test + 1e-6
            X.extend([np.exp(t_test), np.log(t_eps), np.sqrt(t_eps), np.abs(t_test - 0.5)])
            y_pred_te = np.dot(np.array(X).T, coefs) + intercept
            
            oos_prices = prices.loc[log_centered.index].iloc[split:]
            oos_returns = oos_prices.pct_change().fillna(0)
            raw_signals = np.where(y_pred_te > 0, 1.0, -1.0)
            
            # Dynamic Signal Calculation
            ticker_strat_returns = []
            current_cum_ret = 1.0
            peak_cum_ret = 1.0
            
            # Match SPY vol to OOS period
            oos_spy_vol = spy_vol.loc[oos_prices.index]
            
            for i in range(len(oos_returns)):
                # 1. Regime-Aware Target
                market_v = oos_spy_vol.iloc[i] if not np.isnan(oos_spy_vol.iloc[i]) else 0.15
                target = risk_engine.dynamic_vol_target(market_v)
                
                # 2. Drawdown Protection
                current_dd = (current_cum_ret / peak_cum_ret) - 1
                protected_target = risk_engine.drawdown_protector(current_dd, base_target=target)
                
                # 3. Position Sizing
                trailing_v = np.std(oos_returns.iloc[max(0, i-20):i+1]) * np.sqrt(252) + 1e-9
                scale = protected_target / trailing_v
                scale = min(scale, 2.0) # Leverage cap
                
                trade_ret = raw_signals[i] * scale * oos_returns.iloc[i]
                ticker_strat_returns.append(trade_ret)
                
                current_cum_ret *= (1 + trade_ret)
                peak_cum_ret = max(peak_cum_ret, current_cum_ret)
                
            all_oos_returns[ticker] = np.array(ticker_strat_returns)
            
        except Exception as e:
            logger.error(f"Failed for {ticker}: {e}")

    # Portfolio Stats
    weights = risk_engine.apply_risk_parity(all_oos_returns)
    port_rets = np.zeros_like(next(iter(all_oos_returns.values())))
    for ticker, weight in weights.items():
        port_rets += all_oos_returns[ticker] * weight
        
    cum_ret = (1 + port_rets).cumprod()
    sharpe = (np.mean(port_rets) / (np.std(port_rets) + 1e-9)) * np.sqrt(252)
    mdd = (cum_ret / np.maximum.accumulate(cum_ret) - 1).min()
    
    # Save Report
    report = "# Dynamic Volatility Management Report\n\n"
    report += "## Strategy: Regime-Aware Target + Drawdown Protector\n\n"
    report += f"- **Portfolio Sharpe Ratio:** {sharpe:.2f}\n"
    report += f"- **Portfolio Max Drawdown:** {mdd:.2%}\n"
    report += f"- **Final Portfolio Return:** {cum_ret[-1]-1:.2%}\n\n"
    
    report += "### Dynamic Mechanisms Used\n"
    report += "1. **Regime-Aware Target**: Scales down target vol when SPY volatility exceeds 20%.\n"
    report += "2. **Drawdown Protector**: Reduces target vol quadratically as the portfolio approaches the 7.5% limit.\n"
    
    with open("Dynamic_Vol_Report.md", "w") as f:
        f.write(report)
        
    logger.info(f"Dynamic Vol Benchmark complete. Sharpe: {sharpe:.2f}, MDD: {mdd:.2%}")

if __name__ == "__main__":
    run_dynamic_vol_benchmark()
