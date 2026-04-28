import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sagan.models.robust_fitter import LSTMRobustFitter
from sagan.portfolio.asymmetric_risk import AsymmetricRiskEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagan.benchmark_asymmetric")

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "META", "AMD", "GS", "JPM", "XOM"]

def run_asymmetric_benchmark():
    risk_engine = AsymmetricRiskEngine(target_vol=0.15, max_drawdown_limit=0.075)
    all_oos_returns = {}
    ticker_data = {}
    
    logger.info("Starting Asymmetric Benchmark (Riding the Upside)...")
    
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            prices = df['Close']
            ma_20 = prices.rolling(window=20).mean()
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
            oos_ma_20 = ma_20.loc[log_centered.index].iloc[split:]
            oos_returns = oos_prices.pct_change().fillna(0)
            raw_signals = np.where(y_pred_te > 0, 1.0, -1.0)
            
            # Asymmetric Execution
            strat_returns = []
            for i in range(len(oos_returns)):
                # Calculate asymmetric scale based on downside vol and momentum
                lookback_rets = oos_returns.iloc[max(0, i-20):i+1]
                scale = risk_engine.calculate_asymmetric_scale(
                    lookback_rets, 
                    oos_prices.iloc[i], 
                    oos_ma_20.iloc[i]
                )
                
                trade_ret = raw_signals[i] * scale * oos_returns.iloc[i]
                strat_returns.append(trade_ret)
                
            all_oos_returns[ticker] = np.array(strat_returns)
            
        except Exception as e:
            logger.error(f"Failed for {ticker}: {e}")

    # Portfolio Weighting (Asymmetric Sortino-style)
    current_dd = 0.0 # Start with 0 drawdown
    weights = risk_engine.asymmetric_weighting(all_oos_returns, current_dd)
    
    port_rets = np.zeros_like(next(iter(all_oos_returns.values())))
    cum_ret_val = 1.0
    peak_val = 1.0
    
    final_port_rets = []
    for i in range(len(port_rets)):
        # Dynamic Weight Adjustment for Portfolio Drawdown
        current_dd = (cum_ret_val / peak_val) - 1
        dynamic_weights = risk_engine.asymmetric_weighting(all_oos_returns, current_dd)
        
        step_ret = 0
        for ticker, w in dynamic_weights.items():
            step_ret += all_oos_returns[ticker][i] * w
            
        final_port_rets.append(step_ret)
        cum_ret_val *= (1 + step_ret)
        peak_val = max(peak_val, cum_ret_val)
        
    final_port_rets = np.array(final_port_rets)
    cum_ret = (1 + final_port_rets).cumprod()
    sharpe = (np.mean(final_port_rets) / (np.std(final_port_rets) + 1e-9)) * np.sqrt(252)
    mdd = (cum_ret / np.maximum.accumulate(cum_ret) - 1).min()
    
    # Save Report
    report = "# Asymmetric Portfolio Report\n\n"
    report += "## Strategy: Upside Vol Capture + Downside Semi-Deviation Shield\n\n"
    report += f"- **Portfolio Sharpe Ratio:** {sharpe:.2f}\n"
    report += f"- **Portfolio Max Drawdown:** {mdd:.2%}\n"
    report += f"- **Final Portfolio Return:** {cum_ret[-1]-1:.2%}\n\n"
    
    report += "### The Asymmetric Edge\n"
    report += "1. **Upside Vol Capture**: Positions are levered up (to 3.0x) during positive momentum regimes where downside volatility is low.\n"
    report += "2. **Downside Shield**: Exposure is aggressively cut by 50% or more when prices drop below the 20-day MA or when semi-deviation spikes.\n"
    report += "3. **Sortino Weighting**: Portfolio allocation favors assets with the highest Upside/Downside variance ratios.\n"
    
    with open("Asymmetric_Portfolio_Report.md", "w") as f:
        f.write(report)
        
    logger.info(f"Asymmetric Benchmark complete. Sharpe: {sharpe:.2f}, Return: {cum_ret[-1]-1:.2%}")

if __name__ == "__main__":
    run_asymmetric_benchmark()
