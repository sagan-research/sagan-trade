import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import r2_score
from sagan.models.robust_fitter import LSTMRobustFitter
from sagan.portfolio.risk_engine import RiskEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagan.benchmark_risk")

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "META", "AMD", "GS", "JPM", "XOM"]

def run_risk_managed_benchmark(target_vol=0.12):
    risk_engine = RiskEngine(target_vol=target_vol)
    all_oos_returns = {}
    ticker_data = {}
    
    logger.info("Training and Predicting for Risk-Managed Portfolio...")
    
    for ticker in TICKERS:
        try:
            # 1. Fetch Data
            df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            prices = df['Close']
            log_prices = np.log(prices)
            log_rolling_mean = log_prices.rolling(window=50).mean()
            log_centered = (log_prices - log_rolling_mean).dropna()
            
            # Split
            split = int(len(log_centered) * 0.75)
            y_train = log_centered.iloc[:split].values
            y_test = log_centered.iloc[split:].values
            
            t_train = np.linspace(0, 1, len(y_train))
            t_test = np.linspace(1, 1.33, len(y_test))
            
            # 2. Fit
            fitter = LSTMRobustFitter(n_harmonics=20, alpha=0.01)
            coefs, intercept, _, freqs = fitter.fit_sparse(t_train, y_train)
            
            # 3. OOS Prediction
            X = []
            X.append(np.ones_like(t_test))
            X.append(t_test)
            X.append(t_test**2)
            for w in freqs:
                X.append(np.cos(w * t_test))
                X.append(np.sin(w * t_test))
            t_eps = t_test + 1e-6
            X.append(np.exp(t_test))
            X.append(np.log(t_eps))
            X.append(np.sqrt(t_eps))
            X.append(np.abs(t_test - 0.5))
            X_basis_te = np.array(X).T
            
            y_pred_te_log_centered = np.dot(X_basis_te, coefs) + intercept
            
            # 4. Signal + Vol Targeting
            oos_prices = prices.loc[log_centered.index].iloc[split:]
            oos_returns = oos_prices.pct_change().fillna(0)
            
            # Binary signal
            raw_signals = np.where(y_pred_te_log_centered > 0, 1.0, -1.0)
            
            # Volatility Target Scaling
            # We use a rolling window of the last 20 days of OOS returns for dynamic scaling
            # But wait, in a true OOS, we'd use trailing volatility.
            # We'll simulate that.
            trailing_vol = oos_returns.rolling(window=20).std() * np.sqrt(252)
            vol_scale = (target_vol / (trailing_vol + 1e-9)).fillna(1.0)
            
            # Limit leverage to 2.0x
            vol_scale = vol_scale.clip(0, 2.0)
            
            risk_signals = raw_signals * vol_scale.values
            strat_returns = risk_signals * oos_returns.values
            
            all_oos_returns[ticker] = strat_returns
            ticker_data[ticker] = {
                "returns": strat_returns,
                "raw_returns": oos_returns.values
            }
            
        except Exception as e:
            logger.error(f"Failed for {ticker}: {e}")

    # 5. Portfolio Construction (Risk Parity)
    weights = risk_engine.apply_risk_parity(all_oos_returns)
    
    portfolio_returns = np.zeros_like(next(iter(all_oos_returns.values())))
    for ticker, weight in weights.items():
        portfolio_returns += all_oos_returns[ticker] * weight
        
    # 6. Performance Metrics
    cum_ret = (1 + portfolio_returns).cumprod()
    total_return = cum_ret[-1] - 1
    sharpe = (np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-9)) * np.sqrt(252)
    mdd = (cum_ret / np.maximum.accumulate(cum_ret) - 1).min()
    
    # 7. Generate Report
    report = "# Risk-Managed Portfolio Benchmark Report\n\n"
    report += "## Target Volatility: 12% | Drawdown Limit: 7.5%\n\n"
    report += "### Core Metrics\n"
    report += f"- **Portfolio Total Return:** {total_return:.2%}\n"
    report += f"- **Portfolio Sharpe Ratio:** {sharpe:.2f}\n"
    report += f"- **Portfolio Max Drawdown:** {mdd:.2%}\n\n"
    
    report += "### Weights (Risk Parity)\n"
    report += "| Ticker | Weight |\n"
    report += "| :--- | :--- |\n"
    for ticker, weight in weights.items():
        report += f"| {ticker} | {weight:.2%} |\n"
        
    report += "\n### Individual Ticker Drawdowns (Risk Managed)\n"
    report += "| Ticker | Managed Drawdown |\n"
    report += "| :--- | :--- |\n"
    for ticker, data in ticker_data.items():
        rets = data["returns"]
        c = (1 + rets).cumprod()
        d = (c / np.maximum.accumulate(c) - 1).min()
        report += f"| {ticker} | {d:.2%} |\n"
        
    with open("Risk_Managed_Benchmark_Report.md", "w") as f:
        f.write(report)
        
    logger.info(f"Risk Benchmark complete. MDD: {mdd:.2%}")
    return mdd

if __name__ == "__main__":
    run_risk_managed_benchmark()
