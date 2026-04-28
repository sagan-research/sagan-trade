import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import r2_score
from sagan.models.robust_fitter import LSTMRobustFitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagan.benchmark_lognormal")

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "META", "AMD", "GS", "JPM", "XOM"]

def train_and_backtest_lognormal(ticker, window=50, n_harmonics=20, alpha=0.01):
    logger.info(f"Processing {ticker} (Log-Normal Centered)...")
    
    # 1. Fetch Data
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    prices = df['Close']
    
    # 2. Log-Centering (Accounting for Log-Normal Distribution)
    log_prices = np.log(prices)
    log_rolling_mean = log_prices.rolling(window=window).mean()
    log_centered = (log_prices - log_rolling_mean).dropna()
    
    # Split
    split = int(len(log_centered) * 0.75)
    y_train = log_centered.iloc[:split].values
    y_test = log_centered.iloc[split:].values
    
    t_train = np.linspace(0, 1, len(y_train))
    t_test = np.linspace(1, 1.33, len(y_test))
    
    # 3. Fit Robust LSTM on Log-Centered Data (Expanded Basis)
    fitter = LSTMRobustFitter(n_harmonics=n_harmonics, alpha=alpha)
    coefs, intercept, X_basis_tr, freqs = fitter.fit_sparse(t_train, y_train)
    
    # 4. OOS Prediction (Log-Residuals)
    def get_basis_matrix(t_vals):
        X = []
        X.append(np.ones_like(t_vals))
        X.append(t_vals)
        X.append(t_vals**2)
        for w in freqs:
            X.append(np.cos(w * t_vals))
            X.append(np.sin(w * t_vals))
        t_eps = t_vals + 1e-6
        X.append(np.exp(t_vals))
        X.append(np.log(t_eps))
        X.append(np.sqrt(t_eps))
        X.append(np.abs(t_vals - 0.5))
        return np.array(X).T

    X_basis_te = get_basis_matrix(t_test)
    y_pred_te_log_centered = np.dot(X_basis_te, coefs) + intercept
    
    # Revert to Prices
    # P = exp(log_residual + log_MA50)
    y_pred_te_reverted = np.exp(y_pred_te_log_centered + log_rolling_mean.loc[log_centered.index].iloc[split:].values)
    y_true_te = prices.loc[log_centered.index].iloc[split:].values
    
    oos_r2 = r2_score(y_true_te, y_pred_te_reverted)
    
    # 5. Trading Backtest (on Log-Residuals)
    oos_prices = prices.loc[log_centered.index].iloc[split:]
    oos_returns = oos_prices.pct_change().shift(-1).fillna(0)
    
    # Signal: Long if predicted log-residual > 0 (Price expected above MA50 trend in log-space)
    signals = np.where(y_pred_te_log_centered > 0, 1.0, -1.0)
    strat_returns = signals * oos_returns
    
    total_return = np.prod(1 + strat_returns) - 1
    sharpe = (np.mean(strat_returns) / (np.std(strat_returns) + 1e-9)) * np.sqrt(252)
    mdd = ((1 + strat_returns).cumprod() / (1 + strat_returns).cumprod().cummax() - 1).min()
    
    return {
        "ticker": ticker,
        "oos_r2": oos_r2,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "formula": fitter.get_sparse_formula(coefs, intercept, freqs)
    }

def run_basket_lognormal():
    results = []
    for t in TICKERS:
        try:
            res = train_and_backtest_lognormal(t)
            results.append(res)
        except Exception as e:
            logger.error(f"Failed for {t}: {e}")
            
    # Generate Report
    report = "# Diversified Basket Benchmark (Log-Normal Centered Method)\n\n"
    report += "## Performance Metrics (OOS 6-Month Period)\n\n"
    report += "| Ticker | OOS R2 | Total Return | Sharpe | Max Drawdown |\n"
    report += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    rets = []
    sharpes = []
    for res in results:
        report += f"| {res['ticker']} | {res['oos_r2']:.4f} | {res['total_return']:.2%} | {res['sharpe']:.2f} | {res['max_drawdown']:.2%} |\n"
        rets.append(res['total_return'])
        sharpes.append(res['sharpe'])
        
    report += f"\n**Portfolio Avg Return:** {np.mean(rets):.2%}\n"
    report += f"**Portfolio Avg Sharpe:** {np.mean(sharpes):.2f}\n\n"
    
    report += "## Discovered Log-Residual Formulas (Expanded Search Space)\n\n"
    for res in results:
        report += f"### {res['ticker']}\n`{res['formula']}`\n\n"
        
    with open("Basket_Benchmark_Lognormal_Report.md", "w") as f:
        f.write(report)
    
    logger.info("Log-Normal Basket Benchmark complete.")

if __name__ == "__main__":
    run_basket_lognormal()
