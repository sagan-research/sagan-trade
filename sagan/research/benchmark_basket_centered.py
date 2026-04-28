import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import r2_score
from sagan.models.robust_fitter import LSTMRobustFitter
from sagan.research import BacktestEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagan.benchmark_centered")

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "META", "AMD", "GS", "JPM", "XOM"]

def train_and_backtest_centered(ticker, window=50, n_harmonics=20, alpha=0.01):
    logger.info(f"Processing {ticker} (Centered)...")
    
    # 1. Fetch Data
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    prices = df['Close']
    rolling_mean = prices.rolling(window=window).mean()
    centered = (prices - rolling_mean).dropna()
    
    # Split
    split = int(len(centered) * 0.75)
    y_train = centered.iloc[:split].values
    y_test = centered.iloc[split:].values
    
    t_train = np.linspace(0, 1, len(y_train))
    t_test = np.linspace(1, 1.33, len(y_test))
    
    # 2. Fit Robust LSTM on Centered Data
    fitter = LSTMRobustFitter(n_harmonics=n_harmonics, alpha=alpha)
    coefs, intercept, X_basis_tr, freqs = fitter.fit_sparse(t_train, y_train)
    
    # 3. OOS Prediction (Residuals)
    X_basis_te = []
    X_basis_te.append(np.ones_like(t_test))
    X_basis_te.append(t_test)
    X_basis_te.append(t_test**2)
    for w in freqs:
        X_basis_te.append(np.cos(w * t_test))
        X_basis_te.append(np.sin(w * t_test))
    X_basis_te = np.array(X_basis_te).T
    
    y_pred_te_centered = np.dot(X_basis_te, coefs) + intercept
    
    # Revert for R2 calculation
    y_pred_te_reverted = y_pred_te_centered + rolling_mean.loc[centered.index].iloc[split:].values
    y_true_te = prices.loc[centered.index].iloc[split:].values
    
    oos_r2 = r2_score(y_true_te, y_pred_te_reverted)
    
    # 4. Generate Trading Signals (on Residuals)
    # If predicted residual > current residual, we expect mean reversion or trend continuation?
    # Actually, simpler: if predicted residual > 0, we expect price to be above MA(50) -> Long?
    # Or: if predicted residual > current residual -> Buy.
    
    # For backtest engine, we'll pass a custom signal series
    # But wait, BacktestEngine runs on formula. 
    # We'll create a "proxy formula" or just manually backtest.
    
    # Manual Backtest for the OOS period
    oos_prices = prices.loc[centered.index].iloc[split:]
    oos_returns = oos_prices.pct_change().shift(-1).fillna(0)
    
    # Signal: Long if predicted centered value > 0 (Price expected above MA50)
    signals = np.where(y_pred_te_centered > 0, 1.0, -1.0)
    strat_returns = signals * oos_returns
    
    total_return = np.prod(1 + strat_returns) - 1
    daily_std = np.std(strat_returns)
    sharpe = (np.mean(strat_returns) / (daily_std + 1e-9)) * np.sqrt(252)
    
    # Max Drawdown
    cum_ret = (1 + strat_returns).cumprod()
    mdd = (cum_ret / cum_ret.cummax() - 1).min()
    
    return {
        "ticker": ticker,
        "oos_r2": oos_r2,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "formula": fitter.get_sparse_formula(coefs, intercept, freqs)
    }

def run_basket_centered():
    results = []
    for t in TICKERS:
        try:
            res = train_and_backtest_centered(t)
            results.append(res)
        except Exception as e:
            logger.error(f"Failed for {t}: {e}")
            
    # Generate Report
    report = "# Diversified Basket Benchmark (Centered Rolling Method)\n\n"
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
    
    report += "## Discovered Residual Formulas\n\n"
    for res in results:
        report += f"### {res['ticker']}\n`{res['formula']}`\n\n"
        
    with open("Basket_Benchmark_Centered_Report.md", "w") as f:
        f.write(report)
    
    logger.info("Centered Basket Benchmark complete.")

if __name__ == "__main__":
    run_basket_centered()
