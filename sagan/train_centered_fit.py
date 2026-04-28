import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import r2_score
from sagan.models.robust_fitter import LSTMRobustFitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagan.centered_train")

def run_centered_comparison(ticker="AAPL", window=50):
    logger.info(f"Running Centered Rolling Fit for {ticker} (Window: {window})...")
    
    # 1. Fetch Data
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    prices = df['Close'].copy()
    
    # 2. Rolling Mean Centering
    rolling_mean = prices.rolling(window=window).mean()
    centered_prices = prices - rolling_mean
    centered_prices = centered_prices.dropna()
    
    # Align original prices and rolling mean for later reversion
    valid_prices = prices.loc[centered_prices.index]
    valid_rolling_mean = rolling_mean.loc[centered_prices.index]
    
    # Split
    split = int(len(centered_prices) * 0.75)
    y_train = centered_prices.iloc[:split].values
    y_test = centered_prices.iloc[split:].values
    
    t_train = np.linspace(0, 1, len(y_train))
    t_test = np.linspace(1, 1.33, len(y_test))
    
    # 3. Fit LSTM Robust Fitter on Centered Data
    fitter = LSTMRobustFitter(n_harmonics=20, alpha=0.01) # Lower alpha for centered residuals
    coefs, intercept, X_basis_tr, freqs = fitter.fit_sparse(t_train, y_train)
    
    # 4. Predict and Revert
    # In-Sample
    y_pred_tr_centered = np.dot(X_basis_tr, coefs) + intercept
    y_pred_tr_reverted = y_pred_tr_centered + valid_rolling_mean.iloc[:split].values
    
    # Out-of-Sample
    X_basis_te = []
    X_basis_te.append(np.ones_like(t_test))
    X_basis_te.append(t_test)
    X_basis_te.append(t_test**2)
    for w in freqs:
        X_basis_te.append(np.cos(w * t_test))
        X_basis_te.append(np.sin(w * t_test))
    X_basis_te = np.array(X_basis_te).T
    
    y_pred_te_centered = np.dot(X_basis_te, coefs) + intercept
    # NOTE: In OOS, we need the 50-day rolling mean to revert. 
    # For a true forecast, we'd need to forecast the rolling mean too, 
    # but here we use the actual rolling mean to see how the "residual math" sticks.
    y_pred_te_reverted = y_pred_te_centered + valid_rolling_mean.iloc[split:].values
    
    # 5. Evaluate
    y_true_tr = valid_prices.iloc[:split].values
    y_true_te = valid_prices.iloc[split:].values
    
    r2_tr = r2_score(y_true_tr, y_pred_tr_reverted)
    r2_te = r2_score(y_true_te, y_pred_te_reverted)
    
    logger.info("="*30)
    logger.info(f"CENTERED ROLLING RESULTS ({ticker})")
    logger.info(f"In-Sample R2 (Reverted): {r2_tr:.4f}")
    logger.info(f"Out-of-Sample R2 (Reverted): {r2_te:.4f}")
    logger.info(f"Non-zero Coefficients: {np.count_nonzero(coefs)} / {len(coefs)}")
    logger.info("="*30)
    
    formula = fitter.get_sparse_formula(coefs, intercept, freqs)
    logger.info(f"Centered Math Function (Residuals):")
    logger.info(formula)
    
    return r2_te

if __name__ == "__main__":
    run_centered_comparison("AAPL")
