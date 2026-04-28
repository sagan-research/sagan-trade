import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import r2_score
from sagan.utils.stationarity import frac_diff_ffd
from sagan.models.robust_fitter import LSTMRobustFitter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagan.robust_train")

def run_robust_comparison(ticker="NVDA"):
    logger.info(f"Comparing fit robustness for {ticker}...")
    
    # 1. Fetch Data
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    prices = df[['Close']].copy()
    
    # 2. Stationary Transformation
    # We'll use d=0.4 as a common middle ground for FracDiff
    prices_diff = frac_diff_ffd(prices, d=0.4)
    
    # Split
    split = int(len(prices_diff) * 0.75)
    train_diff = prices_diff.iloc[:split]
    test_diff = prices_diff.iloc[split:]
    
    y_train = train_diff['Close'].values
    y_test = test_diff['Close'].values
    t_train = np.linspace(0, 1, len(y_train))
    t_test = np.linspace(1, 1.33, len(y_test))
    
    # 3. Robust Fitting
    fitter = LSTMRobustFitter(n_harmonics=20, alpha=0.05)
    coefs, intercept, X_basis_tr, freqs = fitter.fit_sparse(t_train, y_train)
    
    # Evaluate OOS
    # Construct OOS basis matrix
    X_basis_te = []
    X_basis_te.append(np.ones_like(t_test))
    X_basis_te.append(t_test)
    X_basis_te.append(t_test**2)
    for w in freqs:
        X_basis_te.append(np.cos(w * t_test))
        X_basis_te.append(np.sin(w * t_test))
    X_basis_te = np.array(X_basis_te).T
    
    y_pred_tr = np.dot(X_basis_tr, coefs) + intercept
    y_pred_te = np.dot(X_basis_te, coefs) + intercept
    
    r2_tr = r2_score(y_train, y_pred_tr)
    r2_te = r2_score(y_test, y_pred_te)
    
    logger.info("="*30)
    logger.info(f"ROBUST RESULTS ({ticker})")
    logger.info(f"Stationary (FracDiff d=0.4)")
    logger.info(f"In-Sample R2: {r2_tr:.4f}")
    logger.info(f"Out-of-Sample R2: {r2_te:.4f}")
    logger.info(f"Non-zero Coefficients: {np.count_nonzero(coefs)} / {len(coefs)}")
    logger.info("="*30)
    
    formula = fitter.get_sparse_formula(coefs, intercept, freqs)
    logger.info(f"Robust Math Function:")
    logger.info(formula)
    
    return r2_te

if __name__ == "__main__":
    run_robust_comparison("NVDA")
