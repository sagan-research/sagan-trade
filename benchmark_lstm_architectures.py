import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
import logging
from sagan.models.symbolic_fitter import LegacyLSTMSymbolicFitter, HybridSymbolicFitter, TCNSymbolicFitter
from sagan.utils import sharpe_ratio, max_drawdown, annualised_return

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("benchmark")

def prepare_data(ticker="SPY"):
    logger.info(f"Downloading {ticker} data...")
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Use returns as our target to fit
    returns = df['Close'].pct_change().dropna().values
    
    # Normalize for NN training
    y_mean = returns.mean()
    y_std = returns.std()
    y_norm = (returns - y_mean) / (y_std + 1e-8)
    
    split = int(len(y_norm) * 0.8)
    y_train = y_norm[:split]
    y_test = y_norm[split:]
    actual_returns_test = returns[split:]
    
    return y_train, y_test, actual_returns_test, (y_mean, y_std)

def train_and_evaluate(model, name, y_train, y_test, actual_returns_test, epochs=100):
    logger.info(f"\n--- Testing {name} ---")
    
    # Tensors
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(1, -1, 1)
    t_train = torch.linspace(0, 1, len(y_train)).view(-1, 1)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Measure Training Speed
    start_time = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        coeffs = model(y_train_tensor)
        y_fit = model.evaluate_math(t_train.squeeze(), coeffs[0], n_harmonics=3)
        loss = criterion(y_fit, torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()
    train_time = time.perf_counter() - start_time
    
    # Measure Inference Speed and Financial Metrics
    model.eval()
    inf_start = time.perf_counter()
    with torch.no_grad(): # Use no_grad for inference speedup
        final_coeffs = model(y_train_tensor)
        
        # OOS Grid
        t_test = torch.linspace(1, 1.25, len(y_test)).view(-1, 1)
        y_fit_test = model.evaluate_math(t_test.squeeze(), final_coeffs[0], n_harmonics=3).numpy()
    inf_time = time.perf_counter() - inf_start
    
    # Generate Trading Signals (Momentum / Reversion combo based on prediction)
    # Simple strategy: If predicted normalized return is positive, go long. If negative, go short.
    signals = np.where(y_fit_test > 0, 1.0, -1.0)
    
    # Calculate Strategy Returns (Shift signals by 1 to avoid lookahead bias)
    strategy_returns = signals[:-1] * actual_returns_test[1:]
    
    metrics = {
        "Name": name,
        "Train Time (s)": train_time,
        "Inference Time (s)": inf_time,
        "Ann. Return": annualised_return(strategy_returns),
        "Sharpe": sharpe_ratio(strategy_returns),
        "Max Drawdown": max_drawdown(strategy_returns)
    }
    
    logger.info(f"Train Time: {train_time:.4f}s | Inference Time: {inf_time:.6f}s")
    logger.info(f"Ann Return: {metrics['Ann. Return']:.2%} | Sharpe: {metrics['Sharpe']:.2f} | Max DD: {metrics['Max Drawdown']:.2%}")
    return metrics

def run_benchmark():
    y_train, y_test, actual_returns_test, stats = prepare_data("SPY")
    
    # Legacy Architecture
    legacy_model = LegacyLSTMSymbolicFitter(n_harmonics=3)
    metrics_legacy = train_and_evaluate(legacy_model, "Legacy LSTMFitter (5-Layer)", y_train, y_test, actual_returns_test)
    
    # Hybrid Architecture
    hybrid_model = HybridSymbolicFitter(n_harmonics=3)
    metrics_hybrid = train_and_evaluate(hybrid_model, "Hybrid GRU/Conv1d Fitter", y_train, y_test, actual_returns_test)
    
    # TCN Architecture
    tcn_model = TCNSymbolicFitter(n_harmonics=3)
    metrics_tcn = train_and_evaluate(tcn_model, "TCNSymbolicFitter (Linear Causal)", y_train, y_test, actual_returns_test)
    
    logger.info("\n=== FINAL COMPARISON ===")
    logger.info(f"{'Metric':<20} | {'Legacy':<15} | {'Hybrid':<15} | {'TCN':<15}")
    logger.info("-" * 75)
    keys = ["Train Time (s)", "Inference Time (s)", "Ann. Return", "Sharpe", "Max Drawdown"]
    for k in keys:
        v_leg = metrics_legacy[k]
        v_hyb = metrics_hybrid[k]
        v_tcn = metrics_tcn[k]
        logger.info(f"{k:<20} | {v_leg:<15.4f} | {v_hyb:<15.4f} | {v_tcn:<15.4f}")

if __name__ == "__main__":
    run_benchmark()
