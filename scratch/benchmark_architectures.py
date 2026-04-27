import time
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from sagan.models.math_engine import MathematicalEngine, CenteredModelBasis
from sagan.models.controller_arch import ControllerLSTM
from sagan.models.lstm_direct import DirectLSTM
from sagan.symbolic_lib.download_models import CENTERED_MODEL_PATH, CONTROLLER_MODEL_PATH
from sklearn.preprocessing import StandardScaler
import os

def run_benchmark():
    ticker = "SPY"
    print(f"--- Benchmarking Architectures for {ticker} ---")
    
    # 1. Fetch Data
    data = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    close_col = 'Close' if 'Close' in data.columns else 'Adj Close'
    prices = data[close_col].values
    returns = data[close_col].pct_change().dropna().values
    
    # Prepare features for LSTM (last 30 days)
    window_size = 30
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns.reshape(-1, 1)).flatten()
    
    X, y = [], []
    for i in range(len(scaled_returns) - window_size):
        X.append(scaled_returns[i:i+window_size])
        y.append(scaled_returns[i+window_size])
    X = np.array(X)
    y = np.array(y)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 2. Setup Architectures
    
    # A. Pre-trained Symbolic (Centered Model)
    cm_engine = CenteredModelBasis(str(CENTERED_MODEL_PATH))
    
    # B. Pre-trained Controller (3-layer LSTM)
    controller = ControllerLSTM()
    controller.load_state_dict(torch.load(str(CONTROLLER_MODEL_PATH), map_location='cpu'))
    controller.eval()
    
    # C. New Direct LSTM (5-layer)
    direct_lstm = DirectLSTM(input_size=1, num_layers=5)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(direct_lstm.parameters(), lr=0.001)
    
    # Train Direct LSTM (Quickly for benchmark)
    print("Training 5-layer Direct LSTM...")
    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(-1)
    
    direct_lstm.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = direct_lstm(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    direct_lstm.eval()
    
    # 3. Latency Benchmarking
    print("\n--- Latency Benchmark (1000 inferences) ---")
    
    # Symbolic Latency
    t = np.arange(1000)
    start = time.time()
    for _ in range(1000):
        _ = cm_engine.evaluate(t)
    symbolic_time = (time.time() - start) / 1000
    print(f"Symbolic (Centered Model) Latency: {symbolic_time*1000:.4f} ms")
    
    # Controller Latency (3-layer LSTM)
    dummy_input = torch.randint(0, 67, (1, 30))
    start = time.time()
    with torch.no_grad():
        for _ in range(1000):
            _ = controller(dummy_input)
    controller_time = (time.time() - start) / 1000
    print(f"Controller (3-layer LSTM) Latency: {controller_time*1000:.4f} ms")
    
    # Direct LSTM Latency (5-layer)
    dummy_input_direct = torch.randn(1, 30, 1)
    start = time.time()
    with torch.no_grad():
        for _ in range(1000):
            _ = direct_lstm(dummy_input_direct)
    direct_time = (time.time() - start) / 1000
    print(f"Direct LSTM (5-layer) Latency: {direct_time*1000:.4f} ms")
    
    # 4. Returns Benchmarking (Simulated)
    print("\n--- Returns Benchmark (Simulated on Test Set) ---")
    
    X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
    with torch.no_grad():
        direct_preds = direct_lstm(X_test_t).numpy().flatten()
    
    # Simple Strategy: Long if prediction > 0, Short if < 0
    test_returns = returns[split + window_size:]
    
    # Direct LSTM Returns
    direct_signals = np.where(direct_preds > 0, 1, -1)
    direct_strat_returns = direct_signals * test_returns
    direct_sharpe = np.mean(direct_strat_returns) / (np.std(direct_strat_returns) + 1e-8) * np.sqrt(252)
    
    # Symbolic Returns (Centered Model)
    # We use the centered model on the test time range
    t_test = np.arange(len(test_returns))
    cm_preds = cm_engine.evaluate(t_test)
    cm_signals = np.where(cm_preds > 0, 1, -1)
    cm_strat_returns = cm_signals * test_returns
    cm_sharpe = np.mean(cm_strat_returns) / (np.std(cm_strat_returns) + 1e-8) * np.sqrt(252)
    
    print(f"Direct LSTM (5-layer) Sharpe: {direct_sharpe:.4f}")
    print(f"Symbolic (Centered Model) Sharpe: {cm_sharpe:.4f}")
    
    # 5. Report Generation
    with open("Benchmark_Comparison_Report.md", "w") as f:
        f.write("# Architecture Benchmark Report\n\n")
        f.write(f"**Ticker:** {ticker}\n")
        f.write(f"**Date:** {pd.Timestamp.now()}\n\n")
        f.write("## Latency Comparison (ms per inference)\n")
        f.write(f"- **Symbolic (Centered Model):** {symbolic_time*1000:.4f} ms\n")
        f.write(f"- **Controller (3-layer LSTM):** {controller_time*1000:.4f} ms\n")
        f.write(f"- **Direct LSTM (5-layer):** {direct_time*1000:.4f} ms\n\n")
        f.write("## Performance Comparison (Annualized Sharpe)\n")
        f.write(f"- **Symbolic (Centered Model):** {cm_sharpe:.4f}\n")
        f.write(f"- **Direct LSTM (5-layer):** {direct_sharpe:.4f}\n\n")
        f.write("## Conclusion\n")
        if direct_sharpe > cm_sharpe:
            f.write("The **5-layer Direct LSTM** outperformed the pre-trained symbolic model in returns, despite slightly higher latency.\n")
        else:
            f.write("The **Symbolic Centered Model** remains more efficient and provided better risk-adjusted returns on this sample.\n")

if __name__ == "__main__":
    run_benchmark()
