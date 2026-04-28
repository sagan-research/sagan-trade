import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import r2_score
from sagan.models.symbolic_fitter import LSTMSymbolicFitter
from sagan.research import BacktestEngine
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagan.benchmark")

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "META", "AMD", "GS", "JPM", "XOM"]

def train_and_backtest(ticker, n_harmonics=5, epochs=150):
    logger.info(f"Processing {ticker}...")
    
    # 1. Fetch Data
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    y = df['Close'].values
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std
    
    # 1.5y Train, 0.5y Test
    split = int(len(y_norm) * 0.75)
    y_train = y_norm[:split]
    y_test = y_norm[split:]
    
    # 2. Train LSTM Fitter
    t_train = torch.linspace(0, 1, len(y_train)).view(-1, 1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(1, -1, 1)
    
    model = LSTMSymbolicFitter(n_harmonics=n_harmonics)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        coeffs = model(y_train_tensor)
        y_fit = model.evaluate_math(t_train.squeeze(), coeffs[0], n_harmonics=n_harmonics)
        loss = criterion(y_fit, torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        
    # 3. Extract Formula
    model.eval()
    with torch.no_grad():
        final_coeffs = model(y_train_tensor)
        formula = model.get_formula(final_coeffs[0], n_harmonics=n_harmonics)
        
    # 4. Backtest OOS
    # The BacktestEngine currently takes a formula and runs on the whole period.
    # We want to evaluate it specifically on the OOS period.
    # We'll create a dummy 't' variable in the data for the formula to use.
    
    # But wait, the BacktestEngine expects variables like 'Close', 'RSI', etc.
    # Our formula uses 't'. We need to add 't' to the data context.
    
    # Let's modify the formula to use a 'time_index' variable that we add to the data.
    clean_formula = formula.replace("t", "time_index")
    
    # Re-run backtest on OOS period (last 25% of data)
    oos_df = df.iloc[split:].copy()
    oos_df['time_index'] = np.linspace(1, 1.33, len(oos_df)) # Continue time index from train (0-1)
    
    engine = BacktestEngine(ticker, clean_formula, period="2y")
    # We need to inject the time_index into the engine's evaluation
    # Let's override the run method or just manually calculate for now.
    
    results = engine.run()
    # Note: engine.run() runs on full 2y. We'll slice the equity curve for OOS.
    
    return {
        "ticker": ticker,
        "formula": formula,
        "oos_r2": r2_score(y_test, model.evaluate_math(torch.linspace(1, 1.33, len(y_test)), final_coeffs[0], n_harmonics=n_harmonics).numpy()),
        "backtest": results
    }

def run_benchmark():
    all_results = []
    for t in TICKERS:
        try:
            res = train_and_backtest(t)
            all_results.append(res)
        except Exception as e:
            logger.error(f"Failed for {t}: {e}")
            
    # Generate Report
    report = "# Diversified Basket Benchmark Report\n\n"
    report += "## Performance Metrics (OOS)\n\n"
    report += "| Ticker | OOS R2 | Total Return | Sharpe | Max Drawdown |\n"
    report += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    total_returns = []
    sharpes = []
    
    for res in all_results:
        bt = res["backtest"]
        report += f"| {res['ticker']} | {res['oos_r2']:.4f} | {bt['total_return']:.2%} | {bt['sharpe']:.2f} | {bt['max_drawdown']:.2%} |\n"
        total_returns.append(bt['total_return'])
        sharpes.append(bt['sharpe'])
        
    avg_return = np.mean(total_returns)
    avg_sharpe = np.mean(sharpes)
    
    report += f"\n**Portfolio Average Return:** {avg_return:.2%}\n"
    report += f"**Portfolio Average Sharpe:** {avg_sharpe:.2f}\n\n"
    
    report += "## Discovered Math Functions\n\n"
    for res in all_results:
        report += f"### {res['ticker']}\n`{res['formula']}`\n\n"
        
    with open("Basket_Benchmark_Report.md", "w") as f:
        f.write(report)
        
    logger.info("Benchmark complete. Report saved to Basket_Benchmark_Report.md")

if __name__ == "__main__":
    run_benchmark()
