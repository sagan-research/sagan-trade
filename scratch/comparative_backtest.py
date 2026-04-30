import pandas as pd
import numpy as np
import logging
import time
from typing import List, Dict, Any
from sagan.ensemble import SymbolicRegressor
from sagan.fundamental import FundamentalAnalyzer
from sagan.signals import fetch_signal_data
from sagan.research import BacktestEngine

# Set up logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='comparative_backtest.log',
    filemode='w'
)
logger = logging.getLogger("comparative_backtest")

# 10 NYSE Stocks (Diversified)
NYSE_TICKERS = ["JPM", "V", "WMT", "PG", "UNH", "XOM", "JNJ", "KO", "ABBV", "BAC"]
class ComparativeBacktest:
    def __init__(self, tickers: List[str], period: str = "5y"):
        self.tickers = tickers
        self.period = period
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.results = []

    def run(self):
        print(f"Starting Comparative Backtest for {len(self.tickers)} tickers (Memory Optimized)...")
        for ticker in self.tickers:
            try:
                print(f"Processing {ticker}...")
                
                # 1. Fundamental Bias
                f_data = self.fundamental_analyzer.calculate_bias(ticker)
                bias_score = f_data["score"]
                bias_str = f_data["bias"]
                
                # Map bias string to numerical
                # Bullish: 1, Bearish: -1, Neutral: 0
                bias_val = 1 if bias_str == "Bullish" else (-1 if bias_str == "Bearish" else 0)
                
                # 2. Train Symbolic Model (Technical Baseline)
                # Using 'eco' profile to save memory
                regressor = SymbolicRegressor([ticker], signals=["Adj Close", "Volume", "RSI", "SMA_20"], period=self.period, profile="eco")
                model_meta = regressor.train()
                formula = model_meta["composite_formula"]
                
                # 3. Fetch Data for Custom Backtest
                data = fetch_signal_data(ticker, ["Adj Close", "Volume", "RSI", "SMA_20", "Close"], period=self.period)
                if data.empty:
                    continue
                
                # 4. Evaluate Technical Signals
                eval_context = {col.replace(" ", "_"): data[col].values for col in data.columns}
                eval_context.update({"np": np, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos, "abs": np.abs, "sqrt": np.sqrt})
                clean_formula = formula.replace("^", "**")
                
                # Sanitize variable names in formula
                for col in data.columns:
                    if " " in col:
                        clean_formula = clean_formula.replace(col, col.replace(" ", "_"))
                
                technical_raw = eval(clean_formula, {"__builtins__": {}}, eval_context)
                tech_signals = np.where(technical_raw > 0, 1.0, -1.0)
                
                # 5. Calculate Returns
                asset_returns = data["Close"].pct_change().shift(-1).fillna(0)
                
                # --- Strategy A: Technical Only (Previous) ---
                returns_a = tech_signals * asset_returns
                cum_a = (1 + returns_a).cumprod()
                total_a = float(cum_a.iloc[-1] - 1)
                sharpe_a = (np.mean(returns_a) / (np.std(returns_a) + 1e-9)) * np.sqrt(252)
                
                # --- Strategy B: Fundamental Gated (New) ---
                # Logic: Only trade in the direction of fundamental bias if signal exists.
                # If bias is neutral, follow technicals.
                if bias_val == 1: # Bullish Bias
                    gated_signals = np.where(tech_signals == 1, 1.0, 0.0) # Only Long
                elif bias_val == -1: # Bearish Bias
                    gated_signals = np.where(tech_signals == -1, -1.0, 0.0) # Only Short
                else: # Neutral Bias
                    gated_signals = tech_signals # Follow technicals
                
                returns_b = gated_signals * asset_returns
                cum_b = (1 + returns_b).cumprod()
                total_b = float(cum_b.iloc[-1] - 1)
                sharpe_b = (np.mean(returns_b) / (np.std(returns_b) + 1e-9)) * np.sqrt(252)
                
                self.results.append({
                    "Ticker": ticker,
                    "Bias": bias_str,
                    "Bias Score": bias_score,
                    "Return (Tech Only)": total_a,
                    "Return (Gated)": total_b,
                    "Sharpe (Tech Only)": sharpe_a,
                    "Sharpe (Gated)": sharpe_b,
                    "Improvement (%)": (total_b - total_a) * 100
                })
                
            except Exception as e:
                print(f"Error on {ticker}: {e}")
                logger.error(f"Error on {ticker}: {e}")

        # Summary
        df = pd.DataFrame(self.results)
        print("\n--- Comparative Results ---")
        print(df.to_string(index=False))
        
        avg_a = df["Return (Tech Only)"].mean()
        avg_b = df["Return (Gated)"].mean()
        avg_sharpe_a = df["Sharpe (Tech Only)"].mean()
        avg_sharpe_b = df["Sharpe (Gated)"].mean()
        
        print("\n--- Aggregated Performance ---")
        print(f"Average Return (Tech Only): {avg_a:.2%}")
        print(f"Average Return (Gated):     {avg_b:.2%}")
        print(f"Average Sharpe (Tech Only): {avg_sharpe_a:.2f}")
        print(f"Average Sharpe (Gated):     {avg_sharpe_b:.2f}")
        
        if avg_b > avg_a:
            print("\nResult: The Fundamental-Gated strategy is BETTER on average.")
        else:
            print("\nResult: The Technical-Only strategy performed better on average.")
            
        df.to_csv("comparative_results.csv", index=False)

if __name__ == "__main__":
    tester = ComparativeBacktest(NYSE_TICKERS)
    tester.run()