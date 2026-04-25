import sys
import os
import logging

# Ensure sagan is in path
sys.path.append(os.getcwd())

import sagan
from sagan.desk import run_desk_backtest

def main():
    logging.basicConfig(level=logging.ERROR)
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    print(f"--- Developing Alpha Desk Foundations for {tickers} ---")
    
    model_ids = []
    for t in tickers:
        print(f"Training {t}...")
        # Use simple signals for speed in this demo
        mid = sagan.train([t], signals=["Close", "Volume"], target_r2=0.92)
        model_ids.append(mid)
        print(f"OK: {mid}")
        
    print("\n--- Initializing Trading Desk and Running Backtest ---")
    run_desk_backtest(tickers, model_ids, years=2)

if __name__ == "__main__":
    main()
