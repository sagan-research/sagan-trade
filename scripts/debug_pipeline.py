import sys
import os
import logging
import json
import numpy as np
from datetime import datetime

# Ensure sagan is in path
sys.path.append(os.getcwd())

import sagan
from sagan.desk import run_research_backtest

def main():
    # Set logging to INFO to see what's happening
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    tickers = ["AAPL"] # Just one for debugging
    
    print(f"--- Debug Research Pipeline for {tickers} ---")
    
    model_ids = []
    for t in tickers:
        print(f"Developing symbolic foundation for {t}...")
        try:
            mid = sagan.train([t], signals=["Close", "Volume"], target_r2=0.92)
            model_ids.append(mid)
            print(f"  OK: {mid}")
        except Exception as e:
            print(f"  FAILED {t}: {e}")
            
    if not model_ids:
        print("Error: No models were trained.")
        return
        
    print("\n--- Executing Backtest ---")
    results = run_research_backtest(tickers, model_ids, years=2, commission=0.0005)
    print("Backtest results obtained.")
    print(results)

if __name__ == "__main__":
    main()
