import time
import logging
from sagan.autonomous import AutonomousResearcher

logging.basicConfig(level=logging.INFO)

def verify():
    researcher = AutonomousResearcher()
    ticker = "GOOG"
    
    print(f"Launching Optimized Autonomous Pipeline for {ticker}...")
    start_time = time.time()
    
    results = researcher.run_full_pipeline(ticker, gating_mode="balanced")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n--- RESULTS ---")
    print(f"Ticker: {results['ticker']}")
    print(f"Formula: {results['formula']}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Status: {results['status']}")
    
    print("\n--- DEEP REASONING (XAI) ---")
    print(results.get("reasoning", "No reasoning found."))
    
    print("\n--- ADVICE ---")
    print(results.get("advice", "No advice found."))

if __name__ == "__main__":
    verify()
