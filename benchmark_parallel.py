import time
import numpy as np
import sagan
from sagan.ensemble import PortfolioSymbolicEngine
import logging

logging.basicConfig(level=logging.ERROR)

def run_parallel_benchmark():
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]
    # We'll use a small period to make it faster but still measurable
    period = "6mo"
    
    print(f"Benchmarking Portfolio Parallelism with {len(tickers)} tickers...")
    
    # 1. Sequential (Simulated by running one by one)
    print("Running Sequential Training...")
    start_seq = time.perf_counter()
    for t in tickers:
        engine = PortfolioSymbolicEngine([t], period=period, profile="balanced")
        engine.train_all()
    duration_seq = time.perf_counter() - start_seq
    
    # 2. Parallel (New Architecture)
    print("Running Parallel Training...")
    start_par = time.perf_counter()
    engine = PortfolioSymbolicEngine(tickers, period=period, profile="balanced")
    engine.train_all()
    duration_par = time.perf_counter() - start_par
    
    speedup = duration_seq / duration_par
    
    print(f"\n--- Results ---")
    print(f"Sequential Duration: {duration_seq:.2f}s")
    print(f"Parallel Duration:   {duration_par:.2f}s")
    print(f"Speedup:             {speedup:.2f}x")
    print(f"Significant:        {speedup > 1.2} (Expect >1x improvement on multi-core)")

if __name__ == "__main__":
    run_parallel_benchmark()
