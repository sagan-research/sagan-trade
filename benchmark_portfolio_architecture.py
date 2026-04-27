import time
import numpy as np
import pandas as pd
import logging
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sagan.models.math_engine import MathematicalEngine
from sagan.models.llm_bridge import FunctionGemmaBridge
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("benchmark")

class PortfolioArchitecturalBenchmark:
    def __init__(self, n_tickers: int = 5, n_days: int = 1000):
        self.n_tickers = n_tickers
        self.n_days = n_days
        self.tickers = [f"TICKER_{i:02d}" for i in range(n_tickers)]
        self.signals = ["Open", "High", "Low", "Close", "Volume"]
        self.engine = MathematicalEngine()
        self.llm = FunctionGemmaBridge()
        
        # Generate synthetic data
        self.data_map = self._generate_data()
        
        # Define complex symbolic formulas for each ticker using FunctionGemma
        self.formulas = self._generate_formulas()
        
        # Simulated fitted signals (basis functions)
        self.fitted_signals = self._generate_fitted_signals()

    def _generate_data(self) -> Dict[str, pd.DataFrame]:
        data_map = {}
        for ticker in self.tickers:
            df = pd.DataFrame(
                np.random.randn(self.n_days, len(self.signals)),
                columns=self.signals
            )
            data_map[ticker] = df
        return data_map

    def _generate_formulas(self) -> Dict[str, str]:
        formulas = {}
        print(f"Synthesizing Portfolio Strategy with FunctionGemma (n={self.n_tickers})...")
        for i, ticker in enumerate(self.tickers):
            print(f"  [WAIT] Ticker {i:02d} formula synthesis...", end="", flush=True)
            start = time.time()
            try:
                formula = self.llm.suggest_composite_function("Adj_Close", self.signals)
                formulas[ticker] = formula
                print(f" done in {time.time() - start:.2f}s")
            except Exception as e:
                print(f" failed ({e})")
                formulas[ticker] = "(Open + Close) / 2.0"
            print(f"    {ticker}: {formulas[ticker]}")
        return formulas

    def _generate_fitted_signals(self) -> Dict[str, Dict[str, dict]]:
        fitted = {}
        for ticker in self.tickers:
            t_fitted = {}
            for sig in self.signals:
                t_fitted[sig] = {
                    "func": "polynomial",
                    "params": [1.0, 0.5, 0.1]
                }
            fitted[ticker] = t_fitted
        return fitted

    def run_legacy_benchmark(self) -> float:
        """Simulates the old row-by-row, ticker-by-ticker evaluation."""
        start_time = time.perf_counter()
        for i in range(self.n_days):
            for ticker in self.tickers:
                formula = self.formulas[ticker]
                fitted = self.fitted_signals[ticker]
                t_val = np.array([float(i)])
                ctx = {s: self.engine.evaluate(f["func"], t_val, f["params"])[0] for s, f in fitted.items()}
                ctx.update({"np": np, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos, "abs": np.abs, "sqrt": np.sqrt})
                clean_formula = formula.replace("^", "**")
                try:
                    _ = eval(clean_formula, ctx)
                except:
                    pass
        return time.perf_counter() - start_time

    def run_new_benchmark(self) -> float:
        """Uses the new vectorized architecture."""
        start_time = time.perf_counter()
        for ticker in self.tickers:
            formula = self.formulas[ticker]
            fitted = self.fitted_signals[ticker]
            data = self.data_map[ticker]
            _ = self.engine.evaluate_ensemble(formula, fitted, data)
        return time.perf_counter() - start_time

    def run(self):
        print(f"\n--- Sagan Architecture Benchmark ---")
        print(f"Portfolio Size: {self.n_tickers} stocks")
        print(f"Data Horizon:   {self.n_days} days")
        print(f"Total Evals:    {self.n_tickers * self.n_days:,}\n")
        
        print("Running Legacy Workflow (Row-by-Row)...")
        legacy_time = self.run_legacy_benchmark()
        print(f"Legacy Time:    {legacy_time:.4f}s")
        
        print("Running New Workflow (Vectorized)...")
        new_time = self.run_new_benchmark()
        print(f"New Time:       {new_time:.4f}s")
        
        speedup = legacy_time / new_time
        print(f"\n--- Results ---")
        print(f"Speedup:        {speedup:.2f}x")
        print(f"Latency/Day (Legacy): { (legacy_time / self.n_days) * 1000:.4f}ms")
        print(f"Latency/Day (New):    { (new_time / self.n_days) * 1000:.4f}ms")

if __name__ == "__main__":
    # Use 5 tickers for a reliable test with the LLM
    benchmark = PortfolioArchitecturalBenchmark(n_tickers=5)
    benchmark.run()
