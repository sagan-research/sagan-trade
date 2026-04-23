import logging
from typing import Any, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sagan.config import config
from sagan.signals import get_available_signals, fetch_signal_data
from sagan.models.math_engine import MathematicalEngine, fit_signal_worker
from sagan.models.llm_bridge import FunctionGemmaBridge
from sagan.models.manager import ResourceManager
from sagan.registry import save_model
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger("sagan.ensemble")

class SymbolicRegressor:
    """
    Symbolic Regressor that fits math functions to features and combines them.
    """
    def __init__(
        self,
        tickers: List[str],
        signals: List[str] = None,
        target_r2: float = 0.95,
        period: str = "1y",
        profile: str = "balanced",
    ):
        self.tickers = tickers
        self.target_r2 = target_r2
        self.period = period
        self.signals = signals or ["Open", "High", "Low", "Adj Close", "Volume"]
        self.scaler = StandardScaler()
        self.meta = {}
        
        self.resource_manager = ResourceManager(profile)
        self.llm = FunctionGemmaBridge()
        
        # Stores fitted parameters for each signal: {signal: (func_name, params, r2)}
        self.fitted_signals = {}
        self.composite_formula = None

    def train(self, progress_callback: Any = None):
        """
        Executes the symbolic training workflow with OS-level optimizations.
        """
        self.resource_manager.apply_optimizations()
        if progress_callback: progress_callback(0.05)
        
        # 1. Fetch Data
        ticker = self.tickers[0]
        data = fetch_signal_data(ticker, self.signals, period=self.period)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
            
        if progress_callback: progress_callback(0.15)
        
        # 2. Parallel Fitting (Max Throughput)
        worker_count = self.resource_manager.get_worker_count()
        logger.info(f"Parallellizing fit across {worker_count} workers...")
        
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(fit_signal_worker, data[s].values, s, self.target_r2)
                for s in self.signals
            ]
            
            for i, future in enumerate(futures):
                try:
                    s_name, result = future.result()
                    self.fitted_signals[s_name] = result
                    if progress_callback:
                        progress_callback(0.15 + (0.5 * (i+1)/len(futures)))
                except Exception as e:
                    logger.error(f"Fitting failed for signal: {e}")

        # 3. Discover Composite Function via LLM
        logger.info("Discovering composite function via FunctionGemma...")
        self.composite_formula = self.llm.suggest_composite_function("Adj_Close_Trend", self.signals)
        logger.info(f"Discovered: {self.composite_formula}")
        
        if progress_callback: progress_callback(0.8)
        
        # 4. Finalize Metadata
        self.meta = {
            "tickers": self.tickers,
            "signals": self.signals,
            "fitted_signals": self.fitted_signals,
            "composite_formula": self.composite_formula,
            "val_r2": np.mean([v["r2"] for v in self.fitted_signals.values()]),
            "created_at": pd.Timestamp.now().isoformat(),
        }
        
        return self.meta

    def save(self) -> str:
        # For symbolic, model_buy/sell/hold can all share the same logic or be variations
        # Here we save the fitted signals and formula
        return save_model(
            model_buy=self.meta,
            model_sell=self.meta,
            model_hold=self.meta,
            scaler=self.scaler,
            metadata=self.meta,
            is_symbolic=True
        )

class PortfolioSymbolicEngine:
    """
    Manages a basket of tickers, fitting independent symbolic models for each.
    """
    def __init__(self, tickers: List[str], **kwargs):
        self.tickers = tickers
        self.kwargs = kwargs
        self.regressors = {t: SymbolicRegressor([t], **kwargs) for t in tickers}
        self.results = {}

    def train_all(self, progress_callback: Any = None):
        """
        Trains all tickers in the portfolio independently.
        """
        n = len(self.tickers)
        for i, (t, reg) in enumerate(self.regressors.items()):
            logger.info(f"Training Portfolio Component: {t}")
            
            # Simple wrapper for per-ticker progress
            def sub_callback(p):
                if progress_callback:
                    # Map 0-1 of subtask to i/n -> (i+1)/n of total
                    total_p = (i + p) / n
                    progress_callback(total_p)
            
            self.results[t] = reg.train(progress_callback=sub_callback)
            
        return self.results

    def save_all(self) -> dict:
        return {t: reg.save() for t, reg in self.regressors.items()}

def train(tickers: List[str], **kwargs) -> str:
    """Primary entry point for training."""
    regressor = SymbolicRegressor(tickers, **kwargs)
    regressor.train()
    return regressor.save()
