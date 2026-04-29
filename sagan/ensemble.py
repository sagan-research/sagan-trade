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
        period: str = "1y",
        profile: str = "balanced",
    ):
        self.tickers = tickers
        self.period = period
        self.signals = signals or ["Open", "High", "Low", "Adj Close", "Volume"]
        self.scaler = StandardScaler()
        self.meta = {}
        
        self.resource_manager = ResourceManager(profile)
        self.llm = FunctionGemmaBridge()
        
        # Stores fitted parameters for each signal: {signal: (func_name, params, r2)}
        self.fitted_signals = {}
        self.composite_formula = None

    def train(self, progress_callback: Any = None, data: pd.DataFrame = None):
        """
        Executes the symbolic training workflow with OS-level optimizations.
        """
        self.resource_manager.apply_optimizations()
        if progress_callback: progress_callback(0.05)
        
        # 1. Fetch Data if not provided
        if data is None:
            ticker = self.tickers[0]
            data = fetch_signal_data(ticker, self.signals, period=self.period)
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
        
        # Ensure self.signals matches data columns
        self.signals = [s for s in self.signals if s in data.columns]
            
        if progress_callback: progress_callback(0.15)
        
        # 2. Parallel Fitting (Max Throughput)
        worker_count = self.resource_manager.get_worker_count()
        logger.info(f"Parallellizing fit across {worker_count} workers...")
        
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(fit_signal_worker, data[s].values, s)
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

        # 3. Discover Composite Function (Nonlinear Search)
        logger.info("Discovering optimal nonlinear composition...")
        
        # Split data for out-of-sample validation
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        # Get candidates from LLM
        candidates = self.llm.suggest_candidates("Adj_Close_Trend", self.signals, count=5)
        
        # Add robust default nonlinear forms
        if len(self.signals) >= 2:
            s1, s2 = self.signals[0], self.signals[1]
            candidates.extend([
                f"({s1} * {s2})",
                f"np.log(np.abs({s1}) + 1) * {s2}",
                f"({s1} / (np.abs({s2}) + 1))",
                f"np.sin({s1}) * np.exp({s2} / np.max(np.abs({s2})))",
                f"({s1} ** 2) - ({s2} ** 2)",
                f"np.sqrt(np.abs({s1})) + np.sqrt(np.abs({s2}))",
                f"({s1} + {s2}) / 2", # Simple average
                f"({s1} * 0.7) + ({s2} * 0.3)" # Weighted linear
            ])
        elif len(self.signals) == 1:
            s1 = self.signals[0]
            candidates.extend([
                f"np.log(np.abs({s1}) + 1)",
                f"np.sin({s1})",
                f"({s1} ** 2)"
            ])
        
        # Select the best one based on validation R2
        engine = MathematicalEngine()
        target_signal = self.signals[0] # Use the primary signal as target
        best_formula, best_r2 = engine.find_best_composition(train_data, val_data, target_signal, candidates)
        
        self.composite_formula = best_formula
        logger.info(f"Optimal Formula: {self.composite_formula} (Val R2: {best_r2:.4f})")
        
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
        Trains all tickers in the portfolio in parallel.
        Uses process-level concurrency for maximum throughput.
        """
        n = len(self.tickers)
        worker_count = self.regressors[self.tickers[0]].resource_manager.get_worker_count()
        
        # Parallelize across tickers
        logger.info(f"Parallellizing portfolio training across {worker_count} tickers...")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            # Pass only the necessary data to the worker to avoid pickling complex objects
            future_to_ticker = {
                executor.submit(_train_ticker_worker_static, t, self.kwargs): t 
                for t in self.tickers
            }
            
            for i, future in enumerate(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    self.results[ticker] = future.result()
                    # Re-initialize the local regressor's state with the result if needed
                    self.regressors[ticker].meta = self.results[ticker]
                    if progress_callback:
                        progress_callback((i + 1) / n)
                except Exception as e:
                    logger.error(f"Portfolio: Training failed for {ticker}: {e}")
                    
        return self.results

    def save_all(self) -> dict:
        return {t: reg.save() for t, reg in self.regressors.items()}

def _train_ticker_worker_static(ticker: str, kwargs: dict):
    """Static worker function to avoid pickling self."""
    # Ensure logging is configured in the child process if needed
    reg = SymbolicRegressor([ticker], **kwargs)
    return reg.train()

def train(tickers: List[str], **kwargs) -> str:
    """Primary entry point for training. Handles arbitrary kwargs by filtering."""
    # Filter kwargs for SymbolicRegressor
    valid_keys = {"signals", "period", "profile"}
    reg_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    
    regressor = SymbolicRegressor(tickers, **reg_kwargs)
    regressor.train()
    return regressor.save()
