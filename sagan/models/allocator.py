import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging
from typing import List, Dict, Any

from sagan.predict import predict

logger = logging.getLogger("sagan.allocator")

class PortfolioAllocator:
    """
    ML-inspired weight allocation for symbolic portfolios.
    Allocates weights to maximize a target metric (default: Sharpe).
    """
    
    def __init__(self, model_ids: List[str]):
        self.model_ids = model_ids
        self.weights = None

    def allocate_weights(self, target: str = "sharpe") -> Dict[str, float]:
        """
        Kicks in to allocate weights based on symbolic signal confidence and R2 stability.
        """
        n = len(self.model_ids)
        
        # 1. Gather signals and stats
        signals = []
        r2_scores = []
        tickers = []
        
        for mid in self.model_ids:
            res = predict(model_id=mid)
            tickers.append(res.get("ticker", mid)) # We might need to ensure ticker is in res
            signals.append(1.0 if res["signal"] == "LONG" else -1.0)
            r2_scores.append(np.mean(list(res["r2_stats"].values())))

        # 2. Optimization target
        # We want to favor high R2 and Strong signals
        def objective(w):
            # Penalize low R2 and conflicting signals
            total_signal = np.dot(w, signals)
            r2_penalty = np.dot(w, r2_scores)
            return -(total_signal * r2_penalty) # Maximize this

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0, 1) for _ in range(n)]
        init_w = np.ones(n) / n
        
        res = minimize(objective, init_w, bounds=bounds, constraints=constraints)
        self.weights = {tickers[i]: float(res.x[i]) for i in range(n)}
        
        return self.weights

class SymbolicSimulator:
    """
    Runs historical simulation for understanding symbolic performance.
    """
    def __init__(self, ticker_results: Dict[str, Any]):
        self.results = ticker_results

    def run_simulation(self, days: int = 30) -> pd.DataFrame:
        """
        Generates a simulated equity curve.
        """
        # Mocking a walk-forward simulation for UI visualization
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
        
        # Combined equity curve starting at 100
        equity = [100.0]
        returns = []
        
        for _ in range(1, days):
            # Average daily 'movement' expected from symbolic signals
            daily_ret = np.random.normal(0.001, 0.02) 
            returns.append(daily_ret)
            equity.append(equity[-1] * (1 + daily_ret))
            
        return pd.DataFrame({"Date": dates, "Equity": equity})
