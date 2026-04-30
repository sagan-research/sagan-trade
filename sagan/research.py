import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from sagan.signals import fetch_signal_data
from sagan.models.math_engine import MathematicalEngine
from sagan.models.llm_bridge import FunctionGemmaBridge

logger = logging.getLogger("sagan.research")

class BacktestEngine:
    """
    Evaluates a symbolic formula on historical data and calculates performance metrics.
    """
    def __init__(self, ticker: str, formula: str, period: str = "2y", fundamental_score: float = 0.0, gating_mode: str = "none"):
        self.ticker = ticker
        self.formula = formula
        self.period = period
        self.fundamental_score = fundamental_score
        self.gating_mode = gating_mode
        self.engine = MathematicalEngine()

    def run(self) -> Dict[str, Any]:
        """
        Runs the backtest and returns a dictionary of metrics and equity curve data.
        """
        # 1. Fetch common signals + any signals in the formula
        common_signals = ["Close", "Volume", "RSI", "SMA_20", "Open", "High", "Low", "Adj Close"]
        
        # Improved extraction of likely signals from formula
        import re
        tokens = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', self.formula)
        
        # Filter out keywords, math functions, and standard historical parts
        excluded = [
            "np", "exp", "log", "sin", "cos", "abs", "sqrt", "t", "time_index", 
            "max", "min", "mean", "std", "var",
            "open", "high", "low", "close", "volume", "adj"
        ]
        formula_signals = [t for t in tokens if t.lower() not in excluded]
        
        # Convert underscores back to spaces for yfinance if it's a known historical signal
        historical_map = {"Adj_Close": "Adj Close", "Adj_High": "Adj High", "Adj_Low": "Adj Low", "Adj_Open": "Adj Open"}
        final_signals = []
        for s in formula_signals:
            if s in historical_map:
                final_signals.append(historical_map[s])
            else:
                final_signals.append(s)
        
        all_signals = list(set(common_signals + final_signals))
        
        try:
            data = fetch_signal_data(self.ticker, all_signals, period=self.period)
            if data.empty:
                return {"status": "error", "message": "No data found for ticker."}
            
            # Ensure "Close" exists for returns calculation
            if "Close" not in data.columns and "Adj Close" in data.columns:
                data["Close"] = data["Adj Close"]
            
            # 2. Evaluate the formula
            data["time_index"] = np.linspace(0, 1, len(data))
            eval_context = {col.replace(" ", "_"): data[col].values for col in data.columns}
            eval_context.update({
                "np": np, 
                "exp": np.exp, 
                "log": np.log, 
                "sin": np.sin, 
                "cos": np.cos,
                "abs": np.abs,
                "sqrt": np.sqrt
            })
            
            # Clean formula (replace ^ with **)
            clean_formula = self.formula.replace("^", "**")
            # Replace 't' with 'time_index' for consistency
            if " t " in f" {clean_formula} ":
                clean_formula = clean_formula.replace(" t ", " time_index ")
            
            # Replace spaces in variable names in formula if any (e.g. 'Adj Close' -> 'Adj_Close')
            for col in data.columns:
                if " " in col:
                    clean_formula = clean_formula.replace(col, col.replace(" ", "_"))
            
            signal_values = eval(clean_formula, {"__builtins__": {}}, eval_context)
            
            # 3. Generate Trading Signals
            # Raw technical signal: 1 for long, -1 for short
            tech_signals = np.where(signal_values > 0, 1.0, -1.0)
            
            # Apply Fundamental Gating if requested
            if self.gating_mode != "none":
                from sagan.fundamental import FundamentalAnalyzer
                fa = FundamentalAnalyzer()
                # Vectorized application of gating
                signals = np.array([fa.get_hybrid_weight(s, self.fundamental_score, self.gating_mode) for s in tech_signals])
            else:
                signals = tech_signals
            
            # 4. Calculate Returns
            # Using daily returns of the asset
            asset_returns = data["Close"].pct_change().shift(-1).fillna(0) # Forward daily returns
            
            # Strategy returns = Signal * Next Day's Asset Return
            strat_returns = signals * asset_returns
            
            # Cumulative returns
            cum_returns = (1 + strat_returns).cumprod()
            
            # Benchmarks (Buy & Hold)
            bh_returns = (1 + asset_returns).cumprod()
            
            # 5. Metrics
            total_return = float(cum_returns.iloc[-1] - 1) if not cum_returns.empty else 0
            bh_total_return = float(bh_returns.iloc[-1] - 1) if not bh_returns.empty else 0
            
            # Annualized Sharpe
            daily_std = np.std(strat_returns)
            sharpe = (np.mean(strat_returns) / (daily_std + 1e-9)) * np.sqrt(252) if daily_std > 0 else 0
            
            # Max Drawdown
            rolling_max = cum_returns.cummax()
            drawdown = (cum_returns - rolling_max) / (rolling_max + 1e-9)
            max_drawdown = float(drawdown.min())
            
            # Win Rate
            win_rate = float(np.sum(strat_returns > 0) / np.sum(strat_returns != 0)) if np.sum(strat_returns != 0) > 0 else 0
            
            return {
                "ticker": self.ticker,
                "formula": self.formula,
                "total_return": total_return,
                "bh_return": bh_total_return,
                "sharpe": float(sharpe),
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "equity_curve": cum_returns.tolist(),
                "bh_curve": bh_returns.tolist(),
                "dates": [d.strftime("%Y-%m-%d") for d in data.index],
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Backtest failed for {self.ticker}: {e}")
            return {"status": "error", "message": str(e)}
