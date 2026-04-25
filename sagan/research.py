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
    def __init__(self, ticker: str, formula: str, period: str = "2y"):
        self.ticker = ticker
        self.formula = formula
        self.period = period
        self.engine = MathematicalEngine()

    def run(self) -> Dict[str, Any]:
        """
        Runs the backtest and returns a dictionary of metrics and equity curve data.
        """
        # 1. Fetch common signals to ensure we have enough data context
        common_signals = ["Close", "Volume", "RSI", "SMA_20", "Open", "High", "Low", "Adj Close"]
        
        try:
            data = fetch_signal_data(self.ticker, common_signals, period=self.period)
            if data.empty:
                return {"status": "error", "message": "No data found for ticker."}
            
            # Ensure "Close" exists for returns calculation
            if "Close" not in data.columns and "Adj Close" in data.columns:
                data["Close"] = data["Adj Close"]
            
            # 2. Evaluate the formula
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
            # Replace spaces in variable names in formula if any (e.g. 'Adj Close' -> 'Adj_Close')
            for col in data.columns:
                if " " in col:
                    clean_formula = clean_formula.replace(col, col.replace(" ", "_"))
            
            signal_values = eval(clean_formula, {"__builtins__": {}}, eval_context)
            
            # 3. Generate Trading Signals (LONG if value > 0, else SHORT)
            # Binary signal: 1 for long, -1 for short
            signals = np.where(signal_values > 0, 1.0, -1.0)
            
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

class StrategyRefiner:
    """
    Communicates with FunctionGemma to iterate on a strategy based on backtest performance.
    """
    def __init__(self, bridge: FunctionGemmaBridge = None):
        self.bridge = bridge or FunctionGemmaBridge()

    def refine(self, current_formula: str, backtest_results: Dict[str, Any]) -> str:
        """
        Asks FunctionGemma to improve the formula based on performance metrics.
        """
        if backtest_results["status"] != "success":
            return current_formula

        prompt = f"""
        [INST] <<SYS>>
        You are a quantitative researcher specializing in symbolic regression.
        Your task is to improve a trading formula based on its backtest performance.
        Return ONLY the improved Python/NumPy formula. Do not explain.
        <</SYS>>

        Current Formula: `{current_formula}`
        Ticker: {backtest_results['ticker']}
        Performance:
        - Total Return: {backtest_results['total_return']:.2%}
        - Sharpe Ratio: {backtest_results['sharpe']:.2f}
        - Max Drawdown: {backtest_results['max_drawdown']:.2%}
        - Win Rate: {backtest_results['win_rate']:.2%}

        Task: Suggest a mathematically refined version of this formula.
        If Sharpe is low (< 1.0), consider adding momentum (SMA_20) or mean-reversion (RSI).
        If Drawdown is high (> 20%), consider volatility damping or regime filters.
        Available variables: Close, Volume, RSI, SMA_20, Open, High, Low.

        Improved Formula: [/INST]"""
        
        try:
            # We'll use the bridge's client directly for custom prompt
            response = self.bridge.client.generate(model=self.bridge.model, prompt=prompt)
            new_formula = response['response'].strip().replace("```python", "").replace("```", "").strip()
            return new_formula
        except Exception as e:
            logger.error(f"Refinement call failed: {e}")
            return current_formula
