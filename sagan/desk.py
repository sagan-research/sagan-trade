import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Any
from sagan.registry import load_ensemble
from sagan.models.math_engine import MathematicalEngine
from sagan.utils import sharpe_ratio, max_drawdown, annualised_return, win_rate

logger = logging.getLogger("sagan.desk")

class AlphaDesk:
    """
    Coordinated Trading Desk that manages multiple symbolic models
    and applies statistical execution logic.
    Optimized for high-fidelity Vanilla Symbolic execution.
    """
    def __init__(self, model_ids: List[str], mode: str = "coordinated"):
        self.model_ids = model_ids
        self.models = {}
        self.thresholds = {}
        self.mode = mode # coordinated, market_neutral, long_only
        
        self._initialize_desk()

    def _initialize_desk(self):
        """Load all models and initialize tracking."""
        for mid in self.model_ids:
            try:
                _, _, _, _, metadata = load_ensemble(mid)
                ticker = metadata["tickers"][0]
                self.models[ticker] = metadata
                # Default thresholds (overwritten by optimize_thresholds)
                self.thresholds[ticker] = {"buy": 1.0, "sell": -1.0} 
            except Exception as e:
                logger.error(f"Desk: Failed to load model {mid}: {e}")

    def optimize_thresholds(self, training_data: Dict[str, pd.DataFrame]):
        """
        Statistical method to determine optimal Buy/Sell levels based on formula variance.
        """
        for ticker, data in training_data.items():
            if ticker not in self.models: continue
            
            meta = self.models[ticker]
            formula = meta["composite_formula"]
            fitted = meta["fitted_signals"]
            
            t = np.arange(len(data))
            eval_context = {s: MathematicalEngine.evaluate(f["func"], t, f["params"]) for s, f in fitted.items()}
            eval_context.update({"np": np, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos})
            
            try:
                clean_formula = formula.replace("^", "**")
                outputs = []
                for i in range(len(data)):
                    ctx = {s: eval_context[s][i] for s in fitted}
                    ctx.update({"np": np, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos})
                    outputs.append(eval(clean_formula, {"__builtins__": {}}, ctx))
                
                outputs = np.array(outputs)
                mu, sigma = np.mean(outputs), np.std(outputs)
                
                # Set statistical buy/sell triggers at 1.2 standard deviations
                self.thresholds[ticker] = {
                    "buy": mu + (1.2 * sigma),
                    "sell": mu - (1.2 * sigma),
                    "mean": mu, "std": sigma
                }
            except Exception as e:
                logger.error(f"Desk: Threshold optimization failed for {ticker}: {e}")

    def coordinate_signals(self, current_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Main execution logic: evaluates symbolic formulas for each asset
        and aggregates into a portfolio decision.
        """
        signals = {}
        
        # Vanilla Symbolic Evaluation
        for ticker, meta in self.models.items():
            if ticker not in current_data: continue
            
            formula = meta["composite_formula"]
            val_context = current_data[ticker]
            val_context.update({"np": np, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos})
            
            try:
                clean_formula = formula.replace("^", "**")
                # Direct deterministic evaluation
                raw_signal = eval(clean_formula, {"__builtins__": {}}, val_context)
                
                thresh = self.thresholds.get(ticker, {"buy": 1.0, "sell": -1.0})
                if raw_signal > thresh["buy"]: signals[ticker] = 1.0
                elif raw_signal < thresh["sell"]: signals[ticker] = -1.0
                else: signals[ticker] = 0.0
            except:
                signals[ticker] = 0.0
                
        # Apply Mode-specific logic
        if self.mode == "long_only":
            signals = {t: max(0, s) for t, s in signals.items()}
        elif self.mode == "market_neutral":
            net_val = sum(signals.values())
            if len(signals) > 0:
                adj = net_val / len(signals)
                signals = {t: s - adj for t, s in signals.items()}

        # Institutional Exposure Capping (Vanilla default: 2.5x)
        total_exposure = sum(abs(s) for s in signals.values())
        max_exp = 2.5
        
        if total_exposure > max_exp:
            scale = max_exp / total_exposure
            signals = {t: s * scale for t, s in signals.items()}
            
        return signals

def run_research_backtest(tickers: List[str], model_ids: List[str], years: int = 2, commission: float = 0.0005, mode: str = "coordinated"):
    """
    Institutional backtest with transaction costs and statistical benchmarks.
    """
    import yfinance as yf
    desk = AlphaDesk(model_ids, mode=mode)
    
    data_map = {}
    for t in tickers:
        df = yf.download(t, period=f"{years}y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0) if 'Close' in df.columns.get_level_values(0) else df.columns.get_level_values(1)
        
        close_col = 'Close' if 'Close' in df.columns else 'Adj Close'
        df['Returns'] = df[close_col].pct_change()
        data_map[t] = df.dropna()
        
    train_map = {t: df.iloc[:len(df)//2] for t, df in data_map.items()}
    desk.optimize_thresholds(train_map)
    
    test_map = {t: df.iloc[len(df)//2:] for t, df in data_map.items()}
    common_dates = None
    for df in test_map.values():
        if common_dates is None: common_dates = df.index
        else: common_dates = common_dates.intersection(df.index)
            
    if common_dates is None or len(common_dates) == 0:
        return None
        
    test_map = {t: df.loc[common_dates] for t, df in test_map.items()}
    min_len = len(common_dates)
    
    strategy_returns = []
    benchmark_returns = []
    prev_signals = {t: 0.0 for t in tickers}
    
    for i in range(min_len):
        current_data = {}
        day_bench_ret = 0
        for t in tickers:
            row = test_map[t].iloc[i]
            current_data[t] = {col: row[col] for col in row.index if col != 'Returns'}
            day_bench_ret += row['Returns'] / len(tickers)
            
        signals = desk.coordinate_signals(current_data)
        
        daily_ret = 0
        if i < min_len - 1:
            for t, sig in signals.items():
                ret = test_map[t]['Returns'].iloc[i+1]
                cost = abs(sig - prev_signals[t]) * commission
                daily_ret += (sig * ret - cost) / len(tickers)
                prev_signals[t] = sig
        
        strategy_returns.append(daily_ret)
        benchmark_returns.append(day_bench_ret)
        
    s_ret = np.array(strategy_returns)
    b_ret = np.array(benchmark_returns)
    
    t_stat, p_val = stats.ttest_ind(s_ret, b_ret)
    return {
        "strategy": {
            "annual_return": annualised_return(s_ret),
            "sharpe": sharpe_ratio(s_ret),
            "mdd": max_drawdown(s_ret),
            "win_rate": win_rate(s_ret),
            "cum_return": np.prod(1 + s_ret) - 1
        },
        "benchmark": {
            "annual_return": annualised_return(b_ret),
            "sharpe": sharpe_ratio(b_ret),
            "mdd": max_drawdown(b_ret),
            "cum_return": np.prod(1 + b_ret) - 1
        },
        "stats": {
            "t_stat": t_stat, "p_value": p_val,
            "alpha": annualised_return(s_ret) - annualised_return(b_ret),
            "info_ratio": (np.mean(s_ret - b_ret) / np.std(s_ret - b_ret)) * np.sqrt(252) if np.std(s_ret - b_ret) != 0 else 0
        }
    }
