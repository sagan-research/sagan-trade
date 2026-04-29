import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from sagan.models.llm_bridge import FunctionGemmaBridge
from sagan.ensemble import SymbolicRegressor
from sagan.research import BacktestEngine
from sagan.signals import fetch_signal_data
from sagan.fundamental import FundamentalAnalyzer
from sagan.explain import XAIOrchestrator

logger = logging.getLogger("sagan.autonomous")

class AutonomousResearcher:
    """
    Orchestrates the autonomous alpha discovery loop:
    Discovery -> Optimization -> Backtest -> Advice.
    """
    def __init__(self, bridge: FunctionGemmaBridge = None):
        self.llm = bridge or FunctionGemmaBridge()
        self.fundamental = FundamentalAnalyzer()

    def run_full_pipeline(self, ticker: str, period: str = "2y", gating_mode: str = "balanced") -> Dict[str, Any]:
        """
        Executes the end-to-end research pipeline for a given ticker.
        """
        logger.info(f"--- Starting Autonomous Pipeline for {ticker} ---")
        
        # 1. Discovery Phase
        logger.info("[1/4] Discovering relevant signals...")
        signals = self.llm.suggest_relevant_signals(ticker)
        # Ensure we always have Close
        if "Adj Close" not in signals and "Close" not in signals:
            signals.append("Adj Close")
        
        # 2. Optimization Phase (Auto-Train)
        logger.info(f"[2/4] Fetching data for signals: {signals}")
        data = fetch_signal_data(ticker, signals, period=period)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
            
        # Filter signals that actually exist in the data
        valid_signals = [s for s in signals if s in data.columns]
        logger.info(f"Optimizing symbolic models using valid signals: {valid_signals}")
        
        regressor = SymbolicRegressor([ticker], signals=valid_signals, period=period, profile="balanced")
        model_meta = regressor.train(data=data)
        model_id = regressor.save()
        
        # 2.5 Fundamental Analysis (The 'WHY' - Moved up for Gating)
        logger.info(f"[2.5/4] Analyzing fundamentals for {ticker}...")
        fundamental_data = self.fundamental.calculate_bias(ticker)
        execution_risk = self.fundamental.check_execution_risk(ticker)
        
        # 3. Backtest Phase
        logger.info(f"[3/4] Validating strategy via backtest (Gating: {gating_mode})...")
        engine = BacktestEngine(ticker, model_meta["composite_formula"], period=period, fundamental_score=fundamental_data["score"], gating_mode=gating_mode)
        backtest_results = engine.run()
        
        # 4. Narrative & Advice Phase
        logger.info("[4/4] Generating XAI Decision Narrative...")
        xai = XAIOrchestrator(self.llm)
        reasoning = xai.generate_narrative(ticker, model_meta["composite_formula"], fundamental_data, gating_mode)
        
        advice = self.generate_advice(ticker, model_meta, backtest_results, fundamental_data, execution_risk, reasoning)
        
        return {
            "ticker": ticker,
            "model_id": model_id,
            "signals": signals,
            "formula": model_meta["composite_formula"],
            "backtest": backtest_results,
            "fundamental": fundamental_data,
            "risk": execution_risk,
            "reasoning": reasoning,
            "advice": advice,
            "status": "success"
        }

    def generate_advice(self, ticker: str, model_meta: Dict[str, Any], backtest: Dict[str, Any], fundamental: Dict[str, Any] = None, risk: Dict[str, Any] = None, reasoning: str = None) -> str:
        """
        Uses FunctionGemma to provide a technical summary and recommendation.
        """
        fundamental_context = ""
        if fundamental:
            fundamental_context = f"Fundamental Bias: {fundamental['bias']} (Score: {fundamental['score']})"
        
        prompt = f"""
        [INST] <<SYS>>
        You are a Signal Analysis Advisor. 
        Summarize the numerical processing results for {ticker}.
        <</SYS>>

        Object: {ticker}
        Bias Parameter: {fundamental_context}
        Processing Formula: `{model_meta['composite_formula']}`
        Backtest Output: {backtest['total_return']:.2%}
        Efficiency Metric: {backtest['sharpe']:.2f}
        
        Logic Summary: {reasoning}
        
        Task: Based on the array transformations above, state the priority (High/Low/Neutral) for this signal.
        [/INST]"""
        
        try:
            response = self.llm.client.generate(model=self.llm.model, prompt=prompt)
            text = response['response'].strip()
            refusal_keywords = ["sorry", "cannot assist", "cannot fulfill", "apologize", "unable to", "limitation"]
            if any(k in text.lower() for k in refusal_keywords) or len(text) < 20:
                return self._get_fallback_advice(ticker, model_meta, backtest, fundamental)
            return text
        except Exception as e:
            logger.error(f"Advice generation failed: {e}")
            return self._get_fallback_advice(ticker, model_meta, backtest, fundamental)

    def _get_fallback_advice(self, ticker: str, model_meta: dict, backtest: dict, fundamental: dict) -> str:
        """
        Provides a data-driven fallback recommendation.
        """
        bias = fundamental.get("bias", "Neutral")
        ret = backtest.get("total_return", 0)
        
        rec = "NEUTRAL"
        if bias == "Bullish" and ret > 0: rec = "LONG (Strong Alignment)"
        elif bias == "Bearish" and ret > 0: rec = "NEUTRAL (Fundamental Warning)"
        elif bias == "Bullish" and ret < 0: rec = "WAIT (Technical Underperformance)"
        elif bias == "Bearish" and ret < 0: rec = "SHORT (Strong Alignment)"
        
        return f"""
        **Recommendation: {rec}**
        
        The mathematical model for {ticker} shows a 2-year total return of {ret:.2%}. 
        This technical signal is currently cross-validated against a **{bias}** fundamental bias. 
        Execution is optimized via the TCN-Parallel symbolic engine.
        """.strip()
