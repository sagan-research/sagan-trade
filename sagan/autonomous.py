import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from sagan.models.llm_bridge import FunctionGemmaBridge
from sagan.ensemble import SymbolicRegressor
from sagan.research import BacktestEngine
from sagan.signals import fetch_signal_data

logger = logging.getLogger("sagan.autonomous")

class AutonomousResearcher:
    """
    Orchestrates the autonomous alpha discovery loop:
    Discovery -> Optimization -> Backtest -> Advice.
    """
    def __init__(self, bridge: FunctionGemmaBridge = None):
        self.llm = bridge or FunctionGemmaBridge()

    def run_full_pipeline(self, ticker: str, period: str = "2y") -> Dict[str, Any]:
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
        logger.info(f"[2/4] Optimizing symbolic models using: {signals}")
        regressor = SymbolicRegressor([ticker], signals=signals, period=period, profile="balanced")
        model_meta = regressor.train()
        model_id = regressor.save()
        
        # 3. Backtest Phase
        logger.info(f"[3/4] Validating strategy via backtest...")
        engine = BacktestEngine(ticker, model_meta["composite_formula"], period=period)
        backtest_results = engine.run()
        
        # 4. Advice Phase
        logger.info("[4/4] Generating positioning advice...")
        advice = self.generate_advice(ticker, model_meta, backtest_results)
        
        return {
            "ticker": ticker,
            "model_id": model_id,
            "signals": signals,
            "formula": model_meta["composite_formula"],
            "backtest": backtest_results,
            "advice": advice,
            "status": "success"
        }

    def generate_advice(self, ticker: str, model_meta: Dict[str, Any], backtest: Dict[str, Any]) -> str:
        """
        Uses FunctionGemma to provide a human-readable summary and recommendation.
        """
        prompt = f"""
        [INST] <<SYS>>
        You are a senior quantitative investment advisor. 
        Provide a concise, professional summary and positioning advice based on the symbolic research.
        Keep it under 150 words.
        <</SYS>>

        Ticker: {ticker}
        Discovered Formula: `{model_meta['composite_formula']}`
        Backtest Results:
        - Total Return: {backtest['total_return']:.2%}
        - Sharpe: {backtest['sharpe']:.2f}
        - Max Drawdown: {backtest['max_drawdown']:.2%}
        
        Variables used: {', '.join(model_meta['signals'])}

        Positioning Advice: [/INST]"""
        
        try:
            response = self.llm.client.generate(model=self.llm.model, prompt=prompt)
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Advice generation failed: {e}")
            return "Positioning: Data inconclusive. Maintain neutral exposure."
