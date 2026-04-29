import logging
from typing import Dict, Any, List
from sagan.models.llm_bridge import FunctionGemmaBridge

logger = logging.getLogger("sagan.explain")

class XAIOrchestrator:
    """
    Orchestrates explainability for the Sagan trading engine.
    Translates mathematical formulas and fundamental biases into human-readable narratives.
    """
    def __init__(self, bridge: FunctionGemmaBridge = None):
        self.llm = bridge or FunctionGemmaBridge()

    def generate_narrative(self, ticker: str, formula: str, fundamental_data: Dict[str, Any], gating_mode: str) -> str:
        """
        Generates a comprehensive "How and Why" narrative for a specific decision.
        """
        
        # 1. Formula Decomposition
        formula_meaning = self._interpret_formula(formula)
        
        # 2. Fundamental Justification
        fundamental_bias = fundamental_data.get("bias", "Neutral")
        fundamental_reason = fundamental_data.get("drivers", ["No significant fundamental drivers detected."])
        
        # 3. LLM Synthesis
        prompt = f"""
        [INST] <<SYS>>
        You are a Numerical Analysis Engine. 
        Explain the mathematical structure of a signal transformation.
        <</SYS>>

        Object: {ticker}
        Signal Formula: `{formula}`
        Component Interpretation: {formula_meaning}
        
        Bias Parameter: {fundamental_bias}
        Input Drivers: {", ".join(fundamental_reason)}
        
        Task: Describe how the input drivers ({fundamental_bias}) and the mathematical components of the formula interact to transform the input signals. 
        Note: The formula was discovered using a parallel TCN (Temporal Convolutional Network) search.
        [/INST]"""
        
        try:
            response = self.llm.client.generate(model=self.llm.model, prompt=prompt)
            text = response['response'].strip()
            refusal_keywords = ["sorry", "cannot assist", "cannot fulfill", "apologize", "unable to", "limitation"]
            if any(k in text.lower() for k in refusal_keywords) or len(text) < 20:
                return self._get_fallback_narrative(ticker, formula, formula_meaning, fundamental_bias, fundamental_reason)
            return text
        except Exception as e:
            logger.error(f"XAI Narrative generation failed: {e}")
            return self._get_fallback_narrative(ticker, formula, formula_meaning, fundamental_bias, fundamental_reason)

    def _get_fallback_narrative(self, ticker: str, formula: str, meaning: str, bias: str, drivers: List[str]) -> str:
        """
        Provides a structured fallback explanation when the LLM refuses.
        """
        return f"""
        ### Decision Narrative for {ticker}
        
        **Technical Foundation (The HOW):**
        The signal was discovered using a parallelized Temporal Convolutional Network (TCN) symbolic search. 
        The resulting formula is `{formula}`, which incorporates:
        - {meaning}
        
        **Fundamental Justification (The WHY):**
        The current fundamental bias is **{bias}**. 
        Key identified drivers include: {", ".join(drivers)}.
        
        **Methodology Note:**
        This strategy represents the optimal mathematical fit for {ticker} price action, cross-validated against recent fundamental health scores.
        """.strip()

    def _interpret_formula(self, formula: str) -> str:
        """
        Briefly interprets the mathematical components of a formula.
        """
        interpretation = []
        if "np.log" in formula:
            interpretation.append("Logarithmic volatility damping (sensitivity reduction for large moves)")
        if "np.sin" in formula or "np.cos" in formula:
            interpretation.append("Cyclical/Harmonic pattern recognition (detecting market seasonality)")
        if "np.exp" in formula:
            interpretation.append("Exponential weighting (prioritizing recent surge/momentum)")
        if "** 2" in formula:
            interpretation.append("Quadratic acceleration (sensitive to rapid price changes)")
        if "*" in formula and "Volume" in formula:
            interpretation.append("Volume-price confirmation logic")
            
        return " | ".join(interpretation) if interpretation else "Standard linear/non-linear relationship."

def get_explanation(results: Dict[str, Any]) -> str:
    """
    Helper function to get explanation from autonomous pipeline results.
    """
    orchestrator = XAIOrchestrator()
    ticker = results.get("ticker", "Unknown")
    formula = results.get("backtest", {}).get("formula", "N/A")
    fundamental_data = results.get("fundamentals", {})
    gating_mode = results.get("gating_mode", "balanced")
    
    return orchestrator.generate_narrative(ticker, formula, fundamental_data, gating_mode)
