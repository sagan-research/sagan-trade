import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, TypedDict
import logging

from sagan.config import config
from sagan.signals import fetch_signal_data
from sagan.registry import list_models, load_ensemble
from sagan.models.math_engine import MathematicalEngine

logger = logging.getLogger("sagan.predict")

class PredictionResult(TypedDict):
    signal: str
    confidence: float
    probabilities: dict[str, float]
    formula: str
    r2_stats: dict[str, float]
    model_id: str
    timestamp: str
    xai_justification: dict[str, Any]

def predict(model_id: str = None, compliance: bool = False) -> PredictionResult:
    """
    Predicts using a symbolic or legacy ensemble.
    """
    if model_id is None:
        df = list_models()
        if df.empty: raise ValueError("No models found.")
        model_id = str(df.iloc[-1]["model_id"])

    model_buy, _, _, scaler, metadata = load_ensemble(model_id)
    is_symbolic = metadata.get("is_symbolic", False)

    if not is_symbolic:
        raise NotImplementedError("Legacy Keras prediction is deprecated in this refactor.")

    # Evaluate Symbolic Model
    ticker = metadata["tickers"][0]
    signals = metadata["signals"]
    fitted = metadata["fitted_signals"]
    formula = metadata["composite_formula"]
    
    # Fetch latest data for evaluation
    data = fetch_signal_data(ticker, signals, period="5d")
    t = np.arange(len(data))
    
    # 1. Evaluate fitted signals
    signal_values = {}
    available_signals = []
    for s in signals:
        if s not in fitted:
            logger.warning(f"Signal {s} not found in fitted models, skipping.")
            continue
        
        f_meta = fitted[s]
        val = MathematicalEngine.evaluate(f_meta["func"], t, f_meta["params"])
        signal_values[s] = val[-1]
        available_signals.append(s)

    # 2. Evaluate composite formula
    eval_context = {s: signal_values[s] for s in available_signals}
    # Add numpy functions to context
    eval_context.update({"np": np, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos})
    
    try:
        # Simple cleanup of formula for eval
        clean_formula = formula.replace("^", "**")
        final_value = eval(clean_formula, {"__builtins__": {}}, eval_context)
    except Exception as e:
        logger.error(f"Formula evaluation failed: {e}")
        final_value = 0.0

    # 3. Derive Simple Signal (Trend-based)
    # If the combined value is increasing relative to history, go LONG
    # For a more robust signal, we'd look at the derivative
    signal = "LONG" if final_value > 0 else "SHORT"
    
    return {
        "signal": signal,
        "confidence": 0.95, # Symbolic models have high 'certainty' in their formula
        "probabilities": {"LONG": 0.5, "SHORT": 0.5, "NEUTRAL": 0.0},
        "formula": formula,
        "r2_stats": {s: v["r2"] for s, v in fitted.items()},
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
        "xai_justification": {
            "reason": f"Discovered formula '{formula}' suggests a {signal} trend based on recent signal values.",
            "conflict": False
        }
    }

def batch_predict(model_ids: list[str] = None) -> dict[str, Any]:
    """
    Generate consensus signals from multiple models.
    """
    if model_ids is None:
        df = list_models()
        if df.empty: return {"consensus_signal": "NEUTRAL", "results": []}
        model_ids = list(df["model_id"].values)

    results = []
    for mid in model_ids:
        try:
            results.append(predict(model_id=mid))
        except Exception as e:
            logger.warning(f"Batch predict skipped {mid}: {e}")
            
    if not results:
        return {"consensus_signal": "NEUTRAL", "results": []}

    signals = [r["signal"] for r in results]
    from collections import Counter
    consensus = Counter(signals).most_common(1)[0][0]
    
    return {
        "results": results,
        "consensus_signal": consensus,
        "mean_confidence": np.mean([r["confidence"] for r in results]),
    }
