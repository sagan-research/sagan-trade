from typing import Any, Dict

def format_explain_prompt(prediction: Dict[str, Any]) -> str:
    """Construct an Ollama prompt from prediction output in plain English."""
    
    signal = prediction["signal"]
    confidence = prediction["confidence"]
    override = prediction.get("override", False)
    features = prediction.get("xai_justification", {}).get("selection_weights", {})
    
    # Sort features by importance
    sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:5]
    feature_list = ", ".join([f"{k}" for k, v in sorted_features])

    # Plain English translations for common metrics (if present in metadata/results)
    # Note: These might come from a separate metrics object in a real flow
    sharpe_desc = "Your strategy earned units of return per unit of risk."
    drawdown_desc = "At its worst, this strategy lost % from its peak."
    var_desc = "On a bad day (bottom 5%), you could lose up to %."

    prompt = f"""
Construct an explanation for a trading signal based on the following model output.
Use plain English and avoid financial jargon where possible.

Signal: {signal}
Confidence: {confidence:.1%}
Top Factors: {feature_list}
Override Flag: {override}

Please structure your response into three sections:
1. What this signal means in simple terms.
2. Why the model thinks this — key factors (focusing on {feature_list}).
3. What the user should watch out for — risks and regime flags.
"""

    if override:
        prompt += "\nIMPORTANT: The model flagged low confidence on this signal — a regime change may be occurring. Human review recommended."

    return prompt
