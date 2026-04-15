"""Prediction interface for trained Sagan XAI ensembles.

This module provides two public functions:

- :func:`predict` — generate a single LONG / SHORT / NEUTRAL signal from the
  most recent price window using one saved ensemble.
- :func:`batch_predict` — compare signals across multiple saved model IDs and
  return a consensus summary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

import numpy as np
import tensorflow as tf

from sagan.config import config
from sagan.data import fetch_prices
from sagan.exceptions import InsufficientDataError, ModelNotFoundError
from sagan.registry import list_models, load_ensemble
from sagan.indicators import compute_technical_snapshot
from sagan.database import log_action
from sagan.compliance.report import generate_compliance_report

__all__ = ["predict", "batch_predict", "PredictionResult"]


class PredictionResult(TypedDict):
    """Typed structure returned by :func:`predict`.

    Attributes:
        signal: Trading signal — ``"LONG"``, ``"SHORT"``, or ``"NEUTRAL"``.
        confidence: Softmax probability of the winning action class (0–1).
        probabilities: Dict mapping each action label to its softmax probability.
        regime_uncertainty: ``1 - confidence``, useful as a risk indicator.
        override: ``True`` when confidence is below
            :attr:`~sagan.config.SaganConfig.xai_confidence_threshold`.
        xai_justification: Dict with ``reason`` string and
            ``confidence_threshold`` value.
        model_id: The model identifier used to generate this prediction.
        tickers: Ticker symbols the model was constructed for.
        timestamp: ISO 8601 timestamp of the prediction.
    """

    signal: str
    confidence: float
    probabilities: dict[str, float]
    portfolio_weights: dict[str, float]
    regime_uncertainty: float
    override: bool
    xai_justification: dict[str, Any]
    model_id: str
    tickers: list[str]
    timestamp: str


def predict(
    model_id: str | None = None,
    tickers: list[str] | None = None,
    days_back: int = 20,
    compliance: bool = False,
) -> PredictionResult:
    """Generate a trading signal from a trained ensemble.

    Loads the specified model (or the most recently registered one), fetches
    the latest price data, applies the same scaling used during training, and
    returns a structured prediction.

    Args:
        model_id: ID of the model to use. If ``None``, the most recently
            registered model is selected automatically.
        tickers: Override the tickers stored in model metadata. By default
            the tickers from training are used.
        days_back: Unused; kept for API stability. The last ``window`` trading
            days of returns are always used.

    Returns:
        A :class:`PredictionResult` TypedDict with signal, confidence,
        probabilities, override flag, and XAI justification.

    Raises:
        ModelNotFoundError: If *model_id* is specified but not found in the
            registry.
        ValueError: If no models have been trained yet.
        InsufficientDataError: If fewer than *window* days of data are
            available for the requested tickers.

    Example:
        >>> import sagan
        >>> model_id = sagan.train(["AAPL", "MSFT"])
        >>> result = sagan.predict(model_id=model_id)
        >>> print(result["signal"], f"{result['confidence']:.1%}")
    """
    if model_id is None:
        df = list_models()
        if df.empty:
            raise ValueError("No trained model found. Run `sagan.train()` first.")
        model_id = str(df.iloc[-1]["model_id"])

    model_buy, model_sell, model_hold, scaler, metadata = load_ensemble(model_id)
    window: int = int(metadata["window"])
    model_tickers: list[str] = metadata["tickers"]
    if tickers is None:
        tickers = model_tickers

    prices = fetch_prices(tickers, years=1)
    returns = prices.pct_change().dropna()
    if len(returns) < window:
        raise InsufficientDataError(available=len(returns), required=window)

    n_stocks = returns.shape[1]
    last_window = returns.iloc[-window:].values
    scaled = scaler.transform(last_window.reshape(-1, n_stocks))
    X_input = tf.convert_to_tensor(scaled.reshape(1, window, n_stocks), dtype=tf.float32)

    preds_buy = model_buy(X_input, training=False)
    preds_sell = model_sell(X_input, training=False)
    preds_hold = model_hold(X_input, training=False)

    # Logits and Softmax
    logit_buy = float(preds_buy["logit"][0][0])
    logit_sell = float(preds_sell["logit"][0][0])
    logit_hold = float(preds_hold["logit"][0][0])

    logits = np.array([logit_buy, logit_sell, logit_hold])
    probs = tf.nn.softmax(logits).numpy()
    max_prob = float(np.max(probs))
    signal_map = {0: "LONG", 1: "SHORT", 2: "NEUTRAL"}
    signal_idx = int(np.argmax(probs))
    signal = signal_map[signal_idx]
    override = max_prob < config.xai_confidence_threshold

    # Portfolio Weights from Selection Network
    winning_preds = [preds_buy, preds_sell, preds_hold][signal_idx]
    vsn_weights = winning_preds["selection_weights"][0].numpy().tolist()

    direction = 1.0 if signal == "LONG" else (-1.0 if signal == "SHORT" else 0.0)
    port_weights = {
        tickers[i]: float(vsn_weights[i] * max_prob * direction)
        for i in range(len(tickers))
    }

    # Technical Indicators and Thresholds
    tech_snapshot = compute_technical_snapshot(prices)
    ticker_thresholds = metadata.get("ticker_thresholds", {})
    
    # Simple Rule-based Signal Generation from Thresholds
    # We aggregate across tickers; here we just look at the primary ticker or mean
    rule_signals = []
    for t in tickers:
        t_meta = ticker_thresholds.get(t, {"rsi_buy": 35, "rsi_sell": 65})
        t_tech = tech_snapshot.get(t, {})
        
        if t_tech.get("rsi", 50) < t_meta["rsi_buy"]:
            rule_signals.append("LONG")
        elif t_tech.get("rsi", 50) > t_meta["rsi_sell"]:
            rule_signals.append("SHORT")
        else:
            rule_signals.append("NEUTRAL")
            
    from collections import Counter
    rule_signal = Counter(rule_signals).most_common(1)[0][0]
    
    # Conflict Detection
    conflict = (signal != rule_signal)
    
    # Gating Logic: Use VSN selection weights to weight the ML signal
    # If the model is focused on a specific ticker, trust the ML more.
    # Otherwise, if it's dispersed, the conflict matters more.
    vsn_dispersion = np.std(vsn_weights)  
    gating_reason = "ML Focused" if vsn_dispersion > 0.1 else "Diffused - Trusting Thresholds more"
    
    if conflict:
        res_signal = "NEUTRAL (Conflict)" if not override else signal
        just_reason = f"Conflict detected: ML={signal}, Rule={rule_signal}. Gating: {gating_reason}"
    else:
        res_signal = signal
        just_reason = f"Signals align: {signal}. Gating: {gating_reason}"

    res = PredictionResult(
        signal=res_signal,
        confidence=max_prob,
        probabilities={
            "LONG": float(probs[0]),
            "SHORT": float(probs[1]),
            "NEUTRAL": float(probs[2]),
        },
        portfolio_weights=port_weights,
        regime_uncertainty=1.0 - max_prob,
        override=bool(override),
        xai_justification={
            "reason": just_reason,
            "confidence_threshold": config.xai_confidence_threshold,
            "selection_weights": {tickers[i]: float(vsn_weights[i]) for i in range(len(tickers))},
            "technical_indicators": tech_snapshot,
            "thresholds": ticker_thresholds,
            "conflict": conflict
        },
        model_id=model_id,
        tickers=tickers,
        timestamp=datetime.now().isoformat(),
    )

    # Log to local DB
    log_action(
        model_id=model_id,
        tickers=tickers,
        action=res_signal,
        confidence=max_prob,
        conflict=conflict,
        justification=just_reason,
        extra_meta={"tech": tech_snapshot, "probs": res["probabilities"]}
    )

    if compliance:
        generate_compliance_report(res)

    return res


def batch_predict(
    model_ids: list[str] | None = None,
    tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Generate signals from multiple models and return a consensus summary.

    Useful for comparing different trained models or running an ensemble-of-
    ensembles strategy.

    Args:
        model_ids: List of model IDs to query. If ``None``, **all** registered
            models are used.
        tickers: Override tickers for all models. Each model still uses its own
            scaler and window.

    Returns:
        A dict with keys:

        - ``"results"`` – ``list[PredictionResult]`` one entry per model.
        - ``"consensus_signal"`` – majority-vote signal.
        - ``"mean_confidence"`` – average confidence across all models.
        - ``"agreement_rate"`` – fraction of models that agree with the
          consensus signal.
        - ``"override_count"`` – number of models that triggered the XAI
          override.

    Example:
        >>> import sagan
        >>> summary = sagan.batch_predict()
        >>> summary["consensus_signal"]
        'LONG'
        >>> summary["agreement_rate"]
        0.75
    """
    if model_ids is None:
        df = list_models()
        if df.empty:
            raise ValueError("No trained models found.")
        model_ids = list(df["model_id"].values)

    results: list[PredictionResult] = []
    for mid in model_ids:
        try:
            results.append(predict(model_id=mid, tickers=tickers))
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger("sagan").warning("Skipping model %s: %s", mid, exc)

    if not results:
        raise ValueError("No valid predictions could be generated.")

    signals = [r["signal"] for r in results]
    from collections import Counter
    consensus = Counter(signals).most_common(1)[0][0]

    return {
        "results": results,
        "consensus_signal": consensus,
        "mean_confidence": float(np.mean([r["confidence"] for r in results])),
        "agreement_rate": float(signals.count(consensus) / len(signals)),
        "override_count": int(sum(1 for r in results if r["override"])),
    }
