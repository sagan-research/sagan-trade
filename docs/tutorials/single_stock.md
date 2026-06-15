# Tutorial: Single Stock

This tutorial walks you through training **one ensemble** on a single ticker
end-to-end, from data fetch to interpreting the prediction.

---

## Setup

```python
import sagan
from sagan.logging_config import setup_logging

# Enable console logs so you can follow training progress
setup_logging(level="INFO")
```

---

## 1. Validate your ticker

Before starting a long training run, check that your ticker resolves:

```python
from sagan.data import validate_tickers

valid = validate_tickers(["RELIANCE.NS", "BAD_TICKER"])
print(valid)   # ['RELIANCE.NS']
```

---

## 2. Peek at the data

```python
from sagan.data import fetch_prices, prepare_probabilistic_data

prices = fetch_prices(["RELIANCE.NS"], years=5)
print(prices.tail())
print(f"Shape: {prices.shape}")   # (1258, 1) approximately

X, y_probs, y_ret, symbols, n_stocks = prepare_probabilistic_data(
    prices, window=10, horizon=3, threshold=0.01
)
print(f"Samples: {len(X)}, window={X.shape[1]}, n_stocks={n_stocks}")
print(f"BUY labels:  {(y_probs[:, 0] == 1).mean():.1%}")
print(f"SELL labels: {(y_probs[:, 1] == 1).mean():.1%}")
print(f"HOLD labels: {(y_probs[:, 2] == 1).mean():.1%}")
```

---

## 3. Train the ensemble

```python
model_id = sagan.train(
    tickers=["RELIANCE.NS"],
    years=5,
    window=10,
    horizon=3,
    threshold=0.01,
    head_dim=32,
    num_heads=4,
    ff_dim=64,
    dropout=0.1,
    epochs=30,
    verbose=True,
)
print(f"Saved: {model_id}")
```

!!! tip "Estimated time"
    About 2–4 minutes on CPU for 5 years of daily data with 30 epochs.

---

## 4. Inspect the result

```python
import sagan

df = sagan.list_models()
print(df[["model_id", "tickers", "val_sharpe", "override_fraction"]].tail(3))
```

A `val_sharpe > 0.3` on a single stock is a reasonable signal that the model
has learned a meaningful pattern.

---

## 5. Generate a trading signal

```python
result = sagan.predict(model_id=model_id)

print("=" * 40)
print(f"Signal:      {result['signal']}")
print(f"Confidence:  {result['confidence']:.1%}")
print(f"Override:    {result['override']}")
print(f"Probs:       LONG={result['probabilities']['LONG']:.2f}  "
      f"SHORT={result['probabilities']['SHORT']:.2f}  "
      f"NEUTRAL={result['probabilities']['NEUTRAL']:.2f}")
print(f"Uncertainty: {result['regime_uncertainty']:.3f}")
print(f"Reason:      {result['xai_justification']['reason']}")
```

---

## 6. Interpret the override flag

| Condition | Meaning | Suggested action |
|---|---|---|
| `override=False`, `signal="LONG"` | High-confidence buy | Full position |
| `override=False`, `signal="SHORT"` | High-confidence sell | Short or reduce |
| `override=True`, any signal | Low confidence | Halve size or wait |
| `signal="NEUTRAL"` | No edge detected | Stay flat |

---

## 7. Analyse strategy metrics

```python
from sagan.utils import sharpe_ratio, max_drawdown, win_rate
import numpy as np

# Hypothetical: you ran the model daily and collected these returns
# (replace with your actual backtest returns)
strategy_returns = np.random.normal(0.0005, 0.01, 252)

print(f"Sharpe:   {sharpe_ratio(strategy_returns):.2f}")
print(f"Max DD:   {max_drawdown(strategy_returns):.2%}")
print(f"Win rate: {win_rate(strategy_returns):.1%}")
```
