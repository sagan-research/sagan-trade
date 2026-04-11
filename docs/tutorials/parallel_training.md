# Tutorial: Parallel Training

This tutorial shows how to train **one independent model per ticker**
simultaneously using Python's multiprocessing, then run a consensus signal
across the resulting portfolio.

---

## When to use parallel training

| Use case | Recommended mode |
|---|---|
| Single basket signal | `sagan.train(["A", "B", "C"])` |
| Per-ticker alpha hunting | `sagan.train_parallel_from_fetch(...)` |
| Already have price DataFrames | `sagan.train_parallel(tickers, prices_dict)` |

---

## Option A — Fetch + train in one call

The easiest path: Sagan fetches prices for each ticker, then dispatches one
training worker per ticker.

```python
import sagan
from sagan.logging_config import setup_logging

setup_logging("INFO")

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
           "NVDA", "TSLA", "JPM", "V", "JNJ"]

results = sagan.train_parallel_from_fetch(
    tickers,
    years=5,
    num_processes=8,   # adjust to your CPU core count
    window=10,
    horizon=3,
    epochs=25,
)

# results is a dict {ticker: model_id or None}
for ticker, mid in results.items():
    if mid:
        print(f"✅ {ticker:10s} → {mid}")
    else:
        print(f"❌ {ticker:10s} → FAILED")
```

!!! warning "Windows multiprocessing"
    On Windows, parallel training **must** be inside an `if __name__ == "__main__":` guard:

    ```python
    if __name__ == "__main__":
        results = sagan.train_parallel_from_fetch(tickers, num_processes=4)
    ```

---

## Option B — Pre-fetch, then train

Useful when you already have price data or want to apply custom cleaning.

```python
from sagan.data import fetch_prices
from sagan.parallel import train_parallel

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# Step 1: fetch (and optionally process)
prices_dict = {}
for t in tickers:
    prices_dict[t] = fetch_prices([t], years=5)

# Step 2: dispatch training
config_params = {
    "window": 15,
    "horizon": 5,
    "threshold": 0.015,
    "epochs": 30,
}

if __name__ == "__main__":
    results = train_parallel(
        tickers, prices_dict,
        config_params=config_params,
        num_processes=3,
    )
```

---

## Reviewing results

```python
import sagan

df = sagan.list_models()
# Filter to models trained today
today = df["created_at"].str[:10] == "2024-04-11"
print(df[today][["model_id", "tickers", "val_sharpe"]].to_string())
```

---

## Portfolio consensus signal

After parallel training you can query all models at once and get a
majority-vote consensus:

```python
import sagan

model_ids = list(results.values())
model_ids = [m for m in model_ids if m]   # drop failed

summary = sagan.batch_predict(model_ids=model_ids)

print(f"Consensus:       {summary['consensus_signal']}")
print(f"Agreement:       {summary['agreement_rate']:.0%}")
print(f"Mean confidence: {summary['mean_confidence']:.1%}")
print(f"Override count:  {summary['override_count']}/{len(model_ids)}")
```

---

## Tips for large universes

- Set `num_processes` to the number of **physical** CPU cores (not logical).
- If fetching >50 tickers, consider batching in groups of 20 to avoid
  Yahoo Finance rate-limits.
- Use `sagan.data.validate_tickers(tickers)` to pre-filter bad symbols before
  starting a long parallel run.
