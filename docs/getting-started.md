# Getting Started

This guide walks you through installing Sagan XAI, training your first model,
and generating your first trading signal in under five minutes.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| TensorFlow | ≥ 2.10 |
| pip | ≥ 22 |

!!! note "GPU support"
    Sagan works on CPU out of the box. If you have an NVIDIA GPU, install a
    CUDA-enabled TensorFlow build **before** installing Sagan:

    ```bash
    pip install tensorflow[and-cuda]>=2.10
    pip install sagan
    ```

---

## Installation

```bash
pip install sagan
```

Verify the installation:

```python
import sagan
print(sagan.__version__)   # e.g. "0.1.0"
```

---

## Step 1 — Train a model

::: sagan.train
    options:
      show_source: false
      heading_level: 4

```python
import sagan

model_id = sagan.train(
    tickers=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
    years=5,        # 5 years of history
    window=10,      # 10-day look-back window
    horizon=3,      # 3-day forward return for labelling
    epochs=30,      # max training epochs
)
print(f"Model saved: {model_id}")
```

Training logs will be printed to the console. Early stopping is applied
automatically (patience = 5 epochs). Expect ~2–5 minutes on CPU for 5 years
of 3 tickers.

---

## Step 2 — Inspect trained models

```python
df = sagan.list_models()
print(df[["model_id", "tickers", "val_sharpe", "created_at"]])
```

---

## Step 3 — Generate a signal

```python
result = sagan.predict()

print(result["signal"])            # "LONG", "SHORT", or "NEUTRAL"
print(f"{result['confidence']:.1%}")   # e.g. "74.3%"
print(result["override"])          # True if confidence < threshold
print(result["regime_uncertainty"])
```

The returned :class:`~sagan.predict.PredictionResult` dict contains:

| Key | Type | Description |
|---|---|---|
| `signal` | `str` | `"LONG"` / `"SHORT"` / `"NEUTRAL"` |
| `confidence` | `float` | Max softmax probability (0–1) |
| `probabilities` | `dict` | Per-action probabilities |
| `regime_uncertainty` | `float` | `1 - confidence` |
| `override` | `bool` | `True` when `confidence < xai_confidence_threshold` |
| `xai_justification` | `dict` | Reason and threshold |
| `model_id` | `str` | Which model produced this result |
| `tickers` | `list[str]` | Tickers used for prediction |
| `timestamp` | `str` | ISO 8601 generation time |

---

## Step 4 — Use the CLI

```bash
# Train
sagan --train RELIANCE.NS TCS.NS --epochs 30 --window 10

# Predict using latest model
sagan --predict

# Predict using a specific model
sagan --predict --model-id sagan_20240411_120000_abc123

# List all models
sagan --list
```

---

## What's next?

<div class="grid cards" markdown>

-   :material-book-open-variant: **Tutorials**

    ---

    [Single Stock](tutorials/single_stock.md) · [Parallel Training](tutorials/parallel_training.md) · [Custom Config](tutorials/custom_config.md)

-   :material-cog: **Architecture**

    ---

    Deep dive into the PINN loss, TFT blocks, and XAI-RL override logic.
    [Architecture →](architecture.md)

-   :material-code-tags: **API Reference**

    ---

    Every public function and class, auto-generated from source.
    [API →](api/index.md)

</div>
