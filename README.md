# Sagan XAI

> **Explainable probabilistic ensemble for mean-reversion trading**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sagan XAI combines three state-of-the-art techniques into a single, production-ready Python library:

| Component | Role |
|---|---|
| **Physics-Informed Neural Networks (PINN)** | Encode Ornstein–Uhlenbeck mean-reversion as a regularisation penalty |
| **Temporal Fusion Transformer (TFT)** | Multi-head self-attention over price return windows |
| **XAI-RL Override** | Flag low-confidence regime changes for human review |

---

## Installation

```bash
pip install sagan-xai
```

Or in editable mode from source:

```bash
git clone https://github.com/sagan-labs/sagan-xai
cd sagan-xai
pip install -e ".[dev]"
```

---

## Quick Start

### Python API

```python
import sagan

# Train a single ensemble across all tickers
model_id = sagan.train(["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])

# Parallel training – one independent model per ticker
results = sagan.train_parallel_from_fetch(
    ["AAPL", "MSFT", "GOOGL"],
    num_processes=8,
)

# Predict using the latest saved model
signal = sagan.predict()
print(signal["signal"])        # "LONG" | "SHORT" | "NEUTRAL"
print(signal["confidence"])    # e.g. 0.74
print(signal["override"])      # True if confidence < threshold
```

### Command-Line Interface

```bash
# Train on Indian equities
sagan --train RELIANCE.NS TCS.NS INFY.NS

# Parallel training
sagan --train AAPL MSFT GOOGL --parallel --num-processes 8

# Get Trading Signal from latest model
sagan --predict

# Use a specific model
sagan --predict --model-id sagan_20240101_120000_abc123

# List all trained models
sagan --list
```

---

## Architecture

```
Input prices (T × N)
       │
       ▼
VariableSelectionNetwork   ← soft feature gating
       │
       ▼
TemporalFusionBlock        ← multi-head self-attention + FFN
       │
       ▼
 ┌─────┴──────┬──────────┐
 │            │          │
Buy head  Sell head  Hold head
 │            │          │
 └─────┬──────┴──────────┘
       │
       ▼
   Softmax ensemble  →  LONG / SHORT / NEUTRAL
       │
       ▼
  XAI-RL override check (confidence < threshold → flag)
```

**Loss function:**

```
L = BCE(y_true, logits) + λ · OU_penalty(logits)
```

where `OU_penalty` penalises deviation from 0.5 probability (mean-reversion prior).

---

## Configuration

All defaults live in `sagan.config`:

```python
from sagan import config

config.default_window = 10          # look-back window (days)
config.default_horizon = 3          # forward horizon for labelling
config.default_epochs = 30
config.pinn_lambda = 0.01           # strength of OU penalty
config.xai_confidence_threshold = 0.6  # override trigger
```

Models are stored in `~/.sagan/xai_models/` by default.

---

## Running Tests

```bash
pytest tests/ -v --cov=sagan
```

---

## API Reference

| Function | Description |
|---|---|
| `sagan.train(tickers, **kwargs)` | Train & save a new ensemble; returns `model_id` |
| `sagan.predict(model_id=None, tickers=None)` | Get trading signal from a saved model |
| `sagan.list_models()` | Return a DataFrame of all registered models |
| `sagan.train_parallel(tickers, prices_dict, ...)` | Parallel training from pre-fetched prices |
| `sagan.train_parallel_from_fetch(tickers, ...)` | Fetch + parallel train in one call |

---

## License

[MIT](LICENSE) © 2024 Sagan Labs
