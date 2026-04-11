# Sagan XAI

**Explainable probabilistic ensemble for mean-reversion trading.**

<div class="grid cards" markdown>

-   :material-chart-line: **Physics-Informed**

    ---

    Ornstein–Uhlenbeck mean-reversion penalty baked directly into the loss
    function — no post-hoc corrections needed.

-   :material-brain: **Temporal Fusion Transformer**

    ---

    Multi-head self-attention with learned variable-selection gating operates
    over sliding return windows for rich temporal context.

-   :material-shield-check: **XAI-RL Override**

    ---

    Low-confidence predictions are automatically flagged for human review,
    giving you a built-in safety net in volatile regimes.

-   :material-lightning-bolt: **Parallel Training**

    ---

    Train independent models for hundreds of tickers simultaneously using
    Python's multiprocessing — one core, one stock.

</div>

---

## Quick Start

```bash
pip install sagan
```

```python
import sagan

# 1 ─ Train
model_id = sagan.train(["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])

# 2 ─ Predict
result = sagan.predict()
print(result["signal"])       # "LONG" | "SHORT" | "NEUTRAL"
print(result["confidence"])   # e.g. 0.74
print(result["override"])     # True if regime is uncertain

# 3 ─ Parallel training across many stocks
ids = sagan.train_parallel_from_fetch(["AAPL", "MSFT", "GOOGL"], num_processes=8)
```

```bash
# CLI
sagan --train RELIANCE.NS TCS.NS
sagan --predict
sagan --list
```

---

## How It Works

```
Raw prices (T × N tickers)
         │
         ▼  pct_change + sliding window
     Dataset (samples × window × n_stocks)
         │
         ▼  VariableSelectionNetwork
   Soft feature-gated inputs
         │
         ▼  TemporalFusionBlock  (multi-head attention + FFN)
   Context-aware representations
         │
   ┌─────┴──────┬──────────┐
   │            │          │
Buy head    Sell head  Hold head
   │            │          │
   └─────┬──────┴──────────┘
         │  softmax ensemble
         ▼
   LONG / SHORT / NEUTRAL  +  confidence
         │
         ▼  XAI-RL override check
   confidence < θ → override = True (flag for review)
```

**Loss function:**

$$
\mathcal{L} = \underbrace{\text{BCE}(y, \hat{y})}_{\text{classification}} + \lambda \cdot \underbrace{\mathbb{E}[\,(p - 0.5)^2\,]}_{\text{OU mean-reversion penalty}}
$$

The penalty term $\lambda \cdot \mathbb{E}[\,(p - 0.5)^2]$ pulls predicted
probabilities toward 0.5, encoding the prior that financial returns are
mean-reverting. See [Architecture](architecture.md) for a full derivation.

---

## Features at a Glance

| Feature | Details |
|---|---|
| Model architecture | Temporal Fusion Transformer (TFT) |
| Loss | BCE + Ornstein–Uhlenbeck PINN penalty |
| Signals | LONG · SHORT · NEUTRAL with calibrated probabilities |
| Explainability | XAI-RL override flag + regime uncertainty score |
| Training modes | Single ensemble · Parallel (one model per ticker) |
| Data source | Yahoo Finance via `yfinance` (auto-retry, back-off) |
| Storage | Local filesystem registry (`~/.sagan/xai_models/`) |
| CLI | `sagan --train`, `--predict`, `--list`, `--parallel` |
| Python support | 3.9 · 3.10 · 3.11 · 3.12 |
| Type safety | PEP 561 `py.typed`, fully annotated |

---

## Installation

=== "pip (stable)"

    ```bash
    pip install sagan
    ```

=== "pip (latest from source)"

    ```bash
    pip install git+https://github.com/sagan-labs/sagan.git
    ```

=== "editable (developer)"

    ```bash
    git clone https://github.com/sagan-labs/sagan
    cd sagan
    pip install -e ".[dev]"
    ```

---

!!! tip "Next steps"
    - Follow the [Getting Started](getting-started.md) guide for a full walkthrough.
    - Explore [Tutorials](tutorials/single_stock.md) for end-to-end examples.
    - Browse the [API Reference](api/index.md) for every public function and class.
