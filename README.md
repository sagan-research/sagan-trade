# Sagan Trade

> **High-throughput symbolic mathematical trading engine**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/sagan-trade.svg)](https://pypi.org/project/sagan-trade/)

Sagan Trade replaces black-box neural networks with transparent, human-readable mathematical equations discovered via **FunctionGemma** (via Ollama). 

| Component | Role |
|---|---|
| **Symbolic Regressor** | Fits variables to R2 > 0.95 using Polynomial and Fourier basis functions. |
| **FunctionGemma** | AI architect that suggests optimal mathematical compositions of signals. |
| **Power Hub** | OS-level optimization for maximum throughput (Eco, Balanced, Turbo). |

---

## Installation

```bash
pip install sagan-trade
```

Or in editable mode from source:

```bash
git clone https://github.com/That-Tech-Geek/sagan-trade
cd sagan-trade
pip install -e ".[dev]"
```

---

## 📊 Performance & Research

The **SymbolicBasis** framework achieves high-fidelity trend fitting by decomposing signals into Polynomial and Fourier series. 

| Asset | Discovered Function | Fidelity ($R^2$) |
|---|---|---|
| **AAPL** | Polynomial (Deg 4) | **0.9337** |
| **MSFT** | Polynomial (Deg 5) | **0.9370** |
| **GOOGL** | Polynomial (Deg 3) | **0.9617** |
| **NVDA** | Polynomial (Deg 3) | **0.9211** |

> [!NOTE]
> Higher $R^2$ values indicate greater trend stability, allowing the **PortfolioAllocator** to prioritize assets with minimal mathematical uncertainty.

---

## Quick Start

### Python API

```python
import sagan

# Train a symbolic ensemble with high-accuracy math fitting
model_id = sagan.train(
    ["AAPL"], 
    signals=["Close", "Volume", "RSI"], 
    target_r2=0.95,
    profile="turbo"
)

# Predict using the latest symbolic expression
result = sagan.predict()
print(result["signal"])     # "LONG" | "SHORT"
print(result["formula"])    # e.g. "(Close * 0.5) + log(Volume)"
```

### Command-Line Interface

```bash
# List available math signals for a ticker
sagan vars AAPL

# Train symbolic model
sagan train AAPL --signals Close,Volume --r2 0.95 --profile turbo

# Get Trading Signal
sagan predict
```

---

## Architecture

```
yfinance Data
       │
       ▼
[Parallel Fitting] → Each variable fitted to R2 > 0.95
       │
       ▼
[FunctionGemma]   → Suggests composite math formula
       │
       ▼
[Evaluation]      → Trend-based signal generation
```

---

## Configuration

All defaults live in `sagan.config`:

```python
from sagan import config

config.models_dir = "~/.sagan/models/"
```

---

## License

[MIT](LICENSE) © 2024 Sagan Labs
