---
name: sagan-native
description: >
  Equips a coding agent with the specialized syntax and architectural knowledge 
  required to develop, extend, and optimize the Sagan Trading Engine.
---

# Sagan Native Developer Skill

You are a **Sagan Native Developer**. This skill provides the definitive syntax, mathematical patterns, and architectural blueprints for the Sagan Trading Engine (SymbolicBasis™). Adhering to these standards ensures every line of code is performant, explainable, and mathematically sound.

---

## 🏛️ Model Reference

### 1. `TCNSymbolicFitter` (Production SOTA)
The primary engine for mapping sequences to mathematical coefficients.
- **Architecture**: 4-layer Temporal Convolutional Network.
- **Causality**: Uses `CausalConv1d` to prevent look-ahead bias.
- **Receptive Field**: Exponential dilations (1, 2, 4, 8) capture multi-scale dependencies.
- **Output**: 3 polynomial coefficients + $3 \times N$ Fourier parameters (Amplitude, Phase, Frequency).

### 2. `MathematicalEngine` (The Basis)
The logic layer that defines the "Universe of Discovery".
- **Polynomial Basis**: $y = a_0 + a_1 t + a_2 t^2$
- **Fourier Basis**: $y = \sum A_i \cos(w_i t) + B_i \sin(w_i t)$
- **Evaluation**: JIT-compiled via `numba` for near-C speeds.

### 3. `AutonomousResearcher` (The Orchestrator)
The high-level pipeline that bridges LLMs and Math.
- **Pipeline**: Discovery (LLM) -> Ingestion -> Optimization (TCN) -> Gating (Fundamental) -> Advice (LLM).
- **Quantamental Logic**: Cross-validates technical R2 scores with fundamental bias metrics.

### 4. Configuration & Hyper-parameters
Standard configurations stored in `sagan/config.py`:
- **`default_window`**: 10 steps (history length for TCN).
- **`pinn_lambda`**: 0.01 (Physics-Informed Loss weighting).
- **`xai_confidence_threshold`**: 0.6 (Threshold for "High Confidence" narratives).

---

## 💻 Syntax & API Reference

### 1. Training a Single-Ticker Model
```python
from sagan.ensemble import SymbolicRegressor

# Initialize with performance profile
reg = SymbolicRegressor(
    tickers=["AAPL"], 
    signals=["Adj Close", "Volume", "RSI", "^VIX"], 
    profile="turbo" # eco, balanced, turbo
)

# Execute parallel symbolic fitting
meta = reg.train()
model_id = reg.save()
```

### 2. Multi-Ticker Portfolio Optimization
```python
from sagan.ensemble import PortfolioSymbolicEngine

tickers = ["AAPL", "MSFT", "TSLA", "BTC-USD"]
engine = PortfolioSymbolicEngine(tickers, profile="balanced")

# Parallel training across all tickers
results = engine.train_all()
model_ids = engine.save_all()
```

### 3. Autonomous Alpha Pipeline
```python
from sagan.autonomous import AutonomousResearcher

researcher = AutonomousResearcher()
res = researcher.run_full_pipeline("NVDA", gating_mode="balanced")
```

### 4. Low-Level Symbolic Evaluation
```python
from sagan.models.math_engine import MathematicalEngine
import numpy as np

engine = MathematicalEngine()
data = {"Close": np.array([150, 155, 152]), "RSI": np.array([60, 65, 62])}

# Vectorized evaluation
result = engine.evaluate_formula("(Close / RSI) * 100", data)
```

---

## 🛠️ Native Coding Patterns

### 1. Vectorized Performance
Sagan handles massive ticker baskets. Always use vectorized NumPy operations. Avoid Python loops in the data path. Use `@numba.jit` for custom kernels.

### 2. Symbolic Representation
Never implement a "hard-coded" heuristic if it can be represented as a symbolic formula.
- **Wrong**: `if rsi > 70: sell()`
- **Native**: `formula = "np.where(RSI > 70, -1, 0)"`

### 3. XAI Integration
A feature is not "Sagan Native" unless it is explainable.
- Use `explain_formula` to split complex strings into additive components.
- Use `XAIOrchestrator` to bridge math to human-readable narratives.

### 4. Karpathy-Style Aesthetics
Maintain the high-fidelity aesthetic. Use premium typography in UI components, rigorous LaTeX in documentation, and clean, modular Python in the backend.

---

## 📋 Error Handling & Robustness
- **Data Alignment**: Always perform an `inner-join` reindexing when combining macro indicators (like ^VIX) with assets.
- **Formula Safety**: Use `eval` only within a restricted context (NumPy/Pandas only).
- **Throughput**: Use `ProcessPoolExecutor` for CPU-bound symbolic fitting tasks.
