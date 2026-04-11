# Tutorial: Custom Configuration

Sagan's :class:`~sagan.config.SaganConfig` gives you fine-grained control
over every model parameter, storage path, and XAI threshold — without
modifying library source code.

---

## Method 1 — Mutate the singleton

The simplest approach: change fields on the global `config` object before
calling `train()`.

```python
import sagan

# Increase regularisation strength
sagan.config.pinn_lambda = 0.05

# Lower confidence threshold (more predictions get flagged as uncertain)
sagan.config.xai_confidence_threshold = 0.5

# Use a larger TFT
sagan.config.default_head_dim = 64
sagan.config.default_num_heads = 8
sagan.config.default_ff_dim = 128

# Train with updated defaults
model_id = sagan.train(["AAPL"])
```

---

## Method 2 — Environment variables

Set variables before running your script or in a `.env` file loaded by
`python-dotenv`:

```bash
export SAGAN_DEFAULT_WINDOW=20
export SAGAN_DEFAULT_EPOCHS=50
export SAGAN_PINN_LAMBDA=0.005
export SAGAN_XAI_CONFIDENCE_THRESHOLD=0.65
export SAGAN_HOME_DIR=/data/sagan
```

```python
from sagan.config import SaganConfig
import sagan

# Replace the global config with env-sourced config
import sagan.config as _cfg_mod
_cfg_mod.config = SaganConfig.from_env()

model_id = sagan.train(["RELIANCE.NS"])
```

---

## Method 3 — `from_dict()` for tests and notebooks

Create isolated config instances for testing or experimentation:

```python
from sagan.config import SaganConfig
from sagan.ensemble import ExplainableEnsemble

cfg = SaganConfig.from_dict({
    "default_window": 5,
    "default_horizon": 1,
    "default_epochs": 3,
    "pinn_lambda": 0.0,            # disable PINN penalty
    "xai_confidence_threshold": 0.4,
})

ens = ExplainableEnsemble(
    tickers=["AAPL"],
    window=cfg.default_window,
    horizon=cfg.default_horizon,
    epochs=cfg.default_epochs,
)
ens.train()
```

---

## Custom storage path

By default models are stored in `~/.sagan/xai_models/`. To use a different
location (e.g. an external drive or a shared network path):

```python
from pathlib import Path
from sagan.config import SaganConfig
import sagan.config as _cfg_mod

_cfg_mod.config = SaganConfig.from_dict({
    "home_dir": Path("/mnt/ml_storage/sagan"),
    "models_dir": Path("/mnt/ml_storage/sagan/models"),
})
```

Or via environment variable:

```bash
export SAGAN_HOME_DIR=/mnt/ml_storage/sagan
export SAGAN_MODELS_DIR=/mnt/ml_storage/sagan/models
```

---

## Hyper-parameter reference

| Parameter | Default | Effect |
|---|---|---|
| `default_window` | 10 | Look-back days per sample |
| `default_horizon` | 3 | Forward days for label |
| `default_threshold` | 0.01 | Return threshold for BUY/SELL |
| `default_head_dim` | 32 | Attention head dimension |
| `default_num_heads` | 4 | Number of attention heads |
| `default_ff_dim` | 64 | Feed-forward hidden size |
| `default_dropout` | 0.1 | Dropout inside TFT |
| `default_epochs` | 30 | Max training epochs |
| `pinn_lambda` | 0.01 | OU regularisation strength |
| `xai_confidence_threshold` | 0.6 | Override trigger threshold |

---

## Logging configuration

```python
from sagan.logging_config import setup_logging

# Debug all data fetch and training steps
setup_logging(level="DEBUG")

# Quiet (WARNING + above only)
setup_logging(level="WARNING")

# Log to file
setup_logging(level="INFO", log_file="sagan_run.log")
```
