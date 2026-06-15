# Config

The `sagan.config` module exposes the :class:`~sagan.config.SaganConfig` dataclass
and the module-level `config` singleton that all Sagan modules share.

---

## SaganConfig

::: sagan.config.SaganConfig

---

## config (singleton)

```python
from sagan import config

# Inspect defaults
print(config.default_window)             # 10
print(config.xai_confidence_threshold)  # 0.6
print(config.models_dir)                # ~/.sagan/xai_models

# Mutate for the current session
config.default_epochs = 50
config.pinn_lambda = 0.005
```

---

## Environment Variables

`SaganConfig.from_env()` reads the following variables:

| Variable | Field | Type |
|---|---|---|
| `SAGAN_HOME_DIR` | `home_dir` | `Path` |
| `SAGAN_MODELS_DIR` | `models_dir` | `Path` |
| `SAGAN_DEFAULT_WINDOW` | `default_window` | `int` |
| `SAGAN_DEFAULT_HORIZON` | `default_horizon` | `int` |
| `SAGAN_DEFAULT_EPOCHS` | `default_epochs` | `int` |
| `SAGAN_DEFAULT_HEAD_DIM` | `default_head_dim` | `int` |
| `SAGAN_DEFAULT_NUM_HEADS` | `default_num_heads` | `int` |
| `SAGAN_DEFAULT_FF_DIM` | `default_ff_dim` | `int` |
| `SAGAN_DEFAULT_DROPOUT` | `default_dropout` | `float` |
| `SAGAN_DEFAULT_THRESHOLD` | `default_threshold` | `float` |
| `SAGAN_PINN_LAMBDA` | `pinn_lambda` | `float` |
| `SAGAN_XAI_CONFIDENCE_THRESHOLD` | `xai_confidence_threshold` | `float` |
| `SAGAN_REGIME_CHANGE_THRESHOLD` | `regime_change_threshold` | `float` |

---

## Example: Programmatic override

```python
from sagan.config import SaganConfig
import sagan

# Override via dict
cfg = SaganConfig.from_dict({
    "default_window": 20,
    "default_epochs": 50,
    "pinn_lambda": 0.005,
})

# Use with ExplainableEnsemble
from sagan import ExplainableEnsemble
ens = ExplainableEnsemble(
    ["AAPL"],
    window=cfg.default_window,
    epochs=cfg.default_epochs,
)
```
