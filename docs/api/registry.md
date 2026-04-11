# Registry

The `sagan.registry` module manages the local model store at
`~/.sagan/xai_models/`.

```
~/.sagan/xai_models/
├── registry.json                        ← model index
├── sagan_20240411_120000_abc123/
│   ├── model_buy.h5
│   ├── model_sell.h5
│   ├── model_hold.h5
│   ├── scaler.pkl
│   └── metadata.json
└── ...
```

---

## save_model

::: sagan.registry.save_model

---

## load_ensemble

::: sagan.registry.load_ensemble

---

## list_models

::: sagan.registry.list_models

---

## get_model

::: sagan.registry.get_model

---

## delete_model

::: sagan.registry.delete_model

!!! danger "Irreversible"
    `delete_model()` permanently removes the model directory from disk.
    This cannot be undone.

---

## export_model

::: sagan.registry.export_model

---

## get_model_id

::: sagan.registry.get_model_id

---

## Example: Model lifecycle

```python
import sagan

# Train and register
model_id = sagan.train(["AAPL", "MSFT"])

# Inspect metadata without loading weights
meta = sagan.get_model(model_id)
print(meta["val_sharpe"])

# Export for sharing
path = sagan.export_model(model_id, "/mnt/shared/models/")
print(path)  # /mnt/shared/models/sagan_20240411_120000_abc123

# Clean up
sagan.delete_model(model_id)
```
