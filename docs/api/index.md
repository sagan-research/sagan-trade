# API Reference

Sagan XAI exposes a clean, flat public API. All functions listed here can be
imported directly from the top-level `sagan` package:

```python
import sagan

# All public symbols
sagan.train(...)
sagan.predict(...)
sagan.batch_predict(...)
sagan.list_models()
sagan.delete_model(...)
sagan.export_model(...)
sagan.get_model(...)
sagan.train_parallel(...)
sagan.train_parallel_from_fetch(...)
sagan.setup_logging(...)

# Metrics
sagan.sharpe_ratio(...)
sagan.max_drawdown(...)
sagan.annualised_return(...)
sagan.calmar_ratio(...)
sagan.win_rate(...)
sagan.profit_factor(...)

# Config singleton
sagan.config

# Exceptions
sagan.SaganError
sagan.ModelNotFoundError
sagan.InsufficientDataError
sagan.FetchError
```

Use the sidebar to navigate to the detailed reference for each module.

---

## Module Map

| Module | Contents |
|---|---|
| [`sagan.ensemble`](ensemble.md) | `ExplainableEnsemble`, `train()` |
| [`sagan.predict`](predict.md) | `predict()`, `batch_predict()`, `PredictionResult` |
| [`sagan.data`](data.md) | `fetch_prices()`, `prepare_probabilistic_data()`, `validate_tickers()` |
| [`sagan.registry`](registry.md) | `save_model()`, `load_ensemble()`, `list_models()`, `delete_model()`, `export_model()`, `get_model()` |
| [`sagan.config`](config.md) | `SaganConfig`, `config` singleton |
| [`sagan.cli`](cli.md) | CLI flag reference |
| [`sagan.exceptions`](exceptions.md) | Exception hierarchy |
| [`sagan.utils`](utils.md) | `sharpe_ratio()`, `max_drawdown()`, `annualised_return()`, `calmar_ratio()`, `win_rate()`, `profit_factor()` |
| [`sagan.logging_config`](logging.md) | `setup_logging()` |
