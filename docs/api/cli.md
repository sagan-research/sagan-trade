# CLI Reference

The `sagan` command-line tool is installed automatically with the package.

```bash
sagan --help
```

---

## Commands

### `--train`

Train a new ensemble on one or more tickers.

```bash
sagan --train RELIANCE.NS TCS.NS HDFCBANK.NS
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--train TICKERS…` | `str+` | — | **Required.** Ticker symbols |
| `--years INT` | int | 5 | Years of historical data |
| `--window INT` | int | 10 | Look-back window (days) |
| `--horizon INT` | int | 3 | Forward horizon for labelling |
| `--epochs INT` | int | 30 | Max training epochs |
| `--parallel` | flag | off | Train one model per ticker in parallel |
| `--num-processes INT` | int | 12 | Parallel worker processes |

---

### `--predict`

Generate a trading signal using the latest (or a specific) saved model.

```bash
sagan --predict
sagan --predict --model-id sagan_20240411_120000_abc123
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--predict` | flag | — | **Required.** Activate predict mode |
| `--model-id STR` | str | latest | Specific model to query |

---

### `--list`

Display a table of all registered models.

```bash
sagan --list
```

Output columns: `model_id`, `tickers`, `val_sharpe`, `override_fraction`, `created_at`.

---

## Examples

```bash
# Train on NSE blue-chips, 5 years
sagan --train RELIANCE.NS TCS.NS INFY.NS HDFCBANK.NS --years 5

# Parallel training on US tech (8 workers)
sagan --train AAPL MSFT GOOGL AMZN META --parallel --num-processes 8

# Aggressive hyper-params
sagan --train AAPL --window 20 --horizon 5 --epochs 100

# Predict with a specific model
sagan --predict --model-id sagan_20240411_120000_abc123

# List all models
sagan --list
```
