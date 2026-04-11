# Data

The `sagan.data` module handles price data acquisition and supervised dataset preparation.

---

## fetch_prices

::: sagan.data.fetch_prices

---

## validate_tickers

::: sagan.data.validate_tickers

---

## prepare_probabilistic_data

::: sagan.data.prepare_probabilistic_data

---

## Labelling scheme

Given a forward return $\bar{r}$ and a threshold $\delta$:

| Condition | Label | Class index |
|---|---|---|
| $\bar{r} > \delta$ | **BUY** | 0 |
| $\bar{r} < -\delta$ | **SELL** | 1 |
| $-\delta \le \bar{r} \le \delta$ | **HOLD** | 2 |

The one-hot label vector `y_probs[i]` has shape `(3,)` and sums to 1.

---

## Example: Manual data preparation

```python
from sagan.data import fetch_prices, prepare_probabilistic_data

prices = fetch_prices(["RELIANCE.NS", "TCS.NS"], years=3)
X, y_probs, y_ret, symbols, n_stocks = prepare_probabilistic_data(
    prices, window=10, horizon=3, threshold=0.01
)

print(X.shape)        # (N, 10, 2)
print(y_probs.shape)  # (N, 3)
print(symbols)        # ['RELIANCE.NS', 'TCS.NS']
```
