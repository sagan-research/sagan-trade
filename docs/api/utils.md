# Utilities

The `sagan.utils` module provides reusable financial performance metric
helpers. All functions accept 1-D NumPy arrays of **period returns**
(fractions, not percentages).

---

## sharpe_ratio

::: sagan.utils.sharpe_ratio

---

## max_drawdown

::: sagan.utils.max_drawdown

---

## annualised_return

::: sagan.utils.annualised_return

---

## calmar_ratio

::: sagan.utils.calmar_ratio

---

## win_rate

::: sagan.utils.win_rate

---

## profit_factor

::: sagan.utils.profit_factor

---

## Example: Strategy analysis

```python
import numpy as np
from sagan.utils import sharpe_ratio, max_drawdown, calmar_ratio, win_rate

# Simulate a strategy's daily returns
np.random.seed(42)
strategy_returns = np.random.normal(0.0008, 0.012, 252)

print(f"Sharpe:      {sharpe_ratio(strategy_returns):.2f}")
print(f"Max DD:      {max_drawdown(strategy_returns):.2%}")
print(f"Calmar:      {calmar_ratio(strategy_returns):.2f}")
print(f"Win Rate:    {win_rate(strategy_returns):.1%}")
```
