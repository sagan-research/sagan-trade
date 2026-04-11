# Predict

The `sagan.predict` module provides the prediction interface for trained ensembles.

---

## PredictionResult

::: sagan.predict.PredictionResult

---

## predict

::: sagan.predict.predict

---

## batch_predict

::: sagan.predict.batch_predict

---

## Example: Comparing models

```python
import sagan

# Train two models with different window sizes
id1 = sagan.train(["AAPL", "MSFT"], window=10, epochs=20)
id2 = sagan.train(["AAPL", "MSFT"], window=20, epochs=20)

# Compare their signals
summary = sagan.batch_predict(model_ids=[id1, id2])
print(summary["consensus_signal"])   # majority vote
print(summary["agreement_rate"])     # 0.5 or 1.0 for 2 models
print(summary["mean_confidence"])
```
