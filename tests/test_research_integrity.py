import numpy as np
import pandas as pd

from sagan_trade.symbolic_regressor import SymbolicRegressor


def _prices(n=240):
    rng = np.random.default_rng(7)
    returns = 0.0002 + rng.normal(0, 0.01, n)
    close = 100 * np.cumprod(1 + returns)
    volume = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame(
        {"Close": close, "Volume": volume},
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )


def test_symbolic_regressor_uses_chronological_validation():
    model = SymbolicRegressor()
    data = _prices()
    model.train("SYNTH", ["RSI"], data=data, validation_fraction=0.2)

    assert model.best_formula_name is not None
    assert model.fitted_params is not None
    assert np.isfinite(model.validation_mse)
    prediction, formula = model.predict(data.tail(20))
    assert len(prediction) == 20
    assert prediction.index.equals(data.tail(20).index)
    assert formula


def test_symbolic_regressor_does_not_use_backward_fill():
    import inspect

    source = inspect.getsource(SymbolicRegressor)
    assert ".bfill(" not in source
    assert "shift(10).bfill" not in source
