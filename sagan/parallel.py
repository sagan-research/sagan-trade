"""Parallel training across multiple stocks via multiprocessing"""

import multiprocessing as mp
import traceback

from sagan.data import fetch_prices
from sagan.ensemble import ExplainableEnsemble
from sagan.registry import save_model


def _train_stock_process(symbol: str, prices, config_params: dict, result_queue: mp.Queue):
    """Worker function: train one ensemble for *symbol* and push result to queue."""
    try:
        ensemble = ExplainableEnsemble(tickers=[symbol], **config_params)
        # Skip re-fetching; re-use prices passed in
        from sagan.data import prepare_probabilistic_data
        X, y_probs, y_ret, symbols, n = prepare_probabilistic_data(
            prices,
            window=ensemble.window,
            horizon=ensemble.horizon,
            threshold=ensemble.threshold,
        )
        ensemble.X = X
        ensemble.y_probs = y_probs
        ensemble.y_ret = y_ret
        ensemble.symbols = symbols
        ensemble.n_stocks = n

        # Train without re-fetching
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from sagan.config import config as cfg
        from sagan.models.pinn_loss import pinn_loss
        from sagan.models.tft import build_tft_action_model
        import pandas as pd

        split = int(0.8 * len(ensemble.X))
        X_train, X_val = ensemble.X[:split], ensemble.X[split:]
        y_train, y_val = ensemble.y_probs[:split], ensemble.y_probs[split:]

        ensemble.scaler = StandardScaler()
        X_train = ensemble.scaler.fit_transform(
            X_train.reshape(-1, ensemble.n_stocks)
        ).reshape(X_train.shape)
        X_val = ensemble.scaler.transform(
            X_val.reshape(-1, ensemble.n_stocks)
        ).reshape(X_val.shape)

        def _build_and_train(y_tr, y_v):
            m = build_tft_action_model(
                ensemble.window, ensemble.n_stocks,
                ensemble.head_dim, ensemble.num_heads,
                ensemble.ff_dim, ensemble.dropout,
            )
            lp = cfg.pinn_lambda

            def loss_fn(yt, yp):
                return pinn_loss(yt, yp, lambda_pinn=lp)

            m.compile(optimizer="adam", loss=loss_fn)
            m.fit(
                X_train, y_tr,
                epochs=ensemble.epochs,
                batch_size=32,
                validation_data=(X_val, y_v),
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0,
            )
            return m

        ensemble.model_buy = _build_and_train(y_train[:, 0], y_val[:, 0])
        ensemble.model_sell = _build_and_train(y_train[:, 1], y_val[:, 1])
        ensemble.model_hold = _build_and_train(y_train[:, 2], y_val[:, 2])

        logits = np.stack([
            ensemble.model_buy.predict(X_val, verbose=0).flatten(),
            ensemble.model_sell.predict(X_val, verbose=0).flatten(),
            ensemble.model_hold.predict(X_val, verbose=0).flatten(),
        ], axis=1)
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        final_action = np.argmax(probs, axis=1)
        val_ret = ensemble.y_ret[split: split + len(final_action)]
        strat = np.where(final_action == 0, val_ret,
                         np.where(final_action == 1, -val_ret, 0))
        sharpe = np.sqrt(252) * np.mean(strat) / (np.std(strat) + 1e-8)

        ensemble.metadata = {
            "tickers": [symbol],
            "window": ensemble.window,
            "horizon": ensemble.horizon,
            "threshold": ensemble.threshold,
            "head_dim": ensemble.head_dim,
            "num_heads": ensemble.num_heads,
            "ff_dim": ensemble.ff_dim,
            "dropout": ensemble.dropout,
            "val_sharpe": float(sharpe),
            "override_fraction": float(np.mean(probs.max(axis=1) < cfg.xai_confidence_threshold)),
            "created_at": pd.Timestamp.now().isoformat(),
        }

        model_id = save_model(
            ensemble.model_buy, ensemble.model_sell, ensemble.model_hold,
            ensemble.scaler, ensemble.metadata,
        )
        result_queue.put((symbol, model_id, None))
    except Exception as e:
        result_queue.put((symbol, None, f"{e}\n{traceback.format_exc()}"))


def train_parallel(
    tickers,
    prices_dict: dict,
    config_params: dict = None,
    num_processes: int = 12,
) -> dict:
    """Train one ensemble per ticker in parallel subprocesses."""
    if config_params is None:
        config_params = {}
    mp.set_start_method("spawn", force=True)
    result_queue: mp.Queue = mp.Queue()
    processes = []

    for sym in tickers:
        if sym not in prices_dict:
            print(f"⚠️  {sym} missing from prices_dict – skipping.")
            continue
        p = mp.Process(
            target=_train_stock_process,
            args=(sym, prices_dict[sym], config_params, result_queue),
        )
        processes.append(p)
        p.start()

    results = {}
    for _ in range(len(processes)):
        sym, mid, err = result_queue.get()
        if err:
            print(f"❌ {sym} failed: {err}")
            results[sym] = None
        else:
            print(f"✅ {sym} -> {mid}")
            results[sym] = mid

    for p in processes:
        p.join()

    return results


def train_parallel_from_fetch(
    tickers,
    years: int = 5,
    num_processes: int = 12,
    **kwargs,
) -> dict:
    """Fetch data first, then dispatch parallel training."""
    prices_dict = {}
    for sym in tickers:
        print(f"Fetching {sym}…")
        prices_dict[sym] = fetch_prices([sym], years=years)

    config_params = {
        "years": years,
        "window": kwargs.get("window", 10),
        "horizon": kwargs.get("horizon", 3),
        "threshold": kwargs.get("threshold", 0.01),
        "head_dim": kwargs.get("head_dim", 32),
        "num_heads": kwargs.get("num_heads", 4),
        "ff_dim": kwargs.get("ff_dim", 64),
        "dropout": kwargs.get("dropout", 0.1),
        "epochs": kwargs.get("epochs", 30),
    }
    return train_parallel(tickers, prices_dict, config_params, num_processes)
