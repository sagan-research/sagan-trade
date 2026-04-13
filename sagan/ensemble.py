"""Main ensemble trainer for Sagan XAI.

The :class:`ExplainableEnsemble` trains three independent TFT action models
— buy, sell, and hold — each using a Physics-Informed Neural Network loss
that encodes mean-reversion as an Ornstein–Uhlenbeck penalty.

After training, call :meth:`~ExplainableEnsemble.save` to persist the
ensemble to the local registry, and use :func:`~sagan.predict.predict` to
generate signals from any saved model.

Typical usage::

    from sagan.ensemble import ExplainableEnsemble

    ens = ExplainableEnsemble(
        tickers=["RELIANCE.NS", "TCS.NS"],
        window=15,
        epochs=40,
    )
    metadata = ens.train()
    model_id = ens.save()
    print(f"Saved as {model_id}  Sharpe={metadata['val_sharpe']:.2f}")
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

from sagan.config import config
from sagan.data import fetch_prices, prepare_probabilistic_data
from sagan.models.pinn_loss import pinn_loss
from sagan.models.tft import build_tft_action_model
from sagan.registry import save_model

logger = logging.getLogger("sagan")

__all__ = ["ExplainableEnsemble", "train"]


class ExplainableEnsemble:
    """Three-head TFT ensemble with PINN mean-reversion loss.

    This class manages the full lifecycle of a Sagan model: fetching price
    data, building and scaling the dataset, training three independent action
    heads, computing validation metrics, and saving to the registry.

    Args:
        tickers: List of Yahoo Finance ticker symbols to train on.
        years: Years of historical data to download. Defaults to 5.
        window: Look-back window in trading days. Defaults to
            :attr:`~sagan.config.SaganConfig.default_window`.
        horizon: Forward horizon for label generation. Defaults to
            :attr:`~sagan.config.SaganConfig.default_horizon`.
        threshold: Return threshold for buy/sell/hold labelling. Defaults to
            :attr:`~sagan.config.SaganConfig.default_threshold`.
        head_dim: Key/value dimension per attention head. Defaults to
            :attr:`~sagan.config.SaganConfig.default_head_dim`.
        num_heads: Number of parallel attention heads. Defaults to
            :attr:`~sagan.config.SaganConfig.default_num_heads`.
        ff_dim: Feed-forward hidden dimension. Defaults to
            :attr:`~sagan.config.SaganConfig.default_ff_dim`.
        dropout: Dropout probability inside the TFT. Defaults to
            :attr:`~sagan.config.SaganConfig.default_dropout`.
        epochs: Maximum training epochs (early stopping may stop earlier).
            Defaults to :attr:`~sagan.config.SaganConfig.default_epochs`.
        verbose: If ``True``, Keras training progress is printed to stdout.

    Example:
        >>> ens = ExplainableEnsemble(["AAPL", "MSFT"], window=15, epochs=20)
        >>> meta = ens.train()
        >>> model_id = ens.save()
    """

    def __init__(
        self,
        tickers: list[str],
        years: int = 5,
        window: int | None = None,
        horizon: int | None = None,
        threshold: float | None = None,
        head_dim: int | None = None,
        num_heads: int | None = None,
        ff_dim: int | None = None,
        dropout: float | None = None,
        epochs: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.tickers = tickers
        self.years = years
        self.window = window or config.default_window
        self.horizon = horizon or config.default_horizon
        self.threshold = threshold or config.default_threshold
        self.head_dim = head_dim or config.default_head_dim
        self.num_heads = num_heads or config.default_num_heads
        self.ff_dim = ff_dim or config.default_ff_dim
        self.dropout = dropout or config.default_dropout
        self.epochs = epochs or config.default_epochs
        self.verbose = verbose

        self.model_buy: tf.keras.Model | None = None
        self.model_sell: tf.keras.Model | None = None
        self.model_hold: tf.keras.Model | None = None
        self.scaler: StandardScaler | None = None
        self.metadata: dict[str, Any] = {}

        # Internal dataset arrays (populated by train())
        self.X: np.ndarray | None = None
        self.y_probs: np.ndarray | None = None
        self.y_ret: np.ndarray | None = None
        self.symbols: list[str] = []
        self.n_stocks: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_and_prepare(self) -> None:
        """Download prices and build the supervised dataset.

        Populates ``self.X``, ``self.y_probs``, ``self.y_ret``,
        ``self.symbols``, and ``self.n_stocks``.
        """
        prices = fetch_prices(self.tickers, self.years)
        X, y_probs, y_ret, symbols, n = prepare_probabilistic_data(
            prices, self.window, self.horizon, self.threshold
        )
        self.X = X
        self.y_probs = y_probs
        self.y_ret = y_ret
        self.symbols = symbols
        self.n_stocks = n
        logger.info("Prepared %d samples across %d stocks.", len(X), n)

    def _train_action_model(
        self,
        X_train: np.ndarray,
        y_train_bin: np.ndarray,
        X_val: np.ndarray,
        y_val_bin: np.ndarray,
        name: str,
        progress_callback: Any = None,
        progress_increment: float = 0.0,
    ) -> tf.keras.Model:
        """Build and train one action head.

        Args:
            X_train: Training inputs ``(N, window, n_stocks)``.
            y_train_bin: Binary training labels ``(N,)``.
            X_val: Validation inputs.
            y_val_bin: Validation labels.
            name: Human-readable head name for logging (``"buy"`` / ``"sell"`` / ``"hold"``).
            progress_callback: Optional callable for UI progress updates.
            progress_increment: Value to add to the progress bar after completion.

        Returns:
            The trained Keras model.
        """
        logger.info("Training '%s' head…", name)
        model = build_tft_action_model(
            self.window, self.n_stocks,
            self.head_dim, self.num_heads, self.ff_dim, self.dropout,
        )
        lambda_pinn = config.pinn_lambda

        def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            return pinn_loss(y_true, y_pred, lambda_pinn=lambda_pinn)

        model.compile(
            optimizer="adam",
            loss={"logit": loss_fn, "selection_weights": None},
            metrics={"logit": "accuracy"}
        )
        early = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(
            X_train, {"logit": y_train_bin},
            epochs=self.epochs,
            batch_size=32,
            validation_data=(X_val, {"logit": y_val_bin}),
            callbacks=[early],
            verbose=1 if self.verbose else 0,
        )
        if progress_callback:
            progress_callback(progress_increment)
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, progress_callback: Any = None) -> dict[str, Any]:
        """Fetch data, train all three action heads, and compute metrics.

        This method:

        1. Downloads price data (via :func:`~sagan.data.fetch_prices`).
        2. Builds the sliding-window dataset.
        3. Splits 80/20 into train / validation.
        4. Standardises inputs using :class:`~sklearn.preprocessing.StandardScaler`.
        5. Trains BUY, SELL, and HOLD models with PINN loss.
        6. Computes validation Sharpe ratio and XAI override fraction.

        Returns:
            A metadata dict with keys: ``tickers``, ``window``, ``horizon``,
            ``threshold``, TFT hyper-parameters, ``val_sharpe``,
            ``override_fraction``, ``created_at``.

        Raises:
            FetchError: If price data cannot be downloaded.
            InsufficientDataError: If there is not enough data for training.
        """
        self._fetch_and_prepare()
        if progress_callback:
            progress_callback(0.1)
        assert self.X is not None and self.y_probs is not None

        split = int(0.8 * len(self.X))
        X_train, X_val = self.X[:split], self.X[split:]
        y_train, y_val = self.y_probs[:split], self.y_probs[split:]

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(
            X_train.reshape(-1, self.n_stocks)
        ).reshape(X_train.shape)
        X_val = self.scaler.transform(
            X_val.reshape(-1, self.n_stocks)
        ).reshape(X_val.shape)
        self.X_train, self.X_val = X_train, X_val

        self.model_buy = self._train_action_model(
            X_train, y_train[:, 0], X_val, y_val[:, 0], "buy", 
            progress_callback, 0.3
        )
        self.model_sell = self._train_action_model(
            X_train, y_train[:, 1], X_val, y_val[:, 1], "sell", 
            progress_callback, 0.3
        )
        self.model_hold = self._train_action_model(
            X_train, y_train[:, 2], X_val, y_val[:, 2], "hold", 
            progress_callback, 0.3
        )

        # Validation metrics
        preds_buy = self.model_buy.predict(X_val, verbose=0)
        preds_sell = self.model_sell.predict(X_val, verbose=0)
        preds_hold = self.model_hold.predict(X_val, verbose=0)

        logits_buy = preds_buy["logit"].flatten()
        logits_sell = preds_sell["logit"].flatten()
        logits_hold = preds_hold["logit"].flatten()
        
        logits = np.stack([logits_buy, logits_sell, logits_hold], axis=1)
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        final_action = np.argmax(probs, axis=1)

        val_returns = self.y_ret[split : split + len(final_action)]
        strategy_returns = np.where(
            final_action == 0, val_returns,
            np.where(final_action == 1, -val_returns, 0),
        )
        sharpe = (
            float(np.sqrt(252) * np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8))
        )
        override_frac = float(np.mean(probs.max(axis=1) < config.xai_confidence_threshold))

        self.metadata = {
            "tickers": self.tickers,
            "window": self.window,
            "horizon": self.horizon,
            "threshold": self.threshold,
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "val_sharpe": sharpe,
            "override_fraction": override_frac,
            "created_at": pd.Timestamp.now().isoformat(),
        }
        logger.info(
            "Training complete — val Sharpe=%.3f  override_frac=%.1f%%",
            sharpe, override_frac * 100,
        )
        return self.metadata

    # Sklearn-style alias
    fit = train

    def save(self) -> str:
        """Persist the trained ensemble to the registry.

        Must be called after :meth:`train`.

        Returns:
            The generated ``model_id`` string.

        Raises:
            RuntimeError: If :meth:`train` has not been called yet.
        """
        if self.model_buy is None:
            raise RuntimeError("Call .train() before .save().")
        model_id = save_model(
            self.model_buy, self.model_sell, self.model_hold,
            self.scaler, self.metadata,
        )
        logger.info("Ensemble saved → %s (Sharpe=%.3f)", model_id, self.metadata["val_sharpe"])
        return model_id


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def train(tickers: list[str], progress_callback: Any = None, **kwargs: Any) -> str:
    """Train a new ensemble and save it to the registry in one call.

    This is the primary entry-point for most users. It creates an
    :class:`ExplainableEnsemble`, calls :meth:`~ExplainableEnsemble.train`,
    and returns the saved ``model_id``.

    Args:
        tickers: List of Yahoo Finance ticker symbols.
        progress_callback: Optional callable for UI progress updates.
        **kwargs: Any keyword argument accepted by :class:`ExplainableEnsemble`
            (e.g. ``window``, ``epochs``, ``head_dim``).

    Returns:
        The ``model_id`` string of the saved ensemble.

    Example:
        >>> import sagan
        >>> model_id = sagan.train(["RELIANCE.NS", "TCS.NS"], epochs=20, window=15)
        >>> print(model_id)
        sagan_20240411_120000_abc123
    """
    ensemble = ExplainableEnsemble(tickers, **kwargs)
    ensemble.train(progress_callback=progress_callback)
    return ensemble.save()
