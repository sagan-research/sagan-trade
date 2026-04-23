"""Model saving, loading, listing, and lifecycle management.

The registry is a JSON file at
``~/.sagan/xai_models/registry.json`` that records metadata for every saved
ensemble. Each model is stored in its own subdirectory::

    ~/.sagan/xai_models/
    ├── registry.json
    ├── sagan_20240411_120000_abc123/
    │   ├── model_buy.h5
    │   ├── model_sell.h5
    │   ├── model_hold.h5
    │   ├── scaler.pkl
    │   └── metadata.json
    └── ...
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import tensorflow as tf

from sagan.config import config
from sagan.exceptions import ModelNotFoundError, RegistryCorruptedError
from sagan.models.tft import VariableSelectionNetwork, TemporalFusionBlock

__all__ = [
    "save_model",
    "load_ensemble",
    "list_models",
    "get_model",
    "delete_model",
    "export_model",
    "get_model_id",
]


def _generate_model_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = hashlib.md5(os.urandom(8)).hexdigest()[:6]
    return f"sagan_{ts}_{rand}"


def get_model_id() -> str:
    """Generate a unique, time-stamped model identifier.

    Returns:
        A string of the form ``sagan_YYYYMMDD_HHMMSS_xxxxxx``.
    """
    return _generate_model_id()


def _load_registry() -> dict:
    registry_file = config.models_dir / "registry.json"
    if not registry_file.exists():
        return {"models": []}
    try:
        with open(registry_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise RegistryCorruptedError(str(registry_file), str(exc)) from exc


def _save_registry(registry: dict) -> None:
    registry_file = config.models_dir / "registry.json"
    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def save_model(
    model_buy: Any,
    model_sell: Any,
    model_hold: Any,
    scaler: Any,
    metadata: dict,
    is_symbolic: bool = False,
) -> str:
    """Persist a trained ensemble to disk and register it.
    
    Now supports both Keras models and Symbolic/Math expressions.
    """
    model_id = _generate_model_id()
    model_dir: Path = config.models_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    if is_symbolic:
        # Save as JSON/Pickle
        with open(model_dir / "model_buy.json", "w") as f:
            json.dump(model_buy, f)
        with open(model_dir / "model_sell.json", "w") as f:
            json.dump(model_sell, f)
        with open(model_dir / "model_hold.json", "w") as f:
            json.dump(model_hold, f)
    else:
        # Legacy Keras paths
        model_buy.save(str(model_dir / "model_buy.h5"))
        model_sell.save(str(model_dir / "model_sell.h5"))
        model_hold.save(str(model_dir / "model_hold.h5"))

    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    metadata["is_symbolic"] = is_symbolic
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    registry = _load_registry()
    registry["models"].append({"model_id": model_id, **metadata})
    _save_registry(registry)

    return model_id


@functools.lru_cache(maxsize=16)
def load_ensemble(model_id: str):
    """Load a saved ensemble from disk."""
    model_dir: Path = config.models_dir / model_id
    if not model_dir.exists():
        raise ModelNotFoundError(model_id, str(config.models_dir))

    with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    is_symbolic = metadata.get("is_symbolic", False)

    if is_symbolic:
        with open(model_dir / "model_buy.json", "r") as f:
            model_buy = json.load(f)
        with open(model_dir / "model_sell.json", "r") as f:
            model_sell = json.load(f)
        with open(model_dir / "model_hold.json", "r") as f:
            model_hold = json.load(f)
    else:
        # Legacy Keras loading
        import tensorflow as tf
        from sagan.models.tft import VariableSelectionNetwork, TemporalFusionBlock
        custom_objects = {
            "VariableSelectionNetwork": VariableSelectionNetwork,
            "TemporalFusionBlock": TemporalFusionBlock,
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            model_buy = tf.keras.models.load_model(str(model_dir / "model_buy.h5"), compile=False, safe_mode=False)
            model_sell = tf.keras.models.load_model(str(model_dir / "model_sell.h5"), compile=False, safe_mode=False)
            model_hold = tf.keras.models.load_model(str(model_dir / "model_hold.h5"), compile=False, safe_mode=False)

    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model_buy, model_sell, model_hold, scaler, metadata


def list_models() -> "pd.DataFrame":
    """Return a DataFrame of all registered models, ordered by creation time.

    Returns:
        A :class:`~pandas.DataFrame` with one row per model. Returns an empty
        DataFrame if no models have been trained yet.

    Example:
        >>> import sagan
        >>> df = sagan.list_models()
        >>> print(df[["model_id", "tickers", "val_sharpe"]].to_string())
    """
    import pandas as pd

    registry = _load_registry()
    df = pd.DataFrame(registry["models"])
    if not df.empty and "created_at" in df.columns:
        df = df.sort_values("created_at").reset_index(drop=True)
    return df


def get_model(model_id: str) -> dict:
    """Return the metadata dict for a single model without loading weights.

    Args:
        model_id: The model identifier.

    Returns:
        Metadata dict as stored in ``metadata.json``.

    Raises:
        ModelNotFoundError: If *model_id* is not found.
    """
    model_dir: Path = config.models_dir / model_id
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise ModelNotFoundError(model_id, str(config.models_dir))
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_model(model_id: str) -> None:
    """Permanently delete a model from disk and remove it from the registry.

    Args:
        model_id: The model identifier to delete.

    Raises:
        ModelNotFoundError: If *model_id* does not exist.

    Warning:
        This operation is **irreversible**. The model directory and all its
        contents (weights, scaler, metadata) are removed.

    Example:
        >>> import sagan
        >>> sagan.delete_model("sagan_20240411_120000_abc123")
    """
    model_dir: Path = config.models_dir / model_id
    if not model_dir.exists():
        raise ModelNotFoundError(model_id, str(config.models_dir))

    shutil.rmtree(model_dir)

    registry = _load_registry()
    registry["models"] = [m for m in registry["models"] if m.get("model_id") != model_id]
    _save_registry(registry)


def export_model(model_id: str, destination: str | Path) -> Path:
    """Copy a trained model directory to an arbitrary destination.

    Useful for sharing a model or archiving it outside the default data
    directory.

    Args:
        model_id: The model identifier to export.
        destination: Target directory path. The model will be saved as a
            subdirectory named *model_id* inside *destination*.

    Returns:
        Path to the exported model directory.

    Raises:
        ModelNotFoundError: If *model_id* does not exist.

    Example:
        >>> import sagan
        >>> path = sagan.export_model("sagan_20240411_120000_abc123", "/mnt/share/")
        >>> print(path)
        /mnt/share/sagan_20240411_120000_abc123
    """
    model_dir: Path = config.models_dir / model_id
    if not model_dir.exists():
        raise ModelNotFoundError(model_id, str(config.models_dir))

    dest = Path(destination) / model_id
    shutil.copytree(model_dir, dest)
    return dest
