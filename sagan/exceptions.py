"""Custom exception hierarchy for the Sagan XAI library.

All exceptions inherit from :class:`SaganError`, so callers can catch the
base class to handle any library error, or catch specific subclasses for
fine-grained control.

Example:
    >>> from sagan.exceptions import ModelNotFoundError
    >>> try:
    ...     sagan.predict(model_id="non_existent")
    ... except ModelNotFoundError as exc:
    ...     print(f"Model missing: {exc}")
"""

from __future__ import annotations


class SaganError(Exception):
    """Base exception for all Sagan XAI errors."""


class ModelNotFoundError(SaganError):
    """Raised when a requested ``model_id`` does not exist in the registry.

    Args:
        model_id: The identifier that was not found.
        registry_path: Path to the registry JSON file that was searched.
    """

    def __init__(self, model_id: str, registry_path: str | None = None) -> None:
        self.model_id = model_id
        self.registry_path = registry_path
        loc = f" (searched {registry_path})" if registry_path else ""
        super().__init__(f"Model '{model_id}' not found{loc}.")


class InsufficientDataError(SaganError):
    """Raised when there are not enough data points for the configured window/horizon.

    Args:
        available: Number of rows actually available.
        required: Minimum rows required.
    """

    def __init__(self, available: int, required: int) -> None:
        self.available = available
        self.required = required
        super().__init__(
            f"Insufficient data: need at least {required} rows, got {available}."
        )


class FetchError(SaganError):
    """Raised when downloading price data from Yahoo Finance fails.

    Args:
        tickers: The ticker symbols that failed to fetch.
        cause: The underlying exception, if any.
    """

    def __init__(self, tickers: list[str], cause: Exception | None = None) -> None:
        self.tickers = tickers
        self.cause = cause
        msg = f"Failed to fetch data for tickers: {tickers}"
        if cause:
            msg += f" — {cause}"
        super().__init__(msg)


class ConfigurationError(SaganError):
    """Raised for invalid or conflicting configuration values.

    Args:
        field: The configuration field name that is invalid.
        value: The invalid value provided.
        reason: Human-readable description of why the value is invalid.
    """

    def __init__(self, field: str, value: object, reason: str) -> None:
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid config '{field}={value}': {reason}")


class RegistryCorruptedError(SaganError):
    """Raised when the registry JSON file is missing required fields or malformed.

    Args:
        path: Path to the registry file.
        detail: Description of the corruption.
    """

    def __init__(self, path: str, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"Registry at '{path}' is corrupted: {detail}")



