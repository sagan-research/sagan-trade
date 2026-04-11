"""Logging configuration helper for the Sagan XAI library.

By default the ``sagan`` logger is configured with a ``NullHandler`` so it
does not produce any output unless the consuming application configures its
own logging. Call :func:`setup_logging` once at application start to enable
human-readable log output.

Example:
    >>> from sagan.logging_config import setup_logging
    >>> setup_logging(level="DEBUG", log_file="sagan.log")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGER_NAME = "sagan"

# Attach a NullHandler so library users don't see "No handler found" warnings.
logging.getLogger(_LOGGER_NAME).addHandler(logging.NullHandler())

_DEFAULT_FMT = "%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s"
_DEFAULT_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str | int = "INFO",
    log_file: Path | str | None = None,
    fmt: str = _DEFAULT_FMT,
    date_fmt: str = _DEFAULT_DATE_FMT,
    propagate: bool = False,
) -> logging.Logger:
    """Configure and return the root ``sagan`` logger.

    Calling this function installs a :class:`~logging.StreamHandler` (stdout)
    and optionally a :class:`~logging.FileHandler`. All previous handlers on
    the logger are replaced.

    Args:
        level: Log level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, …)
            or a :mod:`logging` integer constant. Defaults to ``"INFO"``.
        log_file: If provided, also write logs to this file path.
        fmt: :mod:`logging` format string.
        date_fmt: Date format string for the formatter.
        propagate: Whether to propagate records to the root logger.
            Defaults to ``False`` to avoid duplicate output.

    Returns:
        The configured ``sagan`` logger instance.

    Example:
        >>> import sagan
        >>> from sagan.logging_config import setup_logging
        >>> logger = setup_logging(level="DEBUG")
        >>> logger.debug("Verbose output enabled")
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = propagate

    # Remove existing handlers (idempotent re-configuration).
    logger.handlers.clear()

    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # Console handler  -------------------------------------------------------
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler  -------------------------------------------------
    if log_file is not None:
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
