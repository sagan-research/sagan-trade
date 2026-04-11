# Logging

The `sagan.logging_config` module configures the `sagan` logger.

By default, Sagan uses a `NullHandler` so it produces **no output** in
library mode. Call :func:`~sagan.logging_config.setup_logging` once at
application start to enable human-readable logs.

---

## setup_logging

::: sagan.logging_config.setup_logging

---

## Example

```python
from sagan.logging_config import setup_logging

# Console-only at INFO level
setup_logging(level="INFO")

# Verbose debug with file output
setup_logging(level="DEBUG", log_file="sagan_debug.log")

# Suppress all output (restore NullHandler behaviour)
import logging
logging.getLogger("sagan").handlers.clear()
logging.getLogger("sagan").addHandler(logging.NullHandler())
```

---

## Log format

The default format is:

```
2024-04-11 12:00:00  sagan                INFO      Fetching 5 year(s) for 3 tickers: ...
2024-04-11 12:00:05  sagan                INFO      Prepared 1237 samples across 3 stocks.
2024-04-11 12:00:05  sagan                INFO      Training 'buy' head…
```
