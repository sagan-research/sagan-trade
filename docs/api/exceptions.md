# Exceptions

All Sagan exceptions inherit from :class:`~sagan.exceptions.SaganError`,
allowing callers to catch any library error with a single `except SaganError`.

---

## Hierarchy

```
SaganError
├── ModelNotFoundError
├── InsufficientDataError
├── FetchError
├── ConfigurationError
└── RegistryCorruptedError
```

---

## SaganError

::: sagan.exceptions.SaganError

---

## ModelNotFoundError

::: sagan.exceptions.ModelNotFoundError

---

## InsufficientDataError

::: sagan.exceptions.InsufficientDataError

---

## FetchError

::: sagan.exceptions.FetchError

---

## ConfigurationError

::: sagan.exceptions.ConfigurationError

---

## RegistryCorruptedError

::: sagan.exceptions.RegistryCorruptedError

---

## Catching exceptions

```python
from sagan.exceptions import (
    SaganError,
    ModelNotFoundError,
    FetchError,
    InsufficientDataError,
)
import sagan

# Catch all Sagan errors
try:
    result = sagan.predict(model_id="bad_id")
except ModelNotFoundError as exc:
    print(f"Model '{exc.model_id}' doesn't exist")
except FetchError as exc:
    print(f"Data fetch failed for {exc.tickers}")
except SaganError as exc:
    print(f"Sagan error: {exc}")
```
