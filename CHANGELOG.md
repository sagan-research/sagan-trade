# Changelog

All notable changes to **Sagan XAI** will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.1] – 2026-04-23

### Added
- **SymbolicBasis Framework**: Hierarchical Polynomial and Fourier basis discovery for high-fidelity fitting ($R^2 \ge 0.92$).
- **FunctionGemma Integration**: AI-orchestrated mathematical composition of signals.
- **Portfolio Studio**: New dashboard for multi-asset portfolio mathematical discovery.
- **Improved CLI**: New commands for portfolio training and model management.

## [0.3.0] – 2026-04-23

### Changed
- Major rebranding to **Sagan Trade**.
- Transitioned to `pyproject.toml` as the primary build configuration.

## [0.2.1] – 2026-04-15

### Fixed
- Dashboard SyntaxError: Fixed `nonlocal` scoping issue in `sagan/app.py` that occurred when using certain Python versions or execution contexts.

## [0.2.0] – 2026-04-15

### Added
- SEBI Compliance Engine: Dual export of signals to `.md` and `.json`.
- Automated Threshold Optimization: Model-generated RSI and Volatility boundaries for 2000+ NSE stocks.
- HITL Audit Logic: Local SQLite database (`sagan.db`) recording all actions and model justifications.
- Gating-Driven Conflict resolution: Using TFT Variable Selection Network (VSN) weights to manage signal discrepancies.
- Consolidated Typer CLI: New commands `userlogs`, `metrics`, and `dash`.

## [0.0.1] – 2026-04-15

### Changed
- **Completely removed paywall and credit tracking**: Firestore-based billing and tiered feature gating have been stripped. The library is now fully open and free.
- **Fixed TensorFlow retracing issue**: Optimized prediction pipeline to use direct model calls and added an LRU cache for model loading, resolving expensive tracing warnings during inference.
- **Reset versioning**: Reset package version to `0.0.1` as per state transition.

### Removed
- `sagan.billing` module and all related Firestore/sync logic.
- `auth`, `sync`, and `usage` CLI commands.
- Dependencies: `authlib`, `python-jose`, `httpx`.

---

## [0.1.0] – 2024-04-11

### Added
- **`ExplainableEnsemble`** – three-head TFT model (buy / sell / hold) with PINN mean-reversion loss.
- **`train()`** – convenience wrapper that fetches, trains, and registers an ensemble in one call.
- **`predict()`** – generates a `LONG / SHORT / NEUTRAL` signal with confidence and XAI override flag.
- **`batch_predict()`** – compares signals across multiple saved model IDs.
- **`train_parallel()`** / **`train_parallel_from_fetch()`** – multiprocessing dispatch for training one model per ticker simultaneously.
- **`list_models()`** / **`delete_model()`** / **`export_model()`** – full registry lifecycle management.
- **`SaganConfig`** – centralised configuration dataclass with `from_env()` and `from_dict()` factory methods.
- **`setup_logging()`** – configurable logger helper.
- **`sharpe_ratio()`**, **`max_drawdown()`**, **`annualised_return()`**, **`calmar_ratio()`** – reusable financial metrics.
- **Custom exception hierarchy**: `SaganError`, `ModelNotFoundError`, `InsufficientDataError`, `FetchError`, `ConfigurationError`, `RegistryCorruptedError`.
- **CLI** (`sagan --train`, `--predict`, `--list`, `--parallel`, `--model-id`, `--epochs`, `--window`, `--horizon`).
- **Full MkDocs Material documentation** hosted at <https://sagan-docs.vercel.app>.
- **GitHub Actions** CI (lint + type-check + test) and PyPI publish workflows.
- `py.typed` marker for PEP 561 compliance.

### Notes
- Initial public alpha. API may evolve before `1.0.0`.
- TensorFlow ≥ 2.10 required; GPU is auto-detected but not required.

[0.1.0]: https://github.com/sagan-labs/sagan/releases/tag/v0.1.0
