# Changelog

<!-- include the project CHANGELOG here -->

All notable changes to **Sagan XAI** are documented below.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] – 2024-04-11

### Added
- **`ExplainableEnsemble`** – three-head TFT model (buy / sell / hold) with PINN mean-reversion loss.
- **`train()`** – convenience wrapper that fetches, trains, and registers an ensemble in one call.
- **`predict()`** – generates `LONG / SHORT / NEUTRAL` signal with confidence and XAI override flag.
- **`batch_predict()`** – compares signals across multiple saved models with majority-vote consensus.
- **`train_parallel()`** / **`train_parallel_from_fetch()`** – multiprocessing dispatch.
- **`list_models()`** / **`delete_model()`** / **`export_model()`** / **`get_model()`** – full registry lifecycle.
- **`SaganConfig`** – centralised config with `from_env()` and `from_dict()` factories.
- **`setup_logging()`** – configurable logger helper.
- Financial metrics: `sharpe_ratio()`, `max_drawdown()`, `annualised_return()`, `calmar_ratio()`, `win_rate()`, `profit_factor()`.
- Exception hierarchy: `SaganError`, `ModelNotFoundError`, `InsufficientDataError`, `FetchError`, `ConfigurationError`, `RegistryCorruptedError`.
- **CLI** (`sagan --train`, `--predict`, `--list`, `--parallel`, `--model-id`, `--epochs`, `--window`, `--horizon`).
- Full **MkDocs Material** documentation at <https://sagan-docs.vercel.app>.
- **GitHub Actions** CI (lint + type-check + matrix test) and Trusted Publisher PyPI release workflow.
- `py.typed` marker for PEP 561 compliance.

### Notes
- Initial public alpha — API may evolve before `1.0.0`.
- TensorFlow ≥ 2.10 required; GPU is auto-detected.

[0.1.0]: https://github.com/sagan-labs/sagan/releases/tag/v0.1.0
