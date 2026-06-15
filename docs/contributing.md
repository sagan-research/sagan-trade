# Contributing

Thank you for considering a contribution to Sagan XAI! This guide covers
everything from setting up your development environment to submitting a
pull request.

---

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/sagan
cd sagan

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev extras
pip install -e ".[dev]"
```

---

## Code Style

Sagan uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check
ruff check sagan/ tests/

# Auto-fix
ruff check --fix sagan/ tests/
```

Type annotations are enforced with [Mypy](https://mypy.readthedocs.io/):

```bash
mypy sagan/
```

All public functions and classes **must** have Google-style docstrings.

---

## Running Tests

```bash
pytest tests/ -v --cov=sagan --cov-report=term-missing
```

Test files live in `tests/`. Mirror the `sagan/` package structure when
adding new tests (e.g. `tests/test_utils.py` for `sagan/utils.py`).

!!! tip "Speed up tests"
    Use `pytest -x` to stop at the first failure, and `pytest -k "data"` to
    run only tests matching a keyword.

---

## Adding a New Feature

1. **Open an issue** to discuss the feature before writing code.
2. **Create a branch**: `git checkout -b feat/my-feature`
3. **Write the code** with Google-style docstrings and type annotations.
4. **Add tests** with ≥ 90 % coverage for the new code.
5. **Update docs** — add or edit a page in `docs/`.
6. **Run the full check**:
   ```bash
   ruff check sagan/ tests/
   mypy sagan/
   pytest tests/ -v
   ```
7. **Submit a pull request** targeting `main`.

---

## Building Docs Locally

```bash
# Install doc dependencies
pip install -r docs/requirements.txt

# Serve with live reload
mkdocs serve
```

Browse to http://127.0.0.1:8000 to preview the docs site. Changes to any
Markdown file or to Python docstrings (when the source package is installed
in editable mode) will trigger a live reload.

---

## Publishing a Release (Maintainers Only)

1. Update `CHANGELOG.md` with the new version heading.
2. Bump `version` in `pyproject.toml`.
3. Commit: `git commit -am "chore: release v0.2.0"`
4. Tag: `git tag v0.2.0 && git push --tags`

The GitHub Actions `publish.yml` workflow will automatically build the wheel,
publish to TestPyPI for verification, and then publish to PyPI.

!!! warning "Trusted Publisher"
    Publication uses **OIDC Trusted Publisher** — no API tokens or secrets
    are required. Make sure the Sagan PyPI project has the GitHub repository
    configured as a trusted publisher under *Publishing → Trusted publishers*.

---

## Code of Conduct

Sagan follows the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Be kind, respectful, and constructive.
