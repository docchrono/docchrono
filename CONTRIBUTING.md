# Contributing

DocChrono welcomes focused bug fixes, tests, documentation, and deterministic extraction rules.

## Development setup

Use CPython 3.11, 3.12, or 3.13:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Run the local quality gates:

```bash
ruff check .
ruff format --check .
pyright
pytest --cov=docchrono --cov-report=term-missing
python -m build
python -m twine check --strict dist/*
```

New behavior should include deterministic tests and exact provenance assertions. Do not add a
network call to the standard pipeline. Keep optional AGPL or commercial dependencies out of base
installation and document their downstream obligations.

By submitting a contribution, you agree that it may be distributed under Apache-2.0.
