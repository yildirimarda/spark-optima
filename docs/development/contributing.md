# Contributing to Spark Optima

The full contributing guide lives at [CONTRIBUTING.md](contributing.md) in the repository root.

It covers:

- **Bug reports & feature requests** — how to open an issue
- **Pull request process** — branch naming, commit style, review checklist
- **Development setup** — prerequisites, `uv sync`, pre-commit hooks
- **Project structure** — where to find each module
- **Coding standards** — line length, type hints, docstrings, test requirements

## Quick Setup

```bash
git clone https://github.com/yildirimarda/spark-optima.git
cd spark-optima
uv sync
uv run pre-commit install
```

## Running Checks

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy src/spark_optima # type check
uv run pytest                # tests
```
