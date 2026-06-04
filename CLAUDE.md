# Spark Optima — Claude Code Guide

## Project Overview

Spark Optima is an intelligent Apache Spark configuration optimization tool. It uses a **Hybrid (Heuristic + Bayesian/Optuna)** approach to find the optimal Spark configuration for a given workload, platform, and resource budget.

Key components:
- `src/spark_optima/core/` — optimizer pipeline (heuristics → Bayesian → result)
- `src/spark_optima/platforms/` — platform adapters (Local, AWS Glue, Databricks, Azure Synapse)
- `src/spark_optima/analysis/` — AST-based Spark code smell detection
- `src/spark_optima/api/` — FastAPI REST API
- `src/spark_optima/cli/` — Typer CLI + interactive wizard
- `data/configs/` — Spark 3.x/4.x parameter databases (YAML)

## Development Environment

This project uses **uv** for dependency management. Always use `uv run` — never activate the venv manually.

```bash
# Install all dependencies (including dev)
uv sync

# Run any tool
uv run pytest
uv run ruff check .
uv run mypy src/spark_optima
uv run mkdocs serve
```

Never use `python`, `pip`, or `poetry` directly.

## Code Style

- Follow PEP 8 for Python code
- Use type hints for all function signatures (public and private)
- Write docstrings for all public classes and methods (Google style)
- Maximum line length: **120 characters** (enforced by ruff)
- Use **ruff** for both linting and formatting — not black, not isort separately
- All code comments must be in **English**

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage (CI requires 80%+)
uv run pytest --cov=src/spark_optima --cov-fail-under=80 --cov-report=html

# Run only unit tests
uv run pytest -m unit -v

# Run only integration tests
uv run pytest -m integration -v

# Run a specific file
uv run pytest tests/unit/core/test_optimizer.py -v
```

- Write unit tests for all new functionality
- **CI enforces 80%+ coverage** via `--cov-fail-under=80`
- Test files mirror source structure: `src/spark_optima/core/optimizer.py` → `tests/unit/core/test_optimizer.py`
- Use `pytest.mark.skipif` with `importlib.util.find_spec` for optional dependencies (boto3, etc.)
- Do not use `|| true` on test commands — failures must be visible

## Code Quality Checks

Run all of these before committing:

```bash
uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy src/spark_optima     # type check
uv run bandit -r src/spark_optima  # security scan
```

The pre-commit hooks run ruff, mypy, and bandit automatically. Install once with:

```bash
uv run pre-commit install
```

## Project Structure

```
src/spark_optima/
├── core/
│   ├── optimizer.py          # Top-level Optimizer class (entry point)
│   ├── result.py             # OptimizationResult, CodeSuggestion
│   ├── bayesian/             # Optuna integration, search space, trial runner
│   ├── config_engine/        # Spark param database, loader, validator
│   ├── execution/            # Real Spark run engine, metrics, monitor
│   ├── heuristics/           # Rule engine, EvaluationContext, rules
│   └── simulation/           # Performance model, predictor
├── platforms/
│   ├── base.py               # Abstract Platform class
│   ├── models.py             # ResourceSpec, WorkerType, CostModel, ClusterConfig
│   ├── local.py              # Local mode
│   ├── aws_glue.py           # AWS Glue
│   ├── databricks.py         # Databricks
│   └── azure_synapse.py      # Azure Synapse Analytics
├── analysis/
│   ├── parser.py             # SparkCodeParser (AST)
│   ├── smell_detector.py     # CodeSmell detection rules
│   ├── recommender.py        # RecommendationEngine
│   └── models.py             # AnalysisResult, CodeSmell, CodeRecommendation
├── api/                      # FastAPI app + routes (health, optimize, platforms)
├── cli/                      # Typer CLI, wizard, formatters
└── data/                     # DataGenerator, DataProfiler, samplers
```

## Adding a New Platform

1. Create `src/spark_optima/platforms/<name>.py` inheriting from `Platform` (base.py)
2. Implement all abstract methods: `constraints`, `get_worker_types()`, `recommend_config()`, `translate_to_spark_config()`, `estimate_cost()`
3. Register it in `src/spark_optima/platforms/__init__.py` → `PLATFORM_REGISTRY`
4. Add test file at `tests/unit/platforms/test_<name>.py`
5. Add docs page at `docs/platforms/<name>.md`

## Adding a New Heuristic Rule

Rules live in `src/spark_optima/core/heuristics/rules.py`. Add a `HeuristicRuleDef` to the appropriate `_register_*_rules()` method:

```python
HeuristicRuleDef(
    param_name="spark.some.config",
    category=ParameterCategory.MEMORY,  # MEMORY, CPU, SHUFFLE, SQL, RUNTIME, etc.
    formula="some_var * 0.1",
    base_value="1g",
    priority="high",  # high / medium / low
    depends_on=["some_var"],
    conditions={"is_pyspark": True},  # optional — omit for universal rules
    description="What this rule does and why",
),
```

## Adding a Code Smell

Smell detection lives in `src/spark_optima/analysis/smell_detector.py`. Add a new `_detect_*` method and register it in `self._detection_rules`.

## Git Workflow

- Branch names: `feature/<description>`, `fix/<description>`, `docs/<description>`
- Commit messages: imperative form, explain the *why* not the *what*
- All CI checks must pass before merging (ruff, mypy, bandit, tests, coverage, lockfile)
- Run `uv lock --check` before pushing to ensure lockfile is up to date

## Dependencies

- Manage with `uv add <pkg>` (production) or `uv add --dev <pkg>` (dev)
- Update `pyproject.toml` and commit `uv.lock` together
- Optional dependencies (boto3, scikit-learn) must be guarded with `try/import` and tests must use `pytest.importorskip` or `pytest.mark.skipif`

## Docker

```bash
# Build production image
make docker-build

# Build development image
docker build -f docker/Dockerfile --target development -t spark-optima:dev .

# Run dev container
docker-compose -f docker/docker-compose.yml up spark-optima
```

Three Dockerfile stages: `builder` (deps only), `production` (non-root, tini), `development` (includes dev deps + git + Java).

## Documentation

```bash
# Serve locally
uv run mkdocs serve

# Build (strict — warnings become errors)
uv run mkdocs build --strict
```

Docs source: `docs/`. API reference is auto-generated from docstrings via `mkdocstrings[python]`. Always verify `mkdocs build --strict` passes before committing doc changes.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every PR and push to `main`/`develop`:

| Job | What it checks |
|-----|----------------|
| `lockfile-check` | `uv lock --check` — lockfile in sync |
| `lint-and-format` | ruff + mypy |
| `security` | bandit (fails on medium+ severity) |
| `test` | pytest on Python 3.10, 3.11, 3.12, 3.13 — **80% coverage enforced** |
| `integration-test` | end-to-end tests (on PR / main) |
| `docker-build` | Docker production image smoke test |
| `docs` | mkdocs build |

## Kubernetes

Manifests in `kubernetes/base/` (raw) and `kubernetes/helm/spark-optima/` (Helm chart). Production deployment guide: `kubernetes/PRODUCTION.md`.
