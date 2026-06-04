# Spark Optima - Project Plan

## Overview

**Spark Optima** is an intelligent configuration optimization tool for Apache Spark applications. It uses a hybrid Heuristic + Bayesian approach to find the optimal Spark configuration for a given workload, platform, and resource budget — without manual trial-and-error.

The tool collects the user's Spark code, target platform, resource constraints, data characteristics, and optional sample data, then outputs the best configuration along with code improvement suggestions.

---

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project Foundation & Architecture | ✅ Done |
| 2 | Spark Configuration Knowledge Base | ✅ Done |
| 3 | Platform Resource Models | ✅ Done |
| 4 | Heuristic Optimization Engine | ✅ Done |
| 5 | Bayesian Optimization Engine (Optuna) | ✅ Done |
| 6 | Simulation & Execution Engine | ✅ Done |
| 7 | Code Analysis Module | ✅ Done |
| 8 | CLI & API Interface | ✅ Done |
| 9 | Testing & Quality Assurance | ✅ Done |
| 10 | Kubernetes & Production | ✅ Done |
| 11 | Documentation & Examples | ✅ Done |
| — | UV Migration (Poetry → UV) | ✅ Done |

**Active:** Post-audit improvements (see below).

---

## Decisions & Design Choices

| Feature | Choice | Rationale |
|---------|--------|-----------|
| Optimization Algorithm | Hybrid (Heuristic + Bayesian) | Heuristics provide a warm start; Optuna refines from there |
| Run Mode | Simulation + Execution | Fast prediction for exploration, real runs for validation |
| Spark Versions | 3.x, 4.x | Broad version support; new versions loaded via config scraper |
| Platforms | Local, AWS Glue, Databricks, Azure Synapse | Covers the primary managed and self-hosted targets |
| Language Support | Python (Phase 1) | Scala support planned for a future release |
| Architecture | Modular plugin-based | Platform adapters and optimization strategies are swappable |

---

## System Architecture

### Directory Layout

```
spark-optima/
├── src/spark_optima/
│   ├── core/
│   │   ├── bayesian/          # Optuna-based optimizer, search space, trial runner
│   │   ├── config_engine/     # Parameter database, loader, validator
│   │   ├── execution/         # Real Spark run engine, metrics collector, monitor
│   │   ├── heuristics/        # Rule engine, context, evaluator
│   │   ├── simulation/        # Performance model, predictor
│   │   ├── optimizer.py       # Top-level hybrid optimizer
│   │   └── result.py
│   ├── analysis/              # AST parser, smell detector, recommender
│   ├── api/                   # FastAPI app, routes, models
│   ├── cli/                   # Typer CLI, interactive wizard, formatters
│   ├── data/                  # Sample data generators, profiler, samplers
│   └── platforms/             # Local, AWS Glue, Databricks, Azure Synapse adapters
├── tests/
│   ├── unit/                  # 48 test modules
│   └── integration/
├── docker/
├── kubernetes/                # Base manifests + Helm chart
├── docs/                      # MkDocs site
└── examples/                  # 9 example scripts
```

### Optimization Flow

```
+-------------------------------------------------------------+
|  Phase 1: Input Collection                                  |
|  - User Spark code (Python)                                 |
|  - Platform selection (Local / AWS Glue / Databricks /      |
|    Azure Synapse)                                           |
|  - Resource constraints (memory, CPU, cost limits)          |
|  - Data characteristics (size, format, schema)              |
|  - Sample data (optional)                                   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  Phase 2: Code Analysis                                     |
|  - Parse Spark code via AST                                 |
|  - Detect code smells (broadcast hints, caching, UDFs,      |
|    shuffle overhead, skew risk)                             |
|  - Identify data operations (shuffle, join, aggregation)    |
|  - Generate code improvement recommendations                |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  Phase 3: Heuristic Initial Config                          |
|  - Memory heuristics (driver / executor / overhead)         |
|  - CPU / core heuristics (parallelism = 2-3x cores)         |
|  - Shuffle heuristics (spill, compression)                  |
|  - Serialization heuristics (Kryo vs Java)                  |
|  - Platform-specific rules                                  |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  Phase 4: Bayesian Optimization (Optuna)                    |
|  - Define search space from heuristic seed config           |
|  - Run trials (Simulation mode or Execution mode)           |
|  - Evaluate objective (minimize runtime / cost / OOM risk)  |
|  - Converge to optimal configuration                        |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  Phase 5: Output & Export                                   |
|  - Optimal Spark configuration                              |
|  - Code improvement suggestions                             |
|  - Performance prediction                                   |
|  - Platform-specific export (JSON, YAML, native UI config)  |
+-------------------------------------------------------------+
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Dependency Management | UV (astral-sh/uv) |
| Optimization | Optuna (Bayesian), Custom Heuristics |
| API Framework | FastAPI |
| CLI Framework | Typer |
| AST Parsing | ast, astor |
| Data Processing | Pandas, PyArrow |
| Configuration | Pydantic, PyYAML |
| Testing | pytest, pytest-cov, pytest-asyncio |
| Linting | ruff, mypy, bandit |
| Documentation | MkDocs, Material theme |
| Container | Docker, Docker Compose |
| Orchestration | Kubernetes, Helm |
| CI/CD | GitHub Actions |

---

## UV Migration

Migrated from Poetry to UV (astral-sh/uv) on 2026-06-04.

| Item | Before | After |
|------|--------|-------|
| Lock file | `poetry.lock` | `uv.lock` |
| Config | `[tool.poetry]` + `poetry.toml` | PEP 621 `[project]` |
| Install | `poetry install` | `uv sync` |
| Run | `poetry run <cmd>` | `uv run <cmd>` |
| Add dep | `poetry add <pkg>` | `uv add <pkg>` |
| Build | `poetry build` | `uv build` |
| CI | `snok/install-poetry@v1` | `astral-sh/setup-uv@v5` |
| Docker | pip + poetry export | `ghcr.io/astral-sh/uv` layer |

---

## Post-Audit Improvements

### Round 1 — Structural fixes (2026-06-04)

| # | Issue | File | Severity | Status |
|---|-------|------|----------|--------|
| 1 | Dockerfile missing `development` stage (referenced by docker-compose.yml) | `docker/Dockerfile` | High | ✅ Fixed |
| 2 | Redundant `pip install pyspark` in production stage (already in venv) | `docker/Dockerfile` | Medium | ✅ Fixed |
| 3 | Missing `[tool.bandit]` section (pre-commit hook uses `-c pyproject.toml`) | `pyproject.toml` | Medium | ✅ Fixed |
| 4 | Broken mkdocs nav link to `development/contributing.md` (file missing) | `mkdocs.yml` + `docs/` | Medium | ✅ Fixed |
| 5 | Missing `.python-version` file | project root | Low | ✅ Fixed |
| 6 | `requirements.txt` contained `-e .` editable entry (inappropriate for Docker) | `requirements.txt` | Low | ✅ Fixed |

### Round 2 — Deep audit fixes (2026-06-04)

| # | Issue | File | Severity | Status |
|---|-------|------|----------|--------|
| 7 | `azure_synapse.py` missing `Path` import and `logger` → runtime `NameError` | `platforms/azure_synapse.py` | Critical | ✅ Fixed |
| 8 | `engine.py` `ResourceSpec` in `TYPE_CHECKING` but used at runtime (TC004) | `core/execution/engine.py` | High | ✅ Fixed |
| 9 | `local.py` driver+executor memory exceeded available RAM | `platforms/local.py` | High | ✅ Fixed |
| 10 | PySpark overhead factor 10% for all workloads (should be 25% for PySpark) | `core/heuristics/rules.py` | High | ✅ Fixed |
| 11 | 5 failing tests: boto3 skip, CLI exit code, ML model error message | `tests/` | High | ✅ Fixed |
| 12 | `mkdocstrings[python]` not installed → `mkdocs build` failed | `pyproject.toml` | High | ✅ Fixed |
| 13 | API docs referenced wrong class names (`HeuristicContext`, `CodeAnalysis`, etc.) | `docs/api/core.md` | Medium | ✅ Fixed |
| 14 | 4 broken internal links in docs (strict build failed) | `docs/` | Medium | ✅ Fixed |
| 15 | CI: no coverage threshold, bandit always passed, missing Python 3.10, no lockfile check | `.github/workflows/ci.yml` | Medium | ✅ Fixed |
| 16 | `mypy python_version = "3.10"` mismatched with 3.12 Docker/CI | `pyproject.toml` | Medium | ✅ Fixed |
| 17 | Missing GC tuning rules (G1GC) and spill config rules | `core/heuristics/rules.py` | Medium | ✅ Fixed |
| 18 | `smell_detector.py` skew detection false positives + missing `collect()` smell | `analysis/smell_detector.py` | Medium | ✅ Fixed |
| 19 | `scikit-learn` not in dev deps → ML prediction disabled, tests skipping | `pyproject.toml` | Low | ✅ Fixed |

---

## Spark Configuration Parameters (Key Categories)

```yaml
memory:
  - spark.driver.memory
  - spark.executor.memory
  - spark.executor.memoryOverhead
  - spark.memory.fraction
  - spark.memory.storageFraction
  - spark.sql.adaptive.enabled
  - spark.sql.adaptive.coalescePartitions.enabled

cpu:
  - spark.executor.cores
  - spark.default.parallelism
  - spark.sql.shuffle.partitions
  - spark.scheduler.mode

shuffle:
  - spark.shuffle.file.buffer
  - spark.shuffle.spill.compress
  - spark.shuffle.spill.diskWriteBufferSize
  - spark.reducer.maxSizeInFlight
  - spark.shuffle.compress

serialization:
  - spark.serializer
  - spark.kryo.registrator
  - spark.kryoserializer.buffer.max

sql:
  - spark.sql.adaptive.skewJoin.enabled
  - spark.sql.autoBroadcastJoinThreshold
  - spark.sql.broadcastTimeout
  - spark.sql.files.maxPartitionBytes
  - spark.sql.files.openCostInBytes

dynamic_allocation:
  - spark.dynamicAllocation.enabled
  - spark.dynamicAllocation.minExecutors
  - spark.dynamicAllocation.maxExecutors
  - spark.dynamicAllocation.initialExecutors
```

The full parameter database (200+ entries, covering Spark 3.x and 4.x) lives in `src/spark_optima/core/config_engine/database.py`. New Spark releases can be loaded by updating the config scraper in `core/config_engine/loader.py`.
