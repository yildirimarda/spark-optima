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
| 12 | v1.1 Improvements (EMR, history, new smells, exports, warm-start) | ✅ Done |

**Active:** v1.5 implementation (see "Backlog" at the bottom).

---

## Decisions & Design Choices

| Feature | Choice | Rationale |
|---------|--------|-----------|
| Optimization Algorithm | Hybrid (Heuristic + Bayesian) | Heuristics provide a warm start; Optuna refines from there |
| Run Mode | Simulation + Execution | Fast prediction for exploration, real runs for validation |
| Spark Versions | 3.x, 4.x | Broad version support; new versions loaded via config scraper |
| Platforms | Local, AWS Glue, AWS EMR, Databricks, Azure Synapse | Covers the primary managed and self-hosted targets |
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

---

## v1.1 Improvement Plan (2026-06-10)

Findings from a deep audit of the codebase (core pipeline, analysis module, platforms/exports, API/CLI/tests/CI). Work is split into five independent workstreams implemented in parallel.

### Workstream A — Code Analysis Upgrades

The smell detector covers 9 patterns but misses roughly half of the well-known PySpark anti-patterns, and `spark.sql("...")` strings are never inspected.

| # | Item | Detail | Status |
|---|------|--------|--------|
| A1 | Cartesian/cross join smell | Flag `crossJoin()` (HIGH severity) | ✅ Done |
| A2 | `toPandas()` smell | Driver OOM risk on large data (HIGH) | ✅ Done |
| A3 | `count()` emptiness check smell | `df.count() == 0` → recommend `isEmpty()` / `limit(1)` | ✅ Done |
| A4 | Single-partition write smell | `repartition(1)` / `coalesce(1)` before write | ✅ Done |
| A5 | `inferSchema=True` smell | Recommend explicit schema | ✅ Done |
| A6 | `withColumn` in loop smell | Track loop context in AST visitor (HIGH) | ✅ Done |
| A7 | `select("*")` smell | Column pruning recommendation | ✅ Done |
| A8 | `orderBy` without `limit` smell | Full-sort warning | ✅ Done |
| A9 | UDF discrimination | `pandas_udf` → MEDIUM, plain Python UDF → HIGH with pandas_udf recommendation | ✅ Done |
| A10 | Lightweight SQL string analysis | Inspect `spark.sql()` literals for `SELECT *` and `CROSS JOIN` | ✅ Done |
| A11 | Bug fix: skew detection skips ops with empty arguments | `smell_detector.py` | ✅ Done |
| A12 | Bug fix: `large_collect` smell has `location=None` | Extract line from AST node | ✅ Done |

### Workstream B — AWS EMR Platform Adapter

Local, Glue, Databricks, and Synapse are covered; EMR (one of the most common Spark targets) is missing.

| # | Item | Detail | Status |
|---|------|--------|--------|
| B1 | `platforms/aws_emr.py` | EMR on EC2: m5/r5/c5 worker types, YARN config translation, EC2 + EMR surcharge cost model | ✅ Done |
| B2 | Registry + validation wiring | `PLATFORM_REGISTRY`, `Optimizer` platform validation, API platform list | ✅ Done |
| B3 | Tests + docs | `tests/unit/platforms/test_aws_emr.py`, `docs/platforms/aws-emr.md`, mkdocs nav | ✅ Done |

### Workstream C — Optimization History & New CLI Commands

No persistence of optimization results exists; every run is lost. CLI lacks compare/explain/history.

| # | Item | Detail | Status |
|---|------|--------|--------|
| C1 | `core/history.py` | SQLite-backed `OptimizationHistory` (save/list/get/clear), auto-save from CLI `optimize` | ✅ Done |
| C2 | `spark-optima history` | List/show/clear past runs | ✅ Done |
| C3 | `spark-optima compare` | Diff two result JSON files: config deltas + metric deltas | ✅ Done |
| C4 | `spark-optima explain` | Per-parameter rationale from heuristic rule descriptions | ✅ Done |

### Workstream D — Export Formats & Heuristic Rules

| # | Item | Detail | Status |
|---|------|--------|--------|
| D1 | Airflow DAG export | Platform-aware operator snippet (SparkSubmitOperator / DatabricksSubmitRunOperator / GlueJobOperator) | ✅ Done |
| D2 | Kubernetes ConfigMap export | For spark-on-k8s deployments | ✅ Done |
| D3 | AWS EMR export | `aws emr create-cluster --configurations` JSON | ✅ Done |
| D4 | Speculation rules | `spark.speculation.*` conditioned on skew | ✅ Done |
| D5 | Data-aware dynamic allocation bounds | `maxExecutors` scaled from data size | ✅ Done |
| D6 | AQE fine-tuning rules | Skew factor + advisory partition size, data-aware | ✅ Done |

### Workstream E — Bayesian Optimizer Improvements

| # | Item | Detail | Status |
|---|------|--------|--------|
| E1 | Heuristic seed trial | Enqueue heuristic config as trial #1 so Optuna never regresses below the warm start | ✅ Done |
| E2 | Warm-start from stored study | When `storage_path` exists, resume/load past trials instead of starting cold | ✅ Done |

### Integration & Cleanup

| # | Item | Status |
|---|------|--------|
| I1 | Wire new export formats into CLI `export` command | ✅ Done |
| I2 | Consolidate duplicate `tests/unit/test_optimizer.py` into `tests/unit/core/test_optimizer.py` | ✅ Done |
| I3 | Quality gates: ruff, mypy, bandit, pytest (80%+ coverage), mkdocs build --strict | ✅ Done |
| I4 | Update CHANGELOG.md | ✅ Done |

## v1.2 Improvement Plan (2026-06-10)

Scope drawn from the v1.1 backlog. Five parallel workstreams.

### Workstream F — Spark Event Log Analyzer

Real metrics instead of stubs: parse Spark event logs to extract stage/task metrics, GC time, shuffle volumes, spill, and skew indicators, and feed them back into optimization.

| # | Item | Detail | Status |
|---|------|--------|--------|
| F1 | `core/execution/event_log.py` | `EventLogParser` for JSON-lines event logs (plain + gzip), `EventLogSummary` with stage/GC/shuffle/skew metrics | ✅ Done |
| F2 | Context bridge | Map summary onto heuristic context hints (data size, skew factor, large shuffles) | ✅ Done |
| F3 | `spark-optima analyze-log` CLI | Summary + tuning hints, `--output json` | ✅ Done |
| F4 | `optimize --event-log` | Enrich the heuristic context from a real past run | ✅ Done |
| F5 | Replace metrics stubs | `metrics_collector.py` GC/shuffle/CPU stubs populated from event log when available | ✅ Done |

### Workstream G — API: Async Jobs, Auth, Rate Limiting

| # | Item | Detail | Status |
|---|------|--------|--------|
| G1 | Async job API | `POST /api/v1/optimize/async` (202 + job id), `GET /api/v1/jobs/{id}`, `GET /api/v1/jobs` — thread-pool execution, in-memory job store | ✅ Done |
| G2 | API key auth | Optional `X-API-Key` enforcement via `SPARK_OPTIMA_API_KEYS` env (open when unset — current behavior) | ✅ Done |
| G3 | Rate limiting | In-memory per-client limiter, `SPARK_OPTIMA_RATE_LIMIT` env (req/min, 0 = disabled), 429 responses | ✅ Done |

### Workstream H — New Platform Adapters: Google Dataproc + Spark-on-K8s

| # | Item | Detail | Status |
|---|------|--------|--------|
| H1 | `platforms/gcp_dataproc.py` | n2 machine types, Compute Engine + Dataproc fee cost model, optional preemptible workers, `clusters.create` config export | ✅ Done |
| H2 | `platforms/spark_k8s.py` | Pod size presets, `spark.kubernetes.*` config translation, SparkApplication CRD (Spark Operator) export, user-provided $/vCPU-hour cost | ✅ Done |
| H3 | Wiring + tests + docs | Registry, optimizer, heuristic `applies_to`, docs pages, mkdocs nav (API enum/metadata wired at integration) | ✅ Done |

### Workstream J — Full SQL Analysis (sqlglot)

| # | Item | Detail | Status |
|---|------|--------|--------|
| J1 | `analysis/sql_analyzer.py` | sqlglot (spark dialect) AST analysis of `spark.sql()` literals | ✅ Done |
| J2 | SQL smells | select *, cartesian (explicit + comma joins), ORDER BY w/o LIMIT, UNION vs UNION ALL, leading-wildcard LIKE, IN (subquery) | ✅ Done |
| J3 | Integration | Replaces the v1.1 lightweight literal scan; findings flow through existing CodeSmell/recommendation pipeline | ✅ Done |

### Workstream K — Regional Pricing

| # | Item | Detail | Status |
|---|------|--------|--------|
| K1 | `platforms/pricing.py` | Curated static region multiplier tables per platform + `get_region_multiplier()` (default 1.0, warn on unknown) | ✅ Done |
| K2 | Wire `region` into cost | `estimate_cost()` in aws_glue / aws_emr / databricks / azure_synapse applies the multiplier; region + multiplier in breakdown | ✅ Done |

### Integration & Cleanup (v1.2)

| # | Item | Status |
|---|------|--------|
| I5 | API Platform enum + PLATFORM_METADATA for dataproc/k8s (after G & H land) | ✅ Done |
| I6 | CLI help text platform list | ✅ Done |
| I7 | Quality gates + end-to-end smoke + CHANGELOG | ✅ Done |

## v1.3 Improvement Plan (2026-06-10)

Scope drawn from the v1.2 backlog. Five parallel workstreams.

### Workstream L — ML Predictor End-to-End

`MLPerformancePredictor` exists but was never wired into the simulation pipeline. Make it a real surrogate model that learns from trial history.

| # | Item | Detail | Status |
|---|------|--------|--------|
| L1 | Feature extraction | Config + data-profile → numeric feature vector (shared, deterministic) | ✅ Done |
| L2 | Online training in SimulationEngine | Collect (features, predicted/measured time) per trial; train after N samples; blend analytical + ML predictions with confidence weighting | ✅ Done |
| L3 | Model persistence | Save/load via joblib under ~/.spark_optima/models/; scikit-learn stays an optional guarded dependency | ✅ Done |

### Workstream M — Performance Model Improvements

| # | Item | Detail | Status |
|---|------|--------|--------|
| M1 | GC time modeling | GC overhead as a function of memory pressure (data-to-heap ratio, GC algorithm from config) | ✅ Done |
| M2 | Network bandwidth | Shuffle transfer time bounded by per-node network throughput, not just disk | ✅ Done |
| M3 | Straggler/skew distribution | Task-time distribution model: stage time driven by the slowest partition under skew instead of a flat linear penalty | ✅ Done |

### Workstream N — Multi-Objective & Pareto Frontier

| # | Item | Detail | Status |
|---|------|--------|--------|
| N1 | CLI `--objective` (repeatable) on `optimize` | Pass objectives through to the Bayesian engine; multi-objective runs persist the Pareto frontier into result metadata | ✅ Done |
| N2 | Pareto export | Frontier → JSON/CSV via the export pipeline | ✅ Done |
| N3 | `spark-optima pareto -r result.json` | Rich table of frontier points + trade-off summary (no new plotting deps) | ✅ Done |

### Workstream O — REST API Reference Docs

| # | Item | Detail | Status |
|---|------|--------|--------|
| O1 | docs/user-guide/rest-api.md | All endpoints (sync, async jobs, platforms, health), auth + rate-limit env vars, request/response examples | ✅ Done |
| O2 | mkdocs nav + cross-links | Link from user-guide/api.md and index | ✅ Done |

### Workstream P — Persistent Job Store

| # | Item | Detail | Status |
|---|------|--------|--------|
| P1 | SQLite-backed JobStore | `SPARK_OPTIMA_JOB_STORE=memory|sqlite` (+ path env); jobs survive restarts and work across multi-process workers on one node | ✅ Done |
| P2 | Store selection + docs note | Factory keeps in-memory default; PRODUCTION.md note updated | ✅ Done |

## v1.4 Improvement Plan (2026-06-10)

Scope: the two remaining backlog items plus the highest-value gaps left from the v1.1 audits (missing CLI commands, stale wizard, History Server integration). Five parallel workstreams.

### Workstream Q — Live Pricing (opt-in, cached, fallback)

| # | Item | Detail | Status |
|---|------|--------|--------|
| Q1 | `platforms/live_pricing.py` | Azure Retail Prices API (public, no auth) + AWS Pricing API (guarded boto3); GCP stays static (API key required — documented) | ✅ Done |
| Q2 | Cache + fallback | `~/.spark_optima/pricing_cache.json`, 24h TTL; any failure → static tables, never raises | ✅ Done |
| Q3 | Opt-in wiring | `SPARK_OPTIMA_LIVE_PRICING=1` (default off); live rate replaces static baseline×multiplier in `estimate_cost`, breakdown labels the source | ✅ Done |

### Workstream R — Redis Job Store + Webhooks

| # | Item | Detail | Status |
|---|------|--------|--------|
| R1 | `RedisJobStore` | `SPARK_OPTIMA_JOB_STORE=redis` + `SPARK_OPTIMA_REDIS_URL`; guarded `redis` import (no new hard deps); same BaseJobStore contract | ✅ Done |
| R2 | Webhooks | optional `webhook_url` on `POST /optimize/async` → POST result on completion/failure (httpx, retries, SSRF guard) | ✅ Done |

### Workstream S — Config Validate / Import / Templates

| # | Item | Detail | Status |
|---|------|--------|--------|
| S1 | `spark-optima validate` | Validate a spark-defaults.conf / JSON config against the parameter DB + platform constraints + anti-pattern checks | ✅ Done |
| S2 | `spark-optima import` | Import an existing config, run optimization, diff current vs recommended | ✅ Done |
| S3 | Workload templates | Curated baselines (etl-batch, streaming, ml-training, interactive) in `data/templates/*.yaml` + `spark-optima templates list/show` | ✅ Done |

### Workstream U — Spark History Server Client

| # | Item | Detail | Status |
|---|------|--------|--------|
| U1 | `core/execution/history_server.py` | httpx client for the History Server REST API (`/api/v1/applications/...`) producing the same summary + tuning hints as the v1.2 event-log parser | ✅ Done |
| U2 | CLI wiring | `analyze-log --history-server URL --app-id ID` (wired at integration) | ✅ Done |

### Workstream W — Wizard Refresh

| # | Item | Detail | Status |
|---|------|--------|--------|
| W1 | Wizard catch-up | Surface v1.1–v1.3 features: optimization objectives step, optional event-log path, current export formats; platforms stay dynamic | ✅ Done |

### Integration & Cleanup (v1.4)

| # | Item | Status |
|---|------|--------|
| I8 | history-server option into analyze-log; docs (cli.md/rest-api.md) updates | ✅ Done |
| I9 | Quality gates + end-to-end smoke + CHANGELOG | ✅ Done |

## v1.5 Improvement Plan (2026-06-11)

Scope: the four v1.5 backlog items plus an examples/docs refresh. Five parallel workstreams.

### Workstream X — Scala Code Analysis

| # | Item | Detail | Status |
|---|------|--------|--------|
| X1 | `analysis/scala_parser.py` | Lightweight lexer-based Scala Spark parser producing the existing SparkOperation models (comment/string masking, val-assignment tracking, chained calls, triple-quoted strings) | ✅ Done |
| X2 | Smell coverage for Scala | Operation-based smells work as-is; Python-AST-specific detectors skip gracefully; `spark.sql("...")` strings reuse the sqlglot analyzer; new `groupbykey_usage` smell | ✅ Done |
| X3 | CLI language detection | `analyze`/`optimize` accept `.scala` files | ✅ Done |

### Workstream Y — GCP Live Pricing

| # | Item | Detail | Status |
|---|------|--------|--------|
| Y1 | Cloud Billing Catalog client | `SPARK_OPTIMA_GCP_API_KEY`-gated; N2 core/RAM SKU rates per region → hourly machine price; same cache/fallback rules | ✅ Done |
| Y2 | Dataproc wiring | Live compute rate replaces static baseline×multiplier (Dataproc fee stays $0.01/vCPU-h); `pricing_source` labeling | ✅ Done |

### Workstream Z — SSE Progress + Templates API

| # | Item | Detail | Status |
|---|------|--------|--------|
| Z1 | Trial progress plumbing | Optional `progress_callback` on BayesianOptimizer/Optimizer (per-trial: n/total, best value) — additive | ✅ Done |
| Z2 | Job progress + SSE | `progress` on job records (all 3 stores); `GET /api/v1/jobs/{id}/events` — text/event-stream polling the store until terminal state | ✅ Done |
| Z3 | Templates API | `GET /api/v1/templates`, `GET /api/v1/templates/{name}` (parity with the CLI) | ✅ Done |

### Workstream AA — Config DB Unit Normalization

| # | Item | Detail | Status |
|---|------|--------|--------|
| AA1 | Audit + normalize | BYTES/DURATION params with mixed-unit min/max in database.py + data/configs/*.yaml normalized to one canonical convention the validator actually compares | ✅ Done |
| AA2 | Validator + regression tests | Range checks correct for unit-suffixed values; CLI `validate` re-enables numeric range checks for BYTES/DURATION (wired at integration) | ✅ Done |

### Workstream BB — Examples & Docs Refresh

| # | Item | Detail | Status |
|---|------|--------|--------|
| BB1 | New examples | event-log analysis, multi-objective/Pareto, EMR/Dataproc/K8s platform examples, templates usage | ✅ Done |
| BB2 | README + getting-started refresh | Command catalogue (history/compare/explain/analyze-log/validate/import/templates/pareto), env-var table, feature list current | ✅ Done |

### Integration & Cleanup (v1.5)

| # | Item | Status |
|---|------|--------|
| I10 | CLI validate range-check re-enable; review-finding fixes; quality gates + smoke + CHANGELOG | ✅ Done |

### Backlog (v1.6+) — identified but deliberately deferred

- **Java code analysis** — v1.5 adds Scala; Java Spark sources remain unsupported
- **API validate endpoint** — needs the CLI validate logic extracted into a core module first


