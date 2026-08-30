# Spark Optima - Project Plan

## Overview

**Spark Optima** is an intelligent configuration optimization tool for Apache
Spark applications. It uses a hybrid Heuristic + Bayesian approach to find the
optimal Spark configuration for a given workload, platform, and resource
budget — without manual trial-and-error.

The tool collects the user's Spark code, target platform, resource constraints,
data characteristics, and optional sample data, then outputs the best
configuration along with code improvement suggestions.

---

## Decisions & Design Choices

| Feature | Choice | Rationale |
|---------|--------|-----------|
| Optimization Algorithm | Hybrid (Heuristic + Bayesian) | Heuristics provide a warm start; Optuna refines from there |
| Run Mode | Simulation + Execution | Fast prediction for exploration, real runs for validation |
| Spark Versions | 3.x, 4.x | Broad version support; new versions loaded via config scraper |
| Platforms | Local, AWS Glue, AWS EMR, Databricks, Azure Synapse, GCP Dataproc, Spark-on-K8s | Covers the primary managed and self-hosted targets |
| Language Support | Python, Scala | Python from v1.0; Scala added in v1.5. Java support is on the roadmap |
| Architecture | Modular plugin-based | Platform adapters and optimization strategies are swappable |

---

## System Architecture

### Directory Layout

```
spark-optima/
├── src/spark_optima/
│   ├── core/
│   │   ├── bayesian/          # Optuna-based optimizer, search space, trial runner
│   │   ├── config_engine/     # Parameter database, loader, validator, unit helpers
│   │   ├── execution/         # Real Spark run engine, metrics collector, event log
│   │   │                      # parser, History Server client, monitor
│   │   ├── heuristics/        # Rule engine, context, evaluator
│   │   ├── simulation/        # Performance model (GC/network/skew), ML predictor
│   │   ├── history.py         # SQLite-backed optimization history
│   │   ├── optimizer.py       # Top-level hybrid optimizer
│   │   ├── result.py          # Result model + export formats
│   │   └── templates.py       # Workload templates (etl-batch, streaming, ...)
│   ├── analysis/              # AST parser, smell detector, recommender,
│   │                          # sqlglot SQL analyzer, Scala lexer parser
│   ├── api/                   # FastAPI app, routes, in-memory/sqlite/redis
│   │                          # job stores, dependencies, webhooks, security
│   ├── cli/                   # Typer CLI, interactive wizard, formatters
│   ├── data/                  # Sample data generators, profiler, samplers
│   └── platforms/             # Local, AWS Glue, AWS EMR, Databricks,
│                              # Azure Synapse, GCP Dataproc, Spark-on-K8s,
│                              # regional pricing, opt-in live pricing
├── tests/
│   ├── unit/                  # test_*.py modules per area
│   └── integration/
├── docker/
├── kubernetes/                # Base manifests + Helm chart
├── docs/                      # MkDocs site
└── examples/                  # basic + advanced + per-platform scripts
```

### Optimization Flow

```
+-------------------------------------------------------------+
|  Phase 1: Input Collection                                  |
|  - User Spark code (Python or Scala)                        |
|  - Platform selection (Local / AWS Glue / AWS EMR /         |
|    Databricks / Azure Synapse / GCP Dataproc / Spark-on-K8s)|
|  - Resource constraints (memory, CPU, cost limits)          |
|  - Data characteristics (size, format, schema)              |
|  - Sample data (optional)                                   |
|  - Optional event log or History Server URL                 |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  Phase 2: Code Analysis                                     |
|  - Parse Spark code via AST (Python) or lexer (Scala)      |
|  - Detect code smells (broadcast hints, caching, UDFs,      |
|    shuffle overhead, skew risk, +9 patterns from v1.1,      |
|    groupByKey from v1.5)                                    |
|  - SQL literal analysis via sqlglot (spark dialect)         |
|  - Generate code improvement recommendations                |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  Phase 3: Heuristic Initial Config                          |
|  - Memory heuristics (driver / executor / overhead,         |
|    PySpark-aware 25% overhead)                              |
|  - CPU / core heuristics (parallelism = 2-3x cores)         |
|  - Shuffle heuristics (spill, compression)                  |
|  - Serialization heuristics (Kryo vs Java)                  |
|  - GC tuning (G1GC) + platform-specific rules               |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  Phase 4: Bayesian Optimization (Optuna)                    |
|  - Enqueue heuristic config as trial #1 (warm start)        |
|  - Optionally resume from JournalStorage (storage_path)     |
|  - Run trials (Simulation mode or Execution mode)           |
|  - Optional progress_callback for live progress / SSE       |
|  - Single or multi-objective (Pareto frontier persisted)    |
|  - ML surrogate blends with analytical prediction when      |
|    enough samples are available                             |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  Phase 5: Output & Export                                   |
|  - Optimal Spark configuration                              |
|  - Code improvement suggestions                             |
|  - Performance prediction                                   |
|  - Multi-format export (JSON, YAML, Airflow DAG,            |
|    Kubernetes ConfigMap, EMR --configurations, native UI    |
|    config, pareto-json, pareto-csv)                         |
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
| AST Parsing | ast, astor; sqlglot (SQL); custom Scala lexer |
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

The full parameter database (200+ entries, covering Spark 3.x and 4.x) lives in
`src/spark_optima/core/config_engine/database.py`. New Spark releases can be
loaded by updating the config scraper in `core/config_engine/loader.py`.

---

## Milestone 1: Project Foundation & Architecture

Initial scaffolding, packaging, dependency management, code-style tooling, and
the core plugin/registry abstractions that everything else is built on.

- [x] Python package layout under `src/spark_optima/` (core, analysis, api,
      cli, data, platforms)
- [x] UV-based dependency management with PEP 621 `[project]` metadata
- [x] ruff, mypy, bandit quality gates wired into CI
- [x] Pydantic settings + typed config throughout
- [x] Docker image, docker-compose, Kubernetes manifests, Helm chart
- [x] MkDocs site with Material theme

## Milestone 2: Spark Configuration Knowledge Base

- [x] Parameter database (200+ entries) covering Spark 3.x and 4.x
- [x] Version loader / config scraper for adding new Spark releases
- [x] Validator with parameter-type, range, and unit-aware checks
- [x] Curated YAML configs in `data/configs/spark_{3.0..4.1}_configs.yaml`

## Milestone 3: Platform Resource Models

- [x] `platforms/local.py` — single-node, executor + driver sized to RAM
- [x] `platforms/aws_glue.py` — G.x workers, DPU cost model
- [x] `platforms/aws_emr.py` — m5/r5/c5 workers, YARN translation, EMR surcharge
- [x] `platforms/databricks.py` — per-cloud region defaulting, DBU pricing
- [x] `platforms/azure_synapse.py` — Spark pool sizing, DWU cost model
- [x] `platforms/gcp_dataproc.py` — n2 machine types, Dataproc fee pricing
- [x] `platforms/spark_k8s.py` — pod size presets, K8s config translation
- [x] `platforms/pricing.py` — regional multiplier tables + breakdown labels
- [x] `platforms/live_pricing.py` — opt-in Azure/AWS/GCP live pricing with
      24h cache and static fallback

## Milestone 4: Heuristic Optimization Engine

- [x] Rule engine (`core/heuristics/engine.py`) and `HeuristicContext`
- [x] Memory heuristics: driver / executor / overhead with PySpark-aware 25%
      overhead
- [x] CPU / parallelism heuristics (parallelism = 2-3x cores)
- [x] Shuffle heuristics (spill, compression, partition sizing)
- [x] Serialization heuristics (Kryo vs Java)
- [x] GC tuning rules (G1GC) and platform-specific rules
- [x] Speculation rules (skew-conditioned)
- [x] Data-aware `spark.dynamicAllocation.maxExecutors`
- [x] AQE fine-tuning (advisory partition size, skew factor override)

## Milestone 5: Bayesian Optimization Engine (Optuna)

- [x] Optuna-based optimizer with search space derived from heuristic seed
- [x] Heuristic config enqueued as trial #1 (warm start)
- [x] `JournalStorage` resume from `storage_path`
- [x] Per-trial `progress_callback` plumbed through `Optimizer`
- [x] Multi-objective support (Pareto frontier persisted into metadata)
- [x] Optional pruners and trial runner

## Milestone 6: Simulation & Execution Engine

- [x] `core/simulation/performance_model.py` — analytical runtime model
- [x] GC overhead modeled from memory pressure (G1GC relief)
- [x] Shuffle transfer bounded by per-node network bandwidth (10 Gbit/node)
- [x] Straggler / skew model (AQE skew mitigation caps effective skew)
- [x] `core/simulation/predictor.py` — ML surrogate, online training,
      R²-gated blend with analytical prediction
- [x] Joblib model persistence under `SPARK_OPTIMA_MODEL_DIR`
- [x] `core/execution/engine.py` — real Spark run engine
- [x] `core/execution/metrics_collector.py` — GC/shuffle/CPU metrics
      populated from event logs when available
- [x] `core/execution/event_log.py` — JSON-lines event log parser
      (plain + gzip) with stage/GC/shuffle/skew summary
- [x] `core/execution/history_server.py` — httpx client for the History
      Server REST API producing the same summary + tuning hints
- [x] `core/execution/monitor.py` and `spark_runner.py`

## Milestone 7: Code Analysis Module

- [x] Python AST parser (`analysis/parser.py`) producing
      `SparkOperation` models with location + loop context
- [x] Scala lexer-based parser (`analysis/scala_parser.py`) — comment/string
      masking, val lineage tracking, multi-line fluent chains
- [x] `analysis/smell_detector.py` — 9 baseline + 9 v1.1 smells + new
      `groupbykey_usage` smell (Python and Scala)
- [x] `analysis/sql_analyzer.py` — sqlglot (spark dialect) AST analysis
      of `spark.sql()` literals
- [x] `analysis/recommender.py` — converts smells into actionable
      recommendations
- [x] Language detection in CLI `analyze`/`optimize` for `.scala` files

## Milestone 8: CLI & API Interface

- [x] Typer CLI with subcommands: `optimize`, `analyze`, `analyze-log`,
      `platforms`, `wizard`, `export`, `pareto`, `history`, `compare`,
      `explain`, `validate`, `import`, `templates`, `version`
- [x] Rich-formatted tables, panels, and stderr-routed output for
      machine-readable modes
- [x] Interactive wizard with objectives, event-log, and export-format
      steps
- [x] FastAPI app with v1 routes: `optimize`, `optimize/async`, `jobs`,
      `jobs/{id}/events` (SSE), `platforms`, `templates`, `health`
- [x] X-API-Key auth via `SPARK_OPTIMA_API_KEYS` (off by default)
- [x] Per-client rate limiting via `SPARK_OPTIMA_RATE_LIMIT` (off by default)
- [x] In-memory, SQLite, and Redis job stores selectable via
      `SPARK_OPTIMA_JOB_STORE`
- [x] Webhooks on async job completion/failure (httpx + SSRF guard)

## Milestone 9: Testing & Quality Assurance

- [x] `pytest` + `pytest-cov` + `pytest-asyncio` unit tests (2805 passing)
- [x] Coverage threshold enforced in CI
- [x] ruff lint + format, mypy strict-ish, bandit security checks
- [x] CI matrix on Python 3.10/3.11/3.12 with lockfile check
- [x] mkdocs build --strict
- [x] Optional-dep tests skip cleanly when boto3 / redis / pyspark missing

## Milestone 10: Kubernetes & Production

- [x] Dockerfile with `development` and production stages
- [x] docker-compose for local stack
- [x] Kubernetes base manifests
- [x] Helm chart
- [x] PRODUCTION.md notes covering job-store selection, rate limits,
      auth, and live-pricing flags
- [x] Container image CI smoke test

## Milestone 11: Documentation & Examples

- [x] MkDocs site (installation, getting-started, configuration, CLI,
      REST API, platform guides, development)
- [x] REST API reference with auth + rate-limit + job-store env vars
      and full async flow
- [x] Per-platform docs pages (local, AWS Glue, AWS EMR, Databricks,
      Azure Synapse, GCP Dataproc, Spark-on-K8s)
- [x] Example scripts: `examples/basic/*`, `examples/advanced/*`,
      `examples/platforms/*`, `examples/data/generate_sample_data.py`
- [x] README command catalogue and environment-variable table
- [x] CHANGELOG entries per release

## Milestone 12: v1.1 — Code Analysis, EMR, History, Exports, Warm Start

- [x] A1: cartesian/cross-join smell (HIGH)
- [x] A2: `toPandas()` smell (HIGH)
- [x] A3: `count()`-for-emptiness smell
- [x] A4: `repartition(1)` / `coalesce(1)` before write
- [x] A5: `inferSchema=True` smell
- [x] A6: `withColumn` in loop (HIGH)
- [x] A7: `select("*")` smell
- [x] A8: `orderBy` without `limit` smell
- [x] A9: UDF discrimination (`pandas_udf` MEDIUM, plain UDF HIGH)
- [x] A10: lightweight `spark.sql()` literal scan for `SELECT *` /
      `CROSS JOIN` (replaced by sqlglot in v1.2 J1)
- [x] A11: skew detection skips ops with empty arguments (bug fix)
- [x] A12: `large_collect` smell extracts `location` from AST (bug fix)
- [x] B1: `platforms/aws_emr.py`
- [x] B2: EMR registered in `PLATFORM_REGISTRY`, optimizer validation,
      API platform list
- [x] B3: EMR tests + `docs/platforms/aws-emr.md` + mkdocs nav
- [x] C1: `core/history.py` SQLite-backed `OptimizationHistory`
- [x] C2: `spark-optima history` (list / `--show` / `--clear`)
- [x] C3: `spark-optima compare` — diff two result JSON files
- [x] C4: `spark-optima explain` — per-parameter rationale from rules
- [x] D1: Airflow DAG export
- [x] D2: Kubernetes ConfigMap export
- [x] D3: AWS EMR `--configurations` JSON export
- [x] D4: speculation rules conditioned on skew
- [x] D5: data-aware `spark.dynamicAllocation.maxExecutors`
- [x] D6: AQE fine-tuning (skew factor + advisory partition size)
- [x] E1: heuristic seed enqueued as trial #1
- [x] E2: warm-start from `JournalStorage` on `storage_path`
- [x] I1: new export formats wired into CLI `export`
- [x] I2: consolidate `tests/unit/test_optimizer.py` into
      `tests/unit/core/test_optimizer.py`
- [x] I3: quality gates (ruff, mypy, bandit, pytest 80%+, mkdocs
      --strict)
- [x] I4: CHANGELOG entry

## Milestone 13: v1.2 — Event Log, Async API, New Platforms, SQL, Pricing

- [x] F1: `core/execution/event_log.py` `EventLogParser` and
      `EventLogSummary`
- [x] F2: context bridge from event log to heuristic hints
- [x] F3: `spark-optima analyze-log` CLI (`--output json`)
- [x] F4: `optimize --event-log` enriches the heuristic context
- [x] F5: `metrics_collector.collect_from_event_log` replaces stubs
- [x] G1: async job API (`POST /optimize/async`, `GET /jobs/{id}`,
      `GET /jobs`) with in-memory job store
- [x] G2: X-API-Key auth via `SPARK_OPTIMA_API_KEYS`
- [x] G3: per-client rate limiting via `SPARK_OPTIMA_RATE_LIMIT`
      (429 + Retry-After)
- [x] H1: `platforms/gcp_dataproc.py` (n2 machine types, optional
      preemptibles, `clusters.create` export)
- [x] H2: `platforms/spark_k8s.py` (pod presets, `spark.kubernetes.*`
      translation, SparkApplication CRD export)
- [x] H3: registry, optimizer, heuristic `applies_to`, docs + nav
- [x] J1: `analysis/sql_analyzer.py` sqlglot-based analysis
- [x] J2: SQL smells (select *, cartesian, ORDER BY w/o LIMIT,
      UNION vs UNION ALL, leading-wildcard LIKE, IN subquery)
- [x] J3: SQL analyzer replaces the v1.1 substring scan
- [x] K1: `platforms/pricing.py` curated region multiplier tables
- [x] K2: `region` wired into `estimate_cost` for the cloud adapters
- [x] I5: API `Platform` enum + `PLATFORM_METADATA` for dataproc/k8s
- [x] I6: CLI help text platform list
- [x] I7: quality gates + end-to-end smoke + CHANGELOG

## Milestone 14: v1.3 — ML Predictor, Performance Model, Pareto, REST Docs

- [x] L1: shared deterministic feature extraction
- [x] L2: online training in `SimulationEngine`, R²-gated blend of
      analytical + ML predictions
- [x] L3: joblib model persistence under `~/.spark_optima/models/`
- [x] M1: GC time modeled from memory pressure (G1GC relief)
- [x] M2: shuffle transfer bounded by per-node network bandwidth
- [x] M3: straggler/skew distribution model with AQE mitigation cap
- [x] N1: CLI `--objective` (repeatable) on `optimize`; multi-objective
      runs persist the Pareto frontier
- [x] N2: Pareto export to JSON/CSV
- [x] N3: `spark-optima pareto -r result.json` with trade-off summary
- [x] O1: `docs/user-guide/rest-api.md`
- [x] O2: mkdocs nav + cross-links
- [x] P1: SQLite-backed `JobStore` via `SPARK_OPTIMA_JOB_STORE=sqlite`
- [x] P2: store selection + PRODUCTION.md note

## Milestone 15: v1.4 — Live Pricing, Redis, Validate/Import/Templates, History Server

- [x] Q1: `platforms/live_pricing.py` (Azure Retail Prices, AWS Pricing,
      GCP Cloud Billing)
- [x] Q2: 24h cache + static fallback, never raises
- [x] Q3: opt-in `SPARK_OPTIMA_LIVE_PRICING=1`, breakdown labels
      `pricing_source: live|static`
- [x] R1: `RedisJobStore` via `SPARK_OPTIMA_JOB_STORE=redis` +
      `SPARK_OPTIMA_REDIS_URL`
- [x] R2: webhooks on async job completion/failure (10s timeout,
      3 attempts, SSRF guard)
- [x] S1: `spark-optima validate` (parameter DB + platform + anti-pattern
      checks)
- [x] S2: `spark-optima import` (import + optimize + diff)
- [x] S3: workload templates in `data/templates/*.yaml` and
      `spark-optima templates list/show`
- [x] U1: `core/execution/history_server.py` httpx client
- [x] U2: `analyze-log --history-server URL [--app-id ID]`
- [x] W1: wizard catch-up (objectives, event log, exports, dynamic
      platforms)
- [x] I8: history-server option into `analyze-log`; CLI / REST API
      doc updates
- [x] I9: quality gates + smoke + CHANGELOG

## Milestone 16: v1.5 — Scala, GCP Live Pricing, SSE, Config Unit Normalization

- [x] X1: `analysis/scala_parser.py` lexer-based Scala Spark parser
- [x] X2: Scala smell coverage + new `groupbykey_usage` smell
- [x] X3: CLI language detection for `.scala` files
- [x] Y1: GCP Cloud Billing Catalog client (API-key gated)
- [x] Y2: Dataproc live pricing wired into `estimate_cost`
- [x] Z1: per-trial `progress_callback` on BayesianOptimizer/Optimizer
- [x] Z2: `progress` on job records (all 3 stores) +
      `GET /api/v1/jobs/{id}/events` SSE
- [x] Z3: `GET /api/v1/templates` and `GET /api/v1/templates/{name}`
- [x] AA1: BYTES/DURATION bounds audited and normalized across all 8
      Spark config YAMLs
- [x] AA2: validator + regression tests + CLI `validate` re-enables
      numeric range checks
- [x] BB1: new examples (event-log analysis, multi-objective/Pareto,
      EMR/Dataproc/K8s, templates)
- [x] BB2: README + getting-started refresh (command catalogue,
      env-var table)
- [x] I10: CLI `validate` range-check re-enable; review-finding fixes;
      quality gates + smoke + CHANGELOG

## Milestone 17: v1.6 — Backlog

These items were identified during v1.0–v1.5 and explicitly deferred. They
are real gaps, not nice-to-haves.

- [x] Add Java code analysis. Java is the third major Spark source
      language. v1.5 added Scala; the lexer-based approach in
      `analysis/scala_parser.py` is a reasonable template. Java-specific
      concerns include checked-exception handling, `SparkSession.builder`
      chaining, and Java UDFs (`org.apache.spark.api.java.function.MapFunction`).
- [ ] Remove hardcoded version strings from tests: no test may assert a literal
      package version. Read the version via importlib.metadata (or
      spark_optima.__version__) and assert consistency between package metadata
      and pyproject instead of literal strings. Fix
      tests/unit/api/test_dependencies.py::TestAPIMetadata accordingly.
- [ ] Switch to git-tag-based dynamic versioning with hatch-vcs so release PRs
      no longer desync uv.lock: drop the static version field from
      pyproject.toml (dynamic = ["version"], hatch-vcs as the source), make
      __version__ resolve from package metadata at runtime, refresh uv.lock,
      and verify the package builds with a correct version from a git tag.
      In the PR description, note that release automation must then switch to
      release-type "simple" — a human applies that workflow change.
- [ ] Add `POST /api/v1/validate` and `POST /api/v1/import` API endpoints
      mirroring the CLI commands. Depends on extracting the CLI validate /
      import logic into a reusable core module first (the logic currently
      lives in `cli/main.py` around the `validate` and `import_config`
      commands).
- [ ] Extend the supported PySpark range in pyproject.toml to include 4.2 
      (keep the current lower bound), refresh uv.lock, and make the full test 
      suite pass against PySpark 4.2.0, fixing any incompatibilities
- [ ] Audit the code for Spark APIs deprecated or behavior-changed in 4.x 
      that we rely on, using the official 4.2 migration guide, 
      and add regression tests for each affected path
- [ ] Write a CI matrix proposal (ci-proposals/spark-42-matrix.yml) that adds a 
      PySpark 4.2 test leg alongside the existing Python versions, 
      with a PR description explaining the git mv to apply it      

## Discovered

Nothing added this session.
