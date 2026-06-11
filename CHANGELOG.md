# Changelog

All notable changes to the Spark Optima project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (v1.5)
- **Scala code analysis** (planned since v1.0): lexer-based Scala Spark parser (comment/string masking incl. interpolated strings, `val` lineage tracking, multi-line fluent chains) feeding the existing smell/recommendation pipeline; `spark.sql("...")` literals in Scala flow into the sqlglot analyzer; CLI `analyze`/`optimize` accept `.scala` files. New `groupbykey_usage` smell (Python + Scala).
- **GCP live pricing**: Cloud Billing Catalog API client (`SPARK_OPTIMA_GCP_API_KEY`-gated, key sent via header), N2 core/RAM SKU rates per region → hourly machine price; Dataproc `estimate_cost` uses live rates under the same cache/fallback/labeling rules as the other adapters.
- **Live optimization progress over SSE**: per-trial `progress_callback` on the optimizer chain, `progress` persisted on async jobs (all three stores), and `GET /api/v1/jobs/{id}/events` streaming progress/done events.
- **Templates API**: `GET /api/v1/templates` and `GET /api/v1/templates/{name}` (parity with the CLI).
- **Config DB unit normalization**: all BYTES/DURATION parameter bounds audited across the 8 Spark config YAMLs and canonicalized (the `spark.kryoserializer.buffer.max` default previously failed its own max; memory minimums were compared in the wrong unit). `validate` now range-checks BYTES/DURATION values correctly.
- **Examples & docs refresh**: new examples (event-log analysis, multi-objective Pareto, EMR/Dataproc/K8s platforms, workload templates), README command catalogue + environment-variable table.

### Fixed (v1.5)
- `optimize`/`analyze` with `--output json` now route all decorative output to stderr — `... --output json > result.json` produces a clean machine-readable file.
- GCP pricing lookups are bounded by a total deadline and the API key no longer appears in URLs (header-based auth).
- SSE streaming no longer blocks the event loop on job-store I/O; non-finite objective values are mapped to null in progress events.

### Added (v1.4)
- **Live pricing (opt-in)**: `SPARK_OPTIMA_LIVE_PRICING=1` fetches real hourly rates — Azure Retail Prices API (public) for Synapse, AWS Pricing API (optional boto3) for EMR/Glue — with a 24h JSON cache and silent fallback to the static tables on any failure. Cost breakdowns now carry `pricing_source: live|static`.
- **Redis job store**: `SPARK_OPTIMA_JOB_STORE=redis` (+ `SPARK_OPTIMA_REDIS_URL`) — the real multi-replica answer for async jobs; guarded optional `redis` import, startup fallback to memory on connection failure.
- **Webhooks**: optional `webhook_url` on `POST /optimize/async` — result POSTed on completion/failure (10s timeout, 3 attempts with backoff, best-effort SSRF guard, `webhook_status` surfaced on the job).
- **New CLI commands**: `validate` (config vs parameter DB + platform constraints + anti-patterns), `import` (current vs recommended config diff), `templates` (curated etl-batch / streaming / ml-training / interactive baselines with per-param rationale).
- **Spark History Server client**: `analyze-log --history-server URL [--app-id ID]` produces the same metrics + tuning hints as event-log files, straight from a running History Server.
- **Wizard refresh**: objectives multi-select (Pareto), optional event-log enrichment with inferred data size/hints, current export format catalogue (7 steps now).

### Fixed (v1.4)
- `httpx` was a dev-only dependency while being imported at runtime by the Databricks/Synapse adapters — promoted to a runtime dependency.
- The wizard never passed the selected objectives through to the optimizer.

### Added (v1.3)
- **ML surrogate predictor wired end-to-end**: the simulation engine now learns online from trials (RandomForest, R²-gated blending of analytical + ML predictions, 20-sample minimum), persists models via joblib (`SPARK_OPTIMA_MODEL_DIR`), and execution-mode trials feed real measured runtimes into the surrogate. scikit-learn stays optional — everything degrades silently to pure-analytical without it.
- **Performance model physics**: GC overhead modeled from memory pressure (with G1GC relief), shuffle transfer bounded by network bandwidth (10 Gbit/node) in addition to disk, and a straggler/wave-based skew model replacing the flat penalty (AQE skew mitigation caps effective skew).
- **Multi-objective optimization surfaced end-to-end**: repeatable `optimize --objective` flag, Pareto frontier persisted into result metadata (capped at 50 points), new `spark-optima pareto -r result.json` command, and `pareto-json`/`pareto-csv` export formats.
- **REST API reference docs**: `docs/user-guide/rest-api.md` — all endpoints, auth/rate-limit/job-store env vars, full async flow with curl examples.
- **Persistent job store**: `SPARK_OPTIMA_JOB_STORE=sqlite` (+ `SPARK_OPTIMA_JOB_DB`) keeps async jobs across restarts (WAL mode, 2h worker-lost staleness rule). In-memory remains the default.

### Fixed (v1.3)
- An advisory "parallelism very high relative to cores" finding was treated as a feasibility *failure*, making Bayesian trials on sparse configs silently evaluate to `inf`. It is now advisory-only.
- `spark-optima` console script was missing from `[project.scripts]` (docs referenced it but only `spark-optima-api` existed) and `spark-optima-api` pointed at a nonexistent `main()` — both entry points now work.
- API OpenAPI metadata listed only 4 platforms and pinned a stale hardcoded version; it now lists all 7 and reads `spark_optima.__version__`.

### Added (v1.2)
- **Spark event log analyzer** (`core/execution/event_log.py`): parses real event logs (plain/gzip) into stage/task metrics — GC time, shuffle volumes, spill, task skew. New `spark-optima analyze-log` command and `optimize --event-log` to enrich optimization with real run data. `MetricsCollector.collect_from_event_log()` replaces the previously stubbed GC/shuffle metrics.
- **Async job API**: `POST /api/v1/optimize/async` (202 + job id), `GET /api/v1/jobs/{id}`, `GET /api/v1/jobs` — thread-pool execution with an in-memory TTL job store.
- **API security (opt-in)**: `SPARK_OPTIMA_API_KEYS` enables X-API-Key auth on `/api/v1/*`; `SPARK_OPTIMA_RATE_LIMIT` enables per-client rate limiting (429 + Retry-After). Both off by default.
- **GCP Dataproc platform adapter**: n2-standard/highmem machine types, Compute Engine + $0.01/vCPU-h Dataproc fee cost model, optional preemptible workers, `clusters.create` config export.
- **Spark-on-Kubernetes platform adapter**: pod size presets, `spark.kubernetes.*` config translation with shuffle-tracking dynamic allocation (no external shuffle service), SparkApplication CRD export (Spark Operator), user-supplied $/vCPU-hour pricing.
- **Full SQL analysis** via sqlglot (spark dialect): replaces the v1.1 substring scan. Six AST-based findings — select *, cartesian joins (explicit CROSS JOIN and implicit comma joins without WHERE), ORDER BY without LIMIT, UNION vs UNION ALL, leading-wildcard LIKE, IN (subquery).
- **Regional pricing**: the previously ignored `region` parameter now scales cost estimates via curated static multiplier tables (AWS, Azure, GCP regions); breakdowns include `region` and `region_multiplier`.

### Fixed (v1.2)
- The sync optimize endpoint passed Pydantic's deprecated `schema` classmethod instead of the `schema_info` field into the data profile.
- `DatabricksPlatform(cloud_provider="azure")` defaulted to the AWS region name `us-east-1`; the default region is now per-cloud (`eastus` for Azure).

### Added (v1.1)
- **AWS EMR platform adapter** (`platforms/aws_emr.py`): m5/r5/c5 worker types, YARN-style config translation, EC2 + EMR surcharge cost model, `run_job_flow` cluster config export, optional boto3 job submission. Registered across CLI, API, optimizer, and heuristic rules.
- **Optimization history** (`core/history.py`): SQLite-backed persistence of every CLI optimization (auto-saved, best-effort). New `spark-optima history` command (list / `--show` / `--clear`).
- **New CLI commands**: `compare` (diff two result files: config + metric deltas, `--output json` supported) and `explain` (per-parameter rationale sourced from heuristic rule descriptions and the parameter database).
- **New export formats**: `airflow` (platform-aware DAG snippet — SparkSubmitOperator / DatabricksSubmitRunOperator / GlueJobOperator), `kubernetes` (ConfigMap with spark-defaults.conf), `emr` (`aws emr create-cluster --configurations` JSON).
- **9 new code smells**: cartesian/cross join, `toPandas()`, `count()`-for-emptiness, `repartition(1)`/`coalesce(1)` before write, `inferSchema=True`, `withColumn` in loop, `select("*")`, `orderBy` without `limit`, plus `pandas_udf` discrimination (MEDIUM) vs plain Python UDF (HIGH). Lightweight SQL literal analysis of `spark.sql()` strings (SELECT *, CROSS JOIN).
- **New heuristic rules**: task speculation (`spark.speculation.*`), data-aware `spark.dynamicAllocation.maxExecutors`, AQE fine-tuning (advisory partition size, skew factor override).
- **Bayesian warm-start**: the heuristic config is enqueued as trial #1 (Optuna can never return worse than the heuristic baseline), and studies resume from `storage_path` with prior trial counts surfaced in result metadata.

### Fixed (v1.1)
- CLI `analyze` passed the file *path* string to the analyzer instead of the file *contents* — it could never analyze a real file (existing tests only asserted broad exit-code ranges and mocked the analyzer).
- Machine-readable CLI output (`optimize -o json|yaml`, `analyze -o json`, `export` to stdout) now bypasses Rich: long lines were soft-wrapped at terminal width, producing unparseable JSON when piped to files.
- Bytes search-space upper bounds are aligned to the Optuna suggestion grid (silences repeated `step` warnings).
- Data-skew smell detection no longer skips operations with empty argument lists.
- `large_collect` smell now reports its code location.
- Resumed Bayesian studies now reconstruct the best config from stored trials (previously returned `{}`).
- mypy now passes with 0 errors (added `types-PyYAML`/`types-psutil`, overrides for optional untyped deps, fixed `Any` returns in execution module).
- Removed duplicate test module `tests/unit/test_optimizer.py` (fully covered by `tests/unit/core/`).

### Security
- **BREAKING**: Remove insecure `exec()` fallback in `SparkRunner`. Docker is now **required** for secure code execution. If Docker is not available, a `RuntimeError` is raised.
- Add AST-based code safety validation as an additional layer of defense.

### Fixed
- Fix `_execute_code_docker` return type annotation to satisfy mypy (`dict[str, Any]`).
- Fix failing test `test_execute_code_with_exception` by properly mocking Docker execution.
- Fix failing test `test_generate_float_value` by setting `null_ratio=0.0`.
- Remove duplicate `imagePullPolicy` in Kubernetes deployment manifest.
- Update CLI documentation to match actual CLI options.

### Changed
- Update `README.md` to reflect that Docker is required (no insecure fallback).
- Update `docs/index.md` to remove references to exec mode fallback.
- Improve Dockerfile for production: add `tini` for signal handling, fix `JAVA_HOME`, create `/tmp/spark-events` directory.

### Added
- Coverage increased to 95.31% (1745 tests passed).
- Ruff and mypy checks now pass with 0 errors.

## [0.1.0] - 2024-05-04

### Added
- Initial release with hybrid optimization (heuristic + Bayesian).
- Support for Local, AWS Glue, Databricks, Azure Synapse platforms.
- Code analysis with smell detection and recommendations.
- Simulation and execution modes.
- Docker-based secure code execution.
- REST API with FastAPI.
- CLI with Typer.
- Kubernetes manifests and Helm charts.
- Comprehensive test suite with 1700+ tests.
