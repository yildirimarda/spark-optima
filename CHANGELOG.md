# Changelog

All notable changes to the Spark Optima project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
