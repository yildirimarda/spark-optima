# REST API Reference

This page is the authoritative reference for the Spark Optima HTTP REST API. For the in-process Python interface, see the [Python API Guide](api.md).

## Overview

The REST API is a [FastAPI](https://fastapi.tiangolo.com/) application that exposes the full optimization pipeline over HTTP:

- **Synchronous optimization** — `POST /api/v1/optimize`
- **Asynchronous optimization jobs** — `POST /api/v1/optimize/async` plus the `/api/v1/jobs` polling endpoints
- **Live job progress (SSE)** — `GET /api/v1/jobs/{job_id}/events`
- **Code analysis** — `POST /api/v1/analyze`
- **Platform discovery** — `GET /api/v1/platforms`
- **Workload templates** — `GET /api/v1/templates`
- **Health and probes** — `GET /health`, `/health/ready`, `/health/live`

The API always runs optimizations in **simulation mode** (fast performance prediction, no real Spark execution).

### Running the server

```bash
# Console script
uv run spark-optima-api

# Or directly with uvicorn
uv run uvicorn spark_optima.api.main:app --host 127.0.0.1 --port 8000
```

The server binds to `127.0.0.1:8000` by default. Host and port are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARK_OPTIMA_HOST` | `127.0.0.1` | Bind address. Set to `0.0.0.0` only when deliberately exposing the API. |
| `SPARK_OPTIMA_PORT` | `8000` | Listen port. |

All examples below assume a base URL of `http://localhost:8000`.

### Interactive documentation

The running server serves its own OpenAPI-based documentation:

| URL | Description |
|-----|-------------|
| `/docs` | Swagger UI (interactive, try requests in the browser) |
| `/redoc` | ReDoc rendering of the same schema |
| `/openapi.json` | Raw OpenAPI schema |

## Authentication and rate limiting

Both mechanisms are **opt-in and disabled by default** — a freshly started server is fully open. They are enabled per environment variable and apply to all `/api/v1/*` endpoints. Health endpoints (`/health`, `/health/ready`, `/health/live`) and the root endpoint (`/`) are never protected.

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARK_OPTIMA_API_KEYS` | unset (auth disabled) | Comma-separated list of accepted API keys. When set, every `/api/v1/*` request must carry a matching `X-API-Key` header. |
| `SPARK_OPTIMA_RATE_LIMIT` | unset (limiting disabled) | Allowed requests per minute (fixed 60-second window). Unset, empty, or `0` disables rate limiting. |
| `SPARK_OPTIMA_JOB_STORE` | `memory` | Backend for the asynchronous job store: `memory` (default, process-local), `sqlite` (persists jobs on local disk), or `redis` (shared across replicas; requires the optional `redis` package — `uv add redis`). When the redis backend is selected but the package is missing or the server is unreachable at startup, a warning is logged and the API falls back to the in-memory store. |
| `SPARK_OPTIMA_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL used when `SPARK_OPTIMA_JOB_STORE=redis`. Jobs are stored as JSON strings under `spark_optima:job:<job_id>`; finished jobs expire natively via Redis `EXPIRE`. |

### API keys

When `SPARK_OPTIMA_API_KEYS` is set, clients authenticate with the `X-API-Key` header:

```bash
export SPARK_OPTIMA_API_KEYS="key-one,key-two"

curl -s http://localhost:8000/api/v1/platforms \
  -H "X-API-Key: key-one"
```

A missing or invalid key returns `401 Unauthorized` with a `WWW-Authenticate: ApiKey` header. The error message is identical for missing and wrong keys, so the response does not reveal which case occurred:

```json
{"detail": "Invalid or missing API key."}
```

### Rate limiting

When `SPARK_OPTIMA_RATE_LIMIT` is set to a positive integer, a fixed-window limiter counts requests per minute. The limit is keyed by API key when authentication is enabled, otherwise by client IP. Requests over the budget receive `429 Too Many Requests` with a `Retry-After` header (seconds until the window resets):

```json
{"detail": "Rate limit exceeded. Please retry later."}
```

!!! note
    Both settings are read from the environment on every request, so they can be changed without restarting the server. The rate limiter state is in-memory and process-local.

## Error format

Errors raised by route handlers use the standard FastAPI shape — a JSON body with a `detail` field:

```json
{"detail": "Unsupported Spark version: 9.9.9. Available: ['3.5.0', '4.0.0']"}
```

Request validation failures return `422 Unprocessable Entity` with a structured `detail` array (one entry per invalid field):

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "code"],
      "msg": "String should have at least 10 characters",
      "input": "x"
    }
  ]
}
```

Unhandled server errors are caught by a global exception handler and return `500` with an `error`/`message` body:

```json
{
  "error": "internal_server_error",
  "message": "An unexpected error occurred. Please try again later."
}
```

## Endpoints

### POST /api/v1/optimize

Run the full optimization pipeline synchronously — code analysis, heuristic configuration, optional Bayesian search, and performance estimation — and return the recommended configuration in the response.

**Request body**

| Field | Type | Required | Constraints / Default | Description |
|-------|------|----------|-----------------------|-------------|
| `code` | string | yes | min length 10 | Spark application code (PySpark source) |
| `platform` | string | yes | one of `local`, `aws_glue`, `aws_emr`, `databricks`, `azure_synapse`, `gcp_dataproc`, `kubernetes` | Target platform |
| `spark_version` | string | no | pattern `\d+.\d+.\d+`, default `"3.5.0"` | Spark version to optimize for |
| `resources` | object | yes | see below | Available resources |
| `data_profile` | object | no | see below | Data characteristics |
| `constraints` | object | no | see below | Resource and cost constraints |
| `use_bayesian` | boolean | no | default `true` | Enable Bayesian optimization |
| `bayesian_trials` | integer | no | 1–500, default `50` | Number of Bayesian trials |
| `objectives` | array of string | no | default `["minimize_time"]`; allowed: `minimize_time`, `minimize_cost`, `maximize_throughput` | Optimization objectives |

`resources` object:

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `cpu_cores` | integer | yes | 1–128 | Number of CPU cores |
| `memory_gb` | number | yes | 1.0–2048.0 | Memory in GB |
| `disk_gb` | number | no | ≥ 0, default `0.0` | Local disk space in GB |
| `gpu_count` | integer | no | ≥ 0, default `0` | Number of GPUs |

`data_profile` object (optional):

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `size_gb` | number | yes | ≥ 0.001 | Data size in GB |
| `format` | string | yes | one of `parquet`, `delta`, `json`, `csv`, `orc`, `avro` | Data format |
| `schema` | object | no | — | Schema information |
| `compression` | string | no | — | Compression codec |
| `partitioning` | array of string | no | — | Partition columns |

`constraints` object (optional, all fields optional):

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `max_memory_gb` | number | ≥ 1.0 | Maximum memory in GB |
| `max_cost_per_hour` | number | ≥ 0.0 | Maximum cost per hour in USD |
| `max_executors` | integer | ≥ 1 | Maximum executor count |
| `timeout_minutes` | integer | ≥ 1 | Optimization timeout |

**Status codes**

| Code | Meaning |
|------|---------|
| `200` | Optimization completed; body is the optimization result |
| `400` | Unsupported Spark version (response lists available versions) |
| `422` | Request body failed validation |
| `500` | Optimization pipeline failed |

**Example**

```bash
curl -s -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "code": "from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()\ndf = spark.read.parquet(\"s3://bucket/data\")\ndf.groupBy(\"key\").count().write.parquet(\"s3://bucket/out\")",
    "platform": "databricks",
    "spark_version": "3.5.0",
    "resources": {"cpu_cores": 8, "memory_gb": 32},
    "data_profile": {"size_gb": 100, "format": "parquet"},
    "use_bayesian": true,
    "bayesian_trials": 50,
    "objectives": ["minimize_time"]
  }'
```

**Response body** (`200`)

| Field | Type | Description |
|-------|------|-------------|
| `optimization_id` | string | Unique identifier (`opt-<12 hex chars>`) |
| `status` | string | `"success"` |
| `configuration` | object | Recommended Spark configuration (parameter → value) |
| `estimated_time_minutes` | number | Predicted execution time (≥ 0) |
| `confidence_score` | number | Confidence level, 0.0–1.0 |
| `code_suggestions` | array | Code improvement suggestions (see below) |
| `platform_specific` | object | Platform-specific configuration (`platform`, `spark_version`, `cluster_config`, `glue_version`, `spark_pool_version`, `spark_config`) |
| `metadata` | object | Run metadata (`platform`, `spark_version`, `optimization_mode`, `bayesian_used`, `bayesian_trials`, `resources`, `data_profile`, `code_analysis`) |

Each entry in `code_suggestions`:

| Field | Type | Description |
|-------|------|-------------|
| `line_number` | integer | Line number of the issue |
| `issue_type` | string | Code smell category |
| `description` | string | Human-readable description |
| `suggestion` | string | Recommended fix |
| `severity` | string | `low`, `medium`, `high`, or `critical` |

```json
{
  "optimization_id": "opt-1a2b3c4d5e6f",
  "status": "success",
  "configuration": {
    "spark.executor.memory": "8g",
    "spark.executor.cores": "4",
    "spark.sql.shuffle.partitions": "200"
  },
  "estimated_time_minutes": 12.4,
  "confidence_score": 0.87,
  "code_suggestions": [
    {
      "line_number": 4,
      "issue_type": "missing_partitioning",
      "description": "Output written without explicit partitioning",
      "suggestion": "Consider partitionBy() on a frequently filtered column",
      "severity": "medium"
    }
  ],
  "platform_specific": {
    "platform": "databricks",
    "spark_version": "3.5.0",
    "cluster_config": {"num_workers": 4},
    "glue_version": null,
    "spark_pool_version": null,
    "spark_config": {"spark.executor.memory": "8g"}
  },
  "metadata": {
    "platform": "databricks",
    "spark_version": "3.5.0",
    "optimization_mode": "simulation",
    "bayesian_used": true,
    "bayesian_trials": 50,
    "resources": {"cpu_cores": 8, "memory_gb": 32.0},
    "data_profile": {"size_gb": 100, "format": "parquet"},
    "code_analysis": null
  }
}
```

### POST /api/v1/analyze

Run static code analysis only — detect code smells, anti-patterns, and missing optimizations (broadcast hints, caching, etc.) without producing a configuration.

**Request body**

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `code` | string | yes | min length 10 | Spark application code |

**Status codes**

| Code | Meaning |
|------|---------|
| `200` | Analysis completed |
| `422` | Request body failed validation |
| `500` | Analysis failed |

**Example**

```bash
curl -s -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "code": "from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()\ndf1.join(df2, \"id\").collect()"
  }'
```

**Response body** (`200`)

| Field | Type | Description |
|-------|------|-------------|
| `operations_count` | integer | Number of Spark operations detected |
| `smells_count` | integer | Number of code smells detected |
| `recommendations_count` | integer | Number of recommendations |
| `suggestions` | array | Code suggestions (same shape as in the optimize response) |

```json
{
  "operations_count": 3,
  "smells_count": 1,
  "recommendations_count": 1,
  "suggestions": [
    {
      "line_number": 3,
      "issue_type": "collect_on_large_dataset",
      "description": "collect() pulls the entire dataset to the driver",
      "suggestion": "Use take(n), show(), or write the result out instead",
      "severity": "high"
    }
  ]
}
```

### POST /api/v1/optimize/async

Submit an optimization as a background job. The request body is identical to `POST /api/v1/optimize`, plus one optional field. The request is validated upfront (an unsupported Spark version fails fast with `400`), queued on a small worker pool, and the response returns immediately with a job id to poll.

**Additional request field**

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `webhook_url` | string | no | http(s) only; internal targets rejected | Callback URL that receives a POST notification when the job finishes (completed **or** failed) |

**Status codes**

| Code | Meaning |
|------|---------|
| `202` | Job accepted; body contains the job id and polling URL |
| `400` | Unsupported Spark version |
| `422` | Request body failed validation (including a rejected `webhook_url`) |

**Response body** (`202`)

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Unique job identifier |
| `status` | string | Job status at submission time (usually `pending`) |
| `status_url` | string | Relative URL to poll (`/api/v1/jobs/{job_id}`) |

#### Webhook callbacks

When `webhook_url` is provided, the API POSTs a JSON notification to that URL once the job finishes — on completion *and* on failure:

```json
{
  "job_id": "9f8e7d6c5b4a39281706f5e4d3c2b1a0",
  "status": "completed",
  "submitted_at": "2026-06-10T10:00:00+00:00",
  "finished_at": "2026-06-10T10:04:30+00:00",
  "result": {"optimization_id": "opt-1a2b3c4d5e6f", "configuration": {"spark.executor.memory": "8g"}}
}
```

`result` is present only when the job completed (same shape as the synchronous optimize response); for failed jobs it is replaced by an `error` string:

```json
{
  "job_id": "9f8e7d6c5b4a39281706f5e4d3c2b1a0",
  "status": "failed",
  "submitted_at": "2026-06-10T10:00:00+00:00",
  "finished_at": "2026-06-10T10:00:12+00:00",
  "error": "Optimization failed: ..."
}
```

Delivery semantics:

- Any **2xx** response from your endpoint counts as delivered.
- Each attempt has a **10-second timeout**; up to **3 attempts** are made with exponential backoff (1s, then 2s).
- Delivery failures are logged and **never affect the job state** — the outcome is recorded as `webhook_status` (`delivered` or `failed`) on the job record, visible via `GET /api/v1/jobs/{job_id}`.

!!! warning "SSRF guard is best-effort"
    `webhook_url` must use the `http` or `https` scheme, and obvious internal targets are rejected with `422`: `localhost`, loopback addresses (`127.0.0.0/8`, `::1`), link-local addresses (including the cloud metadata endpoint `169.254.169.254`), the unspecified address `0.0.0.0`, and `*.internal` / `*.localhost` hostnames. This check operates on the URL hostname only — it does **not** resolve DNS, so a public hostname pointing at an internal address is not caught. If the API runs inside a sensitive network, enforce egress restrictions at the network level as well.

### GET /api/v1/jobs/{job_id}

Poll the status and result of a job submitted via `POST /api/v1/optimize/async`.

**Status codes**

| Code | Meaning |
|------|---------|
| `200` | Job found; body contains the current state |
| `404` | Unknown job id (never existed, or already evicted) |

**Response body** (`200`)

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Unique job identifier |
| `status` | string | `pending`, `running`, `completed`, or `failed` |
| `submitted_at` | string | UTC ISO timestamp of submission |
| `started_at` | string or null | UTC ISO timestamp when execution began |
| `finished_at` | string or null | UTC ISO timestamp when execution ended |
| `platform` | string | Requested platform |
| `spark_version` | string | Requested Spark version |
| `result` | object or null | Full optimization result (same shape as the synchronous response) when `status` is `completed` |
| `error` | string or null | Failure message when `status` is `failed` |
| `webhook_status` | string or null | Webhook delivery outcome (`delivered` or `failed`); `null` when no `webhook_url` was given or delivery has not finished yet |
| `progress` | object or null | Latest optimization progress snapshot from the Bayesian phase; `null` before the first trial finishes (or when `use_bayesian` is `false`) |

The `progress` object carries per-trial counters, updated while the job is `running` (writes are throttled to roughly one every 0.5 seconds):

| Field | Type | Description |
|-------|------|-------------|
| `trial_number` | integer | Number of the trial that just finished (0-based) |
| `n_trials` | integer | Total trials requested for the run |
| `trials_completed` | integer | Trials recorded in the study so far |
| `state` | string | Optuna state of the trial (`COMPLETE`, `PRUNED`, `FAIL`) |
| `best_value` | number or null | Best objective value so far (single-objective runs) |
| `best_values` | array of number or null | Objective values of one Pareto-optimal trial (multi-objective runs; replaces `best_value`) |

```json
{
  "job_id": "9f8e7d6c5b4a39281706f5e4d3c2b1a0",
  "status": "running",
  "progress": {
    "trial_number": 17,
    "n_trials": 50,
    "trials_completed": 18,
    "best_value": 9.21,
    "state": "COMPLETE"
  }
}
```

### GET /api/v1/jobs/{job_id}/events

Stream live job progress as [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) until the job reaches a terminal state. This avoids polling `GET /api/v1/jobs/{job_id}` in a loop: connect once and receive a push whenever the progress snapshot changes.

**Status codes**

| Code | Meaning |
|------|---------|
| `200` | Stream opened; body is a `text/event-stream` |
| `404` | Unknown job id (checked before streaming starts) |

**Event format**

The stream emits three kinds of frames:

- `event: progress` — sent whenever the job's `progress` snapshot changes. The `data` line is the JSON progress object described above.
- `: keep-alive` — comment heartbeat sent after roughly 10 seconds of silence so proxies and load balancers do not close the idle connection. SSE clients ignore comment lines automatically.
- `event: done` — final frame sent once the job reaches a terminal state (`completed` or `failed`); the stream closes right after. The `data` payload is `{"job_id": ..., "status": ..., "error": ...}` (`error` is `null` for completed jobs). Fetch the full result with `GET /api/v1/jobs/{job_id}`.

**Example**

Use `curl -N` (no buffering) to watch the stream:

```bash
curl -N http://localhost:8000/api/v1/jobs/9f8e7d6c5b4a39281706f5e4d3c2b1a0/events
```

```text
event: progress
data: {"trial_number": 0, "n_trials": 50, "trials_completed": 1, "best_value": 12.06, "state": "COMPLETE"}

event: progress
data: {"trial_number": 9, "n_trials": 50, "trials_completed": 10, "best_value": 9.21, "state": "COMPLETE"}

: keep-alive

event: done
data: {"job_id": "9f8e7d6c5b4a39281706f5e4d3c2b1a0", "status": "completed", "error": null}
```

With API-key authentication enabled, pass the header as usual: `curl -N -H "X-API-Key: key-one" ...`.

!!! note
    The server polls the job store internally (about twice per second), so events may lag the underlying trial completions slightly. If a finished job is evicted while a stream is open, the stream closes with a `done` frame whose `status` is `expired`. Multi-objective runs report `best_values` (a list) instead of `best_value` in each progress frame.

### GET /api/v1/jobs

List recently submitted jobs, newest first.

**Query parameters**

| Parameter | Type | Constraints | Default | Description |
|-----------|------|-------------|---------|-------------|
| `limit` | integer | 1–500 | `50` | Maximum number of jobs to return |

**Response body** (`200`): `{"jobs": [...]}` where each entry is a job summary (the detail fields above without `result` and `error`).

!!! note "Job retention"
    Finished jobs (completed or failed) are retained for a limited time (6 hours by default) and evicted afterwards, so older jobs may return `404` or disappear from the list. The job store backend is selected via `SPARK_OPTIMA_JOB_STORE` (`memory` default, `sqlite` for on-disk persistence, `redis` for a store shared across replicas — see the environment variable table above).

### Full async flow example

```bash
# 1. Submit the job
curl -s -X POST http://localhost:8000/api/v1/optimize/async \
  -H "Content-Type: application/json" \
  -d '{
    "code": "from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()\ndf = spark.read.parquet(\"s3://bucket/data\")\ndf.groupBy(\"key\").count().write.parquet(\"s3://bucket/out\")",
    "platform": "aws_emr",
    "resources": {"cpu_cores": 16, "memory_gb": 64}
  }'
# {"job_id": "9f8e7d6c5b4a39281706f5e4d3c2b1a0", "status": "pending",
#  "status_url": "/api/v1/jobs/9f8e7d6c5b4a39281706f5e4d3c2b1a0"}

# 2. Poll until status is "completed" or "failed"
curl -s http://localhost:8000/api/v1/jobs/9f8e7d6c5b4a39281706f5e4d3c2b1a0
# {"job_id": "...", "status": "running", "submitted_at": "...", ...}

# 3. Fetch the result once completed
curl -s http://localhost:8000/api/v1/jobs/9f8e7d6c5b4a39281706f5e4d3c2b1a0 \
  | python3 -c 'import json,sys; job=json.load(sys.stdin); print(json.dumps(job["result"]["configuration"], indent=2))'

# 4. List recent jobs
curl -s "http://localhost:8000/api/v1/jobs?limit=10"
```

### GET /api/v1/platforms

List all supported platforms with their capabilities.

**Response body** (`200`): a JSON array, one entry per platform:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Platform identifier (use this in optimize requests) |
| `display_name` | string | Human-readable name |
| `description` | string | Platform description |
| `supported_spark_versions` | array of string | Supported Spark versions |
| `supported_features` | array of string | Feature flags (e.g. `bayesian_optimization`, `cost_estimation`) |

```bash
curl -s http://localhost:8000/api/v1/platforms
```

### GET /api/v1/platforms/{platform_name}

Get details for a single platform. Returns `404` for unknown platform names.

```bash
curl -s http://localhost:8000/api/v1/platforms/databricks
```

### GET /api/v1/platforms/spark-versions

List all Spark versions supported by the configuration database.

```bash
curl -s http://localhost:8000/api/v1/platforms/spark-versions
# {"versions": ["3.5.0", "4.0.0", ...]}
```

### GET /api/v1/templates

List the curated workload templates — the same baselines exposed by the `spark-optima templates` CLI command (batch ETL, streaming, ML training, interactive analytics).

**Response body** (`200`): `{"templates": [...]}` where each entry is a summary:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Template identifier (use it in the detail endpoint) |
| `display_name` | string | Human-readable name |
| `description` | string | What the template is for |
| `parameter_count` | integer | Number of curated Spark parameters |

```bash
curl -s http://localhost:8000/api/v1/templates
```

```json
{
  "templates": [
    {
      "name": "etl-batch",
      "display_name": "Batch ETL",
      "description": "Throughput-oriented baseline for scheduled batch ETL pipelines...",
      "parameter_count": 10
    },
    {"name": "interactive", "display_name": "Interactive Analytics", "description": "...", "parameter_count": 9}
  ]
}
```

### GET /api/v1/templates/{name}

Get a single template including its full configuration and the rationale behind every parameter. Returns `404` for unknown template names (the error message lists the available templates).

**Response body** (`200`)

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Template identifier |
| `display_name` | string | Human-readable name |
| `description` | string | What the template is for |
| `workload_traits` | array of string | Characteristics of the targeted workload |
| `config` | object | Parameter name → `{"value": ..., "comment": ...}` (curated value plus rationale) |
| `recommended_for` | array of string | Scenarios where the template is a good fit |
| `not_recommended_for` | array of string | Scenarios where the template should be avoided |

```bash
curl -s http://localhost:8000/api/v1/templates/etl-batch
```

```json
{
  "name": "etl-batch",
  "display_name": "Batch ETL",
  "description": "Throughput-oriented baseline for scheduled batch ETL pipelines...",
  "workload_traits": ["Large sequential reads and writes", "..."],
  "config": {
    "spark.sql.adaptive.enabled": {
      "value": "true",
      "comment": "AQE re-plans shuffles at runtime using real statistics, the single biggest win for batch SQL."
    }
  },
  "recommended_for": ["Nightly or hourly scheduled pipelines", "..."],
  "not_recommended_for": ["Latency-sensitive interactive queries", "..."]
}
```

### GET /health

Service health including version, uptime, and component status. Never requires authentication.

```bash
curl -s http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600.5,
  "timestamp": "2026-06-10T12:00:00.000000",
  "components": {
    "config_database": "healthy"
  }
}
```

`status` is `healthy`, `degraded`, or `unhealthy` depending on component checks.

### GET /health/ready

Readiness probe (for Kubernetes / load balancers). Returns `{"status": "ready"}` when the service can accept traffic, or `{"status": "not_ready: <reason>"}` otherwise.

```bash
curl -s http://localhost:8000/health/ready
```

### GET /health/live

Liveness probe. Always returns `{"status": "alive"}` while the process is running.

```bash
curl -s http://localhost:8000/health/live
```

### GET /

Root endpoint with basic API information and documentation links.

```bash
curl -s http://localhost:8000/
# {"name": "Spark Optima API", "version": "...",
#  "description": "Intelligent Apache Spark configuration optimization",
#  "docs": "/docs", "health": "/health"}
```

## Supported platforms

| `name` | Display name | Supported Spark versions | Notable features |
|--------|--------------|--------------------------|------------------|
| `local` | Local Mode | 3.0.0 – 3.5.0, 4.0.0 | execution mode (in addition to simulation) |
| `aws_glue` | AWS Glue | 3.0.0 – 3.5.0 | cost estimation |
| `aws_emr` | AWS EMR | 3.3.0, 3.4.1, 3.5.0, 3.5.2 | cost estimation, cluster autoscaling |
| `databricks` | Databricks | 3.0.0 – 3.5.0, 4.0.0 | cost estimation, cluster autoscaling |
| `azure_synapse` | Azure Synapse Analytics | 3.0.0 – 3.5.0 | — |
| `gcp_dataproc` | GCP Dataproc | 3.1.0, 3.1.3, 3.3.0, 3.3.2, 3.5.0, 3.5.3 | cost estimation, cluster autoscaling |
| `kubernetes` | Spark on Kubernetes | 3.0.0 – 3.5.0, 4.0.0, 4.1.0 | cost estimation, cluster autoscaling |

All platforms support heuristic optimization, Bayesian optimization, code analysis, and simulation mode. Use `GET /api/v1/platforms` for the live, complete list.

## Deployment notes

!!! warning "Async jobs and multiple replicas"
    The default asynchronous job store is **process-local**: jobs are tracked in the memory of the API process that accepted them. When running multiple API replicas behind a load balancer, `GET /api/v1/jobs/{job_id}` requests may randomly hit a replica that does not know the job and return `404`. A `sqlite`-backed store (`SPARK_OPTIMA_JOB_STORE=sqlite`) persists jobs across restarts and worker processes on a single node, but does not help across nodes. For multi-replica deployments, use the Redis-backed store (`SPARK_OPTIMA_JOB_STORE=redis` plus `SPARK_OPTIMA_REDIS_URL`) so every replica shares the same job state — sticky sessions are then unnecessary for job polling. Without Redis, either run a single API replica, enable sticky sessions on the ingress, or use the synchronous `POST /api/v1/optimize` endpoint when scaling out. See the "API Security & Async Jobs" section in [kubernetes/PRODUCTION.md](https://github.com/yildirimarda/spark-optima/blob/main/kubernetes/PRODUCTION.md) for the full production deployment guide.

---

**Next Steps:**

- Use the [Python API Guide](api.md) for in-process integration
- See [CLI Usage Guide](cli.md)
- Read the [Configuration Guide](configuration.md)
