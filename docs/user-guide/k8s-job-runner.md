# Kubernetes Job Runner for the API

Platform-team guide for running Spark Optima optimization jobs as Kubernetes Jobs with the existing Redis job store.

## Overview

The Kubernetes Job runner lets platform teams execute long-running Spark configuration optimizations outside the API pod lifecycle. It uses the same `RedisJobStore` interface (`SPARK_OPTIMA_JOB_STORE=redis`) that the async API uses, so job state is shared across all replicas and runners.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐ │
│  │  API Pod 1  │    │  API Pod 2  │    │  Optimization   │ │
│  │ (FastAPI)   │◄──►│ (FastAPI)   │◄──►│  Job (batch/v1) │ │
│  │             │    │             │    │                 │ │
│  │ Redis Store │    │ Redis Store │    │  CLI / API      │ │
│  └──────┬──────┘    └──────┬──────┘    └────────┬────────┘ │
│         │                  │                    │            │
│         └──────────────────┴────────────────────┘            │
│                         Redis (Service)                     │
└─────────────────────────────────────────────────────────────┘
```

- The Job reads `SPARK_OPTIMA_JOB_STORE=redis` and connects to the Redis Service configured in `SPARK_OPTIMA_REDIS_URL`.
- Results are written to the PVC-backed `/app/data/results/` path.
- The Job does not need direct API connectivity; it operates through the CLI (`spark_optima.cli.main`) and writes results to Redis and disk.

## Quick Start

### 1. Verify Redis

Ensure a Redis instance is available in the cluster:

```bash
kubectl get svc redis -n spark-optima
```

### 2. Deploy the Job

Using Helm:

```bash
helm upgrade spark-optima kubernetes/helm/spark-optima \
  -n spark-optima \
  --set optimizationJob.enabled=true \
  --set optimizationJob.platform=kubernetes \
  --set optimizationJob.sparkVersion=3.5.0 \
  --set optimizationJob.maxTrials=50 \
  --set optimizationJob.redisUrl=redis://redis.spark-optima.svc.cluster.local:6379/0
```

Using raw manifests:

```bash
# Edit the manifest for your environment
cp kubernetes/base/job-optimization.yaml /tmp/
# Modify SPARK_OPTIMA_REDIS_URL, resources, etc.
kubectl apply -f /tmp/job-optimization.yaml -n spark-optima
```

### 3. Monitor

```bash
# Job status
kubectl get job spark-optima-optimization -n spark-optima

# Pods
kubectl get pods -n spark-optima -l app.kubernetes.io/component=optimization-job

# Logs
kubectl logs -n spark-optima -l app.kubernetes.io/component=optimization-job --tail=50

# Results (after Job completes)
kubectl get job spark-optima-optimization -n spark-optima -o yaml
```

## Configuration Reference

### Environment Variables

| Variable | Default (Helm) | Purpose |
|----------|---------------|---------|
| `SPARK_OPTIMA_JOB_STORE` | `redis` | Must be `redis` for shared state |
| `SPARK_OPTIMA_REDIS_URL` | `redis://redis.spark-optima.svc.cluster.local:6379/0` | Redis connection URL |
| `SPARK_OPTIMA_LOG_LEVEL` | `INFO` | Logging verbosity |
| `SPARK_OPTIMA_MAX_TRIALS` | `50` | Bayesian optimization trials |
| `SPARK_OPTIMA_TIMEOUT_MINUTES` | `30` | Max optimization duration |

### Helm Values (`values.yaml` / `values-production.yaml`)

| Path | Default | Description |
|------|---------|-------------|
| `optimizationJob.enabled` | `false` | Deploy the optimization Job |
| `optimizationJob.platform` | `local` | Spark platform target |
| `optimizationJob.sparkVersion` | `3.5.0` | Spark version |
| `optimizationJob.outputFormat` | `json` | Output format (`json`, `yaml`) |
| `optimizationJob.maxTrials` | `50` | Trial count |
| `optimizationJob.timeoutMinutes` | `30` | Timeout |
| `optimizationJob.redisUrl` | `redis://...` | Redis URL |
| `optimizationJob.resources.requests.cpu` | `500m` | CPU request |
| `optimizationJob.resources.requests.memory` | `1Gi` | Memory request |
| `optimizationJob.resources.limits.cpu` | `1000m` | CPU limit |
| `optimizationJob.resources.limits.memory` | `2Gi` | Memory limit |
| `optimizationJob.ttlSecondsAfterFinished` | `3600` | Job TTL |
| `optimizationJob.backoffLimit` | `2` | Retry attempts |
| `optimizationJob.activeDeadlineSeconds` | `3600` | Max run time |

## Integration with Async API

When the API and the optimization Job share the same Redis instance:

1. A client submits a job via `POST /api/v1/optimize/async`.
2. The API stores the job in Redis (`SPARK_OPTIMA_JOB_STORE=redis`).
3. The Kubernetes Job can independently read that state, run optimization, and update the result back to Redis.
4. Any API replica can serve `GET /api/v1/jobs/{id}` without sticky sessions.

### Example: Submitting and polling

```bash
# Submit via API
JOB=$(curl -s -X POST http://spark-optima-api.svc.cluster.local:8000/api/v1/optimize/async \
  -H "Content-Type: application/json" \
  -d '{"code":"from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()","platform":"local","resources":{"cpu_cores":4,"memory_gb":16}}')

JOB_ID=$(echo $JOB | python3 -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')
echo "Job ID: $JOB_ID"

# Poll via API (any replica works because of Redis store)
while true; do
  STATUS=$(curl -s "http://spark-optima-api.svc.cluster.local:8000/api/v1/jobs/$JOB_ID" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])')
  echo "Status: $STATUS"
  [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ] && break
  sleep 2
done
```

## Security

- The Job uses the same `serviceAccountName` (`spark-optima-cli`) as the CLI Job template.
- Network policies (`networkPolicy`) should allow egress from the Job pod to Redis (port 6379) and to any webhook endpoints if webhooks are enabled.
- Redis connections should use TLS (`rediss://`) in production when available.

## Troubleshooting

### Job stays in `pending`

- Check that the image tag (`spark-optima`) and `pullPolicy` are correct.
- Verify the `serviceAccountName` exists (`kubectl get sa spark-optima-cli -n spark-optima`).
- Confirm the PVC claim (`spark-optima-data`) is bound.

### Redis connection errors

- Check the Job environment: `SPARK_OPTIMA_REDIS_URL` must point to a reachable Redis endpoint.
- Verify the `redis` Python package is installed in the image (`pip list | grep redis`).
- Check Redis Service: `kubectl get svc redis -n spark-optima`.

### Results file not found

- The Job writes to `/app/data/results/optimization_result.json` via the `data` PVC.
- After the Job completes (`ttlSecondsAfterFinished`), the Job pod is deleted but the PVC and file remain.
- Read results via a temporary pod or by mounting the same PVC:

```bash
kubectl run reader --rm -i --tty --restart=Never \
  --image=busybox:1.36 \
  --overrides='{"spec":{"serviceAccountName":"spark-optima-cli","containers":[{"name":"reader","image":"busybox:1.36","command":["cat","/app/data/results/optimization_result.json"],"volumeMounts":[{"name":"data","mountPath":"/app/data"}]}],"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"spark-optima-data"}}]}}' \
  -n spark-optima -- cat /app/data/results/optimization_result.json
```

## See Also

- `kubernetes/PRODUCTION.md` — full production deployment guide
- `docs/user-guide/rest-api.md` — REST API async job endpoints
- `docs/platforms/spark-k8s.md` — Spark-on-K8s platform adapter
