# Spark on Kubernetes Platform Guide

This guide covers using Spark Optima with self-hosted Apache Spark running natively on Kubernetes.

## Overview

Spark runs natively on Kubernetes: the driver and executors are pods scheduled by the cluster. Spark Optima helps you pick the right executor pod size, the right executor count, and an optimized `spark.kubernetes.*` configuration — and can emit a ready-to-use `SparkApplication` custom resource for the Spark Operator.

The Kubernetes adapter models:

- **Executor pod size presets** instead of cloud instance types: small, medium, large, xlarge
- **Dynamic allocation via shuffle tracking** — Kubernetes has no external shuffle service, so `spark.dynamicAllocation.shuffleTracking.enabled=true` is required
- **Cost as zero by default** — self-hosted clusters have no universal price, so pricing is user-supplied via `cost_per_vcpu_hour`

!!! note
    The adapter never sets `spark.master`. It depends on your cluster's API server URL (`k8s://https://<api-server>:<port>`) and must be supplied at submit time — or is implied when running through the Spark Operator.

## Prerequisites

1. A Kubernetes cluster (v1.24+) with enough capacity for your workload
2. A namespace and service account for Spark with RBAC permissions to create/delete pods:

```bash
kubectl create namespace data-jobs
kubectl create serviceaccount spark -n data-jobs
kubectl create clusterrolebinding spark-role \
  --clusterrole=edit \
  --serviceaccount=data-jobs:spark
```

3. A container image with Spark and your application dependencies (the official `apache/spark` images are a starting point)
4. Optionally, the [Kubeflow Spark Operator](https://github.com/kubeflow/spark-operator) for declarative job management

## Pod Size Presets

The adapter ships four executor pod size presets:

| Preset | CPUs | Memory | Ephemeral Storage |
|--------|------|--------|-------------------|
| small | 2 | 8 GB | 20 GB |
| medium | 4 | 16 GB | 40 GB |
| large | 8 | 32 GB | 80 GB |
| xlarge | 16 | 64 GB | 160 GB |

Presets are sized to fit comfortably on common node shapes while leaving room for the kubelet and system daemons.

## Configuration

### Basic Kubernetes Configuration

```python
from spark_optima.platforms import SparkOnK8sPlatform
from spark_optima.platforms.models import ResourceSpec

platform = SparkOnK8sPlatform(namespace="data-jobs")

cluster_config = platform.recommend_config(
    resources=ResourceSpec(cpu_cores=64, memory_gb=256.0),
    spark_version="3.5.0",
)

print(cluster_config.worker_type.name)   # e.g. "medium"
print(cluster_config.worker_count)       # executor pod count
print(cluster_config.driver_type.name)   # driver pod ("small" by default)
```

### Spark Configuration Translation

The adapter translates the cluster config into a Kubernetes-native Spark configuration:

- `spark.kubernetes.container.image` — a placeholder (`apache/spark:<version>`) you should replace with your own image
- `spark.kubernetes.namespace` — from the constructor (default `"default"`)
- `spark.kubernetes.executor.request.cores` / `spark.kubernetes.executor.limit.cores` — pod CPU requests/limits
- Executor/driver memory with a 10% overhead factor
- Dynamic allocation with `spark.dynamicAllocation.shuffleTracking.enabled=true`
- Adaptive Query Execution (AQE) enabled

```python
spark_config = platform.translate_to_spark_config(cluster_config)
# {
#   "spark.kubernetes.container.image": "apache/spark:3.5.0",
#   "spark.kubernetes.namespace": "data-jobs",
#   "spark.executor.instances": "...",
#   "spark.kubernetes.executor.request.cores": "...",
#   "spark.kubernetes.executor.limit.cores": "...",
#   "spark.dynamicAllocation.enabled": "true",
#   "spark.dynamicAllocation.shuffleTracking.enabled": "true",
#   "spark.sql.adaptive.enabled": "true",
#   ...
# }
```

!!! warning "No external shuffle service"
    Unlike YARN platforms (EMR, Dataproc), Kubernetes has no external shuffle service. Never set `spark.shuffle.service.enabled=true` on Kubernetes — dynamic allocation works through shuffle tracking instead, which the adapter configures automatically.

## Usage Examples

### Example 1: Basic Kubernetes Optimization

```python
from spark_optima import Optimizer
from spark_optima.platforms.models import ResourceSpec

optimizer = Optimizer(platform="kubernetes", spark_version="3.5.0")

result = optimizer.optimize(
    code_path="./spark_job.py",
    resources=ResourceSpec(cpu_cores=32, memory_gb=128.0),
    data_profile={"size_gb": 200, "format": "parquet"},
)

print(result.configuration)
```

### Example 2: CLI Usage

```bash
spark-optima optimize \
  --platform kubernetes \
  --spark-version 3.5.0 \
  --code ./spark_job.py \
  --data-size 200
```

### Example 3: Generating a SparkApplication CR (Spark Operator)

```python
from spark_optima.platforms import SparkOnK8sPlatform
from spark_optima.platforms.models import ResourceSpec

platform = SparkOnK8sPlatform(namespace="data-jobs")
cluster_config = platform.recommend_config(
    resources=ResourceSpec(cpu_cores=32, memory_gb=128.0),
    spark_version="3.5.0",
)

manifest = platform.get_spark_application_crd(
    cluster_config,
    app_name="nightly-etl",
    main_application_file="local:///opt/spark/app/etl.py",
)
# apiVersion: sparkoperator.k8s.io/v1beta2
# kind: SparkApplication
# spec.sparkConf carries the optimized config as strings;
# spec.driver / spec.executor carry cores, memory, and instances.
```

Write it out and apply it:

```python
import yaml

with open("spark-app.yaml", "w") as f:
    yaml.safe_dump(manifest, f)
```

```bash
kubectl apply -f spark-app.yaml
kubectl get sparkapplications -n data-jobs
```

!!! note
    Replace the `image` placeholder (`apache/spark:<version>`) with your own image containing application code, and adjust `spec.driver.serviceAccount` to your Spark service account.

### Example 4: spark-submit Without the Operator

```bash
spark-submit \
  --master k8s://https://<api-server>:6443 \
  --deploy-mode cluster \
  --conf spark.kubernetes.container.image=registry.example.com/spark-app:1.0 \
  --conf spark.kubernetes.namespace=data-jobs \
  $(python -c "
from spark_optima.platforms import SparkOnK8sPlatform
from spark_optima.platforms.models import ResourceSpec
p = SparkOnK8sPlatform(namespace='data-jobs')
cc = p.recommend_config(ResourceSpec(cpu_cores=32, memory_gb=128.0), '3.5.0')
print(' '.join(f'--conf {k}={v}' for k, v in p.translate_to_spark_config(cc).items()))
") \
  local:///opt/spark/app/etl.py
```

## Cost Estimation

Self-hosted clusters have no built-in pricing — **pricing is user-supplied**. Pass your blended infrastructure cost per vCPU-hour to get estimates; with the default of `0.0` every estimate is zero.

```python
# e.g. your nodes cost roughly $0.04 per vCPU per hour
platform = SparkOnK8sPlatform(namespace="data-jobs", cost_per_vcpu_hour=0.04)

cost = platform.estimate_cost(cluster_config, duration_hours=2.0)

print(cost["total_cost"])
print(cost["breakdown"]["total_vcpus"])
print(cost["breakdown"]["driver_cost"])
print(cost["breakdown"]["worker_cost"])
```

Total cost = (driver vCPUs + executor vCPUs) x `cost_per_vcpu_hour` x hours.

## Best Practices

### 1. Build Your Own Image

The `apache/spark:<version>` image is a placeholder. Bake your application code, Python dependencies, and connectors into your own image and set `spark.kubernetes.container.image` accordingly.

### 2. Keep Requests and Limits Equal

The adapter sets `request.cores` equal to `limit.cores` so executor pods get the `Guaranteed` QoS class, which avoids CPU throttling and eviction surprises for long-running shuffles.

### 3. Rely on Shuffle Tracking, Not the Shuffle Service

Dynamic allocation on Kubernetes requires `spark.dynamicAllocation.shuffleTracking.enabled=true` (set automatically). Executors holding shuffle data are kept alive until their data is no longer needed.

### 4. Use the Spark Operator for Production

The `SparkApplication` CR gives you declarative job specs, retries, and kubectl-native observability instead of bare `spark-submit`.

## Troubleshooting

### Issue: Executors stuck in Pending

The cluster lacks capacity for the requested pod size. Use a smaller preset or fewer executors:

```python
config = platform.recommend_config(
    resources=ResourceSpec(cpu_cores=16, memory_gb=64.0),
    spark_version="3.5.0",
    worker_count=4,
)
```

### Issue: Executors killed with OOMKilled

Increase the pod preset size or the overhead headroom — the adapter already reserves 10% via `spark.executor.memoryOverheadFactor`.

### Issue: Dynamic allocation does not scale down

Verify `spark.dynamicAllocation.shuffleTracking.enabled=true` is present and that you have not set `spark.shuffle.service.enabled=true` (there is no external shuffle service on Kubernetes).

## Next Steps

- Learn about [GCP Dataproc](./gcp-dataproc.md) and [AWS EMR](./aws-emr.md) for managed YARN platforms
- Read the [Configuration Guide](../user-guide/configuration.md)
- See [Python API Guide](../user-guide/api.md)
