# GCP Dataproc Platform Guide

This guide covers using Spark Optima with Google Cloud Dataproc, GCP's managed Spark-on-Compute-Engine platform.

## Overview

Google Cloud Dataproc runs Apache Spark on YARN on top of Compute Engine instances. Spark Optima helps you pick the right N2 machine family and size, the right worker count, and an optimized Spark configuration — and can emit a ready-to-use `clusters.create` REST body.

The Dataproc adapter models:

- **N2 machine types** across two families: n2-standard (general purpose) and n2-highmem (memory-optimized)
- **One master node** in addition to the primary worker nodes
- **Cost** as the on-demand us-central1 Compute Engine hourly price plus the Dataproc fee ($0.01 per vCPU per hour)
- **Optional preemptible workers** as an approximate compute discount

!!! note
    Dataproc requires at least **2 primary workers** in a standard cluster. The adapter enforces this via `constraints.min_workers = 2`.

## Prerequisites

### GCP Project Setup

1. **GCP project** with the Dataproc API enabled
2. **gcloud CLI** installed and configured:

```bash
# Install the gcloud CLI, then authenticate
gcloud auth login
gcloud config set project your-project-id

# Enable the Dataproc API
gcloud services enable dataproc.googleapis.com

# Verify access
gcloud dataproc clusters list --region=us-central1
```

### Required IAM Permissions

Your user/service account needs the `roles/dataproc.editor` role (or equivalent permissions such as `dataproc.clusters.create` and `dataproc.jobs.submit`), plus `roles/compute.viewer` for machine type introspection.

## Dataproc Image Versions

Spark Optima maps Spark versions to Dataproc image versions:

| Dataproc Image | Spark Version | Status |
|----------------|---------------|--------|
| 2.0 | 3.1.3 | Supported |
| 2.1 | 3.3.2 | Supported |
| 2.2 | 3.5.3 | Supported |

Version matching is exact first, then by major.minor (e.g. `3.5.0` maps to image `2.2`). Unknown Spark versions fall back to the latest image (`2.2`).

## Machine Types

The adapter ships a representative set of N2 machine types:

| Machine Type | vCPUs | Memory | Family | GCE $/h (us-central1) |
|--------------|-------|--------|--------|------------------------|
| n2-standard-4 | 4 | 16 GB | General purpose | 0.1942 |
| n2-standard-8 | 8 | 32 GB | General purpose | 0.3885 |
| n2-standard-16 | 16 | 64 GB | General purpose | 0.7769 |
| n2-standard-32 | 32 | 128 GB | General purpose | 1.5539 |
| n2-standard-64 | 64 | 256 GB | General purpose | 3.1078 |
| n2-highmem-4 | 4 | 32 GB | Memory-optimized | 0.2620 |
| n2-highmem-8 | 8 | 64 GB | Memory-optimized | 0.5241 |
| n2-highmem-16 | 16 | 128 GB | Memory-optimized | 1.0481 |
| n2-highmem-32 | 32 | 256 GB | Memory-optimized | 2.0962 |
| n2-highmem-64 | 64 | 512 GB | Memory-optimized | 4.1924 |

!!! note
    The hourly cost used by Spark Optima is the Compute Engine price plus the Dataproc fee ($0.01 per vCPU per hour). Prices are approximate, vary by region, and change over time.

## Configuration

### Basic Dataproc Configuration

```python
from spark_optima.platforms import GCPDataprocPlatform
from spark_optima.platforms.models import ResourceSpec

platform = GCPDataprocPlatform(region="us-central1")

cluster_config = platform.recommend_config(
    resources=ResourceSpec(cpu_cores=64, memory_gb=256.0),
    spark_version="3.5.0",
)

print(cluster_config.worker_type.name)   # e.g. "n2-standard-8"
print(cluster_config.worker_count)       # primary worker count (>= 2)
print(cluster_config.driver_type.name)   # master node (n2-standard-4 by default)
```

### Machine Family Selection

```python
# Memory-intensive workloads (joins, caching, wide aggregations)
worker = platform.recommend_worker_type(
    target_memory_gb=64.0,
    target_cores=8,
    prefer_memory_optimized=True,   # n2-highmem family
)
```

### Spark Configuration Translation

Dataproc runs Spark on YARN, so the adapter:

- Does **not** set `spark.master` (the Dataproc runtime configures YARN itself)
- Reserves one vCPU per node for YARN/OS daemons
- Leaves ~10% of node memory as headroom
- Enables dynamic allocation with the external shuffle service
- Enables Adaptive Query Execution (AQE)

```python
spark_config = platform.translate_to_spark_config(cluster_config)
# {
#   "spark.executor.instances": "...",
#   "spark.executor.cores": "...",
#   "spark.executor.memory": "...",
#   "spark.dynamicAllocation.enabled": "true",
#   "spark.shuffle.service.enabled": "true",
#   "spark.sql.adaptive.enabled": "true",
#   ...
# }
```

## Usage Examples

### Example 1: Basic Dataproc Optimization

```python
from spark_optima import Optimizer
from spark_optima.platforms.models import ResourceSpec

optimizer = Optimizer(platform="gcp_dataproc", spark_version="3.5.0")

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
  --platform gcp_dataproc \
  --spark-version 3.5.0 \
  --code ./spark_job.py \
  --data-size 200
```

### Example 3: Generating a clusters.create Request Body

The adapter can emit a cluster definition ready for the Dataproc REST API (or `google-cloud-dataproc` client):

```python
from spark_optima.platforms import GCPDataprocPlatform
from spark_optima.platforms.models import ResourceSpec

platform = GCPDataprocPlatform()
cluster_config = platform.recommend_config(
    resources=ResourceSpec(cpu_cores=32, memory_gb=128.0),
    spark_version="3.5.0",
)

body = platform.get_dataproc_cluster_config(
    cluster_config,
    cluster_name="nightly-etl",
)
# body["config"]["softwareConfig"]["properties"] carries the optimized
# config as "spark:" prefixed cluster properties, e.g.
# {"spark:spark.executor.memory": "28g", ...}
```

The generated body contains `clusterName`, `gceClusterConfig` (zone placeholder), `masterConfig` / `workerConfig` with `machineTypeUri`, and `softwareConfig` with the image version and properties.

## Cost Estimation

### Understanding Dataproc Pricing

Dataproc pricing has two components per instance:

1. The **Compute Engine on-demand price** for the machine
2. The **Dataproc fee** ($0.01 per vCPU per hour)

Total cost = (master + workers) x (Compute Engine price + Dataproc fee) x hours.

```python
cost = platform.estimate_cost(cluster_config, duration_hours=2.0)

print(cost["total_cost"])
print(cost["breakdown"]["compute_cost"])
print(cost["breakdown"]["dataproc_fee"])
print(cost["breakdown"]["master_cost"])
print(cost["breakdown"]["worker_cost"])
```

### Preemptible (Spot) Workers

```python
platform = GCPDataprocPlatform(use_preemptible_workers=True)
cost = platform.estimate_cost(cluster_config, duration_hours=2.0)
```

When `use_preemptible_workers=True`, an approximate **~65% discount** is applied to the worker Compute Engine portion. This is an approximation:

- Actual Spot discounts vary (~60-91%) by machine type and over time
- The Dataproc fee is charged on all vCPUs regardless
- The master node always stays on-demand
- Real clusters keep at least 2 non-preemptible primary workers (preemptible instances go into the secondary worker group), so actual savings will be somewhat lower

## Best Practices

### 1. Match the Machine Family to the Workload

- **n2-standard (general purpose)** — balanced default for most ETL jobs
- **n2-highmem (memory-optimized)** — large joins, caching, memory-heavy aggregations

### 2. Use Ephemeral Clusters for Batch Jobs

Create a cluster per job (or use Dataproc workflow templates) and delete it when the job finishes — Dataproc bills per second with a 1-minute minimum.

### 3. Consider Preemptible Secondary Workers

For fault-tolerant batch workloads, adding preemptible secondary workers can reduce Compute Engine costs significantly. Keep the 2 required primary workers on-demand.

### 4. Keep Scripts in Cloud Storage

Submit jobs with `gs://bucket/...` paths — Dataproc jobs read scripts from Cloud Storage, not from your local machine.

## Troubleshooting

### Issue: Job fails with out-of-memory

```python
# Prefer memory-optimized machines
worker = platform.recommend_worker_type(
    target_memory_gb=128.0,
    target_cores=16,
    prefer_memory_optimized=True,
)
```

### Issue: Cluster creation rejected with fewer than 2 workers

Dataproc standard clusters require at least 2 primary workers. The adapter enforces this in `recommend_config` and `validate_config`; if you build a `ClusterConfig` manually, keep `worker_count >= 2`.

### Issue: High costs

```python
# Constrain the budget, or enable the preemptible approximation
result = optimizer.optimize(
    code_path="./spark_job.py",
    resource_constraints={"max_cost_usd": 25.0},
)
```

## Next Steps

- Learn about [AWS EMR](./aws-emr.md) for the AWS equivalent
- Explore [Spark on Kubernetes](./spark-k8s.md) for self-hosted clusters
- Read the [Configuration Guide](../user-guide/configuration.md)
- See [Python API Guide](../user-guide/api.md)
