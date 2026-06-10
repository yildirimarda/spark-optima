# AWS EMR Platform Guide

This guide covers using Spark Optima with Amazon EMR, AWS's managed Spark-on-EC2 platform.

## Overview

Amazon EMR (Elastic MapReduce) runs Apache Spark on YARN on top of EC2 instances. Spark Optima helps you pick the right EC2 instance family and size, the right core node count, and an optimized Spark configuration — and can emit a ready-to-use `run_job_flow` cluster definition.

The EMR adapter models:

- **EC2 instance types** across three families: m5 (general purpose), r5 (memory-optimized), and c5 (compute-optimized)
- **One master node** in addition to the core (worker) nodes
- **Cost** as the on-demand us-east-1 EC2 hourly price plus the EMR surcharge (~25% of the EC2 price)

## Prerequisites

### AWS Account Setup

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured:

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# AWS Access Key ID: [your-access-key]
# AWS Secret Access Key: [your-secret-key]
# Default region: us-east-1
# Default output format: json

# Verify access
aws sts get-caller-identity
```

### Required IAM Permissions

Your AWS user/role needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "elasticmapreduce:RunJobFlow",
        "elasticmapreduce:DescribeCluster",
        "elasticmapreduce:DescribeStep",
        "elasticmapreduce:ListSteps",
        "elasticmapreduce:AddJobFlowSteps",
        "elasticmapreduce:TerminateJobFlows"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["iam:PassRole"],
      "Resource": [
        "arn:aws:iam::*:role/EMR_DefaultRole",
        "arn:aws:iam::*:role/EMR_EC2_DefaultRole"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::your-bucket/*",
        "arn:aws:s3:::your-bucket"
      ]
    }
  ]
}
```

## EMR Releases

Spark Optima maps Spark versions to EMR release labels:

| EMR Release | Spark Version | Status |
|-------------|---------------|--------|
| emr-6.9.0 | 3.3.0 | Supported |
| emr-6.15.0 | 3.4.1 | Supported |
| emr-7.0.0 | 3.5.0 | Supported |
| emr-7.5.0 | 3.5.2 | Supported |

Unknown Spark versions fall back to the latest release (`emr-7.5.0`).

## Instance Types

The adapter ships a representative set of EC2 instance types:

| Instance Type | vCPUs | Memory | Family | EC2 $/h (us-east-1) |
|---------------|-------|--------|--------|---------------------|
| m5.xlarge | 4 | 16 GB | General purpose | 0.192 |
| m5.2xlarge | 8 | 32 GB | General purpose | 0.384 |
| m5.4xlarge | 16 | 64 GB | General purpose | 0.768 |
| m5.8xlarge | 32 | 128 GB | General purpose | 1.536 |
| r5.xlarge | 4 | 32 GB | Memory-optimized | 0.252 |
| r5.2xlarge | 8 | 64 GB | Memory-optimized | 0.504 |
| r5.4xlarge | 16 | 128 GB | Memory-optimized | 1.008 |
| r5.8xlarge | 32 | 256 GB | Memory-optimized | 2.016 |
| c5.xlarge | 4 | 8 GB | Compute-optimized | 0.170 |
| c5.2xlarge | 8 | 16 GB | Compute-optimized | 0.340 |
| c5.4xlarge | 16 | 32 GB | Compute-optimized | 0.680 |
| c5.9xlarge | 36 | 72 GB | Compute-optimized | 1.530 |

!!! note
    The c5 family has no `8xlarge` size — `c5.9xlarge` is the closest larger size. The hourly cost used by Spark Optima is the EC2 price plus the EMR surcharge (~25%). Prices vary by region and over time.

## Configuration

### Basic EMR Configuration

```python
from spark_optima.platforms import AWSEMRPlatform
from spark_optima.platforms.models import ResourceSpec

platform = AWSEMRPlatform(region="us-east-1")

cluster_config = platform.recommend_config(
    resources=ResourceSpec(cpu_cores=64, memory_gb=256.0),
    spark_version="3.5.0",
)

print(cluster_config.worker_type.name)   # e.g. "m5.2xlarge"
print(cluster_config.worker_count)       # core node count
print(cluster_config.driver_type.name)   # master node (m5.xlarge by default)
```

### Instance Family Selection

```python
# Memory-intensive workloads (joins, caching, wide aggregations)
worker = platform.recommend_worker_type(
    target_memory_gb=64.0,
    target_cores=8,
    prefer_memory_optimized=True,   # r5 family
)

# CPU-bound workloads (parsing, transformation-heavy pipelines)
worker = platform.recommend_worker_type(
    target_memory_gb=16.0,
    target_cores=8,
    prefer_compute_optimized=True,  # c5 family
)
```

### Spark Configuration Translation

EMR runs Spark on YARN, so the adapter:

- Does **not** set `spark.master` (the EMR runtime configures YARN itself)
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

### Example 1: Basic EMR Optimization

```python
from spark_optima import Optimizer
from spark_optima.platforms.models import ResourceSpec

optimizer = Optimizer(platform="aws_emr", spark_version="3.5.0")

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
  --platform aws_emr \
  --spark-version 3.5.0 \
  --code ./spark_job.py \
  --data-size 200
```

### Example 3: Generating a run_job_flow Definition

The adapter can emit a cluster definition ready for `boto3`:

```python
import boto3

from spark_optima.platforms import AWSEMRPlatform
from spark_optima.platforms.models import ResourceSpec

platform = AWSEMRPlatform()
cluster_config = platform.recommend_config(
    resources=ResourceSpec(cpu_cores=32, memory_gb=128.0),
    spark_version="3.5.0",
)

job_flow = platform.get_emr_cluster_config(
    cluster_config,
    cluster_name="nightly-etl",
)
# job_flow["Configurations"] carries the optimized config in the
# spark-defaults classification.

client = boto3.client("emr", region_name="us-east-1")
response = client.run_job_flow(**job_flow)
print(response["JobFlowId"])
```

### Example 4: Submitting a Job Directly

With `boto3` installed (optional dependency), the adapter can create a transient cluster and run a script via spark-submit:

```python
result = platform.submit_job(
    code_path="s3://your-bucket/jobs/spark_job.py",
    cluster_name="spark-optima-run",
    cluster_config=cluster_config,
)

status = platform.get_job_status(result["cluster_id"], result["step_id"])
print(status["status"])  # starting / running / completed / failed
```

!!! note
    `boto3` is an optional dependency. Install it with `uv add boto3` if you want real job submission; everything else (recommendation, translation, cost estimation) works without it.

## Cost Estimation

### Understanding EMR Pricing

EMR pricing has two components per instance:

1. The **EC2 on-demand price** for the instance
2. The **EMR surcharge** (modeled as ~25% of the EC2 price)

Total cost = (master + core nodes) x (EC2 price + EMR surcharge) x hours.

```python
cost = platform.estimate_cost(cluster_config, duration_hours=2.0)

print(cost["total_cost"])
print(cost["breakdown"]["ec2_cost"])
print(cost["breakdown"]["emr_surcharge"])
print(cost["breakdown"]["master_cost"])
print(cost["breakdown"]["worker_cost"])
```

### Cost-Aware Optimization

```python
result = optimizer.optimize(
    code_path="./spark_job.py",
    resources=ResourceSpec(cpu_cores=32, memory_gb=128.0),
    resource_constraints={
        "max_cost_usd": 50.0,
    },
)
```

## Best Practices

### 1. Match the Instance Family to the Workload

- **m5 (general purpose)** — balanced default for most ETL jobs
- **r5 (memory-optimized)** — large joins, caching, memory-heavy aggregations
- **c5 (compute-optimized)** — CPU-bound parsing and transformation pipelines

### 2. Use Transient Clusters for Batch Jobs

The generated `run_job_flow` definition sets `KeepJobFlowAliveWhenNoSteps: false`, so the cluster terminates when the step finishes — you only pay for what the job uses.

### 3. Consider Spot Instances for Core Nodes

The cost model assumes on-demand pricing. For fault-tolerant batch workloads, switching the core instance group to spot instances can reduce EC2 costs significantly.

### 4. Keep Scripts in S3

Pass S3 paths (`s3://bucket/...`) to `submit_job` — EMR steps read the script from S3, not from your local machine.

## Troubleshooting

### Issue: Step fails with out-of-memory

```python
# Prefer memory-optimized instances
worker = platform.recommend_worker_type(
    target_memory_gb=128.0,
    target_cores=16,
    prefer_memory_optimized=True,
)
```

### Issue: High costs

```python
# Use a smaller instance type with more nodes, or constrain the budget
result = optimizer.optimize(
    code_path="./spark_job.py",
    resource_constraints={"max_cost_usd": 25.0},
)
```

### Issue: Cluster fails to start with role errors

The generated definition uses the default roles `EMR_DefaultRole` and `EMR_EC2_DefaultRole`. Create them once with:

```bash
aws emr create-default-roles
```

## Next Steps

- Learn about [AWS Glue](./aws-glue.md) for serverless Spark ETL
- Explore [Databricks](./databricks.md)
- Read the [Configuration Guide](../user-guide/configuration.md)
- See [Python API Guide](../user-guide/api.md)
