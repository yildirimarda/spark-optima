# AWS Glue Platform Guide

This guide covers using Spark Optima with AWS Glue, AWS's serverless Spark platform.

## Overview

AWS Glue is a fully managed ETL service that makes it easy to prepare and load data for analytics. Spark Optima helps you optimize Glue job configurations for better performance and cost efficiency.

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
        "glue:GetJob",
        "glue:GetJobs",
        "glue:CreateJob",
        "glue:UpdateJob",
        "glue:StartJobRun",
        "glue:GetJobRun",
        "glue:GetJobRuns",
        "glue:BatchStopJobRun"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket/*",
        "arn:aws:s3:::your-bucket"
      ]
    }
  ]
}
```

## Glue Versions

Spark Optima supports AWS Glue versions:

| Glue Version | Spark Version | Python Version | Status |
|--------------|---------------|----------------|--------|
| Glue 3.0 | 3.1.1 | 3.7 | Supported |
| Glue 4.0 | 3.3.0 | 3.9 | Supported |
| Glue 5.0 | 3.5.0 | 3.9 | Supported |

## Configuration

### Basic Glue Configuration

```python
from spark_optima import Optimizer

optimizer = Optimizer(
    platform="aws_glue",
    spark_version="3.5.0"  # Maps to Glue 5.0
)
```

### Worker Type Selection

AWS Glue offers different worker types:

| Worker Type | vCPU | Memory | Disk | Use Case |
|-------------|------|--------|------|----------|
| G.1X | 4 | 16 GB | 64 GB | Small-medium jobs |
| G.2X | 8 | 32 GB | 128 GB | Medium-large jobs |
| G.4X | 16 | 64 GB | 256 GB | Large jobs |
| G.8X | 32 | 128 GB | 512 GB | Very large jobs |

Spark Optima automatically recommends the optimal worker type based on your workload.

### Resource Specification

```python
from spark_optima.platforms.models import ResourceSpec

resources = ResourceSpec(
    cpu_cores=32,          # Will map to G.8X workers
    memory_gb=128.0,
    disk_gb=512.0
)

result = optimizer.optimize(
    code_path="./glue_job.py",
    resources=resources,
    data_profile={"size_gb": 1000, "format": "parquet"}
)
```

## Usage Examples

### Example 1: Basic Glue Optimization

```python
from spark_optima import Optimizer

optimizer = Optimizer(
    platform="aws_glue",
    spark_version="3.5.0"
)

result = optimizer.optimize(
    code_path="./etl_job.py",
    data_profile={
        "size_gb": 500,
        "format": "parquet",
        "location": "s3://my-bucket/input/"
    },
    resource_constraints={
        "max_cost_usd": 50.0,
        "max_executors": 100
    }
)

print(f"Estimated cost: ${result.metadata.get('estimated_cost', 0):.2f}")
print(f"Worker type: {result.platform_specific.get('worker_type', 'G.1X')}")
```

### Example 2: CLI Usage

```bash
# Optimize Glue job
spark-optima optimize \
  -c ./glue_etl.py \
  -p aws_glue \
  -s 3.5.0 \
  -d 1000 \
  -f parquet \
  -o json > glue_result.json

# Export for AWS Glue
spark-optima export -r glue_result.json -f aws-glue -o glue_job_config.json

# Deploy using AWS CLI
aws glue update-job \
  --job-name my-etl-job \
  --job-update file://glue_job_config.json
```

### Example 3: Programmatic Deployment

```python
import boto3
from spark_optima import Optimizer
import json

# Optimize
optimizer = Optimizer(platform="aws_glue", spark_version="3.5.0")
result = optimizer.optimize(
    code_path="./daily_etl.py",
    data_profile={"size_gb": 200, "format": "json"}
)

# Create Glue job
client = boto3.client('glue', region_name='us-east-1')

job_config = {
    'Name': 'optimized-etl-job',
    'Role': 'AWSGlueServiceRole-Default',
    'Command': {
        'Name': 'glueetl',
        'ScriptLocation': 's3://my-bucket/scripts/daily_etl.py',
        'PythonVersion': '3.9'
    },
    'DefaultArguments': {
        '--job-language': 'python',
        '--enable-continuous-cloudwatch-log': 'true',
        '--enable-metrics': '',
        '--enable-spark-ui': 'true',
        '--spark-event-logs-path': 's3://my-bucket/spark-logs/',
        # Add optimized configurations
        **{f'--{k}': str(v) for k, v in result.configuration.items()}
    },
    'WorkerType': result.platform_specific.get('worker_type', 'G.1X'),
    'NumberOfWorkers': result.platform_specific.get('num_workers', 10),
    'GlueVersion': '4.0'
}

response = client.create_job(**job_config)
print(f"Created job: {response['Name']}")
```

## Glue-Specific Configurations

### Job Bookmarks

Enable job bookmarks for incremental processing:

```python
resource_constraints = {
    "enable_job_bookmark": True,
    "job_bookmark_keys": ["timestamp", "id"]
}

result = optimizer.optimize(
    code_path="./incremental_etl.py",
    resource_constraints=resource_constraints
)
```

### Connection Configuration

For jobs using JDBC connections:

```python
resource_constraints = {
    "connections": ["my-redshift-connection", "my-rds-connection"],
    "connection_timeout": 300
}
```

### Security Configuration

```python
resource_constraints = {
    "security_configuration": "my-security-config",
    "encryption_at_rest": True,
    "encryption_in_transit": True
}
```

## Cost Optimization

### Understanding Glue Pricing

AWS Glue charges based on:
- **DPUs** (Data Processing Units) per hour
- **Worker type** (G.1X = 1 DPU, G.2X = 2 DPUs, etc.)
- **Job duration**

### Cost-Aware Optimization

```python
# Optimize for cost
result = optimizer.optimize(
    code_path="./job.py",
    objectives=["minimize_cost"],
    resource_constraints={
        "max_cost_usd": 25.0,
        "cost_per_dpu_hour": 0.44  # Standard pricing
    }
)

print(f"Estimated cost: ${result.metadata.get('estimated_cost', 0):.2f}")
print(f"DPUs: {result.metadata.get('estimated_dpus', 0)}")
```

### Dynamic Allocation on Glue

```python
# Enable dynamic allocation for variable workloads
result = optimizer.optimize(
    code_path="./variable_load_job.py",
    resource_constraints={
        "enable_dynamic_allocation": True,
        "min_workers": 2,
        "max_workers": 50
    }
)
```

## Monitoring and Logging

### CloudWatch Integration

```python
# Optimized configuration includes monitoring settings
config = result.configuration

# Key monitoring configurations:
# spark.metrics.conf.*.sink.cloudwatch.class
# spark.metrics.conf.*.sink.cloudwatch.namespace
# spark.eventLog.enabled
```

### Spark UI on Glue

Enable Spark UI for debugging:

```bash
# In your job arguments
--enable-spark-ui true
--spark-event-logs-path s3://your-bucket/spark-logs/
```

Then access via AWS Console → Glue → Jobs → [Job Name] → Spark UI

## Best Practices

### 1. Use Appropriate Worker Types

```python
# Let Spark Optima choose based on workload
result = optimizer.optimize(
    code_path="./job.py",
    data_profile={"size_gb": 100},
    resource_constraints={
        "max_cost_usd": 20.0  # Cost constraint helps select worker type
    }
)
```

### 2. Enable Job Bookmarks for Incremental Loads

```python
# For incremental ETL
resource_constraints = {
    "enable_job_bookmark": True,
    "bookmark_options": "job-bookmark-enable"
}
```

### 3. Partition Data Appropriately

```python
data_profile = {
    "size_gb": 500,
    "format": "parquet",
    "partitioning": {
        "type": "date",
        "column": "event_date",
        "granularity": "day"
    }
}
```

### 4. Use Glue Data Catalog

```python
# Optimize for Data Catalog integration
resource_constraints = {
    "use_data_catalog": True,
    "catalog_database": "my_database",
    "catalog_table": "my_table"
}
```

## CI/CD Integration

### GitHub Actions for Glue

```yaml
# .github/workflows/glue-deploy.yml
name: Deploy to AWS Glue

on:
  push:
    branches: [ main ]

jobs:
  optimize-and-deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: arn:aws:iam::123456789:role/GitHubActionsRole
          aws-region: us-east-1
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install Spark Optima
        run: pip install spark-optima[aws]
      
      - name: Optimize configuration
        run: |
          spark-optima optimize \
            -c ./jobs/glue_etl.py \
            -p aws_glue \
            -d 500 \
            -o json > glue_config.json
      
      - name: Deploy to Glue
        run: |
          aws glue update-job \
            --job-name production-etl \
            --job-update file://glue_config.json
```

## Troubleshooting

### Issue: Job fails with out-of-memory

```python
# Increase memory allocation
resource_constraints = {
    "max_memory_gb": 256,
    "worker_type": "G.4X"  # Larger workers
}

result = optimizer.optimize(
    code_path="./job.py",
    resource_constraints=resource_constraints
)
```

### Issue: High costs

```python
# Optimize for cost
result = optimizer.optimize(
    code_path="./job.py",
    objectives=["minimize_cost"],
    resource_constraints={
        "max_cost_usd": 30.0
    }
)
```

### Issue: Slow execution

```python
# Optimize for speed
result = optimizer.optimize(
    code_path="./job.py",
    objectives=["minimize_time"],
    resource_constraints={
        "max_cost_usd": 100.0  # Higher budget for speed
    }
)
```

### Issue: Connection timeouts

```python
resource_constraints = {
    "connection_timeout": 600,  # Increase timeout
    "retry_limit": 3
}
```

## Next Steps

- Learn about [Databricks platform](./databricks.md)
- Explore [Azure Synapse](./azure-synapse.md)
- Read the [Configuration Guide](../user-guide/configuration.md)
- See [Python API Guide](../user-guide/api.md)