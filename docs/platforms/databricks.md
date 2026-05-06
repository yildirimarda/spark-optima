# Databricks Platform Guide

This guide covers using Spark Optima with Databricks, the leading managed Spark platform.

## Overview

Databricks provides a unified analytics platform powered by Apache Spark. Spark Optima helps you optimize cluster configurations for Databricks workspaces, balancing performance and cost.

## Prerequisites

### Databricks Account Setup

1. **Databricks Workspace** - Cloud-based (AWS, Azure, or GCP)
2. **Access Token** or **OAuth** configured

### Databricks CLI Setup

```bash
# Install Databricks CLI
pip install databricks-cli

# Configure with personal access token
databricks configure --token
# Host: https://myworkspace.cloud.databricks.com
# Token: dapiXXXXXXXXXXXXXXXX

# Verify connection
databricks workspace ls /
```

### Databricks SDK (Python)

```bash
pip install databricks-sdk
```

## Databricks Runtime (DBR) Versions

Spark Optima supports Databricks Runtime versions:

| DBR Version | Spark Version | Scala | Status |
|-------------|---------------|-------|--------|
| 10.4 LTS | 3.2.1 | 2.12 | Supported |
| 11.3 LTS | 3.3.0 | 2.12 | Supported |
| 12.2 LTS | 3.3.2 | 2.12 | Supported |
| 13.3 LTS | 3.4.1 | 2.12 | Supported |
| 14.0+ | 3.5.0 | 2.12 | Supported |

## Configuration

### Basic Databricks Configuration

```python
from spark_optima import Optimizer

optimizer = Optimizer(
    platform="databricks",
    spark_version="3.5.0"
)
```

### Node Type Selection

Databricks offers various node types:

| Node Type | vCPU | Memory | Use Case |
|-----------|------|--------|----------|
| Standard_DS3_v2 | 4 | 14 GB | Development |
| Standard_DS4_v2 | 8 | 28 GB | Small jobs |
| Standard_DS5_v2 | 16 | 56 GB | Medium jobs |
| Standard_D8s_v3 | 8 | 32 GB | General purpose |
| Standard_D16s_v3 | 16 | 64 GB | Large jobs |
| Standard_D32s_v3 | 32 | 128 GB | Very large jobs |
| Standard_D64s_v3 | 64 | 256 GB | Massive jobs |

Spark Optima recommends optimal node types based on workload characteristics.

### Cluster Configuration

```python
from spark_optima.platforms.models import ResourceSpec

resources = ResourceSpec(
    cpu_cores=64,          # Total cores needed
    memory_gb=256.0,       # Total memory needed
    disk_gb=1000.0
)

result = optimizer.optimize(
    code_path="./databricks_job.py",
    resources=resources,
    data_profile={"size_gb": 2000, "format": "delta"},
    resource_constraints={
        "max_cost_usd": 200.0,
        "enable_autoscaling": True
    }
)
```

## Usage Examples

### Example 1: Basic Databricks Optimization

```python
from spark_optima import Optimizer

optimizer = Optimizer(
    platform="databricks",
    spark_version="3.5.0"
)

result = optimizer.optimize(
    code_path="./ml_training.py",
    data_profile={
        "size_gb": 1000,
        "format": "delta",
        "location": "/mnt/data/training"
    },
    resource_constraints={
        "max_cost_usd": 100.0,
        "max_workers": 50
    }
)

# Export for Databricks
import json
with open("cluster_config.json", "w") as f:
    json.dump(result.platform_specific, f, indent=2)
```

### Example 2: CLI Usage

```bash
# Optimize for Databricks
spark-optima optimize \
  -c ./ml_pipeline.py \
  -p databricks \
  -s 3.5.0 \
  -d 2000 \
  -f delta \
  -o json > databricks_result.json

# Export cluster configuration
spark-optima export \
  -r databricks_result.json \
  -f databricks-json \
  -o cluster.json

# Create cluster via Databricks CLI
databricks clusters create --json-file cluster.json
```

### Example 3: Programmatic Cluster Creation

```python
from spark_optima import Optimizer
from databricks.sdk import WorkspaceClient
import json

# Optimize
optimizer = Optimizer(platform="databricks", spark_version="3.5.0")
result = optimizer.optimize(
    code_path="./analytics_job.py",
    data_profile={"size_gb": 500, "format": "parquet"}
)

# Create cluster using Databricks SDK
client = WorkspaceClient()

cluster_config = result.platform_specific.get('cluster_config', {})

cluster = client.clusters.create(
    cluster_name="optimized-cluster",
    spark_version="13.3.x-scala2.12",
    node_type_id=cluster_config.get('node_type_id', 'Standard_DS4_v2'),
    autoscale=cluster_config.get('autoscale', {
        'min_workers': 2,
        'max_workers': 20
    }),
    spark_conf=result.configuration,
    enable_elastic_disk=True,
    enable_local_disk_encryption=True
)

print(f"Created cluster: {cluster.cluster_id}")
```

## Databricks-Specific Features

### Unity Catalog Integration

```python
resource_constraints = {
    "unity_catalog_enabled": True,
    "catalog": "main",
    "schema": "production",
    "data_sensitivity": "high"  # Enables additional security configs
}

result = optimizer.optimize(
    code_path="./uc_job.py",
    resource_constraints=resource_constraints
)
```

### Autoscaling Configuration

```python
# Enable autoscaling with bounds
resource_constraints = {
    "enable_autoscaling": True,
    "min_workers": 2,
    "max_workers": 100,
    "autoscale_mode": "standard"  # or "enhanced"
}

result = optimizer.optimize(
    code_path="./variable_load_job.py",
    resource_constraints=resource_constraints
)
```

### Delta Lake Optimizations

For Delta Lake workloads:

```python
data_profile = {
    "size_gb": 1000,
    "format": "delta",
    "delta_properties": {
        "enable_change_data_feed": True,
        "enable_deletion_vectors": True,
        "auto_optimize": True
    }
}

result = optimizer.optimize(
    code_path="./delta_etl.py",
    data_profile=data_profile
)
```

### Photon Acceleration

```python
# Optimize for Photon (vectorized execution)
resource_constraints = {
    "enable_photon": True,
    "photon_priority": "high"  # Prioritize Photon-optimized configs
}

result = optimizer.optimize(
    code_path="./photon_job.py",
    resource_constraints=resource_constraints
)
```

## Cost Optimization

### Understanding Databricks Pricing

Databricks pricing components:
- **DBUs** (Databricks Units) per hour
- **Compute type** (Jobs Compute vs All-Purpose Compute)
- **Instance type**
- **Cloud provider costs** (AWS/Azure/GCP)

### Cost-Aware Optimization

```python
# Optimize for cost
result = optimizer.optimize(
    code_path="./job.py",
    objectives=["minimize_cost"],
    resource_constraints={
        "max_cost_usd": 50.0,
        "dbu_rate": 0.15,  # Your DBU rate
        "compute_type": "jobs"  # or "all-purpose"
    }
)

print(f"Estimated DBUs: {result.metadata.get('estimated_dbus', 0):.2f}")
print(f"Estimated cost: ${result.metadata.get('estimated_cost', 0):.2f}")
```

### Spot Instances

```python
# Use spot instances for cost savings
resource_constraints = {
    "use_spot_instances": True,
    "spot_bid_price": 0.50,  # Max spot price
    "spot_fallback": "on-demand"  # Fallback to on-demand if spot unavailable
}
```

## Job Clusters vs All-Purpose Clusters

### Job Clusters (Recommended for Production)

```python
resource_constraints = {
    "cluster_type": "job",  # vs "all-purpose"
    "auto_termination_minutes": 10
}

result = optimizer.optimize(
    code_path="./production_etl.py",
    resource_constraints=resource_constraints
)
```

**Benefits:**
- Lower cost (Jobs Compute pricing)
- Automatic termination
- Optimized for batch workloads

### All-Purpose Clusters

```python
resource_constraints = {
    "cluster_type": "all-purpose",
    "auto_termination_minutes": 120  # Longer for interactive use
}
```

**Use for:**
- Interactive development
- Notebooks
- Ad-hoc analysis

## Best Practices

### 1. Use Jobs Compute for Production

```python
resource_constraints = {
    "cluster_type": "job",
    "auto_termination_minutes": 10
}
```

### 2. Enable Autoscaling

```python
resource_constraints = {
    "enable_autoscaling": True,
    "min_workers": 2,
    "max_workers": 50
}
```

### 3. Use Delta Lake for Large Datasets

```python
data_profile = {
    "format": "delta",
    "size_gb": 1000,
    "delta_properties": {
        "auto_optimize": True,
        "auto_compact": True
    }
}
```

### 4. Enable Adaptive Query Execution

Spark Optima automatically configures AQE:

```python
# These are automatically set by Spark Optima:
# spark.sql.adaptive.enabled=true
# spark.sql.adaptive.coalescePartitions.enabled=true
# spark.sql.adaptive.skewJoin.enabled=true
```

### 5. Use Instance Pools for Faster Startup

```python
resource_constraints = {
    "use_instance_pool": True,
    "instance_pool_id": "pool-id-from-databricks"
}
```

## CI/CD Integration

### GitHub Actions for Databricks

```yaml
# .github/workflows/databricks-deploy.yml
name: Deploy to Databricks

on:
  push:
    branches: [ main ]

jobs:
  optimize-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install spark-optima[databricks]
          pip install databricks-sdk
      
      - name: Configure Databricks
        run: |
          cat > ~/.databrickscfg << EOF
          [DEFAULT]
          host = ${{ secrets.DATABRICKS_HOST }}
          token = ${{ secrets.DATABRICKS_TOKEN }}
          EOF
      
      - name: Optimize configuration
        run: |
          spark-optima optimize \
            -c ./jobs/databricks_job.py \
            -p databricks \
            -d 1000 \
            -o json > cluster_config.json
      
      - name: Deploy cluster
        run: |
          python -c "
          from databricks.sdk import WorkspaceClient
          import json
          
          client = WorkspaceClient()
          
          with open('cluster_config.json') as f:
              config = json.load(f)
          
          cluster = client.clusters.create(
              cluster_name='production-etl',
              spark_version='13.3.x-scala2.12',
              **config['platform_specific']['cluster_config']
          )
          
          print(f'Created cluster: {cluster.cluster_id}')
          "
```

## Troubleshooting

### Issue: Cluster startup too slow

```python
# Use instance pools
resource_constraints = {
    "use_instance_pool": True,
    "instance_pool_id": "your-pool-id"
}

# Or reduce autoscaling minimum
resource_constraints = {
    "enable_autoscaling": True,
    "min_workers": 0,  # Start with 0, scale up
    "max_workers": 20
}
```

### Issue: High costs

```python
# Switch to Jobs Compute
resource_constraints = {
    "cluster_type": "job",
    "auto_termination_minutes": 10
}

# Optimize for cost
result = optimizer.optimize(
    code_path="./job.py",
    objectives=["minimize_cost"]
)
```

### Issue: Out of memory errors

```python
# Use larger node types
resources = ResourceSpec(
    cpu_cores=64,
    memory_gb=256.0
)

# Or enable memory-intensive configs
resource_constraints = {
    "memory_intensive": True,
    "off_heap_enabled": True
}
```

### Issue: Slow query performance

```python
# Enable Photon
resource_constraints = {
    "enable_photon": True
}

# Optimize for speed
result = optimizer.optimize(
    code_path="./job.py",
    objectives=["minimize_time"]
)
```

## Next Steps

- Explore [AWS Glue](./aws-glue.md)
- Learn about [Azure Synapse](./azure-synapse.md)
- Read the [Configuration Guide](../user-guide/configuration.md)
- See [Python API Guide](../user-guide/api.md)