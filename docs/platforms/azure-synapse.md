# Azure Synapse Platform Guide

This guide covers using Spark Optima with Azure Synapse Analytics, Microsoft's analytics service.

## Overview

Azure Synapse Analytics provides a unified experience for big data and data warehousing. Spark Optima helps you optimize Spark pool configurations in Synapse for better performance and cost efficiency.

## Prerequisites

### Azure Account Setup

1. **Azure Subscription** with appropriate permissions
2. **Azure Synapse Workspace** created
3. **Apache Spark Pool** provisioned

### Azure CLI Setup

```bash
# Install Azure CLI
# Windows
winget install Microsoft.AzureCLI

# macOS
brew install azure-cli

# Ubuntu
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Set default subscription
az account set --subscription "My Subscription"

# Verify access
az account show
```

### Required Permissions

Your Azure account needs these permissions:
- **Synapse Contributor** on the Synapse workspace
- **Storage Blob Data Contributor** on associated storage
- **Spark Pool Contributor** for Spark pool management

## Synapse Spark Versions

Spark Optima supports Azure Synapse Spark versions:

| Synapse Version | Spark Version | Scala | Status |
|-----------------|---------------|-------|--------|
| Spark 2.4 | 2.4.4 | 2.11 | Deprecated |
| Spark 3.1 | 3.1.2 | 2.12 | Supported |
| Spark 3.2 | 3.2.0 | 2.12 | Supported |
| Spark 3.3 | 3.3.1 | 2.12 | Supported |
| Spark 3.4 | 3.4.1 | 2.12 | Supported |

## Configuration

### Basic Synapse Configuration

```python
from spark_optima import Optimizer

optimizer = Optimizer(
    platform="azure_synapse",
    spark_version="3.4.0"
)
```

### Node Size Selection

Azure Synapse offers different node sizes:

| Node Size | vCPU | Memory | Use Case |
|-----------|------|--------|----------|
| Small | 4 | 32 GB | Development, testing |
| Medium | 8 | 64 GB | Small production jobs |
| Large | 16 | 128 GB | Medium production jobs |
| XLarge | 32 | 256 GB | Large production jobs |
| XXLarge | 64 | 512 GB | Very large jobs |

Spark Optima recommends optimal node sizes based on workload.

### Spark Pool Configuration

```python
from spark_optima.platforms.models import ResourceSpec

resources = ResourceSpec(
    cpu_cores=32,          # Total cores needed
    memory_gb=256.0,       # Total memory needed
    disk_gb=1000.0
)

result = optimizer.optimize(
    code_path="./synapse_job.py",
    resources=resources,
    data_profile={"size_gb": 1000, "format": "parquet"},
    resource_constraints={
        "max_cost_usd": 150.0,
        "enable_autoscaling": True
    }
)
```

## Usage Examples

### Example 1: Basic Synapse Optimization

```python
from spark_optima import Optimizer

optimizer = Optimizer(
    platform="azure_synapse",
    spark_version="3.4.0"
)

result = optimizer.optimize(
    code_path="./synapse_etl.py",
    data_profile={
        "size_gb": 500,
        "format": "parquet",
        "location": "abfss://container@storage.dfs.core.windows.net/data/"
    },
    resource_constraints={
        "max_cost_usd": 75.0,
        "max_executors": 50
    }
)

print(f"Estimated cost: ${result.metadata.get('estimated_cost', 0):.2f}")
print(f"Node size: {result.platform_specific.get('node_size', 'Medium')}")
```

### Example 2: CLI Usage

```bash
# Optimize for Azure Synapse
spark-optima optimize \
  -c ./synapse_job.py \
  -p azure_synapse \
  -s 3.4.0 \
  -d 1000 \
  -f parquet \
  -o json > synapse_result.json

# Export configuration
spark-optima export \
  -r synapse_result.json \
  -f azure-synapse \
  -o synapse_config.json
```

### Example 3: Programmatic Pool Configuration

```python
from spark_optima import Optimizer
from azure.identity import DefaultAzureCredential
from azure.synapse.spark import SparkClient
import json

# Optimize
optimizer = Optimizer(platform="azure_synapse", spark_version="3.4.0")
result = optimizer.optimize(
    code_path="./analytics_job.py",
    data_profile={"size_gb": 500, "format": "parquet"}
)

# Create or update Spark pool
credential = DefaultAzureCredential()
client = SparkClient(
    credential=credential,
    subscription_id="your-subscription-id",
    workspace_name="your-workspace",
    resource_group_name="your-resource-group"
)

# Extract configuration
pool_config = result.platform_specific.get('spark_pool_config', {})

# Update pool configuration
pool = client.spark_pool.begin_create_or_update_spark_pool(
    spark_pool_name="optimized-pool",
    spark_pool_info={
        "location": "eastus",
        "node_size": pool_config.get('node_size', 'Medium'),
        "node_size_family": "MemoryOptimized",
        "auto_scale": {
            "enabled": True,
            "min_node_count": pool_config.get('min_nodes', 3),
            "max_node_count": pool_config.get('max_nodes', 20)
        },
        "spark_version": "3.4",
        "spark_config_properties": {
            "configuration_type": "Customized",
            "content": result.configuration
        }
    }
)

print(f"Pool configuration updated")
```

## Synapse-Specific Features

### Auto-Pause Configuration

Save costs with auto-pause:

```python
resource_constraints = {
    "enable_auto_pause": True,
    "auto_pause_delay_minutes": 15  # Pause after 15 min idle
}

result = optimizer.optimize(
    code_path="./synapse_job.py",
    resource_constraints=resource_constraints
)
```

### Dynamic Allocation

```python
resource_constraints = {
    "enable_dynamic_allocation": True,
    "min_executors": 2,
    "max_executors": 50,
    "executor_memory": "8g"
}

result = optimizer.optimize(
    code_path="./variable_load_job.py",
    resource_constraints=resource_constraints
)
```

### ADLS Gen2 Integration

```python
data_profile = {
    "size_gb": 1000,
    "format": "parquet",
    "location": "abfss://container@storage.dfs.core.windows.net/data/",
    "storage_account": "mystorageaccount",
    "file_system": "data"
}

resource_constraints = {
    "storage_endpoints": [
        "abfss://container@storage.dfs.core.windows.net/"
    ]
}

result = optimizer.optimize(
    code_path="./adls_job.py",
    data_profile=data_profile,
    resource_constraints=resource_constraints
)
```

### Workspace Packages

```python
# For jobs requiring additional libraries
resource_constraints = {
    "workspace_packages": [
        "library1-1.0-py3-none-any.whl",
        "library2-2.0-py3-none-any.whl"
    ],
    "session_level_packages": True
}
```

## Cost Optimization

### Understanding Synapse Pricing

Azure Synapse Spark pricing components:
- **vCore-hours** - Based on node size and duration
- **Storage** - ADLS Gen2 storage costs
- **Data transfer** - Egress charges

### Cost-Aware Optimization

```python
# Optimize for cost
result = optimizer.optimize(
    code_path="./job.py",
    objectives=["minimize_cost"],
    resource_constraints={
        "max_cost_usd": 50.0,
        "vcore_hour_rate": 0.18,  # Your region's rate
        "enable_auto_pause": True,
        "auto_pause_delay_minutes": 10
    }
)

print(f"Estimated vCore-hours: {result.metadata.get('estimated_vcore_hours', 0):.2f}")
print(f"Estimated cost: ${result.metadata.get('estimated_cost', 0):.2f}")
```

### Session-Level vs Job-Level

```python
# Session-level (interactive notebooks)
resource_constraints = {
    "session_type": "session",  # vs "job"
    "driver_memory": "28g",
    "driver_cores": 4
}

# Job-level (production pipelines)
resource_constraints = {
    "session_type": "job",
    "enable_auto_pause": True
}
```

## Monitoring and Diagnostics

### Synapse Studio Integration

Access Spark UI and logs:
1. Open Synapse Studio
2. Navigate to **Monitor** → **Apache Spark applications**
3. Select your job
4. View **Spark History Server** and **Driver Logs**

### Log Analytics Integration

```python
resource_constraints = {
    "enable_diagnostic_logs": True,
    "log_analytics_workspace_id": "your-workspace-id",
    "diagnostic_settings": {
        "spark_logs": True,
        "driver_logs": True,
        "executor_logs": True
    }
}
```

## Best Practices

### 1. Use Auto-Pause for Development

```python
resource_constraints = {
    "enable_auto_pause": True,
    "auto_pause_delay_minutes": 15
}
```

### 2. Enable Autoscaling

```python
resource_constraints = {
    "enable_autoscaling": True,
    "min_nodes": 3,
    "max_nodes": 50
}
```

### 3. Use Delta Lake for Large Datasets

```python
data_profile = {
    "format": "delta",
    "size_gb": 1000,
    "location": "abfss://container@storage.dfs.core.windows.net/delta/"
}
```

### 4. Configure ADLS Gen2 Access

```python
resource_constraints = {
    "storage_endpoints": [
        "abfss://raw@storage.dfs.core.windows.net/",
        "abfss://processed@storage.dfs.core.windows.net/"
    ],
    "credential_type": "ManagedIdentity"  # or "ServicePrincipal"
}
```

### 5. Use Workspace Packages for Dependencies

```python
resource_constraints = {
    "workspace_packages": [
        "custom-lib-1.0-py3-none-any.whl"
    ]
}
```

## CI/CD Integration

### GitHub Actions for Azure Synapse

```yaml
# .github/workflows/synapse-deploy.yml
name: Deploy to Azure Synapse

on:
  push:
    branches: [ main ]

jobs:
  optimize-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install spark-optima
          pip install azure-identity azure-synapse-spark
      
      - name: Optimize configuration
        run: |
          spark-optima optimize \
            -c ./jobs/synapse_job.py \
            -p azure_synapse \
            -d 500 \
            -o json > synapse_config.json
      
      - name: Deploy to Synapse
        run: |
          python -c "
          from azure.identity import DefaultAzureCredential
          from azure.synapse.spark import SparkClient
          import json
          
          credential = DefaultAzureCredential()
          client = SparkClient(
              credential=credential,
              subscription_id='${{ secrets.AZURE_SUBSCRIPTION_ID }}',
              workspace_name='${{ secrets.SYNAPSE_WORKSPACE }}',
              resource_group_name='${{ secrets.RESOURCE_GROUP }}'
          )
          
          with open('synapse_config.json') as f:
              config = json.load(f)
          
          # Update Spark pool
          client.spark_pool.begin_create_or_update_spark_pool(
              spark_pool_name='production-pool',
              spark_pool_info=config['platform_specific']['spark_pool_config']
          )
          
          print('Spark pool updated successfully')
          "
```

## Troubleshooting

### Issue: Pool startup too slow

```python
# Keep minimum nodes running
resource_constraints = {
    "enable_autoscaling": True,
    "min_nodes": 3,  # Keep at least 3 nodes warm
    "max_nodes": 20
}

# Or disable auto-pause for critical jobs
resource_constraints = {
    "enable_auto_pause": False
}
```

### Issue: High costs

```python
# Enable aggressive auto-pause
resource_constraints = {
    "enable_auto_pause": True,
    "auto_pause_delay_minutes": 10
}

# Optimize for cost
result = optimizer.optimize(
    code_path="./job.py",
    objectives=["minimize_cost"]
)
```

### Issue: Out of memory

```python
# Use larger node size
resources = ResourceSpec(
    cpu_cores=32,
    memory_gb=256.0
)

# Or increase executor memory
resource_constraints = {
    "executor_memory": "16g",
    "executor_cores": 4
}
```

### Issue: Storage access denied

```python
# Ensure proper storage configuration
resource_constraints = {
    "storage_endpoints": [
        "abfss://container@storage.dfs.core.windows.net/"
    ],
    "credential_type": "ManagedIdentity"
}

# Verify RBAC permissions in Azure Portal
# Storage Account → Access Control (IAM)
```

## Next Steps

- Explore [AWS Glue](./aws-glue.md)
- Learn about [Databricks](./databricks.md)
- Read the [Configuration Guide](../user-guide/configuration.md)
- See [Python API Guide](../user-guide/api.md)