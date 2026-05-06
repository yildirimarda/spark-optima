# Configuration Guide

This guide covers all aspects of configuring Spark Optima for your optimization tasks.

## Overview

Spark Optima optimizes over **200+ Spark configuration parameters** across multiple categories. Understanding these configurations helps you get the most out of the optimization process.

## Configuration Categories

### Memory Configuration

Memory settings are critical for Spark performance:

```python
# Key memory parameters managed by Spark Optima
memory_configs = {
    "spark.driver.memory": "4g",                    # Driver memory
    "spark.executor.memory": "8g",                  # Executor memory
    "spark.executor.memoryOverhead": "1g",          # Overhead for non-heap
    "spark.memory.fraction": "0.6",                 # Fraction for execution/storage
    "spark.memory.storageFraction": "0.5",          # Storage fraction
    "spark.driver.memoryOverhead": "1g",            # Driver overhead
}
```

**How Spark Optima optimizes memory:**
- Calculates optimal driver/executor memory based on available resources
- Adjusts memory overhead based on executor count and data size
- Tunes memory fractions based on workload (caching-heavy vs compute-heavy)

### CPU and Parallelism Configuration

```python
cpu_configs = {
    "spark.executor.cores": "4",                    # Cores per executor
    "spark.default.parallelism": "200",             # Default partitions
    "spark.sql.shuffle.partitions": "400",          # Shuffle partitions
    "spark.scheduler.mode": "FIFO",                 # Scheduling mode
}
```

**Optimization strategies:**
- Sets executor cores based on platform and data size
- Calculates optimal parallelism (typically 2-3x total cores)
- Adjusts shuffle partitions based on data characteristics

### Shuffle Configuration

```python
shuffle_configs = {
    "spark.shuffle.file.buffer": "32k",             # Shuffle buffer size
    "spark.shuffle.spill.compress": "true",         # Compress spilled data
    "spark.reducer.maxSizeInFlight": "48m",         # Reducer buffer
    "spark.shuffle.compress": "true",               # Compress shuffle
}
```

**When Spark Optima tunes these:**
- Large shuffles (joins, aggregations)
- Spill-heavy workloads
- Network-constrained environments

### SQL and Adaptive Query Execution

```python
sql_configs = {
    "spark.sql.adaptive.enabled": "true",           # Enable AQE
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.sql.autoBroadcastJoinThreshold": "10MB",
    "spark.sql.broadcastTimeout": "300",
}
```

**AQE Benefits:**
- Dynamically coalesces partitions
- Handles skewed joins automatically
- Optimizes query plans at runtime

### Dynamic Allocation

```python
dynamic_configs = {
    "spark.dynamicAllocation.enabled": "true",
    "spark.dynamicAllocation.minExecutors": "2",
    "spark.dynamicAllocation.maxExecutors": "50",
    "spark.dynamicAllocation.initialExecutors": "4",
    "spark.dynamicAllocation.executorIdleTimeout": "60s",
}
```

**Spark Optima recommendations:**
- Enables dynamic allocation for variable workloads
- Sets min/max based on resource constraints
- Configures idle timeout based on job patterns

## Resource Constraints

### Specifying Resource Limits

```python
from spark_optima import Optimizer

optimizer = Optimizer(platform="databricks", spark_version="3.5.0")

result = optimizer.optimize(
    code_path="./job.py",
    data_profile={"size_gb": 500, "format": "parquet"},
    resource_constraints={
        "max_memory_gb": 256,           # Maximum total memory
        "max_cost_usd": 100.0,          # Maximum cost per run
        "max_executors": 100,           # Maximum parallel executors
        "max_cores": 400,               # Maximum total cores
    }
)
```

### Platform-Specific Resource Specifications

Using `ResourceSpec` for more control:

```python
from spark_optima.platforms.models import ResourceSpec

resources = ResourceSpec(
    cpu_cores=64,           # Total CPU cores available
    memory_gb=256.0,        # Total memory in GB
    disk_gb=1000.0,         # Disk space in GB
    network_gbps=10.0,      # Network bandwidth
)

result = optimizer.optimize(
    code_path="./job.py",
    resources=resources,
)
```

## Data Profile Configuration

### Basic Data Profile

```python
data_profile = {
    "size_gb": 1000,                    # Data size in GB
    "format": "parquet",                # Data format
    "compression": "snappy",            # Compression codec
    "partitioning": {"type": "date", "column": "created_at"},
}
```

### Advanced Data Profile

```python
data_profile = {
    "size_gb": 500,
    "format": "delta",
    "compression": "zstd",
    "schema": {
        "fields": [
            {"name": "user_id", "type": "long", "nullable": False},
            {"name": "event_type", "type": "string", "nullable": True},
            {"name": "timestamp", "type": "timestamp", "nullable": False},
            {"name": "amount", "type": "decimal(10,2)", "nullable": True},
        ]
    },
    "partitioning": {
        "type": "multi_column",
        "columns": ["year", "month", "day"],
    },
    "statistics": {
        "num_files": 10000,
        "avg_file_size_mb": 50,
        "num_partitions": 5000,
    },
}
```

## Optimization Objectives

### Available Objectives

```python
# Minimize execution time (default)
objectives = ["minimize_time"]

# Minimize cost
objectives = ["minimize_cost"]

# Multi-objective optimization
objectives = ["minimize_time", "minimize_cost"]

# Custom weighting
objectives = ["balanced"]  # Balances time and cost
```

### Using Objectives

```python
result = optimizer.optimize(
    code_path="./job.py",
    objectives=["minimize_cost"],  # Prioritize cost savings
    resource_constraints={"max_cost_usd": 50.0},
)
```

## Bayesian Optimization Settings

### Controlling the Search

```python
result = optimizer.optimize(
    code_path="./job.py",
    use_bayesian=True,
    bayesian_trials=100,               # Number of trials
    bayesian_timeout_minutes=60,       # Timeout limit
)
```

### When to Adjust These Settings

| Scenario | Recommended Trials | Timeout |
|----------|-------------------|---------|
| Quick exploration | 20-30 | 15 min |
| Standard optimization | 50 | 30 min |
| Deep optimization | 100+ | 60+ min |
| Production tuning | 200 | No limit |

### Disabling Bayesian Optimization

For fast heuristic-only optimization:

```python
result = optimizer.optimize(
    code_path="./job.py",
    use_bayesian=False,  # Use only heuristics
)
```

## Platform-Specific Configuration

### Local Platform

```python
optimizer = Optimizer(
    platform="local",
    spark_version="3.5.0",
    optimization_mode="simulation",  # or "execution"
)
```

**Local-specific considerations:**
- Limited by local machine resources
- Execution mode requires Spark installation
- Good for development and testing

### AWS Glue

```python
optimizer = Optimizer(
    platform="aws_glue",
    spark_version="3.5.0",
)

# Export configuration for Glue
# Spark Optima will generate Glue-specific settings
```

**Glue-specific configurations:**
- Glue version compatibility (3.0, 4.0)
- Worker type selection (G.1X, G.2X, G.4X, G.8X)
- Job bookmark settings
- Connection configurations

### Databricks

```python
optimizer = Optimizer(
    platform="databricks",
    spark_version="3.5.0",
)
```

**Databricks-specific features:**
- DBR (Databricks Runtime) compatibility
- Cluster node type recommendations
- Autoscaling configurations
- Unity Catalog integration

### Azure Synapse

```python
optimizer = Optimizer(
    platform="azure_synapse",
    spark_version="3.5.0",
)
```

**Synapse-specific features:**
- Spark pool configuration
- Node size recommendations
- Auto-pause settings

## Advanced Configuration

### Custom Heuristic Rules

For advanced users, you can influence heuristic decisions:

```python
# Provide hints to the optimizer
resource_constraints = {
    "prefer_caching": True,             # Prefer memory caching
    "io_intensive": True,               # Workload is I/O bound
    "shuffle_heavy": True,              # Heavy shuffle operations
    "skewed_data": True,                # Data has skew
}
```

### Configuration Validation

Spark Optima validates all configurations:

```python
# Access validation results
heuristic_config = optimizer.get_heuristic_config()
validation_errors = optimizer.heuristic_engine.validate_config(heuristic_config)

if validation_errors:
    for error in validation_errors:
        print(f"Warning: {error}")
```

### Inspecting Intermediate Results

```python
# Get heuristic baseline (without Bayesian)
heuristic_config = optimizer.get_heuristic_config()
print("Heuristic configuration:")
for key, value in heuristic_config.items():
    print(f"  {key} = {value}")

# Get Bayesian optimization details
bayesian_result = optimizer.get_bayesian_result()
if bayesian_result:
    print(f"Trials completed: {len(bayesian_result.all_trials)}")
    print(f"Best trial value: {bayesian_result.best_value}")
```

## Best Practices

### 1. Start with Simulation Mode

```python
# Use simulation for initial exploration
optimizer = Optimizer(platform="databricks", optimization_mode="simulation")
result = optimizer.optimize(code_path="./job.py")

# Switch to execution for final tuning
optimizer = Optimizer(platform="databricks", optimization_mode="execution")
result = optimizer.optimize(code_path="./job.py")
```

### 2. Provide Accurate Data Profiles

More accurate data profiles yield better optimizations:

```python
# Good - specific information
data_profile = {
    "size_gb": 250,
    "format": "parquet",
    "compression": "snappy",
}

# Less optimal - vague information
data_profile = {"size_gb": 200}  # Underestimated
```

### 3. Use Resource Constraints Wisely

```python
# Be realistic about constraints
resource_constraints = {
    "max_memory_gb": 128,      # What you actually have
    "max_cost_usd": 25.0,      # Your budget
}

# Too restrictive may prevent good solutions
# Too loose wastes resources
```

### 4. Iterate and Refine

```python
# First optimization
result1 = optimizer.optimize(code_path="./job.py", bayesian_trials=30)

# Analyze results, then deeper optimization
result2 = optimizer.optimize(
    code_path="./job.py",
    bayesian_trials=100,
    resource_constraints={"max_cost_usd": result1.metadata.get("estimated_cost", 50) * 0.9}
)
```

## Configuration Reference

### All Spark Versions Supported

Spark Optima supports configuration databases for:
- Spark 3.0.x
- Spark 3.1.x
- Spark 3.2.x
- Spark 3.3.x
- Spark 3.4.x
- Spark 3.5.x
- Spark 4.0.x
- Spark 4.1.x

### Getting Available Versions

```python
from spark_optima.core.config_engine.database import ConfigDatabase

db = ConfigDatabase()
versions = db.get_available_versions()
print(f"Available versions: {versions}")
```

## Troubleshooting Configuration Issues

### Issue: Optimization produces unrealistic configurations

**Solution**: Add realistic constraints:

```python
resource_constraints = {
    "max_memory_gb": 64,      # Set realistic limits
    "max_executors": 20,
}
```

### Issue: Configuration fails on target platform

**Solution**: Verify platform compatibility:

```python
# Check platform-specific settings
platform_config = result.platform_specific
print(f"Platform: {platform_config['platform']}")
print(f"Spark version: {platform_config['spark_version']}")
```

### Issue: Bayesian optimization takes too long

**Solution**: Reduce trials or disable:

```python
result = optimizer.optimize(
    code_path="./job.py",
    use_bayesian=True,
    bayesian_trials=20,        # Fewer trials
    bayesian_timeout_minutes=10,  # Shorter timeout
)
```

---

**Next Steps:**
- Learn about [CLI Usage](cli.md)
- Explore [Python API](api.md)
- Read [Platform-Specific Guides](../platforms/)