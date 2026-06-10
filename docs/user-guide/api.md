# Python API Guide

This guide covers using Spark Optima programmatically via the Python API. For the HTTP/REST interface, see the [REST API Reference](rest-api.md).

## Overview

The Python API provides full access to Spark Optima's optimization capabilities, allowing you to integrate configuration optimization into your applications, notebooks, and pipelines.

## Quick Start

### Basic Usage

```python
from spark_optima import Optimizer

# Initialize optimizer
optimizer = Optimizer(
    platform="databricks",
    spark_version="3.5.0"
)

# Run optimization
result = optimizer.optimize(
    code_path="./my_spark_job.py",
    data_profile={"size_gb": 100, "format": "parquet"}
)

# Access results
print(f"Estimated time: {result.estimated_time_minutes} minutes")
print(f"Confidence: {result.confidence_score:.0%}")
```

## The Optimizer Class

### Initialization

```python
from spark_optima import Optimizer

optimizer = Optimizer(
    platform="databricks",              # Target platform
    spark_version="3.5.0",              # Spark version
    optimization_mode="simulation"      # "simulation" or "execution"
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `platform` | str | required | Target platform: "local", "aws_glue", "databricks", "azure_synapse" |
| `spark_version` | str | "3.5.0" | Spark version (e.g., "3.5.0", "4.0.0") |
| `optimization_mode` | str | "simulation" | Optimization mode: "simulation" or "execution" |

### The `optimize()` Method

```python
result = optimizer.optimize(
    # Input options
    code_path="./job.py",                       # Path to Spark code
    resources=None,                             # ResourceSpec or None
    data_profile=None,                          # Data characteristics
    resource_constraints=None,                  # Resource limits
    
    # Optimization options
    use_bayesian=True,                          # Enable Bayesian optimization
    bayesian_trials=50,                         # Number of trials
    bayesian_timeout_minutes=None,              # Timeout (None = no limit)
    objectives=None,                            # Optimization objectives
)
```

#### Complete Example

```python
from spark_optima import Optimizer
from spark_optima.platforms.models import ResourceSpec

# Initialize
optimizer = Optimizer(
    platform="aws_glue",
    spark_version="3.5.0",
    optimization_mode="simulation"
)

# Define resources
resources = ResourceSpec(
    cpu_cores=32,
    memory_gb=128.0,
    disk_gb=500.0
)

# Define data profile
data_profile = {
    "size_gb": 500,
    "format": "parquet",
    "compression": "snappy",
    "partitioning": {
        "type": "date",
        "column": "created_date"
    }
}

# Define constraints
constraints = {
    "max_memory_gb": 256,
    "max_cost_usd": 100.0,
    "max_executors": 50
}

# Run optimization
result = optimizer.optimize(
    code_path="./etl_job.py",
    resources=resources,
    data_profile=data_profile,
    resource_constraints=constraints,
    use_bayesian=True,
    bayesian_trials=75,
    bayesian_timeout_minutes=45,
    objectives=["minimize_time"]
)
```

## Working with Results

### OptimizationResult Object

```python
from spark_optima import OptimizationResult

# The optimize() method returns an OptimizationResult
result: OptimizationResult = optimizer.optimize(...)
```

#### Properties

```python
# Optimal configuration (dict of Spark parameters)
config = result.configuration
print(config["spark.executor.memory"])  # "8g"

# Estimated execution time in minutes
print(f"Estimated time: {result.estimated_time_minutes:.1f} min")

# Confidence score (0.0 to 1.0)
print(f"Confidence: {result.confidence_score:.0%}")

# Code improvement suggestions
for suggestion in result.code_suggestions:
    print(f"Line {suggestion.line_number}: {suggestion.suggestion}")

# Platform-specific configuration
platform_config = result.platform_specific
print(f"Platform: {platform_config['platform']}")

# Metadata
metadata = result.metadata
print(f"Trials: {metadata['bayesian_trials']}")
print(f"Heuristic config: {metadata['heuristic_config']}")
```

### Exporting Results

```python
# Export to JSON
import json
with open("result.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)

# Export to YAML
import yaml
with open("result.yaml", "w") as f:
    yaml.dump(result.to_dict(), f)

# Get specific platform config
databricks_config = result.get_platform_config("databricks")
aws_glue_config = result.get_platform_config("aws_glue")
```

## Code Suggestions

### Accessing Suggestions

```python
result = optimizer.optimize(code_path="./job.py")

for suggestion in result.code_suggestions:
    print(f"Line {suggestion.line_number}: {suggestion.issue_type}")
    print(f"  Issue: {suggestion.description}")
    print(f"  Suggestion: {suggestion.suggestion}")
    print(f"  Severity: {suggestion.severity}")
```

### Severity Levels

- **critical** - Must fix, will likely cause failures
- **high** - Strongly recommended, significant impact
- **medium** - Good practice, moderate impact
- **low** - Minor improvement, optional

### Example Suggestions

```python
# Example: Missing broadcast hint
suggestion = CodeSuggestion(
    line_number=25,
    issue_type="missing_broadcast_hint",
    description="Small DataFrame being joined without broadcast hint",
    suggestion="Use broadcast(df_small) for DataFrames < 10MB",
    severity="high"
)

# Example: Unnecessary shuffle
suggestion = CodeSuggestion(
    line_number=40,
    issue_type="unnecessary_shuffle",
    description="Repartition after filter may be unnecessary",
    suggestion="Remove repartition() if downstream operations don't require it",
    severity="medium"
)
```

## Advanced Usage

### Multi-Objective Optimization

```python
# Optimize for both time and cost
result = optimizer.optimize(
    code_path="./job.py",
    objectives=["minimize_time", "minimize_cost"],
    resource_constraints={"max_cost_usd": 50.0}
)
```

### Iterative Optimization

```python
# Step 1: Quick exploration with simulation
optimizer = Optimizer(platform="databricks", optimization_mode="simulation")
result_sim = optimizer.optimize(
    code_path="./job.py",
    bayesian_trials=20
)

print(f"Simulation result: {result_sim.estimated_time_minutes} min")

# Step 2: Fine-tune with execution mode
optimizer = Optimizer(platform="databricks", optimization_mode="execution")
result_exec = optimizer.optimize(
    code_path="./job.py",
    bayesian_trials=100,
    resource_constraints={
        "max_memory_gb": 128,
        "max_cost_usd": result_sim.metadata.get("estimated_cost", 100)
    }
)

print(f"Execution result: {result_exec.estimated_time_minutes} min")
```

### Batch Optimization

```python
import os
from pathlib import Path

# Optimize multiple jobs
jobs_dir = Path("./spark_jobs")
results = {}

for job_file in jobs_dir.glob("*.py"):
    job_name = job_file.stem
    print(f"Optimizing: {job_name}")
    
    result = optimizer.optimize(
        code_path=str(job_file),
        data_profile={"size_gb": 100, "format": "parquet"}
    )
    
    results[job_name] = result
    
    # Save individual result
    with open(f"results/{job_name}_config.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

# Summary report
print("\nOptimization Summary:")
print("-" * 60)
for name, result in results.items():
    print(f"{name:20s} | {result.estimated_time_minutes:6.1f} min | {result.confidence_score:.0%}")
```

### Custom Objective Functions

```python
# Define custom objective weights
def custom_objective(metrics):
    """Custom objective combining time, cost, and memory efficiency."""
    time_weight = 0.5
    cost_weight = 0.3
    memory_weight = 0.2
    
    return (
        time_weight * metrics.execution_time_seconds +
        cost_weight * metrics.estimated_cost_usd * 10 +
        memory_weight * metrics.peak_memory_gb * 100
    )

# Use custom objective (advanced usage)
# Note: This requires direct Bayesian optimizer access
from spark_optima.core.bayesian.optimizer import BayesianOptimizer
```

## Integration Examples

### Jupyter Notebook

```python
# In a Jupyter notebook
from spark_optima import Optimizer
import matplotlib.pyplot as plt

optimizer = Optimizer(platform="local")
result = optimizer.optimize(code_path="./job.py")

# Visualize configuration
config = result.configuration
memory_settings = {k: v for k, v in config.items() if "memory" in k}

plt.figure(figsize=(10, 6))
plt.bar(memory_settings.keys(), memory_settings.values())
plt.xticks(rotation=45)
plt.title("Recommended Memory Settings")
plt.tight_layout()
plt.show()
```

### Apache Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def optimize_spark_config(**context):
    from spark_optima import Optimizer
    
    optimizer = Optimizer(
        platform="databricks",
        spark_version="3.5.0"
    )
    
    result = optimizer.optimize(
        code_path="/opt/airflow/dags/spark_jobs/daily_etl.py",
        data_profile={"size_gb": 500, "format": "delta"}
    )
    
    # Push configuration to XCom
    context['ti'].xcom_push(key='spark_config', value=result.configuration)
    
    return f"Optimization complete. Estimated time: {result.estimated_time_minutes} min"

with DAG(
    'spark_optimization',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily'
) as dag:
    
    optimize_task = PythonOperator(
        task_id='optimize_config',
        python_callable=optimize_spark_config
    )
```

### MLflow Tracking

```python
import mlflow
from spark_optima import Optimizer

mlflow.set_experiment("spark_optimization")

with mlflow.start_run():
    optimizer = Optimizer(platform="databricks")
    
    result = optimizer.optimize(
        code_path="./ml_training.py",
        data_profile={"size_gb": 1000, "format": "parquet"}
    )
    
    # Log parameters
    mlflow.log_param("platform", "databricks")
    mlflow.log_param("spark_version", "3.5.0")
    mlflow.log_param("data_size_gb", 1000)
    
    # Log metrics
    mlflow.log_metric("estimated_time_minutes", result.estimated_time_minutes)
    mlflow.log_metric("confidence_score", result.confidence_score)
    mlflow.log_metric("num_trials", result.metadata.get("bayesian_trials", 0))
    
    # Log configuration as artifact
    with open("spark_config.json", "w") as f:
        json.dump(result.configuration, f, indent=2)
    mlflow.log_artifact("spark_config.json")
```

## Error Handling

### Common Exceptions

```python
from spark_optima import Optimizer

optimizer = Optimizer(platform="databricks")

try:
    result = optimizer.optimize(code_path="./job.py")
except FileNotFoundError as e:
    print(f"Code file not found: {e}")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except Exception as e:
    print(f"Optimization failed: {e}")
    # Fall back to heuristic configuration
    result = optimizer.get_heuristic_config()
```

### Graceful Degradation

```python
optimizer = Optimizer(platform="aws_glue")

# Attempt full optimization
try:
    result = optimizer.optimize(
        code_path="./job.py",
        use_bayesian=True,
        bayesian_trials=50
    )
except Exception as e:
    print(f"Bayesian optimization failed: {e}")
    print("Falling back to heuristic optimization...")
    
    # Retry with heuristics only
    result = optimizer.optimize(
        code_path="./job.py",
        use_bayesian=False
    )

# Results are always available
print(f"Configuration: {result.configuration}")
```

## Best Practices

### 1. Reuse Optimizer Instances

```python
# Good: Reuse optimizer for multiple jobs
optimizer = Optimizer(platform="databricks", spark_version="3.5.0")

for job in jobs:
    result = optimizer.optimize(code_path=job.path)

# Less optimal: Creating new optimizer each time
for job in jobs:
    optimizer = Optimizer(platform="databricks")  # Overhead each iteration
    result = optimizer.optimize(code_path=job.path)
```

### 2. Cache Results

```python
import hashlib
import json
from pathlib import Path

def get_cached_or_optimize(code_path, data_profile, cache_dir="./cache"):
    # Create cache key
    key_data = f"{code_path}:{json.dumps(data_profile, sort_keys=True)}"
    cache_key = hashlib.md5(key_data.encode()).hexdigest()
    cache_file = Path(cache_dir) / f"{cache_key}.json"
    
    # Check cache
    if cache_file.exists():
        with open(cache_file) as f:
            return OptimizationResult(**json.load(f))
    
    # Run optimization
    optimizer = Optimizer(platform="databricks")
    result = optimizer.optimize(code_path=code_path, data_profile=data_profile)
    
    # Cache result
    cache_file.parent.mkdir(exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(result.to_dict(), f)
    
    return result
```

### 3. Async Optimization (Advanced)

```python
import asyncio

async def optimize_async(optimizer, job_path):
    # Run optimization in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        optimizer.optimize,
        job_path
    )

# Optimize multiple jobs concurrently
async def optimize_all(jobs):
    optimizer = Optimizer(platform="databricks")
    
    tasks = [optimize_async(optimizer, job) for job in jobs]
    results = await asyncio.gather(*tasks)
    
    return results

# Run
results = asyncio.run(optimize_all(job_paths))
```

## API Reference

### Classes

#### `Optimizer`
Main optimization class. See initialization and `optimize()` method above.

#### `OptimizationResult`
Result container with properties:
- `configuration: dict[str, Any]` - Optimal Spark configuration
- `estimated_time_minutes: float` - Estimated execution time
- `confidence_score: float` - Confidence in results (0.0-1.0)
- `code_suggestions: list[CodeSuggestion]` - Code improvements
- `platform_specific: dict[str, Any]` - Platform configuration
- `metadata: dict[str, Any]` - Additional metadata

Methods:
- `to_dict()` - Convert to dictionary
- `get_platform_config(platform: str)` - Get platform-specific config
- `get_top_suggestions(n: int = 5)` - Get top N suggestions

#### `CodeSuggestion`
Code improvement suggestion with properties:
- `line_number: int` - Line in source file
- `issue_type: str` - Type of issue
- `description: str` - Issue description
- `suggestion: str` - Suggested fix
- `severity: str` - Severity level

---

**Next Steps:**
- Explore [Platform-Specific Guides](../platforms/)
- See the [REST API Reference](rest-api.md)
- See [CLI Usage Guide](cli.md)
- Read [Configuration Guide](configuration.md)