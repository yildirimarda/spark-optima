# Optimizer API Reference

This page documents the `Optimizer` class, the main entry point for Spark configuration optimization.

::: spark_optima.core.optimizer.Optimizer
    handler: python
    options:
      members:
        - __init__
        - optimize
        - get_heuristic_config
        - get_bayesian_result
        - get_last_result
      show_root_heading: true
      show_source: true

## Class Overview

The `Optimizer` class provides the primary interface for optimizing Apache Spark configurations. It combines heuristic rules with Bayesian optimization to find optimal configurations for your specific workload.

## Initialization

```python
from spark_optima import Optimizer

optimizer = Optimizer(
    platform="databricks",
    spark_version="3.5.0",
    optimization_mode="simulation"
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `platform` | `str` | **required** | Target platform: `"local"`, `"aws_glue"`, `"databricks"`, `"azure_synapse"` |
| `spark_version` | `str` | `"3.5.0"` | Spark version to optimize for |
| `optimization_mode` | `str` | `"simulation"` | Optimization mode: `"simulation"` or `"execution"` |

### Raises

- `ValueError` - If platform or optimization_mode is invalid
- `ValueError` - If spark_version is not available

## Methods

### optimize()

Run optimization for the given Spark code.

```python
result = optimizer.optimize(
    code_path="./my_job.py",
    resources=ResourceSpec(cpu_cores=16, memory_gb=64),
    data_profile={"size_gb": 100, "format": "parquet"},
    resource_constraints={"max_memory_gb": 128},
    use_bayesian=True,
    bayesian_trials=50,
    bayesian_timeout_minutes=30,
    objectives=["minimize_time"]
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `code_path` | `str \| Path \| None` | `None` | Path to Spark application code file |
| `resources` | `ResourceSpec \| None` | `None` | Resource specifications |
| `data_profile` | `dict[str, Any] \| None` | `None` | Data characteristics |
| `resource_constraints` | `dict[str, Any] \| None` | `None` | Resource limits |
| `use_bayesian` | `bool` | `True` | Whether to use Bayesian optimization |
| `bayesian_trials` | `int` | `50` | Number of Bayesian optimization trials |
| `bayesian_timeout_minutes` | `int \| None` | `None` | Timeout for Bayesian optimization |
| `objectives` | `list[str] \| None` | `None` | List of optimization objectives |

#### Returns

`OptimizationResult` - Object containing the optimal configuration and metadata.

#### Raises

- `FileNotFoundError` - If the code file does not exist
- `ValueError` - If parameters are invalid

### get_heuristic_config()

Get the heuristic baseline configuration.

```python
heuristic_config = optimizer.get_heuristic_config()
```

#### Returns

`dict[str, Any] \| None` - Heuristic configuration dictionary or None if not run.

### get_bayesian_result()

Get the Bayesian optimization result.

```python
bayesian_result = optimizer.get_bayesian_result()
```

#### Returns

`BayesianOptimizationResult \| None` - Bayesian optimization result or None if not run.

### get_last_result()

Get the last optimization result.

```python
last_result = optimizer.get_last_result()
```

#### Returns

`OptimizationResult \| None` - Last optimization result or None if not run.

## Usage Examples

### Basic Usage

```python
from spark_optima import Optimizer

# Initialize
optimizer = Optimizer(platform="local", spark_version="3.5.0")

# Run optimization
result = optimizer.optimize(code_path="./job.py")

# Access results
print(result.configuration)
print(result.estimated_time_minutes)
```

### With Resource Constraints

```python
from spark_optima import Optimizer
from spark_optima.platforms.models import ResourceSpec

optimizer = Optimizer(platform="databricks")

resources = ResourceSpec(cpu_cores=32, memory_gb=128)

result = optimizer.optimize(
    code_path="./job.py",
    resources=resources,
    resource_constraints={
        "max_memory_gb": 256,
        "max_cost_usd": 100.0
    }
)
```

### Custom Objectives

```python
optimizer = Optimizer(platform="aws_glue")

result = optimizer.optimize(
    code_path="./job.py",
    objectives=["minimize_cost"],
    resource_constraints={"max_cost_usd": 50.0}
)
```

### Without Bayesian Optimization

```python
optimizer = Optimizer(platform="local")

# Fast optimization using only heuristics
result = optimizer.optimize(
    code_path="./job.py",
    use_bayesian=False
)
```

## Attributes

### platform

The target platform.

```python
print(optimizer.platform)  # "databricks"
```

### spark_version

The Spark version.

```python
print(optimizer.spark_version)  # "3.5.0"
```

### optimization_mode

The optimization mode.

```python
print(optimizer.optimization_mode)  # "simulation"
```

### config_database

The configuration database.

```python
db = optimizer.config_database
versions = db.get_available_versions()
```

### heuristic_engine

The heuristic optimization engine.

```python
engine = optimizer.heuristic_engine
config = engine.evaluate(...)
```

## Advanced Usage

### Inspecting Intermediate Results

```python
optimizer = Optimizer(platform="databricks")

# Run optimization
result = optimizer.optimize(code_path="./job.py")

# Get heuristic baseline
heuristic = optimizer.get_heuristic_config()
print("Heuristic config:", heuristic)

# Get Bayesian details
bayesian = optimizer.get_bayesian_result()
if bayesian:
    print(f"Trials: {len(bayesian.all_trials)}")
    print(f"Best value: {bayesian.best_value}")
```

### Iterative Optimization

```python
optimizer = Optimizer(platform="databricks")

# First pass - quick
result1 = optimizer.optimize(
    code_path="./job.py",
    bayesian_trials=20
)

# Second pass - deeper with constraints
result2 = optimizer.optimize(
    code_path="./job.py",
    bayesian_trials=100,
    resource_constraints={
        "max_cost_usd": result1.metadata.get("estimated_cost", 100) * 0.8
    }
)
```

## See Also

- [OptimizationResult](result.md) - Result object documentation
- [Configuration Guide](../user-guide/configuration.md) - Configuration options
- [Python API Guide](../user-guide/api.md) - User guide for Python API