# Core API Reference

This page documents the core modules and classes in Spark Optima.

## Core Modules

### spark_optima.core.optimizer

Main optimizer class for Spark configuration optimization.

See [Optimizer API Reference](optimizer.md) for detailed documentation.

### spark_optima.core.result

Result classes for optimization output.

See [OptimizationResult API Reference](result.md) for detailed documentation.

## Supporting Modules

### Configuration Engine

The configuration engine manages Spark configuration parameters across different versions.

::: spark_optima.core.config_engine.database.ConfigDatabase
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: spark_optima.core.config_engine.models.ConfigSet
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: spark_optima.core.config_engine.models.ConfigParameter
    handler: python
    options:
      show_root_heading: true
      show_source: true

### Heuristics Engine

The heuristics engine applies rule-based optimizations.

::: spark_optima.core.heuristics.engine.HeuristicEngine
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: spark_optima.core.heuristics.context.EvaluationContext
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: spark_optima.core.heuristics.context.DataProfile
    handler: python
    options:
      show_root_heading: true
      show_source: true

### Bayesian Optimizer

The Bayesian optimizer fine-tunes configurations using Optuna.

::: spark_optima.core.bayesian.optimizer.BayesianOptimizer
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: spark_optima.core.bayesian.models.BayesianOptimizationResult
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: spark_optima.core.bayesian.models.SearchSpaceConfig
    handler: python
    options:
      show_root_heading: true
      show_source: true

## Platform Models

::: spark_optima.platforms.models.ResourceSpec
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: spark_optima.platforms.models.WorkerType
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: spark_optima.platforms.models.CostModel
    handler: python
    options:
      show_root_heading: true
      show_source: true

## Analysis Models

::: spark_optima.analysis.models.AnalysisResult
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: spark_optima.analysis.models.CodeSmell
    handler: python
    options:
      show_root_heading: true
      show_source: true

::: spark_optima.analysis.models.CodeRecommendation
    handler: python
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### Using ConfigDatabase

```python
from spark_optima.core.config_engine.database import ConfigDatabase

# Initialize database
db = ConfigDatabase()

# Get available versions
versions = db.get_available_versions()
print(f"Available versions: {versions}")

# Get configuration set for specific version
config_set = db.get_config_set("3.5.0")
print(f"Parameters: {list(config_set.parameters.keys())}")
```

### Using HeuristicContext

```python
from spark_optima.core.heuristics.context import HeuristicContext, DataProfile
from spark_optima.platforms.models import ResourceSpec

# Create data profile
data_profile = DataProfile(
    size_gb=100,
    format="parquet",
    compression="snappy"
)

# Create resource spec
resources = ResourceSpec(
    cpu_cores=16,
    memory_gb=64.0
)

# Create heuristic context
context = HeuristicContext(
    resources=resources,
    data_profile=data_profile,
    platform="databricks",
    spark_version="3.5.0"
)
```

### Using ResourceSpec

```python
from spark_optima.platforms.models import ResourceSpec, WorkerType, CostModel

# Define resource specification
resources = ResourceSpec(
    cpu_cores=32,
    memory_gb=128.0,
    disk_gb=1000.0,
    network_gbps=10.0
)

# Define worker type
worker = WorkerType(
    name="Standard",
    size="medium",
    resources=ResourceSpec(cpu_cores=4, memory_gb=16.0),
    cost=CostModel(
        unit_cost_per_hour=0.50,
        unit_name="instance"
    )
)
```

## Constants and Enums

### Severity Levels

```python
from spark_optima.analysis.models import Severity

# Available severity levels
Severity.CRITICAL  # "critical"
Severity.HIGH      # "high"
Severity.MEDIUM    # "medium"
Severity.LOW       # "low"
```

### Supported Platforms

```python
SUPPORTED_PLATFORMS = [
    "local",
    "aws_glue",
    "databricks",
    "azure_synapse"
]

SPARK_VERSIONS = [
    "3.0.0", "3.1.0", "3.2.0", "3.3.0",
    "3.4.0", "3.5.0", "4.0.0", "4.1.0"
]
```

## See Also

- [Optimizer API Reference](optimizer.md) - Main optimizer class
- [OptimizationResult API Reference](result.md) - Result classes
- [Configuration Guide](../user-guide/configuration.md) - User guide
- [Architecture Overview](../development/architecture.md) - System architecture