# OptimizationResult API Reference

This page documents the `OptimizationResult` class, which contains the results of a Spark configuration optimization.

::: spark_optima.core.result.OptimizationResult
    handler: python
    options:
      members:
        - __init__
        - to_dict
        - get_platform_config
        - get_top_suggestions
      show_root_heading: true
      show_source: true

## Class Overview

The `OptimizationResult` class encapsulates all information about an optimization run, including:
- The optimal Spark configuration
- Estimated performance metrics
- Code improvement suggestions
- Platform-specific settings
- Optimization metadata

## Properties

### configuration

The optimized Spark configuration parameters.

```python
result: OptimizationResult = optimizer.optimize(...)
config = result.configuration

# Access specific parameters
print(config["spark.executor.memory"])  # "8g"
print(config["spark.executor.cores"])   # "4"
print(config["spark.sql.adaptive.enabled"])  # "true"
```

**Type:** `dict[str, Any]`

### estimated_time_minutes

Estimated execution time in minutes based on the optimized configuration.

```python
print(f"Estimated time: {result.estimated_time_minutes:.1f} minutes")
```

**Type:** `float`

### confidence_score

Confidence score for the optimization result (0.0 to 1.0).

```python
print(f"Confidence: {result.confidence_score:.0%}")  # "85%"
```

Higher scores indicate more reliable estimates:
- **0.9+** - Very high confidence
- **0.8-0.9** - High confidence
- **0.7-0.8** - Moderate confidence
- **<0.7** - Lower confidence, consider more trials

**Type:** `float`

### code_suggestions

List of code improvement suggestions.

```python
for suggestion in result.code_suggestions:
    print(f"Line {suggestion.line_number}: {suggestion.issue_type}")
    print(f"  {suggestion.suggestion}")
```

**Type:** `list[CodeSuggestion]`

### platform_specific

Platform-specific configuration and settings.

```python
platform_config = result.platform_specific
print(platform_config["platform"])          # "databricks"
print(platform_config["spark_version"])     # "3.5.0"

# Platform-specific settings
if "cluster_config" in platform_config:
    print(platform_config["cluster_config"])
```

**Type:** `dict[str, Any]`

### metadata

Additional metadata about the optimization run.

```python
metadata = result.metadata
print(f"Platform: {metadata['platform']}")
print(f"Spark Version: {metadata['spark_version']}")
print(f"Optimization Mode: {metadata['optimization_mode']}")
print(f"Bayesian Trials: {metadata['bayesian_trials']}")
```

**Type:** `dict[str, Any]`

## Methods

### to_dict()

Convert the result to a dictionary for serialization.

```python
result_dict = result.to_dict()

# Save to JSON
import json
with open("result.json", "w") as f:
    json.dump(result_dict, f, indent=2)
```

#### Returns

`dict[str, Any]` - Dictionary representation of the result.

### get_platform_config()

Get platform-specific configuration for a given platform.

```python
# Get Databricks-specific config
db_config = result.get_platform_config("databricks")

# Get AWS Glue-specific config
glue_config = result.get_platform_config("aws_glue")
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `platform` | `str` | Platform name |

#### Returns

`dict[str, Any]` - Platform-specific configuration.

### get_top_suggestions()

Get the top N code suggestions by severity.

```python
# Get top 3 suggestions
top_suggestions = result.get_top_suggestions(n=3)

# Get all critical and high severity suggestions
critical = [s for s in result.code_suggestions 
           if s.severity in ("critical", "high")]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | `5` | Number of suggestions to return |

#### Returns

`list[CodeSuggestion]` - List of top N suggestions.

## CodeSuggestion

Individual code improvement suggestion.

### Properties

#### line_number

Line number in the source file where the issue was found.

```python
print(f"Line {suggestion.line_number}: {suggestion.description}")
```

**Type:** `int`

#### issue_type

Type of issue detected.

Common issue types:
- `missing_broadcast_hint` - Missing broadcast hint for small DataFrame
- `unnecessary_shuffle` - Unnecessary shuffle operation
- `caching_issue` - Inefficient caching pattern
- `udf_usage` - UDF that could be replaced with built-in functions
- `skewed_data` - Potential data skew detected

```python
if suggestion.issue_type == "missing_broadcast_hint":
    print("Add broadcast hint to optimize join")
```

**Type:** `str`

#### description

Description of the issue.

```python
print(suggestion.description)
# "Small DataFrame being joined without broadcast hint"
```

**Type:** `str`

#### suggestion

Suggested fix for the issue.

```python
print(suggestion.suggestion)
# "Use broadcast(df_small) for DataFrames < 10MB"
```

**Type:** `str`

#### severity

Severity level of the issue.

Levels:
- `critical` - Must fix, will likely cause failures
- `high` - Strongly recommended, significant impact
- `medium` - Good practice, moderate impact
- `low` - Minor improvement, optional

```python
if suggestion.severity == "critical":
    print("⚠️  Critical issue - must fix!")
```

**Type:** `str`

## Usage Examples

### Basic Result Processing

```python
from spark_optima import Optimizer

optimizer = Optimizer(platform="databricks")
result = optimizer.optimize(code_path="./job.py")

# Print basic info
print(f"Config: {result.configuration}")
print(f"Time: {result.estimated_time_minutes} min")
print(f"Confidence: {result.confidence_score:.0%}")
```

### Processing Code Suggestions

```python
result = optimizer.optimize(code_path="./job.py")

# Group suggestions by severity
from collections import defaultdict
by_severity = defaultdict(list)

for suggestion in result.code_suggestions:
    by_severity[suggestion.severity].append(suggestion)

# Print critical issues first
if "critical" in by_severity:
    print("🚨 Critical Issues:")
    for s in by_severity["critical"]:
        print(f"  Line {s.line_number}: {s.description}")

# Print high priority suggestions
if "high" in by_severity:
    print("⚠️  High Priority:")
    for s in by_severity["high"]:
        print(f"  Line {s.line_number}: {s.suggestion}")
```

### Saving and Loading Results

```python
import json
from spark_optima.core.result import OptimizationResult

# Save result
result = optimizer.optimize(code_path="./job.py")
with open("result.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)

# Load result
with open("result.json") as f:
    data = json.load(f)
    loaded_result = OptimizationResult(**data)

print(loaded_result.configuration)
```

### Platform-Specific Export

```python
result = optimizer.optimize(code_path="./job.py")

# Export for different platforms
platforms = ["databricks", "aws_glue", "azure_synapse"]

for platform in platforms:
    config = result.get_platform_config(platform)
    with open(f"{platform}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Exported {platform} configuration")
```

### Generating Reports

```python
def generate_report(result: OptimizationResult) -> str:
    """Generate a human-readable optimization report."""
    
    lines = []
    lines.append("=" * 60)
    lines.append("SPARK OPTIMA OPTIMIZATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Performance estimates
    lines.append("PERFORMANCE ESTIMATES")
    lines.append("-" * 60)
    lines.append(f"Estimated Time: {result.estimated_time_minutes:.1f} minutes")
    lines.append(f"Confidence Score: {result.confidence_score:.0%}")
    lines.append("")
    
    # Configuration
    lines.append("RECOMMENDED CONFIGURATION")
    lines.append("-" * 60)
    for key, value in result.configuration.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    
    # Code suggestions
    if result.code_suggestions:
        lines.append("CODE IMPROVEMENTS")
        lines.append("-" * 60)
        for s in result.get_top_suggestions(5):
            lines.append(f"Line {s.line_number} [{s.severity}]: {s.suggestion}")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)

# Generate and save report
result = optimizer.optimize(code_path="./job.py")
report = generate_report(result)

with open("optimization_report.txt", "w") as f:
    f.write(report)

print(report)
```

## Validation

### Validating Results

```python
result = optimizer.optimize(code_path="./job.py")

# Check if result is valid
if result.confidence_score < 0.7:
    print("Warning: Low confidence score")
    print("Consider:")
    print("  - Increasing bayesian_trials")
    print("  - Providing more accurate data_profile")
    print("  - Using execution mode for better estimates")

# Validate configuration
required_configs = [
    "spark.executor.memory",
    "spark.executor.cores",
    "spark.driver.memory"
]

missing = [c for c in required_configs if c not in result.configuration]
if missing:
    print(f"Warning: Missing configs: {missing}")
```

## See Also

- [Optimizer](optimizer.md) - Optimizer class documentation
- [Configuration Guide](../user-guide/configuration.md) - Configuration options
- [Python API Guide](../user-guide/api.md) - User guide for Python API