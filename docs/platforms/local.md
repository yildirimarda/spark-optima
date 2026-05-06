# Local Platform Guide

This guide covers using Spark Optima with local/standalone Spark deployments.

## Overview

The Local platform is ideal for:
- Development and testing
- Small to medium workloads
- Learning Spark Optima
- CI/CD pipelines

## Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows (WSL recommended)
- **Python**: 3.10+
- **Java**: 8 or 11 (required for Spark)
- **Memory**: 8GB+ RAM recommended

### Install Java

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install openjdk-11-jdk

# macOS
brew install openjdk@11

# Verify
java -version
```

### Install Apache Spark (Optional)

For execution mode, you need Spark installed:

```bash
# Download Spark 3.5.0
wget https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz

# Extract
tar -xzf spark-3.5.0-bin-hadoop3.tgz
sudo mv spark-3.5.0-bin-hadoop3 /opt/spark

# Set environment variables
export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# Verify
spark-submit --version
```

## Configuration

### Basic Local Configuration

```python
from spark_optima import Optimizer

optimizer = Optimizer(
    platform="local",
    spark_version="3.5.0",
    optimization_mode="simulation"  # or "execution"
)
```

### Resource Constraints

```python
from spark_optima.platforms.models import ResourceSpec

# Define local machine resources
resources = ResourceSpec(
    cpu_cores=8,           # Your machine's CPU cores
    memory_gb=32.0,        # Available memory
    disk_gb=500.0,         # Available disk
)

optimizer = Optimizer(platform="local")

result = optimizer.optimize(
    code_path="./my_job.py",
    resources=resources,
    data_profile={"size_gb": 10, "format": "parquet"}
)
```

## Usage Examples

### Example 1: Simple Local Optimization

```python
# optimize_local.py
from spark_optima import Optimizer

optimizer = Optimizer(
    platform="local",
    spark_version="3.5.0"
)

result = optimizer.optimize(
    code_path="./word_count.py",
    data_profile={"size_gb": 5, "format": "text"}
)

print("Optimal Configuration:")
for key, value in result.configuration.items():
    print(f"  {key} = {value}")

# Export for local execution
import json
with open("local_config.json", "w") as f:
    json.dump(result.configuration, f, indent=2)
```

### Example 2: Using spark-submit

```python
# After optimization, create spark-submit command
result = optimizer.optimize(code_path="./job.py")

# Build spark-submit command
config = result.configuration
spark_submit_cmd = f"""spark-submit \\
  --master local[*] \\
  --driver-memory {config.get('spark.driver.memory', '2g')} \\
  --executor-memory {config.get('spark.executor.memory', '4g')} \\
  --executor-cores {config.get('spark.executor.cores', '2')} \\
  --conf spark.sql.adaptive.enabled={config.get('spark.sql.adaptive.enabled', 'true')} \\
  job.py"""

print(spark_submit_cmd)
```

### Example 3: CLI Usage

```bash
# Optimize for local execution
spark-optima optimize \
  -c ./my_job.py \
  -p local \
  -d 10 \
  -f parquet \
  -m 16 \
  -o json > local_result.json

# Export as spark-submit command
spark-optima export -r local_result.json -f spark-submit -o run_local.sh
chmod +x run_local.sh
./run_local.sh
```

## Docker-Based Local Execution

### Using Docker

```bash
# Build image
docker build -f docker/Dockerfile -t spark-optima:local .

# Run optimization
docker run --rm \
  -v $(pwd)/jobs:/jobs \
  spark-optima:local \
  spark-optima optimize -c /jobs/my_job.py -p local -d 10
```

### Docker Compose Setup

```yaml
# docker-compose.local.yml
version: '3.8'
services:
  spark-optima:
    build:
      context: .
      dockerfile: docker/Dockerfile
    volumes:
      - ./jobs:/jobs
      - ./data:/data
    command: spark-optima optimize -c /jobs/my_job.py -p local -d 10
```

```bash
docker-compose -f docker-compose.local.yml up
```

## Simulation vs Execution Mode

### Simulation Mode

```python
# Fast estimation without running Spark
optimizer = Optimizer(
    platform="local",
    optimization_mode="simulation"
)

result = optimizer.optimize(code_path="./job.py")
# Fast: Returns in seconds
```

**Pros:**
- Very fast (seconds)
- No Spark installation needed
- Good for initial exploration

**Cons:**
- Estimates may vary from actual performance
- Doesn't measure real execution

### Execution Mode

```python
# Actual Spark execution for precise measurement
optimizer = Optimizer(
    platform="local",
    optimization_mode="execution"
)

result = optimizer.optimize(code_path="./job.py")
# Slower: Actually runs Spark jobs
```

**Pros:**
- Precise measurements
- Real performance data

**Cons:**
- Requires Spark installation
- Takes longer (minutes)
- Uses actual resources

## Local Development Workflow

### Step-by-Step Development

```bash
# 1. Create your Spark job
cat > word_count.py << 'EOF'
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("WordCount").getOrCreate()

text_file = spark.read.text("/path/to/input.txt")
words = text_file.selectExpr("explode(split(value, ' ')) as word")
word_counts = words.groupBy("word").count()
word_counts.write.csv("/path/to/output")

spark.stop()
EOF

# 2. Optimize configuration
spark-optima optimize -c word_count.py -p local -d 1 -o json > result.json

# 3. Review configuration
cat result.json | jq '.configuration'

# 4. Export and run
spark-optima export -r result.json -f spark-submit -o run.sh
bash run.sh

# 5. Iterate based on results
# Modify code, re-optimize, compare results
```

## Best Practices

### 1. Resource Management

```python
# Don't exceed your machine's resources
resources = ResourceSpec(
    cpu_cores=min(8, os.cpu_count()),  # Use available cores
    memory_gb=16,                       # Leave memory for OS
)
```

### 2. Data Location

```python
# Use local paths for local execution
data_profile = {
    "size_gb": 10,
    "format": "parquet",
    "location": "/local/path/to/data"  # Local filesystem
}
```

### 3. Testing Configuration

```python
# Test with simulation first
optimizer_sim = Optimizer(platform="local", optimization_mode="simulation")
result_sim = optimizer_sim.optimize(code_path="./job.py")

# Then validate with execution
optimizer_exec = Optimizer(platform="local", optimization_mode="execution")
result_exec = optimizer_exec.optimize(code_path="./job.py")

# Compare results
print(f"Simulation: {result_sim.estimated_time_minutes:.1f} min")
print(f"Execution:  {result_exec.estimated_time_minutes:.1f} min")
```

### 4. CI/CD Integration

```yaml
# .github/workflows/optimize.yml
name: Optimize Spark Config

on: [push]

jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install Spark Optima
        run: pip install spark-optima
      
      - name: Optimize configuration
        run: |
          spark-optima optimize \
            -c ./jobs/ci_job.py \
            -p local \
            -d 5 \
            -o json > optimized_config.json
      
      - name: Upload configuration
        uses: actions/upload-artifact@v2
        with:
          name: spark-config
          path: optimized_config.json
```

## Troubleshooting

### Issue: Java not found

```bash
# Check Java installation
java -version

# Set JAVA_HOME if needed
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin
```

### Issue: Out of memory

```python
# Reduce resource requirements
resources = ResourceSpec(
    cpu_cores=4,
    memory_gb=8,  # Lower memory
)

resource_constraints = {
    "max_memory_gb": 8
}
```

### Issue: Spark not found (execution mode)

```bash
# Install Spark or use simulation mode
export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin

# Or switch to simulation
optimizer = Optimizer(platform="local", optimization_mode="simulation")
```

## Next Steps

- Try [AWS Glue platform](./aws-glue.md) for serverless Spark
- Explore [Databricks platform](./databricks.md) for managed Spark
- Learn about [Azure Synapse](./azure-synapse.md) for Azure integration
- Read the [Python API Guide](../user-guide/api.md)