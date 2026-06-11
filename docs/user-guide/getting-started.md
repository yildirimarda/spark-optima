# Getting Started with Spark Optima

Welcome to Spark Optima! This guide will help you understand the core concepts and get your first optimization running in minutes.

## What is Spark Optima?

Spark Optima is an intelligent Apache Spark configuration optimization tool that eliminates the guesswork from Spark tuning. Instead of manually trying different configurations through trial-and-error, Spark Optima uses a **hybrid optimization approach** combining:

1. **Heuristic Rules** - Spark best practices based on your workload characteristics
2. **Bayesian Optimization** - Intelligent search to fine-tune configurations

## Key Concepts

### Hybrid Optimization

Spark Optima's optimization process works in two phases:

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Heuristic Configuration                           │
│  ├── Analyzes your code and data characteristics            │
│  ├── Applies Spark best practices                           │
│  └── Generates baseline configuration                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Bayesian Optimization                             │
│  ├── Defines search space around heuristic config           │
│  ├── Runs intelligent trials                                │
│  └── Converges to optimal configuration                     │
└─────────────────────────────────────────────────────────────┘
```

### Optimization Modes

Spark Optima supports two execution modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Simulation** | Uses performance models to estimate execution | Fast iteration, initial exploration |
| **Execution** | Runs actual Spark jobs to measure performance | Production workloads, precise tuning |

### Platforms Supported

Spark Optima supports multiple deployment platforms:

- **Local** - Run on your local machine with standalone Spark
- **AWS Glue** - Serverless Spark on AWS
- **AWS EMR** - Managed Spark on EC2 (m5/r5/c5 instance families)
- **Databricks** - Managed Spark platform
- **Azure Synapse** - Azure's analytics service
- **GCP Dataproc** - Managed Spark on Google Cloud (n2-standard/n2-highmem)
- **Kubernetes** - Self-hosted Spark-on-K8s, including Spark Operator CRD export

## Quick Start Tutorial

### Step 1: Install Spark Optima

```bash
pip install spark-optima
```

### Step 2: Create a Sample Spark Job

Create a file named `sample_job.py`:

```python
# sample_job.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, count

# Initialize Spark session
spark = SparkSession.builder \
    .appName("SalesAnalysis") \
    .getOrCreate()

# Read sales data
df = spark.read.parquet("s3://my-bucket/sales_data/")

# Perform aggregations
daily_sales = df.groupBy("date", "region") \
    .agg(
        spark_sum("amount").alias("total_sales"),
        count("*").alias("transaction_count")
    )

# Filter high-value regions
high_value_regions = daily_sales.filter(col("total_sales") > 10000)

# Write results
high_value_regions.write \
    .mode("overwrite") \
    .parquet("s3://my-bucket/output/high_value_regions/")

spark.stop()
```

### Step 3: Run Your First Optimization

Using the CLI:

```bash
spark-optima optimize \
  --code-path ./sample_job.py \
  --platform aws_glue \
  --data-size 500 \
  --data-format parquet \
  --max-memory 64
```

Or using Python:

```python
from spark_optima import Optimizer

# Initialize optimizer
optimizer = Optimizer(
    platform="aws_glue",
    spark_version="3.5.0"
)

# Run optimization
result = optimizer.optimize(
    code_path="./sample_job.py",
    data_profile={"size_gb": 500, "format": "parquet"},
    resource_constraints={"max_memory_gb": 64}
)

# Print results
print(f"Estimated execution time: {result.estimated_time_minutes:.1f} minutes")
print(f"Confidence score: {result.confidence_score:.0%}")

# View optimal configuration
for key, value in result.configuration.items():
    print(f"  {key} = {value}")
```

### Step 4: Review Results

Spark Optima will output:

1. **Optimal Configuration** - Spark parameters optimized for your workload
2. **Estimated Performance** - Predicted execution time and cost
3. **Code Suggestions** - Improvements to your Spark code
4. **Confidence Score** - How confident the optimizer is in the results

Example output:

```
┌─────────────────────────────────────────────────────────────┐
│ Optimization Results                                        │
├─────────────────────────────────────────────────────────────┤
│ Estimated Time:        12.5 minutes                         │
│ Confidence Score:      85%                                  │
│ Platform:              aws_glue                             │
│ Spark Version:         3.5.0                                │
└─────────────────────────────────────────────────────────────┘

Top Configuration Parameters:
┌──────────────────────────────────────┬──────────┐
│ Parameter                            │ Value    │
├──────────────────────────────────────┼──────────┤
│ spark.executor.memory                │ 8g       │
│ spark.executor.cores                 │ 4        │
│ spark.sql.adaptive.enabled           │ true     │
│ spark.sql.shuffle.partitions         │ 400      │
│ spark.dynamicAllocation.enabled      │ true     │
│ spark.dynamicAllocation.maxExecutors │ 50       │
└──────────────────────────────────────┴──────────┘

Code Suggestions:
  ⚠️  Line 15: Consider adding broadcast hint for small DataFrame
      Suggestion: from pyspark.sql.functions import broadcast
                  df_small = broadcast(spark.read.parquet("small_lookup"))
```

## Understanding the Optimization Workflow

### Input Collection

Spark Optima needs the following information:

1. **Spark Code** - Your PySpark application (optional but recommended)
2. **Platform** - Where you'll run the job
3. **Data Profile** - Size, format, and characteristics of your data
4. **Resource Constraints** - Memory, CPU, and cost limits

### Code Analysis

When you provide code, Spark Optima analyzes it to:

- Detect operations that benefit from specific optimizations
- Identify potential code smells (unnecessary shuffles, missing hints)
- Estimate computational complexity

### Resource-Aware Optimization

The optimizer considers your resource constraints:

```python
# Example: Resource constraints
resource_constraints = {
    "max_memory_gb": 128,        # Maximum memory available
    "max_cost_usd": 50.0,        # Maximum cost per run
    "max_executors": 100,        # Maximum parallel executors
}
```

### Exporting Configurations

After optimization, export to your platform:

```bash
# Export for Databricks
spark-optima export -r result.json -f databricks-json -o cluster_config.json

# Export for AWS Glue
spark-optima export -r result.json -f aws-glue -o glue_job.py

# Export as environment variables
spark-optima export -r result.json -f env -o spark_config.env
```

## Next Steps

Now that you understand the basics:

1. **Learn more about installation** → [Installation Guide](installation.md)
2. **Explore platform-specific guides** → [Platform Guides](../platforms/)
3. **See advanced usage patterns** → [Configuration Guide](configuration.md)
4. **Try code examples** → [Examples](../../examples/)

## Common Questions

### Do I need to run the optimization for every job?

No. Once you optimize a job, you can reuse the configuration for similar workloads. However, re-optimizing is recommended when:
- Data size changes significantly (>50%)
- Code changes substantially
- Moving to a different platform

### How accurate are the estimates?

Simulation mode provides good estimates (typically within ±30%) based on performance models. For precise measurements, use Execution mode which runs actual Spark jobs.

### Can I customize the optimization objectives?

Yes! You can optimize for:
- **`minimize_time`** - fastest run (default)
- **`minimize_cost`** - cheapest run
- **`minimize_memory`** - smallest memory footprint
- **`maximize_success`** - lowest failure/OOM risk

Passing more than one objective switches to multi-objective mode and produces
a Pareto frontier of trade-offs:

```bash
spark-optima optimize -c job.py \
  --objective minimize_time --objective minimize_cost \
  --output json > result.json
spark-optima pareto -r result.json
```

See the [Configuration Guide](configuration.md) for details.

### What if the optimization fails?

Spark Optima gracefully degrades:
- If Bayesian optimization fails, it returns the heuristic configuration
- If code analysis fails, optimization continues without code suggestions
- You can always inspect intermediate results

---

**Ready to dive deeper?** Check out the [Installation Guide](installation.md) to set up Spark Optima for your environment.