# CLI Usage Guide

This guide covers all commands and options available in the Spark Optima CLI.

## Overview

The Spark Optima CLI provides a powerful command-line interface for optimizing Spark configurations. It supports interactive wizards, direct optimization, and configuration export.

## Global Options

These options are available for all commands:

```bash
spark-optima [OPTIONS] COMMAND [ARGS]

Options:
  --version, -v        Show version and exit
  --help               Show help message and exit
```

## Commands

### `optimize` - Run Configuration Optimization

The main command for optimizing Spark configurations.

#### Basic Usage

```bash
spark-optima optimize -c ./my_job.py -p databricks
```

#### All Options

```bash
spark-optima optimize \
  --code-file, -c ./job.py \               # Path to Spark code file (required)
  --platform, -p databricks \             # Target platform (local, databricks, aws_glue, azure_synapse)
  --spark-version, -s 3.5.0 \           # Spark version
  --data-size, -d 500 \                   # Data size in GB
  --data-format, -f parquet \             # Data format (parquet, delta, json, csv, orc)
  --max-memory, -m 128 \                  # Max memory in GB
  --output, -o table \                    # Output format (table, json, yaml)
  --mode simulation                         # Optimization mode (simulation or execution)
```

#### Examples

**Basic optimization:**
```bash
spark-optima optimize -c ./etl_job.py -p aws_glue -d 100
```

**With all options:**
```bash
spark-optima optimize \
  -c ./analytics_job.py \
  -p databricks \
  -s 3.5.0 \
  -d 1000 \
  -f delta \
  -m 256 \
  -o json
```

**Save results to file:**
```bash
spark-optima optimize -c ./job.py -p local -d 50 -o json > result.json
```

#### Output Formats

**Table format (default):**
```
┌─────────────────────────────────────────────────────────────┐
│ Optimization Results                                        │
├─────────────────────────────────────────────────────────────┤
│ Estimated Time:        15.3 minutes                         │
│ Confidence Score:      87%                                  │
└─────────────────────────────────────────────────────────────┘
```

**JSON format:**
```json
{
  "configuration": {
    "spark.executor.memory": "8g",
    "spark.executor.cores": "4"
  },
  "estimated_time_minutes": 15.3,
  "confidence_score": 0.87
}
```

**YAML format:**
```yaml
configuration:
  spark.executor.memory: 8g
  spark.executor.cores: "4"
estimated_time_minutes: 15.3
confidence_score: 0.87
```

### `wizard` - Interactive Configuration

Launch an interactive wizard that guides you through the optimization process.

#### Usage

```bash
spark-optima wizard
```

#### Wizard Steps

1. **Select Platform** - Choose from Local, AWS Glue, Databricks, or Azure Synapse
2. **Specify Spark Version** - Select Spark version (3.5.0, 4.0.0, etc.)
3. **Provide Code Path** - Enter path to your Spark application
4. **Data Information** - Specify data size, format, and characteristics
5. **Resource Constraints** - Set memory, CPU, and cost limits
6. **Optimization Settings** - Choose Bayesian trials and timeout
7. **Review and Run** - Confirm settings and start optimization

#### Example Session

```
$ spark-optima wizard

🔥 Spark Optima - Interactive Configuration Wizard
═══════════════════════════════════════════════════

Step 1/7: Select Platform
Available platforms:
  [1] local
  [2] aws_glue
  [3] databricks
  [4] azure_synapse
Select platform (1-4): 3

Step 2/7: Spark Version
Available versions: 3.4.0, 3.5.0, 4.0.0
Enter Spark version [3.5.0]: 3.5.0

Step 3/7: Code Path
Enter path to Spark code file: ./my_job.py
✓ Code file found

Step 4/7: Data Information
Data size in GB [10]: 500
Data format (parquet/delta/json/csv) [parquet]: parquet

Step 5/7: Resource Constraints
Maximum memory in GB [64]: 128
Maximum cost in USD [unlimited]: 100

Step 6/7: Optimization Settings
Use Bayesian optimization? [Y/n]: y
Number of trials [50]: 50
Timeout in minutes [30]: 30

Step 7/7: Review Configuration
Platform: databricks
Spark Version: 3.5.0
Code Path: ./my_job.py
Data Size: 500 GB
...

Start optimization? [Y/n]: y

[bold green]Running optimization...[/bold green]
```

### `export` - Export Configuration

Convert optimization results to platform-specific formats.

#### Usage

```bash
spark-optima export [OPTIONS]
```

#### Options

```bash
spark-optima export \
  --result-file, -r result.json \         # Path to result JSON file
  --format, -f json \                     # Export format
  --output, -o output.file                # Output file path
```

#### Available Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `json` | Standard JSON format | General purpose |
| `yaml` | YAML format | Configuration files |
| `spark-submit` | spark-submit command line | Local execution |
| `databricks-json` | Databricks cluster JSON | Databricks UI/API |
| `databricks-cli` | Databricks CLI command | Automation scripts |
| `aws-glue` | AWS Glue job configuration | Glue console |
| `aws-cli` | AWS CLI command | AWS automation |
| `azure-synapse` | Azure Synapse configuration | Synapse Studio |
| `env` | Environment variables | Shell scripts |
| `properties` | Java properties file | spark-defaults.conf |

#### Examples

**Export to Databricks JSON:**
```bash
spark-optima export -r result.json -f databricks-json -o cluster.json
```

**Export for AWS Glue:**
```bash
spark-optima export -r result.json -f aws-glue -o glue_config.py
```

**Export as environment variables:**
```bash
spark-optima export -r result.json -f env -o spark_env.sh
source spark_env.sh
spark-submit my_job.py
```

**Export as properties file:**
```bash
spark-optima export -r result.json -f properties -o spark-defaults.conf
```

**View format help:**
```bash
spark-optima export -r result.json -f help
```

## Environment Variables

Spark Optima respects these environment variables:

```bash
# Platform credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

export DATABRICKS_HOST=https://...
export DATABRICKS_TOKEN=...

export AZURE_SUBSCRIPTION_ID=...
export AZURE_RESOURCE_GROUP=...

# Spark Optima settings
export SPARK_OPTIMA_LOG_LEVEL=INFO
export SPARK_OPTIMA_CACHE_DIR=~/.spark-optima

# Optimization defaults
export SPARK_OPTIMA_DEFAULT_TRIALS=50
export SPARK_OPTIMA_DEFAULT_TIMEOUT=30
```

## Common CLI Patterns

### Development Workflow

```bash
# 1. Quick simulation for development
spark-optima optimize -c ./dev_job.py -p local -d 10 -o json > dev_result.json

# 2. Export and test locally
spark-optima export -r dev_result.json -f spark-submit -o run.sh
chmod +x run.sh
./run.sh

# 3. Production optimization
spark-optima optimize -c ./prod_job.py -p databricks -d 1000 -m 512 -o json > prod_result.json

# 4. Export for production deployment
spark-optima export -r prod_result.json -f databricks-json -o prod_cluster.json
```

### CI/CD Integration

```bash
#!/bin/bash
# optimize.sh - Run in CI/CD pipeline

set -e

echo "Optimizing Spark configuration..."

# Run optimization
spark-optima optimize \
  -c "${SPARK_CODE_PATH}" \
  -p "${PLATFORM}" \
  -d "${DATA_SIZE_GB}" \
  -o json > optimization_result.json

# Check confidence score
CONFIDENCE=$(cat optimization_result.json | jq -r '.confidence_score')
if (( $(echo "$CONFIDENCE < 0.7" | bc -l) )); then
  echo "Warning: Low confidence score ($CONFIDENCE)"
fi

# Export for deployment
spark-optima export \
  -r optimization_result.json \
  -f "${EXPORT_FORMAT}" \
  -o "${OUTPUT_PATH}"

echo "Optimization complete. Configuration exported to ${OUTPUT_PATH}"
```

### Batch Optimization

```bash
#!/bin/bash
# batch_optimize.sh - Optimize multiple jobs

JOBS_DIR="./jobs"
RESULTS_DIR="./optimization_results"

mkdir -p "$RESULTS_DIR"

for job_file in "$JOBS_DIR"/*.py; do
  job_name=$(basename "$job_file" .py)
  echo "Optimizing: $job_name"
  
  spark-optima optimize \
    -c "$job_file" \
    -p databricks \
    -d 100 \
    -o json > "$RESULTS_DIR/${job_name}_result.json"
  
  # Export configurations
  spark-optima export \
    -r "$RESULTS_DIR/${job_name}_result.json" \
    -f databricks-json \
    -o "$RESULTS_DIR/${job_name}_cluster.json"
done

echo "Batch optimization complete. Results in $RESULTS_DIR"
```

## Tips and Tricks

### 1. Use JSON Output for Automation

```bash
# Get specific value from result
MEMORY=$(spark-optima optimize -c job.py -p local -o json | jq -r '.configuration."spark.executor.memory"')
echo "Recommended memory: $MEMORY"
```

### 2. Compare Multiple Platforms

```bash
# Optimize for different platforms
for platform in local aws_glue databricks; do
  spark-optima optimize -c job.py -p "$platform" -d 100 -o json > "${platform}_result.json"
done

# Compare results
echo "Platform | Time | Cost"
for platform in local aws_glue databricks; do
  TIME=$(jq -r '.estimated_time_minutes' "${platform}_result.json")
  COST=$(jq -r '.metadata.estimated_cost' "${platform}_result.json")
  echo "$platform | $TIME | $COST"
done
```

### 3. Validate Configuration Before Deployment

```bash
# Export and validate
spark-optima export -r result.json -f spark-submit -o validate.sh

# Dry run (don't actually execute)
./validate.sh --dry-run
```

### 4. Combine with Other Tools

```bash
# Use with jq for processing
spark-optima optimize -c job.py -p local -o json | \
  jq '{memory: .configuration."spark.executor.memory", time: .estimated_time_minutes}'

# Use with watch for monitoring
watch -n 30 'spark-optima optimize -c job.py -p local -d 10 2>&1 | tail -20'
```

### `history` - Past Optimizations

Every successful `optimize` run is saved to a local SQLite history (`~/.spark_optima/history.db`, override with `SPARK_OPTIMA_HISTORY_DB`).

```bash
spark-optima history                 # list recent runs
spark-optima history --platform aws_emr --limit 10
spark-optima history --show 3        # full detail of one entry
spark-optima history --clear --yes   # wipe the history
```

### `compare` - Diff Two Results

Compare two optimization result JSON files: differing parameters, parameters present in only one result, and metric deltas.

```bash
spark-optima compare -a result_local.json -b result_emr.json
spark-optima compare -a a.json -b b.json --output json   # machine-readable diff
```

### `explain` - Parameter Rationale

Explain why each parameter in a result was chosen, sourced from the heuristic rule descriptions and the Spark parameter database.

```bash
spark-optima explain -r result.json
```

### `analyze-log` - Spark Event Log Analysis

Parse a Spark event log (plain or `.gz`) and report real run metrics — GC time, shuffle volumes, spill, task skew — plus tuning hints.

```bash
spark-optima analyze-log -l /path/to/eventlog          # summary tables
spark-optima analyze-log -l eventlog.gz -o json        # pipe-safe JSON
spark-optima analyze-log -l eventlog --top-stages 20
```

You can also feed a log directly into optimization — data size, skew, and shuffle pressure are inferred from the real run:

```bash
spark-optima optimize -c job.py -p databricks --event-log /path/to/eventlog
```

## Troubleshooting CLI Issues

### Issue: Command not found

```bash
# Verify installation
which spark-optima

# If not found, check pip installation
pip list | grep spark-optima

# Reinstall if needed
pip install --force-reinstall spark-optima
```

### Issue: Permission denied

```bash
# Check file permissions
ls -la ./my_job.py

# Fix permissions
chmod +r ./my_job.py
```

### Issue: Output is garbled

```bash
# Force plain text output
spark-optima optimize -c job.py --no-color

# Or redirect to file
spark-optima optimize -c job.py -o json > result.json
```

### Issue: Optimization takes too long

```bash
# Reduce Bayesian trials
spark-optima optimize -c job.py -p local --bayesian-trials 20

# Or disable Bayesian optimization
spark-optima optimize -c job.py -p local --no-bayesian
```

## Getting Help

### Command Help

```bash
# General help
spark-optima --help

# Command-specific help
spark-optima optimize --help
spark-optima wizard --help
spark-optima export --help
spark-optima history --help
spark-optima compare --help
spark-optima explain --help
spark-optima analyze-log --help
```

### Examples

```bash
# Show examples
spark-optima optimize --examples
```

---

**Next Steps:**
- Learn about the [Python API](api.md)
- Read about [Configuration Options](configuration.md)
- Explore [Platform-Specific Guides](../platforms/)