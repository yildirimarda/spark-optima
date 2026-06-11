# Spark Optima Examples

This directory contains example scripts demonstrating various use cases of Spark Optima.

## Available Examples

### Basic Usage
- `basic/simple_optimization.py` - Basic optimization example with CLI and programmatic usage
- `basic/workload_templates.py` - List curated workload templates (etl-batch, streaming, ml-training, interactive) and layer your own settings on top with `apply_to()`

### Platform-Specific Examples
- `platforms/aws_glue_example.py` - Optimizing for AWS Glue
- `platforms/aws_emr_example.py` - Sizing, pricing, and exporting an AWS EMR cluster (boto3 `run_job_flow` definition)
- `platforms/databricks_example.py` - Optimizing for Databricks
- `platforms/azure_synapse_example.py` - Optimizing for Azure Synapse
- `platforms/gcp_dataproc_example.py` - Sizing and pricing a GCP Dataproc cluster, on-demand vs. preemptible, `clusters.create` export
- `platforms/spark_k8s_example.py` - Sizing a Spark-on-Kubernetes deployment and exporting a Spark Operator `SparkApplication` CRD

### Advanced Usage
- `advanced/bayesian_optimization.py` - Advanced Bayesian optimization with custom parameters
- `advanced/simulation_mode.py` - Using simulation mode for fast estimation
- `advanced/code_analysis.py` - Analyzing and improving Spark code
- `advanced/event_log_analysis.py` - Parsing a Spark event log into a run summary and tuning hints (builds its own synthetic log)
- `advanced/multi_objective_pareto.py` - Multi-objective optimization (`minimize_time` + `minimize_cost`) and reading the Pareto frontier from `result.metadata`

### Data Generation
- `data/generate_sample_data.py` - Generate sample data for testing

## Running Examples

All examples run with the project virtualenv; none of them require cloud
credentials or a running Spark cluster.

```bash
# Basic example
uv run python examples/basic/simple_optimization.py --platform local --data-size 100

# Workload templates (runs standalone)
uv run python examples/basic/workload_templates.py

# Platform sizing examples (local computations only)
uv run python examples/platforms/aws_emr_example.py
uv run python examples/platforms/gcp_dataproc_example.py
uv run python examples/platforms/spark_k8s_example.py

# Event log analysis (builds a synthetic log, then parses it)
uv run python examples/advanced/event_log_analysis.py

# Multi-objective optimization with a Pareto frontier
uv run python examples/advanced/multi_objective_pareto.py
```
