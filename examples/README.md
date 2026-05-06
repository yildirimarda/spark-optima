# Spark Optima Examples

This directory contains example scripts demonstrating various use cases of Spark Optima.

## Available Examples

### Basic Usage
- `basic/simple_optimization.py` - Basic optimization example with CLI and programmatic usage

### Platform-Specific Examples
- `platforms/aws_glue_example.py` - Optimizing for AWS Glue
- `platforms/databricks_example.py` - Optimizing for Databricks
- `platforms/azure_synapse_example.py` - Optimizing for Azure Synapse

### Advanced Usage
- `advanced/bayesian_optimization.py` - Advanced Bayesian optimization with custom parameters
- `advanced/simulation_mode.py` - Using simulation mode for fast estimation
- `advanced/code_analysis.py` - Analyzing and improving Spark code

### Data Generation
- `data/generate_sample_data.py` - Generate sample data for testing

## Running Examples

```bash
# Basic example
python examples/basic/simple_optimization.py --platform local --data-size 100

# Platform-specific example
python examples/platforms/aws_glue_example.py --job-file my_job.py

# Advanced example
python examples/advanced/bayesian_optimization.py --trials 50
```
