# 🔥 Spark Optima

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](.pre-commit-config.yaml)

**Intelligent Apache Spark Configuration Optimization Tool**

Spark Optima is a professional, production-ready tool that automatically finds the optimal Apache Spark configuration for your specific workload. Using a hybrid approach combining **heuristic rules** and **Bayesian optimization**, it eliminates the guesswork and tedious trial-and-error from Spark tuning.

---

## ✨ Features

- 🧠 **Hybrid Optimization**: Combines Spark best practices with intelligent Bayesian search
- 🎯 **Multi-Platform Support**: Local, AWS Glue, Databricks, Azure Synapse
- 📊 **Code Analysis**: Detects Spark code smells and suggests improvements
- 🚀 **Dual Mode**: Fast simulation mode or real execution mode with actual measurements
- 🔒 **Secure Execution**: Docker-based isolation for untrusted code execution
- 📋 **200+ Config Parameters**: Comprehensive coverage of Spark 3.x and 4.x configurations
- 🔧 **Professional Grade**: Docker-ready, Kubernetes-ready, production-ready

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Poetry (for development)
- Docker (required for execution mode - secure code isolation)
- Java 17 (for local Spark execution)

### Installation

#### Option 1: Using Poetry (Recommended for development)

```bash
# Clone the repository
git clone https://github.com/yourusername/spark-optima.git
cd spark-optima

# Install with Poetry (uses .venv in project directory)
poetry install --all-extras

# Verify installation
poetry run spark-optima --version
```

#### Option 2: Using pip (Simple installation)

```bash
pip install spark-optima
```

#### Install PySpark (Required for execution mode)

```bash
# Install PySpark in your environment
poetry add pyspark  # If using Poetry
# OR
pip install pyspark  # If using pip
```

---

## 🔧 Local Usage Guide

### 1. Basic Optimization (Simulation Mode)

Simulation mode works out-of-the-box without Docker or Spark installation. It uses intelligent heuristics + Bayesian optimization to predict optimal configurations.

```bash
# Basic usage with your Spark code
poetry run spark-optima optimize \
  --code-path /tmp/simple_spark_job.py \
  --platform local \
  --data-size 10 \
  --max-memory 16

# With JSON output (save for later export)
poetry run spark-optima optimize \
  --code-path /tmp/simple_spark_job.py \
  --platform local \
  --data-size 5 \
  --output json > result.json
```

**Example Output:**
```
🔥 Spark Optima
Intelligent Spark Configuration Optimization

              Input Parameters
┌───────────────┬──────────────────────────┐
│ Code Path     │ /tmp/simple_spark_job.py │
│ Platform      │ local                    │
│ Spark Version │ 3.5.0                    │
│ Data Size     │ 10.0 GB                  │
│ Data Format   │ parquet                  │
│ Max Memory    │ 16.0 GB                  │
│ Mode          │ simulation               │
└───────────────┴──────────────────────────┘

⏱  Estimated Execution Time: 13.3 minutes
🎯 Confidence Score: 85%
💰 Estimated Cost: $0.00

🔧 TOP CONFIGURATION PARAMETERS
──────────────────────────────────────────────
  spark.executor.memory                         = 4g
  spark.executor.cores                         = 4
  spark.driver.memory                          = 1g
  spark.sql.adaptive.enabled                    = True
  spark.sql.shuffle.partitions                = 200
  spark.dynamicAllocation.enabled              = True
```

### 2. Execution Mode (Real Spark Measurements)

Execution mode runs your actual Spark code with different configurations in isolated Docker containers and measures real performance.

**Prerequisites:**
- Docker must be installed and running
- Docker image must be built

```bash
# Build Docker image (first time only)
docker build -f docker/Dockerfile -t spark-optima:latest .

# Run optimization with execution mode
poetry run spark-optima optimize \
  --code-path /tmp/simple_spark_job.py \
  --platform local \
  --data-size 1 \
  --max-memory 4 \
  --mode execution

# Quick test with minimal trials
poetry run python /tmp/test_execution.py
```

**Note:** Execution mode requires Docker for security isolation. Your Spark code runs in an isolated container.

### 3. Interactive Wizard

Let Spark Optima guide you through the optimization process step by step:

```bash
poetry run spark-optima wizard
```

### 4. Code Analysis

Analyze your Spark code for optimization opportunities without running optimization:

```bash
poetry run spark-optima analyze --code-path /tmp/simple_spark_job.py

# JSON output
poetry run spark-optima analyze --code-path /tmp/simple_spark_job.py --output json
```

**Example Output:**
```
🔍 Spark Code Analysis

┌───────────┬───────────┐
│ Operations Count │ 2         │
│ Code Smells      │ 2         │
│ Recommendations  │ 2         │
└───────────┴───────────┘

Code Smells
┌────────┬─────────────────┬─────────┐
│ Line   │ Type            │ Severity │
├────────┼─────────────────┼─────────┤
│ 12     │ data_skew       │ medium   │
│        │ _potential      │          │
└────────┴─────────────────┴─────────┘
```

### 5. Platform Management

List all supported platforms:

```bash
poetry run spark-optima platforms list
```

### 6. Export Configuration

Export your optimization results to various platform-specific formats:

```bash
# Export to Databricks cluster JSON
poetry run spark-optima export \
  --result-file result.json \
  --format databricks-json \
  --output cluster.json

# Export to AWS Glue
poetry run spark-optima export \
  --result-file result.json \
  --format aws-glue

# Export as environment variables
poetry run spark-optima export \
  --result-file result.json \
  --format env \
  --output spark_env.sh

# Export as spark-submit command
poetry run spark-optima export \
  --result-file result.json \
  --format spark-submit

# Export to Azure Synapse Spark pool config
poetry run spark-optima export \
  --result-file result.json \
  --format azure-synapse
```

---

## 🐍 Python API

```python
from pathlib import Path
from spark_optima import Optimizer
from spark_optima.platforms.models import ResourceSpec

# Initialize optimizer
optimizer = Optimizer(
    platform="databricks",
    spark_version="3.5.0",
    optimization_mode="simulation",  # or "execution"
)

# Define resources
resources = ResourceSpec(
    cpu_cores=16,
    memory_gb=64,
)

# Define data profile
data_profile = {
    "size_gb": 100,
    "format": "parquet",
    "compression": "snappy",
}

# Run optimization
result = optimizer.optimize(
    code_path="./my_spark_job.py",
    resources=resources,
    data_profile=data_profile,
    use_bayesian=True,
    bayesian_trials=50,
    objectives=["minimize_time"],
)

# Access results
print(f"Optimal config: {result.configuration}")
print(f"Estimated time: {result.estimated_time_minutes:.1f} min")
print(f"Confidence: {result.confidence_score:.0%}")

# Get code suggestions
for suggestion in result.code_suggestions:
    print(f"Line {suggestion.line_number}: {suggestion.issue_type}")
    print(f"  Suggestion: {suggestion.suggestion}")
```

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────┐
│                    Spark Optima Engine                       │
├───────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Input     │  │   Code      │  │  Hybrid Optimizer   │  │
│  │ Collection  │→ │  Analysis   │→ │ (Heuristic + Bayes) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                           ↓                  │
│  ┌─────────────────────────────────────────────┐    │
│  │              Simulation / Execution                  │    │
│  │         (Performance Measurement & Feedback)         │    │
│  └─────────────────────────────────────────────┘    │
│                                           ↓                  │
│  ┌─────────────────────────────────────────────┐    │
│  │          Optimal Configuration Output                │    │
│  │  (JSON, YAML, Platform-specific formats, Code fixes) │    │
│  └─────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────┘
```

---

## 📋 Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| **Local/Standalone** | ✅ Ready | Full support with Docker |
| **AWS Glue** | ✅ Ready | Including Glue 3.0/4.0 |
| **Databricks** | ✅ Ready | DBR 10.x - 14.x |
| **Azure Synapse** | ✅ Ready | Node sizes: Small to XXLarge |

---

## 📦 Project Structure

```
spark-optima/
├── src/spark_optima/          # Main source code
│   ├── core/                  # Core optimization engine
│   │   ├── optimizer.py      # Main optimizer class
│   │   ├── config_engine/   # Spark config knowledge base
│   │   ├── bayesian/        # Bayesian optimization (Optuna)
│   │   ├── simulation/      # Performance simulator
│   │   └── execution/      # Real Spark execution
│   ├── platforms/             # Platform adapters
│   ├── analysis/              # Code analysis
│   ├── cli/                    # Command-line interface
│   ├── api/                    # REST API (FastAPI)
│   └── data/                  # Sample data & profiling
├── tests/                     # Test suite
├── docker/                    # Docker configurations
├── docs/                      # Documentation
├── kubernetes/                # K8s manifests & Helm charts
└── examples/                  # Usage examples
```

---

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/spark-optima.git
cd spark-optima

# Install dependencies with Poetry
poetry install --all-extras

# Activate virtual environment (optional, poetry run prefix works too)
poetry shell

# Install pre-commit hooks
poetry run pre-commit install
```

### Running Tests

```bash
# Run all tests (takes a while, be patient)
poetry run python -m pytest

# Run with coverage
poetry run python -m pytest --cov=src/spark_optima --cov-report=html

# Run specific test categories
poetry run python -m pytest -m unit -v          # Unit tests only
poetry run python -m pytest -m integration -v   # Integration tests only

# Run a specific test file
poetry run python -m pytest tests/unit/test_optimizer.py -v
```

**Note:** Do not run the full test suite more than once concurrently. Test execution takes time. For specific tests, you can run them in parallel.

### Code Quality

```bash
# Linting
poetry run ruff check .
poetry run ruff check . --fix  # Auto-fix issues

# Formatting
poetry run ruff format .

# Type checking
poetry run mypy src/spark_optima

# Security scan
poetry run bandit -r src/spark_optima
```

---

## 📖 Documentation

Full documentation is available at [https://your-project.readthedocs.io](https://your-project.readthedocs.io)

- [User Guide](docs/user-guide.md)
- [API Reference](docs/api-reference.md)
- [Configuration Guide](docs/configuration.md)
- [Platform Guides](docs/platforms/)
- [Contributing](CONTRIBUTING.md)

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Apache Spark community for the excellent documentation
- Optuna team for the Bayesian optimization framework
- All contributors who helped make this project possible

---

**Made with ❤️ by the Spark Optima Contributors**
