# Spark Optima Documentation

Welcome to the Spark Optima documentation! This is your comprehensive guide to intelligent Apache Spark configuration optimization.

## What is Spark Optima?

Spark Optima is a professional tool that automatically finds the optimal Apache Spark configuration for your specific workload. Using a hybrid approach combining **heuristic rules** and **Bayesian optimization**, it eliminates the guesswork and tedious trial-and-error from Spark tuning.

## Key Features

- 🧠 **Hybrid Optimization**: Combines Spark best practices with intelligent Bayesian search
- 🎯 **Multi-Platform Support**: Local, AWS Glue, Databricks, Azure Synapse
- 📊 **Code Analysis**: Detects Spark code smells and suggests improvements
- 🚀 **Dual Mode**: Fast simulation mode or real execution mode with actual measurements
- 📋 **200+ Config Parameters**: Comprehensive coverage of Spark 3.x and 4.x configurations
- 🔧 **Professional Grade**: Docker-ready, Kubernetes-ready, production-ready

## Quick Start

```bash
# Install Spark Optima
pip install spark-optima

# Run optimization
spark-optima optimize -c ./my_spark_job.py -p databricks -d 100
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Spark Optima Engine                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Input     │  │   Code      │  │  Hybrid Optimizer   │  │
│  │ Collection  │→ │  Analysis   │→ │ (Heuristic + Bayes) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                           ↓                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Simulation / Execution                  │    │
│  │         (Performance Measurement & Feedback)         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                           ↓                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          Optimal Configuration Output                │    │
│  │  (JSON, YAML, Platform-specific formats, Code fixes) │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Where to Go Next

- **New User?** Start with [Getting Started](user-guide/getting-started.md)
- **Want to Install?** Check [Installation](user-guide/installation.md)
- **Using a Specific Platform?** See [Platform Guides](platforms/local.md)
- **Developer?** Read [Contributing](development/contributing.md)

## Support

- GitHub Issues: [github.com/spark-optima/spark-optima/issues](https://github.com/yourusername/spark-optima/issues)
- Documentation: [spark-optima.readthedocs.io](https://your-project.readthedocs.io)

## License

Spark Optima is licensed under the Apache License 2.0. See [LICENSE](https://github.com/yourusername/spark-optima/blob/main/LICENSE) for details.