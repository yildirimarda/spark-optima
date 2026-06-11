# Spark Optima Documentation

Welcome to the Spark Optima documentation! This is your comprehensive guide to intelligent Apache Spark configuration optimization.

## What is Spark Optima?

Spark Optima is a professional tool that automatically finds the optimal Apache Spark configuration for your specific workload. Using a hybrid approach combining **heuristic rules** and **Bayesian optimization**, it eliminates the guesswork and tedious trial-and-error from Spark tuning.

## Key Features

- 🧠 **Hybrid Optimization**: Combines Spark best practices with intelligent Bayesian search
- ⚖️ **Multi-Objective Optimization**: Optimize for time *and* cost simultaneously with a Pareto frontier of trade-offs
- 🎯 **Multi-Platform Support**: Local, AWS Glue, AWS EMR, Databricks, Azure Synapse, GCP Dataproc, Kubernetes
- 📊 **Code Analysis**: Detects Spark code smells (DataFrame and `spark.sql()` SQL) and suggests improvements
- 📜 **Post-Run Analysis**: Parse Spark event logs or query a Spark History Server for GC, shuffle, spill, and skew metrics
- 🤖 **ML Surrogate Model**: Optional scikit-learn model learns from real runs to sharpen simulation estimates
- 💰 **Live Cloud Pricing**: Opt-in live on-demand rates with caching and static fallback
- 🌐 **Async REST API**: Background optimization jobs, webhooks, and pluggable job stores (memory, SQLite, Redis)
- 📋 **Workload Templates**: Curated baselines for batch ETL, streaming, ML training, and interactive analytics
- 🚀 **Dual Mode**: Fast simulation mode or real execution mode with actual measurements
- 📋 **200+ Config Parameters**: Comprehensive coverage of Spark 3.x and 4.x configurations
- 🔧 **Professional Grade**: Docker-ready, Kubernetes-ready, production-ready

## Quick Start

```bash
# Install Spark Optima
pip install spark-optima

# Run optimization
spark-optima optimize -c ./my_spark_job.py -p databricks -d 100

# Analyze a finished run from its event log
spark-optima analyze-log -l ./application_1234_eventlog

# Start from a curated workload template
spark-optima templates --show etl-batch
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
- **Calling over HTTP?** See the [REST API Reference](user-guide/rest-api.md)
- **Developer?** Read [Contributing](development/contributing.md)

## Support

- GitHub Issues: [github.com/yildirimarda/spark-optima/issues](https://github.com/yildirimarda/spark-optima/issues)
- Documentation: [yildirimarda.github.io/spark-optima](https://yildirimarda.github.io/spark-optima)

## License

Spark Optima is licensed under the Apache License 2.0. See [LICENSE](https://github.com/yildirimarda/spark-optima/blob/main/LICENSE) for details.