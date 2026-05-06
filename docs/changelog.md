# Changelog

All notable changes to Spark Optima will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Spark Optima
- Hybrid optimization (Heuristic + Bayesian)
- Multi-platform support (Local, AWS Glue, Databricks, Azure Synapse)
- Spark 3.x and 4.x version support
- CLI with interactive wizard
- Python API
- REST API (FastAPI)
- Code analysis module
- Docker support
- Kubernetes deployment manifests
- Helm charts
- Comprehensive documentation

### Features

#### Core Optimization
- Heuristic rule engine with 50+ optimization rules
- Bayesian optimization using Optuna
- Simulation and Execution modes
- Multi-objective optimization (time, cost)
- Configuration validation

#### Platforms
- **Local**: Standalone Spark support
- **AWS Glue**: Worker type selection, job bookmarks
- **Databricks**: DBR support, Unity Catalog, Photon
- **Azure Synapse**: Spark pools, ADLS Gen2 integration

#### Interfaces
- Interactive CLI with Typer
- Python API for programmatic access
- REST API with OpenAPI documentation
- Export to multiple formats (JSON, YAML, env, platform-specific)

#### Code Analysis
- AST-based code parsing
- Smell detection (broadcast hints, shuffles, caching)
- Automated suggestions
- Severity classification

### Documentation
- User guides (Getting Started, Installation, Configuration)
- Platform-specific guides
- API reference documentation
- Architecture documentation
- Testing guide
- Examples directory with runnable code

## [0.1.0] - 2024-02-11

### Added
- Initial public release
- Basic optimization functionality
- Support for Spark 3.5.0
- CLI with optimize and wizard commands
- Python API
- Local platform support
- Documentation framework (MkDocs)

## Migration Guide

### Upcoming Changes

No breaking changes planned for v0.2.0.

### Deprecated Features

None currently deprecated.

## Contributing to Changelog

When making changes:

1. Add entry under `[Unreleased]` section
2. Use categories: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`
3. Include issue/PR reference when applicable
4. Keep descriptions concise but informative

Example:
```markdown
### Added
- New feature X that does Y (#123)

### Fixed
- Bug where Z would cause error (#456)