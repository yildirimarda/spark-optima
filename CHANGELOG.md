# Changelog

All notable changes to the Spark Optima project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- **BREAKING**: Remove insecure `exec()` fallback in `SparkRunner`. Docker is now **required** for secure code execution. If Docker is not available, a `RuntimeError` is raised.
- Add AST-based code safety validation as an additional layer of defense.

### Fixed
- Fix `_execute_code_docker` return type annotation to satisfy mypy (`dict[str, Any]`).
- Fix failing test `test_execute_code_with_exception` by properly mocking Docker execution.
- Fix failing test `test_generate_float_value` by setting `null_ratio=0.0`.
- Remove duplicate `imagePullPolicy` in Kubernetes deployment manifest.
- Update CLI documentation to match actual CLI options.

### Changed
- Update `README.md` to reflect that Docker is required (no insecure fallback).
- Update `docs/index.md` to remove references to exec mode fallback.
- Improve Dockerfile for production: add `tini` for signal handling, fix `JAVA_HOME`, create `/tmp/spark-events` directory.

### Added
- Coverage increased to 95.31% (1745 tests passed).
- Ruff and mypy checks now pass with 0 errors.

## [0.1.0] - 2024-05-04

### Added
- Initial release with hybrid optimization (heuristic + Bayesian).
- Support for Local, AWS Glue, Databricks, Azure Synapse platforms.
- Code analysis with smell detection and recommendations.
- Simulation and execution modes.
- Docker-based secure code execution.
- REST API with FastAPI.
- CLI with Typer.
- Kubernetes manifests and Helm charts.
- Comprehensive test suite with 1700+ tests.
