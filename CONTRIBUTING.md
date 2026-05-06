# Contributing to Spark Optima

First off, thank you for considering contributing to Spark Optima! It's people like you that make Spark Optima such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our commitment to professionalism and respect. By participating, you are expected to uphold this standard.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to see if the problem has already been reported. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what behavior you expected**
- **Include code samples and Spark configurations**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the enhancement**
- **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repository
2. Create a new branch from `develop` (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`make test`)
5. Run linting (`make check-all`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.10+
- Poetry 1.7+
- Docker (optional, for containerized development)

### Setup

```bash
# Clone your fork
git clone https://github.com/your-username/spark-optima.git
cd spark-optima

# Install dependencies
poetry install --all-extras

# Activate virtual environment
poetry shell

# Install pre-commit hooks
pre-commit install

# Run tests to ensure everything is working
make test
```

### Development Workflow

```bash
# Run all checks before committing
make check-all

# Run tests
make test

# Run specific test
pytest tests/unit/test_optimizer.py -v

# Format code
make format

# Run linter
make lint

# Type checking
make type-check
```

## Project Structure

```
spark-optima/
├── src/spark_optima/      # Main source code
│   ├── core/              # Core optimization engine
│   ├── platforms/         # Platform adapters
│   ├── analysis/          # Code analysis
│   ├── cli/               # Command-line interface
│   ├── api/               # REST API
│   └── data/              # Data handling
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── docs/                  # Documentation
├── docker/                # Docker configurations
└── kubernetes/            # Kubernetes manifests
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:
- Line length: 100 characters
- Use type hints for all function signatures
- Use Google-style docstrings
- Use f-strings for string formatting

### Example

```python
def optimize_configuration(
    code_path: Path,
    data_profile: dict[str, Any] | None = None,
) -> Optimization