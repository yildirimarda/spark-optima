# Testing Guide

This guide covers the testing strategy, test organization, and how to write tests for Spark Optima.

## Testing Philosophy

Spark Optima follows these testing principles:

1. **Comprehensive Coverage** - Target 90%+ code coverage
2. **Fast Feedback** - Unit tests run in seconds
3. **Isolation** - Tests don't depend on external services
4. **Clarity** - Tests document expected behavior
5. **Maintainability** - Tests are easy to update

## Test Organization

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── unit/                          # Unit tests
│   ├── test_optimizer.py
│   ├── test_result.py
│   ├── analysis/
│   │   ├── test_parser.py
│   │   ├── test_smell_detector.py
│   │   └── test_recommender.py
│   ├── api/
│   │   ├── test_models.py
│   │   └── routes/
│   │       ├── test_health.py
│   │       ├── test_optimize.py
│   │       └── test_platforms.py
│   ├── cli/
│   │   └── test_main.py
│   ├── core/
│   │   ├── bayesian/
│   │   │   ├── test_optimizer.py
│   │   │   └── test_search_space.py
│   │   ├── config_engine/
│   │   │   └── test_loader.py
│   │   └── heuristics/
│   │       └── test_engine.py
│   ├── platforms/
│   │   ├── test_local.py
│   │   ├── test_databricks.py
│   │   └── test_aws_glue.py
│   └── data/
│       └── test_generators.py
└── integration/                   # Integration tests
    └── test_end_to_end.py
```

## Running Tests

### Run All Tests

```bash
# Using Poetry
poetry run pytest

# Using Make
make test

# Using pytest directly (in poetry shell)
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Run specific test file
pytest tests/unit/test_optimizer.py

# Run specific test class
pytest tests/unit/test_optimizer.py::TestOptimizer

# Run specific test method
pytest tests/unit/test_optimizer.py::TestOptimizer::test_initialization
```

### Run with Coverage

```bash
# HTML coverage report
pytest --cov=src/spark_optima --cov-report=html

# Terminal coverage report
pytest --cov=src/spark_optima --cov-report=term

# XML report for CI
pytest --cov=src/spark_optima --cov-report=xml

# Using Make
make test-coverage
```

## Test Fixtures

### Available Fixtures

Located in `tests/conftest.py`:

```python
# Path fixtures
@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent

# Resource fixtures
@pytest.fixture
def default_resource_spec() -> ResourceSpec:
    """Return a default resource specification."""
    return ResourceSpec(cpu_cores=8, memory_gb=32.0)

# Configuration fixtures
@pytest.fixture
def sample_spark_config() -> dict[str, Any]:
    """Return a sample Spark configuration."""
    return {
        "spark.executor.memory": "4g",
        "spark.executor.cores": "4",
    }

# Data profile fixtures
@pytest.fixture
def sample_data_profile() -> dict[str, Any]:
    """Return a sample data profile."""
    return {"size_gb": 100, "format": "parquet"}

# Mock fixtures
@pytest.fixture
def mock_config_database() -> MagicMock:
    """Return a mock configuration database."""
    mock_db = MagicMock()
    mock_db.get_available_versions.return_value = ["3.4.0", "3.5.0"]
    return mock_db
```

### Using Fixtures

```python
def test_optimizer_with_resources(
    default_resource_spec: ResourceSpec,
    sample_data_profile: dict[str, Any]
):
    """Test optimizer with resource specifications."""
    optimizer = Optimizer(platform="local")
    
    result = optimizer.optimize(
        resources=default_resource_spec,
        data_profile=sample_data_profile
    )
    
    assert result.configuration is not None
    assert result.estimated_time_minutes > 0
```

## Writing Tests

### Test Structure

Follow the Arrange-Act-Assert pattern:

```python
def test_feature_description():
    """Test description of what is being tested."""
    # Arrange - Set up test data and conditions
    optimizer = Optimizer(platform="local")
    code_path = "test_job.py"
    
    # Act - Execute the code being tested
    result = optimizer.optimize(code_path=code_path)
    
    # Assert - Verify the expected outcome
    assert result is not None
    assert result.confidence_score > 0
```

### Test Markers

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_optimizer_initialization():
    """Fast unit test."""
    pass

@pytest.mark.integration
def test_end_to_end_optimization():
    """Integration test with external dependencies."""
    pass

@pytest.mark.slow
def test_bayesian_optimization_with_many_trials():
    """Slow test that takes significant time."""
    pass
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch, MagicMock

def test_optimizer_with_mocked_database():
    """Test optimizer with mocked configuration database."""
    mock_db = MagicMock()
    mock_db.get_config_set.return_value = ConfigSet(
        version="3.5.0",
        parameters={}
    )
    
    with patch('spark_optima.core.optimizer.ConfigDatabase', return_value=mock_db):
        optimizer = Optimizer(platform="local")
        result = optimizer.optimize(code_path="test.py")
        
        assert result is not None
```

### Testing Exceptions

```python
import pytest

def test_optimizer_raises_on_invalid_platform():
    """Test that optimizer raises ValueError for invalid platform."""
    with pytest.raises(ValueError, match="Invalid platform"):
        Optimizer(platform="invalid_platform")

def test_optimizer_raises_on_missing_file():
    """Test that optimizer raises FileNotFoundError for missing code file."""
    optimizer = Optimizer(platform="local")
    
    with pytest.raises(FileNotFoundError):
        optimizer.optimize(code_path="/nonexistent/file.py")
```

## Test Categories

### Unit Tests

Test individual components in isolation:

```python
# tests/unit/core/test_heuristics.py
class TestHeuristicEngine:
    """Test heuristic optimization engine."""
    
    def test_memory_heuristics(self):
        """Test memory-related heuristic rules."""
        engine = HeuristicEngine(config_set)
        context = HeuristicContext(resources=ResourceSpec(memory_gb=64))
        
        result = engine.evaluate(context)
        
        assert "spark.executor.memory" in result
        assert result["spark.executor.memory"] == "8g"
    
    def test_cpu_heuristics(self):
        """Test CPU-related heuristic rules."""
        pass
```

### Integration Tests

Test component interactions:

```python
# tests/integration/test_end_to_end.py
@pytest.mark.integration
def test_full_optimization_workflow():
    """Test complete optimization workflow."""
    optimizer = Optimizer(platform="local")
    
    result = optimizer.optimize(
        code_path="sample_job.py",
        data_profile={"size_gb": 100},
        bayesian_trials=10
    )
    
    assert result.configuration
    assert result.estimated_time_minutes > 0
    assert result.confidence_score > 0
```

### Property-Based Tests

Use hypothesis for property-based testing:

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=1000))
def test_optimizer_handles_various_data_sizes(size_gb):
    """Test optimizer handles various data sizes."""
    optimizer = Optimizer(platform="local")
    
    result = optimizer.optimize(
        data_profile={"size_gb": size_gb, "format": "parquet"}
    )
    
    assert result.estimated_time_minutes > 0
```

## Best Practices

### 1. Test One Thing at a Time

```python
# Good - Tests one specific behavior
def test_optimizer_initialization_valid_platform():
    optimizer = Optimizer(platform="local")
    assert optimizer.platform == "local"

# Less optimal - Tests multiple behaviors
def test_optimizer():
    optimizer = Optimizer(platform="local")
    assert optimizer.platform == "local"
    result = optimizer.optimize(code_path="job.py")
    assert result is not None
    assert result.configuration
```

### 2. Use Descriptive Test Names

```python
# Good - Clear what is being tested
def test_optimizer_raises_value_error_for_invalid_platform():
    pass

# Less optimal - Vague
def test_optimizer_invalid():
    pass
```

### 3. Keep Tests Independent

```python
# Good - Each test is independent
def test_optimizer_with_small_data():
    optimizer = Optimizer(platform="local")
    result = optimizer.optimize(data_profile={"size_gb": 10})
    assert result.estimated_time_minutes < 10

def test_optimizer_with_large_data():
    optimizer = Optimizer(platform="local")
    result = optimizer.optimize(data_profile={"size_gb": 1000})
    assert result.estimated_time_minutes > 10
```

### 4. Use Parameterized Tests

```python
import pytest

@pytest.mark.parametrize("platform", [
    "local",
    "aws_glue",
    "databricks",
    "azure_synapse"
])
def test_optimizer_supports_all_platforms(platform):
    """Test that optimizer supports all platforms."""
    optimizer = Optimizer(platform=platform)
    assert optimizer.platform == platform
```

### 5. Clean Up After Tests

```python
import tempfile
import os

@pytest.fixture
def temp_code_file():
    """Create a temporary code file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("# Test code")
        temp_path = f.name
    
    yield temp_path
    
    # Clean up
    os.unlink(temp_path)
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install Poetry
        run: pip install poetry
      
      - name: Install dependencies
        run: poetry install --all-extras
      
      - name: Run tests
        run: poetry run pytest -m unit --cov=src/spark_optima --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

## Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| Core optimizer | 95% |
| CLI | 90% |
| API | 90% |
| Platforms | 85% |
| Analysis | 90% |
| **Overall** | **90%** |

## Debugging Tests

### Verbose Output

```bash
pytest -v                    # Verbose
pytest -vv                   # Very verbose
pytest -s                    # Show print statements
pytest --tb=long             # Detailed traceback
pytest --tb=short            # Short traceback
pytest --pdb                 # Drop into debugger on failure
```

### Selective Test Running

```bash
# Run tests matching pattern
pytest -k "test_optimizer"

# Run tests in specific directory
pytest tests/unit/core/

# Run until first failure
pytest -x

# Run with fail fast
pytest --maxfail=3
```

## Common Issues

### Issue: Tests are slow

**Solution:**
- Use `pytest -m "not slow"` to skip slow tests
- Use mocks for external dependencies
- Reduce Bayesian trial counts in tests

### Issue: Tests fail due to missing dependencies

**Solution:**
```bash
# Install all extras
poetry install --all-extras

# Or install specific extras
poetry install --extras "aws databricks"
```

### Issue: Coverage is incomplete

**Solution:**
```bash
# Check which lines are uncovered
pytest --cov=src/spark_optima --cov-report=term-missing

# Generate HTML report
pytest --cov=src/spark_optima --cov-report=html
# Open htmlcov/index.html
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)