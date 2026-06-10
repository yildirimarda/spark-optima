# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Pytest configuration and shared fixtures for Spark Optima tests.

This module provides shared fixtures and configuration for all tests
in the Spark Optima test suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from spark_optima.core.config_engine.models import ConfigSet
from spark_optima.platforms.models import CostModel, ResourceSpec, WorkerType

# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root: Path) -> Path:
    """Return the test data directory."""
    return project_root / "tests" / "data"


@pytest.fixture
def sample_spark_code_file(tmp_path: Path) -> Path:
    """Create a sample Spark code file for testing."""
    code_file = tmp_path / "sample_spark_job.py"
    code_content = """
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \\
    .appName("SampleJob") \\
    .getOrCreate()

# Read data
df = spark.read.parquet("input.parquet")

# Transform
df_filtered = df.filter(df.col1 > 10)
df_grouped = df_filtered.groupBy("col2").agg({"col3": "sum"})

# Write output
df_grouped.write.parquet("output.parquet")

spark.stop()
"""
    code_file.write_text(code_content)
    return code_file


@pytest.fixture
def simple_spark_code_file(tmp_path: Path) -> Path:
    """Create a simple Spark code file for testing."""
    code_file = tmp_path / "simple_job.py"
    code_content = """
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("data.parquet")
df.show()
"""
    code_file.write_text(code_content)
    return code_file


# =============================================================================
# Resource Fixtures
# =============================================================================


@pytest.fixture
def default_resource_spec() -> ResourceSpec:
    """Return a default resource specification."""
    return ResourceSpec(
        cpu_cores=8,
        memory_gb=32.0,
        disk_gb=100.0,
    )


@pytest.fixture
def small_resource_spec() -> ResourceSpec:
    """Return a small resource specification."""
    return ResourceSpec(
        cpu_cores=2,
        memory_gb=8.0,
        disk_gb=20.0,
    )


@pytest.fixture
def large_resource_spec() -> ResourceSpec:
    """Return a large resource specification."""
    return ResourceSpec(
        cpu_cores=64,
        memory_gb=256.0,
        disk_gb=1000.0,
    )


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def sample_spark_config() -> dict[str, Any]:
    """Return a sample Spark configuration."""
    return {
        "spark.executor.memory": "4g",
        "spark.executor.cores": "4",
        "spark.driver.memory": "4g",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.shuffle.partitions": "200",
    }


@pytest.fixture
def minimal_spark_config() -> dict[str, Any]:
    """Return a minimal Spark configuration."""
    return {
        "spark.executor.memory": "2g",
        "spark.executor.cores": "2",
    }


@pytest.fixture
def empty_config_set() -> ConfigSet:
    """Return an empty configuration set."""
    return ConfigSet(
        version="3.5.0",
        parameters={},
        metadata={},
    )


# =============================================================================
# Data Profile Fixtures
# =============================================================================


@pytest.fixture
def sample_data_profile() -> dict[str, Any]:
    """Return a sample data profile."""
    return {
        "size_gb": 100,
        "format": "parquet",
        "schema": {
            "fields": [
                {"name": "id", "type": "integer"},
                {"name": "name", "type": "string"},
            ]
        },
    }


@pytest.fixture
def large_data_profile() -> dict[str, Any]:
    """Return a large data profile."""
    return {
        "size_gb": 1000,
        "format": "parquet",
        "compression": "snappy",
    }


# =============================================================================
# Worker Type Fixtures
# =============================================================================


@pytest.fixture
def sample_worker_type() -> WorkerType:
    """Return a sample worker type."""
    return WorkerType(
        name="Standard",
        size="medium",
        resources=ResourceSpec(
            cpu_cores=4,
            memory_gb=16.0,
        ),
        cost=CostModel(
            unit_cost_per_hour=0.50,
            unit_name="instance",
        ),
    )


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_config_database() -> MagicMock:
    """Return a mock configuration database."""
    mock_db = MagicMock()
    mock_db.get_available_versions.return_value = ["3.4.0", "3.5.0", "4.0.0"]
    mock_db.get_config_set.return_value = ConfigSet(
        version="3.5.0",
        parameters={},
        metadata={},
    )
    return mock_db


@pytest.fixture
def mock_platform() -> MagicMock:
    """Return a mock platform."""
    mock = MagicMock()
    mock.name = "mock_platform"
    mock.display_name = "Mock Platform"
    mock.constraints.min_workers = 1
    mock.constraints.max_workers = 100
    return mock


@pytest.fixture
def sample_config_data() -> dict:
    """Return sample configuration data for testing."""
    return {
        "3.4.0": {
            "version": "3.4.0",
            "parameters": {
                "spark.executor.memory": {
                    "name": "spark.executor.memory",
                    "type": "bytes",
                    "default": "4g",
                    "category": "memory",
                },
                "spark.driver.memory": {
                    "name": "spark.driver.memory",
                    "type": "bytes",
                    "default": "4g",
                    "category": "memory",
                },
            },
        },
        "3.5.0": {
            "version": "3.5.0",
            "parameters": {
                "spark.executor.memory": {
                    "name": "spark.executor.memory",
                    "type": "bytes",
                    "default": "4g",
                    "category": "memory",
                },
                "spark.driver.memory": {
                    "name": "spark.driver.memory",
                    "type": "bytes",
                    "default": "4g",
                    "category": "memory",
                },
            },
        },
    }


@pytest.fixture
def mock_config_dir(tmp_path: Path, sample_config_data: dict) -> Path:
    """Create a mock config directory with sample data."""
    import yaml

    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)

    # Write sample config files
    for version, data in sample_config_data.items():
        config_file = config_dir / f"spark_{version}_configs.yaml"
        with open(config_file, "w") as f:
            yaml.dump(data, f)

    return config_dir


# =============================================================================
# Test Configuration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
