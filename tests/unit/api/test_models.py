# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for API models module.

This module tests the Pydantic models used for API request/response validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spark_optima.api.models import (
    AnalysisRequest,
    DataFormat,
    DataProfileInput,
    OptimizationMode,
    OptimizationRequest,
    Platform,
    ResourceConstraintsInput,
    ResourceSpecInput,
)


class TestPlatformEnum:
    """Tests for Platform enum."""

    def test_platform_values(self) -> None:
        """Test that platform enum has expected values."""
        assert Platform.LOCAL == "local"
        assert Platform.AWS_GLUE == "aws_glue"
        assert Platform.AWS_EMR == "aws_emr"
        assert Platform.DATABRICKS == "databricks"
        assert Platform.AZURE_SYNAPSE == "azure_synapse"
        assert Platform.GCP_DATAPROC == "gcp_dataproc"
        assert Platform.KUBERNETES == "kubernetes"

    def test_platform_enum_count(self) -> None:
        """Test that all expected platforms are defined."""
        platforms = list(Platform)
        assert len(platforms) == 7


class TestDataFormatEnum:
    """Tests for DataFormat enum."""

    def test_data_format_values(self) -> None:
        """Test that data format enum has expected values."""
        assert DataFormat.PARQUET == "parquet"
        assert DataFormat.DELTA == "delta"
        assert DataFormat.JSON == "json"
        assert DataFormat.CSV == "csv"
        assert DataFormat.ORC == "orc"
        assert DataFormat.AVRO == "avro"


class TestOptimizationModeEnum:
    """Tests for OptimizationMode enum."""

    def test_optimization_mode_values(self) -> None:
        """Test that optimization mode enum has expected values."""
        assert OptimizationMode.SIMULATION == "simulation"
        assert OptimizationMode.EXECUTION == "execution"


class TestResourceSpecInput:
    """Tests for ResourceSpecInput model."""

    def test_valid_resource_spec(self) -> None:
        """Test creating valid resource spec."""
        spec = ResourceSpecInput(
            cpu_cores=8,
            memory_gb=32.0,
            disk_gb=100.0,
            gpu_count=0,
        )
        assert spec.cpu_cores == 8
        assert spec.memory_gb == 32.0
        assert spec.disk_gb == 100.0
        assert spec.gpu_count == 0

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        spec = ResourceSpecInput(
            cpu_cores=4,
            memory_gb=16.0,
        )
        assert spec.disk_gb == 0.0
        assert spec.gpu_count == 0

    def test_invalid_cpu_cores(self) -> None:
        """Test validation for invalid CPU cores."""
        with pytest.raises(ValidationError):
            ResourceSpecInput(cpu_cores=0, memory_gb=16.0)

        with pytest.raises(ValidationError):
            ResourceSpecInput(cpu_cores=200, memory_gb=16.0)

    def test_invalid_memory(self) -> None:
        """Test validation for invalid memory."""
        with pytest.raises(ValidationError):
            ResourceSpecInput(cpu_cores=4, memory_gb=0.0)

        with pytest.raises(ValidationError):
            ResourceSpecInput(cpu_cores=4, memory_gb=3000.0)


class TestDataProfileInput:
    """Tests for DataProfileInput model."""

    def test_valid_data_profile(self) -> None:
        """Test creating valid data profile."""
        profile = DataProfileInput(
            size_gb=100.0,
            format=DataFormat.PARQUET,
            schema={"id": "int", "name": "string"},
            compression="snappy",
            partitioning=["date"],
        )
        assert profile.size_gb == 100.0
        assert profile.format == DataFormat.PARQUET
        assert profile.schema_info is not None

    def test_minimal_data_profile(self) -> None:
        """Test creating data profile with minimal fields."""
        profile = DataProfileInput(
            size_gb=10.0,
            format=DataFormat.DELTA,
        )
        assert profile.size_gb == 10.0
        assert profile.format == DataFormat.DELTA
        assert profile.schema_info is None
        assert profile.compression is None

    def test_invalid_size(self) -> None:
        """Test validation for invalid data size."""
        with pytest.raises(ValidationError):
            DataProfileInput(size_gb=0.0, format=DataFormat.CSV)

        with pytest.raises(ValidationError):
            DataProfileInput(size_gb=-1.0, format=DataFormat.CSV)


class TestResourceConstraintsInput:
    """Tests for ResourceConstraintsInput model."""

    def test_valid_constraints(self) -> None:
        """Test creating valid constraints."""
        constraints = ResourceConstraintsInput(
            max_memory_gb=64.0,
            max_cost_per_hour=5.0,
            max_executors=10,
            timeout_minutes=30,
        )
        assert constraints.max_memory_gb == 64.0
        assert constraints.max_cost_per_hour == 5.0
        assert constraints.max_executors == 10
        assert constraints.timeout_minutes == 30

    def test_optional_fields(self) -> None:
        """Test that all fields are optional."""
        constraints = ResourceConstraintsInput()
        assert constraints.max_memory_gb is None
        assert constraints.max_cost_per_hour is None
        assert constraints.max_executors is None
        assert constraints.timeout_minutes is None

    def test_invalid_constraints(self) -> None:
        """Test validation for invalid constraints."""
        with pytest.raises(ValidationError):
            ResourceConstraintsInput(max_memory_gb=0.0)

        with pytest.raises(ValidationError):
            ResourceConstraintsInput(max_cost_per_hour=-1.0)

        with pytest.raises(ValidationError):
            ResourceConstraintsInput(max_executors=0)

        with pytest.raises(ValidationError):
            ResourceConstraintsInput(timeout_minutes=0)


class TestOptimizationRequest:
    """Tests for OptimizationRequest model."""

    def test_valid_request(self) -> None:
        """Test creating valid optimization request."""
        request = OptimizationRequest(
            code="from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()",
            platform=Platform.LOCAL,
            spark_version="3.5.0",
            resources=ResourceSpecInput(cpu_cores=4, memory_gb=16.0),
            data_profile=DataProfileInput(size_gb=10.0, format=DataFormat.PARQUET),
            use_bayesian=True,
            bayesian_trials=50,
            objectives=["minimize_time"],
        )
        assert request.platform == Platform.LOCAL
        assert request.spark_version == "3.5.0"
        assert request.use_bayesian is True

    def test_minimal_request(self) -> None:
        """Test creating request with minimal required fields."""
        request = OptimizationRequest(
            code="from pyspark.sql import SparkSession",
            platform=Platform.DATABRICKS,
            resources=ResourceSpecInput(cpu_cores=8, memory_gb=32.0),
        )
        assert request.platform == Platform.DATABRICKS
        assert request.spark_version == "3.5.0"  # default
        assert request.use_bayesian is True  # default
        assert request.bayesian_trials == 50  # default

    def test_code_min_length(self) -> None:
        """Test that code has minimum length requirement."""
        with pytest.raises(ValidationError):
            OptimizationRequest(
                code="short",
                platform=Platform.LOCAL,
                resources=ResourceSpecInput(cpu_cores=4, memory_gb=16.0),
            )

    def test_invalid_spark_version(self) -> None:
        """Test validation for invalid Spark version format."""
        with pytest.raises(ValidationError):
            OptimizationRequest(
                code="from pyspark.sql import SparkSession",
                platform=Platform.LOCAL,
                spark_version="invalid",
                resources=ResourceSpecInput(cpu_cores=4, memory_gb=16.0),
            )

    def test_bayesian_trials_range(self) -> None:
        """Test validation for bayesian trials range."""
        with pytest.raises(ValidationError):
            OptimizationRequest(
                code="from pyspark.sql import SparkSession",
                platform=Platform.LOCAL,
                resources=ResourceSpecInput(cpu_cores=4, memory_gb=16.0),
                bayesian_trials=0,
            )

        with pytest.raises(ValidationError):
            OptimizationRequest(
                code="from pyspark.sql import SparkSession",
                platform=Platform.LOCAL,
                resources=ResourceSpecInput(cpu_cores=4, memory_gb=16.0),
                bayesian_trials=1000,
            )

    def test_valid_objectives(self) -> None:
        """Test valid optimization objectives."""
        request = OptimizationRequest(
            code="from pyspark.sql import SparkSession",
            platform=Platform.LOCAL,
            resources=ResourceSpecInput(cpu_cores=4, memory_gb=16.0),
            objectives=["minimize_time", "minimize_cost"],
        )
        assert "minimize_time" in request.objectives
        assert "minimize_cost" in request.objectives

    def test_invalid_objectives(self) -> None:
        """Test validation for invalid objectives."""
        with pytest.raises(ValidationError):
            OptimizationRequest(
                code="from pyspark.sql import SparkSession",
                platform=Platform.LOCAL,
                resources=ResourceSpecInput(cpu_cores=4, memory_gb=16.0),
                objectives=["invalid_objective"],
            )


class TestAnalysisRequest:
    """Tests for AnalysisRequest model."""

    def test_valid_request(self) -> None:
        """Test creating valid analysis request."""
        request = AnalysisRequest(
            code="from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()",
        )
        assert len(request.code) > 10

    def test_code_min_length(self) -> None:
        """Test that code has minimum length requirement."""
        with pytest.raises(ValidationError):
            AnalysisRequest(code="short")
