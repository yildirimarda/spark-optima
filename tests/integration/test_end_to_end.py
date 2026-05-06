# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Integration tests for end-to-end workflows.

This module contains integration tests that verify complete workflows
from input to output across multiple components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spark_optima.core.optimizer import Optimizer
from spark_optima.core.result import OptimizationResult

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_spark_job(tmp_path: Path) -> Path:
    """Create a sample Spark job file for integration testing."""
    job_file = tmp_path / "integration_test_job.py"
    job_content = '''
"""
Sample Spark job for integration testing.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum

# Initialize Spark session
spark = SparkSession.builder \\
    .appName("IntegrationTestJob") \\
    .getOrCreate()

# Read input data
sales_df = spark.read.parquet("/data/sales.parquet")
customers_df = spark.read.parquet("/data/customers.parquet")

# Join and aggregate
result_df = sales_df.join(
    customers_df,
    sales_df.customer_id == customers_df.id,
    "inner"
).groupBy("region").agg(
    spark_sum("amount").alias("total_sales"),
    spark_sum("quantity").alias("total_quantity")
)

# Write output
result_df.write.mode("overwrite").parquet("/output/sales_summary.parquet")

spark.stop()
'''
    job_file.write_text(job_content)
    return job_file


@pytest.mark.integration
class TestEndToEndOptimization:
    """End-to-end tests for the optimization workflow."""

    def test_full_optimization_workflow_local(self, sample_spark_job: Path) -> None:
        """Test complete optimization workflow on local platform."""
        optimizer = Optimizer(
            platform="local",
            spark_version="3.5.0",
            optimization_mode="simulation",
        )

        result = optimizer.optimize(
            code_path=sample_spark_job,
            use_bayesian=False,  # Faster for integration test
        )

        assert isinstance(result, OptimizationResult)
        assert len(result.configuration) > 0
        assert result.estimated_time_minutes >= 0
        assert 0 <= result.confidence_score <= 1

    def test_optimization_with_code_analysis(self, sample_spark_job: Path) -> None:
        """Test optimization including code analysis."""
        optimizer = Optimizer(platform="local")

        result = optimizer.optimize(code_path=sample_spark_job)

        assert isinstance(result, OptimizationResult)
        # Should detect join operation and provide suggestions
        assert isinstance(result.code_suggestions, list)

    def test_optimization_with_data_profile(self, sample_spark_job: Path) -> None:
        """Test optimization with data profile."""
        optimizer = Optimizer(platform="local")

        data_profile = {
            "size_gb": 500,
            "format": "parquet",
        }

        result = optimizer.optimize(
            code_path=sample_spark_job,
            data_profile=data_profile,
            use_bayesian=False,
        )

        assert isinstance(result, OptimizationResult)

    def test_optimization_across_all_platforms(self, sample_spark_job: Path) -> None:
        """Test optimization across all supported platforms."""
        platforms = ["local", "databricks", "aws_glue", "azure_synapse"]

        for platform in platforms:
            optimizer = Optimizer(
                platform=platform,
                optimization_mode="simulation",
            )

            result = optimizer.optimize(
                code_path=sample_spark_job,
                use_bayesian=False,
            )

            assert isinstance(result, OptimizationResult)
            assert result.platform_specific["platform"] == platform


@pytest.mark.integration
class TestConfigurationExport:
    """Tests for configuration export workflows."""

    def test_optimizer_result_to_dict(self, sample_spark_job: Path) -> None:
        """Test converting optimization result to dictionary."""
        optimizer = Optimizer(platform="local")
        result = optimizer.optimize(code_path=sample_spark_job, use_bayesian=False)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "configuration" in result_dict
        assert "estimated_time_minutes" in result_dict
        assert "confidence_score" in result_dict


@pytest.mark.integration
class TestHeuristicAndBayesianIntegration:
    """Tests for heuristic and Bayesian optimization integration."""

    def test_heuristic_only_optimization(self, sample_spark_job: Path) -> None:
        """Test optimization with heuristics only."""
        optimizer = Optimizer(platform="local")

        result = optimizer.optimize(
            code_path=sample_spark_job,
            use_bayesian=False,
        )

        assert isinstance(result, OptimizationResult)
        heuristic_config = optimizer.get_heuristic_config()
        assert heuristic_config is not None
        assert result.configuration == heuristic_config

    def test_bayesian_optimization_with_heuristic_seed(self, sample_spark_job: Path) -> None:
        """Test Bayesian optimization seeded with heuristic config."""
        optimizer = Optimizer(platform="local")

        # Run with Bayesian optimization (limited trials for speed)
        result = optimizer.optimize(
            code_path=sample_spark_job,
            use_bayesian=True,
            bayesian_trials=5,
        )

        assert isinstance(result, OptimizationResult)
        # Should have used both heuristic and Bayesian
        assert result.metadata.get("bayesian_used", False) is True


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling in workflows."""

    def test_optimization_with_nonexistent_file(self) -> None:
        """Test error handling for non-existent file."""
        optimizer = Optimizer(platform="local")

        with pytest.raises(FileNotFoundError):
            optimizer.optimize(code_path="/nonexistent/file.py")

    def test_optimization_with_empty_file(self, tmp_path: Path) -> None:
        """Test optimization with empty file."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        optimizer = Optimizer(platform="local")
        result = optimizer.optimize(code_path=empty_file, use_bayesian=False)

        assert isinstance(result, OptimizationResult)

    def test_optimization_with_invalid_code(self, tmp_path: Path) -> None:
        """Test optimization with invalid Python code."""
        invalid_file = tmp_path / "invalid.py"
        invalid_file.write_text("this is not valid python {{{")

        optimizer = Optimizer(platform="local")
        # Should handle gracefully, possibly with warnings
        result = optimizer.optimize(code_path=invalid_file, use_bayesian=False)

        assert isinstance(result, OptimizationResult)
