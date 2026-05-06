# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for CLI formatters module.

This module tests the configuration export and display functionality.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from spark_optima.cli.formatters import ConfigExporter, display_results_table
from spark_optima.core.result import CodeSuggestion, OptimizationResult

if TYPE_CHECKING:
    from pathlib import Path


class TestConfigExporter:
    """Tests for ConfigExporter class."""

    @pytest.fixture
    def sample_result(self) -> OptimizationResult:
        """Create a sample optimization result for testing."""
        return OptimizationResult(
            configuration={
                "spark.executor.memory": "4g",
                "spark.executor.cores": "2",
                "spark.driver.memory": "2g",
                "spark.sql.shuffle.partitions": "200",
            },
            estimated_time_minutes=15.5,
            confidence_score=0.85,
            code_suggestions=[
                CodeSuggestion(
                    line_number=10,
                    issue_type="broadcast",
                    description="Consider broadcasting small table",
                    suggestion="Use broadcast hint",
                    severity="medium",
                ),
            ],
            platform_specific={
                "platform": "local",
                "spark_version": "3.5.0",
                "spark_config": {
                    "spark.executor.memory": "4g",
                    "spark.executor.cores": "2",
                },
            },
        )

    def test_export_json(self, sample_result: OptimizationResult) -> None:
        """Test JSON export functionality."""
        exporter = ConfigExporter(sample_result, "local")
        json_output = exporter.export_json()

        # Verify it's valid JSON
        data = json.loads(json_output)
        assert "configuration" in data
        assert "estimated_time_minutes" in data
        assert data["estimated_time_minutes"] == 15.5

    def test_export_yaml(self, sample_result: OptimizationResult) -> None:
        """Test YAML export functionality."""
        exporter = ConfigExporter(sample_result, "local")
        yaml_output = exporter.export_yaml()

        # Verify YAML contains expected content
        assert "configuration:" in yaml_output
        assert "spark.executor.memory" in yaml_output
        assert "estimated_time_minutes: 15.5" in yaml_output

    def test_export_spark_submit(self, sample_result: OptimizationResult) -> None:
        """Test spark-submit export functionality."""
        exporter = ConfigExporter(sample_result, "local")
        output = exporter.export_spark_submit()

        assert "spark-submit" in output
        assert "spark.executor.memory" in output
        assert "your_script.py" in output

    def test_export_environment_variables(self, sample_result: OptimizationResult) -> None:
        """Test environment variables export."""
        exporter = ConfigExporter(sample_result, "local")
        output = exporter.export_environment_variables()

        assert "export SPARK_EXECUTOR_MEMORY" in output
        assert "export SPARK_DRIVER_MEMORY" in output
        assert "4g" in output

    def test_export_properties_file(self, sample_result: OptimizationResult) -> None:
        """Test properties file export."""
        exporter = ConfigExporter(sample_result, "local")
        output = exporter.export_properties_file()

        assert "spark.executor.memory 4g" in output
        assert "spark-defaults.conf" in output

    def test_export_databricks_json(self, sample_result: OptimizationResult) -> None:
        """Test Databricks JSON export."""
        exporter = ConfigExporter(sample_result, "databricks")
        output = exporter.export_databricks_json()

        data = json.loads(output)
        assert data["cluster_name"] == "spark-optima-optimized"
        assert "spark_conf" in data

    def test_export_aws_glue(self, sample_result: OptimizationResult) -> None:
        """Test AWS Glue export."""
        exporter = ConfigExporter(sample_result, "aws_glue")
        output = exporter.export_aws_glue()

        data = json.loads(output)
        assert data["Name"] == "spark-optima-job"
        assert "DefaultArguments" in data

    def test_export_azure_synapse(self, sample_result: OptimizationResult) -> None:
        """Test Azure Synapse export."""
        exporter = ConfigExporter(sample_result, "azure_synapse")
        output = exporter.export_azure_synapse()

        assert "Azure Synapse" in output
        assert "spark.conf.set" in output

    def test_save_to_file(self, sample_result: OptimizationResult, tmp_path: Path) -> None:
        """Test saving to file functionality."""
        exporter = ConfigExporter(sample_result, "local")
        output_file = tmp_path / "config.json"

        exporter.save_to_file(output_file, "json")

        assert output_file.exists()
        content = output_file.read_text()
        data = json.loads(content)
        assert "configuration" in data

    def test_save_to_file_unsupported_format(
        self, sample_result: OptimizationResult, tmp_path: Path
    ) -> None:
        """Test saving with unsupported format raises error."""
        exporter = ConfigExporter(sample_result, "local")
        output_file = tmp_path / "config.txt"

        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.save_to_file(output_file, "unsupported")

    def test_databricks_autoscaling(self, sample_result: OptimizationResult) -> None:
        """Test Databricks export with autoscaling enabled."""
        # Add dynamic allocation settings
        sample_result.configuration["spark.dynamicAllocation.enabled"] = "true"
        sample_result.configuration["spark.dynamicAllocation.minExecutors"] = "2"
        sample_result.configuration["spark.dynamicAllocation.maxExecutors"] = "10"

        exporter = ConfigExporter(sample_result, "databricks")
        output = exporter.export_databricks_json()

        data = json.loads(output)
        assert "autoscale" in data
        assert data["autoscale"]["min_workers"] == 2
        assert data["autoscale"]["max_workers"] == 10


class TestDisplayResults:
    """Tests for display functions."""

    @pytest.fixture
    def sample_result(self) -> OptimizationResult:
        """Create a sample optimization result."""
        return OptimizationResult(
            configuration={
                "spark.executor.memory": "4g",
                "spark.executor.cores": "2",
            },
            estimated_time_minutes=10.0,
            confidence_score=0.9,
            code_suggestions=[],
            platform_specific={},
        )

    def test_display_results_table(
        self, sample_result: OptimizationResult, capsys: pytest.CaptureFixture
    ) -> None:
        """Test that display_results_table runs without errors."""
        # This mainly tests that the function doesn't raise exceptions
        # since Rich output is hard to capture
        display_results_table(sample_result)
        # If we get here without exception, the test passes

    def test_display_results_table_with_suggestions(self, capsys: pytest.CaptureFixture) -> None:
        """Test display_results_table with code suggestions (lines 406-433)."""
        from spark_optima.core.result import CodeSuggestion

        result = OptimizationResult(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.9,
            code_suggestions=[
                CodeSuggestion(
                    line_number=10,
                    issue_type="broadcast",
                    description="Consider broadcast join",
                    suggestion="Use broadcast hint",
                    severity="critical",
                ),
                CodeSuggestion(
                    line_number=20,
                    issue_type="shuffle",
                    description="Reduce shuffle",
                    suggestion="Increase partitions",
                    severity="high",
                ),
                CodeSuggestion(
                    line_number=30,
                    issue_type="memory",
                    description="Memory issue",
                    suggestion="Increase memory",
                    severity="medium",
                ),
                CodeSuggestion(
                    line_number=40,
                    issue_type="cpu",
                    description="CPU issue",
                    suggestion="Increase cores",
                    severity="low",
                ),
                CodeSuggestion(
                    line_number=50,
                    issue_type="other",
                    description="Another issue",
                    suggestion="Fix it",
                    severity="low",
                ),
                CodeSuggestion(
                    line_number=60,
                    issue_type="more",
                    description="Yet another",
                    suggestion="Fix too",
                    severity="low",
                ),
            ],
            platform_specific={},
        )

        # Should not raise
        display_results_table(result)

    def test_display_results_table_no_suggestions(self, capsys: pytest.CaptureFixture) -> None:
        """Test display_results_table without code suggestions."""
        result = OptimizationResult(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.9,
            code_suggestions=[],
            platform_specific={},
        )

        # Should not raise
        display_results_table(result)

    def test_print_platform_export_help_local(self) -> None:
        """Test print_platform_export_help for local (lines 448-475)."""
        from spark_optima.cli.formatters import print_platform_export_help

        # Should not raise
        print_platform_export_help("local")
        print_platform_export_help("databricks")
        print_platform_export_help("aws_glue")
        print_platform_export_help("azure_synapse")
        # Unknown platform
        print_platform_export_help("unknown")

    def test_export_databricks_cli(self, sample_result: OptimizationResult) -> None:
        """Test Databricks CLI export (lines 130-149)."""
        exporter = ConfigExporter(sample_result, "databricks")
        output = exporter.export_databricks_cli()

        assert "databricks" in output.lower()
        assert "cluster" in output.lower()
        assert "EOF" in output

    def test_export_aws_cli(self, sample_result: OptimizationResult) -> None:
        """Test AWS CLI export (lines 192-205)."""
        exporter = ConfigExporter(sample_result, "aws_glue")
        output = exporter.export_aws_cli()

        assert "aws" in output.lower()
        assert "glue" in output.lower()
        assert "create-job" in output.lower()
