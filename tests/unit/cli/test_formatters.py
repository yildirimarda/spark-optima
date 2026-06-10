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

    def test_save_to_file_unsupported_format(self, sample_result: OptimizationResult, tmp_path: Path) -> None:
        """Test saving with unsupported format raises error."""
        exporter = ConfigExporter(sample_result, "local")
        output_file = tmp_path / "config.txt"

        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.save_to_file(output_file, "unsupported")

    def test_export_airflow_dag_local_uses_spark_submit_operator(self, sample_result: OptimizationResult) -> None:
        """Test Airflow export uses SparkSubmitOperator for local platform."""
        exporter = ConfigExporter(sample_result, "local")
        output = exporter.export_airflow_dag()

        assert "from airflow import DAG" in output
        assert "SparkSubmitOperator" in output
        assert "conf=SPARK_CONF" in output
        assert '"spark.executor.memory": "4g"' in output
        assert "dag_id=" in output
        assert "TODO" in output
        # Other platforms' operators must not leak in
        assert "DatabricksSubmitRunOperator" not in output
        assert "GlueJobOperator" not in output

    def test_export_airflow_dag_databricks_operator(self, sample_result: OptimizationResult) -> None:
        """Test Airflow export uses DatabricksSubmitRunOperator for Databricks."""
        exporter = ConfigExporter(sample_result, "databricks")
        output = exporter.export_airflow_dag()

        assert "DatabricksSubmitRunOperator" in output
        assert "new_cluster=" in output
        assert '"spark_conf": SPARK_CONF' in output
        assert "SparkSubmitOperator" not in output

    def test_export_airflow_dag_glue_operator(self, sample_result: OptimizationResult) -> None:
        """Test Airflow export uses GlueJobOperator with --conf DefaultArguments."""
        exporter = ConfigExporter(sample_result, "aws_glue")
        output = exporter.export_airflow_dag()

        assert "GlueJobOperator" in output
        assert "DEFAULT_ARGUMENTS" in output
        assert '"--conf"' in output
        assert "spark.executor.memory=4g" in output
        assert "SparkSubmitOperator" not in output

    def test_export_kubernetes_configmap(self, sample_result: OptimizationResult) -> None:
        """Test Kubernetes ConfigMap export structure and contents."""
        import yaml

        exporter = ConfigExporter(sample_result, "local")
        output = exporter.export_kubernetes_configmap()

        assert "apiVersion: v1" in output
        assert "kind: ConfigMap" in output
        assert "name: spark-optima-config" in output

        # Must be valid YAML with spark-defaults.conf style data
        manifest = yaml.safe_load(output)
        assert manifest["kind"] == "ConfigMap"
        assert manifest["metadata"]["name"] == "spark-optima-config"
        defaults_conf = manifest["data"]["spark-defaults.conf"]
        assert "spark.executor.memory 4g" in defaults_conf
        assert "spark.sql.shuffle.partitions 200" in defaults_conf

    def test_export_kubernetes_configmap_sorted_keys(self, sample_result: OptimizationResult) -> None:
        """Test ConfigMap output renders config keys deterministically sorted."""
        exporter = ConfigExporter(sample_result, "local")
        output = exporter.export_kubernetes_configmap()

        assert output.index("spark.driver.memory") < output.index("spark.executor.cores")
        assert output.index("spark.executor.cores") < output.index("spark.executor.memory")
        assert output.index("spark.executor.memory") < output.index("spark.sql.shuffle.partitions")

    def test_export_aws_emr(self, sample_result: OptimizationResult) -> None:
        """Test AWS EMR configurations JSON export."""
        exporter = ConfigExporter(sample_result, "local")
        output = exporter.export_aws_emr()

        data = json.loads(output)
        assert isinstance(data, list)
        assert data[0]["Classification"] == "spark-defaults"
        properties = data[0]["Properties"]
        assert properties["spark.executor.memory"] == "4g"
        assert properties["spark.sql.shuffle.partitions"] == "200"
        # All values must be strings for the EMR API
        assert all(isinstance(v, str) for v in properties.values())

    @pytest.mark.parametrize(
        ("format_type", "marker"),
        [
            ("airflow", "with DAG("),
            ("kubernetes", "kind: ConfigMap"),
            ("emr", '"Classification": "spark-defaults"'),
        ],
    )
    def test_save_to_file_new_formats(
        self,
        sample_result: OptimizationResult,
        tmp_path: Path,
        format_type: str,
        marker: str,
    ) -> None:
        """Test save_to_file routing for the new export formats."""
        exporter = ConfigExporter(sample_result, "local")
        output_file = tmp_path / f"config.{format_type}"

        exporter.save_to_file(output_file, format_type)

        assert output_file.exists()
        assert marker in output_file.read_text()

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

    def test_display_results_table(self, sample_result: OptimizationResult, capsys: pytest.CaptureFixture) -> None:
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


class TestParetoExports:
    """Tests for Pareto frontier export formats (Workstream N)."""

    @pytest.fixture
    def pareto_result(self) -> OptimizationResult:
        """Create a multi-objective result with a 3-point Pareto frontier."""
        return OptimizationResult(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.9,
            platform_specific={"platform": "local"},
            metadata={
                "objectives": ["minimize_time", "minimize_cost"],
                "pareto_frontier": [
                    {
                        "trial_number": 7,
                        "objective_values": {"minimize_time": 150.0, "minimize_cost": 0.20},
                        "configuration": {"spark.executor.memory": "2g"},
                    },
                    {
                        "trial_number": 0,
                        "objective_values": {"minimize_time": 120.0, "minimize_cost": 0.30},
                        "configuration": {"spark.executor.memory": "4g", "spark.executor.cores": 4},
                    },
                    {
                        "trial_number": 3,
                        "objective_values": {"minimize_time": 90.0, "minimize_cost": 0.45},
                        "configuration": {"spark.executor.memory": "8g", "spark.executor.cores": 4},
                    },
                ],
            },
        )

    @pytest.fixture
    def single_objective_result(self) -> OptimizationResult:
        """Create a single-objective result (no Pareto frontier)."""
        return OptimizationResult(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.9,
            platform_specific={"platform": "local"},
            metadata={"platform": "local"},
        )

    def test_export_pareto_json(self, pareto_result: OptimizationResult) -> None:
        """JSON export contains objectives, point count, and all points."""
        exporter = ConfigExporter(pareto_result, "local")
        payload = json.loads(exporter.export_pareto_json())

        assert payload["objectives"] == ["minimize_time", "minimize_cost"]
        assert payload["n_points"] == 3
        assert len(payload["points"]) == 3
        assert payload["points"][0]["trial_number"] == 7
        assert payload["points"][0]["objective_values"]["minimize_cost"] == pytest.approx(0.20)

    def test_export_pareto_json_falls_back_to_sorted_objectives(self) -> None:
        """Without metadata objectives, names come from the sorted union of point keys."""
        result = OptimizationResult(
            configuration={},
            estimated_time_minutes=1.0,
            confidence_score=0.5,
            metadata={
                "pareto_frontier": [
                    {
                        "trial_number": 1,
                        "objective_values": {"minimize_time": 5.0, "minimize_cost": 0.1},
                        "configuration": {"spark.executor.memory": "4g"},
                    },
                ],
            },
        )
        exporter = ConfigExporter(result, "local")
        payload = json.loads(exporter.export_pareto_json())

        assert payload["objectives"] == ["minimize_cost", "minimize_time"]

    def test_export_pareto_csv_deterministic_columns(self, pareto_result: OptimizationResult) -> None:
        """CSV header is trial, objectives (metadata order), then sorted config params."""
        exporter = ConfigExporter(pareto_result, "local")
        lines = exporter.export_pareto_csv().strip().split("\n")

        assert lines[0] == "trial,minimize_time,minimize_cost,spark.executor.cores,spark.executor.memory"
        # Rows are sorted by trial number
        assert lines[1] == "0,120.0,0.3,4,4g"
        assert lines[2] == "3,90.0,0.45,4,8g"
        # Missing config params produce empty cells (trial 7 has no cores value)
        assert lines[3] == "7,150.0,0.2,,2g"

    def test_export_pareto_json_raises_without_frontier(
        self,
        single_objective_result: OptimizationResult,
    ) -> None:
        """Single-objective results raise a helpful ValueError on pareto export."""
        exporter = ConfigExporter(single_objective_result, "local")

        with pytest.raises(ValueError, match="no Pareto frontier"):
            exporter.export_pareto_json()
        with pytest.raises(ValueError, match="--objective"):
            exporter.export_pareto_csv()

    def test_save_to_file_routes_pareto_formats(
        self,
        pareto_result: OptimizationResult,
        tmp_path: Path,
    ) -> None:
        """save_to_file dispatches pareto-json and pareto-csv formats."""
        exporter = ConfigExporter(pareto_result, "local")

        json_path = tmp_path / "frontier.json"
        exporter.save_to_file(json_path, "pareto-json")
        payload = json.loads(json_path.read_text())
        assert payload["n_points"] == 3

        csv_path = tmp_path / "frontier.csv"
        exporter.save_to_file(csv_path, "pareto-csv")
        assert csv_path.read_text().startswith("trial,minimize_time,minimize_cost")
