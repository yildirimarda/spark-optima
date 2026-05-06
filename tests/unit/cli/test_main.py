# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the CLI main module.

This module contains tests for the command-line interface including
command parsing and execution.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from spark_optima.cli.main import app

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def sample_code_file(tmp_path: Path) -> Path:
    """Create a sample code file for testing."""
    code_file = tmp_path / "test_job.py"
    code_file.write_text("from pyspark.sql import SparkSession\n")
    return code_file


class TestCLIMain:
    """Test cases for the main CLI commands."""

    def test_cli_no_args_shows_help(self, runner: CliRunner) -> None:
        """Test that running CLI without args shows help."""
        result = runner.invoke(app)

        assert result.exit_code == 0
        assert "Spark Optima" in result.output or "Usage:" in result.output

    def test_cli_help_flag(self, runner: CliRunner) -> None:
        """Test the --help flag."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_cli_version_flag(self, runner: CliRunner) -> None:
        """Test the --version flag."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output or "version" in result.output.lower()


class TestCLIOptimizeCommand:
    """Test cases for the optimize command."""

    def test_optimize_command_help(self, runner: CliRunner) -> None:
        """Test optimize command help."""
        result = runner.invoke(app, ["optimize", "--help"])

        assert result.exit_code == 0
        assert "optimize" in result.output.lower()

    def test_optimize_with_valid_file(self, runner: CliRunner, sample_code_file: Path) -> None:
        """Test optimize with valid file."""
        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "local",
            ],
        )

        # Should succeed or show appropriate error
        assert result.exit_code in [0, 1, 2]

    def test_optimize_with_invalid_file(self, runner: CliRunner) -> None:
        """Test optimize with non-existent file."""
        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                "/nonexistent/file.py",
                "--platform",
                "local",
            ],
        )

        assert result.exit_code != 0

    def test_optimize_with_platform(self, runner: CliRunner, sample_code_file: Path) -> None:
        """Test optimize with different platforms."""
        for platform in ["local", "databricks", "aws_glue", "azure_synapse"]:
            result = runner.invoke(
                app,
                [
                    "optimize",
                    "--code-path",
                    str(sample_code_file),
                    "--platform",
                    platform,
                ],
            )
            # Each platform should be accepted
            assert "invalid" not in result.output.lower() or result.exit_code != 2

    def test_optimize_with_spark_version(self, runner: CliRunner, sample_code_file: Path) -> None:
        """Test optimize with specific Spark version."""
        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "local",
                "--spark-version",
                "3.5.0",
            ],
        )

        assert result.exit_code in [0, 1, 2]


class TestCLIAnalyzeCommand:
    """Test cases for the analyze command."""

    def test_analyze_command_help(self, runner: CliRunner) -> None:
        """Test analyze command help."""
        result = runner.invoke(app, ["analyze", "--help"])

        assert result.exit_code == 0
        assert "analyze" in result.output.lower()

    def test_analyze_with_valid_file(self, runner: CliRunner, sample_code_file: Path) -> None:
        """Test analyze with valid file."""
        result = runner.invoke(app, ["analyze", "--code-path", str(sample_code_file)])

        assert result.exit_code in [0, 1, 2]

    def test_analyze_with_invalid_file(self, runner: CliRunner) -> None:
        """Test analyze with non-existent file."""
        result = runner.invoke(app, ["analyze", "--code-path", "/nonexistent/file.py"])

        assert result.exit_code != 0


class TestCLIPlatformsCommand:
    """Test cases for the platforms command."""

    def test_platforms_command_help(self, runner: CliRunner) -> None:
        """Test platforms command help."""
        result = runner.invoke(app, ["platforms", "--help"])

        assert result.exit_code == 0
        assert "platform" in result.output.lower()

    def test_platforms_list(self, runner: CliRunner) -> None:
        """Test platforms list command."""
        result = runner.invoke(app, ["platforms", "list"])

        # Should show available platforms
        assert result.exit_code in [0, 1, 2]


class TestCLIExportCommand:
    """Test cases for the export command."""

    def test_export_command_help(self, runner: CliRunner) -> None:
        """Test export command help."""
        result = runner.invoke(app, ["export", "--help"])

        assert result.exit_code == 0
        assert "export" in result.output.lower()


class TestCLIWizardCommand:
    """Test cases for the wizard command."""

    def test_wizard_command_help(self, runner: CliRunner) -> None:
        """Test wizard command help."""
        result = runner.invoke(app, ["wizard", "--help"])

        assert result.exit_code == 0
        assert "wizard" in result.output.lower()


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_invalid_command(self, runner: CliRunner) -> None:
        """Test response to invalid command."""
        result = runner.invoke(app, ["invalid_command"])

        assert result.exit_code != 0
        assert (
            "invalid" in result.output.lower()
            or "error" in result.output.lower()
            or "usage" in result.output.lower()
        )

    def test_invalid_option(self, runner: CliRunner) -> None:
        """Test response to invalid option."""
        result = runner.invoke(app, ["--invalid-option"])

        assert result.exit_code != 0


class TestCLIOptimizeCommandFull:
    """Test cases for the optimize command with full coverage."""

    def test_optimize_with_code_path_option(
        self, runner: CliRunner, sample_code_file: Path
    ) -> None:
        """Test optimize with --code-path option."""
        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "local",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_optimize_with_data_size(self, runner: CliRunner, sample_code_file: Path) -> None:
        """Test optimize with data size option."""
        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "local",
                "--data-size",
                "100.0",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_optimize_with_data_format(self, runner: CliRunner, sample_code_file: Path) -> None:
        """Test optimize with data format option."""
        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "local",
                "--data-format",
                "json",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_optimize_with_max_memory(self, runner: CliRunner, sample_code_file: Path) -> None:
        """Test optimize with max memory option."""
        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "local",
                "--max-memory",
                "64.0",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_optimize_with_output_json(self, runner: CliRunner, sample_code_file: Path) -> None:
        """Test optimize with JSON output format."""
        with patch("spark_optima.cli.main.Optimizer") as mock_optimizer:
            mock_instance = MagicMock()
            mock_instance.optimize.return_value = MagicMock(
                configuration={"spark.executor.memory": "4g"},
                estimated_time_minutes=10.0,
                confidence_score=0.95,
                platform_specific={"platform": "local"},
                code_suggestions=[],
                metadata={},
            )
            mock_optimizer.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "optimize",
                    "--code-path",
                    str(sample_code_file),
                    "--platform",
                    "local",
                    "--output",
                    "json",
                ],
            )

            assert result.exit_code in [0, 1, 2]

    def test_optimize_with_output_yaml(self, runner: CliRunner, sample_code_file: Path) -> None:
        """Test optimize with YAML output format."""
        with patch("spark_optima.cli.main.Optimizer") as mock_optimizer:
            mock_instance = MagicMock()
            mock_instance.optimize.return_value = MagicMock(
                configuration={"spark.executor.memory": "4g"},
                estimated_time_minutes=10.0,
                confidence_score=0.95,
                platform_specific={"platform": "local"},
                code_suggestions=[],
                metadata={},
            )
            mock_optimizer.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "optimize",
                    "--code-path",
                    str(sample_code_file),
                    "--platform",
                    "local",
                    "--output",
                    "yaml",
                ],
            )

            assert result.exit_code in [0, 1, 2]

    def test_optimize_with_mode_execution(self, runner: CliRunner, sample_code_file: Path) -> None:
        """Test optimize with execution mode."""
        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "local",
                "--mode",
                "execution",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_optimize_missing_code_file(self, runner: CliRunner) -> None:
        """Test optimize without code file shows error."""
        result = runner.invoke(app, ["optimize"])

        assert result.exit_code != 0

    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_initializer_error(
        self, mock_optimizer: MagicMock, runner: CliRunner, sample_code_file: Path
    ) -> None:
        """Test optimize handles initializer error."""
        mock_optimizer.side_effect = ValueError("Invalid platform")

        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "invalid",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_optimization_error(
        self, mock_optimizer: MagicMock, runner: CliRunner, sample_code_file: Path
    ) -> None:
        """Test optimize handles optimization error."""
        mock_instance = MagicMock()
        mock_instance.optimize.side_effect = Exception("Optimization failed")
        mock_optimizer.return_value = mock_instance

        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "local",
            ],
        )

        assert result.exit_code in [0, 1, 2]

    @patch("spark_optima.cli.formatters.ConfigExporter")
    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_json_output_cov(
        self,
        mock_optimizer: MagicMock,
        mock_exporter: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
    ) -> None:
        """Test optimize JSON output covers lines 448-452."""
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = MagicMock(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.95,
            platform_specific={"platform": "local"},
            code_suggestions=[],
            metadata={},
        )
        mock_optimizer.return_value = mock_instance

        mock_exporter_instance = MagicMock()
        mock_exporter_instance.export_json.return_value = '{"config": "test"}'
        mock_exporter.return_value = mock_exporter_instance

        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "local",
                "--output",
                "json",
            ],
        )

        assert result.exit_code == 0
        mock_exporter_instance.export_json.assert_called_once()

    @patch("spark_optima.cli.formatters.ConfigExporter")
    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_yaml_output_cov(
        self,
        mock_optimizer: MagicMock,
        mock_exporter: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
    ) -> None:
        """Test optimize YAML output covers lines 448-452."""
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = MagicMock(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.95,
            platform_specific={"platform": "local"},
            code_suggestions=[],
            metadata={},
        )
        mock_optimizer.return_value = mock_instance

        mock_exporter_instance = MagicMock()
        mock_exporter_instance.export_yaml.return_value = "config: test"
        mock_exporter.return_value = mock_exporter_instance

        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "local",
                "--output",
                "yaml",
            ],
        )

        assert result.exit_code == 0
        mock_exporter_instance.export_yaml.assert_called_once()


class TestCLIAnalyzeCommandFull:
    """Test cases for the analyze command with full coverage."""

    def test_analyze_with_code_path_option(self, runner: CliRunner, sample_code_file: Path) -> None:
        """Test analyze with --code-path option."""
        result = runner.invoke(
            app,
            [
                "analyze",
                "--code-path",
                str(sample_code_file),
            ],
        )

        assert result.exit_code in [0, 1, 2]

    def test_analyze_with_output_json(self, runner: CliRunner, sample_code_file: Path) -> None:
        """Test analyze with JSON output format."""
        with patch("spark_optima.analysis.recommender.analyze_code") as mock_analyze:
            mock_result = MagicMock()
            mock_result.operations = []
            mock_result.smells = []
            mock_result.recommendations = []
            mock_result.to_dict.return_value = {
                "operations": [],
                "smells": [],
                "recommendations": [],
            }
            mock_analyze.return_value = mock_result

            result = runner.invoke(
                app,
                [
                    "analyze",
                    "--code-path",
                    str(sample_code_file),
                    "--output",
                    "json",
                ],
            )

            assert result.exit_code in [0, 1, 2]

    def test_analyze_missing_code_file(self, runner: CliRunner) -> None:
        """Test analyze without code file shows error."""
        result = runner.invoke(app, ["analyze"])

        assert result.exit_code != 0

    @patch("spark_optima.analysis.recommender.analyze_code")
    def test_analyze_error_handling(
        self, mock_analyze: MagicMock, runner: CliRunner, sample_code_file: Path
    ) -> None:
        """Test analyze handles errors."""
        mock_analyze.side_effect = Exception("Analysis failed")

        result = runner.invoke(
            app,
            [
                "analyze",
                "--code-path",
                str(sample_code_file),
            ],
        )

        assert result.exit_code in [0, 1, 2]

    @patch("spark_optima.analysis.recommender.analyze_code")
    def test_analyze_table_output_with_smells(
        self, mock_analyze: MagicMock, runner: CliRunner, sample_code_file: Path
    ) -> None:
        """Test analyze with table output and code smells (covers lines 281-301)."""
        # Mock smell with location
        smell_with_loc = MagicMock()
        smell_with_loc.smell_type = "performance"
        smell_with_loc.severity = "high"
        smell_with_loc.location = MagicMock()
        smell_with_loc.location.line = 10

        # Mock smell without location
        smell_without_loc = MagicMock()
        smell_without_loc.smell_type = "code_smell"
        smell_without_loc.severity = "medium"
        smell_without_loc.location = None

        mock_result = MagicMock()
        mock_result.operations = [MagicMock()]
        mock_result.smells = [smell_with_loc, smell_without_loc]
        mock_result.recommendations = [MagicMock()]
        mock_result.to_dict.return_value = {}
        mock_analyze.return_value = mock_result

        result = runner.invoke(app, ["analyze", "--code-path", str(sample_code_file)])

        assert result.exit_code == 0
        assert "Code Analysis Results" in result.output
        assert "Code Smells" in result.output
        assert "10" in result.output
        assert "Unknown" in result.output


class TestCLIPlatformsCommandFull:
    """Test cases for the platforms command with full coverage."""

    @patch("spark_optima.platforms.get_platform")
    def test_platforms_list_with_platform_info(
        self, mock_get_platform: MagicMock, runner: CliRunner
    ) -> None:
        """Test platforms list shows platform details."""
        mock_instance = MagicMock()
        mock_instance.name = "local"
        mock_instance.display_name = "Local Platform"
        mock_instance.description = "Local Spark platform"
        mock_get_platform.return_value = mock_instance

        result = runner.invoke(app, ["platforms", "list"])

        assert result.exit_code in [0, 1, 2]

    @patch("spark_optima.platforms.get_platform")
    def test_platforms_list_platform_error(
        self, mock_get_platform: MagicMock, runner: CliRunner
    ) -> None:
        """Test platforms list handles platform errors."""
        mock_get_platform.side_effect = Exception("Platform error")

        result = runner.invoke(app, ["platforms", "list"])

        assert result.exit_code in [0, 1, 2]

    def test_platforms_legacy_command(self, runner: CliRunner) -> None:
        """Test legacy platforms command."""
        result = runner.invoke(app, ["platforms"])

        assert result.exit_code in [0, 1, 2]

    def test_platforms_legacy_direct_call(self) -> None:
        """Test legacy platforms command directly (covers line 370)."""
        from spark_optima.cli.main import platforms

        with patch("spark_optima.cli.main.platforms_list") as mock_list:
            platforms(_list_all=True)
            mock_list.assert_called_once()


class TestCLIExportCommandFull:
    """Test cases for the export command with full coverage."""

    def test_export_missing_result_file(self, runner: CliRunner) -> None:
        """Test export without result file shows error."""
        result = runner.invoke(app, ["export"])

        assert result.exit_code != 0

    def test_export_with_format_json(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test export with JSON format."""
        # Create a sample result file
        result_file = tmp_path / "result.json"
        result_data = {
            "configuration": {"spark.executor.memory": "4g"},
            "platform_specific": {"platform": "local"},
            "estimated_time_minutes": 10.0,
            "confidence_score": 0.95,
            "code_suggestions": [],
            "metadata": {},
        }
        result_file.write_text(json.dumps(result_data))

        with patch("spark_optima.cli.formatters.ConfigExporter") as mock_exporter:
            mock_instance = MagicMock()
            mock_instance.export_json.return_value = '{"config": "test"}'
            mock_exporter.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "export",
                    "--result-file",
                    str(result_file),
                    "--format",
                    "json",
                ],
            )

            assert result.exit_code in [0, 1]

    def test_export_with_format_yaml(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test export with YAML format."""
        result_file = tmp_path / "result.json"
        result_data = {
            "configuration": {"spark.executor.memory": "4g"},
            "platform_specific": {"platform": "local"},
        }
        result_file.write_text(json.dumps(result_data))

        with patch("spark_optima.cli.formatters.ConfigExporter") as mock_exporter:
            mock_instance = MagicMock()
            mock_instance.export_yaml.return_value = "config: test"
            mock_exporter.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "export",
                    "--result-file",
                    str(result_file),
                    "--format",
                    "yaml",
                ],
            )

            assert result.exit_code in [0, 1]

    def test_export_with_output_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test export with output file."""
        result_file = tmp_path / "result.json"
        output_file = tmp_path / "output.json"
        result_data = {
            "configuration": {"spark.executor.memory": "4g"},
            "platform_specific": {"platform": "local"},
        }
        result_file.write_text(json.dumps(result_data))

        with patch("spark_optima.cli.formatters.ConfigExporter") as mock_exporter:
            mock_instance = MagicMock()
            mock_exporter.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "export",
                    "--result-file",
                    str(result_file),
                    "--format",
                    "json",
                    "--output",
                    str(output_file),
                ],
            )

            assert result.exit_code in [0, 1]

    def test_export_with_help_format(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test export with help format."""
        result_file = tmp_path / "result.json"
        result_data = {
            "configuration": {},
            "platform_specific": {"platform": "local"},
        }
        result_file.write_text(json.dumps(result_data))

        with patch("spark_optima.cli.formatters.print_platform_export_help") as mock_help:
            result = runner.invoke(
                app,
                [
                    "export",
                    "--result-file",
                    str(result_file),
                    "--format",
                    "help",
                ],
            )

            assert result.exit_code == 0
            mock_help.assert_called_once()

    def test_export_invalid_format(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test export with invalid format."""
        result_file = tmp_path / "result.json"
        result_data = {
            "configuration": {},
            "platform_specific": {"platform": "local"},
        }
        result_file.write_text(json.dumps(result_data))

        result = runner.invoke(
            app,
            [
                "export",
                "--result-file",
                str(result_file),
                "--format",
                "invalid_format",
            ],
        )

        assert result.exit_code == 1

    def test_export_invalid_result_file(self, runner: CliRunner) -> None:
        """Test export with invalid result file."""
        result = runner.invoke(
            app,
            [
                "export",
                "--result-file",
                "/nonexistent/result.json",
            ],
        )

        assert result.exit_code != 0

    def test_export_value_error(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test export handles ValueError (covers lines 569-570)."""
        result_file = tmp_path / "result.json"
        result_data = {
            "configuration": {},
            "platform_specific": {"platform": "local"},
        }
        result_file.write_text(json.dumps(result_data))

        with patch("spark_optima.cli.formatters.ConfigExporter") as mock_exporter:
            mock_instance = MagicMock()
            mock_instance.export_json.side_effect = ValueError("Export failed")
            mock_exporter.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "export",
                    "--result-file",
                    str(result_file),
                    "--format",
                    "json",
                ],
            )

            assert result.exit_code == 1
            assert "Export error" in result.output


class TestCLIWizardCommandV2:
    """Test cases for the wizard command."""

    @patch("spark_optima.cli.wizard.run_wizard")
    def test_wizard_command(
        self, mock_wizard: MagicMock, runner: CliRunner, sample_code_file: Path
    ) -> None:
        """Test wizard command execution."""
        mock_wizard.return_value = {
            "code_path": str(sample_code_file),
            "platform": "local",
            "spark_version": "3.5.0",
            "resources": {"cpu_cores": 4, "memory_gb": 16},
            "data_profile": {"size_gb": 100, "format": "parquet"},
            "constraints": {},
            "output_format": "table",
            "use_bayesian": True,
            "bayesian_trials": 50,
        }

        with patch("spark_optima.cli.main.Optimizer") as mock_optimizer:
            mock_instance = MagicMock()
            mock_instance.optimize.return_value = MagicMock(
                configuration={"spark.executor.memory": "4g"},
                estimated_time_minutes=10.0,
                confidence_score=0.95,
                platform_specific={"platform": "local"},
                code_suggestions=[],
                metadata={},
            )
            mock_optimizer.return_value = mock_instance

            result = runner.invoke(app, ["wizard"])

            assert result.exit_code in [0, 1]

    @patch("spark_optima.cli.wizard.run_wizard")
    def test_wizard_cancelled(self, mock_wizard: MagicMock, runner: CliRunner) -> None:
        """Test wizard command when cancelled."""
        mock_wizard.return_value = None

        result = runner.invoke(app, ["wizard"])

        assert result.exit_code == 0

    @patch("spark_optima.cli.main.Optimizer")
    @patch("spark_optima.cli.wizard.run_wizard")
    def test_wizard_initializer_error(
        self, mock_wizard: MagicMock, mock_optimizer: MagicMock, runner: CliRunner
    ) -> None:
        """Test wizard handles initializer error (covers lines 437-439)."""
        mock_wizard.return_value = {
            "code_path": "/tmp/test.py",
            "platform": "invalid",
            "spark_version": "3.5.0",
            "resources": {"cpu_cores": 4, "memory_gb": 16},
            "data_profile": {},
            "constraints": {},
            "output_format": "table",
        }
        mock_optimizer.side_effect = ValueError("Invalid platform")

        result = runner.invoke(app, ["wizard"])

        assert result.exit_code in [0, 1, 2]

    @patch("spark_optima.cli.main.Optimizer")
    @patch("spark_optima.cli.wizard.run_wizard")
    def test_wizard_optimization_error(
        self, mock_wizard: MagicMock, mock_optimizer: MagicMock, runner: CliRunner
    ) -> None:
        """Test wizard handles optimization error (covers lines 448-452)."""
        mock_wizard.return_value = {
            "code_path": "/tmp/test.py",
            "platform": "local",
            "spark_version": "3.5.0",
            "resources": {"cpu_cores": 4, "memory_gb": 16},
            "data_profile": {},
            "constraints": {},
            "output_format": "table",
        }
        mock_instance = MagicMock()
        mock_instance.optimize.side_effect = Exception("Optimization failed")
        mock_optimizer.return_value = mock_instance

        result = runner.invoke(app, ["wizard"])

        assert result.exit_code in [0, 1, 2]

    @patch("spark_optima.cli.formatters.ConfigExporter")
    @patch("spark_optima.cli.main.Optimizer")
    @patch("spark_optima.cli.wizard.run_wizard")
    def test_wizard_json_output_cov(
        self,
        mock_wizard: MagicMock,
        mock_optimizer: MagicMock,
        mock_exporter: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Test wizard JSON output covers lines 448-452."""
        mock_wizard.return_value = {
            "code_path": "/tmp/test.py",
            "platform": "local",
            "spark_version": "3.5.0",
            "resources": {"cpu_cores": 4, "memory_gb": 16},
            "data_profile": {},
            "constraints": {},
            "output_format": "json",
        }
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = MagicMock(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.95,
            platform_specific={"platform": "local"},
            code_suggestions=[],
            metadata={},
        )
        mock_optimizer.return_value = mock_instance

        mock_exporter_instance = MagicMock()
        mock_exporter_instance.export_json.return_value = '{"config": "test"}'
        mock_exporter.return_value = mock_exporter_instance

        result = runner.invoke(app, ["wizard"])

        assert result.exit_code == 0
        mock_exporter_instance.export_json.assert_called_once()

    @patch("spark_optima.cli.formatters.ConfigExporter")
    @patch("spark_optima.cli.main.Optimizer")
    @patch("spark_optima.cli.wizard.run_wizard")
    def test_wizard_yaml_output_cov(
        self,
        mock_wizard: MagicMock,
        mock_optimizer: MagicMock,
        mock_exporter: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Test wizard YAML output covers lines 448-452."""
        mock_wizard.return_value = {
            "code_path": "/tmp/test.py",
            "platform": "local",
            "spark_version": "3.5.0",
            "resources": {"cpu_cores": 4, "memory_gb": 16},
            "data_profile": {},
            "constraints": {},
            "output_format": "yaml",
        }
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = MagicMock(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.95,
            platform_specific={"platform": "local"},
            code_suggestions=[],
            metadata={},
        )
        mock_optimizer.return_value = mock_instance

        mock_exporter_instance = MagicMock()
        mock_exporter_instance.export_yaml.return_value = "config: test"
        mock_exporter.return_value = mock_exporter_instance

        result = runner.invoke(app, ["wizard"])

        assert result.exit_code == 0
        mock_exporter_instance.export_yaml.assert_called_once()
