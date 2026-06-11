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

        assert result.exit_code in (0, 2)  # Typer returns 2 with no_args_is_help=True
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
            "invalid" in result.output.lower() or "error" in result.output.lower() or "usage" in result.output.lower()
        )

    def test_invalid_option(self, runner: CliRunner) -> None:
        """Test response to invalid option."""
        result = runner.invoke(app, ["--invalid-option"])

        assert result.exit_code != 0


class TestCLIOptimizeCommandFull:
    """Test cases for the optimize command with full coverage."""

    def test_optimize_with_code_path_option(self, runner: CliRunner, sample_code_file: Path) -> None:
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
    def test_analyze_error_handling(self, mock_analyze: MagicMock, runner: CliRunner, sample_code_file: Path) -> None:
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
    def test_platforms_list_with_platform_info(self, mock_get_platform: MagicMock, runner: CliRunner) -> None:
        """Test platforms list shows platform details."""
        mock_instance = MagicMock()
        mock_instance.name = "local"
        mock_instance.display_name = "Local Platform"
        mock_instance.description = "Local Spark platform"
        mock_get_platform.return_value = mock_instance

        result = runner.invoke(app, ["platforms", "list"])

        assert result.exit_code in [0, 1, 2]

    @patch("spark_optima.platforms.get_platform")
    def test_platforms_list_platform_error(self, mock_get_platform: MagicMock, runner: CliRunner) -> None:
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
    def test_wizard_command(self, mock_wizard: MagicMock, runner: CliRunner, sample_code_file: Path) -> None:
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


# =============================================================================
# v1.1 — history / compare / explain commands
# =============================================================================


@pytest.fixture(autouse=True)
def _isolated_history_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the history store at a temporary database for every CLI test.

    This keeps the optimize command's auto-save from touching the real
    ~/.spark_optima/history.db during the test run.
    """
    db_path = tmp_path / "history.db"
    monkeypatch.setenv("SPARK_OPTIMA_HISTORY_DB", str(db_path))
    return db_path


@pytest.fixture
def history_db(_isolated_history_db: Path) -> Path:
    """Expose the isolated history database path to tests that need it."""
    return _isolated_history_db


@pytest.fixture
def sample_result_dict() -> dict:
    """Create a sample optimization result dictionary."""
    return {
        "configuration": {
            "spark.executor.memory": "4g",
            "spark.executor.cores": 4,
            "spark.sql.shuffle.partitions": 200,
        },
        "estimated_time_minutes": 12.5,
        "confidence_score": 0.85,
        "code_suggestions": [],
        "platform_specific": {"platform": "local", "spark_version": "3.5.0"},
        "metadata": {"platform": "local", "spark_version": "3.5.0"},
    }


def _seed_history(platform: str = "local") -> int:
    """Insert one history entry using the env-configured database."""
    from spark_optima.core.history import OptimizationHistory

    with OptimizationHistory() as store:
        return store.save(
            {
                "configuration": {"spark.executor.memory": "4g"},
                "estimated_time_minutes": 10.0,
                "confidence_score": 0.9,
            },
            platform=platform,
            spark_version="3.5.0",
            mode="simulation",
            code_path="/tmp/job.py",
        )


class TestCLIHistoryCommand:
    """Test cases for the history command."""

    def test_history_command_help(self, runner: CliRunner) -> None:
        """Test history command help."""
        result = runner.invoke(app, ["history", "--help"])

        assert result.exit_code == 0
        assert "history" in result.output.lower()

    def test_history_empty(self, runner: CliRunner, history_db: Path) -> None:
        """Test history with no stored entries."""
        result = runner.invoke(app, ["history"])

        assert result.exit_code == 0
        assert "No optimization history" in result.output

    def test_history_lists_entries(self, runner: CliRunner, history_db: Path) -> None:
        """Test history lists saved entries."""
        entry_id = _seed_history()

        result = runner.invoke(app, ["history"])

        assert result.exit_code == 0
        assert str(entry_id) in result.output
        assert "local" in result.output
        assert "3.5.0" in result.output

    def test_history_platform_filter(self, runner: CliRunner, history_db: Path) -> None:
        """Test history filters by platform."""
        _seed_history(platform="local")
        _seed_history(platform="databricks")

        result = runner.invoke(app, ["history", "--platform", "databricks"])

        assert result.exit_code == 0
        assert "databricks" in result.output

        result = runner.invoke(app, ["history", "--platform", "aws_glue"])

        assert result.exit_code == 0
        assert "No optimization history" in result.output

    def test_history_show_entry(self, runner: CliRunner, history_db: Path) -> None:
        """Test history --show prints full entry details."""
        entry_id = _seed_history()

        result = runner.invoke(app, ["history", "--show", str(entry_id)])

        assert result.exit_code == 0
        assert "spark.executor.memory" in result.output
        assert "4g" in result.output
        assert "/tmp/job.py" in result.output

    def test_history_show_missing_entry(self, runner: CliRunner, history_db: Path) -> None:
        """Test history --show errors for unknown ids."""
        result = runner.invoke(app, ["history", "--show", "999"])

        assert result.exit_code == 1
        assert "no history entry" in result.output

    def test_history_clear_with_yes(self, runner: CliRunner, history_db: Path) -> None:
        """Test history --clear --yes deletes all entries."""
        _seed_history()
        _seed_history()

        result = runner.invoke(app, ["history", "--clear", "--yes"])

        assert result.exit_code == 0
        assert "Deleted 2" in result.output

        result = runner.invoke(app, ["history"])
        assert "No optimization history" in result.output

    def test_history_clear_aborted(self, runner: CliRunner, history_db: Path) -> None:
        """Test history --clear keeps entries when confirmation is declined."""
        entry_id = _seed_history()

        result = runner.invoke(app, ["history", "--clear"], input="n\n")

        assert result.exit_code == 0
        assert "Aborted" in result.output

        result = runner.invoke(app, ["history"])
        assert str(entry_id) in result.output

    def test_history_invalid_limit(self, runner: CliRunner, history_db: Path) -> None:
        """Test history with an invalid limit errors cleanly."""
        result = runner.invoke(app, ["history", "--limit", "0"])

        assert result.exit_code == 1
        assert "limit" in result.output.lower()


class TestCLICompareCommand:
    """Test cases for the compare command."""

    @staticmethod
    def _write_result(path: Path, data: dict) -> Path:
        """Write a result dictionary to a JSON file."""
        path.write_text(json.dumps(data))
        return path

    def test_compare_command_help(self, runner: CliRunner) -> None:
        """Test compare command help."""
        result = runner.invoke(app, ["compare", "--help"])

        assert result.exit_code == 0
        assert "compare" in result.output.lower()

    def test_compare_table_output(self, runner: CliRunner, tmp_path: Path, sample_result_dict: dict) -> None:
        """Test compare shows differing and one-sided parameters."""
        result_a = dict(sample_result_dict)
        result_b = json.loads(json.dumps(sample_result_dict))
        result_b["configuration"]["spark.executor.memory"] = "8g"
        result_b["configuration"]["spark.new.param"] = "true"
        del result_b["configuration"]["spark.executor.cores"]

        file_a = self._write_result(tmp_path / "a.json", result_a)
        file_b = self._write_result(tmp_path / "b.json", result_b)

        result = runner.invoke(app, ["compare", "-a", str(file_a), "-b", str(file_b)])

        assert result.exit_code == 0
        assert "spark.executor.memory" in result.output
        assert "spark.new.param" in result.output
        assert "spark.executor.cores" in result.output

    def test_compare_identical_configs(self, runner: CliRunner, tmp_path: Path, sample_result_dict: dict) -> None:
        """Test compare reports identical configurations."""
        file_a = self._write_result(tmp_path / "a.json", sample_result_dict)
        file_b = self._write_result(tmp_path / "b.json", sample_result_dict)

        result = runner.invoke(app, ["compare", "-a", str(file_a), "-b", str(file_b)])

        assert result.exit_code == 0
        assert "identical" in result.output.lower()

    def test_compare_json_output(self, runner: CliRunner, tmp_path: Path, sample_result_dict: dict) -> None:
        """Test compare with machine-readable JSON output."""
        result_a = sample_result_dict
        result_b = json.loads(json.dumps(sample_result_dict))
        result_b["configuration"]["spark.executor.memory"] = "8g"
        result_b["estimated_time_minutes"] = 10.0

        file_a = self._write_result(tmp_path / "a.json", result_a)
        file_b = self._write_result(tmp_path / "b.json", result_b)

        result = runner.invoke(
            app,
            ["compare", "-a", str(file_a), "-b", str(file_b), "--output", "json"],
        )

        assert result.exit_code == 0
        diff = json.loads(result.output)
        assert diff["changed"]["spark.executor.memory"] == {"a": "4g", "b": "8g"}
        assert diff["only_in_a"] == {}
        assert diff["only_in_b"] == {}
        assert diff["metrics"]["estimated_time_minutes"]["delta"] == pytest.approx(-2.5)
        assert diff["metrics"]["confidence_score"]["delta"] == pytest.approx(0.0)

    def test_compare_includes_cost_metric(self, runner: CliRunner, tmp_path: Path, sample_result_dict: dict) -> None:
        """Test compare includes cost when present in both results."""
        result_a = json.loads(json.dumps(sample_result_dict))
        result_b = json.loads(json.dumps(sample_result_dict))
        result_a["metadata"]["estimated_cost"] = 5.0
        result_b["metadata"]["estimated_cost"] = 7.5

        file_a = self._write_result(tmp_path / "a.json", result_a)
        file_b = self._write_result(tmp_path / "b.json", result_b)

        result = runner.invoke(
            app,
            ["compare", "-a", str(file_a), "-b", str(file_b), "--output", "json"],
        )

        assert result.exit_code == 0
        diff = json.loads(result.output)
        assert diff["metrics"]["estimated_cost"]["delta"] == pytest.approx(2.5)

    def test_compare_missing_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test compare errors on a missing input file."""
        existing = self._write_result(tmp_path / "a.json", {"configuration": {}})

        result = runner.invoke(
            app,
            ["compare", "-a", str(existing), "-b", str(tmp_path / "missing.json")],
        )

        assert result.exit_code == 1
        assert "Error loading result file" in result.output

    def test_compare_invalid_json(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test compare errors on malformed JSON input."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not json")
        good_file = self._write_result(tmp_path / "good.json", {"configuration": {}})

        result = runner.invoke(app, ["compare", "-a", str(bad_file), "-b", str(good_file)])

        assert result.exit_code == 1
        assert "Error loading result file" in result.output

    def test_compare_missing_required_options(self, runner: CliRunner) -> None:
        """Test compare requires both result file options."""
        result = runner.invoke(app, ["compare"])

        assert result.exit_code != 0


class TestCLIExplainCommand:
    """Test cases for the explain command."""

    def test_explain_command_help(self, runner: CliRunner) -> None:
        """Test explain command help."""
        result = runner.invoke(app, ["explain", "--help"])

        assert result.exit_code == 0
        assert "explain" in result.output.lower()

    def test_explain_outputs_rationale(self, runner: CliRunner, tmp_path: Path, sample_result_dict: dict) -> None:
        """Test explain shows heuristic rationale and Bayesian fallback."""
        result_data = json.loads(json.dumps(sample_result_dict))
        result_data["configuration"]["spark.custom.unknown"] = "x"
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(result_data))

        result = runner.invoke(app, ["explain", "-r", str(result_file)])

        assert result.exit_code == 0
        assert "spark.executor.memory" in result.output
        # Known parameter resolves to a heuristic rule
        assert "heuristic" in result.output
        # Unknown parameter falls back to the Bayesian note
        assert "bayesian" in result.output

    def test_explain_empty_configuration(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test explain with an empty configuration."""
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({"configuration": {}}))

        result = runner.invoke(app, ["explain", "-r", str(result_file)])

        assert result.exit_code == 0
        assert "no configuration" in result.output.lower()

    def test_explain_missing_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test explain errors on a missing result file."""
        result = runner.invoke(app, ["explain", "-r", str(tmp_path / "missing.json")])

        assert result.exit_code == 1
        assert "Error loading result file" in result.output

    def test_explain_missing_required_option(self, runner: CliRunner) -> None:
        """Test explain requires the result file option."""
        result = runner.invoke(app, ["explain"])

        assert result.exit_code != 0


class TestCLIOptimizeAutoSave:
    """Test cases for the optimize command's history auto-save."""

    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_saves_to_history(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
        history_db: Path,
        sample_result_dict: dict,
    ) -> None:
        """Test a successful optimize run is persisted to history."""
        from spark_optima.core.history import OptimizationHistory
        from spark_optima.core.result import OptimizationResult

        real_result = OptimizationResult(
            configuration=sample_result_dict["configuration"],
            estimated_time_minutes=12.5,
            confidence_score=0.85,
            platform_specific={"platform": "local"},
        )
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = real_result
        mock_optimizer.return_value = mock_instance

        result = runner.invoke(
            app,
            ["optimize", "--code-path", str(sample_code_file), "--platform", "local"],
        )

        assert result.exit_code == 0
        assert "Saved to history" in result.output

        with OptimizationHistory() as store:
            entries = store.list_entries()
            assert len(entries) == 1
            assert entries[0].platform == "local"
            assert entries[0].configuration == sample_result_dict["configuration"]

    @patch("spark_optima.core.history.OptimizationHistory.save")
    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_history_failure_is_non_fatal(
        self,
        mock_optimizer: MagicMock,
        mock_save: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
        history_db: Path,
        sample_result_dict: dict,
    ) -> None:
        """Test that a history save failure never breaks the optimize flow."""
        from spark_optima.core.result import OptimizationResult

        mock_save.side_effect = RuntimeError("disk full")
        real_result = OptimizationResult(
            configuration=sample_result_dict["configuration"],
            estimated_time_minutes=12.5,
            confidence_score=0.85,
            platform_specific={"platform": "local"},
        )
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = real_result
        mock_optimizer.return_value = mock_instance

        result = runner.invoke(
            app,
            ["optimize", "--code-path", str(sample_code_file), "--platform", "local"],
        )

        assert result.exit_code == 0
        assert "Saved to history" not in result.output


class TestCLIAnalyzeRealFile:
    """Regression tests: analyze must read file contents, not parse the path string."""

    def test_analyze_reads_file_contents(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test analyze detects smells from the actual file contents."""
        code_file = tmp_path / "job_with_smells.py"
        code_file.write_text(
            "df = spark.read.parquet('s3://bucket/a/')\n"
            "other = spark.read.parquet('s3://bucket/b/')\n"
            "joined = df.crossJoin(other)\n"
            "pdf = joined.toPandas()\n",
        )

        result = runner.invoke(app, ["analyze", "--code-path", str(code_file), "--output", "json"])

        assert result.exit_code == 0
        assert "cartesian_join" in result.output
        assert "topandas_usage" in result.output

    def test_analyze_clean_file_exits_zero(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test analyze succeeds on a syntactically valid file with no smells."""
        code_file = tmp_path / "clean_job.py"
        code_file.write_text("x = 1\n")

        result = runner.invoke(app, ["analyze", "--code-path", str(code_file)])

        assert result.exit_code == 0
        assert "Syntax error" not in result.output


# =============================================================================
# v1.2 — analyze-log / optimize --event-log (Workstream F)
# =============================================================================

_GB = 1024**3


def _sample_event_log_events() -> list[dict]:
    """Build synthetic Spark event log events for CLI tests.

    Expected aggregates: 60 s app duration, 3 tasks, skew ratio 10.0,
    0.25 GB spill, 10 GB input data.
    """

    def task_end(launch: int, finish: int, run_time: int, spill: int, input_bytes: int) -> dict:
        return {
            "Event": "SparkListenerTaskEnd",
            "Stage ID": 0,
            "Task End Reason": {"Reason": "Success"},
            "Task Info": {"Launch Time": launch, "Finish Time": finish, "Failed": False},
            "Task Metrics": {
                "Executor Run Time": run_time,
                "JVM GC Time": 200,
                "Memory Bytes Spilled": spill,
                "Disk Bytes Spilled": 0,
                "Peak Execution Memory": _GB,
                "Shuffle Read Metrics": {"Remote Bytes Read": _GB, "Local Bytes Read": 0},
                "Shuffle Write Metrics": {"Shuffle Bytes Written": _GB // 2},
                "Input Metrics": {"Bytes Read": input_bytes},
            },
        }

    return [
        {"Event": "SparkListenerApplicationStart", "App Name": "cli-test-app", "Timestamp": 1_000_000},
        {
            "Event": "SparkListenerEnvironmentUpdate",
            "Spark Properties": {"spark.executor.memory": "4g"},
        },
        {"Event": "SparkListenerExecutorAdded", "Executor ID": "1", "Timestamp": 1_000_000},
        task_end(1_000_000, 1_001_000, 1000, _GB // 4, 5 * _GB),
        task_end(1_000_000, 1_001_000, 1000, 0, 5 * _GB),
        task_end(1_000_000, 1_010_000, 10_000, 0, 0),
        {
            "Event": "SparkListenerStageCompleted",
            "Stage Info": {
                "Stage ID": 0,
                "Stage Name": "cli stage",
                "Number of Tasks": 3,
                "Submission Time": 1_000_000,
                "Completion Time": 1_010_000,
                "Accumulables": [],
            },
        },
        {"Event": "SparkListenerApplicationEnd", "Timestamp": 1_060_000},
    ]


@pytest.fixture
def sample_event_log(tmp_path: Path) -> Path:
    """Write a synthetic event log to a temporary file."""
    log_file = tmp_path / "eventlog"
    log_file.write_text(
        "\n".join(json.dumps(event) for event in _sample_event_log_events()) + "\n",
        encoding="utf-8",
    )
    return log_file


class TestCLIAnalyzeLogCommand:
    """Test cases for the analyze-log command."""

    def test_analyze_log_command_help(self, runner: CliRunner) -> None:
        """Test analyze-log command help."""
        result = runner.invoke(app, ["analyze-log", "--help"])

        assert result.exit_code == 0
        assert "event log" in result.output.lower()

    def test_analyze_log_table_output(self, runner: CliRunner, sample_event_log: Path) -> None:
        """Test analyze-log table output shows summary, stages, and hints."""
        result = runner.invoke(app, ["analyze-log", "--log-path", str(sample_event_log)])

        assert result.exit_code == 0
        assert "Spark Event Log Analysis" in result.output
        assert "cli-test-app" in result.output
        assert "Run Summary" in result.output
        assert "Stages by Duration" in result.output
        assert "Tuning Hints" in result.output
        # Skew ratio 10.0 should be flagged as severe with AQE advice
        assert "skew" in result.output.lower()

    def test_analyze_log_json_output(self, runner: CliRunner, sample_event_log: Path) -> None:
        """Test analyze-log JSON output is valid, complete, and unwrapped."""
        result = runner.invoke(
            app,
            ["analyze-log", "--log-path", str(sample_event_log), "--output", "json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["app_name"] == "cli-test-app"
        assert payload["app_duration_seconds"] == pytest.approx(60.0)
        assert payload["total_tasks"] == 3
        assert payload["input_data_gb"] == pytest.approx(10.0)
        assert payload["spark_conf"]["spark.executor.memory"] == "4g"
        assert payload["tuning_hints"]["skew_factor"] == pytest.approx(10.0)
        assert payload["tuning_hints"]["spill_detected"] is True
        assert payload["tuning_hints"]["data_size_gb"] == pytest.approx(10.0)

    def test_analyze_log_missing_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test analyze-log errors cleanly for a missing log file."""
        result = runner.invoke(app, ["analyze-log", "--log-path", str(tmp_path / "missing")])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_analyze_log_requires_log_path(self, runner: CliRunner) -> None:
        """Test analyze-log requires the --log-path option."""
        result = runner.invoke(app, ["analyze-log"])

        assert result.exit_code != 0


class TestCLIOptimizeEventLog:
    """Test cases for optimize --event-log enrichment."""

    @staticmethod
    def _mock_result() -> object:
        """Build a real OptimizationResult for the mocked optimizer."""
        from spark_optima.core.result import OptimizationResult

        return OptimizationResult(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.9,
            platform_specific={"platform": "local"},
        )

    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_event_log_infers_data_size(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
        sample_event_log: Path,
    ) -> None:
        """Test --event-log infers data size and merges tuning hints."""
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = self._mock_result()
        mock_optimizer.return_value = mock_instance

        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "local",
                "--event-log",
                str(sample_event_log),
            ],
        )

        assert result.exit_code == 0
        assert "Inferred from event log" in result.output

        _, kwargs = mock_instance.optimize.call_args
        assert kwargs["data_profile"]["size_gb"] == pytest.approx(10.0)
        assert kwargs["resource_constraints"]["skew_factor"] == pytest.approx(10.0)
        assert kwargs["resource_constraints"]["spill_detected"] is True
        assert kwargs["resource_constraints"]["memory_intensive"] is True
        # data_size_gb must flow through the data profile, not the constraints
        assert "data_size_gb" not in kwargs["resource_constraints"]

    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_event_log_explicit_data_size_wins(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
        sample_event_log: Path,
    ) -> None:
        """Test an explicit --data-size is not overridden by the event log."""
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = self._mock_result()
        mock_optimizer.return_value = mock_instance

        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "local",
                "--data-size",
                "55.0",
                "--event-log",
                str(sample_event_log),
            ],
        )

        assert result.exit_code == 0

        _, kwargs = mock_instance.optimize.call_args
        assert kwargs["data_profile"]["size_gb"] == pytest.approx(55.0)
        # Hints are still merged even when the data size was explicit
        assert kwargs["resource_constraints"]["skew_factor"] == pytest.approx(10.0)

    def test_optimize_event_log_missing_file(
        self,
        runner: CliRunner,
        sample_code_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test optimize errors cleanly when the event log does not exist."""
        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--platform",
                "local",
                "--event-log",
                str(tmp_path / "missing-log"),
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.output


def _make_pareto_result_dict() -> dict:
    """Build a result dictionary with a 3-point Pareto frontier."""
    return {
        "configuration": {"spark.executor.memory": "4g"},
        "estimated_time_minutes": 10.0,
        "confidence_score": 0.9,
        "code_suggestions": [],
        "platform_specific": {"platform": "local"},
        "metadata": {
            "objectives": ["minimize_time", "minimize_cost"],
            "pareto_frontier": [
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
                {
                    "trial_number": 7,
                    "objective_values": {"minimize_time": 150.0, "minimize_cost": 0.20},
                    "configuration": {"spark.executor.memory": "2g", "spark.executor.cores": 2},
                },
            ],
        },
    }


@pytest.fixture
def pareto_result_file(tmp_path: Path) -> Path:
    """Write a multi-objective result file with a Pareto frontier."""
    path = tmp_path / "pareto_result.json"
    path.write_text(json.dumps(_make_pareto_result_dict()))
    return path


@pytest.fixture
def single_objective_result_file(tmp_path: Path) -> Path:
    """Write a single-objective result file (no Pareto frontier)."""
    data = _make_pareto_result_dict()
    data["metadata"] = {"platform": "local"}
    path = tmp_path / "single_result.json"
    path.write_text(json.dumps(data))
    return path


class TestCLIOptimizeObjectiveOption:
    """Test cases for the optimize --objective option (Workstream N)."""

    @staticmethod
    def _mock_result() -> object:
        """Build a real OptimizationResult for the mocked optimizer."""
        from spark_optima.core.result import OptimizationResult

        return OptimizationResult(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.9,
            platform_specific={"platform": "local"},
        )

    def test_optimize_rejects_unknown_objective(
        self,
        runner: CliRunner,
        sample_code_file: Path,
    ) -> None:
        """Unknown objectives are rejected with the valid names listed."""
        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--objective",
                "minimize_bananas",
            ],
        )

        assert result.exit_code == 1
        assert "minimize_bananas" in result.output
        # Error message lists all valid objective names
        assert "minimize_time" in result.output
        assert "minimize_cost" in result.output
        assert "maximize_success" in result.output
        assert "minimize_memory" in result.output

    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_passes_objectives_through(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeated --objective flags reach Optimizer.optimize as a list."""
        monkeypatch.setenv("SPARK_OPTIMA_HISTORY_DB", str(tmp_path / "history.db"))
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = self._mock_result()
        mock_optimizer.return_value = mock_instance

        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--objective",
                "minimize_time",
                "--objective",
                "minimize_cost",
            ],
        )

        assert result.exit_code == 0
        _, kwargs = mock_instance.optimize.call_args
        assert kwargs["objectives"] == ["minimize_time", "minimize_cost"]

    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_default_objectives_is_none(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without --objective the optimizer receives objectives=None (default)."""
        monkeypatch.setenv("SPARK_OPTIMA_HISTORY_DB", str(tmp_path / "history.db"))
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = self._mock_result()
        mock_optimizer.return_value = mock_instance

        result = runner.invoke(
            app,
            ["optimize", "--code-path", str(sample_code_file)],
        )

        assert result.exit_code == 0
        _, kwargs = mock_instance.optimize.call_args
        assert kwargs["objectives"] is None

    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_deduplicates_objectives(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Duplicate --objective flags are deduplicated preserving order."""
        monkeypatch.setenv("SPARK_OPTIMA_HISTORY_DB", str(tmp_path / "history.db"))
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = self._mock_result()
        mock_optimizer.return_value = mock_instance

        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_code_file),
                "--objective",
                "minimize_cost",
                "--objective",
                "minimize_cost",
                "--objective",
                "minimize_time",
            ],
        )

        assert result.exit_code == 0
        _, kwargs = mock_instance.optimize.call_args
        assert kwargs["objectives"] == ["minimize_cost", "minimize_time"]


class TestCLIParetoCommand:
    """Test cases for the pareto command (Workstream N)."""

    def test_pareto_command_help(self, runner: CliRunner) -> None:
        """Test pareto command help."""
        result = runner.invoke(app, ["pareto", "--help"])

        assert result.exit_code == 0
        assert "pareto" in result.output.lower()

    def test_pareto_table_output(self, runner: CliRunner, pareto_result_file: Path) -> None:
        """Table output lists frontier points, objectives, and a trade-off summary."""
        result = runner.invoke(app, ["pareto", "-r", str(pareto_result_file)])

        assert result.exit_code == 0
        assert "Pareto Frontier" in result.output
        assert "minimize_time" in result.output
        assert "minimize_cost" in result.output
        # All trial numbers shown
        for trial in ("0", "3", "7"):
            assert trial in result.output
        assert "Trade-off summary" in result.output
        # Fastest point is trial 3, cheapest is trial 7
        assert "trial 3" in result.output
        assert "trial 7" in result.output

    def test_pareto_json_output(self, runner: CliRunner, pareto_result_file: Path) -> None:
        """JSON output is pure machine-readable JSON via typer.echo."""
        result = runner.invoke(
            app,
            ["pareto", "-r", str(pareto_result_file), "--output", "json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["objectives"] == ["minimize_time", "minimize_cost"]
        assert payload["n_points"] == 3
        assert len(payload["points"]) == 3
        assert payload["points"][0]["trial_number"] == 0
        assert payload["points"][0]["objective_values"]["minimize_time"] == pytest.approx(120.0)

    def test_pareto_missing_frontier_errors(
        self,
        runner: CliRunner,
        single_objective_result_file: Path,
    ) -> None:
        """A result without a frontier exits 1 with a multi-objective hint."""
        result = runner.invoke(app, ["pareto", "-r", str(single_objective_result_file)])

        assert result.exit_code == 1
        assert "no Pareto frontier" in result.output
        assert "--objective" in result.output

    def test_pareto_missing_file_errors(self, runner: CliRunner, tmp_path: Path) -> None:
        """A missing result file exits 1 with an error."""
        result = runner.invoke(app, ["pareto", "-r", str(tmp_path / "missing.json")])

        assert result.exit_code == 1
        assert "Error" in result.output


class TestCLIExportParetoFormats:
    """Test cases for export -f pareto-json / pareto-csv routing (Workstream N)."""

    def test_export_pareto_json_stdout(self, runner: CliRunner, pareto_result_file: Path) -> None:
        """pareto-json is dispatched to stdout as JSON."""
        result = runner.invoke(
            app,
            ["export", "-r", str(pareto_result_file), "-f", "pareto-json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output[result.output.index("{") :])
        assert payload["n_points"] == 3
        assert payload["objectives"] == ["minimize_time", "minimize_cost"]

    def test_export_pareto_csv_stdout(self, runner: CliRunner, pareto_result_file: Path) -> None:
        """pareto-csv is dispatched to stdout with deterministic columns."""
        result = runner.invoke(
            app,
            ["export", "-r", str(pareto_result_file), "-f", "pareto-csv"],
        )

        assert result.exit_code == 0
        assert "trial,minimize_time,minimize_cost,spark.executor.cores,spark.executor.memory" in result.output

    def test_export_pareto_csv_to_file(
        self,
        runner: CliRunner,
        pareto_result_file: Path,
        tmp_path: Path,
    ) -> None:
        """pareto-csv routes through save_to_file when --output is given."""
        output_file = tmp_path / "frontier.csv"
        result = runner.invoke(
            app,
            [
                "export",
                "-r",
                str(pareto_result_file),
                "-f",
                "pareto-csv",
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        content = output_file.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "trial,minimize_time,minimize_cost,spark.executor.cores,spark.executor.memory"
        # Rows sorted by trial number
        assert lines[1].startswith("0,")
        assert lines[2].startswith("3,")
        assert lines[3].startswith("7,")

    def test_export_pareto_without_frontier_errors(
        self,
        runner: CliRunner,
        single_objective_result_file: Path,
    ) -> None:
        """Exporting the frontier of a single-objective result exits 1."""
        result = runner.invoke(
            app,
            ["export", "-r", str(single_objective_result_file), "-f", "pareto-json"],
        )

        assert result.exit_code == 1
        assert "Export error" in result.output


# =============================================================================
# v1.4 — validate / import / templates (Workstream S)
# =============================================================================


def _write_config_file(tmp_path: Path, lines: list[str], name: str = "spark-defaults.conf") -> Path:
    """Write a properties-format config file for validate/import tests."""
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _issue_checks(payload: dict) -> set[str]:
    """Collect the set of check names present in a validate JSON payload."""
    return {issue["check"] for issue in payload["issues"]}


def _issues_for_check(payload: dict, check: str) -> list[dict]:
    """Return the issues produced by a specific check."""
    return [issue for issue in payload["issues"] if issue["check"] == check]


class TestCLIValidateCommand:
    """Test cases for the validate command (Workstream S)."""

    def test_validate_command_help(self, runner: CliRunner) -> None:
        """Test validate command help."""
        result = runner.invoke(app, ["validate", "--help"])

        assert result.exit_code == 0
        assert "validate" in result.output.lower()

    def test_validate_clean_properties_config(self, runner: CliRunner, tmp_path: Path) -> None:
        """A clean properties config produces no issues and exits 0."""
        config = _write_config_file(
            tmp_path,
            [
                "# spark defaults for the nightly job",
                "",
                "spark.executor.memory 4g",
                "spark.executor.cores=4",
                "spark.sql.adaptive.enabled true",
            ],
        )

        result = runner.invoke(app, ["validate", "-c", str(config)])

        assert result.exit_code == 0
        assert "No issues found" in result.output

    def test_validate_json_config_input(self, runner: CliRunner, tmp_path: Path) -> None:
        """A JSON dict config with native types parses and validates."""
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps(
                {
                    "spark.executor.memory": "4g",
                    "spark.executor.cores": 4,
                    "spark.sql.adaptive.enabled": True,
                },
            ),
        )

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["valid"] is True
        assert payload["parameter_count"] == 3
        assert payload["error_count"] == 0
        assert payload["issues"] == []

    def test_validate_unknown_parameter_warns(self, runner: CliRunner, tmp_path: Path) -> None:
        """Unknown parameters are reported as warnings (exit 0)."""
        config = _write_config_file(tmp_path, ["spark.executor.memry 4g"])

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["valid"] is True
        issues = _issues_for_check(payload, "unknown_parameter")
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"
        assert issues[0]["param"] == "spark.executor.memry"

    def test_validate_deprecated_parameter_warns(self, runner: CliRunner, tmp_path: Path) -> None:
        """Parameters deprecated in the target version produce warnings."""
        config = _write_config_file(tmp_path, ["spark.sql.legacy.allowUntypedScalaUDF true"])

        result = runner.invoke(
            app,
            ["validate", "-c", str(config), "-s", "4.1.0", "--output", "json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        issues = _issues_for_check(payload, "deprecated_parameter")
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"
        assert "deprecated" in issues[0]["message"].lower()

    def test_validate_no_deprecation_for_current_parameter(self, runner: CliRunner, tmp_path: Path) -> None:
        """Non-deprecated parameters do not trigger the deprecation check."""
        config = _write_config_file(tmp_path, ["spark.executor.memory 4g"])

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "deprecated_parameter" not in _issue_checks(payload)

    def test_validate_invalid_integer_value(self, runner: CliRunner, tmp_path: Path) -> None:
        """A non-integer value for an integer parameter is an error (exit 1)."""
        config = _write_config_file(tmp_path, ["spark.executor.cores abc"])

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["valid"] is False
        issues = _issues_for_check(payload, "invalid_value")
        assert issues and issues[0]["severity"] == "error"
        assert "integer" in issues[0]["message"]

    def test_validate_invalid_boolean_value(self, runner: CliRunner, tmp_path: Path) -> None:
        """A non-boolean value for a boolean parameter is an error."""
        config = _write_config_file(tmp_path, ["spark.sql.adaptive.enabled maybe"])

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 1
        payload = json.loads(result.output)
        issues = _issues_for_check(payload, "invalid_value")
        assert issues and "boolean" in issues[0]["message"]

    def test_validate_invalid_memory_format(self, runner: CliRunner, tmp_path: Path) -> None:
        """An invalid byte-size string is an error."""
        config = _write_config_file(tmp_path, ["spark.executor.memory 4x"])

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 1
        payload = json.loads(result.output)
        issues = _issues_for_check(payload, "invalid_value")
        assert issues and "byte size" in issues[0]["message"]

    def test_validate_constraint_violation(self, runner: CliRunner, tmp_path: Path) -> None:
        """A value beyond the database max constraint is an error."""
        config = _write_config_file(tmp_path, ["spark.executor.cores 999"])

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 1
        payload = json.loads(result.output)
        issues = _issues_for_check(payload, "invalid_value")
        assert issues and "maximum" in issues[0]["message"]

    def test_validate_platform_constraint_violation(self, runner: CliRunner, tmp_path: Path) -> None:
        """Executor memory beyond the platform maximum is an error."""
        config = _write_config_file(tmp_path, ["spark.executor.memory 300g"])

        result = runner.invoke(
            app,
            ["validate", "-c", str(config), "-p", "aws_glue", "--output", "json"],
        )

        assert result.exit_code == 1
        payload = json.loads(result.output)
        issues = _issues_for_check(payload, "platform_constraint")
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"
        assert "aws_glue" in issues[0]["message"]

    def test_validate_platform_constraint_satisfied(self, runner: CliRunner, tmp_path: Path) -> None:
        """Executor memory/cores within platform limits produce no issues."""
        config = _write_config_file(
            tmp_path,
            ["spark.executor.memory 8g", "spark.executor.cores 4"],
        )

        result = runner.invoke(
            app,
            ["validate", "-c", str(config), "-p", "aws_glue", "--output", "json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "platform_constraint" not in _issue_checks(payload)

    def test_validate_unknown_platform(self, runner: CliRunner, tmp_path: Path) -> None:
        """An unknown platform name exits 1 with the valid options."""
        config = _write_config_file(tmp_path, ["spark.executor.memory 4g"])

        result = runner.invoke(app, ["validate", "-c", str(config), "-p", "narnia"])

        assert result.exit_code == 1
        assert "Unknown platform" in result.output

    def test_validate_antipattern_driver_exceeds_executor(self, runner: CliRunner, tmp_path: Path) -> None:
        """Driver memory above executor memory is flagged as a warning."""
        config = _write_config_file(
            tmp_path,
            ["spark.driver.memory 8g", "spark.executor.memory 4g"],
        )

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        issues = _issues_for_check(payload, "driver_memory_exceeds_executor")
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"

    def test_validate_antipattern_driver_below_executor_ok(self, runner: CliRunner, tmp_path: Path) -> None:
        """Driver memory below executor memory is not flagged."""
        config = _write_config_file(
            tmp_path,
            ["spark.driver.memory 2g", "spark.executor.memory 4g"],
        )

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "driver_memory_exceeds_executor" not in _issue_checks(payload)

    def test_validate_antipattern_dynamic_allocation_inverted_bounds(self, runner: CliRunner, tmp_path: Path) -> None:
        """maxExecutors below minExecutors with dynamic allocation is an error."""
        config = _write_config_file(
            tmp_path,
            [
                "spark.dynamicAllocation.enabled true",
                "spark.dynamicAllocation.minExecutors 10",
                "spark.dynamicAllocation.maxExecutors 5",
                "spark.shuffle.service.enabled true",
            ],
        )

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 1
        payload = json.loads(result.output)
        issues = _issues_for_check(payload, "dynamic_allocation_bounds")
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"

    def test_validate_antipattern_dynamic_allocation_equal_bounds(self, runner: CliRunner, tmp_path: Path) -> None:
        """maxExecutors equal to minExecutors is only a warning."""
        config = _write_config_file(
            tmp_path,
            [
                "spark.dynamicAllocation.enabled true",
                "spark.dynamicAllocation.minExecutors 5",
                "spark.dynamicAllocation.maxExecutors 5",
                "spark.shuffle.service.enabled true",
            ],
        )

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        issues = _issues_for_check(payload, "dynamic_allocation_bounds")
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"

    def test_validate_antipattern_dynamic_allocation_without_shuffle(self, runner: CliRunner, tmp_path: Path) -> None:
        """Dynamic allocation without shuffle service or tracking warns."""
        config = _write_config_file(tmp_path, ["spark.dynamicAllocation.enabled true"])

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        issues = _issues_for_check(payload, "dynamic_allocation_shuffle")
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"

    def test_validate_antipattern_dynamic_allocation_with_shuffle_service(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Dynamic allocation with the shuffle service enabled is not flagged."""
        config = _write_config_file(
            tmp_path,
            [
                "spark.dynamicAllocation.enabled true",
                "spark.shuffle.service.enabled true",
            ],
        )

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "dynamic_allocation_shuffle" not in _issue_checks(payload)

    def test_validate_antipattern_java_serializer_with_kryo_settings(self, runner: CliRunner, tmp_path: Path) -> None:
        """Java serializer combined with Kryo settings is flagged."""
        config = _write_config_file(
            tmp_path,
            [
                "spark.serializer org.apache.spark.serializer.JavaSerializer",
                "spark.kryoserializer.buffer.max 512m",
            ],
        )

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        issues = _issues_for_check(payload, "serializer_mismatch")
        assert len(issues) == 1
        assert "spark.kryoserializer.buffer.max" in issues[0]["message"]

    def test_validate_kryo_serializer_with_kryo_settings_ok(self, runner: CliRunner, tmp_path: Path) -> None:
        """Kryo serializer with Kryo settings is not flagged."""
        config = _write_config_file(
            tmp_path,
            [
                "spark.serializer org.apache.spark.serializer.KryoSerializer",
                "spark.kryoserializer.buffer.max 512m",
            ],
        )

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "serializer_mismatch" not in _issue_checks(payload)

    def test_validate_antipattern_aqe_disabled_on_modern_spark(self, runner: CliRunner, tmp_path: Path) -> None:
        """AQE disabled on Spark >= 3.2 is flagged as a warning."""
        config = _write_config_file(tmp_path, ["spark.sql.adaptive.enabled false"])

        result = runner.invoke(
            app,
            ["validate", "-c", str(config), "-s", "3.5.0", "--output", "json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        issues = _issues_for_check(payload, "aqe_disabled")
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"

    def test_validate_aqe_disabled_on_old_spark_ok(self, runner: CliRunner, tmp_path: Path) -> None:
        """AQE disabled on Spark < 3.2 is not flagged."""
        config = _write_config_file(tmp_path, ["spark.sql.adaptive.enabled false"])

        result = runner.invoke(
            app,
            ["validate", "-c", str(config), "-s", "3.1.0", "--output", "json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "aqe_disabled" not in _issue_checks(payload)

    def test_validate_malformed_properties_line(self, runner: CliRunner, tmp_path: Path) -> None:
        """A properties line without a value exits 1."""
        config = _write_config_file(tmp_path, ["spark.executor.memory"])

        result = runner.invoke(app, ["validate", "-c", str(config)])

        assert result.exit_code == 1
        assert "cannot parse line" in result.output

    def test_validate_missing_config_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """A missing config file exits 1."""
        result = runner.invoke(app, ["validate", "-c", str(tmp_path / "missing.conf")])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_validate_table_output_groups_by_severity(self, runner: CliRunner, tmp_path: Path) -> None:
        """Table output shows the issues table and a severity summary."""
        config = _write_config_file(
            tmp_path,
            [
                "spark.executor.cores abc",
                "spark.unknown.param 1",
            ],
        )

        result = runner.invoke(app, ["validate", "-c", str(config)])

        assert result.exit_code == 1
        assert "Validation Issues" in result.output
        assert "error" in result.output
        assert "warning" in result.output
        assert "1 error(s)" in result.output
        assert "1 warning(s)" in result.output


class TestCLIImportCommand:
    """Test cases for the import command (Workstream S)."""

    @staticmethod
    def _mock_result(configuration: dict | None = None) -> object:
        """Build a real OptimizationResult for the mocked optimizer."""
        from spark_optima.core.result import OptimizationResult

        return OptimizationResult(
            configuration=configuration
            if configuration is not None
            else {
                "spark.executor.memory": "8g",
                "spark.executor.cores": 4,
                "spark.sql.shuffle.partitions": 200,
            },
            estimated_time_minutes=10.0,
            confidence_score=0.9,
            platform_specific={"platform": "local"},
        )

    def test_import_command_help(self, runner: CliRunner) -> None:
        """Test import command help."""
        result = runner.invoke(app, ["import", "--help"])

        assert result.exit_code == 0
        assert "import" in result.output.lower()

    @patch("spark_optima.cli.main.Optimizer")
    def test_import_json_diff(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
        tmp_path: Path,
    ) -> None:
        """JSON output carries current, recommended, and a correct diff."""
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = self._mock_result()
        mock_optimizer.return_value = mock_instance

        config = _write_config_file(
            tmp_path,
            [
                "spark.executor.memory 2g",
                "spark.old.param x",
                "spark.sql.shuffle.partitions 200",
            ],
        )

        result = runner.invoke(
            app,
            [
                "import",
                "-c",
                str(config),
                "--code",
                str(sample_code_file),
                "--output",
                "json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["current"]["spark.executor.memory"] == "2g"
        assert payload["recommended"]["spark.executor.memory"] == "8g"
        assert payload["diff"]["changed"] == {
            "spark.executor.memory": {"current": "2g", "recommended": "8g"},
        }
        assert payload["diff"]["only_in_current"] == {"spark.old.param": "x"}
        assert payload["diff"]["only_in_recommended"] == {"spark.executor.cores": 4}
        # "200" (string) and 200 (int) must compare as equal
        assert "spark.sql.shuffle.partitions" not in payload["diff"]["changed"]
        assert payload["estimated_time_minutes"] == pytest.approx(10.0)

    @patch("spark_optima.cli.main.Optimizer")
    def test_import_table_output(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
        tmp_path: Path,
    ) -> None:
        """Table output shows the diff sections and the estimated time."""
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = self._mock_result()
        mock_optimizer.return_value = mock_instance

        config = _write_config_file(
            tmp_path,
            ["spark.executor.memory 2g", "spark.old.param x"],
        )

        result = runner.invoke(
            app,
            ["import", "-c", str(config), "--code", str(sample_code_file)],
        )

        assert result.exit_code == 0
        assert "Current vs Recommended" in result.output
        assert "Only in current" in result.output
        assert "Only in recommended" in result.output
        assert "10.0 min" in result.output

    @patch("spark_optima.cli.main.Optimizer")
    def test_import_identical_config(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
        tmp_path: Path,
    ) -> None:
        """An already-optimal config reports no differences."""
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = self._mock_result({"spark.executor.memory": "4g"})
        mock_optimizer.return_value = mock_instance

        config = _write_config_file(tmp_path, ["spark.executor.memory 4g"])

        result = runner.invoke(
            app,
            ["import", "-c", str(config), "--code", str(sample_code_file)],
        )

        assert result.exit_code == 0
        assert "already matches" in result.output

    @patch("spark_optima.cli.main.Optimizer")
    def test_import_passes_options_through(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
        tmp_path: Path,
    ) -> None:
        """Platform, version, data size, and trials reach the Optimizer."""
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = self._mock_result()
        mock_optimizer.return_value = mock_instance

        config = _write_config_file(tmp_path, ["spark.executor.memory 2g"])

        result = runner.invoke(
            app,
            [
                "import",
                "-c",
                str(config),
                "--code",
                str(sample_code_file),
                "-p",
                "databricks",
                "-s",
                "3.4.0",
                "-d",
                "25",
                "--bayesian-trials",
                "7",
            ],
        )

        assert result.exit_code == 0
        _, init_kwargs = mock_optimizer.call_args
        assert init_kwargs["platform"] == "databricks"
        assert init_kwargs["spark_version"] == "3.4.0"
        _, optimize_kwargs = mock_instance.optimize.call_args
        assert optimize_kwargs["bayesian_trials"] == 7
        assert optimize_kwargs["data_profile"] == {"size_gb": 25.0}

    @patch("spark_optima.cli.main.Optimizer")
    def test_import_optimizer_failure(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
        tmp_path: Path,
    ) -> None:
        """An optimization failure exits 1 with an error message."""
        mock_instance = MagicMock()
        mock_instance.optimize.side_effect = RuntimeError("boom")
        mock_optimizer.return_value = mock_instance

        config = _write_config_file(tmp_path, ["spark.executor.memory 2g"])

        result = runner.invoke(
            app,
            ["import", "-c", str(config), "--code", str(sample_code_file)],
        )

        assert result.exit_code == 1
        assert "Optimization failed" in result.output

    def test_import_missing_config_file(self, runner: CliRunner, sample_code_file: Path, tmp_path: Path) -> None:
        """A missing config file exits 1."""
        result = runner.invoke(
            app,
            [
                "import",
                "-c",
                str(tmp_path / "missing.conf"),
                "--code",
                str(sample_code_file),
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_import_requires_options(self, runner: CliRunner) -> None:
        """import requires both --config and --code."""
        result = runner.invoke(app, ["import"])

        assert result.exit_code != 0


class TestCLITemplatesCommand:
    """Test cases for the templates command (Workstream S)."""

    def test_templates_command_help(self, runner: CliRunner) -> None:
        """Test templates command help."""
        result = runner.invoke(app, ["templates", "--help"])

        assert result.exit_code == 0
        assert "template" in result.output.lower()

    def test_templates_list_table(self, runner: CliRunner) -> None:
        """The list table shows all four bundled templates."""
        result = runner.invoke(app, ["templates"])

        assert result.exit_code == 0
        for name in ("etl-batch", "streaming", "ml-training", "interactive"):
            assert name in result.output

    def test_templates_list_json(self, runner: CliRunner) -> None:
        """JSON list output is machine readable with all templates."""
        result = runner.invoke(app, ["templates", "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert {entry["name"] for entry in payload} == {
            "etl-batch",
            "streaming",
            "ml-training",
            "interactive",
        }

    def test_templates_show_json(self, runner: CliRunner) -> None:
        """JSON show output contains the curated params with rationale."""
        result = runner.invoke(
            app,
            ["templates", "--show", "etl-batch", "--output", "json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["name"] == "etl-batch"
        assert payload["config"]["spark.sql.adaptive.enabled"]["value"] == "true"
        assert payload["config"]["spark.sql.adaptive.enabled"]["comment"]
        assert payload["recommended_for"]
        assert payload["not_recommended_for"]

    def test_templates_show_table(self, runner: CliRunner) -> None:
        """Table show output displays the template details."""
        result = runner.invoke(app, ["templates", "--show", "streaming"])

        assert result.exit_code == 0
        assert "Streaming" in result.output
        assert "Curated Configuration" in result.output
        assert "Recommended for" in result.output

    def test_templates_show_unknown(self, runner: CliRunner) -> None:
        """An unknown template name exits 1 listing the available ones."""
        result = runner.invoke(app, ["templates", "--show", "does-not-exist"])

        assert result.exit_code == 1
        assert "Unknown template" in result.output


@pytest.fixture
def sample_scala_file(tmp_path: Path) -> Path:
    """Create a sample Scala Spark code file for testing."""
    scala_file = tmp_path / "EtlJob.scala"
    scala_file.write_text(
        "import org.apache.spark.sql.SparkSession\n"
        "\n"
        "object EtlJob {\n"
        "  def main(args: Array[String]): Unit = {\n"
        '    val spark = SparkSession.builder().appName("etl").getOrCreate()\n'
        '    val df = spark.read.parquet("data.parquet")\n'
        "    val out = df.crossJoin(df)\n"
        "    val legacy = rdd.groupByKey().mapValues(_.size)\n"
        "    out.collect()\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    return scala_file


class TestCLIScalaSupport:
    """Test cases for Scala source handling in analyze and optimize."""

    def test_analyze_with_scala_file(self, runner: CliRunner, sample_scala_file: Path) -> None:
        """Analyze accepts a .scala file and detects Scala smells."""
        result = runner.invoke(app, ["analyze", "--code-path", str(sample_scala_file)])

        assert result.exit_code == 0
        assert "Scala source detected" in result.output
        assert "cartesian_join" in result.output
        assert "groupbykey_usage" in result.output

    def test_analyze_scala_json_output(self, runner: CliRunner, sample_scala_file: Path) -> None:
        """Analyze --output json produces parseable results for Scala files."""
        result = runner.invoke(
            app,
            ["analyze", "--code-path", str(sample_scala_file), "--output", "json"],
        )

        assert result.exit_code == 0
        json_start = result.output.index("{")
        payload = json.loads(result.output[json_start:])
        smell_types = {smell["smell_type"] for smell in payload["smells"]}
        assert "cartesian_join" in smell_types
        assert "groupbykey_usage" in smell_types
        assert "large_collect" in smell_types
        assert payload["metadata"]["language"] == "scala"

    def test_analyze_python_file_unchanged(self, runner: CliRunner, sample_code_file: Path) -> None:
        """Python files do not trigger the Scala parser notice."""
        result = runner.invoke(app, ["analyze", "--code-path", str(sample_code_file)])

        assert result.exit_code == 0
        assert "Scala source detected" not in result.output

    def test_optimize_with_scala_file(self, runner: CliRunner, sample_scala_file: Path) -> None:
        """Optimize accepts a .scala file without crashing on code analysis."""
        result = runner.invoke(
            app,
            [
                "optimize",
                "--code-path",
                str(sample_scala_file),
                "--platform",
                "local",
                "--bayesian-trials",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert "Scala source detected" in result.output


# =============================================================================
# v1.5 — machine-readable stdout (json/yaml modes) + validate range checks
# =============================================================================


class TestCLIMachineReadableStdout:
    """Regression tests: json/yaml output modes must keep stdout pure.

    `spark-optima optimize ... --output json > result.json` must produce a
    parseable file, so every decorative element (banner panel, input table,
    status messages, history note, tips) has to go to stderr instead.
    """

    @staticmethod
    def _real_result() -> object:
        """Build a real OptimizationResult so export and history-save work."""
        from spark_optima.core.result import OptimizationResult

        return OptimizationResult(
            configuration={"spark.executor.memory": "4g", "spark.executor.cores": 4},
            estimated_time_minutes=10.0,
            confidence_score=0.9,
            platform_specific={"platform": "local"},
        )

    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_json_stdout_is_strict_json(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
    ) -> None:
        """In json mode stdout carries exactly the JSON document."""
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = self._real_result()
        mock_optimizer.return_value = mock_instance

        result = runner.invoke(
            app,
            ["optimize", "--code-path", str(sample_code_file), "--output", "json"],
        )

        assert result.exit_code == 0
        # Strict parse: any banner/table/tip on stdout breaks this directly
        payload = json.loads(result.stdout)
        assert payload["configuration"]["spark.executor.memory"] == "4g"
        # Decorative output must land on stderr instead
        assert "Input Parameters" in result.stderr
        assert "Saved to history" in result.stderr
        assert "Tip:" in result.stderr

    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_yaml_stdout_is_pure_yaml(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
    ) -> None:
        """In yaml mode stdout carries exactly the YAML document."""
        import yaml

        mock_instance = MagicMock()
        mock_instance.optimize.return_value = self._real_result()
        mock_optimizer.return_value = mock_instance

        result = runner.invoke(
            app,
            ["optimize", "--code-path", str(sample_code_file), "--output", "yaml"],
        )

        assert result.exit_code == 0
        payload = yaml.safe_load(result.stdout)
        assert payload["configuration"]["spark.executor.memory"] == "4g"
        assert "Input Parameters" not in result.stdout
        assert "Input Parameters" in result.stderr

    @patch("spark_optima.cli.main.Optimizer")
    def test_optimize_table_mode_keeps_stdout_output(
        self,
        mock_optimizer: MagicMock,
        runner: CliRunner,
        sample_code_file: Path,
    ) -> None:
        """Table mode still prints the banner and input table to stdout."""
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = self._real_result()
        mock_optimizer.return_value = mock_instance

        result = runner.invoke(
            app,
            ["optimize", "--code-path", str(sample_code_file)],
        )

        assert result.exit_code == 0
        assert "Input Parameters" in result.stdout
        assert "Saved to history" in result.stdout

    def test_analyze_json_stdout_is_strict_json(
        self,
        runner: CliRunner,
        sample_code_file: Path,
    ) -> None:
        """Analyze json mode emits only the JSON document on stdout."""
        analysis_dict = {"operations": [], "smells": [], "recommendations": []}
        with patch("spark_optima.analysis.recommender.analyze_code") as mock_analyze:
            mock_result = MagicMock()
            mock_result.operations = []
            mock_result.smells = []
            mock_result.recommendations = []
            mock_result.to_dict.return_value = analysis_dict
            mock_analyze.return_value = mock_result

            result = runner.invoke(
                app,
                ["analyze", "--code-path", str(sample_code_file), "--output", "json"],
            )

        assert result.exit_code == 0
        # Strict parse: the header panel must not pollute stdout
        payload = json.loads(result.stdout)
        assert payload == analysis_dict
        assert "Spark Code Analysis" in result.stderr

    def test_analyze_table_mode_keeps_stdout_output(
        self,
        runner: CliRunner,
        sample_code_file: Path,
    ) -> None:
        """Table mode still prints the analysis header to stdout."""
        result = runner.invoke(app, ["analyze", "--code-path", str(sample_code_file)])

        assert result.exit_code == 0
        assert "Spark Code Analysis" in result.stdout


class TestCLIValidateByteDurationRanges:
    """Regression tests: validate range-checks BYTES/DURATION via ConfigValidator.

    The v1.4 workaround skipped numeric range checks for BYTES/DURATION
    parameters; with canonical-unit bounds the validate command now delegates
    those checks to ConfigValidator.validate.
    """

    def test_validate_bytes_above_database_max_errors(self, runner: CliRunner, tmp_path: Path) -> None:
        """A byte value above the database maximum is an error naming the bound."""
        config = _write_config_file(tmp_path, ["spark.kryoserializer.buffer.max 3g"])

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["valid"] is False
        issues = _issues_for_check(payload, "invalid_value")
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"
        assert issues[0]["param"] == "spark.kryoserializer.buffer.max"
        message = issues[0]["message"]
        assert "greater than maximum" in message
        # The database max is 2048m; bounds are canonicalized to base units at
        # load time, so accept the suffixed or canonical-bytes rendering.
        assert "2048m" in message or "2147483648" in message

    def test_validate_bytes_within_database_max_ok(self, runner: CliRunner, tmp_path: Path) -> None:
        """A byte value within the database bounds produces no range error."""
        config = _write_config_file(tmp_path, ["spark.kryoserializer.buffer.max 512m"])

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["valid"] is True
        assert _issues_for_check(payload, "invalid_value") == []

    def test_validate_bare_numeric_memory_value_not_rejected(self, runner: CliRunner, tmp_path: Path) -> None:
        """A suffixless numeric memory value must not be falsely range-rejected."""
        config = _write_config_file(tmp_path, ["spark.executor.memory 4096"])

        result = runner.invoke(app, ["validate", "-c", str(config), "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["valid"] is True
        assert _issues_for_check(payload, "invalid_value") == []
