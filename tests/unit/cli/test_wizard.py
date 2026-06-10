# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the CLI wizard.

This module contains tests for the interactive configuration wizard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from spark_optima.cli.wizard import ConfigurationWizard, run_wizard

if TYPE_CHECKING:
    from pathlib import Path


class TestConfigurationWizardInitialization:
    """Test cases for ConfigurationWizard initialization."""

    def test_wizard_initialization(self) -> None:
        """Test wizard initialization."""
        wizard = ConfigurationWizard()
        assert wizard.config == {}
        assert wizard.current_step == 0
        assert wizard.TOTAL_STEPS == 7


class TestConfigurationWizardSteps:
    """Test cases for wizard steps."""

    @pytest.fixture
    def wizard(self) -> ConfigurationWizard:
        """Create test wizard."""
        return ConfigurationWizard()

    @patch("spark_optima.cli.wizard.get_all_platforms_metadata")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_platform_selection(
        self,
        mock_prompt: MagicMock,
        mock_get_platforms: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test platform selection step."""
        # Mock platforms
        mock_get_platforms.return_value = {
            "local": {"display_name": "Local", "description": "Local machine"},
            "databricks": {"display_name": "Databricks", "description": "Databricks platform"},
        }
        # Mock user input
        mock_prompt.return_value = "1"  # Select local

        wizard._step_platform_selection()

        assert wizard.config["platform"] == "local"
        mock_prompt.assert_called_once()

    @patch("spark_optima.api.dependencies.get_optimization_service")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_spark_version(
        self,
        mock_prompt: MagicMock,
        mock_get_service: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test Spark version selection step."""
        # Mock service
        mock_service = MagicMock()
        mock_service.get_available_spark_versions.return_value = ["3.5.0", "3.4.0", "3.3.0"]
        mock_service.validate_spark_version.return_value = True
        mock_get_service.return_value = mock_service
        # Mock user input
        mock_prompt.return_value = "3.5.0"

        wizard._step_spark_version()

        assert wizard.config["spark_version"] == "3.5.0"
        mock_prompt.assert_called_once()

    @patch("spark_optima.cli.wizard.FloatPrompt.ask")
    @patch("spark_optima.cli.wizard.IntPrompt.ask")
    def test_step_resource_constraints(
        self,
        mock_int_prompt: MagicMock,
        mock_float_prompt: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test resource constraints step."""
        # Mock user inputs
        mock_int_prompt.return_value = 8
        mock_float_prompt.side_effect = [16.0, None, None]  # memory, max_memory, max_cost

        wizard._step_resource_constraints()

        assert wizard.config["resources"]["cpu_cores"] == 8
        assert wizard.config["resources"]["memory_gb"] == 16.0
        assert "constraints" not in wizard.config  # No optional constraints

    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.FloatPrompt.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_data_profile(
        self,
        mock_prompt: MagicMock,
        mock_float_prompt: MagicMock,
        mock_confirm: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test data profile step."""
        # Mock user inputs (no event log available)
        mock_confirm.return_value = False
        mock_float_prompt.return_value = 100.0
        mock_prompt.side_effect = ["parquet", ""]  # format, compression

        wizard._step_data_profile()

        assert wizard.config["data_profile"]["size_gb"] == 100.0
        assert wizard.config["data_profile"]["format"] == "parquet"
        assert "compression" not in wizard.config["data_profile"]

    @patch("spark_optima.cli.wizard.Path")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_spark_code(
        self,
        mock_prompt: MagicMock,
        mock_path: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test Spark code step."""
        # Mock path
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".py"
        mock_path_instance.absolute.return_value = "/path/to/code.py"
        mock_path.return_value = mock_path_instance
        # Mock user input
        mock_prompt.return_value = "/path/to/code.py"

        wizard._step_spark_code()

        assert wizard.config["code_path"] == "/path/to/code.py"
        mock_prompt.assert_called_once()

    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.IntPrompt.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_optimization_settings(
        self,
        mock_prompt: MagicMock,
        mock_int_prompt: MagicMock,
        mock_confirm: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test optimization settings step."""
        # Mock user inputs
        mock_confirm.return_value = True
        mock_int_prompt.return_value = 50
        mock_prompt.return_value = "table"

        wizard._step_optimization_settings()

        assert wizard.config["use_bayesian"] is True
        assert wizard.config["bayesian_trials"] == 50
        assert wizard.config["output_format"] == "table"


class TestConfigurationWizardIntegration:
    """Integration tests for the wizard."""

    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_platform_selection")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_spark_version")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_resource_constraints")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_data_profile")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_spark_code")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_objectives")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_optimization_settings")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._print_summary")
    @patch("spark_optima.cli.wizard.Confirm.ask")
    def test_run_wizard_confirmed(
        self,
        mock_confirm: MagicMock,
        mock_print_summary: MagicMock,
        mock_step_opt_settings: MagicMock,
        mock_step_objectives: MagicMock,
        mock_step_spark_code: MagicMock,
        mock_step_data_profile: MagicMock,
        mock_step_resource: MagicMock,
        mock_step_spark_version: MagicMock,
        mock_step_platform: MagicMock,
    ) -> None:
        """Test running wizard with confirmation."""
        wizard = ConfigurationWizard()
        mock_confirm.return_value = True

        result = wizard.run()

        # All steps should be called
        mock_step_platform.assert_called_once()
        mock_step_spark_version.assert_called_once()
        mock_step_resource.assert_called_once()
        mock_step_data_profile.assert_called_once()
        mock_step_spark_code.assert_called_once()
        mock_step_objectives.assert_called_once()
        mock_step_opt_settings.assert_called_once()
        mock_print_summary.assert_called_once()
        mock_confirm.assert_called_once()

        # Should return config (empty in this case since we mocked steps)
        assert isinstance(result, dict)

    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_platform_selection")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_spark_version")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_resource_constraints")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_data_profile")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_spark_code")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_objectives")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_optimization_settings")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._print_summary")
    @patch("spark_optima.cli.wizard.Confirm.ask")
    def test_run_wizard_cancelled(
        self,
        mock_confirm: MagicMock,
        mock_print_summary: MagicMock,
        mock_step_opt_settings: MagicMock,
        mock_step_objectives: MagicMock,
        mock_step_spark_code: MagicMock,
        mock_step_data_profile: MagicMock,
        mock_step_resource: MagicMock,
        mock_step_spark_version: MagicMock,
        mock_step_platform: MagicMock,
    ) -> None:
        """Test running wizard with cancellation."""
        wizard = ConfigurationWizard()
        mock_confirm.return_value = False

        result = wizard.run()

        # All steps should be called
        mock_step_platform.assert_called_once()
        mock_step_spark_version.assert_called_once()
        mock_step_resource.assert_called_once()
        mock_step_data_profile.assert_called_once()
        mock_step_spark_code.assert_called_once()
        mock_step_objectives.assert_called_once()
        mock_step_opt_settings.assert_called_once()
        mock_print_summary.assert_called_once()
        mock_confirm.assert_called_once()

        # Should return empty dict on cancellation
        assert result == {}


class TestRunWizardFunction:
    """Test cases for the run_wizard convenience function."""

    @patch("spark_optima.cli.wizard.ConfigurationWizard")
    def test_run_wizard(self, mock_wizard_class: MagicMock) -> None:
        """Test run_wizard function."""
        mock_wizard = MagicMock()
        mock_wizard.run.return_value = {"platform": "local"}
        mock_wizard_class.return_value = mock_wizard

        result = run_wizard()

        mock_wizard_class.assert_called_once()
        mock_wizard.run.assert_called_once()
        assert result == {"platform": "local"}


class TestConfigurationWizardEdgeCases:
    """Test edge cases for the wizard."""

    @patch("spark_optima.cli.wizard.get_all_platforms_metadata")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_platform_selection_invalid_then_valid(
        self,
        mock_prompt: MagicMock,
        mock_get_platforms: MagicMock,
    ) -> None:
        """Test platform selection with invalid then valid input."""
        wizard = ConfigurationWizard()
        mock_get_platforms.return_value = {
            "local": {"display_name": "Local", "description": "Local machine"},
        }
        # Mock user input: first invalid, then valid
        mock_prompt.side_effect = ["invalid", "1"]

        wizard._step_platform_selection()

        assert wizard.config["platform"] == "local"
        assert mock_prompt.call_count == 2

    @patch("spark_optima.cli.wizard.Path")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_spark_code_invalid_then_valid(
        self,
        mock_prompt: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Test Spark code step with invalid then valid file."""
        wizard = ConfigurationWizard()

        # Mock path for first call (invalid)
        mock_path_instance1 = MagicMock()
        mock_path_instance1.exists.return_value = False

        # Mock path for second call (valid)
        mock_path_instance2 = MagicMock()
        mock_path_instance2.exists.return_value = True
        mock_path_instance2.suffix = ".py"
        mock_path_instance2.absolute.return_value = "/path/to/code.py"

        mock_path.side_effect = [mock_path_instance1, mock_path_instance2]
        mock_prompt.side_effect = ["invalid.py", "valid.py"]

        wizard._step_spark_code()

        assert wizard.config["code_path"] == "/path/to/code.py"
        assert mock_prompt.call_count == 2

    @patch("spark_optima.api.dependencies.get_optimization_service")
    def test_step_spark_version_invalid(
        self,
        mock_get_service: MagicMock,
    ) -> None:
        """Test Spark version warning (lines 136-140)."""
        wizard = ConfigurationWizard()
        # Mock service with invalid version
        mock_service = MagicMock()
        mock_service.get_available_spark_versions.return_value = ["3.5.0"]
        mock_service.validate_spark_version.return_value = False
        mock_get_service.return_value = mock_service

        with patch("spark_optima.cli.wizard.Prompt.ask", return_value="4.0.0"):
            wizard._step_spark_version()

        # Should still set the version even if not valid
        assert wizard.config["spark_version"] == "4.0.0"

    @patch("spark_optima.cli.wizard.FloatPrompt.ask")
    @patch("spark_optima.cli.wizard.IntPrompt.ask")
    def test_step_resource_constraints_with_optional(
        self,
        mock_int_prompt: MagicMock,
        mock_float_prompt: MagicMock,
    ) -> None:
        """Test resource constraints with optional max_memory and max_cost."""
        wizard = ConfigurationWizard()
        # Mock user inputs: IntPrompt.ask returns 8 (cpu_cores)
        # FloatPrompt.ask returns 16.0 (memory), 32.0 (max_memory), 10.0 (max_cost)
        mock_int_prompt.side_effect = [8]
        mock_float_prompt.side_effect = [16.0, 32.0, 10.0]

        wizard._step_resource_constraints()

        assert wizard.config["resources"]["cpu_cores"] == 8
        assert wizard.config["resources"]["memory_gb"] == 16.0
        assert wizard.config["constraints"]["max_memory_gb"] == 32.0
        assert wizard.config["constraints"]["max_cost_per_hour"] == 10.0

    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.FloatPrompt.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_data_profile_numeric_format(
        self,
        mock_prompt: MagicMock,
        mock_float_prompt: MagicMock,
        mock_confirm: MagicMock,
    ) -> None:
        """Test data profile with numeric format choice."""
        wizard = ConfigurationWizard()
        # Confirm.ask declines the event log; FloatPrompt.ask returns data size
        # Prompt.ask returns format choice "2" (selects delta), then compression "snappy"
        mock_confirm.return_value = False
        mock_float_prompt.return_value = 10.0
        mock_prompt.side_effect = ["2", "snappy"]

        wizard._step_data_profile()

        assert wizard.config["data_profile"]["size_gb"] == 10.0
        assert wizard.config["data_profile"]["format"] == "delta"

    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.FloatPrompt.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_data_profile_with_compression(
        self,
        mock_prompt: MagicMock,
        mock_float_prompt: MagicMock,
        mock_confirm: MagicMock,
    ) -> None:
        """Test data profile with compression."""
        wizard = ConfigurationWizard()
        # Confirm.ask declines the event log; FloatPrompt.ask returns data size (10.0)
        # Prompt.ask returns format ("parquet") and compression ("gzip")
        mock_confirm.return_value = False
        mock_float_prompt.return_value = 10.0
        mock_prompt.side_effect = ["parquet", "gzip"]

        wizard._step_data_profile()

        assert wizard.config["data_profile"]["format"] == "parquet"
        assert wizard.config["data_profile"]["compression"] == "gzip"

    @patch("spark_optima.cli.wizard.Path")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_spark_code_not_py(
        self,
        mock_prompt: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Test Spark code step with non-.py file."""
        wizard = ConfigurationWizard()
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".txt"  # Not .py
        mock_path_instance.absolute.return_value = "/path/to/code.txt"
        mock_path.return_value = mock_path_instance
        mock_prompt.return_value = "/path/to/code.txt"

        wizard._step_spark_code()

        # Should not set code_path for non-.py file
        assert "code_path" not in wizard.config

    def test_print_summary(self) -> None:
        """Test _print_summary method."""
        wizard = ConfigurationWizard()
        wizard.config = {
            "platform": "local",
            "spark_version": "3.5.0",
            "resources": {"cpu_cores": 8, "memory_gb": 16.0},
            "data_profile": {"size_gb": 10.0, "format": "parquet"},
            "use_bayesian": True,
            "bayesian_trials": 50,
            "output_format": "table",
        }

        # Should not raise
        wizard._print_summary()


class TestObjectivesStep:
    """Test cases for the optimization objectives step."""

    @pytest.fixture
    def wizard(self) -> ConfigurationWizard:
        """Create test wizard."""
        return ConfigurationWizard()

    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_objectives_default(
        self,
        mock_prompt: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test objectives step with the default selection."""
        mock_prompt.return_value = "1"

        wizard._step_objectives()

        assert wizard.config["objectives"] == ["minimize_time"]
        mock_prompt.assert_called_once()

    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_objectives_multi_select_numbers_and_names(
        self,
        mock_prompt: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test multi-select mixing numeric and name tokens."""
        mock_prompt.return_value = "2, minimize_memory"

        wizard._step_objectives()

        assert wizard.config["objectives"] == ["minimize_cost", "minimize_memory"]

    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_objectives_invalid_then_valid(
        self,
        mock_prompt: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test that an unknown objective name triggers a re-prompt."""
        mock_prompt.side_effect = ["speed", "1,2"]

        wizard._step_objectives()

        assert wizard.config["objectives"] == ["minimize_time", "minimize_cost"]
        assert mock_prompt.call_count == 2

    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_objectives_deduplicates(
        self,
        mock_prompt: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test that duplicate selections collapse to one entry."""
        mock_prompt.return_value = "1,1,minimize_time"

        wizard._step_objectives()

        assert wizard.config["objectives"] == ["minimize_time"]

    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_objectives_blank_input_falls_back_to_default(
        self,
        mock_prompt: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test that blank/comma-only input falls back to the default objective."""
        mock_prompt.return_value = " , "

        wizard._step_objectives()

        assert wizard.config["objectives"] == ["minimize_time"]


class TestEventLogPrompt:
    """Test cases for the optional event-log enrichment prompt."""

    @pytest.fixture
    def wizard(self) -> ConfigurationWizard:
        """Create test wizard."""
        return ConfigurationWizard()

    @staticmethod
    def _make_summary(**overrides: object) -> MagicMock:
        """Create a mock EventLogSummary with canned tuning hints."""
        hints: dict[str, object] = {
            "data_size_gb": 42.0,
            "skew_factor": 2.5,
            "large_shuffles": True,
            "gc_pressure": True,
            "gc_time_fraction": 0.15,
            "spill_detected": True,
            "spill_gb": 1.5,
            "shuffle_total_gb": 12.0,
            "memory_intensive": True,
        }
        hints.update(overrides)
        summary = MagicMock()
        summary.to_tuning_hints.return_value = hints
        return summary

    @patch("spark_optima.cli.wizard.Confirm.ask")
    def test_event_log_declined(
        self,
        mock_confirm: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test that declining the event log prompt is a no-op."""
        mock_confirm.return_value = False

        assert wizard._prompt_event_log() is None
        assert "event_log" not in wizard.config
        assert "constraints" not in wizard.config

    @patch("spark_optima.cli.wizard.EventLogParser")
    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_event_log_success(
        self,
        mock_prompt: MagicMock,
        mock_confirm: MagicMock,
        mock_parser_cls: MagicMock,
        wizard: ConfigurationWizard,
        tmp_path: Path,
    ) -> None:
        """Test successful parse: size inferred and hints merged into constraints."""
        log_file = tmp_path / "eventlog.json"
        log_file.write_text("{}")
        mock_confirm.return_value = True
        mock_prompt.return_value = str(log_file)
        mock_parser_cls.return_value.parse.return_value = self._make_summary()

        inferred = wizard._prompt_event_log()

        assert inferred == 42.0
        assert wizard.config["event_log"] == str(log_file.absolute())
        assert wizard.config["constraints"]["skew_factor"] == 2.5
        assert wizard.config["constraints"]["memory_intensive"] is True
        # The size hint must not leak into resource constraints
        assert "data_size_gb" not in wizard.config["constraints"]
        assert "data size 42.0 GB" in wizard.config["event_log_inference"]

    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_event_log_missing_file_max_attempts(
        self,
        mock_prompt: MagicMock,
        mock_confirm: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test that repeated missing paths give up after max attempts."""
        mock_confirm.return_value = True
        mock_prompt.side_effect = ["/nope/one", "/nope/two", "/nope/three"]

        assert wizard._prompt_event_log() is None
        assert "event_log" not in wizard.config
        assert mock_prompt.call_count == 3

    @patch("spark_optima.cli.wizard.EventLogParser")
    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_event_log_missing_then_valid(
        self,
        mock_prompt: MagicMock,
        mock_confirm: MagicMock,
        mock_parser_cls: MagicMock,
        wizard: ConfigurationWizard,
        tmp_path: Path,
    ) -> None:
        """Test re-prompt after a missing file followed by a valid one."""
        log_file = tmp_path / "eventlog.json.gz"
        log_file.write_text("{}")
        mock_confirm.return_value = True
        mock_prompt.side_effect = ["/nope/missing", str(log_file)]
        mock_parser_cls.return_value.parse.return_value = self._make_summary()

        inferred = wizard._prompt_event_log()

        assert inferred == 42.0
        assert wizard.config["event_log"] == str(log_file.absolute())
        assert mock_prompt.call_count == 2

    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_event_log_skip_with_empty_input(
        self,
        mock_prompt: MagicMock,
        mock_confirm: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test that an empty path input skips the enrichment."""
        mock_confirm.return_value = True
        mock_prompt.return_value = ""

        assert wizard._prompt_event_log() is None
        assert "event_log" not in wizard.config
        mock_prompt.assert_called_once()

    @patch("spark_optima.cli.wizard.EventLogParser")
    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_event_log_parse_failure_warns_and_continues(
        self,
        mock_prompt: MagicMock,
        mock_confirm: MagicMock,
        mock_parser_cls: MagicMock,
        wizard: ConfigurationWizard,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that a parse failure warns and continues without enrichment."""
        log_file = tmp_path / "eventlog.json"
        log_file.write_text("not an event log")
        mock_confirm.return_value = True
        mock_prompt.return_value = str(log_file)
        mock_parser_cls.return_value.parse.side_effect = ValueError("corrupt log")

        assert wizard._prompt_event_log() is None
        assert "event_log" not in wizard.config
        assert "constraints" not in wizard.config
        assert "Could not parse event log" in capsys.readouterr().out

    @patch("spark_optima.cli.wizard.EventLogParser")
    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.FloatPrompt.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_data_profile_with_event_log(
        self,
        mock_prompt: MagicMock,
        mock_float_prompt: MagicMock,
        mock_confirm: MagicMock,
        mock_parser_cls: MagicMock,
        wizard: ConfigurationWizard,
        tmp_path: Path,
    ) -> None:
        """Test that the inferred size pre-fills the data size prompt."""
        log_file = tmp_path / "eventlog.json"
        log_file.write_text("{}")
        mock_confirm.return_value = True
        # Prompt.ask: event log path, data format, compression
        mock_prompt.side_effect = [str(log_file), "parquet", ""]
        mock_float_prompt.return_value = 42.0
        mock_parser_cls.return_value.parse.return_value = self._make_summary()

        wizard._step_data_profile()

        assert wizard.config["data_profile"]["size_gb"] == 42.0
        # The inferred size must be offered as the prompt default
        assert mock_float_prompt.call_args.kwargs["default"] == 42.0


class TestOutputAndSummary:
    """Test cases for export format rendering and the summary panel."""

    @pytest.fixture
    def wizard(self) -> ConfigurationWizard:
        """Create test wizard."""
        return ConfigurationWizard()

    def test_print_export_formats(
        self,
        wizard: ConfigurationWizard,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that the curated export formats are rendered."""
        wizard._print_export_formats()

        output = capsys.readouterr().out
        for name, _ in ConfigurationWizard.EXPORT_FORMATS:
            assert name in output
        assert "Also available" in output
        assert "azure-synapse" in output

    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.IntPrompt.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_optimization_settings_shows_export_formats(
        self,
        mock_prompt: MagicMock,
        mock_int_prompt: MagicMock,
        mock_confirm: MagicMock,
        wizard: ConfigurationWizard,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that the settings step prints trial guidance and export formats."""
        mock_confirm.return_value = True
        mock_int_prompt.return_value = 50
        mock_prompt.return_value = "table"

        wizard._step_optimization_settings()

        output = capsys.readouterr().out
        assert "Guidance" in output
        assert "spark-submit" in output
        assert wizard.config["bayesian_trials"] == 50

    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.IntPrompt.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_optimization_settings_multi_objective_trial_default(
        self,
        mock_prompt: MagicMock,
        mock_int_prompt: MagicMock,
        mock_confirm: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test that multi-objective selections raise the default trial count."""
        wizard.config["objectives"] = ["minimize_time", "minimize_cost"]
        mock_confirm.return_value = True
        mock_int_prompt.return_value = 100
        mock_prompt.return_value = "table"

        wizard._step_optimization_settings()

        assert mock_int_prompt.call_args.kwargs["default"] == 100

    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.IntPrompt.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_optimization_settings_no_bayesian(
        self,
        mock_prompt: MagicMock,
        mock_int_prompt: MagicMock,
        mock_confirm: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test that declining Bayesian optimization skips the trial prompt."""
        mock_confirm.return_value = False
        mock_prompt.return_value = "json"

        wizard._step_optimization_settings()

        assert wizard.config["use_bayesian"] is False
        assert "bayesian_trials" not in wizard.config
        mock_int_prompt.assert_not_called()

    def test_print_summary_with_objectives_and_event_log(
        self,
        wizard: ConfigurationWizard,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test summary rows for objectives and event-log inference."""
        wizard.config = {
            "platform": "local",
            "spark_version": "3.5.0",
            "resources": {"cpu_cores": 8, "memory_gb": 16.0},
            "data_profile": {"size_gb": 42.0, "format": "parquet"},
            "code_path": "/path/to/job.py",
            "objectives": ["minimize_time", "minimize_cost"],
            "event_log": "/path/to/app.log",
            "event_log_inference": "data size 42.0 GB, skew factor 2.5",
            "use_bayesian": True,
            "bayesian_trials": 100,
            "output_format": "table",
        }

        wizard._print_summary()

        output = capsys.readouterr().out
        assert "minimize_cost" in output
        assert "Event Log" in output
        assert "skew factor 2.5" in output
        assert "100 trials" in output
        # Non-default objectives surface the explicit handoff note
        assert "--objective" in output

    def test_print_summary_default_objective_has_no_note(
        self,
        wizard: ConfigurationWizard,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that the default objective does not trigger the handoff note."""
        wizard.config = {
            "platform": "local",
            "spark_version": "3.5.0",
            "objectives": ["minimize_time"],
        }

        wizard._print_summary()

        output = capsys.readouterr().out
        assert "minimize_time" in output
        assert "--objective" not in output


class TestFullWizardRun:
    """End-to-end wizard run with all prompts mocked."""

    @patch("spark_optima.cli.wizard.Path")
    @patch("spark_optima.cli.wizard.get_optimization_service")
    @patch("spark_optima.cli.wizard.get_all_platforms_metadata")
    @patch("spark_optima.cli.wizard.Confirm.ask")
    @patch("spark_optima.cli.wizard.IntPrompt.ask")
    @patch("spark_optima.cli.wizard.FloatPrompt.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_full_run_collects_all_config(
        self,
        mock_prompt: MagicMock,
        mock_float_prompt: MagicMock,
        mock_int_prompt: MagicMock,
        mock_confirm: MagicMock,
        mock_get_platforms: MagicMock,
        mock_get_service: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Test a full wizard run through every step."""
        mock_get_platforms.return_value = {
            "local": {"display_name": "Local", "description": "Local machine"},
        }
        mock_service = MagicMock()
        mock_service.get_available_spark_versions.return_value = ["3.5.0"]
        mock_service.validate_spark_version.return_value = True
        mock_get_service.return_value = mock_service

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".py"
        mock_path_instance.absolute.return_value = "/path/to/job.py"
        mock_path.return_value = mock_path_instance

        # Prompt.ask: platform, spark version, data format, compression,
        # code path, objectives, output format
        mock_prompt.side_effect = ["1", "3.5.0", "parquet", "", "/path/to/job.py", "1,2", "table"]
        # Confirm.ask: event log?, use bayesian?, final confirmation
        mock_confirm.side_effect = [False, True, True]
        # IntPrompt.ask: cpu cores, trials
        mock_int_prompt.side_effect = [4, 60]
        # FloatPrompt.ask: memory, max memory, max cost, data size
        mock_float_prompt.side_effect = [16.0, None, None, 10.0]

        config = ConfigurationWizard().run()

        assert config["platform"] == "local"
        assert config["spark_version"] == "3.5.0"
        assert config["resources"] == {"cpu_cores": 4, "memory_gb": 16.0}
        assert config["data_profile"] == {"size_gb": 10.0, "format": "parquet"}
        assert config["code_path"] == "/path/to/job.py"
        assert config["objectives"] == ["minimize_time", "minimize_cost"]
        assert config["use_bayesian"] is True
        assert config["bayesian_trials"] == 60
        assert config["output_format"] == "table"
