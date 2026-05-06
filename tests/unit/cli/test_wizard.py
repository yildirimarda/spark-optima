# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the CLI wizard.

This module contains tests for the interactive configuration wizard.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from spark_optima.cli.wizard import ConfigurationWizard, run_wizard


class TestConfigurationWizardInitialization:
    """Test cases for ConfigurationWizard initialization."""

    def test_wizard_initialization(self) -> None:
        """Test wizard initialization."""
        wizard = ConfigurationWizard()
        assert wizard.config == {}
        assert wizard.current_step == 0
        assert wizard.TOTAL_STEPS == 6


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

    @patch("spark_optima.cli.wizard.FloatPrompt.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_data_profile(
        self,
        mock_prompt: MagicMock,
        mock_float_prompt: MagicMock,
        wizard: ConfigurationWizard,
    ) -> None:
        """Test data profile step."""
        # Mock user inputs
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
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_optimization_settings")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._print_summary")
    @patch("spark_optima.cli.wizard.Confirm.ask")
    def test_run_wizard_confirmed(
        self,
        mock_confirm: MagicMock,
        mock_print_summary: MagicMock,
        mock_step_opt_settings: MagicMock,
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
    @patch("spark_optima.cli.wizard.ConfigurationWizard._step_optimization_settings")
    @patch("spark_optima.cli.wizard.ConfigurationWizard._print_summary")
    @patch("spark_optima.cli.wizard.Confirm.ask")
    def test_run_wizard_cancelled(
        self,
        mock_confirm: MagicMock,
        mock_print_summary: MagicMock,
        mock_step_opt_settings: MagicMock,
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

    @patch("spark_optima.cli.wizard.FloatPrompt.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_data_profile_numeric_format(
        self,
        mock_prompt: MagicMock,
        mock_float_prompt: MagicMock,
    ) -> None:
        """Test data profile with numeric format choice."""
        wizard = ConfigurationWizard()
        # FloatPrompt.ask returns data size
        # Prompt.ask returns format choice "2" (selects delta), then compression "snappy"
        mock_float_prompt.return_value = 10.0
        mock_prompt.side_effect = ["2", "snappy"]

        wizard._step_data_profile()

        assert wizard.config["data_profile"]["size_gb"] == 10.0
        assert wizard.config["data_profile"]["format"] == "delta"

    @patch("spark_optima.cli.wizard.FloatPrompt.ask")
    @patch("spark_optima.cli.wizard.Prompt.ask")
    def test_step_data_profile_with_compression(
        self,
        mock_prompt: MagicMock,
        mock_float_prompt: MagicMock,
    ) -> None:
        """Test data profile with compression."""
        wizard = ConfigurationWizard()
        # FloatPrompt.ask returns data size (10.0)
        # Prompt.ask returns format ("parquet") and compression ("gzip")
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
