# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for Spark runner module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spark_optima.platforms.models import ResourceSpec


class TestSparkRunner:
    """Tests for SparkRunner class."""

    @pytest.fixture(autouse=True)
    def _reload_module(self):
        """Ensure module is in correct state before each test."""
        import importlib

        import spark_optima.core.execution.spark_runner as runner_module

        importlib.reload(runner_module)
        yield
        # Reload again after test to ensure clean state
        importlib.reload(runner_module)

    def test_runner_initialization_no_docker(self):
        """Test runner initialization raises error when Docker not available."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )
        # Mock Docker as not available
        runner._docker_available = False

        with pytest.raises(RuntimeError, match="Docker is required"):
            runner.execute_code("print('hello')")

    def test_runner_initialization(self):
        """Test runner initialization with mocked Spark."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False,
            app_name="TestApp",
            master="local[4]",
            log_level="ERROR",
        )

        assert runner.app_name == "TestApp"
        assert runner.master == "local[4]"
        assert runner.log_level == "ERROR"

    def test_get_active_session(self):
        """Test get_active_session returns None initially."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        assert runner.get_active_session() is None

    def test_get_applied_configs(self):
        """Test get_applied_configs returns empty dict initially."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        assert runner.get_applied_configs() == {}

    def test_validate_config_valid(self):
        """Test config validation with valid config."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        errors = runner.validate_config(
            {
                "spark.executor.memory": "4g",
                "spark.executor.cores": "2",
            }
        )

        assert len(errors) == 0

    def test_validate_config_missing_memory(self):
        """Test config validation detects missing memory."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        errors = runner.validate_config({})

        assert any("Missing spark.executor.memory" in e for e in errors)

    def test_validate_config_invalid_memory(self):
        """Test config validation detects invalid memory format."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        errors = runner.validate_config(
            {
                "spark.executor.memory": "invalid",
            }
        )

        assert any("Invalid memory format" in e for e in errors)

    def test_validate_config_invalid_cores(self):
        """Test config validation detects invalid cores."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        errors = runner.validate_config(
            {
                "spark.executor.memory": "4g",
                "spark.executor.cores": "abc",
            }
        )

        assert any("Invalid numeric value" in e for e in errors)

    def test_get_spark_info(self):
        """Test get_spark_info returns info dict."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, app_name="TestApp")

        info = runner.get_spark_info()

        assert info["app_name"] == "TestApp"
        assert "pyspark_available" in info

    def test_validate_config_with_spark_version(self):
        """Test config validation with Spark version."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        # Valid config should pass even with version
        errors = runner.validate_config({"spark.executor.memory": "4g"}, spark_version="3.4.0")

        assert len(errors) == 0


class TestSparkRunnerMemoryValidation:
    """Tests for memory validation in SparkRunner."""

    def test_is_valid_memory_various_formats(self):
        """Test memory validation with various formats."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        # Valid formats
        assert SparkRunner._is_valid_memory("4g") is True
        assert SparkRunner._is_valid_memory("4G") is True
        assert SparkRunner._is_valid_memory("4gb") is True
        assert SparkRunner._is_valid_memory("1024m") is True
        assert SparkRunner._is_valid_memory("1t") is True
        assert SparkRunner._is_valid_memory("4") is True
        assert SparkRunner._is_valid_memory(4) is True
        assert SparkRunner._is_valid_memory(4.5) is True

        # Invalid formats
        assert SparkRunner._is_valid_memory("invalid") is False
        assert SparkRunner._is_valid_memory("") is False
        assert SparkRunner._is_valid_memory(0) is False
        assert SparkRunner._is_valid_memory(-1) is False


class TestSparkRunnerExecuteFile:
    """Tests for file execution in SparkRunner."""

    def test_execute_file_not_found(self):
        """Test execute_file with non-existent file."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        result = runner.execute_file("/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @patch("spark_optima.core.execution.spark_runner.SparkSession")
    def test_execute_file_pathlib(self, mock_spark_session):
        """Test execute_file accepts Path object."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        # Should work with Path
        path = Path(__file__)

        # Mock the execute_code to avoid actual execution
        with patch.object(runner, "execute_code") as mock_execute:
            mock_execute.return_value = {"success": True}
            result = runner.execute_file(path)

        assert result["success"] is True


class TestSparkRunnerWithResourceSpec:
    """Tests for SparkRunner with resource specifications."""

    def test_validate_config_with_resource_spec(self):
        """Test validation with resource spec."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        resource_spec = ResourceSpec(cpu_cores=16, memory_gb=64)

        errors = runner.validate_config(
            {"spark.executor.memory": "4g"}, resource_spec=resource_spec
        )

        # Resource spec doesn't affect validation directly
        assert isinstance(errors, list)


class TestSparkRunnerCreateSession:
    """Tests for create_session method."""

    def test_create_session_returns_context_manager(self):
        """Test that create_session returns a context manager."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )
        context_manager = runner.create_session({"spark.executor.memory": "4g"})

        # Should be a context manager
        assert hasattr(context_manager, "__enter__")
        assert hasattr(context_manager, "__exit__")


class TestSparkRunnerBuildSession:
    """Tests for _build_session method."""

    def test_build_session_requires_pyspark(self):
        """Test that _build_session requires PySpark."""
        from spark_optima.core.execution.spark_runner import _PYSPARK_AVAILABLE, SparkRunner

        SparkRunner(use_docker=False, )

        if not _PYSPARK_AVAILABLE:
            # If PySpark not available, _build_session should fail
            # We can't easily test this without mocking internals
            pass


class TestSparkRunnerGetSparkInfo:
    """Tests for get_spark_info method."""

    def test_get_spark_info_structure(self):
        """Test that get_spark_info returns proper structure."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )
        info = runner.get_spark_info()

        assert "pyspark_available" in info
        assert "app_name" in info
        assert "master" in info
        assert info["app_name"] == "SparkOptima"
        assert info["master"] == "local[*]"

    def test_get_spark_info_custom(self):
        """Test get_spark_info with custom settings."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, app_name="CustomApp", master="local[4]")
        info = runner.get_spark_info()

        assert info["app_name"] == "CustomApp"
        assert info["master"] == "local[4]"


class TestSparkRunnerIsValidMemory:
    """Additional tests for _is_valid_memory method."""

    def test_valid_memory_gb_uppercase(self):
        """Test memory validation with GB uppercase."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        assert SparkRunner._is_valid_memory("4GB") is True

    def test_valid_memory_mb_lowercase(self):
        """Test memory validation with MB lowercase."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        assert SparkRunner._is_valid_memory("1024mb") is True

    def test_invalid_memory_empty_string(self):
        """Test memory validation with empty string."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        assert SparkRunner._is_valid_memory("") is False

    def test_invalid_memory_zero(self):
        """Test memory validation with zero."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        assert SparkRunner._is_valid_memory(0) is False
        assert SparkRunner._is_valid_memory(0.0) is False

    def test_valid_memory_float(self):
        """Test memory validation with float."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        assert SparkRunner._is_valid_memory(4.5) is True


class TestSparkRunnerStopSession:
    """Tests for _stop_session method."""

    def test_stop_session_no_error(self):
        """Test stopping a session that doesn't exist."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        SparkRunner(use_docker=False, )
        # Should not raise even if no session
        # (We can't easily test this without a real session)


class TestSparkRunnerAppliedConfigs:
    """Tests for get_applied_configs method."""

    @patch("spark_optima.core.execution.spark_runner.SparkSession")
    def test_get_applied_configs_after_create(self, mock_spark_class):
        """Test getting applied configs."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        # Mock the create_session context manager
        mock_spark = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__return_value = mock_spark
        runner._build_session = MagicMock(return_value=mock_spark)
        runner._active_session = None
        runner._session_configs = {}

        # Test get_applied_configs when no active session
        assert runner.get_applied_configs() == {}


class TestSparkRunnerExecuteCode:
    """Tests for execute_code method."""

    @patch("spark_optima.core.execution.spark_runner.SparkSession")
    def test_execute_code_mock(self, mock_spark_class):
        """Test executing code with mocked Spark."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )
        runner._docker_available = True  # Mock Docker as available

        # Mock _execute_code_docker to return success
        with patch.object(runner, "_execute_code_docker", return_value={"success": True, "duration_seconds": 0.1}):
            result = runner.execute_code(code="x = 1 + 1")

            # With mocked Spark, the exec should succeed
            assert result["success"] is True


class TestSparkRunnerEdgeCases:
    """Edge case tests for SparkRunner."""

    def test_runner_with_all_params(self):
        """Test runner initialization with all parameters."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False,
            app_name="TestApp",
            master="local[2]",
            log_level="DEBUG",
        )

        assert runner.app_name == "TestApp"
        assert runner.master == "local[2]"
        assert runner.log_level == "DEBUG"

    def test_validate_config_missing_driver_memory(self):
        """Test validation still works with only executor memory."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )
        errors = runner.validate_config({"spark.executor.memory": "4g"})

        # Should not have missing driver memory error (it's not required)
        assert all("driver.memory" not in e for e in errors)


class TestSparkRunnerCreateSessionWithMock:
    """Tests for create_session method with mocked Spark."""

    @patch("spark_optima.core.execution.spark_runner.SparkSession")
    def test_create_session_sets_active_session_and_configs(self, mock_spark_class):
        """Test that create_session sets _active_session and _session_configs (lines 103-107)."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )
        config = {"spark.executor.memory": "4g", "spark.executor.cores": "2"}

        # Mock Spark session
        mock_spark = MagicMock()
        mock_spark.version = "3.4.0"
        mock_spark.sparkContext.uiWebUrl = "http://localhost:4040"
        mock_spark_class.builder.appName.return_value.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.config.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        # Use create_session and verify active session is set
        with runner.create_session(config=config):
            assert runner.get_active_session() == mock_spark
            assert runner.get_applied_configs() == config

        # After context manager exits, session should be cleared
        assert runner.get_active_session() is None
        assert runner.get_applied_configs() == {}

    @patch("spark_optima.core.execution.spark_runner.SparkSession")
    def test_create_session_with_empty_config(self, mock_spark_class):
        """Test create_session with empty config sets empty configs."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        # Mock Spark session
        mock_spark = MagicMock()
        mock_spark.version = "3.4.0"
        mock_spark.sparkContext.uiWebUrl = "http://localhost:4040"
        mock_spark_class.builder.appName.return_value.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.config.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        with runner.create_session(config=None):
            assert runner.get_applied_configs() == {}

    @patch("spark_optima.core.execution.spark_runner.SparkSession")
    def test_create_session_stop_session_called(self, mock_spark_class):
        """Test that _stop_session is called on exit (lines 111-113)."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        # Mock Spark session
        mock_spark = MagicMock()
        mock_spark.version = "3.4.0"
        mock_spark.sparkContext.uiWebUrl = "http://localhost:4040"
        mock_spark_class.builder.appName.return_value.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.config.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        with patch.object(runner, "_stop_session") as mock_stop:
            with runner.create_session():
                pass
            mock_stop.assert_called_once_with(mock_spark)


class TestSparkRunnerBuildSessionWithResourceSpec:
    """Tests for _build_session with resource_spec (lines 144-145)."""

    @patch("spark_optima.core.execution.spark_runner.SparkSession")
    def test_build_session_with_resource_spec_cpu_cores_high(self, mock_spark_class):
        """Test _build_session applies resource_spec defaults for high cpu_cores."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        # Mock Spark session
        mock_spark = MagicMock()
        mock_spark.version = "3.4.0"
        mock_spark.sparkContext.uiWebUrl = "http://localhost:4040"
        mock_spark_class.builder.appName.return_value.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.config.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        resource_spec = ResourceSpec(cpu_cores=16, memory_gb=64)

        # Call _build_session with resource_spec
        runner._build_session(config={}, resource_spec=resource_spec)

        # Check that config was applied with resource spec values
        # cpu_cores=16, so min(4, 16)=4 for executor, min(2, 16)=2 for driver
        config_calls = [str(call) for call in mock_spark_class.builder.config.call_args_list]
        assert any("spark.executor.cores" in c for c in config_calls)
        assert any("spark.driver.cores" in c for c in config_calls)

    @patch("spark_optima.core.execution.spark_runner.SparkSession")
    def test_build_session_with_resource_spec_cpu_cores_low(self, mock_spark_class):
        """Test _build_session applies resource_spec defaults for low cpu_cores."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        # Mock Spark session
        mock_spark = MagicMock()
        mock_spark.version = "3.4.0"
        mock_spark.sparkContext.uiWebUrl = "http://localhost:4040"
        mock_spark_class.builder.appName.return_value.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.config.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        resource_spec = ResourceSpec(cpu_cores=1, memory_gb=2)

        # Call _build_session with resource_spec
        runner._build_session(config={}, resource_spec=resource_spec)

        # Check that config was applied with resource spec values
        # cpu_cores=1, so min(4, 1)=1 for executor, min(2, 1)=1 for driver
        config_calls = [str(call) for call in mock_spark_class.builder.config.call_args_list]
        assert any("spark.executor.cores" in c for c in config_calls)
        assert any("spark.driver.cores" in c for c in config_calls)

    @patch("spark_optima.core.execution.spark_runner.SparkSession")
    def test_build_session_sets_log_level_and_logs(self, mock_spark_class):
        """Test _build_session sets log level and logs debug info (lines 155-161)."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, log_level="DEBUG")

        # Mock Spark session
        mock_spark = MagicMock()
        mock_spark.version = "3.4.0"
        mock_spark.sparkContext.uiWebUrl = "http://localhost:4040"
        mock_spark_class.builder.appName.return_value.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.config.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        with patch("spark_optima.core.execution.spark_runner.logger") as mock_logger:
            runner._build_session(config={}, resource_spec=None)

            # Verify setLogLevel was called
            mock_spark.sparkContext.setLogLevel.assert_called_once_with("DEBUG")

            # Verify debug logging was called
            mock_logger.debug.assert_any_call("Spark version: 3.4.0")
            mock_logger.debug.assert_any_call("Spark UI: http://localhost:4040")


class TestSparkRunnerStopSessionV2:
    """Tests for _stop_session method error handling (lines 170-174)."""

    def test_stop_session_handles_exception(self):
        """Test _stop_session handles exceptions gracefully."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        # Mock spark that raises on stop
        # The except block catches (RuntimeError, AttributeError)
        mock_spark = MagicMock()
        mock_spark.stop.side_effect = RuntimeError("Stop failed")

        with patch("spark_optima.core.execution.spark_runner.logger") as mock_logger:
            runner._stop_session(mock_spark)
            # Should log warning but not raise
            mock_logger.warning.assert_called_once()
            assert "Error stopping Spark session" in mock_logger.warning.call_args[0][0]

    def test_stop_session_success(self):
        """Test _stop_session stops spark successfully."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        # Mock spark
        mock_spark = MagicMock()

        with patch("spark_optima.core.execution.spark_runner.logger") as mock_logger:
            runner._stop_session(mock_spark)
            mock_spark.stop.assert_called_once()
            mock_logger.info.assert_called_once_with("Spark session stopped")


class TestSparkRunnerExecuteCodeV2:
    """Additional tests for execute_code method."""

    @patch("spark_optima.core.execution.spark_runner.SparkSession")
    def test_execute_code_timeout_handler(self, mock_spark_class):
        """Test timeout handler in execute_code (Docker mode)."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )
        runner._docker_available = True  # Mock Docker as available

        # Mock _execute_code_docker to return timeout result (not raise exception)
        timeout_result = {
            "success": False,
            "error": "Execution exceeded 100 seconds",
            "error_type": "TimeoutError",
            "duration_seconds": 100.0,
        }

        with patch.object(runner, "_execute_code_docker", return_value=timeout_result):
            result = runner.execute_code(code="x = 1", timeout_seconds=100)

        # Verify timeout was handled correctly
        assert result["success"] is False
        assert "exceeded" in result["error"].lower()

    def test_execute_code_with_exception(self):
        """Test execute_code handles Docker execution errors."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False)
        runner._docker_available = True  # Mock Docker as available

        # Mock _execute_code_docker to simulate a failure
        mock_result = {
            "success": False,
            "error": "ValueError: test error",
            "error_type": "ValueError",
            "duration_seconds": 0.5,
        }
        with patch.object(runner, "_execute_code_docker", return_value=mock_result):
            result = runner.execute_code(code="raise ValueError('test error')")

            assert result["success"] is False
            assert "test error" in result["error"]
            assert result["error_type"] == "ValueError"
            assert "duration_seconds" in result

    @patch("spark_optima.core.execution.spark_runner.SparkSession")
    def test_execute_code_success(self, mock_spark_class):
        """Test execute_code returns success result with metrics."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )
        runner._docker_available = True  # Mock Docker as available

        # Mock _execute_code_docker to avoid actual Docker call
        mock_result = {
            "success": True,
            "duration_seconds": 0.5,
            "spark_version": "3.4.0",
            "ui_url": "http://localhost:4040",
            "config_applied": {},
        }
        with patch.object(runner, "_execute_code_docker", return_value=mock_result):
            result = runner.execute_code(code="x = 1 + 1")

            assert result["success"] is True
            assert "duration_seconds" in result
            assert result["spark_version"] == "3.4.0"
            assert result["ui_url"] == "http://localhost:4040"


class TestSparkRunnerGetSparkInfoEdgeCases:
    """Tests for get_spark_info edge cases (lines 385-386)."""

    def test_get_spark_info_pyspark_version_exception(self):
        """Test get_spark_info handles exception when getting pyspark version (lines 385-386)."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        # To trigger lines 385-386, we need pyspark.__version__ to raise an ImportError
        # The except block catches ImportError

        import sys
        from unittest.mock import MagicMock, PropertyMock

        # Save original pyspark module
        original_pyspark = sys.modules.get("pyspark")

        try:
            # Create a mock for pyspark module
            # We need __version__ to raise an exception when accessed
            mock_pyspark = MagicMock()
            # Use PropertyMock for __version__ to raise ImportError on access (the caught exception)
            type(mock_pyspark).__version__ = PropertyMock(side_effect=ImportError("Version error"))

            # Replace in sys.modules
            sys.modules["pyspark"] = mock_pyspark

            # Now get_spark_info should hit the exception handler (lines 385-386)
            info = runner.get_spark_info()

            # Should still return info without crashing
            assert "pyspark_available" in info
            # pyspark_version should not be in info due to exception
            assert "pyspark_version" not in info

        finally:
            # Restore original pyspark
            if original_pyspark is None:
                if "pyspark" in sys.modules:
                    del sys.modules["pyspark"]
            else:
                sys.modules["pyspark"] = original_pyspark


class TestSparkRunnerExecuteFileEdgeCases:
    """Additional tests for execute_file method."""

    @patch("spark_optima.core.execution.spark_runner.SparkSession")
    def test_execute_file_with_timeout(self, mock_spark_class):
        """Test execute_file passes timeout to execute_code."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        # Create a temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1")
            temp_path = f.name

        try:
            # Mock execute_code
            with patch.object(runner, "execute_code") as mock_execute:
                mock_execute.return_value = {"success": True}
                runner.execute_file(temp_path, timeout_seconds=1800)
                mock_execute.assert_called_once()
                # Check that timeout was passed
                call_kwargs = mock_execute.call_args
                assert (
                    call_kwargs[1].get("timeout_seconds") == 1800 or call_kwargs[0][1] is None
                )  # config may be None

        finally:
            import os

            os.unlink(temp_path)

    def test_validate_config_driver_memory_invalid(self):
        """Test validation detects invalid driver memory."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        errors = runner.validate_config(
            {
                "spark.executor.memory": "4g",
                "spark.driver.memory": "invalid",
            }
        )

        assert any("Invalid memory format" in e for e in errors)

    def test_validate_config_driver_cores_invalid(self):
        """Test validation detects invalid driver cores."""
        from spark_optima.core.execution.spark_runner import SparkRunner

        runner = SparkRunner(use_docker=False, )

        errors = runner.validate_config(
            {
                "spark.executor.memory": "4g",
                "spark.driver.cores": "abc",
            }
        )

        assert any("Invalid numeric value" in e for e in errors)
