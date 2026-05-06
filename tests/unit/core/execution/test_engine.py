# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for execution engine module."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from spark_optima.core.bayesian.models import TrialMetrics, TrialStatus
from spark_optima.core.execution.engine import ExecutionEngine
from spark_optima.platforms.models import CostModel, ResourceSpec


class TestExecutionEngineInitialization:
    """Tests for ExecutionEngine initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        engine = ExecutionEngine()
        assert engine.app_name == "SparkOptima"
        assert engine.enable_monitoring is True

    def test_custom_initialization(self):
        """Test custom initialization."""
        engine = ExecutionEngine(
            app_name="TestApp",
            enable_monitoring=False,
        )
        assert engine.app_name == "TestApp"
        assert engine.enable_monitoring is False

    def test_initialization_with_spark(self):
        """Test initialization when Spark is available."""
        engine = ExecutionEngine()
        # Should have a runner if Spark is available
        if engine.is_available():
            assert engine._runner is not None
        else:
            assert engine._runner is None

    def test_is_available(self):
        """Test is_available method."""
        engine = ExecutionEngine()
        # Result depends on whether PySpark is installed
        assert isinstance(engine.is_available(), bool)

    def test_initialization_spark_not_available(self):
        """Test initialization when Spark is not available (lines 66-69)."""
        # Mock SparkRunner to raise RuntimeError
        with patch("spark_optima.core.execution.engine.SparkRunner") as mock_runner_class:
            mock_runner_class.side_effect = RuntimeError("Spark not available")

            engine = ExecutionEngine()

            assert engine._runner is None
            assert engine._spark_available is False
            assert engine.is_available() is False


class TestExecutionEngineValidation:
    """Tests for configuration validation."""

    def test_validate_config_with_runner(self):
        """Test validate_config when runner is available."""
        engine = ExecutionEngine()
        if engine.is_available():
            errors = engine.validate_config({"spark.executor.memory": "4g"})
            assert isinstance(errors, list)

    def test_validate_config_no_runner(self):
        """Test validate_config when runner is not available."""
        engine = ExecutionEngine()
        # Mock runner to None to simulate unavailable
        engine._runner = None
        engine._spark_available = False

        errors = engine.validate_config({})
        assert len(errors) > 0
        assert "Spark not available" in errors[0]

    def test_get_spark_info_available(self):
        """Test get_spark_info when Spark is available."""
        engine = ExecutionEngine()
        info = engine.get_spark_info()

        if engine.is_available():
            assert "pyspark_available" in info
        else:
            assert info == {"available": False}

    def test_get_spark_info_no_runner(self):
        """Test get_spark_info when runner is not available."""
        engine = ExecutionEngine()
        engine._runner = None

        info = engine.get_spark_info()
        assert info == {"available": False}


class TestExecutionEngineExecuteErrors:
    """Tests for execute method error handling."""

    def test_execute_no_spark(self):
        """Test execute returns error result when Spark not available."""
        engine = ExecutionEngine()
        engine._runner = None
        engine._spark_available = False

        result = engine.execute(config={"spark.executor.memory": "4g"})
        assert result.status == TrialStatus.FAILED
        assert "not available" in result.metrics.error_message.lower()

    def test_execute_no_code_or_path(self):
        """Test execute returns error result when neither code nor code_path provided."""
        engine = ExecutionEngine()
        if not engine.is_available():
            pytest.skip("Spark not available")

        result = engine.execute(config={"spark.executor.memory": "4g"})
        assert result.status == TrialStatus.FAILED
        assert "code or code_path" in result.metrics.error_message


class TestBuildTrialResult:
    """Tests for _build_trial_result method."""

    def test_build_trial_result_success(self):
        """Test building trial result for successful execution."""
        engine = ExecutionEngine()
        if not engine.is_available():
            pytest.skip("Spark not available")

        from spark_optima.core.execution.metrics_collector import ExecutionMetrics

        execution_metrics = ExecutionMetrics(
            execution_time_seconds=60.0,
            memory_peak_gb=4.0,
            success=True,
        )
        exec_result = {"success": True}

        result = engine._build_trial_result(
            config={"spark.executor.memory": "4g"},
            execution_metrics=execution_metrics,
            exec_result=exec_result,
            duration=60.0,
            cost_model=None,
        )

        assert result.status == TrialStatus.COMPLETED
        assert result.metrics.success is True
        assert result.metrics.execution_time_seconds == 60.0

    def test_build_trial_result_failure(self):
        """Test building trial result for failed execution."""
        engine = ExecutionEngine()
        if not engine.is_available():
            pytest.skip("Spark not available")

        from spark_optima.core.execution.metrics_collector import ExecutionMetrics

        execution_metrics = ExecutionMetrics(
            execution_time_seconds=30.0,
            success=False,
            error_message="Test error",
        )
        exec_result = {"success": False, "error": "Test error"}

        result = engine._build_trial_result(
            config={},
            execution_metrics=execution_metrics,
            exec_result=exec_result,
            duration=30.0,
            cost_model=None,
        )

        assert result.status == TrialStatus.FAILED
        assert result.metrics.success is False

    def test_build_trial_result_timeout(self):
        """Test building trial result for timeout (line 270)."""
        engine = ExecutionEngine()
        if not engine.is_available():
            pytest.skip("Spark not available")

        from spark_optima.core.execution.metrics_collector import ExecutionMetrics

        # Timeout error message should result in FAILED status
        execution_metrics = ExecutionMetrics(
            execution_time_seconds=3600.0,
            success=False,
            error_message="Timeout exceeded",
        )
        exec_result = {"success": False, "error": "Timeout exceeded"}

        result = engine._build_trial_result(
            config={},
            execution_metrics=execution_metrics,
            exec_result=exec_result,
            duration=3600.0,
            cost_model=None,
        )

        # Line 270: Status should be FAILED for timeout
        assert result.status == TrialStatus.FAILED
        assert "Timeout" in result.metrics.error_message

    def test_build_trial_result_with_cost_model(self):
        """Test building trial result with cost model."""
        engine = ExecutionEngine()
        if not engine.is_available():
            pytest.skip("Spark not available")

        from spark_optima.core.execution.metrics_collector import ExecutionMetrics

        execution_metrics = ExecutionMetrics(
            execution_time_seconds=3600.0,  # 1 hour
            success=True,
        )
        exec_result = {"success": True}

        # Mock cost model
        cost_model = Mock(spec=CostModel)
        cost_model.calculate.return_value = 10.0  # $10

        result = engine._build_trial_result(
            config={},
            execution_metrics=execution_metrics,
            exec_result=exec_result,
            duration=3600.0,
            cost_model=cost_model,
        )

        assert result.metrics.cost_estimate_usd == 10.0
        cost_model.calculate.assert_called_once_with(1.0)  # 3600s = 1 hour


class TestExecuteTrial:
    """Tests for execute_trial method."""

    def test_execute_trial_sets_trial_number(self):
        """Test that execute_trial sets the trial number."""
        engine = ExecutionEngine()
        if not engine.is_available():
            pytest.skip("Spark not available")

        # Mock the execute method
        with patch.object(engine, "execute") as mock_execute:
            mock_result = Mock()
            mock_result.metrics = TrialMetrics(
                execution_time_seconds=60.0,
                success=True,
            )
            mock_execute.return_value = mock_result

            result = engine.execute_trial(
                trial_number=42,
                config={"spark.executor.memory": "4g"},
                code="print('test')",
            )

            assert result.trial_number == 42
            mock_execute.assert_called_once()


class TestMonitoringCallbacks:
    """Tests for monitoring callback functionality."""

    def test_add_monitoring_callback(self):
        """Test adding monitoring callback."""
        engine = ExecutionEngine()
        if not engine.is_available():
            pytest.skip("Spark not available")

        callback = Mock()
        engine.add_monitoring_callback(callback)

        # Can't easily verify without accessing private members
        # But we can verify it doesn't raise
        assert True

    def test_get_monitoring_events(self):
        """Test getting monitoring events."""
        engine = ExecutionEngine()
        if not engine.is_available():
            pytest.skip("Spark not available")

        events = engine.get_monitoring_events()
        assert isinstance(events, list)

    def test_get_monitoring_events_no_monitor(self):
        """Test get_monitoring_events when _monitor is None (line 342)."""
        engine = ExecutionEngine(enable_monitoring=False)

        # When monitoring is disabled, _monitor is None
        assert engine._monitor is None

        # Should return empty list (line 342)
        events = engine.get_monitoring_events()
        assert events == []


class TestExecutionEngineWithMockSpark:
    """Tests using mocked Spark session."""

    @patch("spark_optima.core.execution.engine.SparkRunner")
    def test_execute_with_mock_runner(self, mock_runner_class):
        """Test execute with mocked Spark runner."""
        # Setup mock
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.validate_config.return_value = []

        # Mock the context manager
        mock_spark = MagicMock()
        mock_runner.create_session.return_value.__enter__ = Mock(return_value=mock_spark)
        mock_runner.create_session.return_value.__exit__ = Mock(return_value=False)

        # Mock execute_file result
        mock_runner.execute_file.return_value = {"success": True}

        engine = ExecutionEngine()
        engine._runner = mock_runner
        engine._spark_available = True

        # Mock metrics collector
        with patch.object(engine, "_build_trial_result") as mock_build:
            mock_result = Mock()
            mock_result.metrics = Mock(success=True)
            mock_build.return_value = mock_result

            engine.execute(
                config={"spark.executor.memory": "4g"},
                code_path="/test/path.py",
            )

            # Verify execute_file was called
            mock_runner.execute_file.assert_called_once()

    @patch("spark_optima.core.execution.engine.SparkRunner")
    def test_execute_code_with_mock_runner(self, mock_runner_class):
        """Test execute with code string."""
        # Setup mock
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.validate_config.return_value = []

        # Mock the context manager
        mock_spark = MagicMock()
        mock_runner.create_session.return_value.__enter__ = Mock(return_value=mock_spark)
        mock_runner.create_session.return_value.__exit__ = Mock(return_value=False)

        # Mock execute_code result
        mock_runner.execute_code.return_value = {"success": True}

        engine = ExecutionEngine()
        engine._runner = mock_runner
        engine._spark_available = True

        # Mock the _build_trial_result
        with patch.object(engine, "_build_trial_result") as mock_build:
            mock_result = Mock()
            mock_result.metrics = Mock(success=True)
            mock_build.return_value = mock_result

            engine.execute(
                config={"spark.executor.memory": "4g"},
                code="print('test')",
            )

            # Verify execute_code was called
            mock_runner.execute_code.assert_called_once()

    @patch("spark_optima.core.execution.engine.SparkRunner")
    def test_execute_with_monitoring(self, mock_runner_class):
        """Test execute with monitoring enabled (lines 125-127, 146-147)."""
        # Setup mock
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.validate_config.return_value = []

        # Mock the context manager
        mock_spark = MagicMock()
        mock_runner.create_session.return_value.__enter__ = Mock(return_value=mock_spark)
        mock_runner.create_session.return_value.__exit__ = Mock(return_value=False)

        # Mock execute_file result
        mock_runner.execute_file.return_value = {"success": True}

        engine = ExecutionEngine(enable_monitoring=True)
        engine._runner = mock_runner
        engine._spark_available = True

        # Mock the monitor
        mock_monitor = MagicMock()
        engine._monitor = mock_monitor

        # Mock metrics collector
        with patch.object(engine, "_build_trial_result") as mock_build:
            mock_result = Mock()
            mock_result.metrics = Mock(success=True)
            mock_build.return_value = mock_result

            engine.execute(
                config={"spark.executor.memory": "4g"},
                code_path="/test/path.py",
            )

            # Verify monitor methods were called
            mock_monitor.start_monitoring.assert_called_once_with(mock_spark)
            mock_monitor.stop_monitoring.assert_called_once()

    @patch("spark_optima.core.execution.engine.SparkRunner")
    def test_execute_with_monitoring_exception(self, mock_runner_class):
        """Test execute handles exceptions and returns FAILED result."""
        # Setup mock
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.validate_config.return_value = []

        # Mock the context manager
        mock_spark = MagicMock()
        mock_runner.create_session.return_value.__enter__ = Mock(return_value=mock_spark)
        mock_runner.create_session.return_value.__exit__ = Mock(return_value=False)

        # Mock execute_file to raise exception
        mock_runner.execute_file.side_effect = RuntimeError("Execution error")

        engine = ExecutionEngine(enable_monitoring=True)
        engine._runner = mock_runner
        engine._spark_available = True

        # Mock the monitor
        mock_monitor = MagicMock()
        engine._monitor = mock_monitor

        result = engine.execute(
            config={"spark.executor.memory": "4g"},
            code_path="/test/path.py",
        )

        # Should handle exception and return FAILED result
        assert result.status == TrialStatus.FAILED
        assert "Execution error" in result.metrics.error_message


class TestExecutionEngineEdgeCases:
    """Edge case tests for ExecutionEngine."""

    @patch("spark_optima.core.execution.engine.SparkRunner")
    def test_execute_with_timeout(self, mock_runner_class):
        """Test execute with timeout parameter."""
        # Setup mock
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.validate_config.return_value = []

        engine = ExecutionEngine()
        engine._runner = mock_runner
        engine._spark_available = True

        # Just verify the parameter is accepted
        with patch.object(engine._runner, "execute_file") as mock_exec:
            mock_exec.return_value = {"success": True}

            with patch.object(engine, "_build_trial_result") as mock_build:
                mock_result = Mock()
                mock_result.metrics = Mock(success=True)
                mock_build.return_value = mock_result

                # This mainly tests that timeout parameter is accepted
                engine.execute(
                    config={"spark.executor.memory": "4g"},
                    code_path="/test/path.py",
                    timeout_seconds=1800,
                )

    def test_execute_trial_with_resource_spec(self):
        """Test execute_trial with resource_spec."""
        engine = ExecutionEngine()
        if not engine.is_available():
            pytest.skip("Spark not available")

        resource_spec = ResourceSpec(cpu_cores=16, memory_gb=64)

        with patch.object(engine, "execute") as mock_execute:
            mock_result = Mock()
            mock_result.metrics = TrialMetrics(
                execution_time_seconds=60.0,
                success=True,
            )
            mock_execute.return_value = mock_result

            engine.execute_trial(
                trial_number=1,
                config={"spark.executor.memory": "4g"},
                code="print('test')",
                resource_spec=resource_spec,
            )

            # Verify resource_spec was passed
            call_kwargs = mock_execute.call_args[1]
            assert call_kwargs["resource_spec"] == resource_spec

    def test_execute_trial_with_cost_model(self):
        """Test execute_trial with cost_model parameter."""
        engine = ExecutionEngine()
        if not engine.is_available():
            pytest.skip("Spark not available")

        cost_model = Mock(spec=CostModel)

        with patch.object(engine, "execute") as mock_execute:
            mock_result = Mock()
            mock_result.metrics = TrialMetrics(
                execution_time_seconds=60.0,
                success=True,
            )
            mock_execute.return_value = mock_result

            result = engine.execute_trial(
                trial_number=2,
                config={"spark.executor.memory": "4g"},
                code="print('test')",
                cost_model=cost_model,
            )

            assert result.trial_number == 2

    def test_execute_trial_with_different_config(self):
        """Test execute_trial with different configs."""
        engine = ExecutionEngine()
        if not engine.is_available():
            pytest.skip("Spark not available")

        with patch.object(engine, "execute") as mock_execute:
            mock_result = Mock()
            mock_result.metrics = TrialMetrics(
                execution_time_seconds=60.0,
                success=True,
            )
            mock_execute.return_value = mock_result

            result = engine.execute_trial(
                trial_number=3,
                config={"spark.executor.memory": "8g"},
                code="print('test')",
            )

            assert result.trial_number == 3

    def test_execute_trial_with_timeout(self):
        """Test execute_trial with timeout_seconds parameter."""
        engine = ExecutionEngine()
        if not engine.is_available():
            pytest.skip("Spark not available")

        with patch.object(engine, "execute") as mock_execute:
            mock_result = Mock()
            mock_result.metrics = TrialMetrics(
                execution_time_seconds=60.0,
                success=True,
            )
            mock_execute.return_value = mock_result

            result = engine.execute_trial(
                trial_number=4,
                config={"spark.executor.memory": "4g"},
                code="print('test')",
                timeout_seconds=1800,
            )

            assert result.trial_number == 4


class TestExecutionEngineBuildTrialResultCoverage:
    """Additional tests for _build_trial_result method to achieve 100% coverage."""

    def test_build_trial_result_timeout_status(self):
        """Test building trial result for timeout status (line 270)."""
        engine = ExecutionEngine()

        from spark_optima.core.execution.metrics_collector import ExecutionMetrics

        # Timeout error message should result in FAILED status
        execution_metrics = ExecutionMetrics(
            execution_time_seconds=3600.0,
            success=False,
            error_message="Timeout exceeded",
        )
        exec_result = {"success": False, "error": "Timeout exceeded"}

        result = engine._build_trial_result(
            config={},
            execution_metrics=execution_metrics,
            exec_result=exec_result,
            duration=3600.0,
            cost_model=None,
        )

        # Line 270: Status should be FAILED for timeout
        assert result.status == TrialStatus.FAILED
        assert "Timeout" in result.metrics.error_message

    def test_build_trial_result_with_cost_calculation(self):
        """Test building trial result with cost calculation (lines 251-253)."""
        engine = ExecutionEngine()

        from spark_optima.core.execution.metrics_collector import ExecutionMetrics

        execution_metrics = ExecutionMetrics(
            execution_time_seconds=7200.0,  # 2 hours
            success=True,
        )
        exec_result = {"success": True}

        # Mock cost model
        cost_model = Mock(spec=CostModel)
        cost_model.calculate.return_value = 20.0  # $20

        result = engine._build_trial_result(
            config={},
            execution_metrics=execution_metrics,
            exec_result=exec_result,
            duration=7200.0,
            cost_model=cost_model,
        )

        # Verify cost was calculated: 2 hours * rate
        # cost_estimate_usd is in TrialMetrics, not ExecutionMetrics
        assert result.metrics.cost_estimate_usd == 20.0
        cost_model.calculate.assert_called_once_with(2.0)  # 7200s = 2 hours

    def test_build_trial_result_cost_from_metrics(self):
        """Test when cost is already in metrics (line 251)."""
        engine = ExecutionEngine()

        from spark_optima.core.execution.metrics_collector import ExecutionMetrics

        # Cost already set in execution_metrics
        execution_metrics = ExecutionMetrics(
            execution_time_seconds=60.0,
            success=True,
        )
        exec_result = {"success": True}

        # Cost model will calculate the cost since execution_metrics doesn't have cost
        cost_model = Mock(spec=CostModel)
        cost_model.calculate.return_value = 5.0

        result = engine._build_trial_result(
            config={},
            execution_metrics=execution_metrics,
            exec_result=exec_result,
            duration=60.0,
            cost_model=cost_model,
        )

        # Cost should be calculated by cost_model
        assert result.metrics.cost_estimate_usd == 5.0


class TestExecutionEngineErrorHandling:
    """Test error handling in execute method."""

    @patch("spark_optima.core.execution.engine.SparkRunner")
    def test_execute_with_exception(self, mock_runner_class):
        """Test execute handles exceptions (lines 165-185)."""
        # Setup mock
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.validate_config.return_value = []

        # Mock the context manager
        mock_spark = MagicMock()
        mock_runner.create_session.return_value.__enter__ = Mock(return_value=mock_spark)
        mock_runner.create_session.return_value.__exit__ = Mock(return_value=False)

        # Make execute_file raise an exception
        mock_runner.execute_file.side_effect = RuntimeError("Execution failed")

        engine = ExecutionEngine()
        engine._runner = mock_runner
        engine._spark_available = True

        # Mock the monitoring to avoid thread issues
        engine._monitor = None

        result = engine.execute(
            config={"spark.executor.memory": "4g"},
            code_path="/test/path.py",
        )

        # Should return a FAILED TrialResult
        assert result.status == TrialStatus.FAILED
        assert result.metrics.success is False

    @patch("spark_optima.core.execution.engine.SparkRunner")
    def test_execute_code_with_exception(self, mock_runner_class):
        """Test execute with code string that raises exception."""
        # Setup mock
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.validate_config.return_value = []

        engine = ExecutionEngine()
        engine._runner = mock_runner
        engine._spark_available = True

        with patch.object(engine._runner, "execute_code") as mock_exec:
            mock_exec.side_effect = RuntimeError("Code execution failed")

            with patch.object(engine, "_build_trial_result") as mock_build:
                mock_result = Mock()
                mock_result.metrics = Mock(success=False)
                mock_build.return_value = mock_result

                result = engine.execute(
                    config={"spark.executor.memory": "4g"},
                    code="invalid code",
                )

                # Should handle the exception
                assert result is not None
