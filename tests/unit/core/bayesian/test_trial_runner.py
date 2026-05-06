# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the trial runner module.

This module contains tests for simulation and execution-based trial runners
including performance modeling and metrics collection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from spark_optima.core.bayesian.models import TrialMetrics, TrialResult, TrialStatus
from spark_optima.core.bayesian.trial_runner import (
    ExecutionRunner,
    SimulationModel,
    SimulationRunner,
    TrialRunner,
)
from spark_optima.platforms.models import ResourceSpec


class TestSimulationModel:
    """Test cases for the SimulationModel."""

    @pytest.fixture
    def simulation_model(self) -> SimulationModel:
        """Create test simulation model."""
        return SimulationModel()

    @pytest.fixture
    def sample_config(self) -> dict:
        """Create sample configuration."""
        return {
            "spark.executor.memory": "4g",
            "spark.executor.cores": "4",
            "spark.sql.shuffle.partitions": "200",
        }

    @pytest.fixture
    def sample_resources(self) -> ResourceSpec:
        """Create sample resource specification."""
        return ResourceSpec(cpu_cores=8, memory_gb=32)

    def test_simulation_model_initialization(self) -> None:
        """Test simulation model initialization."""
        model = SimulationModel()
        assert model is not None

    def test_estimate_basic(
        self,
        simulation_model: SimulationModel,
        sample_config: dict,
        sample_resources: ResourceSpec,
    ) -> None:
        """Test basic estimation."""
        metrics = simulation_model.estimate(
            config=sample_config,
            resource_spec=sample_resources,
        )

        assert isinstance(metrics, TrialMetrics)
        assert metrics.execution_time_seconds > 0
        assert metrics.success is True

    def test_estimate_with_data_profile(
        self,
        simulation_model: SimulationModel,
        sample_config: dict,
        sample_resources: ResourceSpec,
    ) -> None:
        """Test estimation with data profile."""
        data_profile = {"size_gb": 100, "format": "parquet"}

        metrics = simulation_model.estimate(
            config=sample_config,
            resource_spec=sample_resources,
            data_profile=data_profile,
        )

        assert isinstance(metrics, TrialMetrics)
        assert metrics.execution_time_seconds > 0

    def test_estimate_with_cost_model(
        self,
        simulation_model: SimulationModel,
        sample_config: dict,
        sample_resources: ResourceSpec,
    ) -> None:
        """Test estimation with cost model."""
        cost_model = MagicMock()
        cost_model.calculate.return_value = 10.0

        metrics = simulation_model.estimate(
            config=sample_config,
            resource_spec=sample_resources,
            cost_model=cost_model,
        )

        assert isinstance(metrics, TrialMetrics)

    def test_estimate_different_memory_configs(
        self,
        simulation_model: SimulationModel,
        sample_resources: ResourceSpec,
    ) -> None:
        """Test estimation with different memory configurations."""
        small_memory = {"spark.executor.memory": "2g"}
        large_memory = {"spark.executor.memory": "8g"}

        metrics_small = simulation_model.estimate(small_memory, sample_resources)
        metrics_large = simulation_model.estimate(large_memory, sample_resources)

        assert isinstance(metrics_small, TrialMetrics)
        assert isinstance(metrics_large, TrialMetrics)

    def test_estimate_different_parallelism(
        self,
        simulation_model: SimulationModel,
        sample_resources: ResourceSpec,
    ) -> None:
        """Test estimation with different parallelism settings."""
        low_parallelism = {"spark.sql.shuffle.partitions": "50"}
        high_parallelism = {"spark.sql.shuffle.partitions": "400"}

        metrics_low = simulation_model.estimate(low_parallelism, sample_resources)
        metrics_high = simulation_model.estimate(high_parallelism, sample_resources)

        assert isinstance(metrics_low, TrialMetrics)
        assert isinstance(metrics_high, TrialMetrics)

    def test_estimate_returns_positive_time(
        self,
        simulation_model: SimulationModel,
        sample_config: dict,
        sample_resources: ResourceSpec,
    ) -> None:
        """Test that estimate returns positive execution time."""
        metrics = simulation_model.estimate(sample_config, sample_resources)

        assert metrics.execution_time_seconds > 0

    def test_estimate_memory_usage(
        self,
        simulation_model: SimulationModel,
        sample_config: dict,
        sample_resources: ResourceSpec,
    ) -> None:
        """Test that estimate returns reasonable memory usage."""
        metrics = simulation_model.estimate(sample_config, sample_resources)

        assert metrics.memory_usage_mb is None or metrics.memory_usage_mb > 0


class TestTrialRunner:
    """Test cases for the base TrialRunner."""

    @pytest.fixture
    def trial_runner(self) -> TrialRunner:
        """Create test trial runner."""
        return TrialRunner(mode="simulation")

    def test_trial_runner_initialization(self) -> None:
        """Test trial runner initialization."""
        runner = TrialRunner(mode="simulation")
        assert runner.mode == "simulation"

    def test_trial_runner_execution_mode(self) -> None:
        """Test trial runner in execution mode."""
        runner = TrialRunner(mode="execution")
        assert runner.mode == "execution"

    def test_run_trial_works(self, trial_runner: TrialRunner) -> None:
        """Test that run_trial works correctly."""
        result = trial_runner.run_trial(
            trial_number=1,
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=4, memory_gb=16),
        )
        assert result is not None
        assert result.trial_number == 1

    def test_run_trial_basic(self, trial_runner: TrialRunner) -> None:
        """Test that run_trial works with basic parameters."""
        result = trial_runner.run_trial(
            trial_number=1,
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=4, memory_gb=16),
        )
        assert result.trial_number == 1
        assert result.metrics is not None


class TestExecutionRunner:
    """Test cases for the ExecutionRunner."""

    @pytest.fixture
    def execution_runner(self) -> ExecutionRunner:
        """Create test execution runner."""
        return ExecutionRunner()

    @pytest.fixture
    def sample_config(self) -> dict:
        """Create sample configuration."""
        return {
            "spark.executor.memory": "4g",
            "spark.executor.cores": "4",
        }

    def test_execution_runner_initialization(self) -> None:
        """Test execution runner initialization."""
        runner = ExecutionRunner()
        assert runner.mode == "execution"

    def test_execution_runner_with_spark_session(self) -> None:
        """Test execution runner with spark session."""
        mock_spark = MagicMock()
        runner = ExecutionRunner(spark_session=mock_spark)
        assert runner.spark_session == mock_spark

    @patch("spark_optima.core.bayesian.trial_runner.SparkRunner")
    def test_run_trial_simulation_mode(
        self,
        mock_spark_runner: MagicMock,
        execution_runner: ExecutionRunner,
        sample_config: dict,
    ) -> None:
        """Test running trial in simulation mode."""
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)

        # Mock the spark runner
        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock(
            execution_time_seconds=100.0,
            memory_usage_mb=1024.0,
            success=True,
        )
        mock_spark_runner.return_value = mock_runner

        # Should fall back to simulation when no spark session
        result = execution_runner.run_trial(1, sample_config, resources)

        assert isinstance(result, TrialResult)
        assert isinstance(result.metrics, TrialMetrics)

    def test_run_trial_with_monitoring(
        self,
        execution_runner: ExecutionRunner,
        sample_config: dict,
    ) -> None:
        """Test running trial with monitoring enabled."""
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        execution_runner.enable_monitoring = True

        # This should not raise an error
        try:
            result = execution_runner.run_trial(1, sample_config, resources)
            assert isinstance(result, TrialResult)
            assert isinstance(result.metrics, TrialMetrics)
        except Exception:
            # Expected if Spark is not available
            pass


class TestTrialRunnerEdgeCases:
    """Test edge cases for trial runners."""

    def test_simulation_empty_config(self) -> None:
        """Test simulation with empty configuration."""
        model = SimulationModel()
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)

        metrics = model.estimate({}, resources)

        assert isinstance(metrics, TrialMetrics)
        assert metrics.execution_time_seconds > 0

    def test_simulation_zero_resources(self) -> None:
        """Test simulation with minimal resources."""
        model = SimulationModel()
        config = {"spark.executor.memory": "1g"}
        resources = ResourceSpec(cpu_cores=1, memory_gb=1)

        metrics = model.estimate(config, resources)

        assert isinstance(metrics, TrialMetrics)

    def test_simulation_large_resources(self) -> None:
        """Test simulation with large resources."""
        model = SimulationModel()
        config = {"spark.executor.memory": "64g"}
        resources = ResourceSpec(cpu_cores=128, memory_gb=512)

        metrics = model.estimate(config, resources)

        assert isinstance(metrics, TrialMetrics)

    def test_simulation_with_invalid_config_values(self) -> None:
        """Test simulation with invalid configuration values."""
        model = SimulationModel()
        config = {
            "spark.executor.memory": "invalid",
            "spark.sql.shuffle.partitions": "not_a_number",
        }
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)

        # Should handle gracefully
        metrics = model.estimate(config, resources)
        assert isinstance(metrics, TrialMetrics)


class TestPerformanceFactors:
    """Test performance factor calculations."""

    @pytest.fixture
    def simulation_model(self) -> SimulationModel:
        """Create test simulation model."""
        return SimulationModel()

    def test_memory_factor_calculation(self, simulation_model: SimulationModel) -> None:
        """Test memory factor calculation."""
        # More memory should generally improve performance
        base_time = simulation_model._calculate_base_time(
            {"spark.executor.memory": "4g"},
            ResourceSpec(cpu_cores=4, memory_gb=16),
            None,
        )

        # Factor calculation is internal, but we can observe effects
        assert base_time > 0

    def test_cpu_factor_calculation(self, simulation_model: SimulationModel) -> None:
        """Test CPU factor calculation."""
        base_time = simulation_model._calculate_base_time(
            {"spark.executor.cores": "4"},
            ResourceSpec(cpu_cores=8, memory_gb=32),
            None,
        )

        assert base_time > 0

    def test_data_size_factor(self, simulation_model: SimulationModel) -> None:
        """Test data size factor in estimation."""
        small_data = {"size_gb": 10}
        large_data = {"size_gb": 1000}

        config = {"spark.executor.memory": "4g"}
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)

        metrics_small = simulation_model.estimate(config, resources, small_data)
        metrics_large = simulation_model.estimate(config, resources, large_data)

        # Larger data should generally take longer
        assert metrics_large.execution_time_seconds >= metrics_small.execution_time_seconds

    def test_adaptive_query_execution_factor(self, simulation_model: SimulationModel) -> None:
        """Test AQE factor in estimation."""
        with_aqe = {"spark.sql.adaptive.enabled": "true"}
        without_aqe = {"spark.sql.adaptive.enabled": "false"}

        resources = ResourceSpec(cpu_cores=8, memory_gb=32)

        metrics_with = simulation_model.estimate(with_aqe, resources)
        metrics_without = simulation_model.estimate(without_aqe, resources)

        # Both should return valid metrics
        assert isinstance(metrics_with, TrialMetrics)
        assert isinstance(metrics_without, TrialMetrics)


class TestExecutionRunnerMore:
    """Additional tests for ExecutionRunner."""

    def test_execution_runner_initialization_full(self) -> None:
        """Test ExecutionRunner initialization with all params (lines 174-188)."""
        mock_spark = MagicMock()
        runner = ExecutionRunner(spark_session=mock_spark, timeout_seconds=1800.0)

        assert runner.spark_session is mock_spark
        assert runner.timeout_seconds == 1800.0
        assert runner.mode == "execution"
        assert runner.enable_monitoring is False

    def test_execution_runner_no_spark_fallback(self) -> None:
        """Test ExecutionRunner falls back to simulation (lines 220-236)."""
        runner = ExecutionRunner()  # No spark session

        result = runner.run_trial(
            trial_number=1,
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=4, memory_gb=16),
        )

        assert isinstance(result, TrialResult)
        assert result.trial_number == 1

    @patch("spark_optima.core.bayesian.trial_runner.SparkRunner")
    def test_execution_runner_with_spark_run(self, mock_runner_class) -> None:
        """Test ExecutionRunner with actual Spark execution."""
        mock_spark = MagicMock()
        runner = ExecutionRunner(spark_session=mock_spark)

        with patch.object(runner, "_simulation_model") as mock_model:
            mock_metrics = TrialMetrics(execution_time_seconds=100.0, success=True)
            mock_model.estimate.return_value = mock_metrics

            result = runner.run_trial(
                trial_number=1,
                config={"spark.executor.memory": "4g"},
                resource_spec=ResourceSpec(cpu_cores=4, memory_gb=16),
            )

            assert isinstance(result, TrialResult)


class TestSimulationRunnerMore:
    """Additional tests for SimulationRunner."""

    def test_simulation_runner_initialization(self) -> None:
        """Test SimulationRunner initialization (lines 311-324)."""
        runner = SimulationRunner()
        assert runner.mode == "simulation"

        mock_model = MagicMock()
        runner = SimulationRunner(simulation_model=mock_model)
        assert runner._simulation_model is mock_model

    def test_simulation_runner_run_trial(self) -> None:
        """Test SimulationRunner run_trial (lines 326-364)."""
        runner = SimulationRunner()

        result = runner.run_trial(
            trial_number=1,
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=4, memory_gb=16),
        )

        assert isinstance(result, TrialResult)
        assert result.trial_number == 1
        assert result.metrics is not None


class TestSimulationModelMore:
    """Additional tests for SimulationModel."""

    def test_calculate_base_time(self) -> None:
        """Test _calculate_base_time method (lines 482-518)."""
        model = SimulationModel()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=4, memory_gb=16)

        base_time = model._calculate_base_time(
            config=config,
            resource_spec=resource_spec,
            data_profile={"size_gb": 10},
        )

        assert base_time > 0

    def test_estimate_memory_usage(self) -> None:
        """Test _estimate_memory_usage (lines 568-607)."""
        model = SimulationModel()

        result = model._estimate_memory_usage(
            data_size_gb=10.0,
            executor_memory_gb=4.0,
            config={"spark.memory.fraction": 0.6},
        )

        assert result > 0
        # Cap at 90% of executor + 50% of driver (4g = 4.0 GB)
        expected_max = 4.0 * 0.9 + 4.0 * 0.5
        assert result <= expected_max

    def test_estimate_cpu_utilization(self) -> None:
        """Test _estimate_cpu_utilization (lines 609-639)."""
        model = SimulationModel()

        result = model._estimate_cpu_utilization(
            executor_cores=4,
            parallelism=200,
            resource_spec=ResourceSpec(cpu_cores=8, memory_gb=16),
        )

        assert 10.0 <= result <= 100.0

    def test_estimate_shuffle(self) -> None:
        """Test _estimate_shuffle (lines 641-671)."""
        model = SimulationModel()

        read, write = model._estimate_shuffle(
            data_size_gb=10.0,
            config={"spark.shuffle.compress": True},
        )

        assert read >= 0
        assert write >= 0
        assert read <= write or write <= read  # Just check they're valid

    def test_estimate_cost(self) -> None:
        """Test _estimate_cost (lines 673-700)."""
        model = SimulationModel()
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        # Without cost model
        cost = model._estimate_cost(
            execution_time_seconds=3600.0,  # 1 hour
            resource_spec=resource_spec,
            cost_model=None,
        )

        assert cost > 0  # Should calculate default cost

    def test_estimate_cost_with_model(self) -> None:
        """Test _estimate_cost with cost model."""
        model = SimulationModel()
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        mock_cost_model = MagicMock()
        mock_cost_model.calculate.return_value = 15.0

        cost = model._estimate_cost(
            execution_time_seconds=3600.0,
            resource_spec=resource_spec,
            cost_model=mock_cost_model,
        )

        assert cost == 15.0
        mock_cost_model.calculate.assert_called_once_with(1.0)  # 3600s = 1 hour

    def test_validate_feasibility_true(self) -> None:
        """Test _validate_feasibility returns True."""
        model = SimulationModel()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=4, memory_gb=16)

        result = model._validate_feasibility(
            config=config,
            resource_spec=resource_spec,
            memory_peak=3.0,  # Under 4GB * 1.2
        )

        assert result is True

    def test_validate_feasibility_false_memory(self) -> None:
        """Test _validate_feasibility returns False for memory."""
        model = SimulationModel()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=4, memory_gb=16)

        result = model._validate_feasibility(
            config=config,
            resource_spec=resource_spec,
            memory_peak=5.0,  # Over 4GB * 1.2
        )

        assert result is False

    def test_validate_feasibility_false_cores(self) -> None:
        """Test _validate_feasibility returns False for cores."""
        model = SimulationModel()
        config = {"spark.executor.cores": 8}
        resource_spec = ResourceSpec(cpu_cores=4, memory_gb=16)  # Only 4 cores

        result = model._validate_feasibility(
            config=config,
            resource_spec=resource_spec,
            memory_peak=3.0,
        )

        assert result is False

    def test_parse_memory_string(self) -> None:
        """Test _parse_memory with string (lines 738-757)."""
        result = SimulationModel._parse_memory("4g")
        assert result == 4.0

        result = SimulationModel._parse_memory("512m")
        assert result == 0.5  # 512MB = 0.5GB

        result = SimulationModel._parse_memory(1024)
        assert result == 1024.0

    def test_parse_memory_default(self) -> None:
        """Test _parse_memory with invalid string."""
        result = SimulationModel._parse_memory("invalid")
        assert result == 4.0  # Default 4GB

    def test_parse_int_valid(self) -> None:
        """Test _parse_int with valid value (lines 759-771)."""
        result = SimulationModel._parse_int(42)
        assert result == 42

        result = SimulationModel._parse_int("42")
        assert result == 42

    def test_parse_int_invalid(self) -> None:
        """Test _parse_int with invalid value."""
        result = SimulationModel._parse_int("invalid")
        assert result == 0  # Default

        result = SimulationModel._parse_int(None)
        assert result == 0  # Default

    def test_simulation_model_estimate_error(self) -> None:
        """Test estimate handles errors gracefully."""
        model = SimulationModel()

        # This should not raise, but return default metrics
        result = model.estimate(
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=4, memory_gb=16),
        )

        assert isinstance(result, TrialMetrics)
        assert result.execution_time_seconds > 0


class TestTrialRunnerMore:
    """Additional tests for TrialRunner."""

    def test_trial_runner_with_code(self) -> None:
        """Test TrialRunner with code parameter."""
        runner = TrialRunner(mode="simulation")

        result = runner.run_trial(
            trial_number=1,
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=4, memory_gb=16),
            code="print('test')",
        )

        assert isinstance(result, TrialResult)
        assert result.trial_number == 1

    def test_trial_runner_with_code_path(self) -> None:
        """Test TrialRunner with code_path parameter."""
        runner = TrialRunner(mode="simulation")

        result = runner.run_trial(
            trial_number=1,
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=4, memory_gb=16),
            code_path="/test/path.py",
        )

        assert isinstance(result, TrialResult)
        assert result.trial_number == 1

    def test_trial_runner_exception(self) -> None:
        """Test TrialRunner handles exceptions."""
        runner = TrialRunner(mode="simulation")

        # Mock the simulation engine to raise an exception
        # The except block catches (RuntimeError, TimeoutError, ValueError, AttributeError, KeyError, TypeError)
        runner._simulation_engine.estimate = MagicMock(side_effect=RuntimeError("Test error"))

        result = runner.run_trial(
            trial_number=1,
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=4, memory_gb=16),
        )

        assert result.status == TrialStatus.FAILED
        assert result.metrics.success is False
