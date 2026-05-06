# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for Bayesian optimization models."""

from __future__ import annotations

import pytest

from spark_optima.core.bayesian.models import (
    BayesianOptimizationResult,
    OptimizationObjective,
    ParetoPoint,
    SearchSpaceConfig,
    TrialMetrics,
    TrialResult,
    TrialStatus,
)


class TestTrialMetrics:
    """Tests for TrialMetrics class."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        metrics = TrialMetrics()
        assert metrics.execution_time_seconds == 0.0
        assert metrics.memory_peak_gb == 0.0
        assert metrics.success is True

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        metrics = TrialMetrics(
            execution_time_seconds=120.5,
            memory_peak_gb=8.5,
            cpu_utilization_percent=75.0,
            success=True,
        )
        assert metrics.execution_time_seconds == 120.5
        assert metrics.memory_peak_gb == 8.5
        assert metrics.cpu_utilization_percent == 75.0

    def test_validation_negative_time(self) -> None:
        """Test validation rejects negative time."""
        with pytest.raises(ValueError, match="non-negative"):
            TrialMetrics(execution_time_seconds=-1.0)

    def test_validation_negative_memory(self) -> None:
        """Test validation rejects negative memory."""
        with pytest.raises(ValueError, match="non-negative"):
            TrialMetrics(memory_peak_gb=-1.0)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = TrialMetrics(
            execution_time_seconds=60.0,
            memory_peak_gb=4.0,
            cost_estimate_usd=0.50,
        )
        data = metrics.to_dict()
        assert data["execution_time_seconds"] == 60.0
        assert data["memory_peak_gb"] == 4.0
        assert data["cost_estimate_usd"] == 0.50

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "execution_time_seconds": 90.0,
            "memory_peak_gb": 6.0,
            "success": False,
            "error_message": "Out of memory",
        }
        metrics = TrialMetrics.from_dict(data)
        assert metrics.execution_time_seconds == 90.0
        assert metrics.memory_peak_gb == 6.0
        assert metrics.success is False
        assert metrics.error_message == "Out of memory"

    def test_memory_usage_mb_property_zero(self) -> None:
        """Test memory_usage_mb property returns None when memory_peak_gb is 0."""
        metrics = TrialMetrics(memory_peak_gb=0.0)
        assert metrics.memory_usage_mb is None

    def test_memory_usage_mb_property_positive(self) -> None:
        """Test memory_usage_mb property returns correct value."""
        metrics = TrialMetrics(memory_peak_gb=2.0)
        assert metrics.memory_usage_mb == 2048.0  # 2.0 * 1024

    def test_memory_usage_mb_parameter(self) -> None:
        """Test memory_usage_mb parameter converts to memory_peak_gb (line 76)."""
        # When memory_usage_mb is provided and memory_peak_gb is 0,
        # it should convert memory_usage_mb to memory_peak_gb
        metrics = TrialMetrics(memory_usage_mb=4096.0)  # 4096 MB
        assert metrics.memory_peak_gb == pytest.approx(4.0, rel=0.01)  # 4096/1024 = 4.0
        assert metrics.memory_usage_mb == 4096.0


class TestTrialResult:
    """Tests for TrialResult class."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        result = TrialResult(trial_number=1)
        assert result.trial_number == 1
        assert result.status == TrialStatus.PENDING
        assert result.timestamp != ""

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        metrics = TrialMetrics(execution_time_seconds=60.0)
        result = TrialResult(
            trial_number=5,
            configuration={"spark.executor.memory": "4g"},
            metrics=metrics,
            status=TrialStatus.COMPLETED,
            objective_values={"minimize_time": 60.0},
        )
        assert result.trial_number == 5
        assert result.configuration["spark.executor.memory"] == "4g"
        assert result.status == TrialStatus.COMPLETED

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = TrialResult(
            trial_number=1,
            configuration={"key": "value"},
            status=TrialStatus.COMPLETED,
        )
        data = result.to_dict()
        assert data["trial_number"] == 1
        assert data["status"] == "completed"
        assert data["configuration"] == {"key": "value"}

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "trial_number": 2,
            "configuration": {"spark.cores": 4},
            "metrics": {"execution_time_seconds": 120.0, "success": True},
            "status": "completed",
            "objective_values": {"time": 120.0},
            "duration_seconds": 125.0,
            "timestamp": "2024-01-01T00:00:00",
        }
        result = TrialResult.from_dict(data)
        assert result.trial_number == 2
        assert result.status == TrialStatus.COMPLETED
        assert result.objective_values["time"] == 120.0

    def test_initialization_with_error(self) -> None:
        """Test initialization with error parameter sets metrics error_message."""
        result = TrialResult(
            trial_number=1,
            config={"spark.memory": "4g"},
            error="Test error message",
        )
        assert result.error == "Test error message"
        assert result.metrics.error_message == "Test error message"
        assert result.status == TrialStatus.FAILED

    def test_error_property_with_error_param(self) -> None:
        """Test error property when _error is set (line 200)."""
        result = TrialResult(
            trial_number=1,
            error="Error from param",
        )
        # error property should return _error when set
        assert result.error == "Error from param"

    def test_error_property_from_metrics(self) -> None:
        """Test error property falls back to metrics.error_message (line 200)."""
        metrics = TrialMetrics(success=False, error_message="Error from metrics")
        result = TrialResult(
            trial_number=1,
            metrics=metrics,
        )
        # error property should return metrics.error_message when _error is None
        assert result.error == "Error from metrics"

    def test_error_property_none(self) -> None:
        """Test error property returns empty string when no error."""
        result = TrialResult(trial_number=1)
        # error_message is empty string by default, so error property returns ''
        assert result.error == ""

    def test_config_property(self) -> None:
        """Test config property returns configuration (line 200)."""
        result = TrialResult(
            trial_number=1,
            configuration={"spark.memory": "4g"},
        )
        # config property should return the configuration
        assert result.config == {"spark.memory": "4g"}
        assert result.config is result.configuration  # Should be the same object


class TestSearchSpaceConfig:
    """Tests for SearchSpaceConfig class."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        config = SearchSpaceConfig()
        assert config.variation_percent == 0.3
        assert config.min_trials == 20
        assert config.max_trials == 100
        assert config.timeout_minutes == 60

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        config = SearchSpaceConfig(
            variation_percent=0.5,
            min_trials=10,
            max_trials=50,
            fixed_params={"spark.app.name": "test"},
        )
        assert config.variation_percent == 0.5
        assert config.min_trials == 10
        assert config.fixed_params["spark.app.name"] == "test"

    def test_validation_variation_percent(self) -> None:
        """Test validation of variation_percent."""
        with pytest.raises(ValueError, match="variation_percent"):
            SearchSpaceConfig(variation_percent=1.5)

        with pytest.raises(ValueError, match="variation_percent"):
            SearchSpaceConfig(variation_percent=0)

    def test_validation_variation_percent_gt_100(self) -> None:
        """Test validation rejects variation_percent > 100."""
        with pytest.raises(ValueError, match="variation_percent"):
            SearchSpaceConfig(variation_percent=150)

    def test_variation_percent_integer_normalization(self) -> None:
        """Test that integer percentages are normalized to decimal."""
        config = SearchSpaceConfig(variation_percent=20)  # 20%
        assert config.variation_percent == pytest.approx(0.2, rel=0.01)

    def test_variation_percent_decimal_1_to_100(self) -> None:
        """Test that decimal values between 1 and 100 raise error."""
        with pytest.raises(ValueError, match="variation_percent"):
            SearchSpaceConfig(variation_percent=1.5)

    def test_validation_min_trials(self) -> None:
        """Test validation of min_trials."""
        with pytest.raises(ValueError, match="min_trials"):
            SearchSpaceConfig(min_trials=0)

    def test_validation_timeout_minutes(self) -> None:
        """Test validation of timeout_minutes."""
        with pytest.raises(ValueError, match="timeout_minutes"):
            SearchSpaceConfig(timeout_minutes=0)

    def test_validation_trials(self) -> None:
        """Test validation of trial counts."""
        with pytest.raises(ValueError, match="max_trials"):
            SearchSpaceConfig(min_trials=100, max_trials=50)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = SearchSpaceConfig(variation_percent=0.4)
        data = config.to_dict()
        assert data["variation_percent"] == 0.4
        assert data["min_trials"] == 20

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "variation_percent": 0.25,
            "min_trials": 30,
            "max_trials": 150,
            "categorical_params": ["spark.serializer"],
        }
        config = SearchSpaceConfig.from_dict(data)
        assert config.variation_percent == 0.25
        assert config.categorical_params == ["spark.serializer"]


class TestParetoPoint:
    """Tests for ParetoPoint class."""

    def test_initialization(self) -> None:
        """Test initialization."""
        point = ParetoPoint(
            trial_number=1,
            objective_values={"time": 100.0, "cost": 50.0},
            configuration={"memory": "4g"},
        )
        assert point.trial_number == 1
        assert point.objective_values["time"] == 100.0

    def test_dominates_minimize(self) -> None:
        """Test dominates check for minimization."""
        point1 = ParetoPoint(
            trial_number=1,
            objective_values={"time": 100.0, "cost": 50.0},
        )
        point2 = ParetoPoint(
            trial_number=2,
            objective_values={"time": 120.0, "cost": 60.0},
        )

        # point1 dominates point2 (both objectives are better)
        assert point1.dominates(point2, ["time", "cost"]) is True
        assert point2.dominates(point1, ["time", "cost"]) is False

    def test_not_dominates_tradeoff(self) -> None:
        """Test non-domination with trade-offs."""
        point1 = ParetoPoint(
            trial_number=1,
            objective_values={"time": 100.0, "cost": 60.0},
        )
        point2 = ParetoPoint(
            trial_number=2,
            objective_values={"time": 120.0, "cost": 50.0},
        )

        # Neither dominates - trade-off between time and cost
        assert point1.dominates(point2, ["time", "cost"]) is False
        assert point2.dominates(point1, ["time", "cost"]) is False


class TestBayesianOptimizationResult:
    """Tests for BayesianOptimizationResult class."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        result = BayesianOptimizationResult()
        assert result.best_trial_number == -1
        assert result.all_trials == []
        assert result.pareto_frontier == []

    def test_with_trials(self) -> None:
        """Test initialization with trial results."""
        trials = [
            TrialResult(trial_number=1, status=TrialStatus.COMPLETED),
            TrialResult(trial_number=2, status=TrialStatus.COMPLETED),
        ]
        result = BayesianOptimizationResult(
            best_config={"memory": "4g"},
            best_trial_number=1,
            all_trials=trials,
            n_trials_completed=2,
        )
        assert result.best_trial_number == 1
        assert len(result.all_trials) == 2

    def test_get_top_configs(self) -> None:
        """Test getting top configurations."""
        trials = [
            TrialResult(
                trial_number=1,
                configuration={"a": 1},
                objective_values={"minimize_time": 100.0},
            ),
            TrialResult(
                trial_number=2,
                configuration={"a": 2},
                objective_values={"minimize_time": 50.0},
            ),
            TrialResult(
                trial_number=3,
                configuration={"a": 3},
                objective_values={"minimize_time": 75.0},
            ),
        ]
        result = BayesianOptimizationResult(all_trials=trials)

        top = result.get_top_configs(n=2, objective="minimize_time")
        assert len(top) == 2
        assert top[0][1] == 50.0  # Best (lowest) time first
        assert top[1][1] == 75.0

    def test_get_config_by_trial(self) -> None:
        """Test getting configuration by trial number."""
        trials = [
            TrialResult(
                trial_number=5,
                configuration={"memory": "8g"},
            ),
        ]
        result = BayesianOptimizationResult(all_trials=trials)

        config = result.get_config_by_trial(5)
        assert config == {"memory": "8g"}

        # Non-existent trial
        assert result.get_config_by_trial(999) is None

    def test_str_representation(self) -> None:
        """Test string representation."""
        result = BayesianOptimizationResult(
            best_trial_number=10,
            n_trials_completed=50,
            n_trials_pruned=5,
            optimization_time_seconds=120.5,
        )
        str_repr = str(result)
        assert "best_trial=10" in str_repr
        assert "completed=50" in str_repr
        assert "time=120.5s" in str_repr

    def test_to_dict_with_pareto_frontier(self) -> None:
        """Test to_dict with pareto frontier (covers line 382)."""
        pareto = [
            ParetoPoint(
                trial_number=1,
                objective_values={"minimize_time": 100.0},
                configuration={"memory": "4g"},
            ),
        ]
        result = BayesianOptimizationResult(
            pareto_frontier=pareto,
            best_config={"memory": "4g"},
        )
        data = result.to_dict()
        assert "pareto_frontier" in data
        assert len(data["pareto_frontier"]) == 1
        assert data["pareto_frontier"][0]["trial_number"] == 1
        assert data["pareto_frontier"][0]["objective_values"]["minimize_time"] == 100.0

    def test_best_score_property(self) -> None:
        """Test best_score property (line 393)."""
        # Test with best_score in metadata
        result = BayesianOptimizationResult(
            metadata={"best_score": 95.5},
        )
        assert result.best_score == 95.5

    def test_best_score_property_none(self) -> None:
        """Test best_score property returns None when not set."""
        result = BayesianOptimizationResult()
        assert result.best_score is None


class TestTrialStatus:
    """Tests for TrialStatus enum."""

    def test_enum_values(self) -> None:
        """Test enum values."""
        assert TrialStatus.PENDING.value == "pending"
        assert TrialStatus.RUNNING.value == "running"
        assert TrialStatus.COMPLETED.value == "completed"
        assert TrialStatus.FAILED.value == "failed"
        assert TrialStatus.PRUNED.value == "pruned"


class TestOptimizationObjective:
    """Tests for OptimizationObjective enum."""

    def test_enum_values(self) -> None:
        """Test enum values."""
        assert OptimizationObjective.MINIMIZE_TIME.value == "minimize_time"
        assert OptimizationObjective.MINIMIZE_COST.value == "minimize_cost"
        assert OptimizationObjective.MAXIMIZE_SUCCESS.value == "maximize_success"
        assert OptimizationObjective.MINIMIZE_MEMORY.value == "minimize_memory"
