# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for objective functions."""

from __future__ import annotations

import pytest

from spark_optima.core.bayesian.models import TrialMetrics
from spark_optima.core.bayesian.objectives import (
    MaximizeSuccessObjective,
    MinimizeCostObjective,
    MinimizeMemoryObjective,
    MinimizeTimeObjective,
    MultiObjectiveFunction,
    ObjectiveFunction,
    ObjectiveFunctionFactory,
)


class TestObjectiveFunction:
    """Tests for base ObjectiveFunction."""

    def test_invalid_direction(self) -> None:
        """Test that invalid direction raises ValueError (covers line 45)."""
        with pytest.raises(ValueError, match="direction must be"):
            ObjectiveFunction(name="test", direction="invalid")


class TestMinimizeTimeObjective:
    """Tests for MinimizeTimeObjective."""

    def test_initialization(self) -> None:
        """Test initialization."""
        obj = MinimizeTimeObjective()
        assert obj.name == "minimize_time"
        assert obj.direction == "minimize"
        assert obj.penalty_for_failure == 1e6

    def test_compute_success(self) -> None:
        """Test computing objective for successful trial."""
        obj = MinimizeTimeObjective()
        metrics = TrialMetrics(
            execution_time_seconds=120.0,
            memory_peak_gb=4.0,
            success=True,
        )
        value = obj.compute(metrics)
        assert value == pytest.approx(120.04, rel=0.01)  # 120 + 4*0.01

    def test_compute_failure(self) -> None:
        """Test computing objective for failed trial."""
        obj = MinimizeTimeObjective(penalty_for_failure=1e6)
        metrics = TrialMetrics(success=False)
        value = obj.compute(metrics)
        assert value == 1e6

    def test_is_better(self) -> None:
        """Test is_better comparison."""
        obj = MinimizeTimeObjective()
        assert obj.is_better(100.0, 120.0) is True
        assert obj.is_better(120.0, 100.0) is False


class TestMinimizeCostObjective:
    """Tests for MinimizeCostObjective."""

    def test_initialization(self) -> None:
        """Test initialization."""
        obj = MinimizeCostObjective()
        assert obj.name == "minimize_cost"
        assert obj.direction == "minimize"

    def test_compute_with_cost_estimate(self) -> None:
        """Test computing objective with cost estimate."""
        obj = MinimizeCostObjective()
        metrics = TrialMetrics(
            cost_estimate_usd=5.50,
            success=True,
        )
        value = obj.compute(metrics)
        assert value == 5.50

    def test_compute_without_cost_estimate(self) -> None:
        """Test computing objective without cost estimate."""
        obj = MinimizeCostObjective(time_cost_weight=0.3)
        metrics = TrialMetrics(
            execution_time_seconds=3600,  # 1 hour
            cost_estimate_usd=0.0,
            success=True,
        )
        value = obj.compute(metrics)
        # 1 hour * 0.3 weight = 0.3
        assert value == pytest.approx(0.3, rel=0.01)

    def test_compute_failure(self) -> None:
        """Test computing objective for failed trial."""
        obj = MinimizeCostObjective(penalty_for_failure=1e6)
        metrics = TrialMetrics(success=False)
        value = obj.compute(metrics)
        assert value == 1e6


class TestMaximizeSuccessObjective:
    """Tests for MaximizeSuccessObjective."""

    def test_initialization(self) -> None:
        """Test initialization."""
        obj = MaximizeSuccessObjective()
        assert obj.name == "maximize_success"
        assert obj.direction == "maximize"

    def test_compute_success(self) -> None:
        """Test computing objective for successful trial."""
        obj = MaximizeSuccessObjective(time_threshold_seconds=3600)
        metrics = TrialMetrics(
            execution_time_seconds=1800,  # Half of threshold
            memory_peak_gb=20.0,
            success=True,
        )
        value = obj.compute(metrics)
        # Should be between 0 and 1
        assert 0.0 <= value <= 1.0

    def test_compute_failure(self) -> None:
        """Test computing objective for failed trial."""
        obj = MaximizeSuccessObjective()
        metrics = TrialMetrics(success=False)
        value = obj.compute(metrics)
        assert value == 0.0

    def test_is_better(self) -> None:
        """Test is_better comparison (maximize)."""
        obj = MaximizeSuccessObjective()
        assert obj.is_better(0.8, 0.6) is True
        assert obj.is_better(0.6, 0.8) is False


class TestMinimizeMemoryObjective:
    """Tests for MinimizeMemoryObjective."""

    def test_initialization(self) -> None:
        """Test initialization."""
        obj = MinimizeMemoryObjective()
        assert obj.name == "minimize_memory"
        assert obj.direction == "minimize"

    def test_compute_success(self) -> None:
        """Test computing objective for successful trial."""
        obj = MinimizeMemoryObjective(time_weight=0.1)
        metrics = TrialMetrics(
            memory_peak_gb=16.0,
            execution_time_seconds=600.0,
            success=True,
        )
        value = obj.compute(metrics)
        # 16.0 + 600*0.1 = 16 + 60 = 76
        assert value == pytest.approx(76.0, rel=0.01)

    def test_compute_failure(self) -> None:
        """Test computing objective for failed trial."""
        obj = MinimizeMemoryObjective(penalty_for_failure=1e6)
        metrics = TrialMetrics(success=False)
        value = obj.compute(metrics)
        assert value == 1e6


class TestMultiObjectiveFunction:
    """Tests for MultiObjectiveFunction."""

    def test_initialization(self) -> None:
        """Test initialization."""
        objectives = [
            MinimizeTimeObjective(),
            MinimizeCostObjective(),
        ]
        multi = MultiObjectiveFunction(objectives)
        assert len(multi.objectives) == 2
        assert "minimize_time" in multi.get_objective_names()
        assert "minimize_cost" in multi.get_objective_names()

    def test_initialization_empty(self) -> None:
        """Test initialization with empty objectives."""
        with pytest.raises(ValueError, match="At least one objective"):
            MultiObjectiveFunction([])

    def test_compute(self) -> None:
        """Test computing multiple objectives."""
        objectives = [
            MinimizeTimeObjective(),
            MinimizeCostObjective(),
        ]
        multi = MultiObjectiveFunction(objectives)

        metrics = TrialMetrics(
            execution_time_seconds=120.0,
            cost_estimate_usd=5.0,
            success=True,
        )
        values = multi.compute(metrics)

        assert "minimize_time" in values
        assert "minimize_cost" in values
        assert values["minimize_cost"] == 5.0

    def test_get_directions(self) -> None:
        """Test getting directions."""
        objectives = [
            MinimizeTimeObjective(),
            MaximizeSuccessObjective(),
        ]
        multi = MultiObjectiveFunction(objectives)

        directions = multi.get_directions()
        assert directions == ["minimize", "maximize"]

    def test_is_pareto_dominated(self) -> None:
        """Test Pareto domination check."""
        objectives = [
            MinimizeTimeObjective(),
            MinimizeCostObjective(),
        ]
        multi = MultiObjectiveFunction(objectives)

        values1 = {"minimize_time": 100.0, "minimize_cost": 50.0}
        values2 = {"minimize_time": 120.0, "minimize_cost": 60.0}

        # values1 dominates values2 (both are better)
        assert multi.is_pareto_dominated(values2, values1) is True
        assert multi.is_pareto_dominated(values1, values2) is False

    def test_not_pareto_dominated_tradeoff(self) -> None:
        """Test non-domination with trade-offs."""
        objectives = [
            MinimizeTimeObjective(),
            MinimizeCostObjective(),
        ]
        multi = MultiObjectiveFunction(objectives)

        values1 = {"minimize_time": 100.0, "minimize_cost": 60.0}
        values2 = {"minimize_time": 120.0, "minimize_cost": 50.0}

        # Neither dominates (trade-off)
        assert multi.is_pareto_dominated(values1, values2) is False
        assert multi.is_pareto_dominated(values2, values1) is False

    def test_is_pareto_dominated_maximize(self) -> None:
        """Test Pareto domination check with maximize direction (covers lines 370-373)."""
        objectives = [
            MinimizeTimeObjective(),  # minimize
            MaximizeSuccessObjective(),  # maximize
        ]
        multi = MultiObjectiveFunction(objectives)

        # values1 dominates values2: time is better (lower), success is better (higher)
        values1 = {"minimize_time": 100.0, "maximize_success": 0.9}
        values2 = {"minimize_time": 120.0, "maximize_success": 0.7}

        assert multi.is_pareto_dominated(values2, values1) is True
        assert multi.is_pareto_dominated(values1, values2) is False

    def test_is_pareto_dominated_maximize_not_dominated(self) -> None:
        """Test non-domination with maximize objective trade-off."""
        objectives = [
            MinimizeTimeObjective(),  # minimize
            MaximizeSuccessObjective(),  # maximize
        ]
        multi = MultiObjectiveFunction(objectives)

        # Trade-off: values1 has better time but worse success
        values1 = {"minimize_time": 100.0, "maximize_success": 0.7}
        values2 = {"minimize_time": 120.0, "maximize_success": 0.9}

        # Neither dominates the other
        assert multi.is_pareto_dominated(values1, values2) is False
        assert multi.is_pareto_dominated(values2, values1) is False


class TestObjectiveFunctionFactory:
    """Tests for ObjectiveFunctionFactory."""

    def test_create_minimize_time(self) -> None:
        """Test creating minimize_time objective."""
        obj = ObjectiveFunctionFactory.create("minimize_time")
        assert isinstance(obj, MinimizeTimeObjective)
        assert obj.name == "minimize_time"

    def test_create_minimize_cost(self) -> None:
        """Test creating minimize_cost objective."""
        obj = ObjectiveFunctionFactory.create("minimize_cost")
        assert isinstance(obj, MinimizeCostObjective)

    def test_create_maximize_success(self) -> None:
        """Test creating maximize_success objective."""
        obj = ObjectiveFunctionFactory.create("maximize_success")
        assert isinstance(obj, MaximizeSuccessObjective)

    def test_create_minimize_memory(self) -> None:
        """Test creating minimize_memory objective."""
        obj = ObjectiveFunctionFactory.create("minimize_memory")
        assert isinstance(obj, MinimizeMemoryObjective)

    def test_create_invalid(self) -> None:
        """Test creating invalid objective."""
        with pytest.raises(ValueError, match="Unknown objective"):
            ObjectiveFunctionFactory.create("invalid_objective")

    def test_create_multi(self) -> None:
        """Test creating multi-objective function."""
        multi = ObjectiveFunctionFactory.create_multi(["minimize_time", "minimize_cost"])
        assert isinstance(multi, MultiObjectiveFunction)
        assert len(multi.objectives) == 2

    def test_get_available_objectives(self) -> None:
        """Test getting available objectives."""
        objectives = ObjectiveFunctionFactory.get_available_objectives()
        assert "minimize_time" in objectives
        assert "minimize_cost" in objectives
        assert "maximize_success" in objectives
        assert "minimize_memory" in objectives
