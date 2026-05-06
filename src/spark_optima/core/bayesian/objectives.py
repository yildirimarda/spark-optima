# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Objective functions for Bayesian optimization.

This module provides objective function implementations for optimizing
Spark configurations across multiple dimensions (time, cost, memory, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

from spark_optima.core.bayesian.models import OptimizationObjective, TrialMetrics

logger = logging.getLogger(__name__)


class ObjectiveFunction:
    """Base class for optimization objective functions.

    Objective functions transform trial metrics into scalar values
    that the Bayesian optimizer seeks to minimize or maximize.

    Example:
        >>> objective = MinimizeTimeObjective()
        >>> value = objective.compute(metrics)
        >>> print(f"Objective value: {value}")

    """

    def __init__(self, name: str, direction: str = "minimize") -> None:
        """Initialize the objective function.

        Args:
            name: Objective name identifier.
            direction: Either "minimize" or "maximize".

        Raises:
            ValueError: If direction is invalid.

        """
        if direction not in ("minimize", "maximize"):
            raise ValueError("direction must be 'minimize' or 'maximize'")

        self.name = name
        self.direction = direction

    def compute(self, metrics: TrialMetrics) -> float:
        """Compute the objective value from trial metrics.

        Args:
            metrics: Trial performance metrics.

        Returns:
            Scalar objective value.

        Raises:
            NotImplementedError: Must be implemented by subclasses.

        """
        raise NotImplementedError("Subclasses must implement compute()")

    def is_better(self, value1: float, value2: float) -> bool:
        """Check if value1 is better than value2.

        Args:
            value1: First objective value.
            value2: Second objective value.

        Returns:
            True if value1 is better than value2.

        """
        if self.direction == "minimize":
            return value1 < value2
        return value1 > value2


class MinimizeTimeObjective(ObjectiveFunction):
    """Objective to minimize execution time.

    This objective prioritizes faster job completion, useful for
    latency-sensitive workloads.

    """

    def __init__(self, penalty_for_failure: float = 1e6) -> None:
        """Initialize the minimize time objective.

        Args:
            penalty_for_failure: Large penalty value for failed trials.

        """
        super().__init__(
            name=OptimizationObjective.MINIMIZE_TIME.value,
            direction="minimize",
        )
        self.penalty_for_failure = penalty_for_failure

    def compute(self, metrics: TrialMetrics) -> float:
        """Compute objective value from execution time.

        Args:
            metrics: Trial performance metrics.

        Returns:
            Execution time in seconds, with penalty for failures.

        """
        if not metrics.success:
            return self.penalty_for_failure

        # Base value is execution time
        value = metrics.execution_time_seconds

        # Add small penalty for high memory usage (encourages efficiency)
        memory_penalty = metrics.memory_peak_gb * 0.01

        return value + memory_penalty


class MinimizeCostObjective(ObjectiveFunction):
    """Objective to minimize execution cost.

    This objective prioritizes cost efficiency, useful for cloud
    deployments with pay-per-use pricing.

    """

    def __init__(
        self,
        penalty_for_failure: float = 1e6,
        time_cost_weight: float = 0.3,
    ) -> None:
        """Initialize the minimize cost objective.

        Args:
            penalty_for_failure: Large penalty value for failed trials.
            time_cost_weight: Weight for time-based cost component.

        """
        super().__init__(
            name=OptimizationObjective.MINIMIZE_COST.value,
            direction="minimize",
        )
        self.penalty_for_failure = penalty_for_failure
        self.time_cost_weight = time_cost_weight

    def compute(self, metrics: TrialMetrics) -> float:
        """Compute objective value from cost estimate.

        Args:
            metrics: Trial performance metrics.

        Returns:
            Total cost in USD, with penalty for failures.

        """
        if not metrics.success:
            return self.penalty_for_failure

        # Use estimated cost if available
        if metrics.cost_estimate_usd > 0:
            return metrics.cost_estimate_usd

        # Otherwise, estimate from execution time and resources
        # Cost = base_cost + (time_cost_weight * time_in_hours * hourly_rate)
        time_hours = metrics.execution_time_seconds / 3600
        estimated_cost = time_hours * self.time_cost_weight

        return estimated_cost


class MaximizeSuccessObjective(ObjectiveFunction):
    """Objective to maximize job success rate.

    This objective prioritizes reliability, useful for critical
    production workloads.

    """

    def __init__(
        self,
        time_threshold_seconds: float = 3600,
    ) -> None:
        """Initialize the maximize success objective.

        Args:
            time_threshold_seconds: Maximum acceptable execution time.

        """
        super().__init__(
            name=OptimizationObjective.MAXIMIZE_SUCCESS.value,
            direction="maximize",
        )
        self.time_threshold_seconds = time_threshold_seconds

    def compute(self, metrics: TrialMetrics) -> float:
        """Compute objective value representing success likelihood.

        Args:
            metrics: Trial performance metrics.

        Returns:
            Success score between 0 and 1, higher is better.

        """
        if not metrics.success:
            return 0.0

        # Success score decreases as execution time approaches threshold
        time_ratio = metrics.execution_time_seconds / self.time_threshold_seconds
        time_penalty = min(1.0, time_ratio)

        # Memory efficiency bonus
        memory_efficiency = max(0.0, 1.0 - (metrics.memory_peak_gb / 100))

        # Combined score
        score = (1.0 - time_penalty * 0.5) * (0.5 + memory_efficiency * 0.5)

        return max(0.0, min(1.0, score))


class MinimizeMemoryObjective(ObjectiveFunction):
    """Objective to minimize memory usage.

    This objective prioritizes memory efficiency, useful for
    resource-constrained environments.

    """

    def __init__(
        self,
        penalty_for_failure: float = 1e6,
        time_weight: float = 0.1,
    ) -> None:
        """Initialize the minimize memory objective.

        Args:
            penalty_for_failure: Large penalty value for failed trials.
            time_weight: Weight for time component in combined score.

        """
        super().__init__(
            name=OptimizationObjective.MINIMIZE_MEMORY.value,
            direction="minimize",
        )
        self.penalty_for_failure = penalty_for_failure
        self.time_weight = time_weight

    def compute(self, metrics: TrialMetrics) -> float:
        """Compute objective value from memory usage.

        Args:
            metrics: Trial performance metrics.

        Returns:
            Memory-based objective value, with penalty for failures.

        """
        if not metrics.success:
            return self.penalty_for_failure

        # Primary: peak memory usage
        memory_score = metrics.memory_peak_gb

        # Secondary: time penalty (to avoid extremely slow low-memory configs)
        time_score = metrics.execution_time_seconds * self.time_weight

        return memory_score + time_score


class MultiObjectiveFunction:
    """Combines multiple objectives for multi-objective optimization.

    This class handles Pareto optimization across multiple dimensions,
    returning a vector of objective values.

    Example:
        >>> objectives = [
        ...     MinimizeTimeObjective(),
        ...     MinimizeCostObjective(),
        ... ]
        >>> multi = MultiObjectiveFunction(objectives)
        >>> values = multi.compute(metrics)
        >>> print(f"Time: {values['minimize_time']}, Cost: {values['minimize_cost']}")

    """

    def __init__(self, objectives: list[ObjectiveFunction]) -> None:
        """Initialize with multiple objective functions.

        Args:
            objectives: List of objective functions to combine.

        Raises:
            ValueError: If objectives list is empty.

        """
        if not objectives:
            raise ValueError("At least one objective is required")

        self.objectives = objectives
        self._objective_names = [obj.name for obj in objectives]

    def compute(self, metrics: TrialMetrics) -> dict[str, float]:
        """Compute all objective values.

        Args:
            metrics: Trial performance metrics.

        Returns:
            Dictionary mapping objective names to values.

        """
        return {obj.name: obj.compute(metrics) for obj in self.objectives}

    def get_objective_names(self) -> list[str]:
        """Get list of objective names.

        Returns:
            List of objective function names.

        """
        return self._objective_names.copy()

    def get_directions(self) -> list[str]:
        """Get optimization directions for all objectives.

        Returns:
            List of directions ("minimize" or "maximize").

        """
        return [obj.direction for obj in self.objectives]

    def is_pareto_dominated(
        self,
        values1: dict[str, float],
        values2: dict[str, float],
    ) -> bool:
        """Check if values1 is Pareto-dominated by values2.

        Args:
            values1: First set of objective values.
            values2: Second set of objective values.

        Returns:
            True if values1 is dominated by values2.

        """
        at_least_one_better = False

        for obj in self.objectives:
            name = obj.name
            val1 = values1.get(name, float("inf") if obj.direction == "minimize" else 0)
            val2 = values2.get(name, float("inf") if obj.direction == "minimize" else 0)

            # Check if values2 is better for this objective
            if obj.direction == "minimize":
                if val2 > val1:
                    return False
                if val2 < val1:
                    at_least_one_better = True
            else:  # maximize
                if val2 < val1:
                    return False
                if val2 > val1:
                    at_least_one_better = True

        return at_least_one_better


class ObjectiveFunctionFactory:
    """Factory for creating objective function instances.

    Provides a convenient way to create objective functions by name.

    Example:
        >>> factory = ObjectiveFunctionFactory()
        >>> objective = factory.create("minimize_time")
        >>> multi = factory.create_multi(["minimize_time", "minimize_cost"])

    """

    _objective_map: dict[str, type[ObjectiveFunction]] = {
        OptimizationObjective.MINIMIZE_TIME.value: MinimizeTimeObjective,
        OptimizationObjective.MINIMIZE_COST.value: MinimizeCostObjective,
        OptimizationObjective.MAXIMIZE_SUCCESS.value: MaximizeSuccessObjective,
        OptimizationObjective.MINIMIZE_MEMORY.value: MinimizeMemoryObjective,
    }

    @classmethod
    def create(cls, objective_name: str, **kwargs: Any) -> ObjectiveFunction:
        """Create an objective function by name.

        Args:
            objective_name: Name of the objective function.
            **kwargs: Additional arguments for the objective constructor.

        Returns:
            Objective function instance.

        Raises:
            ValueError: If objective name is unknown.

        """
        objective_class = cls._objective_map.get(objective_name)
        if objective_class is None:
            valid_names = list(cls._objective_map.keys())
            raise ValueError(
                f"Unknown objective: {objective_name}. Valid options: {valid_names}",
            )

        return objective_class(**kwargs)

    @classmethod
    def create_multi(
        cls,
        objective_names: list[str],
        **kwargs: Any,
    ) -> MultiObjectiveFunction:
        """Create a multi-objective function.

        Args:
            objective_names: List of objective function names.
            **kwargs: Additional arguments for each objective constructor.

        Returns:
            Multi-objective function instance.

        """
        objectives = [cls.create(name, **kwargs.get(name, {})) for name in objective_names]
        return MultiObjectiveFunction(objectives)

    @classmethod
    def get_available_objectives(cls) -> list[str]:
        """Get list of available objective names.

        Returns:
            List of objective function names.

        """
        return list(cls._objective_map.keys())
