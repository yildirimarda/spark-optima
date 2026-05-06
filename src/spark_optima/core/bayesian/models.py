# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Data models for Bayesian optimization.

This module defines dataclasses for representing Bayesian optimization
results, trial outcomes, and search space configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrialStatus(str, Enum):
    """Status of an optimization trial."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"


class OptimizationObjective(str, Enum):
    """Available optimization objectives."""

    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_SUCCESS = "maximize_success"
    MINIMIZE_MEMORY = "minimize_memory"


@dataclass
class TrialMetrics:
    """Metrics collected during a trial execution.

    Attributes:
        execution_time_seconds: Actual or simulated execution time.
        memory_peak_gb: Peak memory usage in gigabytes.
        cpu_utilization_percent: Average CPU utilization.
        shuffle_read_gb: Total shuffle data read.
        shuffle_write_gb: Total shuffle data written.
        cost_estimate_usd: Estimated cost in USD.
        success: Whether the job completed successfully.
        error_message: Error message if failed.

    """

    execution_time_seconds: float = 0.0
    memory_peak_gb: float = 0.0
    cpu_utilization_percent: float = 0.0
    shuffle_read_gb: float = 0.0
    shuffle_write_gb: float = 0.0
    cost_estimate_usd: float = 0.0
    success: bool = True
    error_message: str = ""
    warnings: list[str] = field(default_factory=list)

    def __init__(
        self,
        execution_time_seconds: float = 0.0,
        memory_peak_gb: float = 0.0,
        cpu_utilization_percent: float = 0.0,
        shuffle_read_gb: float = 0.0,
        shuffle_write_gb: float = 0.0,
        cost_estimate_usd: float = 0.0,
        success: bool = True,
        error_message: str = "",
        warnings: list[str] | None = None,
        memory_usage_mb: float | None = None,
    ) -> None:
        """Initialize with support for memory_usage_mb parameter."""
        # If memory_usage_mb is provided, convert to memory_peak_gb
        if memory_usage_mb is not None and memory_peak_gb == 0.0:
            memory_peak_gb = memory_usage_mb / 1024.0

        self.execution_time_seconds = execution_time_seconds
        self.memory_peak_gb = memory_peak_gb
        self.cpu_utilization_percent = cpu_utilization_percent
        self.shuffle_read_gb = shuffle_read_gb
        self.shuffle_write_gb = shuffle_write_gb
        self.cost_estimate_usd = cost_estimate_usd
        self.success = success
        self.error_message = error_message
        self.warnings = warnings if warnings is not None else []

        # Validate
        if self.execution_time_seconds < 0:
            raise ValueError("execution_time_seconds must be non-negative")
        if self.memory_peak_gb < 0:
            raise ValueError("memory_peak_gb must be non-negative")

    @property
    def memory_usage_mb(self) -> float | None:
        """Return memory usage in MB for compatibility.

        Returns:
            Memory usage in MB, or None if not available.

        """
        if self.memory_peak_gb > 0:
            return self.memory_peak_gb * 1024
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "execution_time_seconds": self.execution_time_seconds,
            "memory_peak_gb": self.memory_peak_gb,
            "cpu_utilization_percent": self.cpu_utilization_percent,
            "shuffle_read_gb": self.shuffle_read_gb,
            "shuffle_write_gb": self.shuffle_write_gb,
            "cost_estimate_usd": self.cost_estimate_usd,
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrialMetrics:
        """Create from dictionary representation."""
        return cls(
            execution_time_seconds=data.get("execution_time_seconds", 0.0),
            memory_peak_gb=data.get("memory_peak_gb", 0.0),
            cpu_utilization_percent=data.get("cpu_utilization_percent", 0.0),
            shuffle_read_gb=data.get("shuffle_read_gb", 0.0),
            shuffle_write_gb=data.get("shuffle_write_gb", 0.0),
            cost_estimate_usd=data.get("cost_estimate_usd", 0.0),
            success=data.get("success", True),
            error_message=data.get("error_message", ""),
        )


@dataclass
class TrialResult:
    """Result of a single optimization trial.

    Attributes:
        trial_number: Unique trial identifier.
        configuration: Spark configuration used in this trial.
        metrics: Performance metrics collected.
        status: Trial execution status.
        objective_values: Computed objective values.
        duration_seconds: Time taken to run the trial.
        timestamp: When the trial was executed.

    """

    trial_number: int
    configuration: dict[str, Any] = field(default_factory=dict)
    metrics: TrialMetrics = field(default_factory=TrialMetrics)
    status: TrialStatus = TrialStatus.PENDING
    objective_values: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: str = ""

    def __init__(
        self,
        trial_number: int,
        configuration: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,  # Alias for configuration
        metrics: TrialMetrics | None = None,
        status: TrialStatus = TrialStatus.PENDING,
        objective_values: dict[str, float] | None = None,
        duration_seconds: float = 0.0,
        timestamp: str = "",
        error: str | None = None,
    ) -> None:
        """Initialize with support for config alias and error message."""
        # Use config if configuration not provided (for test compatibility)
        actual_config = (
            configuration if configuration is not None else (config if config is not None else {})
        )

        self.trial_number = trial_number
        self.configuration = actual_config
        self.metrics = metrics if metrics is not None else TrialMetrics()
        self.status = status
        self.objective_values = objective_values if objective_values is not None else {}
        self.duration_seconds = duration_seconds
        self.timestamp = timestamp
        self._error = error

        # Set default timestamp if not provided
        if not self.timestamp:
            from datetime import datetime

            self.timestamp = datetime.now().isoformat()

        # Update metrics error message if provided
        if error and self.metrics:
            self.metrics.error_message = error
            # Set status to FAILED if error is provided
            if self.status == TrialStatus.PENDING:
                self.status = TrialStatus.FAILED

    @property
    def config(self) -> dict[str, Any]:
        """Alias for configuration (for test compatibility)."""
        return self.configuration

    @property
    def error(self) -> str | None:
        """Error message for failed trials (for test compatibility)."""
        return self._error or (self.metrics.error_message if self.metrics else None)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "trial_number": self.trial_number,
            "configuration": self.configuration,
            "metrics": self.metrics.to_dict(),
            "status": self.status.value,
            "objective_values": self.objective_values,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrialResult:
        """Create from dictionary representation."""
        return cls(
            trial_number=data["trial_number"],
            configuration=data.get("configuration", {}),
            metrics=TrialMetrics.from_dict(data.get("metrics", {})),
            status=TrialStatus(data.get("status", "pending")),
            objective_values=data.get("objective_values", {}),
            duration_seconds=data.get("duration_seconds", 0.0),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class SearchSpaceConfig:
    """Configuration for building search spaces.

    Attributes:
        variation_percent: Percentage variation around heuristic values.
        min_trials: Minimum number of trials to run.
        max_trials: Maximum number of trials to run.
        timeout_minutes: Maximum optimization time.
        n_jobs: Number of parallel jobs (-1 for auto).
        categorical_params: List of parameters to treat as categorical.
        fixed_params: Parameters with fixed values.
        param_ranges: Custom ranges for specific parameters.

    """

    variation_percent: float = 0.3  # ±30% default
    min_trials: int = 20
    max_trials: int = 100
    timeout_minutes: int = 60
    n_jobs: int = -1  # Auto-detect
    categorical_params: list[str] = field(default_factory=list)
    fixed_params: dict[str, Any] = field(default_factory=dict)
    param_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Handle invalid variation_percent values
        # Values > 100: error (e.g., 150%)
        # Values between 1.0 and 100: normalize if integer-like
        # (e.g., 20 -> 0.2), error if decimal (e.g., 1.5)
        # Values between 0 and 1.0: valid decimal
        # Values <= 0: error

        if self.variation_percent > 100:
            raise ValueError("variation_percent cannot exceed 100% (1.0)")

        if self.variation_percent > 1.0:
            # Between 1.0 and 100: only allow if it's an
            # integer percentage (e.g., 20, 30, 50)
            if self.variation_percent != int(self.variation_percent):
                raise ValueError(
                    "variation_percent must be between 0 and 1 (or 0-100 as integer percentage)",
                )
            # Normalize integer percentage to decimal (e.g., 20 -> 0.2)
            self.variation_percent = self.variation_percent / 100.0

        # Validate range - strict check for 0 and negative values
        if self.variation_percent <= 0:
            raise ValueError(
                "variation_percent must be positive and between 0 and 1 (or 0-100 as percentage)",
            )
        if self.min_trials < 1:
            raise ValueError("min_trials must be at least 1")
        if self.max_trials < self.min_trials:
            raise ValueError("max_trials must be >= min_trials")
        if self.timeout_minutes < 1:
            raise ValueError("timeout_minutes must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "variation_percent": self.variation_percent,
            "min_trials": self.min_trials,
            "max_trials": self.max_trials,
            "timeout_minutes": self.timeout_minutes,
            "n_jobs": self.n_jobs,
            "categorical_params": self.categorical_params,
            "fixed_params": self.fixed_params,
            "param_ranges": self.param_ranges,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchSpaceConfig:
        """Create from dictionary representation."""
        return cls(
            variation_percent=data.get("variation_percent", 0.3),
            min_trials=data.get("min_trials", 20),
            max_trials=data.get("max_trials", 100),
            timeout_minutes=data.get("timeout_minutes", 60),
            n_jobs=data.get("n_jobs", -1),
            categorical_params=data.get("categorical_params", []),
            fixed_params=data.get("fixed_params", {}),
            param_ranges=data.get("param_ranges", {}),
        )


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier.

    Attributes:
        trial_number: Reference to the trial.
        objective_values: Objective values for this point.
        configuration: Spark configuration.

    """

    trial_number: int
    objective_values: dict[str, float] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)

    def dominates(self, other: ParetoPoint, objectives: list[str]) -> bool:
        """Check if this point dominates another.

        Args:
            other: Other Pareto point to compare.
            objectives: List of objective names.

        Returns:
            True if this point dominates the other.

        """
        at_least_one_better = False
        for obj in objectives:
            self_val = self.objective_values.get(obj, float("inf"))
            other_val = other.objective_values.get(obj, float("inf"))

            if self_val > other_val:
                return False
            if self_val < other_val:
                at_least_one_better = True

        return at_least_one_better


@dataclass
class BayesianOptimizationResult:
    """Result of Bayesian optimization.

    Attributes:
        best_config: Optimal Spark configuration found.
        best_trial_number: Trial number of best configuration.
        all_trials: List of all trial results.
        pareto_frontier: Pareto frontier for multi-objective optimization.
        optimization_time_seconds: Total optimization time.
        study_name: Name of the Optuna study.
        n_trials_completed: Number of completed trials.
        n_trials_pruned: Number of pruned trials.
        n_trials_failed: Number of failed trials.
        metadata: Additional metadata.

    """

    best_config: dict[str, Any] | None = None
    best_trial_number: int = -1
    all_trials: list[TrialResult] = field(default_factory=list)
    pareto_frontier: list[ParetoPoint] = field(default_factory=list)
    optimization_time_seconds: float = 0.0
    study_name: str = ""
    n_trials_completed: int = 0
    n_trials_pruned: int = 0
    n_trials_failed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_score(self) -> float | None:
        """Best score from metadata (for test compatibility)."""
        return self.metadata.get("best_score")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "best_config": self.best_config,
            "best_trial_number": self.best_trial_number,
            "all_trials": [t.to_dict() for t in self.all_trials],
            "pareto_frontier": [
                {
                    "trial_number": p.trial_number,
                    "objective_values": p.objective_values,
                    "configuration": p.configuration,
                }
                for p in self.pareto_frontier
            ],
            "optimization_time_seconds": self.optimization_time_seconds,
            "study_name": self.study_name,
            "n_trials_completed": self.n_trials_completed,
            "n_trials_pruned": self.n_trials_pruned,
            "n_trials_failed": self.n_trials_failed,
            "metadata": self.metadata,
        }

    def get_top_configs(
        self,
        n: int = 5,
        objective: str = "minimize_time",
    ) -> list[tuple[dict[str, Any], float]]:
        """Get top N configurations by objective.

        Args:
            n: Number of configurations to return.
            objective: Objective to sort by.

        Returns:
            List of (config, objective_value) tuples.

        """
        sorted_trials = sorted(
            self.all_trials,
            key=lambda t: t.objective_values.get(objective, float("inf")),
        )
        return [
            (t.configuration, t.objective_values.get(objective, float("inf")))
            for t in sorted_trials[:n]
        ]

    def get_config_by_trial(self, trial_number: int) -> dict[str, Any] | None:
        """Get configuration for a specific trial.

        Args:
            trial_number: Trial number to look up.

        Returns:
            Configuration dict or None if not found.

        """
        for trial in self.all_trials:
            if trial.trial_number == trial_number:
                return trial.configuration
        return None

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"BayesianOptimizationResult("
            f"best_trial={self.best_trial_number}, "
            f"completed={self.n_trials_completed}, "
            f"pruned={self.n_trials_pruned}, "
            f"failed={self.n_trials_failed}, "
            f"time={self.optimization_time_seconds:.1f}s"
            f")"
        )
