# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Custom pruning strategies for Bayesian optimization.

This module provides custom pruning strategies that extend Optuna's
built-in pruners with Spark-specific logic for early stopping.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from spark_optima.core.bayesian.models import TrialStatus

logger = logging.getLogger(__name__)

# Optuna imports with graceful fallback
# TC002: These imports are needed at runtime for class definitions and availability check
try:
    import optuna  # noqa: TC002
    from optuna.pruners import BasePruner  # noqa: TC002
    from optuna.trial import FrozenTrial, TrialState  # noqa: TC002

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    BasePruner = object  # type: ignore


class Pruner(ABC):
    """Base class for trial pruners.

    Pruners determine whether to stop a trial early based on its
    performance relative to completed trials.

    """

    @abstractmethod
    def should_prune(self, trial: Any, study: Any) -> bool:
        """Determine whether to prune a trial.

        Args:
            trial: The trial to evaluate.
            study: The study containing all trials.

        Returns:
            True if the trial should be pruned.

        """


class MedianPruner(Pruner):
    """Pruner based on median performance of completed trials.

    This pruner stops trials that perform worse than the median
    of completed trials, helping to focus resources on promising
    configurations.

    Example:
        >>> pruner = MedianPruner(n_startup_trials=5)
        >>> if pruner.should_prune(trial, study):
        ...     raise TrialPruned()

    """

    def __init__(
        self,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 0,
    ) -> None:
        """Initialize the median pruner.

        Args:
            n_startup_trials: Number of trials to run before pruning.
            n_warmup_steps: Number of steps to wait before pruning.

        """
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps

    def should_prune(self, trial: Any, study: Any) -> bool:
        """Determine whether to prune based on median performance.

        Args:
            trial: The trial to evaluate.
            study: The study containing all trials.

        Returns:
            True if trial should be pruned.

        """
        # Don't prune during startup period
        if trial.number < self.n_startup_trials:
            return False

        # Get completed trials
        completed_trials = [
            t
            for t in study.trials
            if hasattr(t, "state")
            and (
                t.state.name == TrialStatus.COMPLETED.name
                or (hasattr(t.state, "name") and t.state.name == TrialStatus.COMPLETED.name)
            )
        ]

        if len(completed_trials) == 0:
            return False

        # Calculate median of completed trials
        values = [t.value for t in completed_trials if t.value is not None]
        if len(values) == 0:
            return False

        values.sort()
        median = values[len(values) // 2]

        # Prune if current trial is worse than median
        current_value = getattr(trial, "value", None)
        if current_value is None:
            return False

        # Ensure numeric comparison to avoid returning Any
        if isinstance(current_value, int | float) and isinstance(median, int | float):
            return current_value > median
        return False


class PercentilePruner(Pruner):
    """Pruner based on percentile performance.

    This pruner stops trials that fall below a specified percentile
    of completed trial performance.

    Example:
        >>> pruner = PercentilePruner(percentile=25.0)
        >>> if pruner.should_prune(trial, study):
        ...     raise TrialPruned()

    """

    def __init__(
        self,
        percentile: float = 25.0,
        n_startup_trials: int = 5,
    ) -> None:
        """Initialize the percentile pruner.

        Args:
            percentile: Percentile threshold (0-100).
            n_startup_trials: Number of trials to run before pruning.

        Raises:
            ValueError: If percentile is not between 0 and 100.

        """
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")

        self.percentile = percentile
        self.n_startup_trials = n_startup_trials

    def should_prune(self, trial: Any, study: Any) -> bool:
        """Determine whether to prune based on percentile.

        Args:
            trial: The trial to evaluate.
            study: The study containing all trials.

        Returns:
            True if trial should be pruned.

        """
        # Don't prune during startup period
        if trial.number < self.n_startup_trials:
            return False

        # Get completed trials
        completed_trials = [
            t
            for t in study.trials
            if hasattr(t, "state")
            and (
                t.state.name == TrialStatus.COMPLETED.name
                or (hasattr(t.state, "name") and t.state.name == TrialStatus.COMPLETED.name)
            )
        ]

        if len(completed_trials) == 0:
            return False

        # Get values
        values = [t.value for t in completed_trials if t.value is not None]
        if len(values) == 0:
            return False

        values.sort()

        # Calculate percentile threshold - worst performers are pruned
        # Percentile=25 means bottom 25% are pruned
        index = int(len(values) * self.percentile / 100)
        index = max(0, min(index, len(values) - 1))
        threshold = values[index]

        # Prune if worse than threshold (we want to minimize, so higher is worse)
        current_value = getattr(trial, "value", None)
        if current_value is None:
            return False

        # Ensure numeric comparison to avoid returning Any
        if (
            isinstance(current_value, int | float)
            and isinstance(threshold, int | float)
            and isinstance(values[0], int | float)
        ):
            return current_value >= threshold and current_value > values[0]
        return False


class SuccessRatePruner(Pruner):
    """Pruner based on trial success rate.

    This pruner considers the overall success rate of trials when
    deciding whether to prune, helping to avoid configurations
    that frequently fail.

    Example:
        >>> pruner = SuccessRatePruner(min_success_rate=0.5)
        >>> if pruner.should_prune(trial, study):
        ...     raise TrialPruned()

    """

    def __init__(
        self,
        min_success_rate: float = 0.5,
        n_startup_trials: int = 5,
    ) -> None:
        """Initialize the success rate pruner.

        Args:
            min_success_rate: Minimum acceptable success rate (0-1).
            n_startup_trials: Number of trials to run before pruning.

        """
        self.min_success_rate = min_success_rate
        self.n_startup_trials = n_startup_trials

    def should_prune(self, trial: Any, study: Any) -> bool:
        """Determine whether to prune based on success rate.

        Args:
            trial: The trial to evaluate.
            study: The study containing all trials.

        Returns:
            True if trial should be pruned.

        """
        # Don't prune during startup period
        if trial.number < self.n_startup_trials:
            return False

        # Calculate success rate
        total_trials = len(study.trials)
        if total_trials == 0:
            return False

        completed_trials = [
            t
            for t in study.trials
            if hasattr(t, "state")
            and (
                t.state.name == TrialStatus.COMPLETED.name
                or (hasattr(t.state, "name") and t.state.name == TrialStatus.COMPLETED.name)
            )
        ]

        if total_trials < self.n_startup_trials:
            return False

        success_rate = len(completed_trials) / total_trials

        # Don't prune if success rate is acceptable
        if success_rate >= self.min_success_rate:
            return False

        # Prune with probability inversely proportional to success rate
        import random

        return random.random() > success_rate  # nosec B311 - pseudorandom used for optimization, not crypto


class SparkConfigPruner(BasePruner if OPTUNA_AVAILABLE else object):  # type: ignore[misc]
    """Custom pruner for Spark configuration optimization.

    This pruner extends Optuna's pruning capabilities with Spark-specific
    heuristics to identify and stop unpromising trials early.

    It considers:
    - Memory configuration feasibility
    - Core/parallelism balance
    - Historical performance of similar configurations

    Example:
        >>> pruner = SparkConfigPruner(
        ...     warmup_trials=5,
        ...     min_resource=1,
        ... )
        >>> study = optuna.create_study(pruner=pruner)

    """

    def __init__(
        self,
        warmup_trials: int = 5,
        min_resource: int = 1,
        max_resource: int = 1,
        reduction_factor: int = 3,
        min_early_stopping_rate: int = 0,
    ) -> None:
        """Initialize the Spark configuration pruner.

        Args:
            warmup_trials: Number of trials to run before pruning.
            min_resource: Minimum resource allocation for a trial.
            max_resource: Maximum resource allocation for a trial.
            reduction_factor: Factor for reducing resource allocation.
            min_early_stopping_rate: Minimum early stopping rate.

        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is required for SparkConfigPruner")

        self.warmup_trials = warmup_trials
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.min_early_stopping_rate = min_early_stopping_rate

        # Track configuration history for similarity-based pruning
        self._config_history: list[dict[str, Any]] = []
        self._performance_history: list[float] = []

    def prune(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
    ) -> bool:
        """Determine whether to prune a trial.

        Args:
            study: The Optuna study.
            trial: The trial to evaluate.

        Returns:
            True if the trial should be pruned, False otherwise.

        """
        # Don't prune during warmup
        if trial.number < self.warmup_trials:
            return False

        # Get trial's configuration
        config = trial.params

        # Check for obviously infeasible configurations
        if self._is_infeasible_config(config):
            logger.debug(f"Pruning infeasible config in trial {trial.number}")
            return True

        # Check for similar poor-performing configurations
        if self._is_similar_to_poor_performer(config):
            logger.debug(f"Pruning similar to poor performer in trial {trial.number}")
            return True

        # Use Optuna's built-in pruning logic for intermediate values
        if len(trial.intermediate_values) > 0:
            return self._should_prune_based_on_intermediates(study, trial)

        return False

    def _is_infeasible_config(self, config: dict[str, Any]) -> bool:
        """Check if a configuration is obviously infeasible.

        Args:
            config: Configuration dictionary.

        Returns:
            True if configuration is infeasible.

        """
        # Check memory configuration
        executor_mem = self._parse_memory(config.get("spark.executor.memory", "4g"))
        driver_mem = self._parse_memory(config.get("spark.driver.memory", "4g"))

        # Driver memory should not exceed executor memory significantly
        if driver_mem > executor_mem * 2:
            return True

        # Check core configuration
        executor_cores = config.get("spark.executor.cores", 4)
        task_cpus = config.get("spark.task.cpus", 1)

        # Task CPUs should not exceed executor cores
        if task_cpus > executor_cores:
            return True

        # Check parallelism
        parallelism = config.get("spark.default.parallelism", 200)
        shuffle_partitions = config.get("spark.sql.shuffle.partitions", 200)
        _ = shuffle_partitions  # Used for potential future logic

        # Very high parallelism with few cores is inefficient
        return bool(parallelism > 10000 and executor_cores <= 4)

    def _is_similar_to_poor_performer(self, config: dict[str, Any]) -> bool:
        """Check if configuration is similar to historically poor performers.

        Args:
            config: Configuration dictionary.

        Returns:
            True if similar to poor performers.

        """
        if len(self._config_history) < self.warmup_trials:
            return False

        # Find similar configurations
        for i, hist_config in enumerate(self._config_history):
            if self._config_similarity(config, hist_config) > 0.9 and self._performance_history[
                i
            ] >= float("inf"):
                # Very similar to this historical config
                # Historical config failed
                return True

        return False

    def _config_similarity(
        self,
        config1: dict[str, Any],
        config2: dict[str, Any],
    ) -> float:
        """Calculate similarity between two configurations.

        Args:
            config1: First configuration.
            config2: Second configuration.

        Returns:
            Similarity score between 0 and 1.

        """
        # Get common parameters
        common_params = set(config1.keys()) & set(config2.keys())

        if not common_params:
            return 0.0

        matches: float = 0.0
        for param in common_params:
            val1 = config1[param]
            val2 = config2[param]

            # Exact match
            if val1 == val2:
                matches += 1
                continue

            # Numeric similarity for numeric parameters
            try:
                num1 = float(val1)
                num2 = float(val2)
                if num1 > 0 and num2 > 0:
                    ratio = min(num1, num2) / max(num1, num2)
                    if ratio > 0.9:  # Within 10%
                        matches += ratio
            except (ValueError, TypeError):
                pass

        return matches / len(common_params)

    def _should_prune_based_on_intermediates(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
    ) -> bool:
        """Determine pruning based on intermediate values.

        Args:
            study: The Optuna study.
            trial: The trial to evaluate.

        Returns:
            True if trial should be pruned.

        """
        # Get completed trials for comparison
        completed_trials = [
            t for t in study.trials if t.state == TrialState.COMPLETE and t.number < trial.number
        ]

        if len(completed_trials) < self.warmup_trials:
            return False

        # Get best value from completed trials
        try:
            best_value = min(t.value for t in completed_trials if t.value is not None)
        except ValueError:
            return False

        # Get current trial's latest intermediate value
        latest_step = max(trial.intermediate_values.keys())
        current_value = trial.intermediate_values[latest_step]

        # Prune if significantly worse than best
        return bool(best_value > 0 and current_value > best_value * 2)

    def update_history(
        self,
        config: dict[str, Any],
        performance: float,
    ) -> None:
        """Update configuration history after trial completion.

        Args:
            config: Trial configuration.
            performance: Trial performance metric.

        """
        self._config_history.append(config.copy())
        self._performance_history.append(performance)

    @staticmethod
    def _parse_memory(value: str | int | float) -> float:
        """Parse memory value to GB.

        Args:
            value: Memory value (e.g., "4g", "512m", 1024).

        Returns:
            Memory in GB.

        """
        if isinstance(value, int | float):
            return float(value)

        value = str(value).strip().lower()

        import re

        match = re.match(r"^([\d.]+)\s*([kmgt]?)\s*b?$", value)
        if not match:
            return 4.0  # Default 4GB

        number = float(match.group(1))
        unit = match.group(2)

        multipliers = {
            "k": 1 / (1024 * 1024),
            "m": 1 / 1024,
            "g": 1,
            "t": 1024,
        }

        return number * multipliers.get(unit, 1)
