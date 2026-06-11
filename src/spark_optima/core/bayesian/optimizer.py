# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Bayesian optimizer for Spark configuration tuning.

This module provides the main BayesianOptimizer class that uses Optuna
for intelligent hyperparameter optimization of Spark configurations.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

from spark_optima.core.bayesian.models import (
    BayesianOptimizationResult,
    OptimizationObjective,
    ParetoPoint,
    SearchSpaceConfig,
    TrialResult,
    TrialStatus,
)
from spark_optima.core.bayesian.objectives import (
    MultiObjectiveFunction,
    ObjectiveFunctionFactory,
)
from spark_optima.core.bayesian.search_space import SearchSpaceBuilder
from spark_optima.core.bayesian.trial_runner import TrialRunner
from spark_optima.platforms.models import CostModel, ResourceSpec

if TYPE_CHECKING:
    from spark_optima.core.config_engine.models import ConfigSet

logger = logging.getLogger(__name__)

# Optuna imports with graceful fallback
try:
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler
    from optuna.storages import JournalFileStorage, JournalStorage

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Bayesian optimization disabled.")


class BayesianOptimizer:
    """Bayesian optimizer for Spark configuration tuning using Optuna.

    This class implements Bayesian optimization to fine-tune Spark configurations
    starting from heuristic baseline values. It supports both single-objective
    and multi-objective optimization with parallel trial execution.

    Example:
        >>> from spark_optima.core.bayesian import BayesianOptimizer
        >>> optimizer = BayesianOptimizer(
        ...     heuristic_config={"spark.executor.memory": "4g", ...},
        ...     config_set=config_set,
        ...     resource_spec=ResourceSpec(cpu_cores=16, memory_gb=64),
        ... )
        >>> result = optimizer.optimize(n_trials=50)
        >>> print(f"Best config: {result.best_config}")

    """

    def __init__(
        self,
        heuristic_config: dict[str, Any],
        config_set: ConfigSet,
        resource_spec: ResourceSpec | None = None,
        cost_model: CostModel | None = None,
        search_space_config: SearchSpaceConfig | None = None,
        objectives: list[str] | None = None,
        mode: str = "simulation",
        study_name: str | None = None,
        storage_path: str | None = None,
        max_consecutive_failures: int = 10,
        code_path: str | Path | None = None,
        platform: str = "local",
        spark_version: str = "default",
    ) -> None:
        """Initialize the Bayesian optimizer.

        Args:
            heuristic_config: Baseline configuration from heuristic engine.
            config_set: Spark configuration set with parameter metadata.
            resource_spec: Resource specifications for the platform.
            cost_model: Cost model for cost-based optimization.
            search_space_config: Search space configuration options.
            objectives: List of objective names to optimize.
            mode: Execution mode - "simulation" or "execution".
            study_name: Name for the Optuna study.
            storage_path: Path for study storage (None for in-memory).

        Raises:
            RuntimeError: If Optuna is not available.
            ValueError: If parameters are invalid.

        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError(
                "Optuna is required for Bayesian optimization. Install with: pip install optuna",
            )

        self.heuristic_config = heuristic_config
        self.config_set = config_set
        self.resource_spec = resource_spec or ResourceSpec(cpu_cores=4, memory_gb=16)
        self.cost_model = cost_model
        self.search_space_config = search_space_config or SearchSpaceConfig()
        self.objectives = objectives or [OptimizationObjective.MINIMIZE_TIME.value]
        self.mode = mode
        self.study_name = study_name or f"spark_optima_{int(time.time())}"
        self.storage_path = storage_path

        # Build search space from heuristic configuration
        self._search_space_builder = SearchSpaceBuilder()
        self._search_space = self._search_space_builder.build_from_heuristic(
            heuristic_config=heuristic_config,
            config_set=config_set,
            config=self.search_space_config,
        )

        # Initialize trial runner
        self._trial_runner = TrialRunner(mode=mode, platform=platform, spark_version=spark_version)

        # Initialize objective function
        self._objective_func = ObjectiveFunctionFactory.create_multi(self.objectives)

        # Study storage
        self._storage = None
        if storage_path:
            os.makedirs(os.path.dirname(storage_path) or ".", exist_ok=True)
            self._storage = JournalStorage(JournalFileStorage(storage_path))

        # Results tracking
        self._trial_results: list[TrialResult] = []
        self._trials: list[TrialResult] = []  # Alias for test compatibility
        self._is_multi_objective = len(self.objectives) > 1
        self._best_config: dict[str, Any] | None = None
        self._best_score: float | None = None
        self._consecutive_failures: int = 0
        self.max_consecutive_failures: int = 10  # Default: stop after 10 consecutive failures
        self._code_path = code_path

        # Warm-start tracking (E1/E2): seed trial + prior trials loaded from storage
        self._n_prior_trials: int = 0
        self._seed_trial_enqueued: bool = False

    def optimize(
        self,
        n_trials: int = 50,
        timeout_minutes: int | None = None,
        n_jobs: int = 1,
        show_progress: bool = True,
        data_profile: dict[str, Any] | None = None,
        max_consecutive_failures: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> BayesianOptimizationResult:
        """Run Bayesian optimization.

        Args:
            n_trials: Number of optimization trials to run.
            timeout_minutes: Maximum optimization time (None for no limit).
            n_jobs: Number of parallel jobs (-1 for auto).
            show_progress: Whether to show optimization progress.
            data_profile: Data characteristics for simulation.
            max_consecutive_failures: Stop after N consecutive failures (None for default).
            progress_callback: Optional callable invoked after every finished
                trial with a progress event dictionary containing
                ``trial_number`` (int), ``n_trials`` (int requested for this
                run), ``trials_completed`` (int, trials recorded in the study
                so far), and ``state`` (str, Optuna trial state name). For
                single-objective runs the event additionally carries
                ``best_value`` (float | None); multi-objective runs carry
                ``best_values`` (list[float] | None, objective values of one
                Pareto-optimal trial) instead. Exceptions raised by the
                callback are swallowed with a debug log so progress reporting
                can never break the optimization. Default None (no behavior
                change).

        Returns:
            BayesianOptimizationResult with optimal configuration.

        """
        start_time = time.time()

        # Set max consecutive failures
        if max_consecutive_failures is not None:
            self.max_consecutive_failures = max_consecutive_failures

        # Create Optuna study (resumes from storage when available)
        study = self._create_study()

        # Warm start (E2): count completed trials loaded from a stored study
        self._n_prior_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        self._seed_trial_enqueued = False
        if self._n_prior_trials > 0:
            logger.info(
                f"Warm start: loaded {self._n_prior_trials} completed prior trials from study '{self.study_name}'",
            )
        elif n_trials > 0:
            # Heuristic seed trial (E1): enqueue the baseline configuration as the
            # first trial so Optuna never returns a result worse than the warm start.
            # Only seed fresh studies - resumed studies already contain the seed.
            seed_params = self.get_seed_trial_params()
            if seed_params:
                study.enqueue_trial(seed_params, skip_if_exists=True)
                self._seed_trial_enqueued = True
                logger.info(
                    f"Enqueued heuristic baseline as seed trial ({len(seed_params)} parameters)",
                )

        # Optimization callback for progress tracking and error recovery
        def callback(_study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
            if show_progress and _study.trials:
                last_trial = _study.trials[-1]
                if last_trial.number % 10 == 0 and last_trial.number > 0:
                    logger.info(f"Completed trial {last_trial.number}/{n_trials}")

            # Check for consecutive failures
            if self.max_consecutive_failures > 0:
                consecutive = 0
                for t in reversed(_study.trials):
                    if t.state == optuna.trial.TrialState.FAIL:
                        consecutive += 1
                        if consecutive >= self.max_consecutive_failures:
                            logger.warning(
                                f"Stopping optimization: {consecutive} consecutive "
                                f"trials failed (max: {self.max_consecutive_failures})",
                            )
                            _study.stop()
                            break
                    else:
                        break

        # Optional external progress reporting (fired per finished trial).
        # Exceptions are swallowed: progress must never break optimization.
        callbacks = [callback]
        if progress_callback is not None:
            report = progress_callback  # Bind for the closure (and mypy narrowing)

            def _report_progress(_study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
                try:
                    report(self._build_progress_event(_study, _trial, n_trials))
                except Exception:  # noqa: BLE001 — progress reporting must never break optimization
                    logger.debug("Progress callback raised; continuing optimization", exc_info=True)

            callbacks.append(_report_progress)

        # Run optimization
        timeout_seconds = timeout_minutes * 60 if timeout_minutes else None

        study.optimize(
            func=lambda trial: self._objective(trial, data_profile),
            n_trials=n_trials,
            timeout=timeout_seconds,
            n_jobs=n_jobs,
            callbacks=callbacks,
            show_progress_bar=show_progress,
        )

        # Build result
        optimization_time = time.time() - start_time
        result = self._build_result(study, optimization_time)

        logger.info(f"Optimization completed in {optimization_time:.1f}s")
        logger.info(f"Best trial: {result.best_trial_number}")

        return result

    def _create_study(self) -> optuna.Study:
        """Create an Optuna study with appropriate configuration.

        When a storage path is configured and the study already exists in
        storage, the study is loaded instead of recreated so prior trials
        warm-start the optimization.

        Returns:
            Configured Optuna study.

        """
        # Configure sampler (TPE - Tree-structured Parzen Estimator)
        sampler = TPESampler(
            n_startup_trials=max(10, len(self._search_space) // 2),
            multivariate=True,
            seed=42,
        )

        # Configure pruner for early stopping
        pruner = HyperbandPruner(
            min_resource=1,
            reduction_factor=3,
        )

        # Resume existing studies only when persistent storage is configured
        load_if_exists = self._storage is not None

        # Create study
        if self._is_multi_objective:
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self._storage,
                sampler=sampler,
                pruner=pruner,
                directions=self._objective_func.get_directions(),
                load_if_exists=load_if_exists,
            )
        else:
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self._storage,
                sampler=sampler,
                pruner=pruner,
                direction=self._objective_func.get_directions()[0],
                load_if_exists=load_if_exists,
            )

        return study

    def _build_progress_event(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        n_trials: int,
    ) -> dict[str, Any]:
        """Build the progress event dictionary for a finished trial.

        Args:
            study: The Optuna study the trial belongs to.
            trial: The trial that just finished (FrozenTrial in callbacks).
            n_trials: Number of trials requested for the current run.

        Returns:
            Progress event with ``trial_number``, ``n_trials``,
            ``trials_completed``, ``state``, and ``best_value`` (single
            objective) or ``best_values`` (multi-objective). Non-finite
            objective values (the ``inf`` penalty of failed trials) are
            reported as None so the event always serializes to strict JSON.

        """
        event: dict[str, Any] = {
            "trial_number": trial.number,
            "n_trials": n_trials,
            "trials_completed": len(study.trials),
            "state": trial.state.name,
        }
        if self._is_multi_objective:
            # No single best value exists: report the objective values of one
            # Pareto-optimal trial (None until a trial completes).
            try:
                best_trials = study.best_trials
            except ValueError:
                event["best_values"] = None
            else:
                if best_trials:
                    # Failed trials are penalized with inf, which json.dumps
                    # would emit as the invalid token 'Infinity'; map any
                    # non-finite element to None to keep the payload valid.
                    event["best_values"] = [value if math.isfinite(value) else None for value in best_trials[0].values]
                else:
                    event["best_values"] = None
        else:
            try:
                best_value = study.best_value
            except ValueError:
                # No completed trial yet
                event["best_value"] = None
            else:
                # Until the first successful trial, best_value is the inf
                # penalty of a failed trial — not valid JSON; report None.
                event["best_value"] = best_value if math.isfinite(best_value) else None
        return event

    def get_seed_trial_params(self) -> dict[str, Any]:
        """Build Optuna trial parameters for the heuristic baseline configuration.

        The returned dictionary matches the parameter names, categorical choices,
        and numeric units used by the objective's ``trial.suggest_*`` calls, so it
        can be enqueued directly via ``study.enqueue_trial()``. Parameters outside
        the search space are skipped and out-of-range values are clamped.

        Returns:
            Dictionary of Optuna parameter names to seed values. Empty when the
            search space is empty.

        """
        return self._search_space_builder.to_trial_params(
            self.heuristic_config,
            self._search_space,
        )

    def _objective(
        self,
        trial: optuna.Trial,
        data_profile: dict[str, Any] | None,
    ) -> float | list[float]:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object.
            data_profile: Data characteristics.

        Returns:
            Objective value(s) to optimize.

        """
        # Sample configuration from search space
        config = self._sample_config(trial)

        # Run trial
        trial_result = self._trial_runner.run_trial(
            trial_number=trial.number,
            config=config,
            resource_spec=self.resource_spec,
            cost_model=self.cost_model,
            data_profile=data_profile,
            code_path=self._code_path,
        )

        # Store result
        self._trial_results.append(trial_result)
        self._trials = self._trial_results  # Keep alias in sync

        # Report intermediate value for pruning (if trial is running)
        if trial_result.status == TrialStatus.COMPLETED:
            # Calculate objective values
            objective_values = self._objective_func.compute(trial_result.metrics)
            trial_result.objective_values = objective_values

            # Report for pruning
            if len(self.objectives) == 1:
                trial.report(objective_values[self.objectives[0]], step=1)

            # Return objective values
            if self._is_multi_objective:
                return [objective_values[name] for name in self.objectives]
            return objective_values[self.objectives[0]]

        # Failed trial - return penalty values
        if self._is_multi_objective:
            return [float("inf")] * len(self.objectives)
        return float("inf")

    def _sample_config(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample configuration parameters from search space.

        Args:
            trial: Optuna trial object.

        Returns:
            Sampled configuration dictionary.

        """
        config = {}

        for param_name, search_def in self._search_space.items():
            param_type = search_def["type"]
            base_value = search_def.get("base_value")

            if param_type == "categorical":
                # Categorical parameter
                choices = search_def["choices"]
                config[param_name] = trial.suggest_categorical(param_name, choices)

            elif param_type == "int":
                # Integer parameter
                low = int(search_def["low"])
                high = int(search_def["high"])
                step = search_def.get("step", 1)
                config[param_name] = trial.suggest_int(param_name, low, high, step=step)

            elif param_type == "float":
                # Float parameter
                low_float: float = float(search_def["low"])
                high_float: float = float(search_def["high"])
                config[param_name] = trial.suggest_float(param_name, low_float, high_float)

            elif param_type == "bytes":
                # Memory/bytes parameter - suggest as integer bytes
                low_val = int(search_def["low"])
                high_val = int(search_def["high"])

                # Validate and correct range if needed
                if low_val > high_val:
                    logger.warning(
                        f"Invalid range for {param_name}: low={low_val} > high={high_val}. "
                        "Swapping low and high values.",
                    )
                    low_val, high_val = high_val, low_val

                # Step shared with SearchSpaceBuilder.to_trial_params so enqueued
                # seed values land exactly on the suggestion grid
                step = SearchSpaceBuilder.compute_bytes_step(low_val, high_val)
                # Align the upper bound to the step grid to avoid Optuna range warnings
                high_val = low_val + ((high_val - low_val) // step) * step
                bytes_value = trial.suggest_int(param_name, low_val, high_val, step=step)
                config[param_name] = self._format_bytes(bytes_value)

            else:
                # Unknown type - use base value
                config[param_name] = base_value

        # Enforce constraint: spark.executor.cores >= spark.task.cpus
        if "spark.task.cpus" in config and "spark.executor.cores" in config:
            try:
                task_cpus = int(config["spark.task.cpus"])
                exec_cores = int(config["spark.executor.cores"])
                if task_cpus > exec_cores:
                    logger.warning(f"Adjusting spark.task.cpus ({task_cpus}) to <= spark.executor.cores ({exec_cores})")
                    config["spark.task.cpus"] = exec_cores
            except (ValueError, TypeError):
                pass

        return config

    def _build_result(
        self,
        study: Any = None,
        optimization_time: float = 0.0,
    ) -> BayesianOptimizationResult:
        """Build optimization result from study.

        Args:
            study: Completed Optuna study (optional for test compatibility).
            optimization_time: Total optimization time in seconds.

        Returns:
            BayesianOptimizationResult.

        """
        # Handle test compatibility - no study provided
        if study is None:
            # Sync trials if needed (for tests that set _trials directly)
            if not self._trial_results and self._trials:
                self._trial_results = self._trials

            # Count trial statuses
            completed = sum(1 for t in self._trial_results if t.status == TrialStatus.COMPLETED)
            pruned = sum(1 for t in self._trial_results if t.status == TrialStatus.PRUNED)
            failed = sum(1 for t in self._trial_results if t.status == TrialStatus.FAILED)

            # Use stored best config if available
            best_config = getattr(self, "_best_config", None)
            best_score = getattr(self, "_best_score", None)

            return BayesianOptimizationResult(
                best_config=best_config,
                best_trial_number=-1 if best_config is None else 0,
                all_trials=self._trial_results.copy(),
                pareto_frontier=[],
                optimization_time_seconds=optimization_time,
                study_name=self.study_name,
                n_trials_completed=completed,
                n_trials_pruned=pruned,
                n_trials_failed=failed,
                metadata={
                    "objectives": self.objectives,
                    "n_trials": len(self._trial_results),
                    "best_score": best_score,
                    "n_prior_trials": self._n_prior_trials,
                    "seed_trial_enqueued": self._seed_trial_enqueued,
                },
            )

        # Normal flow with study provided
        # Handle case when no trials were run (n_trials=0)
        if len(study.trials) == 0:
            return BayesianOptimizationResult(
                best_config=None,
                best_trial_number=-1,
                all_trials=[],
                pareto_frontier=[],
                optimization_time_seconds=optimization_time,
                study_name=self.study_name,
                n_trials_completed=0,
                n_trials_pruned=0,
                n_trials_failed=0,
                metadata={
                    "objectives": self.objectives,
                    "n_trials": 0,
                    "study_directions": study.directions,
                    "n_prior_trials": self._n_prior_trials,
                    "seed_trial_enqueued": self._seed_trial_enqueued,
                },
            )

        if self._is_multi_objective:
            # For multi-objective, get Pareto frontier
            best_trials = study.best_trials
            best_trial = best_trials[0] if best_trials else None
            pareto_frontier = self._build_pareto_frontier(study)
        else:
            best_trial = study.best_trial
            pareto_frontier = []

        # Get best configuration
        best_config = {}
        best_trial_number = -1
        if best_trial:
            best_config = self._get_trial_config(best_trial)
            best_trial_number = best_trial.number

        # Count trial statuses
        completed = sum(1 for t in self._trial_results if t.status == TrialStatus.COMPLETED)
        pruned = sum(1 for t in self._trial_results if t.status == TrialStatus.PRUNED)
        failed = sum(1 for t in self._trial_results if t.status == TrialStatus.FAILED)

        return BayesianOptimizationResult(
            best_config=best_config,
            best_trial_number=best_trial_number,
            all_trials=self._trial_results.copy(),
            pareto_frontier=pareto_frontier,
            optimization_time_seconds=optimization_time,
            study_name=self.study_name,
            n_trials_completed=completed,
            n_trials_pruned=pruned,
            n_trials_failed=failed,
            metadata={
                "objectives": self.objectives,
                "n_trials": len(self._trial_results),
                "study_directions": study.directions,
                "n_prior_trials": self._n_prior_trials,
                "seed_trial_enqueued": self._seed_trial_enqueued,
            },
        )

    def _build_pareto_frontier(self, study: optuna.Study) -> list[ParetoPoint]:
        """Build Pareto frontier for multi-objective optimization.

        Args:
            study: Optuna study with multi-objective results.

        Returns:
            List of Pareto points.

        """
        pareto_points = []

        for trial in study.best_trials:
            # Get trial result
            trial_result = next(
                (tr for tr in self._trial_results if tr.trial_number == trial.number),
                None,
            )

            if trial_result:
                pareto_points.append(
                    ParetoPoint(
                        trial_number=trial.number,
                        objective_values=trial_result.objective_values,
                        configuration=trial_result.configuration,
                    ),
                )

        return pareto_points

    def _get_trial_config(self, trial: optuna.Trial) -> dict[str, Any]:
        """Get configuration for a trial.

        For trials run in this session, the configuration is taken from the
        recorded trial results. For trials loaded from a stored study
        (warm start), the configuration is reconstructed from the Optuna
        trial parameters.

        Args:
            trial: Optuna trial.

        Returns:
            Configuration dictionary.

        """
        trial_result = next(
            (tr for tr in self._trial_results if tr.trial_number == trial.number),
            None,
        )
        if trial_result:
            return trial_result.configuration

        # Warm-start fallback: rebuild the config from stored Optuna params
        params = getattr(trial, "params", None)
        if isinstance(params, dict):
            return self._config_from_trial_params(params)
        return {}

    def _config_from_trial_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Reconstruct a Spark configuration from raw Optuna trial parameters.

        Byte-size parameters are stored as integers in Optuna and must be
        converted back to Spark memory strings (e.g. "4g").

        Args:
            params: Optuna trial parameter dictionary.

        Returns:
            Spark configuration dictionary.

        """
        config: dict[str, Any] = {}
        for param_name, value in params.items():
            search_def = self._search_space.get(param_name)
            if search_def and search_def.get("type") == "bytes":
                config[param_name] = self._format_bytes(int(value))
            else:
                config[param_name] = value
        return config

    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format bytes as human-readable string suitable for Spark config.

        Args:
            bytes_value: Value in bytes.

        Returns:
            Human-readable string (e.g., "4g", "512m", "64k").
            Always returns integer values to be compatible with Spark.

        """
        # Round to nearest integer value
        for unit, divisor in [("t", 1024**4), ("g", 1024**3), ("m", 1024**2), ("k", 1024)]:
            if bytes_value >= divisor:
                value = round(bytes_value / divisor)
                return f"{int(value)}{unit}"
        # For bytes less than 1024, return as-is
        return f"{int(bytes_value)}b"

    def get_search_space(self) -> dict[str, Any]:
        """Get the current search space.

        Returns:
            Dictionary mapping parameter names to search definitions.

        """
        return self._search_space.copy()

    def get_trial_results(self) -> list[TrialResult]:
        """Get all trial results.

        Returns:
            List of trial results.

        """
        return self._trial_results.copy()

    # Methods for test compatibility
    def _build_search_space(self) -> dict[str, Any]:
        """Build and return search space (for test compatibility).

        Returns:
            Dictionary mapping parameter names to search definitions.

        """
        return self._search_space.copy()

    def _create_objective_function(self) -> MultiObjectiveFunction:
        """Create and return objective function (for test compatibility).

        Returns:
            MultiObjectiveFunction instance.

        """
        return self._objective_func

    def _create_trial_result(
        self,
        trial_number: int,
        config: dict[str, Any] | None = None,
        metrics: Any = None,
        error: str | None = None,
    ) -> TrialResult:
        """Create a trial result (for test compatibility).

        Args:
            trial_number: Trial identifier.
            config: Configuration dictionary.
            metrics: Trial metrics or None.
            error: Error message if failed.

        Returns:
            TrialResult instance.

        """
        from spark_optima.core.bayesian.models import TrialMetrics

        if error:
            return TrialResult(
                trial_number=trial_number,
                configuration=config or {},
                metrics=TrialMetrics(success=False, error_message=error),
                status=TrialStatus.FAILED,
            )

        # Handle metrics - convert if needed
        if metrics is None:
            trial_metrics = TrialMetrics()
        elif isinstance(metrics, TrialMetrics):
            trial_metrics = metrics
        else:
            # Handle dict or other formats
            trial_metrics = TrialMetrics()

        return TrialResult(
            trial_number=trial_number,
            configuration=config or {},
            metrics=trial_metrics,
            status=TrialStatus.COMPLETED if trial_metrics.success else TrialStatus.FAILED,
        )

    def _update_best_config(
        self,
        config: dict[str, Any],
        metrics: Any,
    ) -> None:
        """Update best configuration (for test compatibility).

        Args:
            config: Configuration to consider.
            metrics: Metrics for evaluation.

        """
        # Simple implementation - store as best if no best exists
        if not hasattr(self, "_best_config") or self._best_config is None:
            self._best_config = config
            self._best_score = getattr(metrics, "execution_time_seconds", 0.0)
        else:
            # Compare and update if better
            current_score = getattr(metrics, "execution_time_seconds", float("inf"))
            if self._best_score is None or current_score < self._best_score:
                self._best_config = config
                self._best_score = current_score

    def _sample_configuration(
        self,
        search_space: dict[str, Any],
    ) -> dict[str, Any]:
        """Sample configuration from search space (for test compatibility).

        Args:
            search_space: Search space definition.

        Returns:
            Sampled configuration.

        """
        import random

        config = {}
        for param_name, search_def in search_space.items():
            if "type" not in search_def:
                # No type specified, use base_value
                config[param_name] = search_def.get("base_value")
                continue

            param_type = search_def["type"]

            if param_type == "categorical":
                choices = search_def.get("choices", [])
                if choices:
                    config[param_name] = random.choice(choices)  # nosec B311 - pseudorandom used for optimization, not crypto
                else:
                    config[param_name] = search_def.get("base_value")
            elif param_type == "int":
                low = int(search_def.get("low", 0))
                high = int(search_def.get("high", 100))
                config[param_name] = random.randint(low, high)  # nosec B311 - pseudorandom used for optimization, not crypto
            elif param_type == "float":
                low_float: float = float(search_def.get("low", 0.0))
                high_float: float = float(search_def.get("high", 1.0))
                config[param_name] = random.uniform(low_float, high_float)  # nosec B311 - pseudorandom used for optimization, not crypto
            else:
                # Unrecognized type, use base_value
                config[param_name] = search_def.get("base_value")

        return config
