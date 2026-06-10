# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Simulation engine for Spark configuration optimization.

This module provides the main SimulationEngine class that orchestrates
performance simulation using both analytical models and ML-based prediction.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from spark_optima.core.bayesian.models import TrialMetrics, TrialResult, TrialStatus
from spark_optima.core.simulation.performance_model import (
    DataCharacteristics,
    OperationProfile,
    OperationType,
    PerformanceModel,
)
from spark_optima.core.simulation.predictor import MLPerformancePredictor, extract_features
from spark_optima.platforms.models import CostModel, ResourceSpec

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of a simulation run.

    Attributes:
        metrics: TrialMetrics with performance estimates.
        analytical_estimate: Estimate from analytical model.
        ml_estimate: Estimate from ML model (if available).
        hybrid_estimate: Combined/weighted estimate.
        confidence_score: Confidence in the estimate (0-1).
        confidence_interval: 95% confidence interval as (lower, upper) bounds.
        model_breakdown: Detailed breakdown from each model.
        is_simulation: Always True, indicates this is not real measurement.

    """

    metrics: TrialMetrics
    analytical_estimate: dict[str, Any]
    ml_estimate: dict[str, Any] | None
    hybrid_estimate: dict[str, Any]
    confidence_score: float
    model_breakdown: dict[str, Any]
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    is_simulation: bool = True


class SimulationEngine:
    """Main simulation engine for Spark configuration optimization.

    This engine combines analytical performance modeling with ML-based
    prediction to provide accurate performance estimates. It automatically
    switches between models based on available data and confidence levels.

    Example:
        >>> engine = SimulationEngine()
        >>> result = engine.simulate(
        ...     config={"spark.executor.memory": "4g", ...},
        ...     resource_spec=ResourceSpec(cpu_cores=16, memory_gb=64),
        ...     data_profile={"size_gb": 100, "format": "parquet"},
        ...     operations=["scan", "aggregation"]
        ... )
        >>> print(f"Estimated time: {result.metrics.execution_time_seconds:.1f}s")

    """

    # Weight factors for hybrid estimation
    ANALYTICAL_WEIGHT = 0.6
    ML_WEIGHT = 0.4

    # Online surrogate blending policy (v1.3, Workstream L)
    ML_MIN_SAMPLES_DEFAULT = 20
    ML_RETRAIN_INTERVAL_DEFAULT = 10
    ML_R2_GATE = 0.3
    ML_R2_SATURATION = 0.9
    ML_WEIGHT_CAP_ANALYTICAL = 0.5
    ML_WEIGHT_CAP_MEASURED = 0.8
    ML_RAMP_SAMPLES_MULTIPLIER = 3

    def __init__(
        self,
        use_ml: bool = True,
        ml_model_path: str | Path | None = None,
        hybrid_mode: str = "auto",
        enable_ml_predictor: bool = True,
        ml_min_samples: int = ML_MIN_SAMPLES_DEFAULT,
        ml_retrain_interval: int = ML_RETRAIN_INTERVAL_DEFAULT,
        platform: str = "local",
        spark_version: str = "default",
    ) -> None:
        """Initialize the simulation engine.

        Args:
            use_ml: Whether to use ML-based prediction.
            ml_model_path: Path to saved ML model.
            hybrid_mode: Hybrid mode - "auto", "analytical", "ml", or "ensemble".
            enable_ml_predictor: Whether the online surrogate learns across trials
                and blends its predictions into estimates. Degrades silently to
                pure-analytical when scikit-learn is missing.
            ml_min_samples: Minimum accumulated samples before surrogate training.
            ml_retrain_interval: Retrain after this many new samples post-training.
            platform: Platform identifier used for default model persistence naming.
            spark_version: Spark version used for default model persistence naming.

        """
        self.use_ml = use_ml
        self.hybrid_mode = hybrid_mode
        self.enable_ml_predictor = enable_ml_predictor
        self.ml_min_samples = max(ml_min_samples, 4)
        self.ml_retrain_interval = max(ml_retrain_interval, 1)
        self.platform = platform
        self.spark_version = spark_version

        # Initialize analytical model
        self._analytical_model = PerformanceModel()

        # Initialize ML model (if available)
        self._ml_model: MLPerformancePredictor | None = None
        if use_ml:
            try:
                self._ml_model = MLPerformancePredictor(
                    model_path=ml_model_path,
                    use_ensemble=True,
                    platform=platform,
                    spark_version=spark_version,
                )
                logger.info("ML predictor initialized successfully")
            except RuntimeError as e:
                # Logged once here; all surrogate paths silently degrade to analytical
                logger.warning(f"ML predictor not available: {e}")
                self._ml_model = None

        # Online surrogate is active only when a predictor could be constructed
        self._surrogate_active = bool(use_ml and enable_ml_predictor and self._ml_model is not None)

        # Historical data for ML training
        self._trial_history: list[dict[str, Any]] = []

    def estimate(
        self,
        config: dict[str, Any],
        resource_spec: ResourceSpec | None = None,
        cost_model: CostModel | None = None,
        data_profile: dict[str, Any] | None = None,
    ) -> TrialMetrics:
        """Estimate performance metrics for a configuration (alias for simulate).

        This method provides a simpler interface that returns just TrialMetrics
        instead of the full SimulationResult. Used by TrialRunner.

        Args:
            config: Spark configuration dictionary.
            resource_spec: Resource specifications.
            cost_model: Cost model for cost estimation.
            data_profile: Data characteristics.

        Returns:
            TrialMetrics with performance estimates.

        """
        result = self.simulate(
            config=config,
            resource_spec=resource_spec,
            cost_model=cost_model,
            data_profile=data_profile,
        )
        return result.metrics

    def simulate(
        self,
        config: dict[str, Any],
        resource_spec: ResourceSpec | None = None,
        cost_model: CostModel | None = None,
        data_profile: dict[str, Any] | None = None,
        operations: list[str] | None = None,
    ) -> SimulationResult:
        """Run simulation for a configuration.

        Args:
            config: Spark configuration dictionary.
            resource_spec: Resource specifications.
            cost_model: Cost model for cost estimation.
            data_profile: Data characteristics.
            operations: List of operation names.

        Returns:
            SimulationResult with performance estimates.

        """
        # Set defaults
        resource_spec = resource_spec or ResourceSpec(cpu_cores=4, memory_gb=16)
        data_profile = data_profile or {"size_gb": 10, "format": "parquet"}
        operations = operations or ["scan"]

        # Convert to structured types
        data_chars = self._convert_data_profile(data_profile)
        op_profile = self._convert_operations(operations)

        # Run analytical model
        analytical_result = self._run_analytical_model(
            config=config,
            resource_spec=resource_spec,
            cost_model=cost_model,
            data_profile=data_chars,
            operations=op_profile,
        )

        # Run ML model (if available)
        ml_result = None
        if self._ml_model is not None and self._ml_model.is_trained():
            ml_result = self._run_ml_model(
                config=config,
                resource_spec=resource_spec,
                data_profile=data_profile,
                operations=operations,
            )

        # Combine estimates
        hybrid_estimate = self._combine_estimates(
            analytical_result=analytical_result,
            ml_result=ml_result,
        )

        # Online surrogate: learn from accumulated trials and blend its
        # prediction into the hybrid estimate (additive metadata only)
        surrogate_status = self._apply_ml_surrogate(
            config=config,
            data_profile=data_profile,
            analytical_time=float(analytical_result.get("execution_time_seconds", 0.0)),
            hybrid_estimate=hybrid_estimate,
        )

        # Calculate confidence score
        confidence = self._calculate_confidence(
            analytical_result=analytical_result,
            ml_result=ml_result,
        )

        # Calculate confidence interval (95% CI)
        # Wider interval for lower confidence
        base_time = hybrid_estimate.get("execution_time", 0.0)
        if confidence >= 0.8:
            margin = 0.05  # ±5%
        elif confidence >= 0.5:
            margin = 0.15  # ±15%
        else:
            margin = 0.30  # ±30%
        confidence_interval = (
            base_time * (1 - margin),
            base_time * (1 + margin),
        )

        # Build TrialMetrics
        metrics = self._build_trial_metrics(
            hybrid_estimate=hybrid_estimate,
            analytical_result=analytical_result,
        )

        # Add simulation warning to metrics
        metrics.warnings.append(
            "SIMULATION RESULT: This is a simulation, not a real measurement. Actual performance may vary.",
        )

        # Build detailed breakdown
        model_breakdown = {
            "analytical": analytical_result,
            "ml": ml_result,
            "hybrid_weights": {
                "analytical": self.ANALYTICAL_WEIGHT,
                "ml": self.ML_WEIGHT if ml_result else 0.0,
            },
            "ml_surrogate": surrogate_status,
        }

        return SimulationResult(
            metrics=metrics,
            analytical_estimate=analytical_result,
            ml_estimate=ml_result,
            hybrid_estimate=hybrid_estimate,
            confidence_score=confidence,
            confidence_interval=confidence_interval,
            model_breakdown=model_breakdown,
        )

    def simulate_trial(
        self,
        trial_number: int,
        config: dict[str, Any],
        resource_spec: ResourceSpec | None = None,
        cost_model: CostModel | None = None,
        data_profile: dict[str, Any] | None = None,
    ) -> TrialResult:
        """Run simulation and return TrialResult for Bayesian optimization.

        Args:
            trial_number: Trial identifier.
            config: Spark configuration.
            resource_spec: Resource specifications.
            cost_model: Cost model.
            data_profile: Data characteristics.

        Returns:
            TrialResult compatible with Bayesian optimizer.

        """
        start_time = __import__("time").time()

        # Run simulation
        result = self.simulate(
            config=config,
            resource_spec=resource_spec,
            cost_model=cost_model,
            data_profile=data_profile,
        )

        duration = __import__("time").time() - start_time

        # Store for ML training (provide default if resource_spec is None)
        actual_resource_spec = resource_spec or ResourceSpec(cpu_cores=4, memory_gb=16)
        self._store_trial_result(result, config, actual_resource_spec, data_profile)

        return TrialResult(
            trial_number=trial_number,
            configuration=config,
            metrics=result.metrics,
            status=TrialStatus.COMPLETED if result.metrics.success else TrialStatus.FAILED,
            duration_seconds=duration,
        )

    def _run_analytical_model(
        self,
        config: dict[str, Any],
        resource_spec: ResourceSpec,
        cost_model: CostModel | None,
        data_profile: DataCharacteristics,
        operations: OperationProfile,
    ) -> dict[str, Any]:
        """Run analytical performance model.

        Args:
            config: Spark configuration.
            resource_spec: Resource specifications.
            cost_model: Cost model.
            data_profile: Data characteristics.
            operations: Operation profile.

        Returns:
            Analytical model results.

        """
        return self._analytical_model.estimate(
            config=config,
            resource_spec=resource_spec,
            cost_model=cost_model,
            data_profile=data_profile,
            operations=operations,
        )

    def _run_ml_model(
        self,
        config: dict[str, Any],
        resource_spec: ResourceSpec,
        data_profile: dict[str, Any],
        operations: list[str],
    ) -> dict[str, Any] | None:
        """Run ML-based prediction model.

        Args:
            config: Spark configuration.
            resource_spec: Resource specifications.
            data_profile: Data characteristics.
            operations: List of operations.

        Returns:
            ML model results or None if not available.

        """
        if self._ml_model is None or not self._ml_model.is_trained():
            return None

        try:
            # Convert operations to flags
            op_flags = {
                "has_aggregation": "aggregation" in operations,
                "has_join": any(op in operations for op in ["join", "broadcast_join"]),
                "has_shuffle": any(op in operations for op in ["join", "aggregation", "sort"]),
            }

            prediction = self._ml_model.predict(
                config=config,
                data_profile=data_profile,
                resource_spec=resource_spec.to_dict(),
                operations=op_flags,
            )

            return {
                "execution_time_seconds": prediction.predicted_time,
                "confidence_lower": prediction.confidence_interval[0],
                "confidence_upper": prediction.confidence_interval[1],
                "feature_importance": prediction.feature_importance,
                "model_version": prediction.model_version,
            }
        except (RuntimeError, ValueError, AttributeError, TypeError, KeyError) as e:
            logger.warning(f"ML prediction failed: {e}")
            return None

    def _combine_estimates(
        self,
        analytical_result: dict[str, Any],
        ml_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Combine analytical and ML estimates.

        Args:
            analytical_result: Analytical model results.
            ml_result: ML model results.

        Returns:
            Combined hybrid estimate.

        """
        if ml_result is None:
            # Use only analytical
            return {
                "execution_time_seconds": analytical_result["execution_time_seconds"],
                "memory_peak_gb": analytical_result["memory_peak_gb"],
                "cpu_utilization_percent": analytical_result["cpu_utilization_percent"],
                "shuffle_read_gb": analytical_result["shuffle_read_gb"],
                "shuffle_write_gb": analytical_result["shuffle_write_gb"],
                "cost_estimate_usd": analytical_result["cost_estimate_usd"],
                "method": "analytical_only",
            }

        # Weighted combination
        analytical_time = analytical_result["execution_time_seconds"]
        ml_time = ml_result["execution_time_seconds"]

        # Check if estimates are wildly different (>5x)
        ratio = max(analytical_time, ml_time) / max(min(analytical_time, ml_time), 1)

        if ratio > 5.0:
            # Estimates differ significantly - use analytical as more conservative
            logger.debug(
                f"Large difference between models: analytical={analytical_time:.1f}s, "
                f"ml={ml_time:.1f}s. Using analytical.",
            )
            hybrid_time = analytical_time
            method = "analytical_fallback"
        else:
            # Normal weighted combination
            hybrid_time = self.ANALYTICAL_WEIGHT * analytical_time + self.ML_WEIGHT * ml_time
            method = "hybrid_weighted"

        # For other metrics, use analytical (more detailed)
        return {
            "execution_time_seconds": hybrid_time,
            "memory_peak_gb": analytical_result["memory_peak_gb"],
            "cpu_utilization_percent": analytical_result["cpu_utilization_percent"],
            "shuffle_read_gb": analytical_result["shuffle_read_gb"],
            "shuffle_write_gb": analytical_result["shuffle_write_gb"],
            "cost_estimate_usd": analytical_result["cost_estimate_usd"],
            "method": method,
            "analytical_time": analytical_time,
            "ml_time": ml_time,
        }

    def _apply_ml_surrogate(
        self,
        config: dict[str, Any],
        data_profile: dict[str, Any],
        analytical_time: float,
        hybrid_estimate: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the online surrogate step: train lazily, blend, and record the sample.

        The analytical estimate of every simulated trial is recorded as a
        pseudo-observation after prediction, so the surrogate never sees the
        point it is asked about. When the surrogate is trained and its hold-out
        R² clears the gate, its prediction is blended into ``hybrid_estimate``
        in place. All keys added to ``hybrid_estimate`` are additive.

        Args:
            config: Spark configuration of the current trial.
            data_profile: Data characteristics of the current trial.
            analytical_time: Execution time predicted by the analytical model.
            hybrid_estimate: Hybrid estimate dict, updated in place.

        Returns:
            Status dictionary describing what the surrogate did.

        """
        status: dict[str, Any] = {
            "enabled": self._surrogate_active,
            "trained": False,
            "ml_samples": 0,
            "measured_samples": 0,
            "ml_blend_weight": 0.0,
            "validation_r2": 0.0,
            "ml_prediction_seconds": None,
        }

        if not self._surrogate_active or self._ml_model is None:
            hybrid_estimate["ml_blend_weight"] = 0.0
            hybrid_estimate["ml_samples"] = 0
            return status

        predictor = self._ml_model
        try:
            features = extract_features(config, data_profile)

            # Train lazily on previously accumulated samples
            self._maybe_train_surrogate()

            trained = bool(predictor.has_online_model())
            r2 = float(predictor.online_r2)
            n_samples = int(predictor.sample_count)
            measured_fraction = float(predictor.measured_fraction)

            weight = 0.0
            if trained:
                weight = self._compute_blend_weight(
                    r2=r2,
                    n_samples=n_samples,
                    measured_fraction=measured_fraction,
                )

            applied_weight = 0.0
            if weight > 0.0:
                ml_prediction = predictor.predict_online(features)
                if ml_prediction is not None:
                    pre_blend = float(hybrid_estimate.get("execution_time_seconds", analytical_time))
                    blended = weight * ml_prediction + (1.0 - weight) * pre_blend
                    hybrid_estimate["pre_blend_time_seconds"] = pre_blend
                    hybrid_estimate["ml_surrogate_time_seconds"] = ml_prediction
                    hybrid_estimate["execution_time_seconds"] = max(blended, 1e-6)
                    applied_weight = weight
                    status["ml_prediction_seconds"] = ml_prediction

            # Record the analytical estimate as a pseudo-observation after prediction
            if math.isfinite(analytical_time) and analytical_time > 0:
                predictor.add_sample(features, analytical_time, measured=False)

            status.update(
                {
                    "trained": trained,
                    "ml_samples": int(predictor.sample_count),
                    "measured_samples": int(predictor.measured_sample_count),
                    "ml_blend_weight": applied_weight,
                    "validation_r2": r2,
                },
            )
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as exc:
            logger.warning(f"ML surrogate step failed; using analytical estimate: {exc}")

        hybrid_estimate["ml_blend_weight"] = status["ml_blend_weight"]
        hybrid_estimate["ml_samples"] = status["ml_samples"]
        return status

    def _compute_blend_weight(
        self,
        r2: float,
        n_samples: int,
        measured_fraction: float,
    ) -> float:
        """Compute the surrogate blend weight from validation quality and sample count.

        Policy:
        - A hold-out R² below ``ML_R2_GATE`` keeps the weight at 0 (analytical only).
        - The cap ramps from ``ML_WEIGHT_CAP_ANALYTICAL`` (analytical-only samples)
          to ``ML_WEIGHT_CAP_MEASURED`` as the share of real measurements grows.
        - The weight ramps with sample count, saturating at
          ``ML_RAMP_SAMPLES_MULTIPLIER * ml_min_samples`` samples, and with R²,
          saturating at ``ML_R2_SATURATION``.

        Args:
            r2: Hold-out validation R² of the surrogate model.
            n_samples: Number of accumulated online samples.
            measured_fraction: Fraction of samples backed by real measurements.

        Returns:
            Blend weight in [0, ML_WEIGHT_CAP_MEASURED].

        """
        if not math.isfinite(r2) or r2 < self.ML_R2_GATE:
            return 0.0

        clamped_fraction = min(max(measured_fraction, 0.0), 1.0)
        cap = (
            self.ML_WEIGHT_CAP_ANALYTICAL
            + (self.ML_WEIGHT_CAP_MEASURED - self.ML_WEIGHT_CAP_ANALYTICAL) * clamped_fraction
        )

        ramp_span = max(self.ML_RAMP_SAMPLES_MULTIPLIER * self.ml_min_samples, 1)
        sample_factor = min(1.0, n_samples / ramp_span)
        r2_factor = min(1.0, (r2 - self.ML_R2_GATE) / (self.ML_R2_SATURATION - self.ML_R2_GATE))

        return cap * sample_factor * r2_factor

    def _maybe_train_surrogate(self) -> None:
        """Train or retrain the online surrogate when enough new samples accumulated."""
        predictor = self._ml_model
        if predictor is None:
            return

        n_samples = predictor.sample_count
        if n_samples < self.ml_min_samples:
            return
        if predictor.has_online_model() and n_samples - predictor.online_trained_samples < self.ml_retrain_interval:
            return

        predictor.train_online(min_samples=self.ml_min_samples)

    def record_observation(
        self,
        config: dict[str, Any],
        data_profile: dict[str, Any] | None,
        measured_time: float,
    ) -> bool:
        """Record a real measured execution time for online surrogate learning.

        Intended for execution mode where actual runtimes are available.
        Measured samples raise the surrogate blend-weight cap because they
        carry more signal than analytical pseudo-observations.

        Args:
            config: Spark configuration that was executed.
            data_profile: Data characteristics of the executed workload.
            measured_time: Measured execution time in seconds.

        Returns:
            True if the observation was recorded, False when the surrogate is
            inactive or the measurement is invalid.

        """
        if not self._surrogate_active or self._ml_model is None:
            return False
        if not math.isfinite(measured_time) or measured_time <= 0:
            logger.debug(f"Ignored invalid measured time: {measured_time}")
            return False

        features = extract_features(config, data_profile or {})
        recorded = bool(self._ml_model.add_sample(features, measured_time, measured=True))
        if recorded:
            self._maybe_train_surrogate()
        return recorded

    def save_ml_model(self, path: str | Path | None = None) -> Path | None:
        """Persist the ML predictor state to disk.

        Args:
            path: Target file path. Defaults to the predictor's default path
                derived from platform and Spark version (see
                ``predictor.default_model_path``).

        Returns:
            The path the model was saved to, or None when no predictor exists.

        """
        if self._ml_model is None:
            logger.warning("Cannot save ML model: predictor not available")
            return None
        return self._ml_model.save(path)

    def load_ml_model(self, path: str | Path | None = None) -> bool:
        """Load previously persisted ML predictor state.

        Args:
            path: Source file path. Defaults to the predictor's default path.

        Returns:
            True if the state was loaded, False otherwise.

        """
        if self._ml_model is None:
            return False
        return self._ml_model.load(path)

    def get_ml_surrogate_status(self) -> dict[str, Any]:
        """Get a snapshot of the online surrogate state for diagnostics.

        Returns:
            Dictionary with ``enabled``, ``trained``, ``ml_samples``,
            ``measured_samples`` and ``validation_r2`` keys.

        """
        if not self._surrogate_active or self._ml_model is None:
            return {
                "enabled": False,
                "trained": False,
                "ml_samples": 0,
                "measured_samples": 0,
                "validation_r2": 0.0,
            }
        predictor = self._ml_model
        return {
            "enabled": True,
            "trained": bool(predictor.has_online_model()),
            "ml_samples": int(predictor.sample_count),
            "measured_samples": int(predictor.measured_sample_count),
            "validation_r2": float(predictor.online_r2),
        }

    def _calculate_confidence(
        self,
        analytical_result: dict[str, Any],
        ml_result: dict[str, Any] | None,
    ) -> float:
        """Calculate confidence score for the estimate.

        Args:
            analytical_result: Analytical model results.
            ml_result: ML model results.

        Returns:
            Confidence score between 0 and 1.

        """
        base_confidence = 0.7

        # Boost confidence if ML model agrees
        if ml_result is not None:
            base_confidence += 0.15

            # Check agreement between models
            analytical_time = analytical_result["execution_time_seconds"]
            ml_time = ml_result["execution_time_seconds"]

            if analytical_time > 0 and ml_time > 0:
                ratio = max(analytical_time, ml_time) / min(analytical_time, ml_time)
                if ratio < 1.5:  # Models agree within 50%
                    base_confidence += 0.1

        # Reduce confidence if feasibility issues
        if analytical_result.get("feasibility_issues"):
            base_confidence -= 0.1 * len(analytical_result["feasibility_issues"])

        return max(0.0, min(1.0, base_confidence))

    def _build_trial_metrics(
        self,
        hybrid_estimate: dict[str, Any],
        analytical_result: dict[str, Any],
    ) -> TrialMetrics:
        """Build TrialMetrics from estimates.

        Args:
            hybrid_estimate: Combined estimate.
            analytical_result: Analytical results.

        Returns:
            TrialMetrics object.

        """
        is_feasible = analytical_result.get("success", True)
        issues = analytical_result.get("feasibility_issues", [])

        return TrialMetrics(
            execution_time_seconds=hybrid_estimate["execution_time_seconds"],
            memory_peak_gb=hybrid_estimate["memory_peak_gb"],
            cpu_utilization_percent=hybrid_estimate["cpu_utilization_percent"],
            shuffle_read_gb=hybrid_estimate["shuffle_read_gb"],
            shuffle_write_gb=hybrid_estimate["shuffle_write_gb"],
            cost_estimate_usd=hybrid_estimate["cost_estimate_usd"],
            success=is_feasible,
            error_message="; ".join(issues) if issues else "",
        )

    def _convert_data_profile(
        self,
        data_profile: dict[str, Any],
    ) -> DataCharacteristics:
        """Convert dict to DataCharacteristics.

        Args:
            data_profile: Data profile dictionary.

        Returns:
            DataCharacteristics object.

        """
        return DataCharacteristics(
            size_gb=data_profile.get("size_gb", 10.0),
            num_rows=data_profile.get("num_rows"),
            num_columns=data_profile.get("num_columns", 10),
            avg_row_size_bytes=data_profile.get("avg_row_size_bytes", 100.0),
            format=data_profile.get("format", "parquet"),
            compression=data_profile.get("compression", "snappy"),
            partitioning=data_profile.get("partitioning", 100),
            null_ratio=data_profile.get("null_ratio", 0.05),
            cardinality=data_profile.get("cardinality"),
            skew_factor=data_profile.get("skew_factor", 1.0),
        )

    def _convert_operations(
        self,
        operations: list[str],
    ) -> OperationProfile:
        """Convert operation names to OperationProfile.

        Args:
            operations: List of operation names.

        Returns:
            OperationProfile object.

        """
        op_types = []
        join_details = {}
        has_agg = False
        has_shuffle = False

        for i, op in enumerate(operations):
            op_lower = op.lower()

            if "scan" in op_lower or "read" in op_lower:
                op_types.append(OperationType.SCAN)
            elif "filter" in op_lower:
                op_types.append(OperationType.FILTER)
            elif "project" in op_lower or "select" in op_lower:
                op_types.append(OperationType.PROJECT)
            elif "agg" in op_lower or "group" in op_lower or "reduce" in op_lower:
                op_types.append(OperationType.AGGREGATION)
                has_agg = True
                has_shuffle = True
            elif "join" in op_lower:
                op_types.append(OperationType.JOIN)
                has_shuffle = True
                # Determine join type
                if "broadcast" in op_lower:
                    from spark_optima.core.simulation.performance_model import JoinType

                    join_details[i] = JoinType.BROADCAST_HASH
            elif "sort" in op_lower or "order" in op_lower:
                op_types.append(OperationType.SORT)
                has_shuffle = True
            elif "union" in op_lower:
                op_types.append(OperationType.UNION)
            elif "window" in op_lower:
                op_types.append(OperationType.WINDOW)
            elif "udf" in op_lower:
                op_types.append(OperationType.UDF)
            elif "cache" in op_lower or "persist" in op_lower:
                op_types.append(OperationType.CACHED)
            else:
                # Default to scan
                op_types.append(OperationType.SCAN)

        return OperationProfile(
            operations=op_types,
            join_details=join_details,
            has_aggregation=has_agg,
            has_shuffle=has_shuffle,
            estimated_stages=len(op_types),
        )

    def _store_trial_result(
        self,
        result: SimulationResult,
        config: dict[str, Any],
        resource_spec: ResourceSpec,
        data_profile: dict[str, Any] | None,
    ) -> None:
        """Store trial result for ML training.

        Args:
            result: Simulation result.
            config: Configuration used.
            resource_spec: Resource specifications.
            data_profile: Data characteristics.

        """
        trial_data = {
            "configuration": config,
            "resource_spec": resource_spec.to_dict(),
            "data_profile": data_profile or {},
            "execution_time_seconds": result.metrics.execution_time_seconds,
            "memory_peak_gb": result.metrics.memory_peak_gb,
            "success": result.metrics.success,
        }

        self._trial_history.append(trial_data)

    def train_ml_model(
        self,
        trials: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Train ML model with historical data.

        Args:
            trials: Optional list of trials to use. Uses internal history if None.

        Returns:
            Training metrics.

        """
        if self._ml_model is None:
            logger.warning("ML model not available")
            return {"error": "ML model not available"}

        data = trials if trials is not None else self._trial_history

        if len(data) < 10:
            logger.warning(f"Insufficient data for training: {len(data)} samples")
            return {"error": f"Insufficient data: {len(data)} samples"}

        return self._ml_model.train(data)

    def get_analytical_model(self) -> PerformanceModel:
        """Get the underlying analytical model.

        Returns:
            PerformanceModel instance.

        """
        return self._analytical_model

    def get_ml_model(self) -> MLPerformancePredictor | None:
        """Get the underlying ML model.

        Returns:
            MLPerformancePredictor instance or None.

        """
        return self._ml_model

    def get_trial_history(self) -> list[dict[str, Any]]:
        """Get stored trial history.

        Returns:
            List of trial data dictionaries.

        """
        return self._trial_history.copy()

    def clear_history(self) -> None:
        """Clear trial history."""
        self._trial_history.clear()
        logger.info("Trial history cleared")
