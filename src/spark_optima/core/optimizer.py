# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Main Optimizer class for Spark configuration optimization.

This module provides the primary interface for optimizing Apache Spark
configurations using a hybrid approach combining heuristics and Bayesian
optimization.
"""

import logging
from pathlib import Path
from typing import Any

from spark_optima.analysis import RecommendationEngine
from spark_optima.core.bayesian.models import SearchSpaceConfig
from spark_optima.core.bayesian.optimizer import BayesianOptimizer
from spark_optima.core.config_engine.database import ConfigDatabase
from spark_optima.core.heuristics.context import DataProfile
from spark_optima.core.heuristics.engine import HeuristicEngine
from spark_optima.core.result import CodeSuggestion, OptimizationResult
from spark_optima.platforms.models import ResourceSpec

logger = logging.getLogger(__name__)


class Optimizer:
    """Main optimizer for Spark configuration tuning.

    This class provides an interface to optimize Spark configurations for
    specific workloads using a hybrid optimization approach. It combines
    heuristic rules for initial configuration with Bayesian optimization
    for fine-tuning.

    The optimization process follows these steps:
    1. Generate initial configuration using heuristics (Phase 4)
    2. Fine-tune using Bayesian optimization (Phase 5)
    3. Return optimal configuration with performance estimates

    Attributes:
        platform: Target platform (local, databricks, aws_glue, azure_synapse).
        spark_version: Spark version to optimize for (e.g., "3.5.0").
        optimization_mode: Either "simulation" or "execution".
        config_database: Database of Spark configuration parameters.
        heuristic_engine: Engine for heuristic-based optimization.

    Example:
        >>> optimizer = Optimizer(platform="databricks", spark_version="3.5.0")
        >>> result = optimizer.optimize(
        ...     code_path="./spark_job.py",
        ...     data_profile={"size_gb": 100, "format": "parquet"},
        ...     resources=ResourceSpec(cpu_cores=16, memory_gb=64),
        ...     use_bayesian=True,
        ...     n_trials=50
        ... )
        >>> print(result.configuration)

    """

    def __init__(
        self,
        platform: str,
        spark_version: str = "3.5.0",
        optimization_mode: str = "simulation",
    ) -> None:
        """Initialize the optimizer.

        Args:
            platform: Target platform (local, databricks, aws_glue, azure_synapse).
            spark_version: Spark version to optimize for.
            optimization_mode: Either "simulation" or "execution".

        Raises:
            ValueError: If platform or optimization_mode is invalid.

        """
        valid_platforms = ["local", "databricks", "aws_glue", "azure_synapse"]
        valid_modes = ["simulation", "execution"]

        if platform not in valid_platforms:
            raise ValueError(f"Invalid platform. Must be one of: {valid_platforms}")
        if optimization_mode not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of: {valid_modes}")

        self.platform = platform
        self.spark_version = spark_version
        self.optimization_mode = optimization_mode

        # Initialize configuration database
        self.config_database = ConfigDatabase()

        # Get configuration set for the specified version
        self.config_set = self.config_database.get_config_set(spark_version)
        if self.config_set is None:
            available = self.config_database.get_available_versions()
            raise ValueError(
                f"Spark version {spark_version} not available. Available versions: {available}",
            )

        # Initialize heuristic engine
        self.heuristic_engine = HeuristicEngine(self.config_set)

        # Results storage
        self._last_result: OptimizationResult | None = None
        self._heuristic_config: dict[str, Any] | None = None
        self._bayesian_result: Any | None = None

    def optimize(
        self,
        code_path: str | Path | None = None,
        resources: ResourceSpec | None = None,
        data_profile: dict[str, Any] | None = None,
        resource_constraints: dict[str, Any] | None = None,
        use_bayesian: bool = True,
        bayesian_trials: int = 50,
        bayesian_timeout_minutes: int | None = None,
        objectives: list[str] | None = None,
    ) -> OptimizationResult:
        """Run optimization for the given Spark code.

        This method analyzes the provided Spark code and data characteristics
        to find the optimal configuration using hybrid optimization.

        Args:
            code_path: Path to the Spark application code file (optional).
            resources: Resource specifications (cpu_cores, memory_gb).
            data_profile: Data characteristics including size, format, schema.
            resource_constraints: Resource limits such as max memory or cost.
            use_bayesian: Whether to use Bayesian optimization after heuristics.
            bayesian_trials: Number of Bayesian optimization trials.
            bayesian_timeout_minutes: Timeout for Bayesian optimization.
            objectives: List of optimization objectives (e.g., ["minimize_time"]).

        Returns:
            OptimizationResult containing the optimal configuration and metadata.

        Raises:
            FileNotFoundError: If the code file does not exist.
            ValueError: If parameters are invalid.

        """
        # Validate code path if provided
        code_path_obj: Path | None = None
        if code_path is not None:
            code_path_obj = Path(code_path)
            if not code_path_obj.exists():
                raise FileNotFoundError(f"Code file not found: {code_path_obj}")

        # Set default resources if not provided
        if resources is None:
            resources = ResourceSpec(cpu_cores=4, memory_gb=16)

        # Create data profile
        data_prof = None
        if data_profile:
            data_prof = DataProfile(
                size_gb=data_profile.get("size_gb", 10),
                format=data_profile.get("format", "parquet"),
                schema=data_profile.get("schema"),
                compression=data_profile.get("compression"),
                partitioning=data_profile.get("partitioning"),
            )

        # Phase 1: Generate heuristic-based initial configuration
        heuristic_config = self.heuristic_engine.evaluate(
            resources=resources,
            platform=self.platform,
            spark_version=self.spark_version,
            data_profile=data_prof,
            custom_vars=resource_constraints,
        )
        self._heuristic_config = heuristic_config

        # Validate heuristic configuration
        validation_errors = self.heuristic_engine.validate_config(heuristic_config)
        if validation_errors:
            # Log warnings but continue
            for error in validation_errors:
                logger.warning(f"Validation warning: {error}")

        # Phase 2: Bayesian optimization (if enabled)
        if use_bayesian:
            try:
                # config_set is guaranteed to be non-None here (checked in __init__)
                if self.config_set is None:
                    raise AssertionError("config_set is None")

                bayesian_config = self._run_bayesian_optimization(
                    heuristic_config=heuristic_config,
                    resources=resources,
                    data_profile=data_profile,
                    n_trials=bayesian_trials,
                    timeout_minutes=bayesian_timeout_minutes,
                    objectives=objectives,
                    code_path=code_path_obj,
                )
                final_config = bayesian_config
            except (RuntimeError, ValueError, KeyError, AttributeError, TypeError) as e:
                # Fall back to heuristic if Bayesian fails
                logger.warning(f"Bayesian optimization failed: {e}")
                logger.warning("Falling back to heuristic configuration")
                final_config = heuristic_config
                self._bayesian_result = None
        else:
            final_config = heuristic_config
            self._bayesian_result = None

        # Phase 3: Code analysis (if code path provided)
        code_suggestions = []
        analysis_metadata = {}
        if code_path_obj is not None:
            try:
                code_analysis = self._analyze_code(code_path_obj)
                code_suggestions = code_analysis["suggestions"]
                analysis_metadata = code_analysis["metadata"]
            except (OSError, RuntimeError, ValueError, AttributeError, TypeError, SyntaxError) as e:
                logger.warning(f"Code analysis failed: {e}")

        # Build optimization result
        result = self._build_result(
            final_config=final_config,
            heuristic_config=heuristic_config,
            resources=resources,
            data_profile=data_prof,
            code_suggestions=code_suggestions,
            analysis_metadata=analysis_metadata,
        )

        self._last_result = result
        return result

    def _run_bayesian_optimization(
        self,
        heuristic_config: dict[str, Any],
        resources: ResourceSpec,
        data_profile: dict[str, Any] | None,
        n_trials: int,
        timeout_minutes: int | None,
        objectives: list[str] | None,
        code_path: Path | None = None,
    ) -> dict[str, Any]:
        """Run Bayesian optimization to fine-tune heuristic configuration.

        Args:
            heuristic_config: Initial configuration from heuristics.
            resources: Resource specifications.
            data_profile: Data characteristics.
            n_trials: Number of optimization trials.
            timeout_minutes: Optimization timeout.
            objectives: Optimization objectives.
            code_path: Path to the Spark application code file (for execution mode).

        Returns:
            Optimized configuration dictionary.

        """
        # Ensure config_set is available
        if self.config_set is None:
            raise ValueError("config_set is None - cannot run Bayesian optimization")

        # Create Bayesian optimizer
        bayesian_optimizer = BayesianOptimizer(
            heuristic_config=heuristic_config,
            config_set=self.config_set,
            resource_spec=resources,
            search_space_config=SearchSpaceConfig(),
            objectives=objectives or ["minimize_time"],
            mode=self.optimization_mode,
            study_name=f"spark_optima_{self.platform}_{self.spark_version}",
            code_path=code_path,
        )

        # Run optimization
        bayesian_result = bayesian_optimizer.optimize(
            n_trials=n_trials,
            timeout_minutes=timeout_minutes,
            n_jobs=1,  # Sequential for stability
            show_progress=True,
            data_profile=data_profile,
        )

        # Store result
        self._bayesian_result = bayesian_result

        # Return best configuration
        return bayesian_result.best_config if bayesian_result.best_config else heuristic_config

    def _analyze_code(self, code_path: Path) -> dict[str, Any]:
        """Analyze Spark code for optimization opportunities.

        Args:
            code_path: Path to the Python file.

        Returns:
            Dictionary with suggestions and metadata.

        """
        engine = RecommendationEngine()
        analysis = engine.analyze_file(str(code_path))

        # Convert recommendations to CodeSuggestion objects
        suggestions = []
        for rec in analysis.recommendations:
            suggestion = CodeSuggestion(
                line_number=rec.smell.location.line if rec.smell.location else 0,
                issue_type=rec.smell.smell_type,
                description=rec.smell.description,
                suggestion=rec.suggestion,
                severity=rec.smell.severity.value,
            )
            suggestions.append(suggestion)

        return {
            "suggestions": suggestions,
            "metadata": {
                "operations_count": len(analysis.operations),
                "smells_count": len(analysis.smells),
                "recommendations_count": len(analysis.recommendations),
            },
        }

    def _build_result(
        self,
        final_config: dict[str, Any],
        heuristic_config: dict[str, Any],
        resources: ResourceSpec,
        data_profile: DataProfile | None,
        code_suggestions: list[CodeSuggestion] | None = None,
        analysis_metadata: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Build optimization result from configuration.

        Args:
            final_config: Final optimized configuration.
            heuristic_config: Initial heuristic configuration.
            resources: Resource specifications.
            data_profile: Data profile.
            code_suggestions: List of code improvement suggestions.
            analysis_metadata: Code analysis metadata.

        Returns:
            OptimizationResult with metadata.

        """
        # Estimate performance metrics
        estimated_time = self._estimate_execution_time(final_config, resources, data_profile)

        # Calculate confidence score
        confidence = self._calculate_confidence()

        # Build platform-specific config
        platform_specific = self._build_platform_config(final_config)

        # Build metadata
        metadata: dict[str, Any] = {
            "platform": self.platform,
            "spark_version": self.spark_version,
            "optimization_mode": self.optimization_mode,
            "heuristic_config": heuristic_config,
            "resources": resources.to_dict() if resources else {},
            "data_profile": {
                "size_gb": data_profile.size_gb,
                "format": data_profile.format,
                "schema": data_profile.schema,
                "compression": data_profile.compression,
                "partitioning": data_profile.partitioning,
            }
            if data_profile
            else {},
            "bayesian_used": self._bayesian_result is not None,
            "bayesian_trials": (
                len(self._bayesian_result.all_trials) if self._bayesian_result else 0
            ),
        }

        # Add code analysis metadata if available
        if analysis_metadata:
            metadata["code_analysis"] = analysis_metadata

        # Build result
        return OptimizationResult(
            configuration=final_config,
            estimated_time_minutes=estimated_time,
            confidence_score=confidence,
            code_suggestions=code_suggestions or [],
            platform_specific=platform_specific,
            metadata=metadata,
        )

    def _estimate_execution_time(
        self,
        config: dict[str, Any],
        resources: ResourceSpec,
        data_profile: DataProfile | None,
    ) -> float:
        """Estimate execution time for the configuration.

        Args:
            config: Spark configuration.
            resources: Resource specifications.
            data_profile: Data characteristics.

        Returns:
            Estimated execution time in minutes.

        """
        # Use simulation model from trial_runner
        from spark_optima.core.bayesian.trial_runner import SimulationModel

        model = SimulationModel()
        data_prof_dict = data_profile.__dict__ if data_profile else {"size_gb": 10}

        metrics = model.estimate(
            config=config,
            resource_spec=resources,
            cost_model=None,
            data_profile=data_prof_dict,
        )

        # Convert seconds to minutes
        return metrics.execution_time_seconds / 60.0

    def _calculate_confidence(self) -> float:
        """Calculate confidence score for the optimization.

        Returns:
            Confidence score between 0.0 and 1.0.

        """
        base_confidence = 0.7

        # Increase confidence if Bayesian optimization was used
        if self._bayesian_result is not None:
            base_confidence += 0.15

            # More trials = higher confidence (up to a point)
            n_trials = len(self._bayesian_result.all_trials)
            trial_bonus = min(0.1, n_trials / 1000)
            base_confidence += trial_bonus

        # Cap at 0.95
        return min(0.95, base_confidence)

    def _build_platform_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Build platform-specific configuration.

        Args:
            config: Generic Spark configuration.

        Returns:
            Platform-specific configuration.

        """
        platform_config: dict[str, Any] = {
            "platform": self.platform,
            "spark_version": self.spark_version,
        }

        # Add platform-specific settings
        if self.platform == "databricks":
            platform_config["cluster_config"] = {
                "spark_version": f"{self.spark_version}.x-scala2.12",
                "node_type_id": "Standard_DS3_v2",
            }
        elif self.platform == "aws_glue":
            platform_config["glue_version"] = "4.0"
        elif self.platform == "azure_synapse":
            platform_config["spark_pool_version"] = self.spark_version

        # Add relevant Spark configurations
        relevant_keys = [
            "spark.executor.memory",
            "spark.executor.cores",
            "spark.driver.memory",
            "spark.sql.adaptive.enabled",
            "spark.dynamicAllocation.enabled",
        ]

        platform_config["spark_config"] = {k: v for k, v in config.items() if k in relevant_keys}

        return platform_config

    def get_heuristic_config(self) -> dict[str, Any] | None:
        """Get the heuristic baseline configuration.

        Returns:
            Heuristic configuration or None if not run.

        """
        return self._heuristic_config.copy() if self._heuristic_config else None

    def get_bayesian_result(self) -> Any | None:
        """Get the Bayesian optimization result.

        Returns:
            BayesianOptimizationResult or None if not run.

        """
        return self._bayesian_result

    def get_last_result(self) -> OptimizationResult | None:
        """Get the last optimization result.

        Returns:
            OptimizationResult or None if not run.

        """
        return self._last_result

    def __repr__(self) -> str:
        """Return string representation of the optimizer."""
        return (
            f"Optimizer("
            f"platform='{self.platform}', "
            f"spark_version='{self.spark_version}', "
            f"mode='{self.optimization_mode}'"
            f")"
        )
