# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Main HeuristicEngine for Spark configuration optimization.

This module provides the HeuristicEngine class that orchestrates
the heuristic rule evaluation and generates optimal Spark configurations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from spark_optima.core.config_engine.database import ConfigDatabase
from spark_optima.core.config_engine.models import ConfigSet, ParameterCategory
from spark_optima.core.heuristics.context import DataProfile, EvaluationContext
from spark_optima.core.heuristics.evaluator import FormulaError, FormulaEvaluator
from spark_optima.core.heuristics.rules import RuleRegistry

if TYPE_CHECKING:
    from spark_optima.platforms.models import ResourceSpec

logger = logging.getLogger(__name__)


class HeuristicEngine:
    """Main engine for heuristic-based Spark configuration optimization.

    This class orchestrates the evaluation of heuristic rules to generate
    optimal Spark configurations based on resources, platform, and data characteristics.

    Attributes:
        config_set: Spark configuration set for a specific version.
        rule_registry: Registry of heuristic rules.
        formula_evaluator: Safe formula evaluator.

    Example:
        >>> from spark_optima.core.config_engine import ConfigDatabase
        >>> db = ConfigDatabase()
        >>> config_set = db.get_config_set("3.5.0")
        >>> engine = HeuristicEngine(config_set)
        >>> result = engine.evaluate(
        ...     resources=ResourceSpec(cpu_cores=16, memory_gb=64),
        ...     platform="local",
        ...     data_profile=DataProfile(format="parquet", size_gb=100)
        ... )

    """

    def __init__(self, config_set: ConfigSet | None = None) -> None:
        """Initialize the heuristic engine.

        Args:
            config_set: Spark configuration set. If None, loads default (3.5.0).

        """
        if config_set is None:
            db = ConfigDatabase()
            self.config_set = db.get_config_set("3.5.0")
        else:
            self.config_set = config_set

        self.rule_registry = RuleRegistry()
        self.formula_evaluator = FormulaEvaluator()
        self._evaluated_config: dict[str, Any] = {}

    def evaluate(
        self,
        resources: ResourceSpec,
        platform: str = "local",
        spark_version: str | None = None,
        data_profile: DataProfile | None = None,
        custom_vars: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate heuristics and generate optimal configuration.

        Args:
            resources: Available system resources.
            platform: Target platform (local, aws_glue, databricks, azure_synapse).
            spark_version: Spark version. Uses config_set version if None.
            data_profile: Data characteristics.
            custom_vars: Additional custom variables for formulas.

        Returns:
            Dictionary of recommended configuration parameters.

        """
        if self.config_set is None:
            raise ValueError("config_set must not be None")
        version = spark_version or self.config_set.version
        data_profile = data_profile or DataProfile()

        # Create evaluation context
        context = EvaluationContext(
            resources=resources,
            platform=platform,
            spark_version=version,
            data_profile=data_profile,
        )

        # Add custom variables
        if custom_vars:
            for key, value in custom_vars.items():
                context.set(key, value)

        # Calculate base resource allocation first
        self._calculate_base_resources(context, platform)

        # Get applicable rules sorted by priority
        variables = set(context.to_variables().keys())
        rules = self.rule_registry.get_applicable_rules(
            variables=variables,
            platform=platform,
            version=version,
        )

        # Sort by priority (high -> medium -> low)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        rules.sort(key=lambda r: priority_order.get(r.priority, 3))

        # Evaluate rules and build configuration
        config: dict[str, Any] = {}
        evaluated_vars: dict[str, Any] = context.to_variables()

        for rule in rules:
            try:
                value = self._evaluate_rule(rule, evaluated_vars)
                if value is not None:
                    config[rule.param_name] = value
                    # Make evaluated value available for dependent rules
                    var_name = rule.param_name.replace(".", "_")
                    evaluated_vars[var_name] = value
                    evaluated_vars[rule.param_name] = value
            except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning(f"Failed to evaluate rule {rule.param_name}: {e}")
                # Fall back to base_value if available
                if rule.base_value is not None:
                    config[rule.param_name] = rule.base_value

        self._evaluated_config = config
        return config

    def _calculate_base_resources(self, context: EvaluationContext, platform: str) -> None:
        """Calculate base resource allocation for executors and driver.

        Args:
            context: Evaluation context to update.
            platform: Target platform.

        """
        resources = context.resources

        # Determine number of executors
        if platform == "local":
            num_executors = 1
            executor_cores = max(1, resources.cpu_cores - 1)  # Leave 1 core for OS
        else:
            # For distributed platforms, target ~4-8 cores per executor
            num_executors = max(2, resources.cpu_cores // 4)
            executor_cores = 4

        # Calculate memory per executor
        driver_memory_gb = min(resources.memory_gb * 0.1, 16)
        available_memory = resources.memory_gb - driver_memory_gb
        executor_memory_gb = min(
            (available_memory / num_executors) * 0.9,  # 10% buffer
            64,  # Cap at 64GB per executor
        )

        context.update_calculated_values(
            num_executors=num_executors,
            executor_cores=executor_cores,
            executor_memory_gb=executor_memory_gb,
            driver_memory_gb=driver_memory_gb,
        )

    def _evaluate_rule(
        self,
        rule: Any,
        variables: dict[str, Any],
    ) -> Any | None:
        """Evaluate a single heuristic rule.

        Args:
            rule: Heuristic rule definition.
            variables: Available variables.

        Returns:
            Evaluated value or None if cannot evaluate.

        """
        if not rule.formula:
            return rule.base_value

        try:
            # Evaluate formula
            result = self.formula_evaluator.evaluate(rule.formula, variables)

            # Format result based on parameter type
            if self.config_set is None:
                raise ValueError("config_set must not be None")
            param = self.config_set.parameters.get(rule.param_name)
            if param:
                result = self._format_value(result, param.param_type.value)

            return result
        except FormulaError as e:
            logger.debug(f"Formula evaluation failed for {rule.param_name}: {e}")
            return rule.base_value
        except (ValueError, TypeError, AttributeError, ZeroDivisionError) as e:
            logger.warning(f"Unexpected error evaluating {rule.param_name}: {e}")
            return rule.base_value

    def _format_value(self, value: Any, param_type: str) -> Any:
        """Format value according to parameter type.

        Args:
            value: Raw value.
            param_type: Parameter type (integer, float, bytes, etc.).

        Returns:
            Formatted value.

        """
        if param_type == "integer":
            return int(value)
        elif param_type == "float":
            return float(value)
        elif param_type == "boolean":
            return str(value).lower() in ("true", "1", "yes", "on")
        elif param_type == "bytes":
            # Format as human-readable bytes string
            if isinstance(value, int | float):
                return FormulaEvaluator.format_bytes(int(value))
            return str(value)
        elif param_type == "duration":
            # Format as human-readable duration string
            if isinstance(value, int | float):
                return FormulaEvaluator.format_duration(int(value))
            return str(value)
        else:
            return str(value)

    def get_config_by_category(self, category: ParameterCategory) -> dict[str, Any]:
        """Get evaluated configuration filtered by category.

        Args:
            category: Parameter category.

        Returns:
            Filtered configuration dictionary.

        """
        if not self._evaluated_config:
            return {}

        if self.config_set is None:
            raise ValueError("config_set must not be None")
        params_in_category = self.config_set.get_parameters_by_category(category)
        return {k: v for k, v in self._evaluated_config.items() if k in params_in_category}

    def get_memory_config(self) -> dict[str, Any]:
        """Get memory-related configuration.

        Returns:
            Memory configuration dictionary.

        """
        return self.get_config_by_category(ParameterCategory.MEMORY)

    def get_cpu_config(self) -> dict[str, Any]:
        """Get CPU-related configuration.

        Returns:
            CPU configuration dictionary.

        """
        return self.get_config_by_category(ParameterCategory.CPU)

    def get_shuffle_config(self) -> dict[str, Any]:
        """Get shuffle-related configuration.

        Returns:
            Shuffle configuration dictionary.

        """
        return self.get_config_by_category(ParameterCategory.SHUFFLE)

    def get_sql_config(self) -> dict[str, Any]:
        """Get SQL-related configuration.

        Returns:
            SQL configuration dictionary.

        """
        return self.get_config_by_category(ParameterCategory.SQL)

    def validate_config(self, config: dict[str, Any] | None = None) -> list[str]:
        """Validate configuration against constraints.

        Args:
            config: Configuration to validate. Uses last evaluated if None.

        Returns:
            List of validation errors (empty if valid).

        """
        config = config or self._evaluated_config
        errors = []

        if self.config_set is None:
            raise ValueError("config_set must not be None")
        for param_name, value in config.items():
            param = self.config_set.parameters.get(param_name)
            if param and param.constraints:
                # Validate against constraints
                if param.constraints.min_value is not None:
                    try:
                        if float(value) < param.constraints.min_value:
                            errors.append(
                                f"{param_name}: {value} < min {param.constraints.min_value}",
                            )
                    except (ValueError, TypeError):
                        pass

                if param.constraints.max_value is not None:
                    try:
                        if float(value) > param.constraints.max_value:
                            errors.append(
                                f"{param_name}: {value} > max {param.constraints.max_value}",
                            )
                    except (ValueError, TypeError):
                        pass

        return errors
