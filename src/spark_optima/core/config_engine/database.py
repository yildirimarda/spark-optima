# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Configuration database for Spark parameters.

This module provides the ConfigDatabase class for loading, querying,
and managing Spark configuration parameters across multiple versions.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import yaml

from spark_optima.core.config_engine.models import (
    ConfigParameter,
    ConfigSet,
    ParameterCategory,
    PlatformSupport,
)

logger = logging.getLogger(__name__)


class ConfigDatabase:
    """Database for Spark configuration parameters.

    This class manages configuration parameters across multiple Spark versions
    and provides methods for querying, filtering, and retrieving optimal
    configurations.

    Attributes:
        config_dir: Directory containing configuration YAML files.
        _configs: Dictionary mapping version strings to ConfigSet objects.

    Example:
        >>> db = ConfigDatabase()
        >>> params = db.get_parameters(version="3.5.0", category="memory")
        >>> memory_params = db.get_parameters_by_category("3.5.0", ParameterCategory.MEMORY)

    """

    def __init__(self, config_dir: str | Path | None = None) -> None:
        """Initialize the configuration database.

        Args:
            config_dir: Directory containing configuration YAML files.
                       Defaults to data/configs relative to package root.

        """
        if config_dir is None:
            # Default to package data/configs directory
            package_root = Path(__file__).parent.parent.parent.parent.parent
            config_dir = package_root / "data" / "configs"

        self.config_dir = Path(config_dir)
        self._configs: dict[str, ConfigSet] = {}
        self._load_all_configs()

    def _load_all_configs(self) -> None:
        """Load all configuration files from the config directory."""
        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
            return

        for config_file in sorted(self.config_dir.glob("spark_*_configs.yaml")):
            try:
                self._load_config_file(config_file)
            except (
                OSError,
                FileNotFoundError,
                yaml.YAMLError,
                ValueError,
                KeyError,
                AttributeError,
            ) as e:
                logger.error(f"Failed to load config file {config_file}: {e}")

    def _load_config_file(self, filepath: Path) -> None:
        """Load a single configuration file.

        Args:
            filepath: Path to the YAML configuration file.

        """
        with open(filepath, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Handle case where yaml.safe_load returns a list (e.g., multi-document YAML)
        if isinstance(data, list):
            # Find first dict in the list
            data = next((item for item in data if isinstance(item, dict)), None)
            if data is None:
                raise ValueError(f"Config file {filepath} contains no valid data")

        if not isinstance(data, dict):
            msg = f"Config file {filepath} must contain a YAML dict"
            msg += f", got {type(data).__name__}"
            raise ValueError(msg)

        version = data.get("version")
        if not version:
            raise ValueError(f"Config file {filepath} missing version")

        config_set = ConfigSet.from_dict(data)
        self._configs[version] = config_set
        logger.debug(f"Loaded {len(config_set)} parameters for Spark {version}")

    def get_available_versions(self) -> list[str]:
        """Get list of available Spark versions.

        Returns:
            Sorted list of version strings (e.g., ["3.0.0", "3.5.0", "4.0.0"]).

        """
        return sorted(self._configs.keys(), key=self._version_sort_key)

    @staticmethod
    def _version_sort_key(version: str) -> tuple[int, ...]:
        """Create sort key for version strings.

        Args:
            version: Version string like "3.5.0".

        Returns:
            Tuple of integers for proper version sorting.

        """
        return tuple(int(x) for x in version.split("."))

    def get_config_set(self, version: str) -> ConfigSet | None:
        """Get configuration set for a specific version.

        Args:
            version: Spark version string (e.g., "3.5.0").

        Returns:
            ConfigSet for the version, or None if not found.

        """
        return self._configs.get(version)

    def get_parameter(self, version: str, name: str) -> ConfigParameter | None:
        """Get a specific parameter for a version.

        Args:
            version: Spark version string.
            name: Parameter name (e.g., "spark.executor.memory").

        Returns:
            ConfigParameter if found, None otherwise.

        """
        config_set = self._configs.get(version)
        if not config_set:
            return None
        return config_set.parameters.get(name)

    def get_parameters(
        self,
        version: str,
        category: ParameterCategory | str | None = None,
        platform: PlatformSupport | str | None = None,
    ) -> dict[str, ConfigParameter]:
        """Get parameters filtered by category and/or platform.

        Args:
            version: Spark version string.
            category: Optional category filter.
            platform: Optional platform filter.

        Returns:
            Dictionary of parameter name to ConfigParameter.

        """
        config_set = self._configs.get(version)
        if not config_set:
            return {}

        parameters = config_set.parameters

        if category:
            if isinstance(category, str):
                category = ParameterCategory(category)
            parameters = {name: param for name, param in parameters.items() if param.category == category}

        if platform:
            parameters = {name: param for name, param in parameters.items() if param.is_applicable_for(platform)}

        return parameters

    def get_parameters_by_category(
        self,
        version: str,
        category: ParameterCategory | str,
    ) -> dict[str, ConfigParameter]:
        """Get parameters by category.

        Args:
            version: Spark version string.
            category: Category to filter by.

        Returns:
            Dictionary of matching parameters.

        """
        config_set = self._configs.get(version)
        if not config_set:
            return {}
        return config_set.get_parameters_by_category(category)

    def get_parameters_for_platform(
        self,
        version: str,
        platform: PlatformSupport | str,
    ) -> dict[str, ConfigParameter]:
        """Get parameters applicable for a platform.

        Args:
            version: Spark version string.
            platform: Platform to filter by.

        Returns:
            Dictionary of applicable parameters.

        """
        config_set = self._configs.get(version)
        if not config_set:
            return {}
        return config_set.get_parameters_for_platform(platform)

    def get_default_config(self, version: str) -> dict[str, Any]:
        """Get default configuration values for a version.

        Args:
            version: Spark version string.

        Returns:
            Dictionary of parameter names to default values.

        """
        config_set = self._configs.get(version)
        if not config_set:
            return {}
        return config_set.get_default_config()

    def search_parameters(
        self,
        version: str,
        query: str,
        case_sensitive: bool = False,
    ) -> dict[str, ConfigParameter]:
        """Search parameters by name or description.

        Args:
            version: Spark version string.
            query: Search query string.
            case_sensitive: Whether search is case sensitive.

        Returns:
            Dictionary of matching parameters.

        """
        config_set = self._configs.get(version)
        if not config_set:
            return {}

        if not case_sensitive:
            query = query.lower()

        results = {}
        for name, param in config_set.parameters.items():
            search_text = f"{name} {param.description}"
            if not case_sensitive:
                search_text = search_text.lower()

            if query in search_text:
                results[name] = param

        return results

    def get_recommended_config(
        self,
        version: str,
        resources: dict[str, Any],
        platform: PlatformSupport | str = PlatformSupport.LOCAL,
    ) -> dict[str, Any]:
        """Get recommended configuration based on resources.

        Args:
            version: Spark version string.
            resources: Resource specifications (memory_gb, cores, etc.).
            platform: Target platform.

        Returns:
            Dictionary of recommended configuration values.

        """
        config_set = self._configs.get(version)
        if not config_set:
            return {}

        recommended = {}
        # Handle ResourceSpec or dict
        if hasattr(resources, "cpu_cores"):
            # It's likely a ResourceSpec object
            available_vars = {"cpu_cores", "memory_gb"}
            if hasattr(resources, "disk_gb") and resources.disk_gb:
                available_vars.add("disk_gb")
        else:
            available_vars = set(resources.keys())

        for name, param in config_set.parameters.items():
            if not param.is_applicable_for(platform):
                continue

            if param.heuristic and param.heuristic.can_apply(available_vars):
                # Apply heuristic formula (simplified - would use formula evaluator)
                value = self._apply_heuristic(param.heuristic, resources)
                if value is not None:
                    recommended[name] = value
                    continue

            # Fall back to heuristic base_value or default
            if param.heuristic and param.heuristic.base_value is not None:
                recommended[name] = param.heuristic.base_value
            elif param.default is not None:
                recommended[name] = param.default

        return recommended

    def _apply_heuristic(self, heuristic: Any, resources: dict[str, Any]) -> Any | None:
        """Apply a heuristic rule to calculate a value.

        Args:
            heuristic: HeuristicRule to apply.
            resources: Available resource values.

        Returns:
            Calculated value or None if cannot apply.

        """
        if not heuristic.formula:
            return heuristic.base_value

        try:
            return self._evaluate_formula(heuristic.formula, resources)
        except (ValueError, KeyError, TypeError, ArithmeticError) as e:
            logger.debug(f"Formula evaluation failed for '{heuristic.formula}': {e}")
            return heuristic.base_value

    def _evaluate_formula(self, formula: str, variables: dict[str, Any]) -> Any:
        """Safely evaluate a mathematical formula string.

        Supports basic arithmetic (+, -, *, /, **, %), parentheses,
        and functions like min(), max(), round().

        Args:
            formula: Formula string (e.g., "memory_gb / num_executors * 0.9").
            variables: Dictionary of variable names to values.

        Returns:
            Calculated numeric result.

        Raises:
            ValueError: If formula is invalid or contains unsafe operations.
            KeyError: If a variable is not found in variables dict.

        """
        # Parse the formula into an AST
        try:
            tree = ast.parse(formula, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid formula syntax: {e}") from e

        # Walk the AST and evaluate safely
        return self._eval_node(tree.body, variables)

    def _eval_node(self, node: ast.AST, variables: dict[str, Any]) -> Any:
        """Recursively evaluate an AST node.

        Args:
            node: AST node to evaluate.
            variables: Variable dictionary.

        Returns:
            Evaluated value.

        Raises:
            ValueError: For unsupported or unsafe operations.

        """
        # Number literal
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value

        # Variable name
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise KeyError(f"Variable '{node.id}' not found in resources")

        # Binary operations (+, -, *, /, **, %)
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, variables)
            right = self._eval_node(node.right, variables)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

        # Unary operations (+, -, not)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, variables)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

        # Function calls (min, max, round)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Unsupported function call")
            func_name = node.func.id
            if func_name not in ("min", "max", "round"):
                raise ValueError(f"Unsupported function: {func_name}")
            args = [self._eval_node(arg, variables) for arg in node.args]
            if func_name == "min":
                return min(args)
            if func_name == "max":
                return max(args)
            if func_name == "round":
                if len(args) == 1:
                    return round(args[0])
                return round(args[0], args[1])

        # Comparison operations (for conditions)
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, variables)
            for op, comparator in zip(node.ops, node.comparators, strict=False):
                right = self._eval_node(comparator, variables)
                if isinstance(op, ast.Gt) and not left > right:
                    return False
                if isinstance(op, ast.GtE) and not left >= right:
                    return False
                if isinstance(op, ast.Lt) and not left < right:
                    return False
                if isinstance(op, ast.LtE) and not left <= right:
                    return False
                if isinstance(op, ast.Eq) and left != right:
                    return False
                if isinstance(op, ast.NotEq) and left == right:
                    return False
                left = right
            return True

        # Boolean operations (and, or)
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                result = self._eval_node(value, variables)
                if isinstance(node.op, ast.And) and not result:
                    return False
                if isinstance(node.op, ast.Or) and result:
                    return True
            return bool(result) if node.values else False

        # Parenthesized expression / tuple
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt, variables) for elt in node.elts)

        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    def get_parameter_count(self, version: str | None = None) -> int:
        """Get total number of parameters.

        Args:
            version: Specific version or None for all versions.

        Returns:
            Number of parameters.

        """
        if version:
            config_set = self._configs.get(version)
            return len(config_set) if config_set else 0

        return sum(len(cs) for cs in self._configs.values())

    def has_version(self, version: str) -> bool:
        """Check if a version is available.

        Args:
            version: Spark version string.

        Returns:
            True if version is available.

        """
        return version in self._configs

    def get_heuristic_config(
        self,
        version: str,
        resources: Any,
        platform: str = "local",
        data_profile: Any | None = None,
    ) -> dict[str, Any]:
        """Get optimized configuration using heuristic engine.

        This method uses the HeuristicEngine to generate optimal
        Spark configuration based on resources and platform.

        Args:
            version: Spark version string.
            resources: ResourceSpec with cpu_cores and memory_gb.
            platform: Target platform (local, aws_glue, databricks, azure_synapse).
            data_profile: Optional DataProfile for data-specific optimizations.

        Returns:
            Dictionary of optimized configuration parameters.

        Example:
            >>> db = ConfigDatabase()
            >>> from spark_optima.platforms.models import ResourceSpec
            >>> resources = ResourceSpec(cpu_cores=16, memory_gb=64)
            >>> config = db.get_heuristic_config("3.5.0", resources, "local")

        """
        from spark_optima.core.heuristics import HeuristicEngine

        config_set = self._configs.get(version)
        if not config_set:
            return {}

        engine = HeuristicEngine(config_set)
        return engine.evaluate(
            resources=resources,
            platform=platform,
            spark_version=version,
            data_profile=data_profile,
        )

    def reload(self) -> None:
        """Reload all configuration files."""
        self._configs.clear()
        self._load_all_configs()
        logger.info("Configuration database reloaded")

    def __contains__(self, version: str) -> bool:
        """Check if version is in database."""
        return version in self._configs

    def __len__(self) -> int:
        """Return number of loaded versions."""
        return len(self._configs)
