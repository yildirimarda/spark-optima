# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Pydantic models for Spark configuration engine.

This module defines the data structures for representing Spark configuration
parameters, their validation rules, and heuristic optimization rules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ParameterCategory(str, Enum):
    """Categories of Spark configuration parameters."""

    MEMORY = "memory"
    CPU = "cpu"
    SHUFFLE = "shuffle"
    SERIALIZATION = "serialization"
    SQL = "sql"
    DYNAMIC_ALLOCATION = "dynamic_allocation"
    RUNTIME = "runtime"
    IO = "io"
    NETWORK = "network"
    SECURITY = "security"
    SCHEDULER = "scheduler"
    UI_HISTORY = "ui_history"


class ParameterType(str, Enum):
    """Data types for configuration parameter values."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    BYTES = "bytes"  # String with unit like "4g", "512m"
    DURATION = "duration"  # String like "60s", "5m", "1h"
    LIST = "list"
    MAP = "map"


class PlatformSupport(str, Enum):
    """Supported platforms for Spark deployment."""

    LOCAL = "local"
    DATABRICKS = "databricks"
    AWS_GLUE = "aws_glue"
    AZURE_SYNAPSE = "azure_synapse"
    ALL = "all"


class ValidationConstraint:
    """Validation constraints for a configuration parameter.

    Attributes:
        min_value: Minimum numeric value (for int/float/bytes).
        max_value: Maximum numeric value (for int/float/bytes).
        pattern: Regex pattern for string validation.
        allowed_values: List of allowed enum values.
        depends_on: Other parameters this constraint depends on.

    """

    def __init__(
        self,
        min_value: int | float | None = None,
        max_value: int | float | None = None,
        pattern: str | None = None,
        allowed_values: list[Any] | None = None,
        depends_on: list[str] | None = None,
    ) -> None:
        """Initialize validation constraints.

        Args:
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.
            pattern: Regex pattern for string matching.
            allowed_values: List of valid values.
            depends_on: List of dependent parameter names.

        """
        self.min_value = min_value
        self.max_value = max_value
        self.pattern = pattern
        self.allowed_values = allowed_values
        self.depends_on = depends_on or []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "pattern": self.pattern,
            "allowed_values": self.allowed_values,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationConstraint:
        """Create from dictionary representation."""
        return cls(
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            pattern=data.get("pattern"),
            allowed_values=data.get("allowed_values"),
            depends_on=data.get("depends_on", []),
        )


class HeuristicRule:
    """Heuristic rule for calculating recommended parameter values.

    Attributes:
        formula: Mathematical formula or logic for calculation.
        base_value: Default base value if formula cannot be applied.
        priority: Priority level (high/medium/low) for rule application.
        conditions: Conditions when this rule applies.
        depends_on: Input variables this rule depends on.
        description: Human-readable description of the rule.

    """

    def __init__(
        self,
        formula: str | None = None,
        base_value: Any = None,
        priority: str = "medium",
        conditions: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
        description: str = "",
    ) -> None:
        """Initialize heuristic rule.

        Args:
            formula: Formula string like "memory_gb / num_executors * 0.9".
            base_value: Fallback value if formula fails.
            priority: Rule priority (high/medium/low).
            conditions: Dict of conditions for rule application.
            depends_on: List of required input variables.
            description: Human-readable rule description.

        Raises:
            ValueError: If priority is invalid.

        """
        valid_priorities = ["high", "medium", "low"]
        if priority not in valid_priorities:
            raise ValueError(f"Priority must be one of: {valid_priorities}")

        self.formula = formula
        self.base_value = base_value
        self.priority = priority
        self.conditions = conditions or {}
        self.depends_on = depends_on or []
        self.description = description

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "formula": self.formula,
            "base_value": self.base_value,
            "priority": self.priority,
            "conditions": self.conditions,
            "depends_on": self.depends_on,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeuristicRule:
        """Create from dictionary representation."""
        return cls(
            formula=data.get("formula"),
            base_value=data.get("base_value"),
            priority=data.get("priority", "medium"),
            conditions=data.get("conditions", {}),
            depends_on=data.get("depends_on", []),
            description=data.get("description", ""),
        )

    def can_apply(self, available_vars: set[str]) -> bool:
        """Check if rule can be applied with available variables.

        Args:
            available_vars: Set of available variable names.

        Returns:
            True if all dependencies are satisfied.

        """
        return all(dep in available_vars for dep in self.depends_on)


class ConfigParameter:
    """Represents a single Spark configuration parameter.

    This class encapsulates all metadata about a Spark configuration parameter
    including its type, default value, validation rules, and heuristic rules
    for optimization.

    Attributes:
        name: Full parameter name (e.g., "spark.executor.memory").
        category: Parameter category (memory, cpu, shuffle, etc.).
        param_type: Data type of the parameter value.
        default: Default value if not specified.
        description: Human-readable description.
        since_version: Spark version when parameter was introduced.
        deprecated_in: Spark version when parameter was deprecated.
        alternatives: Alternative parameters if deprecated.
        constraints: Validation constraints.
        applicable_platforms: Platforms where this parameter applies.
        heuristic: Heuristic rule for optimization.
        is_advanced: Whether this is an advanced parameter.
        doc_url: URL to official Spark documentation.

    """

    def __init__(
        self,
        name: str,
        category: ParameterCategory,
        param_type: ParameterType,
        default: Any = None,
        description: str = "",
        since_version: str = "3.0.0",
        deprecated_in: str | None = None,
        alternatives: list[str] | None = None,
        constraints: ValidationConstraint | None = None,
        applicable_platforms: list[PlatformSupport] | None = None,
        heuristic: HeuristicRule | None = None,
        is_advanced: bool = False,
        doc_url: str = "",
    ) -> None:
        """Initialize configuration parameter.

        Args:
            name: Full parameter name.
            category: Parameter category.
            param_type: Data type.
            default: Default value.
            description: Human-readable description.
            since_version: Introduced in version.
            deprecated_in: Deprecated in version.
            alternatives: Alternative parameters.
            constraints: Validation constraints.
            applicable_platforms: Supported platforms.
            heuristic: Heuristic optimization rule.
            is_advanced: Advanced parameter flag.
            doc_url: Documentation URL.

        """
        self.name = name
        self.category = category
        self.param_type = param_type
        self.default = default
        self.description = description
        self.since_version = since_version
        self.deprecated_in = deprecated_in
        self.alternatives = alternatives or []
        self.constraints = constraints or ValidationConstraint()
        self.applicable_platforms = applicable_platforms or [
            PlatformSupport.LOCAL,
            PlatformSupport.DATABRICKS,
            PlatformSupport.AWS_GLUE,
            PlatformSupport.AZURE_SYNAPSE,
        ]
        self.heuristic = heuristic
        self.is_advanced = is_advanced
        self.doc_url = doc_url

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "category": self.category.value,
            "type": self.param_type.value,
            "default": self.default,
            "description": self.description,
            "since_version": self.since_version,
            "deprecated_in": self.deprecated_in,
            "alternatives": self.alternatives,
            "constraints": self.constraints.to_dict(),
            "applicable_platforms": [p.value for p in self.applicable_platforms],
            "heuristic": self.heuristic.to_dict() if self.heuristic else None,
            "is_advanced": self.is_advanced,
            "doc_url": self.doc_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigParameter:
        """Create from dictionary representation."""
        heuristic_data = data.get("heuristic")
        heuristic = HeuristicRule.from_dict(heuristic_data) if heuristic_data else None

        constraints_data = data.get("constraints", {})
        constraints = ValidationConstraint.from_dict(constraints_data)

        return cls(
            name=data["name"],
            category=ParameterCategory(data["category"]),
            param_type=ParameterType(data["type"]),
            default=data.get("default"),
            description=data.get("description", ""),
            since_version=data.get("since_version", "3.0.0"),
            deprecated_in=data.get("deprecated_in"),
            alternatives=data.get("alternatives", []),
            constraints=constraints,
            applicable_platforms=[PlatformSupport(p) for p in data.get("applicable_platforms", [])],
            heuristic=heuristic,
            is_advanced=data.get("is_advanced", False),
            doc_url=data.get("doc_url", ""),
        )

    def is_applicable_for(self, platform: PlatformSupport | str) -> bool:
        """Check if parameter is applicable for given platform.

        Args:
            platform: Platform to check.

        Returns:
            True if parameter applies to the platform.

        """
        if isinstance(platform, str):
            platform = PlatformSupport(platform)
        return platform in self.applicable_platforms

    def is_deprecated_in(self, version: str) -> bool:
        """Check if parameter is deprecated in given version.

        Args:
            version: Spark version to check.

        Returns:
            True if deprecated in or before the version.

        """
        if self.deprecated_in is None:
            return False
        return self._compare_versions(version, self.deprecated_in) >= 0

    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        """Compare two version strings.

        Args:
            v1: First version string.
            v2: Second version string.

        Returns:
            Negative if v1 < v2, 0 if equal, positive if v1 > v2.

        """
        parts1 = [int(x) for x in v1.split(".")]
        parts2 = [int(x) for x in v2.split(".")]

        for p1, p2 in zip(parts1, parts2, strict=False):
            if p1 != p2:
                return p1 - p2

        return len(parts1) - len(parts2)


class ConfigSet:
    """A complete set of configuration parameters for a Spark version.

    Attributes:
        version: Spark version (e.g., "3.5.0").
        parameters: Dictionary of parameter name to ConfigParameter.
        metadata: Additional metadata about this config set.

    """

    def __init__(
        self,
        version: str,
        parameters: dict[str, ConfigParameter] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize configuration set.

        Args:
            version: Spark version string.
            parameters: Dictionary of configuration parameters.
            metadata: Additional metadata.

        """
        self.version = version
        self.parameters = parameters or {}
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "version": self.version,
            "parameters": {name: param.to_dict() for name, param in self.parameters.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigSet:
        """Create from dictionary representation."""
        parameters = {
            name: ConfigParameter.from_dict({**param_data, "name": name})
            for name, param_data in data.get("parameters", {}).items()
        }
        return cls(
            version=data["version"],
            parameters=parameters,
            metadata=data.get("metadata", {}),
        )

    def get_parameters_by_category(
        self,
        category: ParameterCategory | str,
    ) -> dict[str, ConfigParameter]:
        """Get parameters filtered by category.

        Args:
            category: Category to filter by.

        Returns:
            Dictionary of matching parameters.

        """
        if isinstance(category, str):
            category = ParameterCategory(category)

        return {name: param for name, param in self.parameters.items() if param.category == category}

    def get_parameters_for_platform(
        self,
        platform: PlatformSupport | str,
    ) -> dict[str, ConfigParameter]:
        """Get parameters applicable for given platform.

        Args:
            platform: Platform to filter by.

        Returns:
            Dictionary of applicable parameters.

        """
        return {name: param for name, param in self.parameters.items() if param.is_applicable_for(platform)}

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration as key-value pairs.

        Returns:
            Dictionary of parameter names to default values.

        """
        return {name: param.default for name, param in self.parameters.items() if param.default is not None}

    def __len__(self) -> int:
        """Return number of parameters in the set."""
        return len(self.parameters)

    def __contains__(self, name: str) -> bool:
        """Check if parameter exists in the set."""
        return name in self.parameters

    def __getitem__(self, name: str) -> ConfigParameter:
        """Get parameter by name."""
        return self.parameters[name]
