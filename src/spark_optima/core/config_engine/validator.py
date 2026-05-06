# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Configuration validator for Spark parameters.

This module provides the ConfigValidator class for validating Spark
configuration values against their defined constraints and rules.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from spark_optima.core.config_engine.models import (
    ConfigParameter,
    ParameterType,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for configuration validation errors."""

    def __init__(self, message: str, param_name: str = "", value: Any = None) -> None:
        """Initialize validation error.

        Args:
            message: Error message.
            param_name: Name of the parameter that failed validation.
            value: The invalid value.

        """
        self.param_name = param_name
        self.value = value
        super().__init__(message)


class ConfigValidator:
    """Validator for Spark configuration parameters.

    This class validates configuration values against their defined
    constraints including type checking, range validation, pattern
    matching, and dependency checking.

    Example:
        >>> validator = ConfigValidator()
        >>> param = db.get_parameter("3.5.0", "spark.executor.memory")
        >>> is_valid = validator.validate(param, "4g")
        >>> errors = validator.get_errors()

    """

    # Byte size multipliers
    BYTE_UNITS = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
    }

    # Duration multipliers (in seconds)
    DURATION_UNITS = {
        "ms": 0.001,
        "s": 1,
        "sec": 1,
        "m": 60,
        "min": 60,
        "h": 3600,
        "hr": 3600,
        "d": 86400,
        "day": 86400,
    }

    def __init__(self) -> None:
        """Initialize the validator."""
        self._errors: list[ValidationError] = []
        self._warnings: list[str] = []

    def validate(
        self,
        param: ConfigParameter,
        value: Any,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Validate a value against a parameter definition.

        Args:
            param: Configuration parameter definition.
            value: Value to validate.
            context: Optional context for dependency checking.

        Returns:
            True if validation passes, False otherwise.

        """
        self._errors.clear()
        self._warnings.clear()

        try:
            # Type validation
            if not self._validate_type(param.param_type, value):
                return False

            # Constraint validation
            if not self._validate_constraints(param, value, context):
                return False

            # Platform compatibility
            if context and "platform" in context:
                platform = context["platform"]
                if not param.is_applicable_for(platform):
                    self._warnings.append(
                        f"Parameter {param.name} may not be applicable for platform {platform}",
                    )

            return True

        except (ValueError, TypeError, AttributeError) as e:
            self._errors.append(ValidationError(f"Validation error: {e}", param.name, value))
            return False

    def _validate_type(self, param_type: ParameterType, value: Any) -> bool:
        """Validate value type.

        Args:
            param_type: Expected parameter type.
            value: Value to check.

        Returns:
            True if type is valid.

        """
        if value is None:
            return True  # None is valid (use default)

        try:
            if param_type == ParameterType.STRING:
                return isinstance(value, str)

            elif param_type == ParameterType.INTEGER:
                return isinstance(value, int) and not isinstance(value, bool)

            elif param_type == ParameterType.FLOAT:
                return isinstance(value, int | float) and not isinstance(value, bool)

            elif param_type == ParameterType.BOOLEAN:
                return isinstance(value, bool)

            elif param_type == ParameterType.BYTES or param_type == ParameterType.DURATION:
                return isinstance(value, str | int)

            elif param_type == ParameterType.LIST:
                return isinstance(value, list | tuple)

            elif param_type == ParameterType.MAP:
                return isinstance(value, dict)

            return True

        except (ValueError, TypeError, AttributeError) as e:
            self._errors.append(ValidationError(f"Type validation error: {e}"))
            return False

    def _validate_constraints(
        self,
        param: ConfigParameter,
        value: Any,
        context: dict[str, Any] | None,
    ) -> bool:
        """Validate value against constraints.

        Args:
            param: Configuration parameter.
            value: Value to validate.
            context: Validation context.

        Returns:
            True if constraints are satisfied.

        """
        if value is None:
            return True

        constraints = param.constraints
        valid = True

        # Min/Max validation
        numeric_value = self._to_numeric(value, param.param_type)
        if numeric_value is not None:
            if constraints.min_value is not None and numeric_value < constraints.min_value:
                self._errors.append(
                    ValidationError(
                        f"Value {value} is less than minimum {constraints.min_value}",
                        param.name,
                        value,
                    ),
                )
                valid = False

            if constraints.max_value is not None and numeric_value > constraints.max_value:
                self._errors.append(
                    ValidationError(
                        f"Value {value} is greater than maximum {constraints.max_value}",
                        param.name,
                        value,
                    ),
                )
                valid = False

        # Pattern validation
        if (
            constraints.pattern
            and isinstance(value, str)
            and not re.match(constraints.pattern, value)
        ):
            self._errors.append(
                ValidationError(
                    f"Value '{value}' does not match pattern '{constraints.pattern}'",
                    param.name,
                    value,
                ),
            )
            valid = False

        # Allowed values validation
        if constraints.allowed_values is not None and value not in constraints.allowed_values:
            self._errors.append(
                ValidationError(
                    f"Value '{value}' is not in allowed values: {constraints.allowed_values}",
                    param.name,
                    value,
                ),
            )
            valid = False

        # Dependency validation
        if constraints.depends_on and context:
            for dep in constraints.depends_on:
                if dep not in context:
                    self._warnings.append(
                        f"Parameter {param.name} depends on {dep} which is not in context",
                    )

        return valid

    def _to_numeric(self, value: Any, param_type: ParameterType) -> float | None:
        """Convert value to numeric for comparison.

        Args:
            value: Value to convert.
            param_type: Parameter type.

        Returns:
            Numeric value or None if not convertible.

        """
        try:
            if param_type == ParameterType.INTEGER or param_type == ParameterType.FLOAT:
                return float(value)

            elif param_type == ParameterType.BYTES:
                return float(self._parse_bytes(value))

            elif param_type == ParameterType.DURATION:
                return float(self._parse_duration(value))

        except (ValueError, TypeError):
            pass

        return None

    def _parse_bytes(self, value: str | int) -> int:
        """Parse byte string to integer.

        Args:
            value: Byte string like "4g", "512m", or integer.

        Returns:
            Number of bytes.

        Raises:
            ValueError: If string cannot be parsed.

        """
        if isinstance(value, int):
            return value

        if isinstance(value, str):
            value = value.strip().lower()

            # Handle -1 for unlimited
            if value == "-1" or value == "infinity":
                return -1

            # Extract number and unit
            match = re.match(r"^(-?\d+(?:\.\d+)?)\s*([a-z]*)$", value)
            if not match:
                raise ValueError(f"Invalid byte string: {value}")

            num_str = match.group(1)
            unit = match.group(2)

            # Validate that we actually have a number
            try:
                num = float(num_str)
            except ValueError as e:
                raise ValueError(f"Invalid byte string: {value}") from e

            # Empty unit is valid (plain number)
            if unit == "":
                return int(num)

            multiplier = self.BYTE_UNITS.get(unit)
            if multiplier is None:
                raise ValueError(f"Invalid byte unit: {unit}")
            return int(num * multiplier)

        raise ValueError(f"Cannot parse bytes from: {value}")

    def _parse_duration(self, value: str | int) -> int:
        """Parse duration string to seconds.

        Args:
            value: Duration string like "5m", "1h", or integer seconds.

        Returns:
            Duration in seconds.

        Raises:
            ValueError: If string cannot be parsed.

        """
        if isinstance(value, int):
            return value

        if isinstance(value, str):
            value = value.strip().lower()

            # Handle "infinity"
            if value == "infinity":
                return -1

            # Check for special keywords
            if value == "daily":
                return 86400

            # Extract number and unit
            match = re.match(r"^(\d+(?:\.\d+)?)\s*([a-z]+)$", value)
            if match:
                num = float(match.group(1))
                unit = match.group(2)
                multiplier = self.DURATION_UNITS.get(unit, 1)
                return int(num * multiplier)

            # Try parsing as just a number (seconds)
            try:
                return int(value)
            except ValueError:
                pass

        raise ValueError(f"Cannot parse duration from: {value}")

    def validate_config(
        self,
        config: dict[str, Any],
        parameters: dict[str, ConfigParameter],
        context: dict[str, Any] | None = None,
    ) -> dict[str, list[ValidationError]]:
        """Validate an entire configuration.

        Args:
            config: Configuration dictionary to validate.
            parameters: Dictionary of parameter definitions.
            context: Optional validation context.

        Returns:
            Dictionary of parameter names to their validation errors.

        """
        all_errors: dict[str, list[ValidationError]] = {}

        for name, value in config.items():
            param = parameters.get(name)
            if not param:
                all_errors[name] = [ValidationError(f"Unknown parameter: {name}", name, value)]
                continue

            if not self.validate(param, value, context):
                all_errors[name] = self._errors.copy()

        return all_errors

    def get_errors(self) -> list[ValidationError]:
        """Get validation errors from last validation.

        Returns:
            List of ValidationError objects.

        """
        return self._errors.copy()

    def get_warnings(self) -> list[str]:
        """Get validation warnings from last validation.

        Returns:
            List of warning messages.

        """
        return self._warnings.copy()

    def is_valid_bytes(self, value: str) -> bool:
        """Check if a string is valid byte notation.

        Args:
            value: String to check.

        Returns:
            True if valid byte string.

        """
        try:
            self._parse_bytes(value)
            return True
        except ValueError:
            return False

    def is_valid_duration(self, value: str) -> bool:
        """Check if a string is valid duration notation.

        Args:
            value: String to check.

        Returns:
            True if valid duration string.

        """
        try:
            self._parse_duration(value)
            return True
        except ValueError:
            return False

    def normalize_value(self, value: Any, param_type: ParameterType) -> Any:
        """Normalize a value to its standard form.

        Args:
            value: Value to normalize.
            param_type: Parameter type.

        Returns:
            Normalized value.

        """
        if param_type == ParameterType.BYTES and isinstance(value, str):
            return self._parse_bytes(value)

        elif param_type == ParameterType.DURATION and isinstance(value, str):
            return self._parse_duration(value)

        return value
