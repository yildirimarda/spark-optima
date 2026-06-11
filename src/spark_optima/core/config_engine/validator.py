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

from spark_optima.core.config_engine import units
from spark_optima.core.config_engine.models import (
    ConfigParameter,
    ParameterType,
)
from spark_optima.core.config_engine.units import (
    has_byte_unit_suffix,
    parse_bytes,
    parse_duration,
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

    Unit convention: constraint bounds (``min_value`` / ``max_value``) are
    numbers in canonical base units — bytes for BYTES parameters, seconds
    for DURATION parameters (unit-suffixed strings in the YAML database are
    canonicalized to base units at load time). Range comparisons are always
    performed in base units and violation messages name both sides with
    their units.

    Range-check rule for unit-less BYTES candidates: a BYTES candidate
    without an explicit unit suffix (a bare int, or a string like "4096")
    is NOT range-compared, because Spark's default unit for bare numbers
    varies per parameter — ``bytesConf(ByteUnit.MiB)`` parameters such as
    driver/executor memory, memoryOverhead, pyspark.memory and
    kryoserializer.buffer.max read bare numbers as MiB, while others read
    them as bytes — so a unit-less candidate cannot be ranged safely.
    Format/pattern/allowed-value checks still apply to such candidates.
    Unit-suffixed candidates ("512m", "3g") are fully range-checked.

    Example:
        >>> validator = ConfigValidator()
        >>> param = db.get_parameter("3.5.0", "spark.executor.memory")
        >>> is_valid = validator.validate(param, "4g")
        >>> errors = validator.get_errors()

    """

    # Byte size multipliers (shared with load-time canonicalization)
    BYTE_UNITS = units.BYTE_UNITS

    # Duration multipliers in seconds (shared with load-time canonicalization)
    DURATION_UNITS = units.DURATION_UNITS

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

        # Min/Max validation. Unit-less BYTES candidates are excluded:
        # Spark's default unit for bare numbers varies per parameter
        # (MiB for memory params, bytes for others), so they cannot be
        # range-compared safely. See the class docstring.
        numeric_value = self._to_numeric(value, param.param_type)
        if numeric_value is not None and self._is_range_comparable(param.param_type, value):
            min_bound = self._bound_to_numeric(constraints.min_value, param.param_type, param.name, "min_value")
            max_bound = self._bound_to_numeric(constraints.max_value, param.param_type, param.name, "max_value")

            if min_bound is not None and numeric_value < min_bound:
                value_desc = self._describe_quantity(value, numeric_value, param.param_type)
                bound_desc = self._describe_quantity(constraints.min_value, min_bound, param.param_type)
                self._errors.append(
                    ValidationError(
                        f"Value {value_desc} is less than minimum {bound_desc}",
                        param.name,
                        value,
                    ),
                )
                valid = False

            if max_bound is not None and numeric_value > max_bound:
                value_desc = self._describe_quantity(value, numeric_value, param.param_type)
                bound_desc = self._describe_quantity(constraints.max_value, max_bound, param.param_type)
                self._errors.append(
                    ValidationError(
                        f"Value {value_desc} is greater than maximum {bound_desc}",
                        param.name,
                        value,
                    ),
                )
                valid = False

        # Pattern validation
        if constraints.pattern and isinstance(value, str) and not re.match(constraints.pattern, value):
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

    @staticmethod
    def _is_range_comparable(param_type: ParameterType, value: Any) -> bool:
        """Check whether a candidate value can be safely range-compared.

        BYTES candidates without an explicit unit suffix (bare ints, or
        strings like "4096") are not comparable: Spark's default unit for
        bare numbers varies per parameter (MiB for memory params, bytes
        for others). All other candidates are comparable.

        Args:
            param_type: Parameter type being validated.
            value: Candidate value as supplied.

        Returns:
            True if min/max range comparison applies to this candidate.

        """
        if param_type != ParameterType.BYTES:
            return True
        return has_byte_unit_suffix(value)

    def _bound_to_numeric(
        self,
        bound: int | float | str | None,
        param_type: ParameterType,
        param_name: str,
        bound_name: str,
    ) -> float | None:
        """Convert a constraint bound to a numeric base-unit value.

        Database-loaded bounds are already numeric (canonicalized to base
        units at load time). String bounds on directly-constructed
        constraints are still parsed defensively with the same parser used
        for candidate values; bare numbers mean canonical base units
        (bytes for BYTES, seconds for DURATION).

        Args:
            bound: Bound value from the parameter constraints.
            param_type: Parameter type the bound applies to.
            param_name: Parameter name (for error reporting).
            bound_name: Which bound is being converted ("min_value"/"max_value").

        Returns:
            Numeric bound in base units, or None if no bound is set.

        Raises:
            ValueError: If a string bound cannot be parsed for the type.

        """
        if bound is None:
            return None

        if isinstance(bound, str):
            try:
                if param_type == ParameterType.BYTES:
                    return float(self._parse_bytes(bound))
                if param_type == ParameterType.DURATION:
                    return float(self._parse_duration(bound))
                return float(bound)
            except ValueError as e:
                raise ValueError(
                    f"Invalid {bound_name} '{bound}' for parameter {param_name}: "
                    f"expected a number or a unit-suffixed string",
                ) from e

        return float(bound)

    def _describe_quantity(self, raw: Any, numeric: float, param_type: ParameterType) -> str:
        """Format a value or bound for range-violation messages.

        For BYTES/DURATION the description names the unit on both the raw
        form and its canonical base-unit equivalent, e.g. "'2048m'
        (2147483648 bytes)". Other types render the raw value unchanged.

        Args:
            raw: Original value or bound as defined/supplied.
            numeric: Parsed base-unit numeric equivalent.
            param_type: Parameter type being validated.

        Returns:
            Human-readable description string.

        """
        if param_type == ParameterType.BYTES:
            unit = "bytes"
        elif param_type == ParameterType.DURATION:
            unit = "seconds"
        else:
            return str(raw)

        canonical = int(numeric) if float(numeric).is_integer() else numeric
        if isinstance(raw, str):
            return f"'{raw}' ({canonical} {unit})"
        return f"{raw} {unit}"

    def _parse_bytes(self, value: str | int) -> int:
        """Parse byte string to integer.

        Delegates to the shared parser in
        :mod:`spark_optima.core.config_engine.units`, which is also used
        for load-time canonicalization of constraint bounds.

        Args:
            value: Byte string like "4g", "512m", or integer.

        Returns:
            Number of bytes.

        Raises:
            ValueError: If string cannot be parsed.

        """
        return parse_bytes(value)

    def _parse_duration(self, value: str | int) -> int:
        """Parse duration string to seconds.

        Delegates to the shared parser in
        :mod:`spark_optima.core.config_engine.units`, which is also used
        for load-time canonicalization of constraint bounds.

        Args:
            value: Duration string like "5m", "1h", or integer seconds.

        Returns:
            Duration in seconds.

        Raises:
            ValueError: If string cannot be parsed.

        """
        return parse_duration(value)

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
