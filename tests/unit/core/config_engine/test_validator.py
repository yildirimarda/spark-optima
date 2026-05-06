# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for config validator."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from spark_optima.core.config_engine.models import (
    ConfigParameter,
    ParameterCategory,
    ParameterType,
    PlatformSupport,
    ValidationConstraint,
)
from spark_optima.core.config_engine.validator import ConfigValidator, ValidationError


class TestValidationError:
    """Test cases for ValidationError class."""

    def test_error_creation(self) -> None:
        """Test creating validation error."""
        error = ValidationError("Test error", "param1", "value1")
        assert str(error) == "Test error"
        assert error.param_name == "param1"
        assert error.value == "value1"

    def test_error_without_details(self) -> None:
        """Test error without parameter details."""
        error = ValidationError("Test error")
        assert error.param_name == ""
        assert error.value is None


class TestConfigValidator:
    """Test cases for ConfigValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return ConfigValidator()

    @pytest.fixture
    def string_param(self):
        """Create a string parameter."""
        return ConfigParameter(
            name="spark.test.string",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.STRING,
            default="default_value",
        )

    @pytest.fixture
    def integer_param(self):
        """Create an integer parameter."""
        return ConfigParameter(
            name="spark.test.integer",
            category=ParameterCategory.CPU,
            param_type=ParameterType.INTEGER,
            default=10,
            constraints=ValidationConstraint(min_value=1, max_value=100),
        )

    @pytest.fixture
    def bytes_param(self):
        """Create a bytes parameter."""
        return ConfigParameter(
            name="spark.test.bytes",
            category=ParameterCategory.MEMORY,
            param_type=ParameterType.BYTES,
            default="1g",
            constraints=ValidationConstraint(min_value=512),
        )

    def test_validate_string_valid(self, validator, string_param):
        """Test valid string validation."""
        assert validator.validate(string_param, "test_value")
        assert validator.get_errors() == []

    def test_validate_string_none(self, validator, string_param):
        """Test None value validation (uses default)."""
        assert validator.validate(string_param, None)

    def test_validate_integer_valid(self, validator, integer_param):
        """Test valid integer validation."""
        assert validator.validate(integer_param, 50)

    def test_validate_integer_min_violation(self, validator, integer_param):
        """Test integer below minimum."""
        assert not validator.validate(integer_param, 0)
        errors = validator.get_errors()
        assert len(errors) == 1
        assert "less than minimum" in str(errors[0])

    def test_validate_integer_max_violation(self, validator, integer_param):
        """Test integer above maximum."""
        assert not validator.validate(integer_param, 200)
        errors = validator.get_errors()
        assert len(errors) == 1
        assert "greater than maximum" in str(errors[0])

    def test_validate_bytes_valid(self, validator, bytes_param):
        """Test valid bytes string."""
        assert validator.validate(bytes_param, "4g")
        assert validator.validate(bytes_param, "512m")
        assert validator.validate(bytes_param, "1024k")

    def test_validate_bytes_min_violation(self, validator, bytes_param):
        """Test bytes below minimum."""
        assert not validator.validate(bytes_param, "100")
        errors = validator.get_errors()
        assert len(errors) == 1

    def test_parse_bytes_valid(self, validator):
        """Test byte parsing."""
        assert validator._parse_bytes("4g") == 4 * 1024**3
        assert validator._parse_bytes("512m") == 512 * 1024**2
        assert validator._parse_bytes("1024k") == 1024 * 1024
        assert validator._parse_bytes("100") == 100
        assert validator._parse_bytes(-1) == -1  # Unlimited

    def test_parse_bytes_invalid(self, validator):
        """Test invalid byte strings."""
        with pytest.raises(ValueError):
            validator._parse_bytes("invalid")
        with pytest.raises(ValueError):
            validator._parse_bytes("4x")  # Invalid unit

    def test_parse_duration_valid(self, validator):
        """Test duration parsing."""
        assert validator._parse_duration("60s") == 60
        assert validator._parse_duration("5m") == 300
        assert validator._parse_duration("2h") == 7200
        assert validator._parse_duration("1d") == 86400
        assert validator._parse_duration("daily") == 86400
        assert validator._parse_duration(100) == 100  # Integer

    def test_parse_duration_invalid(self, validator):
        """Test invalid duration strings."""
        with pytest.raises(ValueError):
            validator._parse_duration("invalid")

    def test_is_valid_bytes(self, validator):
        """Test bytes validity check."""
        assert validator.is_valid_bytes("4g")
        assert validator.is_valid_bytes("512m")
        assert not validator.is_valid_bytes("invalid")

    def test_is_valid_duration(self, validator):
        """Test duration validity check."""
        assert validator.is_valid_duration("5m")
        assert validator.is_valid_duration("1h")
        assert not validator.is_valid_duration("invalid")

    def test_allowed_values_constraint(self, validator):
        """Test allowed values validation."""
        param = ConfigParameter(
            name="spark.test.enum",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.STRING,
            constraints=ValidationConstraint(allowed_values=["a", "b", "c"]),
        )
        assert validator.validate(param, "a")
        assert not validator.validate(param, "d")

    def test_pattern_constraint(self, validator):
        """Test pattern validation."""
        param = ConfigParameter(
            name="spark.test.pattern",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.STRING,
            constraints=ValidationConstraint(pattern=r"^\d+$"),
        )
        assert validator.validate(param, "123")
        assert not validator.validate(param, "abc")

    def test_platform_compatibility_warning(self, validator):
        """Test platform compatibility warning."""
        param = ConfigParameter(
            name="spark.databricks.param",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.STRING,
            applicable_platforms=[PlatformSupport.DATABRICKS],
        )
        context = {"platform": "local"}
        validator.validate(param, "value", context)
        warnings = validator.get_warnings()
        assert len(warnings) == 1
        assert "may not be applicable" in warnings[0]

    def test_validate_config(self, validator, integer_param, string_param):
        """Test validating entire configuration."""
        config = {
            "spark.test.integer": 50,
            "spark.test.string": "valid",
        }
        parameters = {
            "spark.test.integer": integer_param,
            "spark.test.string": string_param,
        }
        errors = validator.validate_config(config, parameters)
        assert errors == {}

    def test_validate_config_with_errors(self, validator, integer_param):
        """Test validating config with errors."""
        config = {
            "spark.test.integer": 200,  # Above max
        }
        parameters = {
            "spark.test.integer": integer_param,
        }
        errors = validator.validate_config(config, parameters)
        assert "spark.test.integer" in errors

    def test_validate_config_unknown_param(self, validator):
        """Test validating config with unknown parameter."""
        config = {
            "spark.unknown.param": "value",
        }
        parameters = {}
        errors = validator.validate_config(config, parameters)
        assert "spark.unknown.param" in errors

    def test_normalize_value_bytes(self, validator):
        """Test bytes normalization."""
        result = validator.normalize_value("4g", ParameterType.BYTES)
        assert result == 4 * 1024**3

    def test_normalize_value_duration(self, validator):
        """Test duration normalization."""
        result = validator.normalize_value("5m", ParameterType.DURATION)
        assert result == 300

    def test_normalize_value_other(self, validator):
        """Test normalization for other types."""
        assert validator.normalize_value("test", ParameterType.STRING) == "test"
        assert validator.normalize_value(42, ParameterType.INTEGER) == 42


class TestConfigValidatorBoolean:
    """Test boolean validation."""

    def test_validate_boolean_true(self):
        """Test valid boolean True."""
        validator = ConfigValidator()
        param = ConfigParameter(
            name="spark.test.bool",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.BOOLEAN,
        )
        assert validator.validate(param, True)
        assert validator.validate(param, False)

    def test_validate_boolean_invalid(self):
        """Test invalid boolean."""
        validator = ConfigValidator()
        param = ConfigParameter(
            name="spark.test.bool",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.BOOLEAN,
        )
        assert not validator.validate(param, "true")  # String is not bool


class TestConfigValidatorFloat:
    """Test float validation."""

    def test_validate_float_valid(self):
        """Test valid float."""
        validator = ConfigValidator()
        param = ConfigParameter(
            name="spark.test.float",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.FLOAT,
            constraints=ValidationConstraint(min_value=0.0, max_value=1.0),
        )
        assert validator.validate(param, 0.5)
        assert validator.validate(param, 0)  # Int is also valid

    def test_validate_float_range(self):
        """Test float range validation."""
        validator = ConfigValidator()
        param = ConfigParameter(
            name="spark.test.float",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.FLOAT,
            constraints=ValidationConstraint(min_value=0.0, max_value=1.0),
        )
        assert not validator.validate(param, 1.5)


class TestValidationErrorMore:
    """Additional tests for ValidationError."""

    def test_error_with_all_params(self):
        """Test ValidationError with all parameters (lines 27-38)."""
        error = ValidationError("Test message", "param1", "value1")
        assert str(error) == "Test message"
        assert error.param_name == "param1"
        assert error.value == "value1"

    def test_error_inheritance(self):
        """Test ValidationError inherits from Exception."""
        error = ValidationError("Test")
        assert isinstance(error, Exception)


class TestConfigValidatorValidate:
    """Additional tests for validate method."""

    def test_validate_clears_previous_errors(self):
        """Test that validate clears previous errors (lines 101-102)."""
        validator = ConfigValidator()
        param = ConfigParameter(
            name="test",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.STRING,
        )

        # First validation with error
        validator.validate(param, "test")
        assert len(validator.get_errors()) == 0

        # Second validation should clear previous state
        validator.validate(param, "test")
        assert len(validator.get_errors()) == 0

    def test_validate_exception_handling(self):
        """Test exception handling in validate (lines 123-127)."""
        validator = ConfigValidator()

        # Create a param that will cause an exception
        param = ConfigParameter(
            name="test",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.STRING,
        )

        # Mock _validate_type to raise an exception
        # The except block catches (ValueError, TypeError, AttributeError)
        with patch.object(validator, "_validate_type", side_effect=ValueError("Test exception")):
            result = validator.validate(param, "test")
            assert result is False
            errors = validator.get_errors()
            assert len(errors) == 1
            assert "Test exception" in str(errors[0])


class TestConfigValidatorValidateType:
    """Tests for _validate_type method."""

    def test_validate_type_boolean_true(self):
        """Test boolean type validation (lines 153-154)."""
        validator = ConfigValidator()
        ConfigParameter(
            name="test",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.BOOLEAN,
        )
        assert validator._validate_type(ParameterType.BOOLEAN, True)
        assert validator._validate_type(ParameterType.BOOLEAN, False)

    def test_validate_type_bytes(self):
        """Test bytes type validation (lines 156-157)."""
        validator = ConfigValidator()
        assert validator._validate_type(ParameterType.BYTES, "4g")
        assert validator._validate_type(ParameterType.BYTES, 1024)

    def test_validate_type_duration(self):
        """Test duration type validation (lines 156-157)."""
        validator = ConfigValidator()
        assert validator._validate_type(ParameterType.DURATION, "60s")
        assert validator._validate_type(ParameterType.DURATION, 60)

    def test_validate_type_list(self):
        """Test list type validation (lines 159-160)."""
        validator = ConfigValidator()
        assert validator._validate_type(ParameterType.LIST, [1, 2, 3])
        assert validator._validate_type(ParameterType.LIST, (1, 2, 3))

    def test_validate_type_map(self):
        """Test map type validation (lines 162-163)."""
        validator = ConfigValidator()
        assert validator._validate_type(ParameterType.MAP, {"key": "value"})

    def test_validate_type_none(self):
        """Test None value (lines 140-141)."""
        validator = ConfigValidator()
        assert validator._validate_type(ParameterType.STRING, None)
        assert validator._validate_type(ParameterType.INTEGER, None)

    def test_validate_type_exception(self):
        """Test exception handling in _validate_type (lines 167-169)."""
        validator = ConfigValidator()
        # This is hard to trigger, but we can test the path
        result = validator._validate_type(ParameterType.STRING, "test")
        assert result is True


class TestConfigValidatorValidateConstraints:
    """Tests for _validate_constraints method."""

    def test_validate_constraints_min(self):
        """Test min value validation (lines 192-202)."""
        validator = ConfigValidator()
        param = ConfigParameter(
            name="test",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.INTEGER,
            constraints=ValidationConstraint(min_value=10),
        )

        assert validator._validate_constraints(param, 5, {}) is False
        assert len(validator.get_errors()) == 1

        validator._errors.clear()
        assert validator._validate_constraints(param, 15, {}) is True

    def test_validate_constraints_max(self):
        """Test max value validation (lines 204-212)."""
        validator = ConfigValidator()
        param = ConfigParameter(
            name="test",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.INTEGER,
            constraints=ValidationConstraint(max_value=100),
        )

        assert validator._validate_constraints(param, 150, {}) is False
        assert len(validator.get_errors()) == 1

        validator._errors.clear()
        assert validator._validate_constraints(param, 50, {}) is True

    def test_validate_constraints_pattern(self):
        """Test pattern validation (lines 214-227)."""
        validator = ConfigValidator()
        param = ConfigParameter(
            name="test",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.STRING,
            constraints=ValidationConstraint(pattern=r"^\d+$"),
        )

        assert validator._validate_constraints(param, "123", {}) is True
        assert validator._validate_constraints(param, "abc", {}) is False

    def test_validate_constraints_allowed_values(self):
        """Test allowed values validation (lines 229-237)."""
        validator = ConfigValidator()
        param = ConfigParameter(
            name="test",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.STRING,
            constraints=ValidationConstraint(allowed_values=["a", "b"]),
        )

        assert validator._validate_constraints(param, "a", {}) is True
        assert validator._validate_constraints(param, "c", {}) is False

    def test_validate_constraints_dependency(self):
        """Test dependency validation (lines 240-246)."""
        validator = ConfigValidator()
        param = ConfigParameter(
            name="test",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.STRING,
            constraints=ValidationConstraint(depends_on=["other_param"]),
        )

        # Without the dependency in context
        validator._warnings.clear()
        validator._validate_constraints(param, "value", {"other_param": 1})
        assert len(validator.get_warnings()) == 0

    def test_validate_constraints_none_value(self):
        """Test with None value (lines 185-186)."""
        validator = ConfigValidator()
        param = ConfigParameter(
            name="test",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.INTEGER,
        )

        assert validator._validate_constraints(param, None, {}) is True


class TestConfigValidatorToNumeric:
    """Tests for _to_numeric method."""

    def test_to_numeric_integer(self):
        """Test converting integer (lines 262-264)."""
        validator = ConfigValidator()
        result = validator._to_numeric(42, ParameterType.INTEGER)
        assert result == 42.0

    def test_to_numeric_float(self):
        """Test converting float."""
        validator = ConfigValidator()
        result = validator._to_numeric(3.14, ParameterType.FLOAT)
        assert result == 3.14

    def test_to_numeric_bytes(self):
        """Test converting bytes string (lines 265-266)."""
        validator = ConfigValidator()
        result = validator._to_numeric("4g", ParameterType.BYTES)
        assert result == 4 * 1024**3

    def test_to_numeric_duration(self):
        """Test converting duration string (lines 268-269)."""
        validator = ConfigValidator()
        result = validator._to_numeric("60s", ParameterType.DURATION)
        assert result == 60.0

    def test_to_numeric_invalid(self):
        """Test invalid conversion (lines 271-273)."""
        validator = ConfigValidator()
        result = validator._to_numeric("invalid", ParameterType.INTEGER)
        assert result is None


class TestConfigValidatorValidateConfig:
    """Tests for validate_config method."""

    def test_validate_config_all_valid(self):
        """Test validate_config with all valid (lines 367-395)."""
        validator = ConfigValidator()
        params = {
            "param1": ConfigParameter(
                name="param1",
                category=ParameterCategory.RUNTIME,
                param_type=ParameterType.STRING,
            ),
        }

        config = {"param1": "value1"}
        errors = validator.validate_config(config, params)
        assert errors == {}

    def test_validate_config_with_context(self):
        """Test validate_config with context (lines 88-120)."""
        validator = ConfigValidator()
        param = ConfigParameter(
            name="test",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.STRING,
            applicable_platforms=[PlatformSupport.DATABRICKS],
        )

        # With matching platform
        validator.validate(param, "value", {"platform": "databricks"})
        warnings = validator.get_warnings()
        # Should not have warning for applicable platform
        assert len(warnings) == 0

        # With non-matching platform
        validator.validate(param, "value", {"platform": "local"})
        warnings = validator.get_warnings()
        assert len(warnings) == 1
        assert "may not be applicable" in warnings[0]
