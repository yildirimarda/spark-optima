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
        """Test bytes below minimum (unit-suffixed candidate)."""
        assert not validator.validate(bytes_param, "100b")
        errors = validator.get_errors()
        assert len(errors) == 1

    def test_validate_bytes_suffixless_candidate_skips_range(self, validator, bytes_param):
        """Bare-number BYTES candidates are not range-compared.

        Spark's default unit for bare numbers varies per parameter (MiB for
        memory params, bytes for others), so "100" cannot be ranged safely.
        """
        assert validator.validate(bytes_param, "100")
        assert validator.get_errors() == []

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


class TestUnitSuffixedBounds:
    """Tests for unit-suffixed min/max bounds on BYTES/DURATION parameters.

    Regression tests for the v1.4 mixed-unit database issue where bounds like
    ``max_value: 2048`` (meaning MiB) were compared against candidate values
    parsed to bytes, making even the parameter's own default fail validation.
    """

    @pytest.fixture
    def validator(self) -> ConfigValidator:
        """Create a validator instance."""
        return ConfigValidator()

    @pytest.fixture
    def kryo_like_param(self) -> ConfigParameter:
        """Create a BYTES parameter with unit-suffixed bounds."""
        return ConfigParameter(
            name="spark.test.kryoBufferMax",
            category=ParameterCategory.SERIALIZATION,
            param_type=ParameterType.BYTES,
            default="64m",
            constraints=ValidationConstraint(min_value="64k", max_value="2048m"),
        )

    @pytest.fixture
    def duration_param(self) -> ConfigParameter:
        """Create a DURATION parameter with unit-suffixed bounds."""
        return ConfigParameter(
            name="spark.test.timeout",
            category=ParameterCategory.NETWORK,
            param_type=ParameterType.DURATION,
            default="120s",
            constraints=ValidationConstraint(min_value="1s", max_value="10min"),
        )

    def test_bytes_within_suffixed_bounds_passes(self, validator, kryo_like_param) -> None:
        """A value inside unit-suffixed bounds must validate."""
        assert validator.validate(kryo_like_param, "512m")
        assert validator.get_errors() == []

    def test_bytes_default_within_own_bounds(self, validator, kryo_like_param) -> None:
        """Previously broken: default '64m' failed against max_value 2048 (MiB-as-bytes)."""
        assert validator.validate(kryo_like_param, "64m")

    def test_bytes_bounds_are_inclusive(self, validator, kryo_like_param) -> None:
        """Values equal to the bounds must validate."""
        assert validator.validate(kryo_like_param, "64k")
        assert validator.validate(kryo_like_param, "2048m")

    def test_bytes_above_max_fails_with_units(self, validator, kryo_like_param) -> None:
        """A too-large value must fail naming both sides with units."""
        assert not validator.validate(kryo_like_param, "3g")
        errors = validator.get_errors()
        assert len(errors) == 1
        message = str(errors[0])
        assert "is greater than maximum" in message
        assert "'3g'" in message
        assert "'2048m'" in message
        assert "bytes" in message

    def test_bytes_below_min_fails_with_units(self, validator, kryo_like_param) -> None:
        """A too-small value must fail naming both sides with units."""
        assert not validator.validate(kryo_like_param, "16k")
        errors = validator.get_errors()
        assert len(errors) == 1
        message = str(errors[0])
        assert "is less than minimum" in message
        assert "'16k'" in message
        assert "'64k'" in message
        assert "bytes" in message

    def test_unitless_bytes_candidates_skip_range_check(self, validator, kryo_like_param) -> None:
        """Unit-less BYTES candidates are never range-compared.

        Spark's default unit for bare numbers varies per parameter
        (bytesConf(ByteUnit.MiB) params read "4096" as MiB, others as
        bytes), so neither bare ints nor suffixless strings can be ranged
        safely — even when their bytes interpretation would be out of range.
        """
        assert validator.validate(kryo_like_param, 64 * 1024 * 1024)
        assert validator.validate(kryo_like_param, 3 * 1024**3)
        assert validator.validate(kryo_like_param, "4096")
        assert validator.get_errors() == []

    def test_bare_numeric_bounds_mean_bytes(self, validator) -> None:
        """Bare numeric bounds keep meaning canonical base units (bytes)."""
        param = ConfigParameter(
            name="spark.test.buffer",
            category=ParameterCategory.SHUFFLE,
            param_type=ParameterType.BYTES,
            default="32k",
            constraints=ValidationConstraint(min_value=1024),
        )
        assert validator.validate(param, "2k")
        assert not validator.validate(param, "512b")
        message = str(validator.get_errors()[0])
        assert "1024 bytes" in message

    def test_duration_within_suffixed_bounds_passes(self, validator, duration_param) -> None:
        """A duration inside unit-suffixed bounds must validate."""
        assert validator.validate(duration_param, "30s")
        assert validator.validate(duration_param, "120s")

    def test_duration_above_max_fails_with_units(self, validator, duration_param) -> None:
        """A too-long duration must fail naming both sides with units."""
        assert not validator.validate(duration_param, "1h")
        errors = validator.get_errors()
        assert len(errors) == 1
        message = str(errors[0])
        assert "is greater than maximum" in message
        assert "'1h'" in message
        assert "'10min'" in message
        assert "seconds" in message

    def test_duration_bare_numeric_bound_means_seconds(self, validator) -> None:
        """Bare numeric duration bounds keep meaning seconds."""
        param = ConfigParameter(
            name="spark.test.interval",
            category=ParameterCategory.NETWORK,
            param_type=ParameterType.DURATION,
            default="60s",
            constraints=ValidationConstraint(min_value=10, max_value=600),
        )
        assert validator.validate(param, "1min")
        assert not validator.validate(param, "20min")
        message = str(validator.get_errors()[0])
        assert "600 seconds" in message

    def test_invalid_bound_string_is_validation_error(self, validator) -> None:
        """An unparsable bound must surface as a validation error, not pass silently."""
        param = ConfigParameter(
            name="spark.test.badBound",
            category=ParameterCategory.MEMORY,
            param_type=ParameterType.BYTES,
            default="1g",
            constraints=ValidationConstraint(max_value="12parsecs"),
        )
        assert not validator.validate(param, "1g")
        errors = validator.get_errors()
        assert len(errors) == 1
        assert "Invalid max_value '12parsecs'" in str(errors[0])

    def test_integer_messages_unchanged(self, validator) -> None:
        """Non-BYTES/DURATION range messages keep the plain numeric format."""
        param = ConfigParameter(
            name="spark.test.cores",
            category=ParameterCategory.CPU,
            param_type=ParameterType.INTEGER,
            constraints=ValidationConstraint(min_value=1, max_value=100),
        )
        assert not validator.validate(param, 200)
        assert str(validator.get_errors()[0]) == "Value 200 is greater than maximum 100"


class TestDatabaseUnitNormalizationSweep:
    """Mechanical sweep over the shipped parameter database.

    Asserts the canonical unit convention holds for every BYTES/DURATION
    parameter in every Spark version: bounds parse with the same parser used
    for candidate values, min <= max, and each parameter's own default
    validates within its own bounds. Catches any reintroduced mixed-unit
    entry without naming parameters individually.
    """

    @pytest.fixture(scope="class")
    def database(self):
        """Load the shipped configuration database."""
        from spark_optima.core.config_engine.database import ConfigDatabase

        return ConfigDatabase()

    @pytest.fixture
    def validator(self) -> ConfigValidator:
        """Create a validator instance."""
        return ConfigValidator()

    @staticmethod
    def _sized_params(database):
        """Yield (version, name, param) for every BYTES/DURATION parameter."""
        for version in database.get_available_versions():
            config_set = database.get_config_set(version)
            for name, param in config_set.parameters.items():
                if param.param_type in (ParameterType.BYTES, ParameterType.DURATION):
                    yield version, name, param

    def test_database_has_sized_params(self, database) -> None:
        """Sanity check: the sweep actually covers parameters."""
        assert sum(1 for _ in self._sized_params(database)) > 0

    def test_bounds_parse_with_value_parser(self, database, validator) -> None:
        """Every string bound must parse with the candidate-value parser."""
        failures = []
        for version, name, param in self._sized_params(database):
            for bound_name in ("min_value", "max_value"):
                bound = getattr(param.constraints, bound_name)
                try:
                    validator._bound_to_numeric(bound, param.param_type, name, bound_name)
                except ValueError as e:
                    failures.append(f"{version} {name}.{bound_name}: {e}")
        assert not failures, "Unparsable bounds:\n" + "\n".join(failures)

    def test_min_not_greater_than_max(self, database, validator) -> None:
        """Whenever both bounds exist, min must not exceed max in base units."""
        failures = []
        for version, name, param in self._sized_params(database):
            low = validator._bound_to_numeric(param.constraints.min_value, param.param_type, name, "min_value")
            high = validator._bound_to_numeric(param.constraints.max_value, param.param_type, name, "max_value")
            if low is not None and high is not None and low > high:
                failures.append(f"{version} {name}: min {low} > max {high}")
        assert not failures, "Inverted bounds:\n" + "\n".join(failures)

    def test_every_default_validates_within_own_bounds(self, database, validator) -> None:
        """Every BYTES/DURATION default must pass its own range check."""
        failures = []
        for version, name, param in self._sized_params(database):
            if param.default is None:
                continue
            if not validator.validate(param, param.default):
                details = "; ".join(str(e) for e in validator.get_errors())
                failures.append(f"{version} {name} default={param.default!r}: {details}")
        assert not failures, "Defaults failing their own bounds:\n" + "\n".join(failures)

    def test_all_loaded_bounds_are_numeric(self, database) -> None:
        """Load-time canonicalization leaves no string bound in the system."""
        failures = []
        for version in database.get_available_versions():
            config_set = database.get_config_set(version)
            for name, param in config_set.parameters.items():
                for bound_name in ("min_value", "max_value"):
                    bound = getattr(param.constraints, bound_name)
                    if bound is not None and not isinstance(bound, int | float):
                        failures.append(f"{version} {name}.{bound_name}={bound!r}")
        assert not failures, "Non-numeric bounds after load:\n" + "\n".join(failures)

    def test_kryoserializer_buffer_max_regression(self, database, validator) -> None:
        """Regression: the v1.4 mixed-unit entry now range-checks correctly.

        The YAML max bound "2048m" is canonicalized to 2147483648 bytes at
        load time; violations name both sides with units.
        """
        param = database.get_parameter("3.5.0", "spark.kryoserializer.buffer.max")
        assert param is not None
        assert param.constraints.max_value == 2048 * 1024**2
        assert validator.validate(param, "64m")
        assert validator.validate(param, "512m")
        assert not validator.validate(param, "3g")
        message = str(validator.get_errors()[0])
        assert "'3g'" in message
        assert "2147483648 bytes" in message

    def test_executor_memory_minimum_regression(self, database, validator) -> None:
        """Regression: executor memory minimum is 512 MiB, not 512 bytes."""
        param = database.get_parameter("3.5.0", "spark.executor.memory")
        assert param is not None
        assert param.constraints.min_value == 512 * 1024**2
        assert validator.validate(param, "4g")
        assert not validator.validate(param, "1m")
        message = str(validator.get_errors()[0])
        assert "'1m'" in message
        assert "536870912 bytes" in message

    def test_bare_number_candidate_on_driver_memory_passes(self, database, validator) -> None:
        """Regression: a bare "4096" on spark.driver.memory must pass.

        Spark defines driver memory as bytesConf(ByteUnit.MiB), so "4096"
        means 4096 MiB — perfectly valid. Range comparison is skipped for
        unit-less candidates; format/pattern checks still apply.
        """
        param = database.get_parameter("3.5.0", "spark.driver.memory")
        assert param is not None
        assert validator.validate(param, "4096")
        assert validator.get_errors() == []
