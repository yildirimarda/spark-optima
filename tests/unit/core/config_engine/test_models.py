# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for config engine models."""

from __future__ import annotations

import pytest

from spark_optima.core.config_engine.models import (
    ConfigParameter,
    ConfigSet,
    HeuristicRule,
    ParameterCategory,
    ParameterType,
    PlatformSupport,
    ValidationConstraint,
)


class TestParameterCategory:
    """Test cases for ParameterCategory enum."""

    def test_enum_values(self) -> None:
        """Test that all categories are defined correctly."""
        assert ParameterCategory.MEMORY.value == "memory"
        assert ParameterCategory.CPU.value == "cpu"
        assert ParameterCategory.SHUFFLE.value == "shuffle"
        assert ParameterCategory.SQL.value == "sql"

    def test_enum_from_string(self) -> None:
        """Test creating enum from string."""
        cat = ParameterCategory("memory")
        assert cat == ParameterCategory.MEMORY

    def test_all_categories(self) -> None:
        """Test that we have expected categories."""
        categories = list(ParameterCategory)
        expected = [
            "memory",
            "cpu",
            "shuffle",
            "serialization",
            "sql",
            "dynamic_allocation",
            "runtime",
            "io",
            "network",
            "security",
            "scheduler",
            "ui_history",
        ]
        assert len(categories) == 12
        assert all(cat.value in expected for cat in categories)


class TestParameterType:
    """Test cases for ParameterType enum."""

    def test_enum_values(self) -> None:
        """Test that all types are defined correctly."""
        assert ParameterType.STRING.value == "string"
        assert ParameterType.INTEGER.value == "integer"
        assert ParameterType.BYTES.value == "bytes"
        assert ParameterType.DURATION.value == "duration"

    def test_numeric_types(self) -> None:
        """Test numeric parameter types."""
        assert ParameterType.INTEGER.value == "integer"
        assert ParameterType.FLOAT.value == "float"


class TestPlatformSupport:
    """Test cases for PlatformSupport enum."""

    def test_all_platforms(self) -> None:
        """Test that all platforms are defined."""
        platforms = list(PlatformSupport)
        expected = ["local", "databricks", "aws_glue", "azure_synapse", "all"]
        assert all(p.value in expected for p in platforms)


class TestValidationConstraint:
    """Test cases for ValidationConstraint class."""

    def test_default_initialization(self) -> None:
        """Test constraint with default values."""
        constraint = ValidationConstraint()
        assert constraint.min_value is None
        assert constraint.max_value is None
        assert constraint.pattern is None
        assert constraint.allowed_values is None
        assert constraint.depends_on == []

    def test_custom_initialization(self) -> None:
        """Test constraint with custom values."""
        constraint = ValidationConstraint(
            min_value=1,
            max_value=100,
            pattern=r"^\d+$",
            allowed_values=["a", "b", "c"],
            depends_on=["other_param"],
        )
        assert constraint.min_value == 1
        assert constraint.max_value == 100
        assert constraint.pattern == r"^\d+$"
        assert constraint.allowed_values == ["a", "b", "c"]
        assert constraint.depends_on == ["other_param"]

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        constraint = ValidationConstraint(min_value=1, max_value=100)
        data = constraint.to_dict()
        assert data["min_value"] == 1
        assert data["max_value"] == 100

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "min_value": 10,
            "max_value": 1000,
            "pattern": None,
            "allowed_values": ["opt1", "opt2"],
            "depends_on": ["dep1"],
        }
        constraint = ValidationConstraint.from_dict(data)
        assert constraint.min_value == 10
        assert constraint.allowed_values == ["opt1", "opt2"]


class TestHeuristicRule:
    """Test cases for HeuristicRule class."""

    def test_default_initialization(self) -> None:
        """Test rule with default values."""
        rule = HeuristicRule()
        assert rule.formula is None
        assert rule.base_value is None
        assert rule.priority == "medium"
        assert rule.conditions == {}
        assert rule.depends_on == []

    def test_custom_initialization(self) -> None:
        """Test rule with custom values."""
        rule = HeuristicRule(
            formula="memory / 2",
            base_value="4g",
            priority="high",
            conditions={"large_data": True},
            depends_on=["memory"],
            description="Test rule",
        )
        assert rule.formula == "memory / 2"
        assert rule.base_value == "4g"
        assert rule.priority == "high"
        assert rule.depends_on == ["memory"]

    def test_invalid_priority(self) -> None:
        """Test that invalid priority raises error."""
        with pytest.raises(ValueError, match="Priority must be one of"):
            HeuristicRule(priority="invalid")

    def test_can_apply(self) -> None:
        """Test can_apply method."""
        rule = HeuristicRule(depends_on=["memory", "cores"])
        assert rule.can_apply({"memory", "cores", "disk"})
        assert not rule.can_apply({"memory"})
        assert not rule.can_apply(set())

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        rule = HeuristicRule(formula="x * 2", base_value=10, priority="high")
        data = rule.to_dict()
        assert data["formula"] == "x * 2"
        assert data["priority"] == "high"

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "formula": "mem / 4",
            "base_value": "2g",
            "priority": "high",
            "conditions": {},
            "depends_on": ["mem"],
            "description": "Memory allocation",
        }
        rule = HeuristicRule.from_dict(data)
        assert rule.formula == "mem / 4"
        assert rule.priority == "high"


class TestConfigParameter:
    """Test cases for ConfigParameter class."""

    def test_basic_initialization(self) -> None:
        """Test parameter with minimal values."""
        param = ConfigParameter(
            name="spark.executor.memory",
            category=ParameterCategory.MEMORY,
            param_type=ParameterType.BYTES,
        )
        assert param.name == "spark.executor.memory"
        assert param.category == ParameterCategory.MEMORY
        assert param.param_type == ParameterType.BYTES

    def test_full_initialization(self) -> None:
        """Test parameter with all values."""
        heuristic = HeuristicRule(formula="x * 2", base_value="4g")
        param = ConfigParameter(
            name="spark.executor.memory",
            category=ParameterCategory.MEMORY,
            param_type=ParameterType.BYTES,
            default="1g",
            description="Executor memory",
            since_version="2.0.0",
            heuristic=heuristic,
            is_advanced=True,
        )
        assert param.default == "1g"
        assert param.description == "Executor memory"
        assert param.heuristic is not None
        assert param.is_advanced is True

    def test_is_applicable_for(self) -> None:
        """Test platform applicability checking."""
        param = ConfigParameter(
            name="test.param",
            category=ParameterCategory.MEMORY,
            param_type=ParameterType.STRING,
            applicable_platforms=[PlatformSupport.DATABRICKS, PlatformSupport.AWS_GLUE],
        )
        assert param.is_applicable_for(PlatformSupport.DATABRICKS)
        assert param.is_applicable_for("databricks")
        assert not param.is_applicable_for(PlatformSupport.LOCAL)

    def test_is_deprecated_in(self) -> None:
        """Test deprecation checking."""
        param = ConfigParameter(
            name="old.param",
            category=ParameterCategory.MEMORY,
            param_type=ParameterType.STRING,
            deprecated_in="3.5.0",
        )
        assert param.is_deprecated_in("3.5.0")
        assert param.is_deprecated_in("3.6.0")
        assert not param.is_deprecated_in("3.4.0")

    def test_compare_versions(self) -> None:
        """Test version comparison."""
        assert ConfigParameter._compare_versions("3.5.0", "3.5.0") == 0
        assert ConfigParameter._compare_versions("3.5.0", "3.4.0") > 0
        assert ConfigParameter._compare_versions("3.5.0", "3.6.0") < 0
        assert ConfigParameter._compare_versions("4.0.0", "3.9.0") > 0

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        param = ConfigParameter(
            name="spark.test",
            category=ParameterCategory.MEMORY,
            param_type=ParameterType.STRING,
            default="value",
        )
        data = param.to_dict()
        assert data["name"] == "spark.test"
        assert data["category"] == "memory"
        assert data["default"] == "value"

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "name": "spark.executor.cores",
            "category": "cpu",
            "type": "integer",
            "default": 4,
            "description": "Number of cores",
            "since_version": "1.0.0",
            "constraints": {"min_value": 1, "max_value": 64},
            "applicable_platforms": ["local", "databricks"],
            "heuristic": None,
        }
        param = ConfigParameter.from_dict(data)
        assert param.name == "spark.executor.cores"
        assert param.param_type == ParameterType.INTEGER
        assert param.default == 4


class TestConfigSet:
    """Test cases for ConfigSet class."""

    def test_empty_initialization(self) -> None:
        """Test empty config set."""
        config_set = ConfigSet(version="3.5.0")
        assert config_set.version == "3.5.0"
        assert len(config_set) == 0

    def test_with_parameters(self) -> None:
        """Test config set with parameters."""
        params = {
            "spark.executor.memory": ConfigParameter(
                name="spark.executor.memory",
                category=ParameterCategory.MEMORY,
                param_type=ParameterType.BYTES,
                default="1g",
            ),
            "spark.executor.cores": ConfigParameter(
                name="spark.executor.cores",
                category=ParameterCategory.CPU,
                param_type=ParameterType.INTEGER,
                default=2,
            ),
        }
        config_set = ConfigSet(version="3.5.0", parameters=params)
        assert len(config_set) == 2
        assert "spark.executor.memory" in config_set

    def test_get_parameters_by_category(self) -> None:
        """Test filtering by category."""
        params = {
            "param1": ConfigParameter(
                name="param1", category=ParameterCategory.MEMORY, param_type=ParameterType.STRING
            ),
            "param2": ConfigParameter(name="param2", category=ParameterCategory.CPU, param_type=ParameterType.STRING),
            "param3": ConfigParameter(
                name="param3", category=ParameterCategory.MEMORY, param_type=ParameterType.STRING
            ),
        }
        config_set = ConfigSet(version="3.5.0", parameters=params)
        memory_params = config_set.get_parameters_by_category(ParameterCategory.MEMORY)
        assert len(memory_params) == 2
        assert "param1" in memory_params
        assert "param3" in memory_params

    def test_get_parameters_by_category_string(self) -> None:
        """Test filtering by category with string (lines 432-433)."""
        params = {
            "param1": ConfigParameter(
                name="param1", category=ParameterCategory.MEMORY, param_type=ParameterType.STRING
            ),
            "param2": ConfigParameter(name="param2", category=ParameterCategory.CPU, param_type=ParameterType.STRING),
        }
        config_set = ConfigSet(version="3.5.0", parameters=params)
        memory_params = config_set.get_parameters_by_category("memory")
        assert len(memory_params) == 1
        assert "param1" in memory_params

    def test_get_parameters_for_platform(self) -> None:
        """Test filtering by platform."""
        params = {
            "param1": ConfigParameter(
                name="param1",
                category=ParameterCategory.MEMORY,
                param_type=ParameterType.STRING,
                applicable_platforms=[PlatformSupport.DATABRICKS],
            ),
            "param2": ConfigParameter(
                name="param2",
                category=ParameterCategory.MEMORY,
                param_type=ParameterType.STRING,
                applicable_platforms=[PlatformSupport.LOCAL],
            ),
        }
        config_set = ConfigSet(version="3.5.0", parameters=params)
        databricks_params = config_set.get_parameters_for_platform(PlatformSupport.DATABRICKS)
        assert len(databricks_params) == 1
        assert "param1" in databricks_params

    def test_get_parameters_for_platform_string(self) -> None:
        """Test filtering by platform with string (lines 451-452)."""
        params = {
            "param1": ConfigParameter(
                name="param1",
                category=ParameterCategory.MEMORY,
                param_type=ParameterType.STRING,
                applicable_platforms=[PlatformSupport.DATABRICKS],
            ),
        }
        config_set = ConfigSet(version="3.5.0", parameters=params)
        databricks_params = config_set.get_parameters_for_platform("databricks")
        assert len(databricks_params) == 1

    def test_get_default_config(self) -> None:
        """Test getting default configuration."""
        params = {
            "param1": ConfigParameter(
                name="param1",
                category=ParameterCategory.MEMORY,
                param_type=ParameterType.STRING,
                default="value1",
            ),
            "param2": ConfigParameter(
                name="param2",
                category=ParameterCategory.MEMORY,
                param_type=ParameterType.STRING,
                default=None,
            ),
        }
        config_set = ConfigSet(version="3.5.0", parameters=params)
        defaults = config_set.get_default_config()
        assert defaults == {"param1": "value1"}

    def test_to_dict_with_heuristic(self) -> None:
        """Test to_dict with heuristic rule (lines 293-290)."""
        heuristic = HeuristicRule(formula="x * 2", base_value="4g")
        param = ConfigParameter(
            name="spark.test",
            category=ParameterCategory.MEMORY,
            param_type=ParameterType.STRING,
            default="test",
            heuristic=heuristic,
        )
        config_set = ConfigSet(version="3.5.0", parameters={"spark.test": param})
        data = config_set.to_dict()
        assert data["version"] == "3.5.0"
        assert "spark.test" in data["parameters"]
        assert data["parameters"]["spark.test"]["heuristic"] is not None

    def test_to_dict_without_heuristic(self) -> None:
        """Test to_dict without heuristic (lines 287-288)."""
        param = ConfigParameter(
            name="spark.test",
            category=ParameterCategory.MEMORY,
            param_type=ParameterType.STRING,
            default="test",
            heuristic=None,
        )
        config_set = ConfigSet(version="3.5.0", parameters={"spark.test": param})
        data = config_set.to_dict()
        assert data["parameters"]["spark.test"]["heuristic"] is None

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "version": "3.5.0",
            "parameters": {
                "spark.executor.memory": {
                    "name": "spark.executor.memory",
                    "category": "memory",
                    "type": "bytes",
                    "default": "1g",
                    "description": "Executor memory",
                    "since_version": "1.0.0",
                    "constraints": {},
                    "applicable_platforms": ["local"],
                    "heuristic": None,
                }
            },
            "metadata": {"test": True},
        }
        config_set = ConfigSet.from_dict(data)
        assert config_set.version == "3.5.0"
        assert len(config_set) == 1
        assert "spark.executor.memory" in config_set
        assert config_set.metadata == {"test": True}

    def test_getitem(self) -> None:
        """Test dictionary-style access."""
        param = ConfigParameter(name="spark.test", category=ParameterCategory.MEMORY, param_type=ParameterType.STRING)
        config_set = ConfigSet(version="3.5.0", parameters={"spark.test": param})
        assert config_set["spark.test"] == param

    def test_contains(self) -> None:
        """Test 'in' operator."""
        param = ConfigParameter(name="spark.test", category=ParameterCategory.MEMORY, param_type=ParameterType.STRING)
        config_set = ConfigSet(version="3.5.0", parameters={"spark.test": param})
        assert "spark.test" in config_set
        assert "spark.other" not in config_set

    def test_len(self) -> None:
        """Test __len__ method (lines 472-474)."""
        config_set = ConfigSet(version="3.5.0")
        assert len(config_set) == 0

        param = ConfigParameter(name="spark.test", category=ParameterCategory.MEMORY, param_type=ParameterType.STRING)
        config_set.parameters["spark.test"] = param
        assert len(config_set) == 1

    def test_getitem_not_found(self) -> None:
        """Test __getitem__ with missing key (lines 482-483)."""
        config_set = ConfigSet(version="3.5.0")
        try:
            _ = config_set["nonexistent"]
            raise AssertionError("Should have raised KeyError")
        except KeyError:
            pass

    def test_is_deprecated_in_none(self) -> None:
        """Test is_deprecated_in when deprecated_in is None (line 344)."""
        param = ConfigParameter(
            name="current.param",
            category=ParameterCategory.MEMORY,
            param_type=ParameterType.STRING,
            deprecated_in=None,  # No deprecation
        )
        # Should return False when deprecated_in is None
        assert not param.is_deprecated_in("3.5.0")
        assert not param.is_deprecated_in("4.0.0")


class TestLoadTimeBoundCanonicalization:
    """Tests for load-time canonicalization of constraint bounds.

    ``ConfigParameter.from_dict`` parses unit-suffixed string bounds for
    BYTES/DURATION parameters into canonical base units (bytes/seconds), so
    after load every bound in the system is numeric.
    """

    @staticmethod
    def _param_dict(param_type: str, constraints: dict) -> dict:
        """Build a minimal from_dict payload."""
        return {
            "name": "spark.test.param",
            "category": "memory",
            "type": param_type,
            "constraints": constraints,
        }

    def test_bytes_string_bound_canonicalized(self) -> None:
        """A "2048m" max bound becomes 2147483648 bytes."""
        param = ConfigParameter.from_dict(self._param_dict("bytes", {"max_value": "2048m"}))
        assert param.constraints.max_value == 2147483648
        assert isinstance(param.constraints.max_value, int)

    def test_bytes_min_and_max_canonicalized(self) -> None:
        """Both bounds are parsed with the shared byte parser."""
        param = ConfigParameter.from_dict(
            self._param_dict("bytes", {"min_value": "64k", "max_value": "512m"}),
        )
        assert param.constraints.min_value == 64 * 1024
        assert param.constraints.max_value == 512 * 1024**2

    def test_duration_string_bound_canonicalized(self) -> None:
        """A "5m" duration bound becomes 300 seconds."""
        param = ConfigParameter.from_dict(self._param_dict("duration", {"max_value": "5m"}))
        assert param.constraints.max_value == 300

    def test_bare_numeric_bounds_pass_through(self) -> None:
        """Numeric bounds already mean base units and are left unchanged."""
        param = ConfigParameter.from_dict(
            self._param_dict("bytes", {"min_value": 1024, "max_value": 2147483648}),
        )
        assert param.constraints.min_value == 1024
        assert param.constraints.max_value == 2147483648

    def test_non_sized_types_left_untouched(self) -> None:
        """Bounds on non-BYTES/DURATION parameters are not canonicalized."""
        param = ConfigParameter.from_dict(
            {
                "name": "spark.test.cores",
                "category": "cpu",
                "type": "integer",
                "constraints": {"min_value": 1, "max_value": 64},
            },
        )
        assert param.constraints.min_value == 1
        assert param.constraints.max_value == 64

    def test_unparsable_string_bound_raises(self) -> None:
        """An invalid string bound fails loudly at load time, naming the parameter."""
        with pytest.raises(ValueError, match="spark.test.param"):
            ConfigParameter.from_dict(self._param_dict("bytes", {"max_value": "12parsecs"}))

    def test_config_set_from_dict_canonicalizes(self) -> None:
        """Bounds are numeric for parameters loaded through ConfigSet.from_dict."""
        config_set = ConfigSet.from_dict(
            {
                "version": "3.5.0",
                "parameters": {
                    "spark.executor.memory": {
                        "category": "memory",
                        "type": "bytes",
                        "default": "1g",
                        "constraints": {"min_value": "512m"},
                    },
                },
            },
        )
        assert config_set["spark.executor.memory"].constraints.min_value == 512 * 1024**2
