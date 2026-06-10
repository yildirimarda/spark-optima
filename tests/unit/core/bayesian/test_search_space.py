# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for SearchSpaceBuilder."""

from __future__ import annotations

import pytest

from spark_optima.core.bayesian.models import SearchSpaceConfig
from spark_optima.core.bayesian.search_space import SearchSpaceBuilder
from spark_optima.core.config_engine.models import (
    ConfigParameter,
    ConfigSet,
    ParameterCategory,
    ParameterType,
    ValidationConstraint,
)


class TestSearchSpaceBuilder:
    """Tests for SearchSpaceBuilder class."""

    def test_initialization(self) -> None:
        """Test initialization."""
        builder = SearchSpaceBuilder()
        assert builder._search_space == {}

    def test_build_from_heuristic_empty(self) -> None:
        """Test building from empty heuristic config."""
        builder = SearchSpaceBuilder()
        config_set = ConfigSet(version="3.5.0")

        search_space = builder.build_from_heuristic({}, config_set)
        assert search_space == {}

    def test_build_from_heuristic_memory_params(self) -> None:
        """Test building search space for memory parameters."""
        builder = SearchSpaceBuilder()
        config_set = ConfigSet(version="3.5.0")

        heuristic_config = {
            "spark.executor.memory": "4g",
            "spark.driver.memory": "2g",
        }

        search_space = builder.build_from_heuristic(
            heuristic_config, config_set, SearchSpaceConfig(variation_percent=0.3)
        )

        assert "spark.executor.memory" in search_space
        assert search_space["spark.executor.memory"]["type"] == "bytes"
        # 4GB with 30% variation
        assert search_space["spark.executor.memory"]["low"] < 4 * 1024**3
        assert search_space["spark.executor.memory"]["high"] > 4 * 1024**3

    def test_build_from_heuristic_cores(self) -> None:
        """Test building search space for core parameters."""
        builder = SearchSpaceBuilder()
        config_set = ConfigSet(version="3.5.0")

        heuristic_config = {
            "spark.executor.cores": 4,
        }

        search_space = builder.build_from_heuristic(heuristic_config, config_set)

        assert "spark.executor.cores" in search_space
        # Cores should be categorical
        assert search_space["spark.executor.cores"]["type"] == "categorical"
        assert 4 in search_space["spark.executor.cores"]["choices"]

    def test_build_from_heuristic_boolean_params(self) -> None:
        """Test building search space for boolean parameters."""
        builder = SearchSpaceBuilder()
        config_set = ConfigSet(version="3.5.0")

        heuristic_config = {
            "spark.sql.adaptive.enabled": True,
            "spark.shuffle.compress": True,
        }

        search_space = builder.build_from_heuristic(heuristic_config, config_set)

        assert "spark.sql.adaptive.enabled" in search_space
        assert search_space["spark.sql.adaptive.enabled"]["type"] == "categorical"
        assert True in search_space["spark.sql.adaptive.enabled"]["choices"]
        assert False in search_space["spark.sql.adaptive.enabled"]["choices"]

    def test_build_from_heuristic_fixed_params_skipped(self) -> None:
        """Test that fixed parameters are skipped."""
        builder = SearchSpaceBuilder()
        config_set = ConfigSet(version="3.5.0")

        heuristic_config = {
            "spark.app.name": "MyApp",
            "spark.master": "local[*]",
            "spark.executor.memory": "4g",
        }

        search_space = builder.build_from_heuristic(heuristic_config, config_set)

        assert "spark.app.name" not in search_space
        assert "spark.master" not in search_space
        assert "spark.executor.memory" in search_space

    def test_build_with_custom_fixed_params(self) -> None:
        """Test building with custom fixed parameters."""
        builder = SearchSpaceBuilder()
        config_set = ConfigSet(version="3.5.0")

        heuristic_config = {
            "spark.executor.memory": "4g",
            "spark.custom.param": "value",
        }

        config = SearchSpaceConfig(fixed_params={"spark.custom.param": "value"})
        search_space = builder.build_from_heuristic(heuristic_config, config_set, config)

        assert "spark.custom.param" not in search_space

    def test_parse_memory_string(self) -> None:
        """Test parsing memory strings."""
        assert SearchSpaceBuilder._parse_memory_string("4g") == 4 * 1024**3
        assert SearchSpaceBuilder._parse_memory_string("512m") == 512 * 1024**2
        assert SearchSpaceBuilder._parse_memory_string("1024k") == 1024 * 1024
        assert SearchSpaceBuilder._parse_memory_string("1t") == 1024**4

    def test_parse_memory_string_with_spaces(self) -> None:
        """Test parsing memory strings with spaces."""
        assert SearchSpaceBuilder._parse_memory_string(" 4g ") == 4 * 1024**3
        assert SearchSpaceBuilder._parse_memory_string("512 m") == 512 * 1024**2

    def test_parse_memory_string_invalid(self) -> None:
        """Test parsing invalid memory strings."""
        with pytest.raises(ValueError):
            SearchSpaceBuilder._parse_memory_string("invalid")

    def test_parse_duration_string(self) -> None:
        """Test parsing duration strings."""
        assert SearchSpaceBuilder._parse_duration_string("600s") == 600
        assert SearchSpaceBuilder._parse_duration_string("5m") == 300
        assert SearchSpaceBuilder._parse_duration_string("1h") == 3600
        assert SearchSpaceBuilder._parse_duration_string("600") == 600

    def test_parse_duration_string_with_suffix(self) -> None:
        """Test parsing duration strings with 's' suffix."""
        assert SearchSpaceBuilder._parse_duration_string("600s") == 600
        assert SearchSpaceBuilder._parse_duration_string("300s") == 300

    def test_parse_duration_string_invalid(self) -> None:
        """Test parsing invalid duration strings."""
        with pytest.raises(ValueError):
            SearchSpaceBuilder._parse_duration_string("invalid")

    def test_parse_boolean(self) -> None:
        """Test parsing boolean values."""
        assert SearchSpaceBuilder._parse_boolean(True) is True
        assert SearchSpaceBuilder._parse_boolean(False) is False
        assert SearchSpaceBuilder._parse_boolean("true") is True
        assert SearchSpaceBuilder._parse_boolean("false") is False
        assert SearchSpaceBuilder._parse_boolean("yes") is True
        assert SearchSpaceBuilder._parse_boolean("no") is False
        assert SearchSpaceBuilder._parse_boolean(1) is True
        assert SearchSpaceBuilder._parse_boolean(0) is False

    def test_get_search_space(self) -> None:
        """Test getting search space."""
        builder = SearchSpaceBuilder()
        config_set = ConfigSet(version="3.5.0")

        heuristic_config = {"spark.executor.memory": "4g"}
        builder.build_from_heuristic(heuristic_config, config_set)

        space = builder.get_search_space()
        assert "spark.executor.memory" in space

    def test_filter_by_category(self) -> None:
        """Test filtering search space by category."""
        builder = SearchSpaceBuilder()
        config_set = ConfigSet(version="3.5.0")

        heuristic_config = {
            "spark.executor.memory": "4g",
            "spark.executor.cores": 4,
        }
        builder.build_from_heuristic(heuristic_config, config_set)

        memory_space = builder.filter_by_category(config_set, ParameterCategory.MEMORY)
        # Note: Without actual parameters in config_set, this may be empty
        # But the method should work without error
        assert isinstance(memory_space, dict)

    def test_build_memory_search_space(self) -> None:
        """Test _build_memory_search_space method (lines 228-277)."""
        builder = SearchSpaceBuilder()
        config = SearchSpaceConfig(variation_percent=0.3)

        # Test with "4g"
        result = builder._build_memory_search_space("spark.executor.memory", "4g", config)
        assert result is not None
        assert result["type"] == "bytes"
        assert "low" in result
        assert "high" in result

    def test_build_core_search_space_cores(self) -> None:
        """Test _build_core_search_space for executor cores (lines 279-342)."""
        builder = SearchSpaceBuilder()
        config = SearchSpaceConfig(variation_percent=0.3)

        # Test spark.executor.cores
        result = builder._build_core_search_space("spark.executor.cores", "4", config)
        assert result is not None
        assert result["type"] == "categorical"
        assert 4 in result["choices"]

    def test_build_core_search_space_partitions(self) -> None:
        """Test _build_core_search_space for partitions (lines 318-342)."""
        builder = SearchSpaceBuilder()
        config = SearchSpaceConfig(variation_percent=0.3)

        # Test spark.sql.shuffle.partitions
        result = builder._build_core_search_space("spark.sql.shuffle.partitions", "200", config)
        assert result is not None
        assert result["type"] == "int"
        assert "low" in result
        assert "high" in result

    def test_build_float_search_space(self) -> None:
        """Test _build_float_search_space method (lines 384-419)."""
        builder = SearchSpaceBuilder()
        from spark_optima.core.config_engine.models import (
            ConfigParameter,
            ParameterCategory,
            ValidationConstraint,
        )

        param = ConfigParameter(
            name="test.param",
            param_type=ParameterType.FLOAT,
            category=ParameterCategory.RUNTIME,
            default="0.5",
            constraints=ValidationConstraint(min_value=0.0, max_value=1.0),
        )
        config = SearchSpaceConfig(variation_percent=0.3)

        result = builder._build_float_search_space("test.param", "0.5", param, config)
        assert result is not None
        assert result["type"] == "float"
        assert "low" in result
        assert "high" in result

    def test_build_integer_search_space(self) -> None:
        """Test _build_integer_search_space method (lines 421-456)."""
        builder = SearchSpaceBuilder()
        from spark_optima.core.config_engine.models import (
            ConfigParameter,
            ParameterCategory,
            ValidationConstraint,
        )

        param = ConfigParameter(
            name="test.param",
            param_type=ParameterType.INTEGER,
            category=ParameterCategory.RUNTIME,
            default="10",
            constraints=ValidationConstraint(min_value=1, max_value=100),
        )
        config = SearchSpaceConfig(variation_percent=0.3)

        result = builder._build_integer_search_space("test.param", "10", param, config)
        assert result is not None
        assert result["type"] == "int"
        assert "low" in result
        assert "high" in result

    def test_build_bytes_search_space(self) -> None:
        """Test _build_bytes_search_space method (lines 458-490)."""
        builder = SearchSpaceBuilder()
        config = SearchSpaceConfig(variation_percent=0.3)

        result = builder._build_bytes_search_space("test.param", "1m", config)
        assert result is not None
        assert result["type"] == "bytes"
        assert "low" in result
        assert "high" in result

    def test_build_duration_search_space(self) -> None:
        """Test _build_duration_search_space method (lines 344-382)."""
        builder = SearchSpaceBuilder()
        config = SearchSpaceConfig(variation_percent=0.3)

        result = builder._build_duration_search_space("spark.network.timeout", "600s", config)
        assert result is not None
        assert result["type"] == "int"
        assert "low" in result
        assert "high" in result

    def test_parse_memory_string_various_formats(self) -> None:
        """Test _parse_memory_string with various formats (lines 506-531)."""
        # Test various formats
        assert SearchSpaceBuilder._parse_memory_string("1024") == 1024  # Raw bytes
        assert SearchSpaceBuilder._parse_memory_string("1k") == 1024
        assert SearchSpaceBuilder._parse_memory_string("1m") == 1024**2
        assert SearchSpaceBuilder._parse_memory_string("1g") == 1024**3
        assert SearchSpaceBuilder._parse_memory_string("1t") == 1024**4
        assert SearchSpaceBuilder._parse_memory_string(" 4g ") == 4 * 1024**3  # With spaces

    def test_parse_memory_string_b_with_k(self) -> None:
        """Test _parse_memory_string with 'b' suffix (lines 508-515)."""
        assert SearchSpaceBuilder._parse_memory_string("1kb") == 1024
        assert SearchSpaceBuilder._parse_memory_string("1mb") == 1024**2

    def test_parse_duration_string_various_formats(self) -> None:
        """Test _parse_duration_string with various formats (lines 547-588)."""
        assert SearchSpaceBuilder._parse_duration_string("1000ms") == 1  # Milliseconds
        assert SearchSpaceBuilder._parse_duration_string("1s") == 1
        assert SearchSpaceBuilder._parse_duration_string("1m") == 60
        assert SearchSpaceBuilder._parse_duration_string("1h") == 3600
        assert SearchSpaceBuilder._parse_duration_string("1d") == 86400
        assert SearchSpaceBuilder._parse_duration_string(" 60s ") == 60  # With spaces

    def test_build_parameter_search_space_categorical(self) -> None:
        """Test _build_parameter_search_space for categorical (lines 155-226)."""
        builder = SearchSpaceBuilder()
        config = SearchSpaceConfig()

        # Test categorical parameter
        result = builder._build_parameter_search_space(
            "spark.serializer",
            "org.apache.spark.serializer.KryoSerializer",
            None,
            config,
        )
        assert result is not None
        assert result["type"] == "categorical"

    def test_build_parameter_search_space_boolean(self) -> None:
        """Test _build_parameter_search_space for boolean (lines 192-198)."""
        builder = SearchSpaceBuilder()
        config = SearchSpaceConfig()

        # Test boolean parameter
        result = builder._build_parameter_search_space(
            "spark.sql.adaptive.enabled",
            "true",
            None,
            config,
        )
        assert result is not None
        assert result["type"] == "categorical"
        assert True in result["choices"]
        assert False in result["choices"]

    def test_build_parameter_search_space_fixed(self) -> None:
        """Test _build_parameter_search_space skips fixed params (lines 134-135)."""
        builder = SearchSpaceBuilder()
        config = SearchSpaceConfig()

        # Test fixed parameter - should return None
        result = builder._build_parameter_search_space(
            "spark.app.name",
            "MyApp",
            None,
            config,
        )
        assert result is None

    def test_build_parameter_search_space_param_ranges(self) -> None:
        """Test _build_parameter_search_space with custom ranges (lines 175-182)."""
        builder = SearchSpaceBuilder()
        config = SearchSpaceConfig(param_ranges={"custom.param": (10.0, 100.0)})

        result = builder._build_parameter_search_space(
            "custom.param",
            "50.0",
            None,
            config,
        )
        assert result is not None
        assert result["type"] == "float"
        assert result["low"] == 10.0
        assert result["high"] == 100.0


class TestSearchSpaceBuilderMoreCoverage:
    """Additional tests for 100% coverage."""

    def test_build_from_heuristic_with_config_set(self) -> None:
        """Test build_from_heuristic with actual ConfigParameter."""
        builder = SearchSpaceBuilder()
        from spark_optima.core.config_engine.models import (
            ConfigParameter,
            ParameterCategory,
            ValidationConstraint,
        )

        # Create a config set with actual parameters
        param = ConfigParameter(
            name="spark.executor.memory",
            category=ParameterCategory.MEMORY,
            param_type=ParameterType.BYTES,
            default="1g",
            constraints=ValidationConstraint(min_value=512 * 1024 * 1024, max_value=16 * 1024**3),
        )
        config_set = ConfigSet(version="3.5.0", parameters={"spark.executor.memory": param})

        heuristic_config = {"spark.executor.memory": "4g"}

        search_space = builder.build_from_heuristic(heuristic_config, config_set)

        assert "spark.executor.memory" in search_space
        assert search_space["spark.executor.memory"]["type"] == "bytes"

    def test_build_parameter_search_space_with_param(self) -> None:
        """Test _build_parameter_search_space with actual param (lines 213-226)."""
        builder = SearchSpaceBuilder()
        from spark_optima.core.config_engine.models import (
            ParameterCategory,
            ParameterType,
        )

        param = ConfigParameter(
            name="spark.test.float",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.FLOAT,
            default="0.5",
            constraints=ValidationConstraint(min_value=0.0, max_value=1.0),
        )

        result = builder._build_parameter_search_space(
            "spark.test.float",
            "0.5",
            param,
            SearchSpaceConfig(),
        )

        assert result is not None
        assert result["type"] == "float"

    def test_build_parameter_search_space_int_param(self) -> None:
        """Test _build_parameter_search_space with int param."""
        builder = SearchSpaceBuilder()
        from spark_optima.core.config_engine.models import (
            ParameterCategory,
            ParameterType,
        )

        param = ConfigParameter(
            name="spark.test.int",
            category=ParameterCategory.RUNTIME,
            param_type=ParameterType.INTEGER,
            default="10",
            constraints=ValidationConstraint(min_value=1, max_value=100),
        )

        result = builder._build_parameter_search_space(
            "spark.test.int",
            "10",
            param,
            SearchSpaceConfig(),
        )

        assert result is not None
        assert result["type"] == "int"

    def test_build_parameter_search_space_bytes_param(self) -> None:
        """Test _build_parameter_search_space with bytes param."""
        builder = SearchSpaceBuilder()
        from spark_optima.core.config_engine.models import (
            ParameterCategory,
            ParameterType,
        )

        param = ConfigParameter(
            name="spark.test.bytes",
            category=ParameterCategory.MEMORY,
            param_type=ParameterType.BYTES,
            default="1g",
        )

        result = builder._build_parameter_search_space(
            "spark.test.bytes",
            "1g",
            param,
            SearchSpaceConfig(),
        )

        assert result is not None
        assert result["type"] == "bytes"

    def test_build_core_search_space_parallelism(self) -> None:
        """Test _build_core_search_space for parallelism (lines 318-342)."""
        builder = SearchSpaceBuilder()
        config = SearchSpaceConfig(variation_percent=0.3)

        # Test spark.default.parallelism
        result = builder._build_core_search_space(
            "spark.default.parallelism",
            "200",
            config,
        )

        assert result is not None
        assert result["type"] == "int"

    def test_build_core_search_space_task_cpus(self) -> None:
        """Test _build_core_search_space for task cpus."""
        builder = SearchSpaceBuilder()
        config = SearchSpaceConfig(variation_percent=0.3)

        result = builder._build_core_search_space(
            "spark.task.cpus",
            "1",
            config,
        )

        assert result is not None
        assert result["type"] == "int"
        assert 1 in range(result["low"], result["high"] + 1)

    def test_build_memory_search_space_rounding(self) -> None:
        """Test _build_memory_search_space rounding (lines 260-269)."""
        builder = SearchSpaceBuilder()
        config = SearchSpaceConfig(variation_percent=0.3)

        # Test with value that needs rounding
        result = builder._build_memory_search_space(
            "spark.test.mem",
            "3.7g",  # 3.7 * 1024^3 bytes
            config,
        )

        assert result is not None
        assert result["low"] <= result["high"]

    def test_filter_by_category_no_match(self) -> None:
        """Test filter_by_category with no matching params."""
        builder = SearchSpaceBuilder()
        config_set = ConfigSet(version="3.5.0")

        # Build empty search space
        builder.build_from_heuristic({}, config_set)

        result = builder.filter_by_category(config_set, ParameterCategory.MEMORY)
        assert isinstance(result, dict)

    def test_parse_duration_string_ms(self) -> None:
        """Test _parse_duration_string with ms suffix."""
        assert SearchSpaceBuilder._parse_duration_string("500ms") == 0.5  # 500ms = 0.5s
        assert SearchSpaceBuilder._parse_duration_string("1000ms") == 1  # 1000ms = 1s

    def test_parse_duration_string_minutes(self) -> None:
        """Test _parse_duration_string with minute variants."""
        assert SearchSpaceBuilder._parse_duration_string("5min") == 300
        assert SearchSpaceBuilder._parse_duration_string("1minutes") == 60

    def test_parse_duration_string_hours(self) -> None:
        """Test _parse_duration_string with hour variants."""
        assert SearchSpaceBuilder._parse_duration_string("2hour") == 7200
        assert SearchSpaceBuilder._parse_duration_string("1hours") == 3600

    def test_parse_duration_string_days(self) -> None:
        """Test _parse_duration_string with day variants."""
        assert SearchSpaceBuilder._parse_duration_string("2day") == 172800
        assert SearchSpaceBuilder._parse_duration_string("1days") == 86400

    def test_parse_duration_string_integer(self) -> None:
        """Test _parse_duration_string with plain integer."""
        assert SearchSpaceBuilder._parse_duration_string("100") == 100
        assert SearchSpaceBuilder._parse_duration_string("   200  ") == 200  # With spaces

    def test_parse_memory_string_kb(self) -> None:
        """Test _parse_memory_string with kb suffix."""
        assert SearchSpaceBuilder._parse_memory_string("1kb") == 1024
        assert SearchSpaceBuilder._parse_memory_string("2kb") == 2 * 1024

    def test_parse_memory_string_mb(self) -> None:
        """Test _parse_memory_string with mb suffix."""
        assert SearchSpaceBuilder._parse_memory_string("1mb") == 1024**2
        assert SearchSpaceBuilder._parse_memory_string("2mb") == 2 * 1024**2

    def test_parse_memory_string_gb(self) -> None:
        """Test _parse_memory_string with gb suffix."""
        assert SearchSpaceBuilder._parse_memory_string("1gb") == 1024**3
        assert SearchSpaceBuilder._parse_memory_string("2gb") == 2 * 1024**3

    def test_parse_memory_string_tb(self) -> None:
        """Test _parse_memory_string with tb suffix."""
        assert SearchSpaceBuilder._parse_memory_string("1tb") == 1024**4
        assert SearchSpaceBuilder._parse_memory_string("2tb") == 2 * 1024**4

    def test_parse_memory_string_invalid_unit(self) -> None:
        """Test _parse_memory_string with invalid unit."""
        with pytest.raises(ValueError):
            SearchSpaceBuilder._parse_memory_string("4x")  # Invalid unit

    def test_parse_memory_string_no_number(self) -> None:
        """Test _parse_memory_string with no number."""
        with pytest.raises(ValueError):
            SearchSpaceBuilder._parse_memory_string("g")  # No number


class TestSearchSpaceBuilderToTrialParams:
    """Tests for SearchSpaceBuilder.to_trial_params (heuristic seed trial mapping)."""

    @pytest.fixture
    def builder(self) -> SearchSpaceBuilder:
        """Create a search space builder."""
        return SearchSpaceBuilder()

    @pytest.fixture
    def config_set(self) -> ConfigSet:
        """Create an empty config set."""
        return ConfigSet(version="3.5.0")

    def test_memory_param_mapped_to_bytes_on_grid(
        self,
        builder: SearchSpaceBuilder,
        config_set: ConfigSet,
    ) -> None:
        """Memory values are parsed to bytes, clamped, and snapped to the step grid."""
        heuristic = {"spark.executor.memory": "4g"}
        space = builder.build_from_heuristic(heuristic, config_set)

        params = builder.to_trial_params(heuristic)

        assert "spark.executor.memory" in params
        value = params["spark.executor.memory"]
        low = int(space["spark.executor.memory"]["low"])
        high = int(space["spark.executor.memory"]["high"])
        step = SearchSpaceBuilder.compute_bytes_step(low, high)
        assert low <= value <= high
        assert (value - low) % step == 0
        # The snapped value stays within one step of the heuristic baseline
        assert abs(value - 4 * 1024**3) <= step

    def test_boolean_param_parsed_from_string(
        self,
        builder: SearchSpaceBuilder,
        config_set: ConfigSet,
    ) -> None:
        """String boolean values map to the True/False categorical choices."""
        heuristic = {"spark.sql.adaptive.enabled": "true", "spark.shuffle.compress": "false"}
        builder.build_from_heuristic(heuristic, config_set)

        params = builder.to_trial_params(heuristic)

        assert params["spark.sql.adaptive.enabled"] is True
        assert params["spark.shuffle.compress"] is False

    def test_executor_cores_string_matched_to_int_choice(
        self,
        builder: SearchSpaceBuilder,
        config_set: ConfigSet,
    ) -> None:
        """String core counts map to the integer categorical choices."""
        heuristic = {"spark.executor.cores": "4"}
        builder.build_from_heuristic(heuristic, config_set)

        params = builder.to_trial_params(heuristic)

        assert params["spark.executor.cores"] == 4
        assert isinstance(params["spark.executor.cores"], int)

    def test_categorical_value_not_in_choices_is_skipped(
        self,
        builder: SearchSpaceBuilder,
        config_set: ConfigSet,
    ) -> None:
        """Categorical values without a matching choice are skipped."""
        heuristic = {"spark.serializer": "com.example.CustomSerializer"}
        builder.build_from_heuristic(heuristic, config_set)

        params = builder.to_trial_params(heuristic)

        assert "spark.serializer" not in params

    def test_int_param_within_bounds_and_step_aligned(
        self,
        builder: SearchSpaceBuilder,
        config_set: ConfigSet,
    ) -> None:
        """Integer values land within bounds and on the step grid."""
        heuristic = {"spark.sql.shuffle.partitions": "200"}
        space = builder.build_from_heuristic(heuristic, config_set)

        params = builder.to_trial_params(heuristic)

        value = params["spark.sql.shuffle.partitions"]
        low = int(space["spark.sql.shuffle.partitions"]["low"])
        high = int(space["spark.sql.shuffle.partitions"]["high"])
        step = int(space["spark.sql.shuffle.partitions"].get("step", 1))
        assert low <= value <= high
        assert (value - low) % step == 0
        assert abs(value - 200) <= step

    def test_duration_string_parsed_to_seconds(
        self,
        builder: SearchSpaceBuilder,
        config_set: ConfigSet,
    ) -> None:
        """Duration strings (e.g. "600s") are parsed into the seconds-based range."""
        heuristic = {"spark.network.timeout": "600s"}
        space = builder.build_from_heuristic(heuristic, config_set)

        params = builder.to_trial_params(heuristic)

        value = params["spark.network.timeout"]
        assert space["spark.network.timeout"]["low"] <= value <= space["spark.network.timeout"]["high"]
        assert isinstance(value, int)

    def test_params_outside_search_space_are_skipped(
        self,
        builder: SearchSpaceBuilder,
        config_set: ConfigSet,
    ) -> None:
        """Config entries not present in the search space are silently skipped."""
        heuristic = {
            "spark.executor.memory": "4g",
            "spark.app.name": "my-app",  # FIXED_PARAMS - excluded from the space
            "custom.unknown.param": "x",  # No metadata - excluded from the space
        }
        builder.build_from_heuristic(heuristic, config_set)

        params = builder.to_trial_params(heuristic)

        assert "spark.executor.memory" in params
        assert "spark.app.name" not in params
        assert "custom.unknown.param" not in params

    def test_out_of_range_int_value_clamped_to_bounds(
        self,
        builder: SearchSpaceBuilder,
        config_set: ConfigSet,
    ) -> None:
        """Values above/below the search range are clamped to the bounds."""
        space = builder.build_from_heuristic({"spark.sql.shuffle.partitions": "200"}, config_set)
        low = int(space["spark.sql.shuffle.partitions"]["low"])
        high = int(space["spark.sql.shuffle.partitions"]["high"])

        too_high = builder.to_trial_params({"spark.sql.shuffle.partitions": 100000})
        too_low = builder.to_trial_params({"spark.sql.shuffle.partitions": 1})

        assert low <= too_high["spark.sql.shuffle.partitions"] <= high
        assert too_low["spark.sql.shuffle.partitions"] == low

    def test_out_of_range_float_value_clamped(
        self,
        builder: SearchSpaceBuilder,
        config_set: ConfigSet,
    ) -> None:
        """Float values outside a custom range are clamped to the range edges."""
        search_config = SearchSpaceConfig(
            param_ranges={"spark.memory.fraction": (0.5, 0.9)},
        )
        builder.build_from_heuristic({"spark.memory.fraction": 0.95}, config_set, search_config)

        params = builder.to_trial_params({"spark.memory.fraction": 0.95})

        assert params["spark.memory.fraction"] == pytest.approx(0.9)

    def test_explicit_search_space_argument(
        self,
        builder: SearchSpaceBuilder,
    ) -> None:
        """An explicitly passed search space overrides the stored one."""
        space = {
            "spark.task.cpus": {"type": "int", "low": 1, "high": 4, "base_value": 2},
        }

        params = builder.to_trial_params({"spark.task.cpus": 2}, space)

        assert params == {"spark.task.cpus": 2}

    def test_unparseable_value_falls_back_to_base_value(
        self,
        builder: SearchSpaceBuilder,
    ) -> None:
        """Unparseable raw values fall back to the search-space base value."""
        space = {
            "spark.network.timeout": {
                "type": "int",
                "low": 100,
                "high": 1000,
                "base_value": 600,
            },
        }

        params = builder.to_trial_params({"spark.network.timeout": "not-a-number"}, space)

        assert params["spark.network.timeout"] == 600

    def test_unknown_type_skipped(self, builder: SearchSpaceBuilder) -> None:
        """Parameters with unknown types are not enqueued (never suggested)."""
        space = {"some.param": {"type": "mystery", "base_value": "x"}}

        params = builder.to_trial_params({"some.param": "x"}, space)

        assert params == {}

    def test_empty_config_returns_empty_params(self, builder: SearchSpaceBuilder) -> None:
        """An empty configuration produces no trial parameters."""
        assert builder.to_trial_params({}) == {}

    def test_snap_to_grid_alignment(self) -> None:
        """_snap_to_grid clamps and aligns values onto low + k * step."""
        # In range, snapped to nearest grid point
        assert SearchSpaceBuilder._snap_to_grid(200, 138, 258, 6) == 198
        # Above range: clamped to highest aligned point
        assert SearchSpaceBuilder._snap_to_grid(1000, 138, 258, 6) == 258
        # Below range: clamped to low
        assert SearchSpaceBuilder._snap_to_grid(0, 138, 258, 6) == 138
        # Degenerate step is treated as 1
        assert SearchSpaceBuilder._snap_to_grid(5, 1, 10, 0) == 5
