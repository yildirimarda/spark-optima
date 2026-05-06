# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for spark_optima.core.heuristics.context module."""

from spark_optima.core.heuristics.context import DataProfile, EvaluationContext
from spark_optima.platforms.models import ResourceSpec


class TestDataProfile:
    """Tests for DataProfile class."""

    def test_default_initialization(self) -> None:
        """Test DataProfile with default values."""
        profile = DataProfile()
        assert profile.format == "parquet"
        assert profile.size_gb == 0.0
        assert profile.num_files == 0
        assert profile.avg_file_size_mb == 0.0
        assert profile.compression is None
        assert profile.schema is None
        assert profile.schema_complexity == "medium"
        assert profile.has_nulls is True
        assert profile.is_partitioned is False
        assert profile.partition_columns == []
        assert profile.partitioning is None

    def test_initialization_with_partitioning(self) -> None:
        """Test DataProfile initialization with partitioning alias (lines 53-54)."""
        profile = DataProfile(
            partitioning=["year", "month"],
            partition_columns=[],  # Empty, should be filled from partitioning
        )
        assert profile.partition_columns == ["year", "month"]
        assert profile.is_partitioned is True

    def test_initialization_with_partitioning_no_partition_columns(self) -> None:
        """Test DataProfile initialization with partitioning but is_partitioned not set."""
        profile = DataProfile(
            partitioning=["region"],
            partition_columns=[],
        )
        assert profile.is_partitioned is True

    def test_initialization_with_partition_columns_and_partitioning(self) -> None:
        """Test when both partition_columns and partitioning are provided."""
        profile = DataProfile(
            partition_columns=["col1"],
            partitioning=["col2"],
        )
        # partition_columns takes precedence when not empty
        assert profile.partition_columns == ["col1"]

    def test_schema_string_parsing_with_colon(self) -> None:
        """Test schema string parsing with col:type format (lines 60-67)."""
        profile = DataProfile(schema="col1:int,col2:string,col3:double")
        assert profile.schema == {
            "col1": "int",
            "col2": "string",
            "col3": "double",
        }

    def test_schema_string_parsing_without_colon(self) -> None:
        """Test schema string parsing without colon (defaults to string type)."""
        profile = DataProfile(schema="col1,col2,col3")
        assert profile.schema == {
            "col1": "string",
            "col2": "string",
            "col3": "string",
        }

    def test_schema_string_parsing_mixed(self) -> None:
        """Test schema string parsing with mixed format."""
        profile = DataProfile(schema="col1:int,col2")
        assert profile.schema == {
            "col1": "int",
            "col2": "string",
        }

    def test_schema_string_empty_after_parse(self) -> None:
        """Test schema string that results in empty dict."""
        profile = DataProfile(schema="")
        assert profile.schema is None

    def test_schema_string_whitespace_handling(self) -> None:
        """Test schema string parsing handles whitespace."""
        profile = DataProfile(schema=" col1 : int , col2 : string ")
        assert profile.schema == {
            "col1": "int",
            "col2": "string",
        }

    def test_schema_dict_no_parsing(self) -> None:
        """Test that dict schema is not parsed."""
        schema_dict = {"col1": "int", "col2": "string"}
        profile = DataProfile(schema=schema_dict)
        assert profile.schema == schema_dict

    def test_to_dict(self) -> None:
        """Test to_dict method (line 71)."""
        profile = DataProfile(
            format="csv",
            size_gb=10.5,
            num_files=100,
            avg_file_size_mb=128.0,
            compression="gzip",
            schema={"col1": "int"},
            schema_complexity="high",
            has_nulls=False,
            is_partitioned=True,
            partition_columns=["date"],
        )
        result = profile.to_dict()
        assert result == {
            "format": "csv",
            "size_gb": 10.5,
            "num_files": 100,
            "avg_file_size_mb": 128.0,
            "compression": "gzip",
            "schema": {"col1": "int"},
            "schema_complexity": "high",
            "has_nulls": False,
            "is_partitioned": True,
            "partition_columns": ["date"],
        }

    def test_from_dict_with_partitioning_alias(self) -> None:
        """Test from_dict with partitioning alias (lines 88-94)."""
        data = {
            "partitioning": ["year", "month"],
            "format": "parquet",
        }
        profile = DataProfile.from_dict(data)
        assert profile.partition_columns == ["year", "month"]
        assert profile.is_partitioned is True

    def test_from_dict_with_partition_columns(self) -> None:
        """Test from_dict with partition_columns provided."""
        data = {
            "partition_columns": ["region"],
            "format": "delta",
        }
        profile = DataProfile.from_dict(data)
        assert profile.partition_columns == ["region"]

    def test_from_dict_with_both_partitioning_and_partition_columns(self) -> None:
        """Test from_dict when both partitioning and partition_columns exist."""
        data = {
            "partition_columns": ["col1"],
            "partitioning": ["col2"],
            "format": "parquet",
        }
        profile = DataProfile.from_dict(data)
        # partition_columns should be used
        assert profile.partition_columns == ["col1"]

    def test_from_dict_with_partitioning_non_list(self) -> None:
        """Test from_dict with partitioning as non-list (should not crash)."""
        data = {
            "partitioning": "year",  # String instead of list
            "format": "parquet",
        }
        profile = DataProfile.from_dict(data)
        # partitioning is not a list, so partition_columns stays empty
        assert profile.partition_columns == []

    def test_from_dict_defaults(self) -> None:
        """Test from_dict with minimal data uses defaults."""
        profile = DataProfile.from_dict({})
        assert profile.format == "parquet"
        assert profile.size_gb == 0.0
        assert profile.num_files == 0


class TestEvaluationContext:
    """Tests for EvaluationContext class."""

    def test_default_initialization(self) -> None:
        """Test EvaluationContext with default values."""
        context = EvaluationContext()
        assert context.resources.cpu_cores == 4
        assert context.resources.memory_gb == 16
        assert context.platform == "local"
        assert context.spark_version == "3.5.0"
        assert context.num_executors == 2
        assert context.executor_cores == 4
        assert context.executor_memory_gb == 4.0
        assert context.driver_memory_gb == 4.0
        assert context.memory_overhead_factor == 0.1
        assert context.custom_vars == {}

    def test_initialization_with_custom_resources(self) -> None:
        """Test EvaluationContext with custom ResourceSpec."""
        resources = ResourceSpec(cpu_cores=8, memory_gb=32, disk_gb=100)
        context = EvaluationContext(resources=resources)
        assert context.resources.cpu_cores == 8
        assert context.resources.memory_gb == 32

    def test_to_variables(self) -> None:
        """Test to_variables method returns all expected keys."""
        resources = ResourceSpec(
            cpu_cores=16, memory_gb=64, disk_gb=200, gpu_count=2, network_gbps=25.0
        )
        data_profile = DataProfile(size_gb=50.0, num_files=100, avg_file_size_mb=128.0)
        context = EvaluationContext(
            resources=resources,
            platform="databricks",
            spark_version="3.4.0",
            data_profile=data_profile,
            num_executors=4,
            executor_cores=8,
            executor_memory_gb=16.0,
            driver_memory_gb=8.0,
            memory_overhead_factor=0.15,
            custom_vars={"custom_key": "custom_value"},
        )
        variables = context.to_variables()

        # Check resource variables
        assert variables["total_memory_gb"] == 64
        assert variables["total_cores"] == 16
        assert variables["total_disk_gb"] == 200
        assert variables["gpu_count"] == 2
        assert variables["network_gbps"] == 25.0

        # Check calculated variables
        assert variables["num_executors"] == 4
        assert variables["executor_cores"] == 8
        assert variables["executor_memory_gb"] == 16.0
        assert variables["driver_memory_gb"] == 8.0
        assert variables["memory_overhead_factor"] == 0.15
        assert variables["total_executor_memory_gb"] == 64.0  # 16 * 4
        assert variables["total_cores_cluster"] == 32  # 8 * 4

        # Check data profile variables
        assert variables["data_size_gb"] == 50.0
        assert variables["data_num_files"] == 100
        assert variables["data_avg_file_size_mb"] == 128.0

        # Check platform variables
        assert variables["platform"] == "databricks"
        assert variables["spark_version"] == "3.4.0"

        # Check custom variables
        assert variables["custom_key"] == "custom_value"

    def test_get_variable(self) -> None:
        """Test get method to retrieve variable by name (lines 196-197)."""
        context = EvaluationContext()
        context.custom_vars["test_var"] = 123

        # Test getting existing variable (case insensitive)
        assert context.get("total_memory_gb") == 16
        assert context.get("TOTAL_MEMORY_GB") == 16
        assert context.get("custom_vars") is None  # Not directly accessible

        # Test getting custom variable (case insensitive)
        context.set("MyVar", 456)
        assert context.get("myvar") == 456
        assert context.get("MYVAR") == 456

    def test_get_with_default(self) -> None:
        """Test get method with default value."""
        context = EvaluationContext()
        assert context.get("nonexistent", "default_value") == "default_value"
        assert context.get("nonexistent") is None

    def test_set_variable(self) -> None:
        """Test set method to set custom variable (line 207)."""
        context = EvaluationContext()

        # Test setting a variable (should be lowercased)
        context.set("MyCustomVar", 100)
        assert context.custom_vars.get("mycustomvar") == 100
        assert "MyCustomVar" not in context.custom_vars

        # Test overwriting
        context.set("mycustomvar", 200)
        assert context.custom_vars["mycustomvar"] == 200

    def test_update_calculated_values(self) -> None:
        """Test update_calculated_values method."""
        context = EvaluationContext()

        # Update some values
        context.update_calculated_values(
            num_executors=10,
            executor_cores=8,
            executor_memory_gb=8.0,
            driver_memory_gb=2.0,
        )
        assert context.num_executors == 10
        assert context.executor_cores == 8
        assert context.executor_memory_gb == 8.0
        assert context.driver_memory_gb == 2.0

    def test_update_calculated_values_partial(self) -> None:
        """Test update_calculated_values with partial updates."""
        context = EvaluationContext()
        original_cores = context.executor_cores

        context.update_calculated_values(num_executors=5)
        assert context.num_executors == 5
        assert context.executor_cores == original_cores  # Should not change

    def test_is_streaming_true(self) -> None:
        """Test is_streaming method returns True (line 241)."""
        context = EvaluationContext()
        context.custom_vars["streaming"] = True
        assert context.is_streaming() is True

    def test_is_streaming_false(self) -> None:
        """Test is_streaming method returns False when not set."""
        context = EvaluationContext()
        assert context.is_streaming() is False

    def test_is_streaming_false_explicit(self) -> None:
        """Test is_streaming method returns False when explicitly set to False."""
        context = EvaluationContext()
        context.custom_vars["streaming"] = False
        assert context.is_streaming() is False

    def test_is_memory_intensive_true(self) -> None:
        """Test is_memory_intensive method returns True (line 250)."""
        context = EvaluationContext()
        context.custom_vars["memory_intensive"] = True
        assert context.is_memory_intensive() is True

    def test_is_memory_intensive_false(self) -> None:
        """Test is_memory_intensive method returns False when not set."""
        context = EvaluationContext()
        assert context.is_memory_intensive() is False

    def test_is_memory_intensive_false_explicit(self) -> None:
        """Test is_memory_intensive method returns False when explicitly set to False."""
        context = EvaluationContext()
        context.custom_vars["memory_intensive"] = False
        assert context.is_memory_intensive() is False

    def test_has_large_shuffles_true_from_custom_var(self) -> None:
        """Test has_large_shuffles returns True from custom var (line 259)."""
        context = EvaluationContext()
        context.custom_vars["large_shuffles"] = True
        assert context.has_large_shuffles() is True

    def test_has_large_shuffles_true_from_data_size(self) -> None:
        """Test has_large_shuffles returns True from data size > 100GB."""
        context = EvaluationContext()
        context.data_profile.size_gb = 150.0
        assert context.has_large_shuffles() is True

    def test_has_large_shuffles_false(self) -> None:
        """Test has_large_shuffles returns False."""
        context = EvaluationContext()
        context.data_profile.size_gb = 50.0
        assert context.has_large_shuffles() is False

    def test_has_large_shuffles_true_both_conditions(self) -> None:
        """Test has_large_shuffles returns True when both conditions are True."""
        context = EvaluationContext()
        context.custom_vars["large_shuffles"] = True
        context.data_profile.size_gb = 200.0
        assert context.has_large_shuffles() is True

    def test_has_large_shuffles_exactly_100gb(self) -> None:
        """Test has_large_shuffles with data size exactly 100GB (should be False)."""
        context = EvaluationContext()
        context.data_profile.size_gb = 100.0
        assert context.has_large_shuffles() is False

    def test_has_large_shuffles_over_100gb(self) -> None:
        """Test has_large_shuffles with data size just over 100GB."""
        context = EvaluationContext()
        context.data_profile.size_gb = 100.1
        assert context.has_large_shuffles() is True

    def test_repr(self) -> None:
        """Test string representation of EvaluationContext."""
        context = EvaluationContext()
        repr_str = repr(context)
        assert "EvaluationContext" in repr_str
        assert "platform=local" in repr_str
        assert "4c/16.0g" in repr_str or "4c/16" in repr_str
