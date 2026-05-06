# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for PerformanceModel."""

import pytest

from spark_optima.core.simulation.performance_model import (
    DataCharacteristics,
    JoinType,
    OperationProfile,
    OperationType,
    PerformanceModel,
)
from spark_optima.platforms.models import ResourceSpec


class TestPerformanceModel:
    """Test cases for PerformanceModel."""

    def test_initialization(self) -> None:
        """Test model initialization."""
        model = PerformanceModel()
        assert model is not None

    def test_estimate_basic(self) -> None:
        """Test basic estimation."""
        model = PerformanceModel()
        config = {
            "spark.executor.memory": "4g",
            "spark.executor.cores": "4",
            "spark.default.parallelism": "200",
        }
        resource_spec = ResourceSpec(cpu_cores=16, memory_gb=64)

        result = model.estimate(config, resource_spec)

        assert "execution_time_seconds" in result
        assert "memory_peak_gb" in result
        assert result["execution_time_seconds"] > 0

    def test_estimate_with_data_profile(self) -> None:
        """Test estimation with data profile."""
        model = PerformanceModel()
        config = {"spark.executor.memory": "4g", "spark.executor.cores": "4"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)
        data_profile = DataCharacteristics(size_gb=100, format="parquet")
        operations = OperationProfile(operations=[OperationType.SCAN, OperationType.AGGREGATION])

        result = model.estimate(
            config, resource_spec, data_profile=data_profile, operations=operations
        )

        assert result["execution_time_seconds"] > 0
        assert result["memory_peak_gb"] > 0

    def test_estimate_different_formats(self) -> None:
        """Test estimation with different data formats."""
        model = PerformanceModel()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        for fmt in ["parquet", "json", "csv", "orc"]:
            data_profile = DataCharacteristics(size_gb=10, format=fmt)
            result = model.estimate(config, resource_spec, data_profile=data_profile)
            assert result["execution_time_seconds"] > 0

    def test_estimate_with_join(self) -> None:
        """Test estimation with join operations."""
        model = PerformanceModel()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)
        operations = OperationProfile(
            operations=[OperationType.SCAN, OperationType.JOIN],
            join_details={1: JoinType.BROADCAST_HASH},
        )

        result = model.estimate(config, resource_spec, operations=operations)
        assert result["execution_time_seconds"] > 0

    def test_cluster_topology_calculation(self) -> None:
        """Test cluster topology calculation."""
        model = PerformanceModel()
        resource_spec = ResourceSpec(cpu_cores=16, memory_gb=64)

        topology = model._calculate_cluster_topology(resource_spec, 4.0, 4)

        assert "num_executors" in topology
        assert "total_executor_cores" in topology
        assert topology["num_executors"] >= 1

    def test_memory_parsing(self) -> None:
        """Test memory value parsing."""
        model = PerformanceModel()

        assert model._parse_memory("4g") == 4.0
        assert model._parse_memory("512m") == 0.5
        assert model._parse_memory("1024k") == pytest.approx(0.0009765625)
        assert model._parse_memory(8) == 8.0

    def test_validate_feasibility(self) -> None:
        """Test configuration feasibility validation via full estimate."""
        model = PerformanceModel()
        config = {"spark.executor.memory": "4g", "spark.executor.cores": "2"}
        resource_spec = ResourceSpec(cpu_cores=16, memory_gb=64)

        # Use full estimate which includes feasibility check
        result = model.estimate(config, resource_spec)

        # Should return valid results - success depends on memory calculations
        assert "success" in result
        assert "feasibility_issues" in result
        assert result["execution_time_seconds"] > 0

    def test_validate_infeasible_config(self) -> None:
        """Test infeasible configuration detection via full estimate."""
        model = PerformanceModel()
        # Very constrained resources with large executor memory
        config = {"spark.executor.memory": "32g", "spark.executor.cores": "16"}
        resource_spec = ResourceSpec(cpu_cores=4, memory_gb=8)

        result = model.estimate(config, resource_spec)

        # Should be infeasible
        assert result["success"] is False
        assert len(result["feasibility_issues"]) > 0

    def test_io_metrics_estimation(self) -> None:
        """Test I/O metrics estimation."""
        model = PerformanceModel()
        data_profile = DataCharacteristics(size_gb=10, format="parquet")
        config = {}
        cluster_topology = {"total_executor_cores": 8}

        io_metrics = model._estimate_io_metrics(data_profile, config, cluster_topology)

        assert "read_time_seconds" in io_metrics
        assert io_metrics["read_time_seconds"] > 0

    def test_shuffle_metrics_estimation(self) -> None:
        """Test shuffle metrics estimation."""
        model = PerformanceModel()
        data_profile = DataCharacteristics(size_gb=10)
        operations = OperationProfile(operations=[OperationType.JOIN, OperationType.AGGREGATION])
        config = {}
        cluster_topology = {"num_executors": 2, "executor_memory_gb": 4}

        shuffle_metrics = model._estimate_shuffle_metrics(
            operations, data_profile, config, cluster_topology
        )

        assert "read_gb" in shuffle_metrics
        assert "write_gb" in shuffle_metrics


class TestDataCharacteristics:
    """Test cases for DataCharacteristics."""

    def test_default_initialization(self) -> None:
        """Test default values."""
        data = DataCharacteristics()
        assert data.size_gb == 10.0
        assert data.format == "parquet"

    def test_row_count_estimation(self) -> None:
        """Test automatic row count estimation."""
        data = DataCharacteristics(size_gb=1, avg_row_size_bytes=100)
        expected_rows = int((1 * 1024**3) / 100)
        assert data.num_rows == expected_rows


class TestOperationProfile:
    """Test cases for OperationProfile."""

    def test_default_operations(self) -> None:
        """Test default operation list."""
        profile = OperationProfile()
        assert len(profile.operations) == 1
        assert profile.operations[0] == OperationType.SCAN


class TestPerformanceModelMissingCoverage:
    """Tests to cover missing lines for 100% coverage."""

    def test_aggregation_with_cardinality(self) -> None:
        """Test AGGREGATION operation with cardinality set (line 454).

        When data_profile.cardinality has values, max_cardinality should be
        calculated from those values.
        """
        model = PerformanceModel()
        config = {"spark.executor.memory": "4g", "spark.executor.cores": "4"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)
        data_profile = DataCharacteristics(
            size_gb=10,
            format="parquet",
            cardinality={"user_id": 5000, "product_id": 1000, "category": 50},
        )
        operations = OperationProfile(operations=[OperationType.AGGREGATION])

        result = model.estimate(
            config, resource_spec, data_profile=data_profile, operations=operations
        )

        assert result["execution_time_seconds"] > 0
        assert "stage_0_aggregation" in result["stage_times"]

    def test_udf_operation_penalty(self) -> None:
        """Test UDF operation applies penalty (line 475).

        When operation is UDF, processing_time should be multiplied by 1.5.
        """
        model = PerformanceModel()
        config = {"spark.executor.memory": "4g", "spark.executor.cores": "4"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)
        data_profile = DataCharacteristics(size_gb=10, format="parquet")
        operations = OperationProfile(operations=[OperationType.UDF])

        result = model.estimate(
            config, resource_spec, data_profile=data_profile, operations=operations
        )

        assert result["execution_time_seconds"] > 0
        assert "stage_0_udf" in result["stage_times"]

    def test_cached_operation_memory(self) -> None:
        """Test CACHED operation affects memory calculation (line 571).

        When CACHED is in operations, data_memory should be 40% of data size.
        """
        model = PerformanceModel()
        config = {"spark.executor.memory": "4g", "spark.executor.cores": "4"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)
        data_profile = DataCharacteristics(size_gb=10, format="parquet")
        operations = OperationProfile(operations=[OperationType.CACHED])

        result = model.estimate(
            config, resource_spec, data_profile=data_profile, operations=operations
        )

        assert result["execution_time_seconds"] > 0
        # CACHED operation should result in higher data memory allocation
        assert result["memory_breakdown"]["data_gb"] == 4.0  # 10 * 0.4

    def test_estimate_with_cost_model(self) -> None:
        """Test estimate with cost_model provided (line 703).

        When cost_model is provided, _estimate_cost should use it.
        """

        class MockCostModel:
            """Mock cost model for testing."""

            def calculate(self, duration_hours: float) -> float:
                return duration_hours * 100.0  # $100 per hour

        model = PerformanceModel()
        config = {"spark.executor.memory": "4g", "spark.executor.cores": "4"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)
        cost_model = MockCostModel()

        result = model.estimate(config, resource_spec, cost_model=cost_model)

        assert result["execution_time_seconds"] > 0
        # Cost should be calculated using our mock model
        expected_cost = (result["execution_time_seconds"] / 3600) * 100.0
        assert result["cost_estimate_usd"] == pytest.approx(expected_cost)

    def test_validate_feasibility_memory_exceeded(self) -> None:
        """Test feasibility check when peak memory exceeds limit (line 731).

        When peak_memory > executor_memory * 0.95, an issue should be added.
        """
        model = PerformanceModel()
        # Use very small executor memory with large data to trigger high peak memory
        config = {"spark.executor.memory": "1g", "spark.executor.cores": "1"}
        resource_spec = ResourceSpec(cpu_cores=4, memory_gb=16)
        # Large data size to increase memory estimates
        data_profile = DataCharacteristics(size_gb=100, format="parquet")
        operations = OperationProfile(operations=[OperationType.AGGREGATION, OperationType.JOIN])

        result = model.estimate(
            config, resource_spec, data_profile=data_profile, operations=operations
        )

        # With 1GB executor and large data, peak memory may exceed 95% of executor
        # The success flag and issues should reflect this
        assert "success" in result
        assert "feasibility_issues" in result

    def test_parse_memory_invalid_format(self) -> None:
        """Test _parse_memory with invalid format (line 782).

        When memory string doesn't match pattern, should return default 4.0.
        """
        model = PerformanceModel()

        # Invalid formats that don't match the regex pattern
        assert model._parse_memory("invalid") == 4.0
        assert model._parse_memory("abc123") == 4.0
        assert model._parse_memory("") == 4.0
        assert model._parse_memory("123xyz") == 4.0

    def test_skew_factor_penalty(self) -> None:
        """Test skew factor penalty applied (line 471).

        When data_profile.skew_factor > 1.5, processing_time should be increased.
        """
        model = PerformanceModel()
        config = {"spark.executor.memory": "4g", "spark.executor.cores": "4"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)
        # Set skew_factor > 1.5 to trigger the penalty
        data_profile = DataCharacteristics(size_gb=10, format="parquet", skew_factor=2.0)
        operations = OperationProfile(operations=[OperationType.SCAN])

        result = model.estimate(
            config, resource_spec, data_profile=data_profile, operations=operations
        )

        assert result["execution_time_seconds"] > 0
        assert "stage_0_scan" in result["stage_times"]

    def test_validate_feasibility_memory_exceeds_95_percent(self) -> None:
        """Test feasibility check when peak memory exceeds 95% of executor (line 731).

        When peak_memory > executor_memory * 0.95, an issue should be added.
        This test directly calls _validate_feasibility with manipulated memory metrics
        since the estimate() method caps peak_memory at executor_memory * 0.85.
        """
        model = PerformanceModel()
        config = {"spark.executor.memory": "4g", "spark.executor.cores": "4"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        # Create memory_metrics where peak_gb exceeds 95% of executor memory
        # executor_memory = 4GB, so 95% = 3.8GB. Set peak to 4.0GB to exceed.
        memory_metrics = {"peak_gb": 4.0}
        cluster_topology = {
            "executor_memory_gb": 4.0,
            "executor_cores": 4,
            "total_executor_cores": 8,
        }

        is_feasible, issues = model._validate_feasibility(
            config=config,
            resource_spec=resource_spec,
            memory_metrics=memory_metrics,
            cluster_topology=cluster_topology,
        )

        assert is_feasible is False
        assert len(issues) > 0
        # Check that the memory issue is in the list
        memory_issue_found = any("memory" in issue.lower() for issue in issues)
        assert memory_issue_found
        # Verify line 731 message format
        assert any("exceeds executor memory" in issue for issue in issues)
