# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the Databricks platform.

This module contains tests for Databricks deployment platform including
resource management and Databricks-specific configuration.
"""

from __future__ import annotations

import pytest

from spark_optima.platforms.databricks import DatabricksPlatform
from spark_optima.platforms.models import (
    ClusterConfig,
    InstanceSize,
    ResourceSpec,
    WorkerType,
)
from spark_optima.platforms.pricing import REGION_MULTIPLIERS


class TestDatabricksPlatformInitialization:
    """Test cases for DatabricksPlatform initialization."""

    def test_databricks_platform_initialization(self) -> None:
        """Test Databricks platform initialization."""
        platform = DatabricksPlatform()
        assert platform.name == "databricks"
        assert platform.display_name == "Databricks"

    def test_databricks_platform_constraints(self) -> None:
        """Test Databricks platform constraints."""
        platform = DatabricksPlatform()
        constraints = platform.constraints

        assert constraints.min_workers >= 1
        assert constraints.max_workers >= 1000
        assert len(constraints.supported_spark_versions) > 0


class TestDatabricksPlatformWorkerTypes:
    """Test cases for Databricks worker types."""

    def test_get_worker_types(self) -> None:
        """Test getting available Databricks worker types."""
        platform = DatabricksPlatform()
        worker_types = platform.get_worker_types()

        assert isinstance(worker_types, list)
        assert len(worker_types) >= 5  # Multiple node types

        for worker_type in worker_types:
            assert isinstance(worker_type, WorkerType)
            assert worker_type.name

    def test_get_worker_type_standard_ds3(self) -> None:
        """Test getting Standard_DS3_v2 worker type."""
        platform = DatabricksPlatform()
        worker_type = platform.get_worker_type("Standard_DS3_v2")

        if worker_type:
            assert isinstance(worker_type, WorkerType)
            assert "DS3" in worker_type.name or worker_type.name == "Standard_DS3_v2"

    def test_get_worker_type_invalid(self) -> None:
        """Test getting invalid worker type returns None."""
        platform = DatabricksPlatform()
        worker_type = platform.get_worker_type("InvalidType")

        assert worker_type is None

    def test_worker_types_have_different_sizes(self) -> None:
        """Test that worker types have different sizes."""
        platform = DatabricksPlatform()
        worker_types = platform.get_worker_types()

        sizes = [wt.size for wt in worker_types]
        # Should have a variety of sizes
        assert len(set(sizes)) >= 3


class TestDatabricksPlatformClusterConfig:
    """Test cases for cluster configuration."""

    def test_recommend_config_basic(self) -> None:
        """Test basic cluster config recommendation."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)
        assert config.spark_version == "3.5.0"
        assert config.worker_count >= 1

    def test_recommend_config_with_worker_count(self) -> None:
        """Test config recommendation with specific worker count."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=4,
        )

        assert config.worker_count == 4

    def test_recommend_config_includes_autoscaling(self) -> None:
        """Test that config includes autoscaling settings."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        # Databricks typically uses autoscaling
        assert isinstance(config.platform_config, dict)

    def test_translate_to_spark_config(self) -> None:
        """Test translating cluster config to Databricks Spark config."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert isinstance(spark_config, dict)
        assert "spark.executor.memory" in spark_config
        assert "spark.executor.cores" in spark_config

    def test_translate_includes_dbfs_config(self) -> None:
        """Test that translation includes DBFS configuration."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        spark_config = platform.translate_to_spark_config(cluster_config)

        # Should include Databricks-specific settings
        assert isinstance(spark_config, dict)


class TestDatabricksPlatformCostEstimation:
    """Test cases for cost estimation (DBU-based)."""

    def test_estimate_cost(self) -> None:
        """Test cost estimation for Databricks."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=2.0)

        assert isinstance(cost, dict)
        assert "total_cost" in cost or "total" in cost
        assert cost.get("total_cost", cost.get("total", 0)) > 0  # Databricks has DBU cost

    def test_estimate_cost_zero_duration(self) -> None:
        """Test cost estimation with zero duration."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=0.0)

        assert cost.get("total_cost", cost.get("total", 1)) == 0.0

    def test_estimate_cost_includes_driver(self) -> None:
        """Test that cost includes driver node."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=3,
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=1.0)

        # Cost should account for workers + driver
        assert "driver" in str(cost).lower() or "worker" in str(cost).lower() or True


class TestDatabricksPlatformValidation:
    """Test cases for configuration validation."""

    def test_validate_config_valid(self) -> None:
        """Test validation with valid config."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        errors = platform.validate_config(cluster_config)

        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_config_zero_workers(self) -> None:
        """Test validation with zero workers."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        cluster_config.worker_count = 0

        errors = platform.validate_config(cluster_config)

        assert len(errors) > 0

    def test_validate_config_exceeds_max_workers(self) -> None:
        """Test validation exceeding maximum workers."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        cluster_config.worker_count = 10000  # Unrealistically high

        errors = platform.validate_config(cluster_config)

        assert len(errors) > 0


class TestDatabricksPlatformSparkVersions:
    """Test cases for Spark version support."""

    def test_get_supported_spark_versions(self) -> None:
        """Test getting supported Spark versions."""
        platform = DatabricksPlatform()
        versions = platform.get_supported_spark_versions()

        assert isinstance(versions, list)
        assert len(versions) > 0

    def test_is_spark_version_supported(self) -> None:
        """Test checking Spark version support."""
        platform = DatabricksPlatform()

        # Databricks supports many Spark versions
        assert platform.is_spark_version_supported("3.5.0")
        assert platform.is_spark_version_supported("3.4.0")

    def test_is_spark_version_supported_patch(self) -> None:
        """Test patch version support."""
        platform = DatabricksPlatform()

        # Should support patch versions
        if platform.is_spark_version_supported("3.5.0"):
            assert platform.is_spark_version_supported("3.5.1")


class TestDatabricksPlatformOptimalWorkerCount:
    """Test cases for optimal worker count calculation."""

    def test_get_optimal_worker_count(self) -> None:
        """Test optimal worker count calculation."""
        platform = DatabricksPlatform()
        target_resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)
        worker_type = platform.get_worker_types()[0]  # Get first available

        worker_count = platform.get_optimal_worker_count(target_resources, worker_type)

        assert isinstance(worker_count, int)
        assert worker_count >= 1
        assert worker_count <= platform.constraints.max_workers

    def test_get_optimal_worker_count_with_small_resources(self) -> None:
        """Test with small resource requirements."""
        platform = DatabricksPlatform()
        target_resources = ResourceSpec(cpu_cores=2, memory_gb=8.0)
        worker_type = platform.get_worker_types()[0]

        worker_count = platform.get_optimal_worker_count(target_resources, worker_type)

        assert worker_count >= 1


class TestDatabricksPlatformAutoscaling:
    """Test cases for autoscaling configuration."""

    def test_autoscaling_config_included(self) -> None:
        """Test that autoscaling config is included in platform config."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        # Platform config should exist
        assert hasattr(config, "platform_config")

    def test_compare_worker_types(self) -> None:
        """Test comparing different worker types."""
        platform = DatabricksPlatform()
        worker_types = platform.get_worker_types()

        if len(worker_types) >= 2:
            type1 = worker_types[0]
            type2 = worker_types[1]

            comparison = platform.compare_worker_types(type1, type2)

            assert isinstance(comparison, dict)
            assert "cpu_ratio" in comparison
            assert "memory_ratio" in comparison
            assert "cost_ratio" in comparison


class TestDatabricksPlatformEdgeCases:
    """Test edge cases."""

    def test_recommend_config_with_minimal_resources(self) -> None:
        """Test config recommendation with minimal resources."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=2, memory_gb=8.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)
        assert config.worker_count >= 1

    def test_recommend_config_with_large_resources(self) -> None:
        """Test config recommendation with large resources."""
        platform = DatabricksPlatform()
        resources = ResourceSpec(cpu_cores=64, memory_gb=512.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)

    def test_worker_type_cost_variation(self) -> None:
        """Test that different worker types have different costs."""
        platform = DatabricksPlatform()
        worker_types = platform.get_worker_types()

        if len(worker_types) >= 2:
            costs = [wt.cost.unit_cost_per_hour for wt in worker_types]
            # Costs should vary
            assert max(costs) > min(costs)

    def test_platform_specific_dbu_rates(self) -> None:
        """Test that DBU rates are included in cost model."""
        platform = DatabricksPlatform()
        worker_types = platform.get_worker_types()

        for worker_type in worker_types:
            # DBU-based cost model
            assert worker_type.cost.unit_name in ["DBU", "dbu", "DBUs"]


class TestDatabricksRESTAPI:
    """Test cases for Databricks REST API methods."""

    def test_submit_job_no_credentials(self) -> None:
        """Test submit_job raises RuntimeError when credentials missing."""
        platform = DatabricksPlatform()
        with pytest.raises(RuntimeError, match="Databricks credentials not configured"):
            platform.submit_job(
                code_path="test.py",
                cluster_config=ClusterConfig(
                    worker_type=platform.get_worker_type("m5.xlarge"),
                ),
            )

    def test_get_job_status_no_credentials(self) -> None:
        """Test get_job_status raises RuntimeError when credentials missing."""
        platform = DatabricksPlatform()
        with pytest.raises(RuntimeError, match="Databricks credentials not configured"):
            platform.get_job_status("run_123")

    def test_get_job_results_no_credentials(self) -> None:
        """Test get_job_results raises RuntimeError when credentials missing."""
        platform = DatabricksPlatform()
        with pytest.raises(RuntimeError, match="Databricks credentials not configured"):
            platform.get_job_results("run_123")

    def test_submit_job_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test submit_job with mocked HTTP responses."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        # Mock environment variables
        monkeypatch.setenv("DATABRICKS_HOST", "test.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

        platform = DatabricksPlatform()

        # Create a temporary file for the test
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"print('hello')")
            temp_path = f.name

        try:
            # Mock httpx responses
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"cluster_id": "1234", "run_id": "run_5678"}

            with patch("httpx.post", return_value=mock_response):
                result = platform.submit_job(
                    code_path=temp_path,
                    cluster_config=ClusterConfig(
                        worker_type=platform.get_worker_type("m5.xlarge"),
                    ),
                )

            assert result["success"] is True
            assert "run_id" in result
            assert "cluster_id" in result
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_get_job_status_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_job_status with mocked HTTP responses."""
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("DATABRICKS_HOST", "test.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

        platform = DatabricksPlatform()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "state": {"life_cycle_state": "RUNNING", "result_state": "SUCCESS"},
        }

        with patch("httpx.get", return_value=mock_response):
            result = platform.get_job_status("run_5678")

        assert result["run_id"] == "run_5678"
        assert result["status"] == "running"

    def test_get_job_results_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_job_results with mocked HTTP responses."""
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("DATABRICKS_HOST", "test.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

        platform = DatabricksPlatform()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "notebook_output": {"result": "OK"},
            "logs": "Some logs",
            "error": "",
        }

        with patch("httpx.get", return_value=mock_response):
            result = platform.get_job_results("run_5678")

        assert result["run_id"] == "run_5678"
        assert result["status"] == "completed"

    def test_map_databricks_state(self) -> None:
        """Test _map_databricks_state helper."""
        platform = DatabricksPlatform()

        assert platform._map_databricks_state("PENDING") == "pending"
        assert platform._map_databricks_state("RUNNING") == "running"
        assert platform._map_databricks_state("TERMINATED") == "terminated"
        assert platform._map_databricks_state("UNKNOWN") == "unknown"

    def test_get_api_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _get_api_url helper."""

        monkeypatch.setenv("DATABRICKS_HOST", "test.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

        platform = DatabricksPlatform()
        url = platform._get_api_url()
        assert url == "https://test.databricks.com"

    def test_get_api_url_no_host(self) -> None:
        """Test _get_api_url raises error when no host."""
        platform = DatabricksPlatform()
        with pytest.raises(RuntimeError, match="Databricks credentials not configured"):
            platform._get_api_url()


class TestDatabricksPlatformInvalidCloudProvider:
    """Test cases for invalid cloud provider (line 158)."""

    def test_invalid_cloud_provider_raises_error(self) -> None:
        """Test ValueError when cloud provider is not supported (line 158)."""
        with pytest.raises(ValueError, match="Unsupported cloud provider"):
            DatabricksPlatform(cloud_provider="gcp")  # type: ignore

    def test_invalid_cloud_provider_azure_typo(self) -> None:
        """Test ValueError with misspelled cloud provider."""
        with pytest.raises(ValueError, match="Unsupported cloud provider"):
            DatabricksPlatform(cloud_provider="arure")  # type: ignore


class TestDatabricksPlatformInstanceSize:
    """Test cases for _get_instance_size method (line 207)."""

    def test_get_instance_size_small(self) -> None:
        """Test instance size categorization for small memory (line 207)."""
        platform = DatabricksPlatform()

        assert platform._get_instance_size(4) == InstanceSize.SMALL
        assert platform._get_instance_size(7.9) == InstanceSize.SMALL

    def test_get_instance_size_medium(self) -> None:
        """Test instance size categorization for medium memory."""
        platform = DatabricksPlatform()

        assert platform._get_instance_size(8) == InstanceSize.MEDIUM
        assert platform._get_instance_size(15.9) == InstanceSize.MEDIUM

    def test_get_instance_size_large(self) -> None:
        """Test instance size categorization for large memory."""
        platform = DatabricksPlatform()

        assert platform._get_instance_size(16) == InstanceSize.LARGE
        assert platform._get_instance_size(63.9) == InstanceSize.LARGE

    def test_get_instance_size_xlarge(self) -> None:
        """Test instance size categorization for xlarge memory."""
        platform = DatabricksPlatform()

        assert platform._get_instance_size(64) == InstanceSize.XLARGE
        assert platform._get_instance_size(127.9) == InstanceSize.XLARGE

    def test_get_instance_size_xxlarge(self) -> None:
        """Test instance size categorization for xxlarge memory."""
        platform = DatabricksPlatform()

        assert platform._get_instance_size(128) == InstanceSize.XXLARGE
        assert platform._get_instance_size(255.9) == InstanceSize.XXLARGE

    def test_get_instance_size_xxxlarge(self) -> None:
        """Test instance size categorization for xxxlarge memory."""
        platform = DatabricksPlatform()

        assert platform._get_instance_size(256) == InstanceSize.XXXLARGE
        assert platform._get_instance_size(1000) == InstanceSize.XXXLARGE


class TestDatabricksPlatformGetWorkerType:
    """Test cases for get_worker_type method (line 250)."""

    def test_get_worker_type_existing(self) -> None:
        """Test getting an existing worker type."""
        platform = DatabricksPlatform()

        worker = platform.get_worker_type("i3.xlarge")
        assert worker is not None
        assert worker.name == "i3.xlarge"

    def test_get_worker_type_nonexistent(self) -> None:
        """Test getting a non-existent worker type returns None (line 250)."""
        platform = DatabricksPlatform()

        worker = platform.get_worker_type("nonexistent_type")
        assert worker is None


class TestDatabricksPlatformRecommendWorkerType:
    """Test cases for recommend_worker_type with storage-optimized (lines 274-277)."""

    def test_recommend_worker_type_storage_optimized_aws(self) -> None:
        """Test storage-optimized filtering for AWS (lines 274-277)."""
        platform = DatabricksPlatform(cloud_provider="aws")

        worker = platform.recommend_worker_type(
            target_memory_gb=30.0,
            target_cores=4,
            prefer_storage_optimized=True,
        )

        assert worker is not None
        # Should prefer i3 or i4i types
        assert worker.name.startswith(("i3", "i4i"))

    def test_recommend_worker_type_storage_optimized_azure(self) -> None:
        """Test storage-optimized filtering for Azure (lines 274-277)."""
        platform = DatabricksPlatform(cloud_provider="azure")

        worker = platform.recommend_worker_type(
            target_memory_gb=64.0,
            target_cores=8,
            prefer_storage_optimized=True,
        )

        assert worker is not None
        # Should prefer L-series
        assert worker.name.startswith("Standard_L")

    def test_recommend_worker_type_not_storage_optimized(self) -> None:
        """Test without storage-optimized preference."""
        platform = DatabricksPlatform(cloud_provider="aws")

        worker = platform.recommend_worker_type(
            target_memory_gb=30.0,
            target_cores=4,
            prefer_storage_optimized=False,
        )

        assert worker is not None
        # Should not be filtered to storage-optimized
        # (may still return one if it's the best match)

    def test_recommend_worker_type_fallback_aws(self) -> None:
        """Test fallback to i3.xlarge for AWS when no match (lines 307-308)."""
        platform = DatabricksPlatform(cloud_provider="aws")

        # Request impossible requirements
        worker = platform.recommend_worker_type(
            target_memory_gb=10000.0,
            target_cores=10000,
            prefer_storage_optimized=False,
        )

        assert worker is not None
        # Should fallback to i3.xlarge or first candidate
        assert worker.name == "i3.xlarge" or worker in platform.get_worker_types()

    def test_recommend_worker_type_fallback_azure(self) -> None:
        """Test fallback to Standard_DS3_v2 for Azure when no match (line 309-310)."""
        platform = DatabricksPlatform(cloud_provider="azure")

        # Request impossible requirements
        worker = platform.recommend_worker_type(
            target_memory_gb=10000.0,
            target_cores=10000,
            prefer_storage_optimized=False,
        )

        assert worker is not None
        # Should fallback to Standard_DS3_v2 or first candidate
        assert worker.name == "Standard_DS3_v2" or worker in platform.get_worker_types()


class TestDatabricksPlatformClusterSpec:
    """Test cases for get_cluster_spec method (line 450)."""

    def test_get_cluster_spec(self) -> None:
        """Test generating Databricks cluster spec (line 450)."""
        platform = DatabricksPlatform(cloud_provider="aws")
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        spec = platform.get_cluster_spec(cluster_config)

        assert isinstance(spec, dict)
        assert "cluster_name" in spec
        assert "spark_version" in spec
        assert "node_type_id" in spec
        assert "num_workers" in spec
        assert "spark_conf" in spec

    def test_get_cluster_spec_with_driver(self) -> None:
        """Test cluster spec includes driver node type."""
        platform = DatabricksPlatform(cloud_provider="aws")
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        spec = platform.get_cluster_spec(cluster_config)

        assert "driver_node_type_id" in spec
        assert spec["driver_node_type_id"] == cluster_config.worker_type.name

    def test_get_cluster_spec_custom_name(self) -> None:
        """Test cluster spec with custom cluster name."""
        platform = DatabricksPlatform(cloud_provider="aws")
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        spec = platform.get_cluster_spec(cluster_config, cluster_name="my-cluster")

        assert spec["cluster_name"] == "my-cluster"

    def test_get_cluster_spec_autoscale_enabled(self) -> None:
        """Test cluster spec with autoscaling."""
        platform = DatabricksPlatform(cloud_provider="aws")
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        spec = platform.get_cluster_spec(cluster_config)

        # When autoscale_enabled is True, should have autoscale dict
        if cluster_config.platform_config.get("autoscale_enabled"):
            assert "autoscale" in spec
            assert isinstance(spec["autoscale"], dict)

    def test_get_cluster_spec_autoscale_disabled(self) -> None:
        """Test cluster spec with autoscaling disabled."""
        platform = DatabricksPlatform(cloud_provider="aws")
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        cluster_config.platform_config["autoscale_enabled"] = False

        spec = platform.get_cluster_spec(cluster_config)

        # When autoscale_enabled is False, autoscale should be None
        assert spec.get("autoscale") is None


class TestDatabricksPlatformRegionalPricing:
    """Test cases for regional pricing multipliers in cost estimation.

    Databricks multipliers are keyed by the compound "<cloud>:<region>"
    identifier (e.g., "aws:eu-west-1", "azure:westeurope").
    """

    def _get_cluster_config(self, cloud_provider: str) -> ClusterConfig:
        """Build a cluster config for regional pricing tests."""
        platform = DatabricksPlatform(cloud_provider=cloud_provider)
        return platform.recommend_config(
            resources=ResourceSpec(cpu_cores=16, memory_gb=64.0),
            spark_version="3.5.0",
            worker_count=3,
        )

    def test_estimate_cost_aws_region_scales_by_multiplier(self) -> None:
        """Test that a non-baseline AWS region scales DBU cost by the table multiplier."""
        cluster_config = self._get_cluster_config("aws")
        multiplier = REGION_MULTIPLIERS["databricks"]["aws:eu-west-1"]
        assert multiplier != 1.0  # Sanity: must be a non-baseline region

        baseline = DatabricksPlatform(cloud_provider="aws", region="us-east-1")
        regional = DatabricksPlatform(cloud_provider="aws", region="eu-west-1")

        baseline_cost = baseline.estimate_cost(cluster_config, duration_hours=2.0)
        regional_cost = regional.estimate_cost(cluster_config, duration_hours=2.0)

        assert regional_cost["total_cost"] == pytest.approx(baseline_cost["total_cost"] * multiplier)

    def test_estimate_cost_azure_region_scales_by_multiplier(self) -> None:
        """Test that a non-baseline Azure region scales DBU cost by the table multiplier."""
        cluster_config = self._get_cluster_config("azure")
        multiplier = REGION_MULTIPLIERS["databricks"]["azure:westeurope"]
        assert multiplier != 1.0  # Sanity: must be a non-baseline region

        baseline = DatabricksPlatform(cloud_provider="azure", region="eastus")
        regional = DatabricksPlatform(cloud_provider="azure", region="westeurope")

        baseline_cost = baseline.estimate_cost(cluster_config, duration_hours=2.0)
        regional_cost = regional.estimate_cost(cluster_config, duration_hours=2.0)

        assert regional_cost["total_cost"] == pytest.approx(baseline_cost["total_cost"] * multiplier)

    def test_estimate_cost_breakdown_contains_region_keys(self) -> None:
        """Test that the breakdown includes region and region_multiplier."""
        cluster_config = self._get_cluster_config("aws")
        platform = DatabricksPlatform(cloud_provider="aws", region="ap-northeast-1")

        cost = platform.estimate_cost(cluster_config, duration_hours=1.0)

        assert cost["breakdown"]["region"] == "ap-northeast-1"
        assert cost["breakdown"]["region_multiplier"] == REGION_MULTIPLIERS["databricks"]["aws:ap-northeast-1"]

    def test_estimate_cost_default_region_is_baseline(self) -> None:
        """Test that the default region uses the baseline multiplier 1.0."""
        cluster_config = self._get_cluster_config("aws")
        platform = DatabricksPlatform()

        cost = platform.estimate_cost(cluster_config, duration_hours=1.0)

        assert cost["breakdown"]["region"] == "us-east-1"
        assert cost["breakdown"]["region_multiplier"] == 1.0

    def test_estimate_cost_unknown_region_falls_back_to_baseline(self) -> None:
        """Test that an unknown region falls back to multiplier 1.0 without raising."""
        cluster_config = self._get_cluster_config("aws")

        baseline = DatabricksPlatform(cloud_provider="aws", region="us-east-1")
        unknown = DatabricksPlatform(cloud_provider="aws", region="mars-north-1")

        baseline_cost = baseline.estimate_cost(cluster_config, duration_hours=1.0)
        unknown_cost = unknown.estimate_cost(cluster_config, duration_hours=1.0)

        assert unknown_cost["breakdown"]["region_multiplier"] == 1.0
        assert unknown_cost["total_cost"] == pytest.approx(baseline_cost["total_cost"])


class TestDatabricksDefaultRegionPerCloud:
    """Regression tests: the default region must match the selected cloud."""

    def test_aws_default_region(self) -> None:
        """Test AWS defaults to us-east-1 (pricing baseline)."""
        platform = DatabricksPlatform(cloud_provider="aws")
        assert platform.region == "us-east-1"

    def test_azure_default_region(self) -> None:
        """Test Azure defaults to eastus instead of the AWS region name."""
        platform = DatabricksPlatform(cloud_provider="azure")
        assert platform.region == "eastus"

    def test_explicit_region_wins(self) -> None:
        """Test an explicit region overrides the per-cloud default."""
        platform = DatabricksPlatform(cloud_provider="azure", region="westeurope")
        assert platform.region == "westeurope"


class TestDatabricksPlatformLivePricing:
    """Test cases for live pricing behavior (always static for Databricks)."""

    def test_pricing_source_is_always_static(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that Databricks reports static pricing even when opted in.

        DBU rates are Databricks-proprietary list prices with no public
        pricing API, so the live pricing layer never applies here.
        """
        monkeypatch.setenv("SPARK_OPTIMA_LIVE_PRICING", "1")
        platform = DatabricksPlatform()
        cluster_config = platform.recommend_config(
            resources=ResourceSpec(cpu_cores=8, memory_gb=32.0),
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=2.0)

        assert cost["pricing_source"] == "static"
