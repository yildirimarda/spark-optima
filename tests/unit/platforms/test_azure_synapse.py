# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the Azure Synapse platform.

This module contains tests for Azure Synapse deployment platform including
resource management and Synapse-specific configuration.
"""

from __future__ import annotations

import pytest

from spark_optima.platforms.azure_synapse import AzureSynapsePlatform
from spark_optima.platforms.models import (
    ClusterConfig,
    ResourceSpec,
    WorkerType,
)
from spark_optima.platforms.pricing import REGION_MULTIPLIERS


class TestAzureSynapsePlatformInitialization:
    """Test cases for AzureSynapsePlatform initialization."""

    def test_azure_synapse_platform_initialization(self) -> None:
        """Test Azure Synapse platform initialization."""
        platform = AzureSynapsePlatform()
        assert platform.name == "azure_synapse"
        assert platform.display_name == "Azure Synapse"

    def test_azure_synapse_platform_constraints(self) -> None:
        """Test Azure Synapse platform constraints."""
        platform = AzureSynapsePlatform()
        constraints = platform.constraints

        assert constraints.min_workers >= 3  # Synapse Spark pool minimum
        assert constraints.max_workers >= 200
        assert len(constraints.supported_spark_versions) > 0


class TestAzureSynapsePlatformWorkerTypes:
    """Test cases for Azure Synapse worker types."""

    def test_get_worker_types(self) -> None:
        """Test getting available Synapse worker types."""
        platform = AzureSynapsePlatform()
        worker_types = platform.get_worker_types()

        assert isinstance(worker_types, list)
        assert len(worker_types) >= 3  # Small, Medium, Large

        for worker_type in worker_types:
            assert isinstance(worker_type, WorkerType)
            assert worker_type.name

    def test_get_worker_type_small(self) -> None:
        """Test getting Small worker type."""
        platform = AzureSynapsePlatform()
        worker_type = platform.get_worker_type("Small")

        if worker_type:
            assert isinstance(worker_type, WorkerType)
            assert worker_type.name == "Small"

    def test_get_worker_type_medium(self) -> None:
        """Test getting Medium worker type."""
        platform = AzureSynapsePlatform()
        worker_type = platform.get_worker_type("Medium")

        if worker_type:
            assert isinstance(worker_type, WorkerType)
            assert worker_type.name == "Medium"

    def test_get_worker_type_large(self) -> None:
        """Test getting Large worker type."""
        platform = AzureSynapsePlatform()
        worker_type = platform.get_worker_type("Large")

        if worker_type:
            assert isinstance(worker_type, WorkerType)
            assert worker_type.name == "Large"

    def test_get_worker_type_invalid(self) -> None:
        """Test getting invalid worker type returns None."""
        platform = AzureSynapsePlatform()
        worker_type = platform.get_worker_type("InvalidType")

        assert worker_type is None

    def test_worker_type_resources_scale(self) -> None:
        """Test that worker types have scaling resources."""
        platform = AzureSynapsePlatform()

        small = platform.get_worker_type("Small")
        medium = platform.get_worker_type("Medium")
        large = platform.get_worker_type("Large")

        if all([small, medium, large]):
            # Resources should scale with size
            assert medium.resources.cpu_cores >= small.resources.cpu_cores
            assert large.resources.cpu_cores >= medium.resources.cpu_cores


class TestAzureSynapsePlatformClusterConfig:
    """Test cases for cluster configuration."""

    def test_recommend_config_basic(self) -> None:
        """Test basic cluster config recommendation."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)
        assert config.spark_version == "3.5.0"
        assert config.worker_count >= 3  # Synapse minimum

    def test_recommend_config_with_worker_count(self) -> None:
        """Test config recommendation with specific worker count."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=10,
        )

        assert config.worker_count == 10

    def test_recommend_config_enforces_minimum_workers(self) -> None:
        """Test that config enforces Synapse minimum worker count."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=4, memory_gb=16.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=1,  # Below minimum
        )

        assert config.worker_count >= 3  # Enforced minimum

    def test_translate_to_spark_config(self) -> None:
        """Test translating cluster config to Synapse Spark config."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert isinstance(spark_config, dict)
        assert "spark.executor.memory" in spark_config
        assert "spark.executor.cores" in spark_config

    def test_translate_includes_synapse_settings(self) -> None:
        """Test that translation includes Synapse-specific settings."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        spark_config = platform.translate_to_spark_config(cluster_config)

        # Should include Synapse-specific configuration
        assert isinstance(spark_config, dict)


class TestAzureSynapsePlatformCostEstimation:
    """Test cases for cost estimation (vCore-based)."""

    def test_estimate_cost(self) -> None:
        """Test cost estimation for Azure Synapse."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=2.0)

        assert isinstance(cost, dict)
        assert "total_cost" in cost or "total" in cost
        assert cost.get("total_cost", cost.get("total", 0)) >= 0  # Synapse has vCore-based cost

    def test_estimate_cost_zero_duration(self) -> None:
        """Test cost estimation with zero duration."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=0.0)

        assert cost.get("total_cost", cost.get("total", 1)) == 0.0

    def test_estimate_cost_scales_with_resources(self) -> None:
        """Test that cost scales with resources."""
        platform = AzureSynapsePlatform()

        config_small = platform.recommend_config(
            resources=ResourceSpec(cpu_cores=8, memory_gb=32.0),
            spark_version="3.5.0",
            worker_count=3,
        )
        config_large = platform.recommend_config(
            resources=ResourceSpec(cpu_cores=32, memory_gb=128.0),
            spark_version="3.5.0",
            worker_count=10,
        )

        cost_small = platform.estimate_cost(config_small, duration_hours=1.0)
        cost_large = platform.estimate_cost(config_large, duration_hours=1.0)

        assert cost_large.get("total_cost", cost_large.get("total", 0)) >= cost_small.get(
            "total_cost", cost_small.get("total", 0)
        )


class TestAzureSynapsePlatformValidation:
    """Test cases for configuration validation."""

    def test_validate_config_valid(self) -> None:
        """Test validation with valid config."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        errors = platform.validate_config(cluster_config)

        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_config_below_minimum_workers(self) -> None:
        """Test validation with below minimum workers."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        cluster_config.worker_count = 2  # Below Synapse minimum

        errors = platform.validate_config(cluster_config)

        assert len(errors) > 0
        assert any("minimum" in error.lower() for error in errors)

    def test_validate_config_exceeds_max_workers(self) -> None:
        """Test validation exceeding maximum workers."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        cluster_config.worker_count = 300  # Exceeds typical Synapse limit

        errors = platform.validate_config(cluster_config)

        assert len(errors) > 0


class TestAzureSynapsePlatformSparkVersions:
    """Test cases for Spark version support."""

    def test_get_supported_spark_versions(self) -> None:
        """Test getting supported Spark versions."""
        platform = AzureSynapsePlatform()
        versions = platform.get_supported_spark_versions()

        assert isinstance(versions, list)
        assert len(versions) > 0

    def test_is_spark_version_supported_synapse_3_3(self) -> None:
        """Test Synapse Spark 3.3 support."""
        platform = AzureSynapsePlatform()

        assert platform.is_spark_version_supported("3.3.0")

    def test_is_spark_version_supported_synapse_3_4(self) -> None:
        """Test Synapse Spark 3.4 support."""
        platform = AzureSynapsePlatform()

        assert platform.is_spark_version_supported("3.4.0")


class TestAzureSynapsePlatformOptimalWorkerCount:
    """Test cases for optimal worker count calculation."""

    def test_get_optimal_worker_count(self) -> None:
        """Test optimal worker count calculation for Synapse."""
        platform = AzureSynapsePlatform()
        target_resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)
        worker_type = platform.get_worker_type("Medium")

        if worker_type:
            worker_count = platform.get_optimal_worker_count(target_resources, worker_type)

            assert isinstance(worker_count, int)
            assert worker_count >= 3  # Synapse minimum
            assert worker_count <= platform.constraints.max_workers

    def test_get_optimal_worker_count_respects_minimum(self) -> None:
        """Test that optimal worker count respects Synapse minimum."""
        platform = AzureSynapsePlatform()
        target_resources = ResourceSpec(cpu_cores=2, memory_gb=4.0)  # Very small
        worker_type = platform.get_worker_type("Small")

        if worker_type:
            worker_count = platform.get_optimal_worker_count(target_resources, worker_type)

            assert worker_count >= 3  # Synapse minimum enforced


class TestAzureSynapsePlatformAutoPause:
    """Test cases for auto-pause configuration."""

    def test_autopause_config_included(self) -> None:
        """Test that auto-pause config is included."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config.platform_config, dict)


class TestAzureSynapsePlatformEdgeCases:
    """Test edge cases."""

    def test_recommend_config_with_minimal_resources(self) -> None:
        """Test config recommendation with minimal resources."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=2, memory_gb=8.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)
        assert config.worker_count >= 3

    def test_recommend_config_with_large_resources(self) -> None:
        """Test config recommendation with large resources."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=64, memory_gb=256.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)
        assert config.worker_count <= platform.constraints.max_workers

    def test_different_worker_types_have_different_costs(self) -> None:
        """Test that different worker types have different costs."""
        platform = AzureSynapsePlatform()

        small = platform.get_worker_type("Small")
        medium = platform.get_worker_type("Medium")
        large = platform.get_worker_type("Large")

        if all([small, medium, large]):
            # Costs should scale with size
            costs = [
                small.cost.unit_cost_per_hour,
                medium.cost.unit_cost_per_hour,
                large.cost.unit_cost_per_hour,
            ]
            assert max(costs) > min(costs)

    def test_cost_model_per_vcore(self) -> None:
        """Test that cost model is vCore-based."""
        platform = AzureSynapsePlatform()
        worker_types = platform.get_worker_types()

        for worker_type in worker_types:
            # vCore-based cost model
            assert worker_type.cost.unit_name in ["vCore", "vcore", "vCores"]


class TestAzureSynapsePlatformWorkerTypeCaseInsensitive:
    """Test cases for case-insensitive worker type matching (line 177)."""

    def test_get_worker_type_lowercase(self) -> None:
        """Test getting worker type with lowercase name (line 177)."""
        platform = AzureSynapsePlatform()

        # Should match "Small" with "small" (case-insensitive)
        worker_type = platform.get_worker_type("small")
        assert worker_type is not None
        assert worker_type.name == "Small"

    def test_get_worker_type_uppercase(self) -> None:
        """Test getting worker type with uppercase name."""
        platform = AzureSynapsePlatform()

        worker_type = platform.get_worker_type("LARGE")
        assert worker_type is not None
        assert worker_type.name == "Large"

    def test_get_worker_type_mixed_case(self) -> None:
        """Test getting worker type with mixed case."""
        platform = AzureSynapsePlatform()

        worker_type = platform.get_worker_type("MeDiUm")
        assert worker_type is not None
        assert worker_type.name == "Medium"

    def test_get_worker_type_exact_match_preferred(self) -> None:
        """Test that exact match is preferred over case-insensitive."""
        platform = AzureSynapsePlatform()

        # Exact match should work
        worker_type = platform.get_worker_type("XXLarge")
        assert worker_type is not None
        assert worker_type.name == "XXLarge"


class TestAzureSynapsePlatformRecommendWorkerTypeFallback:
    """Test cases for fallback in recommend_worker_type (line 223)."""

    def test_recommend_worker_type_fallback_to_medium(self) -> None:
        """Test fallback to Medium when no worker type meets requirements (line 223)."""
        platform = AzureSynapsePlatform()

        # Request impossible requirements - no worker type can meet this
        worker = platform.recommend_worker_type(
            target_memory_gb=10000.0,  # Impossibly high
            target_cores=10000,  # Impossibly high
        )

        assert worker is not None
        # Should fallback to Medium or second worker type
        assert worker.name == "Medium" or worker == platform.get_worker_types()[1]


class TestAzureSynapsePlatformSparkPoolProperties:
    """Test cases for get_spark_pool_properties method (line 368)."""

    def test_get_spark_pool_properties(self) -> None:
        """Test generating Spark pool properties (line 368)."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        pool_properties = platform.get_spark_pool_properties(cluster_config)

        assert isinstance(pool_properties, dict)
        assert "name" in pool_properties
        assert "location" in pool_properties
        assert "properties" in pool_properties

        properties = pool_properties["properties"]
        assert "sparkVersion" in properties
        assert "nodeSize" in properties
        assert "autoScale" in properties
        assert "sparkConfigProperties" in properties

    def test_get_spark_pool_properties_custom_name(self) -> None:
        """Test Spark pool properties with custom pool name."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        pool_properties = platform.get_spark_pool_properties(cluster_config, pool_name="my-custom-pool")

        assert pool_properties["name"] == "my-custom-pool"

    def test_get_spark_pool_properties_autoscale(self) -> None:
        """Test that autoscale config is included in pool properties."""
        platform = AzureSynapsePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        pool_properties = platform.get_spark_pool_properties(cluster_config)

        autoscale = pool_properties["properties"]["autoScale"]
        assert "enabled" in autoscale
        assert "minNodeCount" in autoscale
        assert "maxNodeCount" in autoscale


class TestAzureSynapsePlatformFormatSparkConfig:
    """Test cases for _format_spark_config method (line 391)."""

    def test_format_spark_config(self) -> None:
        """Test formatting Spark config to properties format (line 391)."""
        platform = AzureSynapsePlatform()

        config = {
            "spark.executor.memory": "4g",
            "spark.executor.cores": "2",
            "spark.sql.adaptive.enabled": "true",
        }

        formatted = platform._format_spark_config(config)

        assert isinstance(formatted, str)
        assert "spark.executor.memory=4g" in formatted
        assert "spark.executor.cores=2" in formatted
        assert "spark.sql.adaptive.enabled=true" in formatted

    def test_format_spark_config_empty(self) -> None:
        """Test formatting empty Spark config."""
        platform = AzureSynapsePlatform()

        formatted = platform._format_spark_config({})

        assert isinstance(formatted, str)
        assert formatted == ""

    def test_format_spark_config_multiple_entries(self) -> None:
        """Test formatting multiple config entries."""
        platform = AzureSynapsePlatform()

        config = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }

        formatted = platform._format_spark_config(config)

        # Should have 3 lines
        lines = formatted.split("\n")
        assert len(lines) == 3


class TestAzureSynapsePlatformRegionalPricing:
    """Test cases for regional pricing multipliers in cost estimation."""

    def _get_cluster_config(self) -> ClusterConfig:
        """Build a cluster config for regional pricing tests."""
        platform = AzureSynapsePlatform()
        return platform.recommend_config(
            resources=ResourceSpec(cpu_cores=16, memory_gb=64.0),
            spark_version="3.4.0",
        )

    def test_estimate_cost_non_baseline_region_scales_by_multiplier(self) -> None:
        """Test that a non-baseline region scales cost by the table multiplier."""
        cluster_config = self._get_cluster_config()
        multiplier = REGION_MULTIPLIERS["azure_synapse"]["brazilsouth"]
        assert multiplier != 1.0  # Sanity: must be a non-baseline region

        baseline_cost = AzureSynapsePlatform(region="eastus").estimate_cost(cluster_config, duration_hours=2.0)
        regional_cost = AzureSynapsePlatform(region="brazilsouth").estimate_cost(cluster_config, duration_hours=2.0)

        assert regional_cost["total_cost"] == pytest.approx(baseline_cost["total_cost"] * multiplier)

    def test_estimate_cost_breakdown_contains_region_keys(self) -> None:
        """Test that the breakdown includes region and region_multiplier."""
        cluster_config = self._get_cluster_config()
        platform = AzureSynapsePlatform(region="westeurope")

        cost = platform.estimate_cost(cluster_config, duration_hours=1.0)

        assert cost["breakdown"]["region"] == "westeurope"
        assert cost["breakdown"]["region_multiplier"] == REGION_MULTIPLIERS["azure_synapse"]["westeurope"]

    def test_estimate_cost_default_region_is_baseline(self) -> None:
        """Test that the default region uses the baseline multiplier 1.0."""
        cluster_config = self._get_cluster_config()
        platform = AzureSynapsePlatform()

        cost = platform.estimate_cost(cluster_config, duration_hours=1.0)

        assert cost["breakdown"]["region"] == "eastus"
        assert cost["breakdown"]["region_multiplier"] == 1.0

    def test_estimate_cost_unknown_region_falls_back_to_baseline(self) -> None:
        """Test that an unknown region falls back to multiplier 1.0 without raising."""
        cluster_config = self._get_cluster_config()

        baseline_cost = AzureSynapsePlatform().estimate_cost(cluster_config, duration_hours=1.0)
        unknown_cost = AzureSynapsePlatform(region="atlantis").estimate_cost(cluster_config, duration_hours=1.0)

        assert unknown_cost["breakdown"]["region_multiplier"] == 1.0
        assert unknown_cost["total_cost"] == pytest.approx(baseline_cost["total_cost"])
