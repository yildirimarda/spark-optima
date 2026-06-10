# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the Local platform.

This module contains tests for local Spark deployment platform including
resource detection and configuration generation.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from spark_optima.platforms.local import LocalPlatform
from spark_optima.platforms.models import (
    ClusterConfig,
    ResourceSpec,
    WorkerType,
)


class TestLocalPlatformInitialization:
    """Test cases for LocalPlatform initialization."""

    def test_local_platform_initialization(self) -> None:
        """Test local platform initialization."""
        platform = LocalPlatform()
        assert platform.name == "local"
        assert platform.display_name == "Local"

    def test_local_platform_constraints(self) -> None:
        """Test local platform constraints."""
        platform = LocalPlatform()
        constraints = platform.constraints

        assert constraints.min_workers == 1
        assert constraints.max_workers >= 1
        assert len(constraints.supported_spark_versions) > 0


class TestLocalPlatformResourceDetection:
    """Test cases for resource detection."""

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_detect_local_resources(
        self,
        mock_disk: MagicMock,
        mock_memory: MagicMock,
        mock_cpu: MagicMock,
    ) -> None:
        """Test detecting local system resources."""
        # Mock system resources
        mock_cpu.return_value = 8
        mock_memory.return_value = MagicMock(total=16 * 1024**3)  # 16 GB
        mock_disk.return_value = MagicMock(total=100 * 1024**3)  # 100 GB

        platform = LocalPlatform()
        resources = platform.detect_local_resources()

        assert isinstance(resources, ResourceSpec)
        assert resources.cpu_cores == 8
        assert resources.memory_gb == 16.0
        assert resources.disk_gb == 100.0

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_detect_local_resources_with_none_cpu(
        self,
        mock_memory: MagicMock,
        mock_cpu: MagicMock,
    ) -> None:
        """Test detection when cpu_count returns None."""
        mock_cpu.return_value = None
        mock_memory.return_value = MagicMock(total=8 * 1024**3)

        with patch("psutil.disk_usage") as mock_disk:
            mock_disk.return_value = MagicMock(total=50 * 1024**3)
            platform = LocalPlatform()
            resources = platform.detect_local_resources()

        assert resources.cpu_cores == 4  # Default fallback

    def test_get_usable_resources(self) -> None:
        """Test calculating usable resources."""
        platform = LocalPlatform()
        total_resources = ResourceSpec(
            cpu_cores=8,
            memory_gb=16.0,
            disk_gb=100.0,
        )

        usable = platform.get_usable_resources(total_resources, headroom_percent=20.0)

        assert usable.cpu_cores < total_resources.cpu_cores
        assert usable.memory_gb < total_resources.memory_gb
        assert usable.disk_gb < total_resources.disk_gb

    def test_get_usable_resources_default_headroom(self) -> None:
        """Test usable resources with default headroom."""
        platform = LocalPlatform()
        total_resources = ResourceSpec(cpu_cores=8, memory_gb=16.0)

        usable = platform.get_usable_resources(total_resources)

        # Default headroom is 20%
        assert usable.memory_gb == pytest.approx(12.8, rel=0.01)  # 80% of 16


class TestLocalPlatformWorkerTypes:
    """Test cases for worker type management."""

    def test_get_worker_types(self) -> None:
        """Test getting available worker types."""
        platform = LocalPlatform()
        worker_types = platform.get_worker_types()

        assert isinstance(worker_types, list)
        assert len(worker_types) > 0

        for worker_type in worker_types:
            assert isinstance(worker_type, WorkerType)
            assert worker_type.name
            assert worker_type.resources

    def test_get_worker_type_valid(self) -> None:
        """Test getting specific worker type."""
        platform = LocalPlatform()
        worker_type = platform.get_worker_type("local")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.name == "local"

    def test_get_worker_type_invalid(self) -> None:
        """Test getting invalid worker type returns None."""
        platform = LocalPlatform()
        worker_type = platform.get_worker_type("nonexistent")

        assert worker_type is None


class TestLocalPlatformClusterConfig:
    """Test cases for cluster configuration."""

    def test_recommend_config_basic(self) -> None:
        """Test basic cluster config recommendation."""
        platform = LocalPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=16.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)
        assert config.spark_version == "3.5.0"
        assert config.worker_count >= 1

    def test_recommend_config_with_worker_count(self) -> None:
        """Test config recommendation with specific worker count."""
        platform = LocalPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=16.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=2,
        )

        assert config.worker_count == 2

    def test_translate_to_spark_config(self) -> None:
        """Test translating cluster config to Spark config."""
        platform = LocalPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=16.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert isinstance(spark_config, dict)
        assert "spark.executor.memory" in spark_config
        assert "spark.executor.cores" in spark_config


class TestLocalPlatformCostEstimation:
    """Test cases for cost estimation."""

    def test_estimate_cost(self) -> None:
        """Test cost estimation for local platform."""
        platform = LocalPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=16.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=2.0)

        assert isinstance(cost, dict)
        assert "total" in cost
        # Local platform should have zero or minimal cost
        assert cost["total"] >= 0

    def test_estimate_cost_zero_duration(self) -> None:
        """Test cost estimation with zero duration."""
        platform = LocalPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=16.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=0.0)

        assert cost["total"] == 0.0


class TestLocalPlatformValidation:
    """Test cases for configuration validation."""

    def test_validate_config_valid(self) -> None:
        """Test validation with valid config."""
        platform = LocalPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=16.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        errors = platform.validate_config(cluster_config)

        assert isinstance(errors, list)
        # Valid config should have no errors
        assert len(errors) == 0

    def test_validate_config_invalid_workers(self) -> None:
        """Test validation with invalid worker count."""
        platform = LocalPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=16.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        # Set invalid worker count
        cluster_config.worker_count = -1

        errors = platform.validate_config(cluster_config)

        assert len(errors) > 0

    def test_validate_config_unsupported_version(self) -> None:
        """Test validation with unsupported Spark version."""
        platform = LocalPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=16.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        # Set unsupported version
        cluster_config.spark_version = "9.9.9"

        errors = platform.validate_config(cluster_config)

        assert len(errors) > 0


class TestLocalPlatformSparkVersions:
    """Test cases for Spark version support."""

    def test_get_supported_spark_versions(self) -> None:
        """Test getting supported Spark versions."""
        platform = LocalPlatform()
        versions = platform.get_supported_spark_versions()

        assert isinstance(versions, list)
        assert len(versions) > 0

    def test_is_spark_version_supported_exact(self) -> None:
        """Test checking exact version support."""
        platform = LocalPlatform()

        # Should support at least 3.5.0
        assert platform.is_spark_version_supported("3.5.0")

    def test_is_spark_version_supported_patch(self) -> None:
        """Test checking patch version support."""
        platform = LocalPlatform()

        # Should support 3.5.1 if 3.5.0 is supported
        if platform.is_spark_version_supported("3.5.0"):
            assert platform.is_spark_version_supported("3.5.1")

    def test_is_spark_version_not_supported(self) -> None:
        """Test checking unsupported version."""
        platform = LocalPlatform()

        assert not platform.is_spark_version_supported("9.9.9")


class TestLocalPlatformOptimalWorkerCount:
    """Test cases for optimal worker count calculation."""

    def test_get_optimal_worker_count(self) -> None:
        """Test optimal worker count calculation."""
        platform = LocalPlatform()
        target_resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)
        worker_type = platform.get_worker_type("local")

        assert worker_type is not None
        worker_count = platform.get_optimal_worker_count(target_resources, worker_type)

        assert isinstance(worker_count, int)
        assert worker_count >= platform.constraints.min_workers
        assert worker_count <= platform.constraints.max_workers


class TestLocalPlatformEdgeCases:
    """Test edge cases."""

    def test_detect_resources_with_zero_values(self) -> None:
        """Test resource detection with zero/edge values."""
        with (
            patch("psutil.cpu_count") as mock_cpu,
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.disk_usage") as mock_disk,
        ):
            mock_cpu.return_value = 1
            mock_memory.return_value = MagicMock(total=1024**3)  # 1 GB
            mock_disk.return_value = MagicMock(total=10 * 1024**3)  # 10 GB

            platform = LocalPlatform()
            resources = platform.detect_local_resources()

            assert resources.cpu_cores == 1
            assert resources.memory_gb == 1.0

    def test_recommend_config_with_minimal_resources(self) -> None:
        """Test config recommendation with minimal resources."""
        platform = LocalPlatform()
        resources = ResourceSpec(cpu_cores=1, memory_gb=2.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)

    def test_recommend_config_with_large_resources(self) -> None:
        """Test config recommendation with large resources."""
        platform = LocalPlatform()
        resources = ResourceSpec(cpu_cores=64, memory_gb=256.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)

    def test_usable_resources_with_zero_headroom(self) -> None:
        """Test usable resources with zero headroom."""
        platform = LocalPlatform()
        total_resources = ResourceSpec(cpu_cores=8, memory_gb=16.0)

        usable = platform.get_usable_resources(total_resources, headroom_percent=0.0)

        # With 0% headroom, should return same resources (minus 1 core minimum)
        assert usable.cpu_cores <= total_resources.cpu_cores
        assert usable.memory_gb == total_resources.memory_gb


class TestLocalPlatformTranslateSparkConfig:
    """Test cases for translate_to_spark_config local_dir (line 208)."""

    def test_translate_includes_local_dir_from_config(self) -> None:
        """Test that local_dir from platform_config is used (line 208)."""
        platform = LocalPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=16.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        # Add local_dir to platform_config
        cluster_config.platform_config["local_dir"] = "/custom/spark-local"

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert spark_config["spark.local.dir"] == "/custom/spark-local"

    def test_translate_default_local_dir(self) -> None:
        """Test default local dir when not specified in platform_config."""
        platform = LocalPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=16.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        # Don't add local_dir to platform_config

        spark_config = platform.translate_to_spark_config(cluster_config)

        # Should use default
        assert spark_config["spark.local.dir"] == tempfile.gettempdir() + "/spark-local"

    def test_translate_platform_config_none(self) -> None:
        """Test translate when platform_config is empty."""
        platform = LocalPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=16.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        # Clear platform_config
        cluster_config.platform_config = {}

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert isinstance(spark_config, dict)
        assert "spark.local.dir" in spark_config


class TestLocalPlatformRecommendedConfig:
    """Test cases for get_recommended_local_config method (lines 269-278)."""

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_get_recommended_local_config_basic(
        self,
        mock_memory: MagicMock,
        mock_cpu: MagicMock,
    ) -> None:
        """Test get_recommended_local_config basic functionality (lines 269-278)."""
        mock_cpu.return_value = 8
        mock_memory.return_value = MagicMock(total=16 * 1024**3)  # 16 GB

        platform = LocalPlatform()

        config = platform.get_recommended_local_config(
            spark_version="3.5.0",
            memory_fraction=0.75,
        )

        assert isinstance(config, dict)
        assert "spark.master" in config
        assert "spark.driver.memory" in config
        assert "spark.serializer" in config
        assert config["spark.master"] == "local[8]"

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_get_recommended_local_config_memory_calculation(
        self,
        mock_memory: MagicMock,
        mock_cpu: MagicMock,
    ) -> None:
        """Test memory calculation in get_recommended_local_config."""
        mock_cpu.return_value = 8
        mock_memory.return_value = MagicMock(total=16 * 1024**3)  # 16 GB

        platform = LocalPlatform()

        config = platform.get_recommended_local_config(
            spark_version="3.5.0",
            memory_fraction=0.75,
        )

        # 16 GB * 0.75 = 12 GB usable, 90% = 10.8 GB driver memory
        expected_memory = max(1, int(16 * 0.75 * 0.9))
        assert config["spark.driver.memory"] == f"{expected_memory}g"

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_get_recommended_local_config_parallelism(
        self,
        mock_memory: MagicMock,
        mock_cpu: MagicMock,
    ) -> None:
        """Test parallelism settings in get_recommended_local_config."""
        mock_cpu.return_value = 8
        mock_memory.return_value = MagicMock(total=16 * 1024**3)

        platform = LocalPlatform()

        config = platform.get_recommended_local_config()

        # parallelism should be cpu_cores * 2 = 16
        assert config["spark.default.parallelism"] == "16"
        assert config["spark.sql.shuffle.partitions"] == str(max(200, 16))

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_get_recommended_local_config_adaptive_sql(
        self,
        mock_memory: MagicMock,
        mock_cpu: MagicMock,
    ) -> None:
        """Test AQE settings in get_recommended_local_config."""
        mock_cpu.return_value = 8
        mock_memory.return_value = MagicMock(total=16 * 1024**3)

        platform = LocalPlatform()

        config = platform.get_recommended_local_config()

        assert config["spark.sql.adaptive.enabled"] == "true"
        assert config["spark.sql.adaptive.coalescePartitions.enabled"] == "true"
        assert config["spark.sql.adaptive.skewJoin.enabled"] == "true"

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_get_recommended_local_config_ui_settings(
        self,
        mock_memory: MagicMock,
        mock_cpu: MagicMock,
    ) -> None:
        """Test UI settings in get_recommended_local_config."""
        mock_cpu.return_value = 8
        mock_memory.return_value = MagicMock(total=16 * 1024**3)

        platform = LocalPlatform()

        config = platform.get_recommended_local_config()

        assert config["spark.ui.enabled"] == "true"
        assert config["spark.ui.port"] == "4040"

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_get_recommended_local_config_custom_version(
        self,
        mock_memory: MagicMock,
        mock_cpu: MagicMock,
    ) -> None:
        """Test with custom Spark version."""
        mock_cpu.return_value = 4
        mock_memory.return_value = MagicMock(total=8 * 1024**3)

        platform = LocalPlatform()

        config = platform.get_recommended_local_config(spark_version="3.4.0")

        assert isinstance(config, dict)
        # Config should still be valid
        assert "spark.master" in config

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_get_recommended_local_config_memory_fraction(
        self,
        mock_memory: MagicMock,
        mock_cpu: MagicMock,
    ) -> None:
        """Test with different memory fractions."""
        mock_cpu.return_value = 8
        mock_memory.return_value = MagicMock(total=32 * 1024**3)  # 32 GB

        platform = LocalPlatform()

        # With 50% memory fraction
        config = platform.get_recommended_local_config(memory_fraction=0.5)

        # 32 * 0.5 * 0.9 = 14.4 -> 14 GB
        expected = max(1, int(32 * 0.5 * 0.9))
        assert config["spark.driver.memory"] == f"{expected}g"
