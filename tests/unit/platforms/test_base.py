# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the Platform base class.

This module contains tests for the abstract base class and LocalPlatformBase
to achieve 100% coverage on base.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from spark_optima.platforms.base import Platform
from spark_optima.platforms.local import LocalPlatform
from spark_optima.platforms.models import (
    ClusterConfig,
    CostModel,
    InstanceSize,
    PlatformConstraints,
    ResourceSpec,
    WorkerType,
)


class TestPlatformStrMethod:
    """Test cases for Platform.__str__ method (line 298)."""

    def test_platform_str(self) -> None:
        """Test Platform __str__ method returns correct format (line 298)."""

        # Create a concrete subclass to test
        class ConcretePlatform(Platform):
            @property
            def constraints(self) -> PlatformConstraints:
                return PlatformConstraints()

            def get_worker_types(self) -> list[WorkerType]:
                return []

            def get_worker_type(self, name: str) -> WorkerType | None:
                return None

            def recommend_config(
                self,
                resources: ResourceSpec,
                spark_version: str,
                worker_count: int | None = None,
            ) -> ClusterConfig:
                raise NotImplementedError

            def translate_to_spark_config(
                self,
                cluster_config: ClusterConfig,
            ) -> dict[str, Any]:
                raise NotImplementedError

            def estimate_cost(
                self,
                cluster_config: ClusterConfig,
                duration_hours: float,
            ) -> dict[str, Any]:
                raise NotImplementedError

        platform = ConcretePlatform(
            name="test_platform",
            display_name="Test Platform",
            description="A test platform",
        )

        result = str(platform)
        assert result == "Test Platform (test_platform)"

    def test_platform_repr(self) -> None:
        """Test Platform __repr__ method."""

        class ConcretePlatform(Platform):
            @property
            def constraints(self) -> PlatformConstraints:
                return PlatformConstraints()

            def get_worker_types(self) -> list[WorkerType]:
                return []

            def get_worker_type(self, name: str) -> WorkerType | None:
                return None

            def recommend_config(
                self,
                resources: ResourceSpec,
                spark_version: str,
                worker_count: int | None = None,
            ) -> ClusterConfig:
                raise NotImplementedError

            def translate_to_spark_config(
                self,
                cluster_config: ClusterConfig,
            ) -> dict[str, Any]:
                raise NotImplementedError

            def estimate_cost(
                self,
                cluster_config: ClusterConfig,
                duration_hours: float,
            ) -> dict[str, Any]:
                raise NotImplementedError

        platform = ConcretePlatform(
            name="my_platform",
            display_name="My Platform",
        )

        result = repr(platform)
        assert result == "ConcretePlatform(name='my_platform')"


class TestLocalPlatformBaseGetUsableResources:
    """Test cases for LocalPlatformBase.get_usable_resources (line 351)."""

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_get_usable_resources_with_none(
        self,
        mock_disk: MagicMock,
        mock_memory: MagicMock,
        mock_cpu: MagicMock,
    ) -> None:
        """Test get_usable_resources with total_resources=None (line 351)."""
        # Mock system resources
        mock_cpu.return_value = 8
        mock_memory.return_value = MagicMock(total=16 * 1024**3)  # 16 GB
        mock_disk.return_value = MagicMock(total=100 * 1024**3)  # 100 GB

        platform = LocalPlatform()

        # Call with total_resources=None - should call detect_local_resources()
        usable = platform.get_usable_resources(total_resources=None)

        assert isinstance(usable, ResourceSpec)
        assert usable.cpu_cores == 6  # max(1, int(8 * 0.8)) = 6 with default 20% headroom
        assert usable.memory_gb == pytest.approx(12.8, rel=0.01)  # 80% of 16

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_get_usable_resources_detects_resources(
        self,
        mock_disk: MagicMock,
        mock_memory: MagicMock,
        mock_cpu: MagicMock,
    ) -> None:
        """Test that detect_local_resources is called when total_resources is None."""
        mock_cpu.return_value = 16
        mock_memory.return_value = MagicMock(total=32 * 1024**3)  # 32 GB
        mock_disk.return_value = MagicMock(total=200 * 1024**3)  # 200 GB

        platform = LocalPlatform()

        # Call with total_resources=None
        usable = platform.get_usable_resources(None)

        # Verify the resources were detected
        assert usable.cpu_cores == 12  # 75% of 16 with default 20% headroom = 12.8, rounded to 12
        assert usable.memory_gb == pytest.approx(25.6, rel=0.01)  # 80% of 32

    def test_get_usable_resources_with_provided_resources(self) -> None:
        """Test get_usable_resources with provided resources (not None)."""
        platform = LocalPlatform()

        total_resources = ResourceSpec(
            cpu_cores=10,
            memory_gb=20.0,
            disk_gb=100.0,
        )

        # Call with provided resources
        usable = platform.get_usable_resources(
            total_resources=total_resources,
            headroom_percent=20.0,
        )

        assert usable.cpu_cores == 8  # 80% of 10
        assert usable.memory_gb == 16.0  # 80% of 20


class TestPlatformCompareWorkerTypes:
    """Test cases for Platform.compare_worker_types method."""

    def test_compare_worker_types(self) -> None:
        """Test comparing two worker types."""

        class ConcretePlatform(Platform):
            @property
            def constraints(self) -> PlatformConstraints:
                return PlatformConstraints()

            def get_worker_types(self) -> list[WorkerType]:
                return []

            def get_worker_type(self, name: str) -> WorkerType | None:
                return None

            def recommend_config(
                self,
                resources: ResourceSpec,
                spark_version: str,
                worker_count: int | None = None,
            ) -> ClusterConfig:
                raise NotImplementedError

            def translate_to_spark_config(
                self,
                cluster_config: ClusterConfig,
            ) -> dict[str, Any]:
                raise NotImplementedError

            def estimate_cost(
                self,
                cluster_config: ClusterConfig,
                duration_hours: float,
            ) -> dict[str, Any]:
                raise NotImplementedError

        platform = ConcretePlatform(name="test", display_name="Test")

        resources1 = ResourceSpec(cpu_cores=4, memory_gb=16)
        resources2 = ResourceSpec(cpu_cores=8, memory_gb=32)

        # Make type1 more cost-effective (lower cost per core)
        # type1: $0.25 per core (1.0 / 4)
        # type2: $0.375 per core (3.0 / 8)
        cost1 = CostModel(unit_cost_per_hour=1.0)
        cost2 = CostModel(unit_cost_per_hour=3.0)

        type1 = WorkerType(
            name="small",
            size=InstanceSize.SMALL,
            resources=resources1,
            cost=cost1,
        )
        type2 = WorkerType(
            name="large",
            size=InstanceSize.LARGE,
            resources=resources2,
            cost=cost2,
        )

        comparison = platform.compare_worker_types(type1, type2)

        assert "cpu_ratio" in comparison
        assert "memory_ratio" in comparison
        assert "cost_ratio" in comparison
        assert comparison["more_cost_effective"] == "small"  # type1 is cheaper per core
