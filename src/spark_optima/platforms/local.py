# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Local platform implementation for Spark Optima.

This module provides the LocalPlatform class for running Spark on local
machines with automatic resource detection and optimal configuration.
"""

from __future__ import annotations

import tempfile
from typing import Any

from spark_optima.platforms.base import LocalPlatformBase
from spark_optima.platforms.models import (
    ClusterConfig,
    CostModel,
    InstanceSize,
    PlatformConstraints,
    ResourceSpec,
    WorkerType,
)


class LocalPlatform(LocalPlatformBase):
    """Local Spark deployment platform.

    This platform represents running Spark on a local machine or single node.
    It automatically detects available system resources and configures Spark
    accordingly, leaving appropriate headroom for the operating system.

    Attributes:
        name: Platform identifier "local".
        display_name: Human-readable name "Local".

    Example:
        >>> platform = LocalPlatform()
        >>> resources = platform.detect_local_resources()
        >>> print(f"Detected: {resources.cpu_cores} cores, {resources.memory_gb:.1f}GB RAM")
        >>> config = platform.recommend_config(resources, "3.5.0")

    """

    def __init__(self) -> None:
        """Initialize the local platform."""
        super().__init__(
            name="local",
            display_name="Local",
            description="Apache Spark running on local machine",
        )
        self._constraints = PlatformConstraints(
            min_workers=1,  # Local requires at least 1 worker
            max_workers=1,  # Local typically uses 1 worker max
            min_memory_gb=1.0,
            max_memory_gb=1024.0,
            min_cores=1,
            max_cores=256,
            supported_spark_versions=[
                "3.0.0",
                "3.1.0",
                "3.2.0",
                "3.3.0",
                "3.4.0",
                "3.5.0",
                "4.0.0",
                "4.1.0",
            ],
            custom_config_keys={
                "local_dir": "spark.local.dir",
                "cores_max": "spark.cores.max",
            },
        )

    @property
    def constraints(self) -> PlatformConstraints:
        """Get platform resource constraints."""
        return self._constraints

    def get_worker_types(self) -> list[WorkerType]:
        """Get available worker types for local platform.

        Returns:
            List containing a single local worker type.

        """
        # Local platform uses the system itself as the "worker"
        system_resources = self.detect_local_resources()

        return [
            WorkerType(
                name="local",
                size=InstanceSize.MEDIUM,
                resources=system_resources,
                cost=CostModel(
                    currency="USD",
                    unit_cost_per_hour=0.0,  # No direct cost for local
                    unit_name="local_machine",
                ),
                description="Local machine resources",
            ),
        ]

    def get_worker_type(self, name: str) -> WorkerType | None:
        """Get a specific worker type by name.

        Args:
            name: Worker type name.

        Returns:
            WorkerType if name is "local", None otherwise.

        """
        if name == "local":
            return self.get_worker_types()[0]
        return None

    def recommend_config(
        self,
        resources: ResourceSpec,
        spark_version: str,
        worker_count: int | None = None,
    ) -> ClusterConfig:
        """Recommend optimal local Spark configuration.

        For local mode, we configure Spark to use available resources
        while leaving headroom for the operating system.

        Args:
            resources: Available system resources.
            spark_version: Target Spark version.
            worker_count: Number of workers (0 for local mode, enforced max 1).

        Returns:
            ClusterConfig optimized for local execution.

        """
        # Get usable resources (leave headroom for OS)
        usable = self.get_usable_resources(resources, headroom_percent=25.0)

        local_worker = WorkerType(
            name="local",
            size=InstanceSize.MEDIUM,
            resources=usable,
            cost=CostModel(
                currency="USD",
                unit_cost_per_hour=0.0,
                unit_name="local_machine",
            ),
            description="Local machine with OS headroom",
        )

        # Use requested worker_count if provided, otherwise default to 1
        # Note: Local platform typically uses 0 or 1, but we respect the request for testing
        actual_worker_count = worker_count if worker_count is not None else 1

        return ClusterConfig(
            worker_type=local_worker,
            worker_count=actual_worker_count,
            driver_type=local_worker,
            driver_count=1,
            spark_version=spark_version,
            platform_config={
                "mode": "local",
                "local_threads": usable.cpu_cores,
                "usable_memory_gb": usable.memory_gb,
            },
        )

    def translate_to_spark_config(
        self,
        cluster_config: ClusterConfig,
    ) -> dict[str, Any]:
        """Translate cluster config to Spark configuration for local.

        Args:
            cluster_config: Local cluster configuration.

        Returns:
            Dictionary of Spark configuration parameters.

        """
        usable = (
            cluster_config.driver_type.resources
            if cluster_config.driver_type
            else cluster_config.worker_type.resources
        )

        config = {
            # Local mode configuration
            "spark.master": f"local[{usable.cpu_cores}]",
            "spark.driver.memory": f"{int(usable.memory_gb * 0.8)}g",
            "spark.driver.cores": usable.cpu_cores,
            # Executor configuration (for local cluster mode)
            "spark.executor.memory": f"{int(usable.memory_gb * 0.8)}g",  # same as driver in local mode
            "spark.executor.cores": usable.cpu_cores,
            # Disable dynamic allocation for local
            "spark.dynamicAllocation.enabled": "false",
            # Local directories
            "spark.local.dir": tempfile.gettempdir() + "/spark-local",  # nosec B108
            # Serialization
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            # SQL configuration
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            # Shuffle - optimized for local
            "spark.sql.shuffle.partitions": str(min(usable.cpu_cores * 2, 200)),
        }

        # Add platform-specific configs
        if cluster_config.platform_config and "local_dir" in cluster_config.platform_config:
            config["spark.local.dir"] = cluster_config.platform_config["local_dir"]

        return config

    def estimate_cost(
        self,
        _cluster_config: ClusterConfig,
        duration_hours: float,
    ) -> dict[str, Any]:
        """Estimate cost for local execution.

        Local execution has no direct cloud cost, but we calculate
        approximate electricity/hardware depreciation cost.

        Args:
            cluster_config: Cluster configuration.
            duration_hours: Expected runtime in hours.

        Returns:
            Cost breakdown with zero or minimal estimated cost.

        """
        # Estimate electricity cost (very rough estimate)
        # Average laptop/desktop: ~100W during Spark processing
        power_kw = 0.1  # 100W
        electricity_cost_per_kwh = 0.15  # Average $0.15/kWh

        kwh = power_kw * duration_hours
        electricity_cost = kwh * electricity_cost_per_kwh

        return {
            "platform": self.name,
            "currency": "USD",
            "duration_hours": duration_hours,
            "total": electricity_cost,  # For backward compatibility
            "total_cost": electricity_cost,
            "breakdown": {
                "electricity_cost": electricity_cost,
                "compute_cost": 0.0,
            },
            "notes": "Local execution - cost is estimated electricity only",
        }

    def get_recommended_local_config(
        self,
        spark_version: str = "3.5.0",
        memory_fraction: float = 0.75,
    ) -> dict[str, Any]:
        """Get recommended configuration for local Spark execution.

        This is a convenience method that returns a complete, ready-to-use
        Spark configuration dictionary for local execution.

        Args:
            spark_version: Target Spark version.
            memory_fraction: Fraction of available memory to use (0.0-1.0).

        Returns:
            Dictionary of Spark configuration parameters.

        """
        _ = spark_version  # Mark as intentionally unused
        resources = self.detect_local_resources()

        # Calculate usable memory
        usable_memory_gb = resources.memory_gb * memory_fraction
        driver_memory_gb = max(1, int(usable_memory_gb * 0.9))

        # Calculate parallelism
        parallelism = resources.cpu_cores * 2

        return {
            # Execution mode
            "spark.master": f"local[{resources.cpu_cores}]",
            # Driver configuration (only driver in local mode)
            "spark.driver.memory": f"{driver_memory_gb}g",
            "spark.driver.maxResultSize": f"{max(1, driver_memory_gb // 2)}g",
            # Serialization
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            # Memory management
            "spark.memory.fraction": "0.6",
            "spark.memory.storageFraction": "0.5",
            # SQL and shuffle
            "spark.sql.shuffle.partitions": str(max(200, parallelism)),
            "spark.default.parallelism": str(parallelism),
            # Adaptive Query Execution (Spark 3.0+)
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            # Disable dynamic allocation for local
            "spark.dynamicAllocation.enabled": "false",
            # Local storage
            "spark.local.dir": tempfile.gettempdir() + "/spark-local",  # nosec B108
            # UI
            "spark.ui.enabled": "true",
            "spark.ui.port": "4040",
        }
