# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Base platform interface for Spark Optima.

This module defines the abstract base class that all platform implementations
must inherit from, providing a consistent interface for resource management,
configuration translation, and cost estimation across different Spark platforms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from spark_optima.platforms.models import (
    ClusterConfig,
    PlatformConstraints,
    ResourceSpec,
    WorkerType,
)


class Platform(ABC):
    """Abstract base class for Spark deployment platforms.

    This class defines the interface that all platform implementations
    (Local, AWS Glue, Databricks, Azure Synapse) must provide. It handles
    resource specifications, configuration translation, cost estimation,
    and platform-specific optimizations.

    Attributes:
        name: Platform identifier (e.g., "local", "aws_glue", "databricks").
        display_name: Human-readable platform name.
        description: Brief description of the platform.

    Example:
        >>> platform = LocalPlatform()
        >>> resources = platform.detect_local_resources()
        >>> config = platform.recommend_config(resources, "3.5.0")

    """

    def __init__(
        self,
        name: str,
        display_name: str,
        description: str = "",
    ) -> None:
        """Initialize the platform.

        Args:
            name: Platform identifier.
            display_name: Human-readable name.
            description: Platform description.

        """
        self.name = name
        self.display_name = display_name
        self.description = description

    @property
    @abstractmethod
    def constraints(self) -> PlatformConstraints:
        """Get platform resource constraints.

        Returns:
            PlatformConstraints defining min/max resources and limits.

        """
        ...

    @abstractmethod
    def get_worker_types(self) -> list[WorkerType]:
        """Get available worker instance types for this platform.

        Returns:
            List of WorkerType definitions available on this platform.

        """
        ...

    @abstractmethod
    def get_worker_type(self, name: str) -> WorkerType | None:
        """Get a specific worker type by name.

        Args:
            name: Worker type identifier.

        Returns:
            WorkerType if found, None otherwise.

        """
        ...

    @abstractmethod
    def recommend_config(
        self,
        resources: ResourceSpec,
        spark_version: str,
        worker_count: int | None = None,
    ) -> ClusterConfig:
        """Recommend an optimal cluster configuration.

        Args:
            resources: Available/requested resources.
            spark_version: Target Spark version.
            worker_count: Optional specific worker count.

        Returns:
            Recommended ClusterConfig.

        """
        ...

    @abstractmethod
    def translate_to_spark_config(
        self,
        cluster_config: ClusterConfig,
    ) -> dict[str, Any]:
        """Translate cluster config to Spark configuration parameters.

        Args:
            cluster_config: Cluster configuration to translate.

        Returns:
            Dictionary of Spark configuration key-value pairs.

        """
        ...

    @abstractmethod
    def estimate_cost(
        self,
        cluster_config: ClusterConfig,
        duration_hours: float,
    ) -> dict[str, Any]:
        """Estimate cost for running a cluster.

        Args:
            cluster_config: Cluster configuration.
            duration_hours: Expected runtime in hours.

        Returns:
            Dictionary with cost breakdown and total.

        """
        ...

    def validate_config(self, cluster_config: ClusterConfig) -> list[str]:
        """Validate a cluster configuration against platform constraints.

        Args:
            cluster_config: Configuration to validate.

        Returns:
            List of validation error messages (empty if valid).

        """
        errors = []
        constraints = self.constraints

        # Validate worker count
        if cluster_config.worker_count < constraints.min_workers:
            errors.append(
                f"Worker count {cluster_config.worker_count} is below minimum "
                f"{constraints.min_workers} for {self.name}",
            )
        if cluster_config.worker_count > constraints.max_workers:
            errors.append(
                f"Worker count {cluster_config.worker_count} exceeds maximum "
                f"{constraints.max_workers} for {self.name}",
            )

        # Validate worker resources
        worker_errors = constraints.validate_resources(
            cluster_config.worker_type.resources,
            cluster_config.worker_count,
        )
        errors.extend(worker_errors)

        # Validate Spark version
        if cluster_config.spark_version not in constraints.supported_spark_versions:
            # Check if it's a patch version difference
            base_version = ".".join(cluster_config.spark_version.split(".")[:2]) + ".0"
            if base_version not in constraints.supported_spark_versions:
                errors.append(
                    f"Spark version {cluster_config.spark_version} is not supported. "
                    f"Supported: {constraints.supported_spark_versions}",
                )

        return errors

    def get_supported_spark_versions(self) -> list[str]:
        """Get list of supported Spark versions.

        Returns:
            Sorted list of supported version strings.

        """
        return sorted(self.constraints.supported_spark_versions)

    def is_spark_version_supported(self, version: str) -> bool:
        """Check if a Spark version is supported.

        Args:
            version: Spark version string (e.g., "3.5.0").

        Returns:
            True if version is supported.

        """
        # Exact match
        if version in self.constraints.supported_spark_versions:
            return True

        # Check base version (major.minor.0)
        parts = version.split(".")
        if len(parts) >= 2:
            base = f"{parts[0]}.{parts[1]}.0"
            if base in self.constraints.supported_spark_versions:
                return True

        return False

    def get_optimal_worker_count(
        self,
        target_resources: ResourceSpec,
        worker_type: WorkerType,
    ) -> int:
        """Calculate optimal worker count for target resources.

        Args:
            target_resources: Desired total resources.
            worker_type: Type of worker to use.

        Returns:
            Recommended number of workers.

        """
        worker_resources = worker_type.resources

        # Calculate based on CPU cores
        cpu_based = target_resources.cpu_cores / max(worker_resources.cpu_cores, 1)

        # Calculate based on memory
        mem_based = target_resources.memory_gb / max(worker_resources.memory_gb, 1)

        # Use the larger of the two, rounded up
        worker_count = int(max(cpu_based, mem_based))

        # Apply constraints
        constraints = self.constraints
        worker_count = max(constraints.min_workers, worker_count)
        worker_count = min(constraints.max_workers, worker_count)

        return worker_count

    def compare_worker_types(
        self,
        type1: WorkerType,
        type2: WorkerType,
    ) -> dict[str, Any]:
        """Compare two worker types.

        Args:
            type1: First worker type.
            type2: Second worker type.

        Returns:
            Comparison results including performance and cost differences.

        """
        r1, r2 = type1.resources, type2.resources

        cpu_ratio = r1.cpu_cores / max(r2.cpu_cores, 1)
        memory_ratio = r1.memory_gb / max(r2.memory_gb, 1)
        cost_ratio = type1.cost.unit_cost_per_hour / max(type2.cost.unit_cost_per_hour, 0.001)

        return {
            "cpu_ratio": cpu_ratio,
            "memory_ratio": memory_ratio,
            "cost_ratio": cost_ratio,
            "cost_per_cpu_1": type1.cost.unit_cost_per_hour / max(r1.cpu_cores, 1),
            "cost_per_cpu_2": type2.cost.unit_cost_per_hour / max(r2.cpu_cores, 1),
            "cost_per_memory_1": type1.cost.unit_cost_per_hour / max(r1.memory_gb, 1),
            "cost_per_memory_2": type2.cost.unit_cost_per_hour / max(r2.memory_gb, 1),
            "more_cost_effective": (
                type1.name
                if (
                    type1.cost.unit_cost_per_hour / max(r1.cpu_cores, 1)
                    < type2.cost.unit_cost_per_hour / max(r2.cpu_cores, 1)
                )
                else type2.name
            ),
        }

    def __repr__(self) -> str:
        """Return string representation of the platform."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return f"{self.display_name} ({self.name})"


class LocalPlatformBase(Platform):
    """Base class for local platforms with resource detection.

    Provides common functionality for local Spark deployments including
    automatic resource detection from the host system.

    """

    def detect_local_resources(self) -> ResourceSpec:
        """Detect available system resources.

        Returns:
            ResourceSpec with detected CPU, memory, and disk.

        """
        import psutil

        # Get CPU info
        cpu_cores = psutil.cpu_count(logical=True)

        # Get memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)

        # Get disk info
        disk = psutil.disk_usage("/")
        disk_gb = disk.total / (1024**3)

        return ResourceSpec(
            cpu_cores=cpu_cores or 4,
            memory_gb=memory_gb,
            disk_gb=disk_gb,
        )

    def get_usable_resources(
        self,
        total_resources: ResourceSpec | None = None,
        headroom_percent: float = 20.0,
    ) -> ResourceSpec:
        """Calculate usable resources leaving system headroom.

        Args:
            total_resources: Total system resources (auto-detected if None).
            headroom_percent: Percentage of resources to reserve for system.

        Returns:
            ResourceSpec with usable resources.

        """
        if total_resources is None:
            total_resources = self.detect_local_resources()

        # Leave headroom for OS and other processes
        usable_factor = (100.0 - headroom_percent) / 100.0

        return ResourceSpec(
            cpu_cores=max(1, int(total_resources.cpu_cores * usable_factor)),
            memory_gb=total_resources.memory_gb * usable_factor,
            disk_gb=total_resources.disk_gb * usable_factor,
        )
