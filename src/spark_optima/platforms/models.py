# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Resource models for Spark platforms.

This module defines data models for representing platform resources,
constraints, and cost calculations across different Spark deployment
platforms including Local, AWS Glue, Databricks, and Azure Synapse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InstanceSize(str, Enum):
    """Standard instance sizes across platforms."""

    XSMALL = "xsmall"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"
    XXLARGE = "xxlarge"
    XXXLARGE = "xxxlarge"


@dataclass
class ResourceSpec:
    """Resource specifications for a compute instance.

    Attributes:
        cpu_cores: Number of CPU cores (vCPUs).
        memory_gb: Total memory in gigabytes.
        disk_gb: Local disk/storage in gigabytes.
        gpu_count: Number of GPUs (0 if none).
        network_gbps: Network bandwidth in Gbps.

    Example:
        >>> spec = ResourceSpec(cpu_cores=4, memory_gb=16, disk_gb=64)
        >>> print(f"{spec.cpu_cores} vCPUs, {spec.memory_gb}GB RAM")

    """

    cpu_cores: int
    memory_gb: float
    disk_gb: float = 0.0
    gpu_count: int = 0
    network_gbps: float = 10.0

    def __post_init__(self) -> None:
        """Validate resource specifications."""
        if self.cpu_cores < 1:
            raise ValueError("CPU cores must be at least 1")
        if self.memory_gb <= 0:
            raise ValueError("Memory must be positive")
        if self.disk_gb < 0:
            raise ValueError("Disk cannot be negative")
        if self.gpu_count < 0:
            raise ValueError("GPU count cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "disk_gb": self.disk_gb,
            "gpu_count": self.gpu_count,
            "network_gbps": self.network_gbps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceSpec:
        """Create from dictionary representation."""
        return cls(
            cpu_cores=data["cpu_cores"],
            memory_gb=data["memory_gb"],
            disk_gb=data.get("disk_gb", 0.0),
            gpu_count=data.get("gpu_count", 0),
            network_gbps=data.get("network_gbps", 10.0),
        )

    def scale(self, factor: float) -> ResourceSpec:
        """Scale resources by a factor.

        Args:
            factor: Multiplication factor (e.g., 2.0 doubles resources).

        Returns:
            New ResourceSpec with scaled values.

        """
        return ResourceSpec(
            cpu_cores=int(self.cpu_cores * factor),
            memory_gb=self.memory_gb * factor,
            disk_gb=self.disk_gb * factor,
            gpu_count=int(self.gpu_count * factor) if self.gpu_count > 0 else 0,
            network_gbps=self.network_gbps,
        )

    def __add__(self, other: ResourceSpec) -> ResourceSpec:
        """Add two resource specifications together."""
        return ResourceSpec(
            cpu_cores=self.cpu_cores + other.cpu_cores,
            memory_gb=self.memory_gb + other.memory_gb,
            disk_gb=self.disk_gb + other.disk_gb,
            gpu_count=self.gpu_count + other.gpu_count,
            network_gbps=max(self.network_gbps, other.network_gbps),
        )


@dataclass
class PlatformConstraints:
    """Constraints and limits for a Spark platform.

    Attributes:
        min_workers: Minimum number of worker instances.
        max_workers: Maximum number of worker instances.
        min_memory_gb: Minimum memory per worker in GB.
        max_memory_gb: Maximum memory per worker in GB.
        min_cores: Minimum CPU cores per worker.
        max_cores: Maximum CPU cores per worker.
        supported_spark_versions: List of supported Spark versions.
        custom_config_keys: Platform-specific configuration keys.

    """

    min_workers: int = 1
    max_workers: int = 1000
    min_memory_gb: float = 1.0
    max_memory_gb: float = 1024.0
    min_cores: int = 1
    max_cores: int = 128
    supported_spark_versions: list[str] = field(
        default_factory=lambda: [
            "3.0.0",
            "3.1.0",
            "3.2.0",
            "3.3.0",
            "3.4.0",
            "3.5.0",
            "4.0.0",
        ],
    )
    custom_config_keys: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate constraints."""
        if self.min_workers < 0:
            raise ValueError("min_workers cannot be negative")
        if self.max_workers < self.min_workers:
            raise ValueError("max_workers must be >= min_workers")
        if self.min_memory_gb <= 0:
            raise ValueError("min_memory_gb must be positive")
        if self.max_memory_gb < self.min_memory_gb:
            raise ValueError("max_memory_gb must be >= min_memory_gb")
        if self.min_cores < 1:
            raise ValueError("min_cores must be at least 1")
        if self.max_cores < self.min_cores:
            raise ValueError("max_cores must be >= min_cores")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "min_memory_gb": self.min_memory_gb,
            "max_memory_gb": self.max_memory_gb,
            "min_cores": self.min_cores,
            "max_cores": self.max_cores,
            "supported_spark_versions": self.supported_spark_versions,
            "custom_config_keys": self.custom_config_keys,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlatformConstraints:
        """Create from dictionary representation."""
        return cls(
            min_workers=data.get("min_workers", 1),
            max_workers=data.get("max_workers", 1000),
            min_memory_gb=data.get("min_memory_gb", 1.0),
            max_memory_gb=data.get("max_memory_gb", 1024.0),
            min_cores=data.get("min_cores", 1),
            max_cores=data.get("max_cores", 128),
            supported_spark_versions=data.get(
                "supported_spark_versions",
                ["3.0.0", "3.1.0", "3.2.0", "3.3.0", "3.4.0", "3.5.0", "4.0.0"],
            ),
            custom_config_keys=data.get("custom_config_keys", {}),
        )

    def validate_resources(self, spec: ResourceSpec, worker_count: int) -> list[str]:
        """Validate resource specifications against constraints.

        Args:
            spec: Resource specification to validate.
            worker_count: Number of workers.

        Returns:
            List of validation error messages (empty if valid).

        """
        errors = []

        if worker_count < self.min_workers:
            errors.append(f"Worker count {worker_count} is below minimum {self.min_workers}")
        if worker_count > self.max_workers:
            errors.append(f"Worker count {worker_count} exceeds maximum {self.max_workers}")
        if spec.memory_gb < self.min_memory_gb:
            errors.append(f"Memory {spec.memory_gb}GB is below minimum {self.min_memory_gb}GB")
        if spec.memory_gb > self.max_memory_gb:
            errors.append(f"Memory {spec.memory_gb}GB exceeds maximum {self.max_memory_gb}GB")
        if spec.cpu_cores < self.min_cores:
            errors.append(f"CPU cores {spec.cpu_cores} is below minimum {self.min_cores}")
        if spec.cpu_cores > self.max_cores:
            errors.append(f"CPU cores {spec.cpu_cores} exceeds maximum {self.max_cores}")

        return errors


@dataclass
class CostModel:
    """Cost calculation model for platform resources.

    Attributes:
        currency: Currency code (e.g., "USD", "EUR").
        unit_cost_per_hour: Base cost per hour for the resource.
        unit_name: Name of the billing unit (e.g., "DPU", "DBU", "vCore").
        granularity_minutes: Billing granularity in minutes.
        minimum_charge_minutes: Minimum charge duration in minutes.

    Example:
        >>> cost = CostModel(unit_cost_per_hour=0.44, unit_name="DPU")
        >>> print(f"Cost for 2 hours: ${cost.calculate(2.0)}")

    """

    currency: str = "USD"
    unit_cost_per_hour: float = 0.0
    unit_name: str = "instance"
    granularity_minutes: int = 60
    minimum_charge_minutes: int = 1

    def __post_init__(self) -> None:
        """Validate cost model."""
        if self.unit_cost_per_hour < 0:
            raise ValueError("Unit cost cannot be negative")
        if self.granularity_minutes < 1:
            raise ValueError("Granularity must be at least 1 minute")

    def calculate(self, duration_hours: float, units: float = 1.0) -> float:
        """Calculate cost for a given duration.

        Args:
            duration_hours: Duration in hours.
            units: Number of units (e.g., workers, DPUs).

        Returns:
            Total cost in the specified currency.

        """
        if duration_hours <= 0:
            return 0.0

        # Apply minimum charge
        duration_minutes = max(duration_hours * 60, self.minimum_charge_minutes)

        # Round up to granularity
        granularity = self.granularity_minutes
        billed_minutes = ((int(duration_minutes) + granularity - 1) // granularity) * granularity

        billed_hours = billed_minutes / 60.0
        return billed_hours * self.unit_cost_per_hour * units

    def estimate_monthly(
        self,
        hours_per_day: float,
        days_per_month: float = 30.0,
        units: float = 1.0,
    ) -> float:
        """Estimate monthly cost.

        Args:
            hours_per_day: Average hours of usage per day.
            days_per_month: Number of days in the month (default 30).
            units: Number of units.

        Returns:
            Estimated monthly cost.

        """
        total_hours = hours_per_day * days_per_month
        return self.calculate(total_hours, units)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "currency": self.currency,
            "unit_cost_per_hour": self.unit_cost_per_hour,
            "unit_name": self.unit_name,
            "granularity_minutes": self.granularity_minutes,
            "minimum_charge_minutes": self.minimum_charge_minutes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostModel:
        """Create from dictionary representation."""
        return cls(
            currency=data.get("currency", "USD"),
            unit_cost_per_hour=data.get("unit_cost_per_hour", 0.0),
            unit_name=data.get("unit_name", "instance"),
            granularity_minutes=data.get("granularity_minutes", 60),
            minimum_charge_minutes=data.get("minimum_charge_minutes", 1),
        )


@dataclass
class WorkerType:
    """Definition of a worker instance type.

    Attributes:
        name: Unique name/identifier for the worker type.
        size: Standard size category.
        resources: Resource specifications.
        cost: Cost model for this worker type.
        description: Human-readable description.
        is_spot: Whether this is a spot/preemptible instance.
        availability_zones: List of available zones/regions.

    """

    name: str
    size: InstanceSize
    resources: ResourceSpec
    cost: CostModel
    description: str = ""
    is_spot: bool = False
    availability_zones: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "size": self.size.value,
            "resources": self.resources.to_dict(),
            "cost": self.cost.to_dict(),
            "description": self.description,
            "is_spot": self.is_spot,
            "availability_zones": self.availability_zones,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerType:
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            size=InstanceSize(data["size"]),
            resources=ResourceSpec.from_dict(data["resources"]),
            cost=CostModel.from_dict(data["cost"]),
            description=data.get("description", ""),
            is_spot=data.get("is_spot", False),
            availability_zones=data.get("availability_zones", []),
        )

    def estimate_job_cost(self, duration_hours: float, worker_count: int = 1) -> float:
        """Estimate cost for a job using this worker type.

        Args:
            duration_hours: Job duration in hours.
            worker_count: Number of workers.

        Returns:
            Estimated total cost.

        """
        return self.cost.calculate(duration_hours, worker_count)


@dataclass
class ClusterConfig:
    """Complete cluster configuration for a platform.

    Attributes:
        worker_type: Type of worker instances.
        worker_count: Number of worker instances.
        driver_type: Type of driver instance (if different from worker).
        driver_count: Number of driver instances (usually 1).
        spark_version: Spark version to use.
        platform_config: Platform-specific configuration.

    """

    worker_type: WorkerType
    worker_count: int = 2
    driver_type: WorkerType | None = None
    driver_count: int = 1
    spark_version: str = "3.5.0"
    platform_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate cluster configuration."""
        if self.worker_count < 0:
            raise ValueError("worker_count cannot be negative")
        if self.driver_count < 1:
            raise ValueError("driver_count must be at least 1")

    @property
    def total_resources(self) -> ResourceSpec:
        """Calculate total cluster resources."""
        driver = self.driver_type or self.worker_type
        workers_total = self.worker_type.resources.scale(float(self.worker_count))
        drivers_total = driver.resources.scale(float(self.driver_count))
        return workers_total + drivers_total

    @property
    def total_cost_per_hour(self) -> float:
        """Calculate total cost per hour for the cluster."""
        driver = self.driver_type or self.worker_type
        worker_cost = self.worker_type.cost.unit_cost_per_hour * self.worker_count
        driver_cost = driver.cost.unit_cost_per_hour * self.driver_count
        return worker_cost + driver_cost

    def estimate_cost(self, duration_hours: float) -> float:
        """Estimate cost for a given duration.

        Args:
            duration_hours: Duration in hours.

        Returns:
            Estimated total cost.

        """
        driver = self.driver_type or self.worker_type
        worker_cost = self.worker_type.cost.calculate(duration_hours, self.worker_count)
        driver_cost = driver.cost.calculate(duration_hours, self.driver_count)
        return worker_cost + driver_cost

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "worker_type": self.worker_type.to_dict(),
            "worker_count": self.worker_count,
            "driver_type": self.driver_type.to_dict() if self.driver_type else None,
            "driver_count": self.driver_count,
            "spark_version": self.spark_version,
            "platform_config": self.platform_config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClusterConfig:
        """Create from dictionary representation."""
        driver_data = data.get("driver_type")
        return cls(
            worker_type=WorkerType.from_dict(data["worker_type"]),
            worker_count=data.get("worker_count", 2),
            driver_type=WorkerType.from_dict(driver_data) if driver_data else None,
            driver_count=data.get("driver_count", 1),
            spark_version=data.get("spark_version", "3.5.0"),
            platform_config=data.get("platform_config", {}),
        )
