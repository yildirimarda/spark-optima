# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Evaluation context for heuristic rules.

This module provides the EvaluationContext class for managing variables,
resources, and state during heuristic evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spark_optima.platforms.models import ResourceSpec


@dataclass
class DataProfile:
    """Data characteristics for heuristic optimization.

    Attributes:
        format: Data format (parquet, orc, json, csv, delta, etc.).
        size_gb: Total data size in gigabytes.
        num_files: Number of files/partitions.
        avg_file_size_mb: Average file size in megabytes.
        compression: Compression codec used.
        schema: Schema information (column names, types, etc.).
        schema_complexity: Simple/medium/complex schema indicator.
        has_nulls: Whether data contains null values.
        is_partitioned: Whether data is partitioned.
        partition_columns: List of partition column names.
        partitioning: Alias for partition_columns (for API compatibility).

    """

    format: str = "parquet"
    size_gb: float = 0.0
    num_files: int = 0
    avg_file_size_mb: float = 0.0
    compression: str | None = None
    schema: dict[str, Any] | None = None
    schema_complexity: str = "medium"
    has_nulls: bool = True
    is_partitioned: bool = False
    partition_columns: list[str] = field(default_factory=list)
    partitioning: list[str] | None = field(default=None)  # Alias for partition_columns

    def __post_init__(self) -> None:
        """Handle partitioning alias and schema conversion."""
        # If partitioning is provided but partition_columns is empty, use partitioning
        if self.partitioning is not None and not self.partition_columns:
            self.partition_columns = self.partitioning
            self.is_partitioned = len(self.partition_columns) > 0

        # Convert schema string to dict if needed
        if isinstance(self.schema, str):
            # Simple schema string parsing
            # Format: "col1:type1,col2:type2,..."
            schema_dict = {}
            for field in self.schema.split(","):
                if ":" in field:
                    col_name, col_type = field.split(":", 1)
                    schema_dict[col_name.strip()] = col_type.strip()
                elif field.strip():
                    schema_dict[field.strip()] = "string"
            self.schema = schema_dict if schema_dict else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "format": self.format,
            "size_gb": self.size_gb,
            "num_files": self.num_files,
            "avg_file_size_mb": self.avg_file_size_mb,
            "compression": self.compression,
            "schema": self.schema,
            "schema_complexity": self.schema_complexity,
            "has_nulls": self.has_nulls,
            "is_partitioned": self.is_partitioned,
            "partition_columns": self.partition_columns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataProfile:
        """Create from dictionary representation."""
        # Handle partitioning alias
        partition_columns = data.get("partition_columns", [])
        if not partition_columns and "partitioning" in data:
            partitioning = data["partitioning"]
            if isinstance(partitioning, list):
                partition_columns = partitioning

        return cls(
            format=data.get("format", "parquet"),
            size_gb=data.get("size_gb", 0.0),
            num_files=data.get("num_files", 0),
            avg_file_size_mb=data.get("avg_file_size_mb", 0.0),
            compression=data.get("compression"),
            schema=data.get("schema"),
            schema_complexity=data.get("schema_complexity", "medium"),
            has_nulls=data.get("has_nulls", True),
            is_partitioned=data.get("is_partitioned", False) or len(partition_columns) > 0,
            partition_columns=partition_columns,
        )


@dataclass
class EvaluationContext:
    """Context for heuristic rule evaluation.

    This class manages all variables and state needed during the evaluation
    of heuristic rules, including resources, platform information, and
    data characteristics.

    Attributes:
        resources: Available system resources.
        platform: Target platform name.
        spark_version: Spark version string.
        data_profile: Data characteristics.
        num_executors: Calculated number of executors.
        executor_cores: Cores per executor.
        executor_memory_gb: Memory per executor in GB.
        driver_memory_gb: Driver memory in GB.
        custom_vars: Additional custom variables.

    """

    resources: ResourceSpec = field(default_factory=lambda: ResourceSpec(cpu_cores=4, memory_gb=16))
    platform: str = "local"
    spark_version: str = "3.5.0"
    data_profile: DataProfile = field(default_factory=DataProfile)

    # Calculated values (filled during evaluation)
    num_executors: int = 2
    executor_cores: int = 4
    executor_memory_gb: float = 4.0
    driver_memory_gb: float = 4.0
    memory_overhead_factor: float = 0.1

    # Custom variables
    custom_vars: dict[str, Any] = field(default_factory=dict)

    def to_variables(self) -> dict[str, Any]:
        """Convert context to variable dictionary for formula evaluation.

        Returns:
            Dictionary of variable names to values.

        """
        variables = {
            # Resource variables
            "total_memory_gb": self.resources.memory_gb,
            "total_cores": self.resources.cpu_cores,
            "total_disk_gb": self.resources.disk_gb,
            "gpu_count": self.resources.gpu_count,
            "network_gbps": self.resources.network_gbps,
            # Calculated variables
            "num_executors": self.num_executors,
            "executor_cores": self.executor_cores,
            "executor_memory_gb": self.executor_memory_gb,
            "driver_memory_gb": self.driver_memory_gb,
            "memory_overhead_factor": self.memory_overhead_factor,
            "total_executor_memory_gb": self.executor_memory_gb * self.num_executors,
            "total_cores_cluster": self.executor_cores * self.num_executors,
            # Data profile variables
            "data_size_gb": self.data_profile.size_gb,
            "data_num_files": self.data_profile.num_files,
            "data_avg_file_size_mb": self.data_profile.avg_file_size_mb,
            # Platform and version
            "platform": self.platform,
            "spark_version": self.spark_version,
        }

        # Add custom variables
        variables.update(self.custom_vars)

        return variables

    def get(self, name: str, default: Any = None) -> Any:
        """Get a variable value by name.

        Args:
            name: Variable name.
            default: Default value if not found.

        Returns:
            Variable value or default.

        """
        variables = self.to_variables()
        return variables.get(name.lower(), default)

    def set(self, name: str, value: Any) -> None:
        """Set a custom variable.

        Args:
            name: Variable name.
            value: Variable value.

        """
        self.custom_vars[name.lower()] = value

    def update_calculated_values(
        self,
        num_executors: int | None = None,
        executor_cores: int | None = None,
        executor_memory_gb: float | None = None,
        driver_memory_gb: float | None = None,
    ) -> None:
        """Update calculated resource values.

        Args:
            num_executors: Number of executors.
            executor_cores: Cores per executor.
            executor_memory_gb: Memory per executor in GB.
            driver_memory_gb: Driver memory in GB.

        """
        if num_executors is not None:
            self.num_executors = num_executors
        if executor_cores is not None:
            self.executor_cores = executor_cores
        if executor_memory_gb is not None:
            self.executor_memory_gb = executor_memory_gb
        if driver_memory_gb is not None:
            self.driver_memory_gb = driver_memory_gb

    def is_streaming(self) -> bool:
        """Check if this is a streaming workload.

        Returns:
            True if streaming context.

        """
        return bool(self.custom_vars.get("streaming", False))

    def is_memory_intensive(self) -> bool:
        """Check if workload is memory intensive.

        Returns:
            True if memory intensive (caching, large joins, etc.).

        """
        return bool(self.custom_vars.get("memory_intensive", False))

    def has_large_shuffles(self) -> bool:
        """Check if workload has large shuffles.

        Returns:
            True if large shuffles expected.

        """
        return self.custom_vars.get("large_shuffles", False) or self.data_profile.size_gb > 100

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"EvaluationContext("
            f"platform={self.platform}, "
            f"resources={self.resources.cpu_cores}c/{self.resources.memory_gb:.1f}g, "
            f"executors={self.num_executors}x{self.executor_cores}c/{self.executor_memory_gb:.1f}g"
            f")"
        )
