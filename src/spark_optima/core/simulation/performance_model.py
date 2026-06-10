# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Analytical performance model for Spark job simulation.

This module provides sophisticated analytical modeling of Spark job performance
based on configuration parameters, data characteristics, and operation types.
It uses queueing theory, statistical models, and Spark internals knowledge.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from spark_optima.platforms.models import CostModel, ResourceSpec

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of Spark operations with different performance characteristics."""

    SCAN = auto()  # Reading data from source
    FILTER = auto()  # Filtering rows
    PROJECT = auto()  # Column selection/projection
    AGGREGATION = auto()  # GroupBy, reduceByKey
    JOIN = auto()  # Join operations (shuffle-heavy)
    SORT = auto()  # OrderBy, sortByKey
    UNION = auto()  # Union operations
    WINDOW = auto()  # Window functions
    UDF = auto()  # User-defined functions
    CACHED = auto()  # Cached/persisted data access


class JoinType(Enum):
    """Types of join operations."""

    BROADCAST_HASH = "broadcast_hash"
    SHUFFLE_HASH = "shuffle_hash"
    SORT_MERGE = "sort_merge"
    CARTESIAN = "cartesian"


@dataclass
class DataCharacteristics:
    """Characteristics of the data being processed.

    Attributes:
        size_gb: Total data size in GB.
        num_rows: Estimated number of rows (if known).
        num_columns: Number of columns.
        avg_row_size_bytes: Average row size in bytes.
        format: Data format (parquet, delta, json, csv, orc).
        compression: Compression codec (snappy, gzip, etc.).
        partitioning: Number of partitions in source.
        null_ratio: Ratio of null values (0-1).
        cardinality: Dict of column cardinalities {column: distinct_count}.
        skew_factor: Data skew factor (1.0 = uniform, >1 = skewed).

    """

    size_gb: float = 10.0
    num_rows: int | None = None
    num_columns: int = 10
    avg_row_size_bytes: float = 100.0
    format: str = "parquet"
    compression: str = "snappy"
    partitioning: int = 100
    null_ratio: float = 0.05
    cardinality: dict[str, int] | None = None
    skew_factor: float = 1.0

    def __post_init__(self) -> None:
        """Calculate derived metrics."""
        if self.num_rows is None and self.size_gb > 0:
            # Estimate row count from size
            self.num_rows = int((self.size_gb * 1024**3) / max(self.avg_row_size_bytes, 1))

        if self.cardinality is None:
            self.cardinality = {}


@dataclass
class OperationProfile:
    """Profile of operations in a Spark job.

    Attributes:
        operations: List of operation types in execution order.
        join_details: Details of join operations {index: JoinType}.
        has_aggregation: Whether job contains aggregations.
        has_shuffle: Whether job contains shuffle operations.
        estimated_stages: Estimated number of stages.

    """

    operations: list[OperationType] = field(default_factory=lambda: [OperationType.SCAN])
    join_details: dict[int, JoinType] = field(default_factory=dict)
    has_aggregation: bool = False
    has_shuffle: bool = False
    estimated_stages: int = 1

    def __post_init__(self) -> None:
        """Initialize defaults."""


class PerformanceModel:
    """Analytical performance model for Spark job estimation.

    This class uses sophisticated analytical models based on:
    - Queueing theory for executor utilization
    - Statistical models for shuffle overhead
    - Spark internals for memory and I/O patterns
    - Data format characteristics

    Example:
        >>> model = PerformanceModel()
        >>> metrics = model.estimate(
        ...     config={"spark.executor.memory": "4g", ...},
        ...     resource_spec=ResourceSpec(cpu_cores=16, memory_gb=64),
        ...     data_profile=DataCharacteristics(size_gb=100, format="parquet"),
        ...     operations=OperationProfile(
            operations=[OperationType.SCAN, OperationType.AGGREGATION]
        )
        ... )

    """

    # Format-specific reading speeds (GB/s per core)
    FORMAT_READ_SPEEDS = {
        "parquet": 0.15,
        "delta": 0.12,
        "orc": 0.13,
        "json": 0.03,
        "csv": 0.05,
        "avro": 0.08,
    }

    # Compression ratios (data size multiplier)
    COMPRESSION_RATIOS = {
        "none": 1.0,
        "snappy": 0.7,
        "gzip": 0.5,
        "lz4": 0.75,
        "zstd": 0.55,
    }

    # Operation complexity factors (relative to scan)
    OPERATION_COMPLEXITY = {
        OperationType.SCAN: 1.0,
        OperationType.FILTER: 1.1,
        OperationType.PROJECT: 1.05,
        OperationType.AGGREGATION: 2.5,
        OperationType.JOIN: 3.0,
        OperationType.SORT: 2.0,
        OperationType.UNION: 1.2,
        OperationType.WINDOW: 2.2,
        OperationType.UDF: 2.0,
        OperationType.CACHED: 0.3,
    }

    # Join type overhead factors
    JOIN_OVERHEAD = {
        JoinType.BROADCAST_HASH: 1.1,
        JoinType.SHUFFLE_HASH: 2.5,
        JoinType.SORT_MERGE: 2.8,
        JoinType.CARTESIAN: 10.0,
    }

    def __init__(self) -> None:
        """Initialize the performance model."""
        self._cache: dict[str, Any] = {}

    def estimate(
        self,
        config: dict[str, Any],
        resource_spec: ResourceSpec | None = None,
        cost_model: CostModel | None = None,
        data_profile: DataCharacteristics | None = None,
        operations: OperationProfile | None = None,
    ) -> dict[str, Any]:
        """Estimate performance metrics for a configuration.

        Args:
            config: Spark configuration dictionary.
            resource_spec: Resource specifications.
            cost_model: Cost model for cost estimation.
            data_profile: Data characteristics.
            operations: Operation profile of the job.

        Returns:
            Dictionary containing detailed performance metrics.

        """
        # Set defaults
        resource_spec = resource_spec or ResourceSpec(cpu_cores=4, memory_gb=16)
        data_profile = data_profile or DataCharacteristics()
        operations = operations or OperationProfile()

        # Extract configuration parameters
        executor_memory_gb = self._parse_memory(config.get("spark.executor.memory", "4g"))
        executor_cores = int(config.get("spark.executor.cores", 4))
        self._parse_memory(config.get("spark.driver.memory", "4g"))
        parallelism = int(config.get("spark.default.parallelism", 200))
        int(config.get("spark.sql.shuffle.partitions", 200))

        # Calculate cluster topology
        cluster_topology = self._calculate_cluster_topology(
            resource_spec=resource_spec,
            executor_memory_gb=executor_memory_gb,
            executor_cores=executor_cores,
        )

        # Calculate data processing metrics
        io_metrics = self._estimate_io_metrics(
            data_profile=data_profile,
            config=config,
            cluster_topology=cluster_topology,
        )

        # Calculate execution time by stage
        stage_times = self._estimate_stage_execution_times(
            operations=operations,
            data_profile=data_profile,
            cluster_topology=cluster_topology,
            config=config,
            io_metrics=io_metrics,
        )

        # Calculate shuffle metrics
        shuffle_metrics = self._estimate_shuffle_metrics(
            operations=operations,
            data_profile=data_profile,
            config=config,
            cluster_topology=cluster_topology,
        )

        # Calculate memory usage
        memory_metrics = self._estimate_memory_usage(
            data_profile=data_profile,
            config=config,
            cluster_topology=cluster_topology,
            operations=operations,
        )

        # Calculate CPU utilization
        cpu_metrics = self._estimate_cpu_utilization(
            cluster_topology=cluster_topology,
            parallelism=parallelism,
            stage_times=stage_times,
        )

        # Aggregate total execution time
        total_time = self._aggregate_execution_time(
            stage_times=stage_times,
            shuffle_metrics=shuffle_metrics,
            cluster_topology=cluster_topology,
        )

        # Estimate cost
        cost = self._estimate_cost(
            execution_time_seconds=total_time,
            resource_spec=resource_spec,
            cost_model=cost_model,
            cluster_topology=cluster_topology,
        )

        # Validate feasibility
        is_feasible, feasibility_issues = self._validate_feasibility(
            config=config,
            resource_spec=resource_spec,
            memory_metrics=memory_metrics,
            cluster_topology=cluster_topology,
        )

        # Calculate confidence interval for simulation
        # Lower confidence for complex operations, higher for simple scans
        confidence = self._calculate_confidence(operations, data_profile, config)
        confidence_interval = self._calculate_confidence_interval(
            total_time,
            confidence,
        )

        # Build comprehensive metrics
        metrics = {
            "execution_time_seconds": total_time,
            "memory_peak_gb": memory_metrics["peak_gb"],
            "memory_average_gb": memory_metrics["average_gb"],
            "cpu_utilization_percent": cpu_metrics["average_utilization"],
            "cpu_peak_percent": cpu_metrics["peak_utilization"],
            "shuffle_read_gb": shuffle_metrics["read_gb"],
            "shuffle_write_gb": shuffle_metrics["write_gb"],
            "shuffle_spill_gb": shuffle_metrics["spill_gb"],
            "cost_estimate_usd": cost,
            "success": is_feasible,
            "error_message": "; ".join(feasibility_issues) if not is_feasible else "",
            # Simulation confidence
            "simulation_confidence": confidence,
            "confidence_interval": confidence_interval,
            "is_simulation": True,
            "simulation_warning": (
                "This is a simulation estimate, not actual measurement. Results may vary in real execution."
                if confidence < 0.7
                else ""
            ),
            # Detailed breakdown
            "stage_times": stage_times,
            "io_metrics": io_metrics,
            "cluster_topology": cluster_topology,
            "memory_breakdown": memory_metrics,
            "cpu_breakdown": cpu_metrics,
            "shuffle_breakdown": shuffle_metrics,
            "feasibility_issues": feasibility_issues,
        }

        return metrics

    def _calculate_cluster_topology(
        self,
        resource_spec: ResourceSpec,
        executor_memory_gb: float,
        executor_cores: int,
    ) -> dict[str, Any]:
        """Calculate cluster topology based on resources.

        Args:
            resource_spec: Total available resources.
            executor_memory_gb: Memory per executor.
            executor_cores: Cores per executor.

        Returns:
            Dictionary with cluster topology details.

        """
        # Calculate number of executors that fit in resources
        # Account for overhead (10% for OS, 15% for Spark overhead)
        usable_memory = resource_spec.memory_gb * 0.75
        usable_cores = max(1, int(resource_spec.cpu_cores * 0.9))

        # Calculate executors based on memory
        memory_based_executors = int(usable_memory / (executor_memory_gb * 1.1))

        # Calculate executors based on cores
        core_based_executors = int(usable_cores / executor_cores)

        # Use the limiting factor
        num_executors = max(1, min(memory_based_executors, core_based_executors))

        # Recalculate based on actual executor count
        total_executor_memory = num_executors * executor_memory_gb
        total_executor_cores = num_executors * executor_cores

        return {
            "num_executors": num_executors,
            "executor_memory_gb": executor_memory_gb,
            "executor_cores": executor_cores,
            "total_executor_memory_gb": total_executor_memory,
            "total_executor_cores": total_executor_cores,
            "usable_memory_gb": usable_memory,
            "usable_cores": usable_cores,
        }

    def _estimate_io_metrics(
        self,
        data_profile: DataCharacteristics,
        config: dict[str, Any],
        cluster_topology: dict[str, Any],
    ) -> dict[str, float]:
        """Estimate I/O metrics for data reading.

        Args:
            data_profile: Data characteristics.
            config: Spark configuration.
            cluster_topology: Cluster topology details.

        Returns:
            Dictionary with I/O metrics.

        """
        # Get format-specific read speed
        read_speed_per_core = self.FORMAT_READ_SPEEDS.get(data_profile.format.lower(), 0.1)

        # Apply compression penalty
        compression_ratio = self.COMPRESSION_RATIOS.get(data_profile.compression.lower(), 0.7)
        effective_read_speed = read_speed_per_core * (1 + (1 - compression_ratio))

        # Calculate parallel read throughput
        total_cores = cluster_topology["total_executor_cores"]
        parallel_throughput = effective_read_speed * total_cores

        # Estimate read time
        read_time = data_profile.size_gb / max(parallel_throughput, 0.01)

        # Apply AQE optimizations
        if config.get("spark.sql.adaptive.enabled", True) and config.get(
            "spark.sql.adaptive.coalescePartitions.enabled",
            True,
        ):
            read_time *= 0.9  # 10% improvement from coalescing

        return {
            "read_speed_per_core_gb_s": read_speed_per_core,
            "parallel_throughput_gb_s": parallel_throughput,
            "read_time_seconds": read_time,
            "effective_data_size_gb": data_profile.size_gb,
        }

    def _estimate_stage_execution_times(
        self,
        operations: OperationProfile,
        data_profile: DataCharacteristics,
        cluster_topology: dict[str, Any],
        config: dict[str, Any],
        io_metrics: dict[str, float],
    ) -> dict[str, float]:
        _ = config  # Mark as intentionally unused
        """Estimate execution time for each stage.

        Args:
            operations: Operation profile.
            data_profile: Data characteristics.
            cluster_topology: Cluster topology.
            config: Spark configuration.
            io_metrics: I/O metrics.

        Returns:
            Dictionary mapping stage names to times.

        """
        if operations is None:
            raise ValueError("Operations must not be None")
        if data_profile is None:
            raise ValueError("Data profile must not be None")
        stage_times = {}

        # Base processing rate (GB/s per core for simple scan)
        base_rate = 0.5  # GB/s per core
        total_cores = cluster_topology["total_executor_cores"]

        for i, op in enumerate(operations.operations):
            stage_name = f"stage_{i}_{op.name.lower()}"

            # Get operation complexity
            complexity = self.OPERATION_COMPLEXITY.get(op, 1.0)

            # Calculate processing time
            if op == OperationType.SCAN:
                # Scan time dominated by I/O
                processing_time = io_metrics["read_time_seconds"]
            elif op == OperationType.JOIN:
                # Join has special handling
                join_type = operations.join_details.get(i, JoinType.SORT_MERGE)
                join_overhead = self.JOIN_OVERHEAD.get(join_type, 2.0)

                # Estimate join data size (assume 2x for typical join)
                join_data_size = data_profile.size_gb * 2
                processing_time = (join_data_size / (base_rate * total_cores)) * join_overhead
            elif op == OperationType.AGGREGATION:
                # Aggregation depends on cardinality
                max_cardinality = max(data_profile.cardinality.values()) if data_profile.cardinality else 1000
                cardinality_factor = min(math.log10(max_cardinality) / 3, 3.0)

                agg_data_size = data_profile.size_gb * 0.3  # Aggregation reduces data
                processing_time = (agg_data_size / (base_rate * total_cores)) * complexity * cardinality_factor
            else:
                # General processing
                processing_time = (data_profile.size_gb / (base_rate * total_cores)) * complexity

            # Apply skew penalty
            if data_profile.skew_factor > 1.5:
                processing_time *= 1 + (data_profile.skew_factor - 1) * 0.3

            # Apply UDF penalty if present
            if op == OperationType.UDF:
                processing_time *= 1.5

            stage_times[stage_name] = processing_time

        return stage_times

    def _estimate_shuffle_metrics(
        self,
        operations: OperationProfile,
        data_profile: DataCharacteristics,
        config: dict[str, Any],
        cluster_topology: dict[str, Any],
    ) -> dict[str, float]:
        """Estimate shuffle read/write metrics.

        Args:
            operations: Operation profile.
            data_profile: Data characteristics.
            config: Spark configuration.
            cluster_topology: Cluster topology.

        Returns:
            Dictionary with shuffle metrics.

        """
        if operations is None:
            raise ValueError("Operations must not be None")
        # Determine shuffle data size
        shuffle_data_gb = 0.0

        for op in operations.operations:
            if op in (OperationType.JOIN, OperationType.AGGREGATION, OperationType.SORT):
                # These operations typically shuffle data
                shuffle_data_gb += data_profile.size_gb * 0.5

        # Apply compression
        compression_enabled = config.get("spark.shuffle.compress", True)
        compression_ratio = 0.3 if compression_enabled else 0.0

        shuffle_write_gb = shuffle_data_gb * (1 - compression_ratio)
        shuffle_read_gb = shuffle_write_gb * 0.95  # Slight reduction for filtering

        # Estimate spill
        executor_memory = cluster_topology["executor_memory_gb"]
        memory_fraction = float(config.get("spark.memory.fraction", 0.6))
        shuffle_fraction = 1 - float(config.get("spark.memory.storageFraction", 0.5))

        available_shuffle_memory = (
            executor_memory * memory_fraction * shuffle_fraction * cluster_topology["num_executors"]
        )

        if shuffle_data_gb > available_shuffle_memory:
            spill_gb = (shuffle_data_gb - available_shuffle_memory) * 0.5
        else:
            spill_gb = 0.0

        return {
            "read_gb": shuffle_read_gb,
            "write_gb": shuffle_write_gb,
            "spill_gb": spill_gb,
            "data_gb": shuffle_data_gb,
            "compression_ratio": compression_ratio,
        }

    def _estimate_memory_usage(
        self,
        data_profile: DataCharacteristics,
        config: dict[str, Any],
        cluster_topology: dict[str, Any],
        operations: OperationProfile,
    ) -> dict[str, float]:
        """Estimate memory usage profile.

        Args:
            data_profile: Data characteristics.
            config: Spark configuration.
            cluster_topology: Cluster topology.
            operations: Operation profile.

        Returns:
            Dictionary with memory metrics.

        """
        if operations is None:
            raise ValueError("Operations must not be None")
        executor_memory = cluster_topology["executor_memory_gb"]
        memory_fraction = float(config.get("spark.memory.fraction", 0.6))
        storage_fraction = float(config.get("spark.memory.storageFraction", 0.5))

        # Memory available for execution
        execution_memory = executor_memory * memory_fraction * (1 - storage_fraction)
        storage_memory = executor_memory * memory_fraction * storage_fraction

        # Estimate data memory (cached/buffered)
        if OperationType.CACHED in operations.operations:
            data_memory = data_profile.size_gb * 0.4  # 40% of data cached
        else:
            data_memory = data_profile.size_gb * 0.1  # 10% buffered

        # Peak memory includes processing overhead
        processing_overhead = data_profile.size_gb * 0.2 * memory_fraction
        peak_memory = min(
            data_memory + processing_overhead + execution_memory * 0.3,
            executor_memory * 0.85,  # Cap at 85%
        )

        # Average memory (typically lower than peak)
        average_memory = peak_memory * 0.7

        # Driver memory estimate
        driver_memory = self._parse_memory(config.get("spark.driver.memory", "4g"))
        driver_peak = driver_memory * 0.6

        return {
            "peak_gb": peak_memory,
            "average_gb": average_memory,
            "execution_gb": execution_memory,
            "storage_gb": storage_memory,
            "data_gb": data_memory,
            "driver_peak_gb": driver_peak,
            "executor_memory_gb": executor_memory,
        }

    def _estimate_cpu_utilization(
        self,
        cluster_topology: dict[str, Any],
        parallelism: int,
        stage_times: dict[str, float],
    ) -> dict[str, float]:
        """Estimate CPU utilization metrics.

        Args:
            cluster_topology: Cluster topology.
            parallelism: Default parallelism setting.
            stage_times: Stage execution times.

        Returns:
            Dictionary with CPU metrics.

        """
        total_cores = cluster_topology["total_executor_cores"]

        # Calculate average utilization based on parallelism vs cores
        # Optimal is 2-3x cores for parallelism
        optimal_parallelism = total_cores * 2.5
        parallelism_ratio = parallelism / optimal_parallelism if optimal_parallelism > 0 else 1.0

        # Base utilization
        base_utilization = min(95.0, 50.0 + (parallelism_ratio * 30))

        # Peak utilization (during heavy operations)
        peak_utilization = min(95.0, base_utilization * 1.2)

        # Adjust based on stage complexity
        if stage_times:
            avg_stage_time = sum(stage_times.values()) / len(stage_times)
            if avg_stage_time > 60:  # Long stages suggest good utilization
                base_utilization = min(95.0, base_utilization * 1.1)

        return {
            "average_utilization": base_utilization,
            "peak_utilization": peak_utilization,
            "total_cores": total_cores,
            "parallelism": parallelism,
        }

    def _aggregate_execution_time(
        self,
        stage_times: dict[str, float],
        shuffle_metrics: dict[str, float],
        cluster_topology: dict[str, Any],
    ) -> float:
        """Aggregate stage times into total execution time.

        Args:
            stage_times: Stage execution times.
            shuffle_metrics: Shuffle metrics.
            cluster_topology: Cluster topology.

        Returns:
            Total execution time in seconds.

        """
        # Sum stage times (simplified - assumes no parallel stages)
        total_stage_time = sum(stage_times.values())

        # Add shuffle overhead
        shuffle_overhead = shuffle_metrics["spill_gb"] * 2  # 2s per GB spill

        # Add coordination overhead
        num_executors = float(cluster_topology["num_executors"])
        coordination_overhead = num_executors * 0.5  # 0.5s per executor

        total_time = total_stage_time + shuffle_overhead + coordination_overhead

        return max(1.0, float(total_time))  # Minimum 1 second

    def _estimate_cost(
        self,
        execution_time_seconds: float,
        resource_spec: ResourceSpec,
        cost_model: CostModel | None,
        cluster_topology: dict[str, Any],
    ) -> float:
        _ = resource_spec  # Mark as intentionally unused
        """Estimate execution cost.

        Args:
            execution_time_seconds: Total execution time.
            resource_spec: Resource specifications.
            cost_model: Cost model.
            cluster_topology: Cluster topology.

        Returns:
            Estimated cost in USD.

        """
        duration_hours = execution_time_seconds / 3600

        if cost_model is None:
            # Default pricing model
            # $0.05 per CPU hour, $0.01 per GB memory hour
            cluster_topology["num_executors"]
            cpu_hours = duration_hours * float(cluster_topology["total_executor_cores"])
            memory_hours = duration_hours * float(cluster_topology["total_executor_memory_gb"])

            return float(cpu_hours * 0.05 + memory_hours * 0.01)

        return cost_model.calculate(duration_hours)

    def _validate_feasibility(
        self,
        config: dict[str, Any],
        resource_spec: ResourceSpec,
        memory_metrics: dict[str, float],
        cluster_topology: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Validate if configuration is feasible.

        Args:
            config: Spark configuration.
            resource_spec: Resource specifications.
            memory_metrics: Memory metrics.
            cluster_topology: Cluster topology.

        Returns:
            Tuple of (is_feasible, list_of_issues).

        """
        issues = []

        executor_memory = cluster_topology["executor_memory_gb"]
        peak_memory = memory_metrics["peak_gb"]

        # Check memory limits
        if peak_memory > executor_memory * 0.95:
            issues.append(
                f"Peak memory ({peak_memory:.1f}GB) exceeds executor memory ({executor_memory:.1f}GB)",
            )

        # Check executor memory against resources
        if executor_memory > resource_spec.memory_gb:
            issues.append(
                f"Executor memory ({executor_memory:.1f}GB) exceeds total memory ({resource_spec.memory_gb:.1f}GB)",
            )

        # Check core configuration
        executor_cores = cluster_topology["executor_cores"]
        if executor_cores > resource_spec.cpu_cores:
            issues.append(
                f"Executor cores ({executor_cores}) exceed total cores ({resource_spec.cpu_cores})",
            )

        # Check for reasonable parallelism
        parallelism = int(config.get("spark.default.parallelism", 200))
        total_cores = cluster_topology["total_executor_cores"]
        if parallelism > total_cores * 10:
            issues.append(
                f"Parallelism ({parallelism}) is very high relative to cores ({total_cores}) - may cause overhead",
            )

        is_feasible = len(issues) == 0
        return is_feasible, issues

    @staticmethod
    def _parse_memory(value: str | int | float) -> float:
        """Parse memory value to GB.

        Args:
            value: Memory value (e.g., "4g", "512m", 1024).

        Returns:
            Memory in GB.

        """
        if isinstance(value, int | float):
            return float(value)

        value = str(value).strip().lower()

        # Extract number and unit
        match = re.match(r"^([\d.]+)\s*([kmgt]?)\s*b?$", value)
        if not match:
            return 4.0  # Default 4GB

        number = float(match.group(1))
        unit = match.group(2)

        multipliers = {
            "k": 1 / (1024 * 1024),
            "m": 1 / 1024,
            "g": 1,
            "t": 1024,
        }

        return number * multipliers.get(unit, 1)

    def _calculate_confidence(
        self,
        operations: OperationProfile,
        data_profile: DataCharacteristics,
        config: dict[str, Any],
    ) -> float:
        """Calculate confidence level for simulation results.

        Higher confidence for:
        - Simple operations (scans, filters)
        - Well-understood configurations
        - Standard data formats

        Lower confidence for:
        - Complex operations (joins, UDFs)
        - Skewed data
        - Unusual configurations

        Args:
            operations: Operation profile.
            data_profile: Data characteristics.
            config: Spark configuration.

        Returns:
            Confidence score (0.0 to 1.0).

        """
        confidence = 0.85  # Base confidence for simulation

        # Reduce confidence for complex operations
        if operations:
            complex_ops = {
                OperationType.JOIN,
                OperationType.UDF,
                OperationType.WINDOW,
            }
            for op in operations.operations:
                if op in complex_ops:
                    confidence -= 0.1

        # Reduce confidence for skewed data
        if data_profile and data_profile.skew_factor > 1.5:
            confidence -= 0.1 * (data_profile.skew_factor - 1.0)

        # Reduce confidence for very large data
        if data_profile and data_profile.size_gb > 1000:
            confidence -= 0.1

        # Reduce confidence for unusual configurations
        if config:
            # Non-standard serializer
            serializer = config.get("spark.serializer", "")
            if "Kryo" not in serializer and "Kyro" not in serializer:
                confidence -= 0.05

            # Disabled AQE
            if not config.get("spark.sql.adaptive.enabled", True):
                confidence -= 0.05

        return max(0.3, min(0.95, confidence))

    def _calculate_confidence_interval(
        self,
        estimated_time: float,
        confidence: float,
    ) -> dict[str, float]:
        """Calculate confidence interval for execution time.

        Args:
            estimated_time: Estimated execution time in seconds.
            confidence: Confidence level (0.0 to 1.0).

        Returns:
            Dictionary with lower and upper bounds.

        """
        # Width of interval based on confidence
        # Lower confidence = wider interval
        width_factor = 0.5 + (1.0 - confidence) * 1.5

        # Also widen for longer jobs (more uncertainty)
        if estimated_time > 300:  # 5 minutes
            width_factor *= 1.3
        elif estimated_time > 60:  # 1 minute
            width_factor *= 1.1

        half_width = estimated_time * width_factor * 0.5

        return {
            "lower_seconds": max(1.0, estimated_time - half_width),
            "upper_seconds": estimated_time + half_width,
            "confidence_level": confidence,
        }
