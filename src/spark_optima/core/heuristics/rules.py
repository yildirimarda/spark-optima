# Licensed under the Apache License, Version 2.0

"""Heuristic rule definitions and registry.

This module provides predefined heuristic rules for Spark configuration
optimization organized by category (memory, cpu, shuffle, sql, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spark_optima.core.config_engine.models import ParameterCategory


@dataclass
class HeuristicRuleDef:
    """Definition of a heuristic rule.

    Attributes:
        param_name: Spark configuration parameter name.
        category: Parameter category.
        formula: Mathematical formula for calculation.
        base_value: Fallback value if formula cannot be applied.
        priority: Priority level (high/medium/low).
        depends_on: Required variable names.
        conditions: Conditions for rule application.
        applies_to: List of applicable platforms.
        description: Human-readable description.
        min_version: Minimum Spark version.
        max_version: Maximum Spark version (None = no limit).

    """

    param_name: str
    category: ParameterCategory
    formula: str | None = None
    base_value: Any = None
    priority: str = "medium"
    depends_on: list[str] = field(default_factory=list)
    conditions: dict[str, Any] = field(default_factory=dict)
    applies_to: list[str] = field(
        default_factory=lambda: [
            "local",
            "databricks",
            "aws_glue",
            "azure_synapse",
        ],
    )
    description: str = ""
    min_version: str = "3.0.0"
    max_version: str | None = None

    def can_apply(self, variables: set[str], platform: str, version: str) -> bool:
        """Check if rule can be applied.

        Args:
            variables: Available variable names.
            platform: Target platform.
            version: Spark version.

        Returns:
            True if all dependencies are satisfied.

        """
        # Check platform
        if platform not in self.applies_to:
            return False

        # Check version
        if not self._version_in_range(version):
            return False

        # Check dependencies
        return all(dep in variables for dep in self.depends_on)

    def _version_in_range(self, version: str) -> bool:
        """Check if version is within rule's supported range.

        Args:
            version: Spark version string.

        Returns:
            True if version is supported.

        """
        parts = [int(x) for x in version.split(".")]
        min_parts = [int(x) for x in self.min_version.split(".")]

        # Check minimum version
        for p, m in zip(parts, min_parts, strict=False):
            if p < m:
                return False
            if p > m:
                break

        # Check maximum version
        if self.max_version:
            max_parts = [int(x) for x in self.max_version.split(".")]
            for p, m in zip(parts, max_parts, strict=False):
                if p > m:
                    return False
                if p < m:
                    break

        return True


class RuleRegistry:
    """Registry of heuristic rules organized by category.

    This class provides a centralized registry for all heuristic rules
    and allows querying rules by category, platform, and priority.

    Example:
        >>> registry = RuleRegistry()
        >>> memory_rules = registry.get_rules_by_category(ParameterCategory.MEMORY)
        >>> high_priority = registry.get_rules_by_priority("high")

    """

    def __init__(self) -> None:
        """Initialize the rule registry with predefined rules."""
        self._rules: list[HeuristicRuleDef] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register all default heuristic rules."""
        # Memory rules
        self._register_memory_rules()
        # CPU rules
        self._register_cpu_rules()
        # Shuffle rules
        self._register_shuffle_rules()
        # SQL rules
        self._register_sql_rules()
        # Dynamic allocation rules
        self._register_dynamic_allocation_rules()
        # Serialization rules
        self._register_serialization_rules()
        # IO rules
        self._register_io_rules()
        # Network rules
        self._register_network_rules()

    def _register_memory_rules(self) -> None:
        """Register memory-related heuristic rules."""
        rules = [
            HeuristicRuleDef(
                param_name="spark.driver.memory",
                category=ParameterCategory.MEMORY,
                formula="min(total_memory_gb * 0.1, 16)",
                base_value="4g",
                priority="high",
                depends_on=["total_memory_gb"],
                description="Driver memory is 10% of total or max 16GB",
            ),
            HeuristicRuleDef(
                param_name="spark.driver.memoryOverhead",
                category=ParameterCategory.MEMORY,
                formula="max(driver_memory_gb * 0.1, 0.384)",
                base_value="1g",
                priority="high",
                depends_on=["driver_memory_gb"],
                description="Overhead is 10% of driver memory or minimum 384MB",
            ),
            HeuristicRuleDef(
                param_name="spark.driver.maxResultSize",
                category=ParameterCategory.MEMORY,
                formula="min(driver_memory_gb * 0.8, 8)",
                base_value="2g",
                priority="medium",
                depends_on=["driver_memory_gb"],
                description="Limit to 80% of driver memory or max 8GB",
            ),
            HeuristicRuleDef(
                param_name="spark.executor.memory",
                category=ParameterCategory.MEMORY,
                formula="min((total_memory_gb - driver_memory_gb) / num_executors * 0.9, 64)",
                base_value="4g",
                priority="high",
                depends_on=["total_memory_gb", "driver_memory_gb", "num_executors"],
                description="Distribute remaining memory across executors with 10% buffer",
            ),
            HeuristicRuleDef(
                param_name="spark.executor.memoryOverhead",
                category=ParameterCategory.MEMORY,
                formula="max(executor_memory_gb * memory_overhead_factor, 0.384)",
                base_value="1g",
                priority="high",
                depends_on=["executor_memory_gb", "memory_overhead_factor"],
                description="Overhead is 10% of executor memory or minimum 384MB",
            ),
            HeuristicRuleDef(
                param_name="spark.executor.memoryOverheadFactor",
                category=ParameterCategory.MEMORY,
                formula="0.1",
                base_value=0.1,
                priority="medium",
                description="Default 10% overhead, increase for Python/PySpark workloads",
            ),
            HeuristicRuleDef(
                param_name="spark.executor.pyspark.memory",
                category=ParameterCategory.MEMORY,
                formula="executor_memory_gb * 0.2",
                base_value="2g",
                priority="medium",
                depends_on=["executor_memory_gb"],
                conditions={"is_pyspark": True},
                description="Allocate 20% of executor memory for Python processes",
            ),
            HeuristicRuleDef(
                param_name="spark.memory.fraction",
                category=ParameterCategory.MEMORY,
                formula="0.6",
                base_value=0.6,
                priority="high",
                description=(
                    "60% for execution/storage, 40% for user data and Spark internal metadata"
                ),
            ),
            HeuristicRuleDef(
                param_name="spark.memory.storageFraction",
                category=ParameterCategory.MEMORY,
                formula="0.5",
                base_value=0.5,
                priority="medium",
                description="50-50 split between storage and execution when cached",
            ),
            HeuristicRuleDef(
                param_name="spark.memory.offHeap.enabled",
                category=ParameterCategory.MEMORY,
                formula="false",
                base_value=False,
                priority="medium",
                conditions={"large_shuffles": True},
                description="Enable for large shuffle operations to avoid GC pressure",
            ),
            HeuristicRuleDef(
                param_name="spark.memory.offHeap.size",
                category=ParameterCategory.MEMORY,
                formula="executor_memory_gb * 0.3",
                base_value="2g",
                priority="medium",
                depends_on=["executor_memory_gb"],
                conditions={"off_heap_enabled": True},
                description="Allocate 30% of executor memory for off-heap",
            ),
        ]
        self._rules.extend(rules)

    def _register_cpu_rules(self) -> None:
        """Register CPU-related heuristic rules."""
        rules = [
            HeuristicRuleDef(
                param_name="spark.executor.cores",
                category=ParameterCategory.CPU,
                formula="min(max(4, total_cores / num_executors), 8)",
                base_value=4,
                priority="high",
                depends_on=["total_cores", "num_executors"],
                description="Use 4-8 cores per executor for optimal parallelism",
            ),
            HeuristicRuleDef(
                param_name="spark.task.cpus",
                category=ParameterCategory.CPU,
                formula="1",
                base_value=1,
                priority="medium",
                description="Usually 1 CPU per task, increase for CPU-intensive tasks",
            ),
            HeuristicRuleDef(
                param_name="spark.default.parallelism",
                category=ParameterCategory.CPU,
                formula="executor_cores * num_executors * 2",
                base_value=200,
                priority="high",
                depends_on=["executor_cores", "num_executors"],
                description="2-3 tasks per core for optimal parallelism",
            ),
            HeuristicRuleDef(
                param_name="spark.sql.shuffle.partitions",
                category=ParameterCategory.SHUFFLE,
                formula="max(200, min(total_cores_cluster * 2, 10000))",
                base_value=200,
                priority="high",
                depends_on=["total_cores_cluster"],
                description="2-3 partitions per core, min 200, max 10000",
            ),
        ]
        self._rules.extend(rules)

    def _register_shuffle_rules(self) -> None:
        """Register shuffle-related heuristic rules."""
        rules = [
            HeuristicRuleDef(
                param_name="spark.shuffle.partitions",
                category=ParameterCategory.SHUFFLE,
                formula="spark_sql_shuffle_partitions",
                base_value=200,
                priority="medium",
                depends_on=["spark_sql_shuffle_partitions"],
                description="Keep consistent with spark.sql.shuffle.partitions",
            ),
            HeuristicRuleDef(
                param_name="spark.shuffle.file.buffer",
                category=ParameterCategory.SHUFFLE,
                formula="max(32, min(1024, executor_memory_gb * 1024 / 1000))",
                base_value="1m",
                priority="medium",
                depends_on=["executor_memory_gb"],
                description="Increase to 1MB for large shuffle operations",
            ),
            HeuristicRuleDef(
                param_name="spark.shuffle.spill.compress",
                category=ParameterCategory.SHUFFLE,
                formula="true",
                base_value=True,
                priority="high",
                description="Always enable to reduce disk I/O during spills",
            ),
            HeuristicRuleDef(
                param_name="spark.shuffle.spill.diskWriteBufferSize",
                category=ParameterCategory.SHUFFLE,
                formula="1024",
                base_value="1m",
                priority="medium",
                description="1MB buffer balances memory and I/O",
            ),
            HeuristicRuleDef(
                param_name="spark.reducer.maxSizeInFlight",
                category=ParameterCategory.SHUFFLE,
                formula="min(96, executor_memory_gb * 1024 / 20)",
                base_value="48m",
                priority="medium",
                depends_on=["executor_memory_gb"],
                description="Increase for large memory executors, up to 96MB",
            ),
            HeuristicRuleDef(
                param_name="spark.reducer.maxReqsInFlight",
                category=ParameterCategory.SHUFFLE,
                formula="2147483647",
                base_value=2147483647,
                priority="low",
                description="Keep default unlimited unless network issues",
            ),
            HeuristicRuleDef(
                param_name="spark.shuffle.compress",
                category=ParameterCategory.SHUFFLE,
                formula="true",
                base_value=True,
                priority="high",
                description="Always enable unless CPU-bound with fast network",
            ),
            HeuristicRuleDef(
                param_name="spark.shuffle.sort.bypassMergeThreshold",
                category=ParameterCategory.SHUFFLE,
                formula="200",
                base_value=200,
                priority="low",
                description="Bypass sort-merge for small number of partitions",
            ),
            HeuristicRuleDef(
                param_name="spark.shuffle.mapStatus.compression.codec",
                category=ParameterCategory.SHUFFLE,
                formula='"lz4"',
                base_value="lz4",
                priority="medium",
                description="lz4 for speed, zstd for better compression",
            ),
        ]
        self._rules.extend(rules)

    def _register_sql_rules(self) -> None:
        """Register SQL and AQE heuristic rules."""
        rules = [
            HeuristicRuleDef(
                param_name="spark.sql.adaptive.enabled",
                category=ParameterCategory.SQL,
                formula="true",
                base_value=True,
                priority="high",
                min_version="3.0.0",
                description="Always enable AQE for better performance",
            ),
            HeuristicRuleDef(
                param_name="spark.sql.adaptive.coalescePartitions.enabled",
                category=ParameterCategory.SQL,
                formula="true",
                base_value=True,
                priority="high",
                min_version="3.0.0",
                description="Reduces small file problem after shuffle",
            ),
            HeuristicRuleDef(
                param_name="spark.sql.adaptive.coalescePartitions.minPartitionSize",
                category=ParameterCategory.SQL,
                formula="max(1, data_size_gb * 1024 / max(spark_sql_shuffle_partitions, 200) / 4)",
                base_value="4m",
                priority="medium",
                depends_on=["data_size_gb", "spark_sql_shuffle_partitions"],
                min_version="3.2.0",
                description="Balance between parallelism and file size",
            ),
            HeuristicRuleDef(
                param_name="spark.sql.adaptive.skewJoin.enabled",
                category=ParameterCategory.SQL,
                formula="true",
                base_value=True,
                priority="high",
                min_version="3.0.0",
                description="Automatically handle skewed joins",
            ),
            HeuristicRuleDef(
                param_name="spark.sql.adaptive.skewJoin.skewedPartitionFactor",
                category=ParameterCategory.SQL,
                formula="5.0",
                base_value=5.0,
                priority="medium",
                min_version="3.0.0",
                description="5x median is a good threshold for skew detection",
            ),
            HeuristicRuleDef(
                param_name="spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes",
                category=ParameterCategory.SQL,
                formula="256m",
                base_value="256m",
                priority="medium",
                min_version="3.0.0",
                description="256MB is a reasonable minimum for skew consideration",
            ),
            HeuristicRuleDef(
                param_name="spark.sql.adaptive.localShuffleReader.enabled",
                category=ParameterCategory.SQL,
                formula="true",
                base_value=True,
                priority="high",
                min_version="3.0.0",
                description="Avoid shuffle when reading from local nodes",
            ),
            HeuristicRuleDef(
                param_name="spark.sql.autoBroadcastJoinThreshold",
                category=ParameterCategory.SQL,
                formula="min(100, max(10, driver_memory_gb * 1024 / 10))",
                base_value="10m",
                priority="high",
                depends_on=["driver_memory_gb"],
                description="Broadcast small tables up to 10-100MB",
            ),
            HeuristicRuleDef(
                param_name="spark.sql.broadcastTimeout",
                category=ParameterCategory.SQL,
                formula="600",
                base_value="600s",
                priority="medium",
                description="Increase for slower networks or large broadcasts",
            ),
            HeuristicRuleDef(
                param_name="spark.sql.files.maxPartitionBytes",
                category=ParameterCategory.SQL,
                formula="128m",
                base_value="128m",
                priority="high",
                description="128MB is optimal for Parquet/ORC files",
            ),
            HeuristicRuleDef(
                param_name="spark.sql.files.openCostInBytes",
                category=ParameterCategory.SQL,
                formula="4m",
                base_value="4m",
                priority="medium",
                description="4MB balances small file overhead vs parallelism",
            ),
        ]
        self._rules.extend(rules)

    def _register_dynamic_allocation_rules(self) -> None:
        """Register dynamic allocation heuristic rules."""
        rules = [
            HeuristicRuleDef(
                param_name="spark.dynamicAllocation.enabled",
                category=ParameterCategory.DYNAMIC_ALLOCATION,
                formula="true",
                base_value=True,
                priority="high",
                applies_to=["databricks", "aws_glue", "azure_synapse"],
                description="Enable for cost optimization and elasticity (cloud platforms)",
            ),
            HeuristicRuleDef(
                param_name="spark.dynamicAllocation.enabled",
                category=ParameterCategory.DYNAMIC_ALLOCATION,
                formula="false",
                base_value=False,
                priority="high",
                applies_to=["local"],
                description="Disable for local execution",
            ),
            HeuristicRuleDef(
                param_name="spark.dynamicAllocation.minExecutors",
                category=ParameterCategory.DYNAMIC_ALLOCATION,
                formula="2",
                base_value=2,
                priority="medium",
                applies_to=["databricks", "aws_glue", "azure_synapse"],
                description="Keep 2 executors minimum for quick startup",
            ),
            HeuristicRuleDef(
                param_name="spark.dynamicAllocation.maxExecutors",
                category=ParameterCategory.DYNAMIC_ALLOCATION,
                formula="max(10, total_cores / executor_cores)",
                base_value=20,
                priority="high",
                depends_on=["total_cores", "executor_cores"],
                applies_to=["databricks", "aws_glue", "azure_synapse"],
                description="Scale with available cluster capacity",
            ),
            HeuristicRuleDef(
                param_name="spark.dynamicAllocation.initialExecutors",
                category=ParameterCategory.DYNAMIC_ALLOCATION,
                formula="spark_dynamicAllocation_minExecutors",
                base_value=2,
                priority="low",
                depends_on=["spark_dynamicAllocation_minExecutors"],
                applies_to=["databricks", "aws_glue", "azure_synapse"],
                description="Start with minimum executors",
            ),
            HeuristicRuleDef(
                param_name="spark.dynamicAllocation.executorIdleTimeout",
                category=ParameterCategory.DYNAMIC_ALLOCATION,
                formula="300",
                base_value="300s",
                priority="medium",
                applies_to=["databricks", "aws_glue", "azure_synapse"],
                description="5 minutes idle before removal for cost savings",
            ),
            HeuristicRuleDef(
                param_name="spark.dynamicAllocation.cachedExecutorIdleTimeout",
                category=ParameterCategory.DYNAMIC_ALLOCATION,
                formula='"infinity"',
                base_value="infinity",
                priority="low",
                applies_to=["databricks", "aws_glue", "azure_synapse"],
                description="Keep executors with cached data indefinitely",
            ),
            HeuristicRuleDef(
                param_name="spark.dynamicAllocation.schedulerBacklogTimeout",
                category=ParameterCategory.DYNAMIC_ALLOCATION,
                formula="5",
                base_value="5s",
                priority="medium",
                applies_to=["databricks", "aws_glue", "azure_synapse"],
                description="Wait 5s of backlog before scaling up",
            ),
            HeuristicRuleDef(
                param_name="spark.shuffle.service.enabled",
                category=ParameterCategory.DYNAMIC_ALLOCATION,
                formula="true",
                base_value=True,
                priority="high",
                conditions={"dynamic_allocation_enabled": True},
                description="Required for dynamic allocation with shuffle",
            ),
        ]
        self._rules.extend(rules)

    def _register_serialization_rules(self) -> None:
        """Register serialization heuristic rules."""
        rules = [
            HeuristicRuleDef(
                param_name="spark.serializer",
                category=ParameterCategory.SERIALIZATION,
                formula='"org.apache.spark.serializer.KryoSerializer"',
                base_value="org.apache.spark.serializer.KryoSerializer",
                priority="high",
                description="Kryo is faster and more compact than JavaSerializer",
            ),
            HeuristicRuleDef(
                param_name="spark.kryoserializer.buffer.max",
                category=ParameterCategory.SERIALIZATION,
                formula="min(512, max(64, executor_memory_gb * 1024 / 8))",
                base_value="64m",
                priority="medium",
                depends_on=["executor_memory_gb"],
                description="Scale with executor memory, max 512MB",
            ),
            HeuristicRuleDef(
                param_name="spark.kryoserializer.buffer",
                category=ParameterCategory.SERIALIZATION,
                formula="64k",
                base_value="64k",
                priority="low",
                description="Increase if objects are large",
            ),
            HeuristicRuleDef(
                param_name="spark.kryo.registrationRequired",
                category=ParameterCategory.SERIALIZATION,
                formula="false",
                base_value=False,
                priority="low",
                description="Enable only for performance critical code",
            ),
        ]
        self._rules.extend(rules)

    def _register_io_rules(self) -> None:
        """Register IO-related heuristic rules."""
        rules = [
            HeuristicRuleDef(
                param_name="spark.io.compression.codec",
                category=ParameterCategory.IO,
                formula='"zstd"',
                base_value="zstd",
                priority="high",
                description="zstd for better compression, lz4 for speed",
            ),
            HeuristicRuleDef(
                param_name="spark.io.compression.zstd.level",
                category=ParameterCategory.IO,
                formula="3",
                base_value=3,
                priority="medium",
                description="Level 3 balances speed and compression",
            ),
            HeuristicRuleDef(
                param_name="spark.sql.parquet.compression.codec",
                category=ParameterCategory.IO,
                formula='"zstd"',
                base_value="zstd",
                priority="high",
                description="zstd for best compression ratio, snappy for speed",
            ),
            HeuristicRuleDef(
                param_name="spark.sql.orc.compression.codec",
                category=ParameterCategory.IO,
                formula='"zstd"',
                base_value="zstd",
                priority="high",
                description="zstd for best compression ratio",
            ),
        ]
        self._rules.extend(rules)

    def _register_network_rules(self) -> None:
        """Register network-related heuristic rules."""
        rules = [
            HeuristicRuleDef(
                param_name="spark.network.timeout",
                category=ParameterCategory.NETWORK,
                formula="600",
                base_value="600s",
                priority="high",
                description="Increase for large shuffles or slow networks",
            ),
            HeuristicRuleDef(
                param_name="spark.rpc.askTimeout",
                category=ParameterCategory.NETWORK,
                formula="600",
                base_value="600s",
                priority="medium",
                description="Match network.timeout for consistency",
            ),
            HeuristicRuleDef(
                param_name="spark.rpc.connect.threads",
                category=ParameterCategory.NETWORK,
                formula="max(64, total_cores_cluster)",
                base_value=64,
                priority="medium",
                depends_on=["total_cores_cluster"],
                description="Scale with cluster size",
            ),
        ]
        self._rules.extend(rules)

    def get_all_rules(self) -> list[HeuristicRuleDef]:
        """Get all registered rules.

        Returns:
            List of all heuristic rules.

        """
        return self._rules.copy()

    def get_rules_by_category(self, category: ParameterCategory) -> list[HeuristicRuleDef]:
        """Get rules filtered by category.

        Args:
            category: Category to filter by.

        Returns:
            List of matching rules.

        """
        return [r for r in self._rules if r.category == category]

    def get_rules_by_priority(self, priority: str) -> list[HeuristicRuleDef]:
        """Get rules filtered by priority.

        Args:
            priority: Priority level (high/medium/low).

        Returns:
            List of matching rules.

        """
        return [r for r in self._rules if r.priority == priority]

    def get_rules_for_platform(self, platform: str) -> list[HeuristicRuleDef]:
        """Get rules applicable for a platform.

        Args:
            platform: Platform name.

        Returns:
            List of applicable rules.

        """
        return [r for r in self._rules if platform in r.applies_to]

    def get_rule(self, param_name: str) -> HeuristicRuleDef | None:
        """Get a specific rule by parameter name.

        Args:
            param_name: Spark configuration parameter name.

        Returns:
            Rule definition if found, None otherwise.

        """
        for rule in self._rules:
            if rule.param_name == param_name:
                return rule
        return None

    def add_rule(self, rule: HeuristicRuleDef) -> None:
        """Add a custom rule to the registry.

        Args:
            rule: Rule definition to add.

        """
        self._rules.append(rule)

    def get_applicable_rules(
        self,
        variables: set[str],
        platform: str,
        version: str,
        category: ParameterCategory | None = None,
    ) -> list[HeuristicRuleDef]:
        """Get rules that can be applied given the context.

        Args:
            variables: Available variable names.
            platform: Target platform.
            version: Spark version.
            category: Optional category filter.

        Returns:
            List of applicable rules.

        """
        rules = self._rules

        if category:
            rules = [r for r in rules if r.category == category]

        return [r for r in rules if r.can_apply(variables, platform, version)]
