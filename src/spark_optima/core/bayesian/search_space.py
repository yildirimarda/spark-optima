# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Search space builder for Bayesian optimization.

This module provides utilities for constructing Optuna search spaces
from heuristic configuration outputs, with category-specific strategies.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from spark_optima.core.bayesian.models import SearchSpaceConfig
from spark_optima.core.config_engine.models import (
    ConfigParameter,
    ConfigSet,
    ParameterCategory,
    ParameterType,
)

logger = logging.getLogger(__name__)


class SearchSpaceBuilder:
    """Builds Optuna search spaces from heuristic configurations.

    This class creates appropriate search ranges for each Spark configuration
    parameter based on its type, category, and the heuristic baseline value.

    Example:
        >>> builder = SearchSpaceBuilder()
        >>> heuristic_config = {"spark.executor.memory": "4g", ...}
        >>> search_space = builder.build_from_heuristic(
        ...     heuristic_config=heuristic_config,
        ...     config_set=config_set,
        ...     variation_percent=0.3
        ... )

    """

    # Parameters that should be treated as categorical
    CATEGORICAL_PARAMS = {
        "spark.serializer": [
            "org.apache.spark.serializer.KryoSerializer",
            "org.apache.spark.serializer.JavaSerializer",
        ],
        "spark.io.compression.codec": ["lz4", "zstd", "snappy", "lzf"],
        "spark.sql.parquet.compression.codec": ["zstd", "snappy", "gzip", "none"],
        "spark.sql.orc.compression.codec": ["zstd", "snappy", "zlib", "none"],
        "spark.shuffle.mapStatus.compression.codec": ["lz4", "zstd"],
        "spark.scheduler.mode": ["FIFO", "FAIR", "DRR"],
    }

    # Boolean parameters (will be converted to categorical True/False)
    BOOLEAN_PARAMS = [
        "spark.sql.adaptive.enabled",
        "spark.sql.adaptive.coalescePartitions.enabled",
        "spark.sql.adaptive.skewJoin.enabled",
        "spark.sql.adaptive.localShuffleReader.enabled",
        "spark.shuffle.compress",
        "spark.shuffle.spill.compress",
        "spark.memory.offHeap.enabled",
        "spark.dynamicAllocation.enabled",
        "spark.kryo.registrationRequired",
        "spark.shuffle.service.enabled",
    ]

    # Memory parameters with specific constraints
    MEMORY_PARAMS = [
        "spark.driver.memory",
        "spark.executor.memory",
        "spark.driver.memoryOverhead",
        "spark.executor.memoryOverhead",
        "spark.executor.pyspark.memory",
        "spark.memory.offHeap.size",
        "spark.driver.maxResultSize",
    ]

    # Core/CPU parameters
    CORE_PARAMS = [
        "spark.executor.cores",
        "spark.task.cpus",
        "spark.default.parallelism",
        "spark.sql.shuffle.partitions",
        "spark.shuffle.partitions",
    ]

    # Time duration parameters
    DURATION_PARAMS = [
        "spark.network.timeout",
        "spark.rpc.askTimeout",
        "spark.sql.broadcastTimeout",
        "spark.dynamicAllocation.executorIdleTimeout",
        "spark.dynamicAllocation.cachedExecutorIdleTimeout",
        "spark.dynamicAllocation.schedulerBacklogTimeout",
    ]

    # Fixed parameters that shouldn't be optimized
    FIXED_PARAMS = [
        "spark.app.name",
        "spark.master",
        "spark.submit.deployMode",
    ]

    def __init__(self) -> None:
        """Initialize the search space builder."""
        self._search_space: dict[str, Any] = {}

    def build_from_heuristic(
        self,
        heuristic_config: dict[str, Any],
        config_set: ConfigSet,
        config: SearchSpaceConfig | None = None,
    ) -> dict[str, Any]:
        """Build search space from heuristic configuration.

        Args:
            heuristic_config: Configuration from heuristic engine.
            config_set: Spark configuration set with parameter metadata.
            config: Search space configuration. Uses defaults if None.

        Returns:
            Dictionary mapping parameter names to search space definitions.

        """
        config = config or SearchSpaceConfig()
        search_space: dict[str, Any] = {}

        for param_name, heuristic_value in heuristic_config.items():
            # Skip fixed parameters
            if param_name in self.FIXED_PARAMS or param_name in config.fixed_params:
                continue

            # Get parameter metadata if available
            param = config_set.parameters.get(param_name)

            # Build search space entry based on parameter type
            search_def = self._build_parameter_search_space(
                param_name=param_name,
                heuristic_value=heuristic_value,
                param=param,
                config=config,
            )

            if search_def:
                search_space[param_name] = search_def

        logger.info(f"Built search space with {len(search_space)} parameters")
        self._search_space = search_space
        return search_space

    def _build_parameter_search_space(
        self,
        param_name: str,
        heuristic_value: Any,
        param: ConfigParameter | None,
        config: SearchSpaceConfig,
    ) -> dict[str, Any] | None:
        """Build search space definition for a single parameter.

        Args:
            param_name: Parameter name.
            heuristic_value: Heuristic baseline value.
            param: Parameter metadata (if available).
            config: Search space configuration.

        Returns:
            Search space definition dictionary or None if skipped.

        """
        # Use custom range if specified
        if param_name in config.param_ranges:
            min_val, max_val = config.param_ranges[param_name]
            return {
                "type": "float",
                "low": min_val,
                "high": max_val,
                "base_value": heuristic_value,
            }

        # Categorical parameters
        if param_name in self.CATEGORICAL_PARAMS:
            return {
                "type": "categorical",
                "choices": self.CATEGORICAL_PARAMS[param_name],
                "base_value": heuristic_value,
            }

        # Boolean parameters
        if param_name in self.BOOLEAN_PARAMS:
            return {
                "type": "categorical",
                "choices": [True, False],
                "base_value": self._parse_boolean(heuristic_value),
            }

        # Memory parameters
        if param_name in self.MEMORY_PARAMS:
            return self._build_memory_search_space(param_name, heuristic_value, config)

        # Core/CPU parameters
        if param_name in self.CORE_PARAMS:
            return self._build_core_search_space(param_name, heuristic_value, config)

        # Duration parameters
        if param_name in self.DURATION_PARAMS:
            return self._build_duration_search_space(param_name, heuristic_value, config)

        # Float parameters (fractions, ratios)
        if param and param.param_type == ParameterType.FLOAT:
            return self._build_float_search_space(param_name, heuristic_value, param, config)

        # Integer parameters
        if param and param.param_type == ParameterType.INTEGER:
            return self._build_integer_search_space(param_name, heuristic_value, param, config)

        # Bytes parameters
        if param and param.param_type == ParameterType.BYTES:
            return self._build_bytes_search_space(param_name, heuristic_value, config)

        # Default: skip unknown parameter types
        logger.debug(f"Skipping unknown parameter type for {param_name}")
        return None

    def _build_memory_search_space(
        self,
        _param_name: str,
        heuristic_value: Any,
        config: SearchSpaceConfig,
    ) -> dict[str, Any]:
        """Build search space for memory parameters.

        Memory parameters use percentage-based ranges with minimum/maximum bounds.

        Args:
            param_name: Parameter name.
            heuristic_value: Heuristic baseline (e.g., "4g", "512m").
            config: Search space configuration.

        Returns:
            Search space definition.

        """
        # Parse memory value to bytes
        base_bytes = self._parse_memory_string(str(heuristic_value))

        # Calculate range with variation
        variation = config.variation_percent
        min_bytes = int(base_bytes * (1 - variation))
        max_bytes = int(base_bytes * (1 + variation))

        # Apply memory-specific constraints
        min_bytes = max(min_bytes, 512 * 1024 * 1024)  # Min 512MB
        max_bytes = min(max_bytes, 128 * 1024 * 1024 * 1024)  # Max 128GB

        # Ensure max is at least min
        if max_bytes < min_bytes:
            max_bytes = min_bytes + 512 * 1024 * 1024

        # Round to reasonable increments (64MB)
        min_bytes = (min_bytes // (64 * 1024 * 1024)) * (64 * 1024 * 1024)
        max_bytes = (max_bytes // (64 * 1024 * 1024)) * (64 * 1024 * 1024)

        # Final check after rounding
        if max_bytes < min_bytes:
            max_bytes = min_bytes

        return {
            "type": "bytes",
            "low": min_bytes,
            "high": max_bytes,
            "base_value": base_bytes,
            "step": 64 * 1024 * 1024,  # 64MB increments
        }

    def _build_core_search_space(
        self,
        param_name: str,
        heuristic_value: Any,
        config: SearchSpaceConfig,
    ) -> dict[str, Any]:
        """Build search space for CPU/core parameters.

        Core parameters use discrete integer ranges with platform-appropriate limits.

        Args:
            param_name: Parameter name.
            heuristic_value: Heuristic baseline (e.g., 4).
            config: Search space configuration.

        Returns:
            Search space definition.

        """
        base_value = int(heuristic_value) if heuristic_value else 4

        if param_name == "spark.executor.cores":
            # Executor cores: discrete values 2, 4, 8
            choices = [c for c in [2, 4, 8] if c <= max(8, base_value + 4)]
            return {
                "type": "categorical",
                "choices": choices,
                "base_value": base_value,
            }

        if param_name == "spark.task.cpus":
            # Task CPUs: 1-4
            return {
                "type": "int",
                "low": 1,
                "high": 4,
                "base_value": base_value,
            }

        # Parallelism/shuffle partitions: range-based
        variation = config.variation_percent
        min_val = max(10, int(base_value * (1 - variation)))
        max_val = min(10000, int(base_value * (1 + variation)))

        # Ensure max is at least min
        if max_val < min_val:
            max_val = min_val + 10

        # Round to reasonable steps
        step = max(1, (max_val - min_val) // 20)
        min_val = (min_val // step) * step
        max_val = (max_val // step) * step

        # Final check after rounding
        if max_val < min_val:
            max_val = min_val

        return {
            "type": "int",
            "low": min_val,
            "high": max_val,
            "base_value": base_value,
            "step": step,
        }

    def _build_duration_search_space(
        self,
        _param_name: str,
        heuristic_value: Any,
        config: SearchSpaceConfig,
    ) -> dict[str, Any]:
        """Build search space for duration parameters.

        Duration parameters use seconds-based ranges.

        Args:
            param_name: Parameter name.
            heuristic_value: Heuristic baseline (e.g., "600s", "5m").
            config: Search space configuration.

        Returns:
            Search space definition.

        """
        base_seconds = self._parse_duration_string(str(heuristic_value))

        # Handle infinity values (e.g., -1 or "infinity" in Spark configs)
        if base_seconds == float("inf") or base_seconds == float("-inf"):
            base_seconds = 86400  # Default to 24 hours

        variation = config.variation_percent
        min_seconds = int(base_seconds * (1 - variation))
        max_seconds = int(base_seconds * (1 + variation))

        # Ensure reasonable bounds
        min_seconds = max(1, min_seconds)
        max_seconds = min(86400, max_seconds)  # Max 24 hours

        # Ensure max is at least min
        if max_seconds < min_seconds:
            max_seconds = min_seconds

        return {
            "type": "int",
            "low": min_seconds,
            "high": max_seconds,
            "base_value": base_seconds,
        }

    def _build_float_search_space(
        self,
        _param_name: str,
        heuristic_value: Any,
        param: ConfigParameter,
        config: SearchSpaceConfig,
    ) -> dict[str, Any]:
        """Build search space for float parameters.

        Args:
            param_name: Parameter name.
            heuristic_value: Heuristic baseline.
            param: Parameter metadata.
            config: Search space configuration.

        Returns:
            Search space definition.

        """
        base_value = float(heuristic_value) if heuristic_value else 0.5

        # Use constraints if available
        min_val = param.constraints.min_value if param.constraints else None
        max_val = param.constraints.max_value if param.constraints else None

        if min_val is None:
            min_val = max(0.0, base_value * (1 - config.variation_percent))
        if max_val is None:
            max_val = min(1.0, base_value * (1 + config.variation_percent))

        return {
            "type": "float",
            "low": min_val,
            "high": max_val,
            "base_value": base_value,
        }

    def _build_integer_search_space(
        self,
        _param_name: str,
        heuristic_value: Any,
        param: ConfigParameter,
        config: SearchSpaceConfig,
    ) -> dict[str, Any]:
        """Build search space for integer parameters.

        Args:
            param_name: Parameter name.
            heuristic_value: Heuristic baseline.
            param: Parameter metadata.
            config: Search space configuration.

        Returns:
            Search space definition.

        """
        base_value = int(heuristic_value) if heuristic_value else 1

        # Use constraints if available
        min_val = param.constraints.min_value if param.constraints else None
        max_val = param.constraints.max_value if param.constraints else None

        if min_val is None:
            min_val = max(1, int(base_value * (1 - config.variation_percent)))
        if max_val is None:
            max_val = int(base_value * (1 + config.variation_percent))

        # Ensure max >= min
        if max_val < min_val:
            max_val = min_val * 2
            if max_val < min_val:
                max_val = min_val + 1

        return {
            "type": "int",
            "low": int(min_val),
            "high": int(max_val),
            "base_value": base_value,
        }

    def _build_bytes_search_space(
        self,
        _param_name: str,
        heuristic_value: Any,
        config: SearchSpaceConfig,
    ) -> dict[str, Any]:
        """Build search space for byte-size parameters.

        Args:
            param_name: Parameter name.
            heuristic_value: Heuristic baseline (e.g., "32k", "1m").
            config: Search space configuration.

        Returns:
            Search space definition.

        """
        base_bytes = self._parse_memory_string(str(heuristic_value))

        variation = config.variation_percent
        min_bytes = int(base_bytes * (1 - variation))
        max_bytes = int(base_bytes * (1 + variation))

        # Reasonable bounds
        min_bytes = max(1024, min_bytes)  # Min 1KB
        max_bytes = min(1024 * 1024 * 1024, max_bytes)  # Max 1GB

        # Ensure max >= min
        if max_bytes < min_bytes:
            max_bytes = min_bytes * 2
            # If still less than min_bytes, set max to min + 1024
            if max_bytes < min_bytes:
                max_bytes = min_bytes + 1024

        return {
            "type": "bytes",
            "low": min_bytes,
            "high": max_bytes,
            "base_value": base_bytes,
        }

    @staticmethod
    def _parse_memory_string(value: str) -> int:
        """Parse memory string to bytes.

        Args:
            value: Memory string (e.g., "4g", "512m", "1024k").

        Returns:
            Number of bytes.

        Raises:
            ValueError: If format is invalid.

        """
        value = str(value).strip().lower()

        # Remove trailing 'b' if present (e.g., "1b", "1kb")
        if value.endswith("b"):
            if len(value) > 1 and not value[-2].isdigit():
                # case like 'kb', 'mb' -> remove 'b' to get 'k', 'm'
                value = value[:-1]
            elif len(value) > 1 and value[-2].isdigit():
                # case like '1b' -> remove 'b' to get '1'
                value = value[:-1]

        multipliers = {
            "k": 1024,
            "m": 1024**2,
            "g": 1024**3,
            "t": 1024**4,
        }

        match = re.match(r"^([\d.]+)\s*([kmgt]?)\s*$", value)
        if not match:
            raise ValueError(f"Invalid memory format: {value}")

        number = float(match.group(1))
        unit = match.group(2)

        return int(number * multipliers.get(unit, 1))

    @staticmethod
    def _parse_duration_string(value: str) -> float:
        """Parse duration string to seconds.

        Args:
            value: Duration string (e.g., "600s", "5m", "1h").

        Returns:
            Number of seconds.

        Raises:
            ValueError: If format is invalid.

        """
        value = str(value).strip().lower()

        # Mapping of all common Spark duration units
        multipliers = {
            "ms": 0.001,
            "milli": 0.001,
            "millis": 0.001,
            "s": 1,
            "sec": 1,
            "secs": 1,
            "second": 1,
            "seconds": 1,
            "m": 60,
            "min": 60,
            "mins": 60,
            "minute": 60,
            "minutes": 60,
            "h": 3600,
            "hour": 3600,
            "hours": 3600,
            "d": 86400,
            "day": 86400,
            "days": 86400,
        }

        # Match number and unit
        match = re.match(r"^([\d.]+)\s*([a-z]*)$", value)
        if match:
            number = float(match.group(1))
            unit = match.group(2)

            if not unit:
                return float(number)

            if unit in multipliers:
                return number * multipliers[unit]

        # Try plain number
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"Invalid duration format: {value}") from e

    @staticmethod
    def _parse_boolean(value: Any) -> bool:
        """Parse value to boolean.

        Args:
            value: Value to parse.

        Returns:
            Boolean representation.

        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def get_search_space(self) -> dict[str, Any]:
        """Get the last built search space.

        Returns:
            Dictionary mapping parameter names to search definitions.

        """
        return self._search_space.copy()

    def filter_by_category(
        self,
        config_set: ConfigSet,
        category: ParameterCategory,
    ) -> dict[str, Any]:
        """Filter search space by parameter category.

        Args:
            config_set: Configuration set with parameter metadata.
            category: Category to filter by.

        Returns:
            Filtered search space.

        """
        params_in_category = set(config_set.get_parameters_by_category(category).keys())
        return {k: v for k, v in self._search_space.items() if k in params_in_category}
