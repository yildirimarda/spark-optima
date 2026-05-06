# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Data profiling utilities for Spark datasets.

This module provides data profiling capabilities to analyze dataset
characteristics including statistics, distributions, and schema information.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# Optional PySpark import

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import avg, col, stddev
    from pyspark.sql.functions import max as spark_max
    from pyspark.sql.functions import min as spark_min

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import avg, col, stddev
    from pyspark.sql.functions import max as spark_max
    from pyspark.sql.functions import min as spark_min

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


@dataclass
class ColumnProfile:
    """Profile for a single column.

    Attributes:
        name: Column name.
        data_type: Data type.
        nullable: Whether column allows nulls.
        null_count: Number of null values.
        null_ratio: Ratio of null values.
        distinct_count: Number of distinct values.
        min_value: Minimum value (for numeric).
        max_value: Maximum value (for numeric).
        mean: Mean value (for numeric).
        std_dev: Standard deviation (for numeric).
        top_values: Most frequent values.

    """

    name: str
    data_type: str = ""
    nullable: bool = True
    null_count: int = 0
    null_ratio: float = 0.0
    distinct_count: int = 0
    min_value: Any = None
    max_value: Any = None
    mean: float | None = None
    std_dev: float | None = None
    top_values: list[tuple[Any, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "null_count": self.null_count,
            "null_ratio": self.null_ratio,
            "distinct_count": self.distinct_count,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "top_values": self.top_values[:5],  # Limit top values
        }


@dataclass
class DataProfile:
    """Complete profile of a dataset.

    Attributes:
        path: Path to dataset.
        format: Data format.
        num_rows: Number of rows.
        num_columns: Number of columns.
        size_bytes: Size in bytes.
        num_partitions: Number of partitions.
        columns: List of column profiles.
        schema: Schema string.

    """

    path: str = ""
    format: str = ""
    num_rows: int = 0
    num_columns: int = 0
    size_bytes: int = 0
    num_partitions: int = 0
    columns: list[ColumnProfile] = field(default_factory=list)
    schema: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "format": self.format,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "size_bytes": self.size_bytes,
            "size_mb": self.size_bytes / (1024 * 1024),
            "num_partitions": self.num_partitions,
            "columns": [c.to_dict() for c in self.columns],
            "schema": self.schema,
        }

    def get_column_names(self) -> list[str]:
        """Get list of column names."""
        return [c.name for c in self.columns]

    def get_column_profile(self, name: str) -> ColumnProfile | None:
        """Get profile for specific column."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def get_numeric_columns(self) -> list[ColumnProfile]:
        """Get numeric columns."""
        numeric_types = ("int", "integer", "long", "double", "float", "decimal")
        return [c for c in self.columns if any(t in c.data_type.lower() for t in numeric_types)]

    def get_categorical_columns(self) -> list[ColumnProfile]:
        """Get categorical (low cardinality) columns."""
        return [c for c in self.columns if c.distinct_count > 0 and c.distinct_count < 100]


class DataProfiler:
    """Profiler for analyzing Spark datasets.

    This class analyzes datasets to extract statistics, distributions,
    and schema information for optimization purposes.

    Example:
        >>> profiler = DataProfiler(spark)
        >>> profile = profiler.profile("./data.parquet")
        >>> print(f"Rows: {profile.num_rows}, Size: {profile.size_bytes} bytes")
        >>> for col in profile.columns:
        ...     print(f"{col.name}: {col.distinct_count} distinct values")

    """

    def __init__(self, spark: Any | None = None) -> None:
        """Initialize data profiler.

        Args:
            spark: SparkSession (optional).

        """
        self.spark = spark
        self._own_spark = False

    def __del__(self) -> None:
        """Cleanup."""
        if self._own_spark and self.spark is not None:
            with contextlib.suppress(Exception):
                self.spark.stop()

    def profile(
        self,
        data_path: str | Path,
        format: str | None = None,
        sample_ratio: float = 1.0,
    ) -> DataProfile:
        """Profile a dataset.

        Args:
            data_path: Path to dataset.
            format: Data format (inferred if not provided).
            sample_ratio: Fraction of data to sample (for large datasets).

        Returns:
            DataProfile with statistics.

        """
        if not PYSPARK_AVAILABLE:
            raise RuntimeError("PySpark required for data profiling")

        # Initialize Spark if needed
        if self.spark is None:
            self.spark = (
                SparkSession.builder.appName("SparkOptimaProfiler").master("local[*]").getOrCreate()
            )
            self._own_spark = True

        data_path = Path(data_path)

        # Infer format if not provided
        if format is None:
            format = self._infer_format(data_path)

        # Load data
        df = self._load_data(data_path, format)

        if sample_ratio < 1.0:
            df = df.sample(False, sample_ratio)

        # Get basic info
        num_rows = df.count()
        num_partitions = df.rdd.getNumPartitions()

        # Get schema
        schema = df.schema
        schema_str = schema.simpleString()

        # Profile each column
        columns = []
        for f in schema.fields:
            col_profile = self._profile_column(df, f)
            columns.append(col_profile)

        # Estimate size
        size_bytes = self._estimate_size(data_path, format)

        return DataProfile(
            path=str(data_path),
            format=format,
            num_rows=num_rows,
            num_columns=len(columns),
            size_bytes=size_bytes,
            num_partitions=num_partitions,
            columns=columns,
            schema=schema_str,
        )

    def profile_quick(
        self,
        data_path: str | Path,
        format: str | None = None,
    ) -> dict[str, Any]:
        """Quick profile with basic statistics only.

        Args:
            data_path: Path to dataset.
            format: Data format.

        Returns:
            Dictionary with basic info.

        """
        if not PYSPARK_AVAILABLE:
            raise RuntimeError("PySpark required")

        if self.spark is None:
            self.spark = (
                SparkSession.builder.appName("SparkOptimaProfiler").master("local[*]").getOrCreate()
            )
            self._own_spark = True

        data_path = Path(data_path)
        format = format or self._infer_format(data_path)

        df = self._load_data(data_path, format)

        return {
            "path": str(data_path),
            "format": format,
            "num_rows": df.count(),
            "num_columns": len(df.columns),
            "columns": df.columns,
            "num_partitions": df.rdd.getNumPartitions(),
        }

    def _load_data(self, data_path: Path, format: str) -> Any:
        """Load data from path."""
        if self.spark is None:
            raise ValueError("Spark session not initialized")
        reader = self.spark.read

        if format == "csv":
            return reader.option("header", "true").option("inferSchema", "true").csv(str(data_path))
        elif format == "json":
            return reader.json(str(data_path))
        elif format == "parquet":
            return reader.parquet(str(data_path))
        elif format == "delta":
            return reader.format("delta").load(str(data_path))
        elif format == "orc":
            return reader.orc(str(data_path))
        else:
            # Try generic load
            return reader.format(format).load(str(data_path))

    def _profile_column(self, df: Any, field: Any) -> ColumnProfile:
        """Profile a single column."""
        col_name = field.name
        col_type = field.dataType.simpleString()
        nullable = field.nullable

        # Count nulls
        null_count = df.filter(col(col_name).isNull()).count()
        total_count = df.count()
        null_ratio = null_count / total_count if total_count > 0 else 0.0

        # Count distinct
        distinct_count = df.select(col_name).distinct().count()

        profile = ColumnProfile(
            name=col_name,
            data_type=col_type,
            nullable=nullable,
            null_count=null_count,
            null_ratio=null_ratio,
            distinct_count=distinct_count,
        )

        # Numeric statistics
        if self._is_numeric_type(col_type):
            stats = df.select(
                spark_min(col_name).alias("min"),
                spark_max(col_name).alias("max"),
                avg(col_name).alias("mean"),
                stddev(col_name).alias("stddev"),
            ).collect()[0]

            profile.min_value = stats["min"]
            profile.max_value = stats["max"]
            profile.mean = stats["mean"]
            profile.std_dev = stats["stddev"]

        return profile

    def _is_numeric_type(self, data_type: str) -> bool:
        """Check if type is numeric."""
        numeric_types = ("int", "integer", "long", "bigint", "double", "float", "decimal")
        return any(t in data_type.lower() for t in numeric_types)

    def _infer_format(self, data_path: Path) -> str:
        """Infer data format from path."""
        suffix = data_path.suffix.lower()

        format_map = {
            ".parquet": "parquet",
            ".csv": "csv",
            ".json": "json",
            ".orc": "orc",
        }

        return format_map.get(suffix, "parquet")

    def _estimate_size(self, _data_path: Path, _format: str) -> int:
        """Estimate data size in bytes."""
        try:
            if _data_path.is_file():
                return _data_path.stat().st_size
            elif _data_path.is_dir():
                total = 0
                for file in _data_path.rglob("*"):
                    if file.is_file():
                        total += file.stat().st_size
                return total
        except (OSError, FileNotFoundError, PermissionError, ValueError) as e:
            logger.warning(f"Could not estimate size: {e}")

        return 0

    def analyze_skew(self, data_path: str | Path, column: str) -> dict[str, Any]:
        """Analyze data skew for a column.

        Args:
            data_path: Path to dataset.
            column: Column to analyze.

        Returns:
            Skew analysis results.

        """
        if not PYSPARK_AVAILABLE:
            raise RuntimeError("PySpark required")

        if self.spark is None:
            self.spark = (
                SparkSession.builder.appName("SparkOptimaProfiler").master("local[*]").getOrCreate()
            )
            self._own_spark = True

        df = self._load_data(Path(data_path), self._infer_format(Path(data_path)))

        # Get value distribution
        value_counts = df.groupBy(column).count().orderBy(col("count").desc())

        # Calculate skew metrics
        total_rows = df.count()
        top_values = value_counts.limit(10).collect()

        if top_values:
            max_count = top_values[0]["count"]
            skew_ratio = max_count / (total_rows / len(top_values)) if len(top_values) > 0 else 1.0
        else:
            skew_ratio = 1.0

        return {
            "column": column,
            "total_rows": total_rows,
            "distinct_values": value_counts.count(),
            "top_values": [(row[column], row["count"]) for row in top_values],
            "skew_ratio": skew_ratio,
            "is_skewed": skew_ratio > 2.0,
        }
