# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Synthetic data generators for Spark optimization testing.

This module provides data generators for creating synthetic datasets
in various formats (Parquet, Delta, JSON, CSV, ORC) with configurable
characteristics for testing Spark configurations.
"""

from __future__ import annotations

import contextlib
import logging
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional PySpark import
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        BooleanType,
        DateType,
        DoubleType,
        IntegerType,
        MapType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    logger.warning("PySpark not available. Data generation limited.")


@dataclass
class DataGeneratorConfig:
    """Configuration for data generation.

    Attributes:
        num_rows: Number of rows to generate.
        num_partitions: Number of partitions for output.
        format: Output format (parquet, delta, json, csv, orc).
        compression: Compression codec (snappy, gzip, etc.).
        null_ratio: Ratio of null values (0-1).
        skew_factor: Data skew factor (1.0 = uniform).
        random_seed: Random seed for reproducibility.

    """

    num_rows: int = 10000
    num_partitions: int = 4
    format: str = "parquet"
    compression: str = "snappy"
    null_ratio: float = 0.05
    skew_factor: float = 1.0
    random_seed: int = 42
    columns: list[ColumnSpec] | None = None


@dataclass
class ColumnSpec:
    """Specification for a data column.

    Attributes:
        name: Column name.
        data_type: Data type (string, int, double, etc.).
        nullable: Whether column can have nulls.
        cardinality: Number of distinct values (for categorical).
        min_value: Minimum value (for numeric).
        max_value: Maximum value (for numeric).

    """

    name: str
    data_type: str = "string"
    nullable: bool = True
    cardinality: int | None = None
    min_value: float | None = None
    max_value: float | None = None


class DataGenerator:
    """Generator for synthetic datasets.

    This class creates synthetic data with specified characteristics
    for testing Spark configurations. Supports multiple formats and
    data distributions.

    Example:
        >>> generator = DataGenerator(spark)
        >>> config = DataGeneratorConfig(num_rows=100000, format="parquet")
        >>> columns = [
        ...     ColumnSpec("id", "int", min_value=1, max_value=1000000),
        ...     ColumnSpec("name", "string", cardinality=1000),
        ... ]
        >>> path = generator.generate(
        ...     output_path="./test_data",
        ...     config=config,
        ...     columns=columns
        ... )
        >>> print(f"Generated data at: {path}")

    """

    # Supported formats
    SUPPORTED_FORMATS = ["parquet", "delta", "json", "csv", "orc"]

    # Format-specific options
    FORMAT_OPTIONS = {
        "parquet": {"compression": "snappy"},
        "delta": {"overwriteSchema": "true"},
        "json": {"compression": "gzip"},
        "csv": {"header": "true", "compression": "gzip"},
        "orc": {"compression": "zlib"},
    }

    def __init__(self, spark: Any | None = None) -> None:
        """Initialize data generator.

        Args:
            spark: SparkSession (optional, will create if not provided).

        """
        self.spark = spark
        self._own_spark = False

    def __del__(self) -> None:
        """Cleanup - stop Spark if we created it."""
        if self._own_spark and self.spark is not None:
            with contextlib.suppress(Exception):
                self.spark.stop()

    def generate(
        self,
        output_path: str | Path,
        config: DataGeneratorConfig | None = None,
        columns: list[ColumnSpec] | None = None,
        schema: Any | None = None,
    ) -> Path:
        """Generate synthetic dataset.

        Args:
            output_path: Output directory path.
            config: Generation configuration.
            columns: Column specifications.
            schema: PySpark schema (alternative to columns).

        Returns:
            Path to generated data.

        Raises:
            RuntimeError: If PySpark not available.
            ValueError: If format not supported.

        """
        if not PYSPARK_AVAILABLE:
            raise RuntimeError("PySpark required for data generation")

        config = config or DataGeneratorConfig()
        output_path = Path(output_path)

        # Validate format
        if config.format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {config.format}. Use one of: {self.SUPPORTED_FORMATS}",
            )

        # Initialize Spark if needed
        if self.spark is None:
            self.spark = SparkSession.builder.appName("SparkOptimaDataGen").master("local[*]").getOrCreate()
            self._own_spark = True

        # Create schema if not provided
        if schema is None:
            if columns is None:
                columns = self._default_columns()
            schema = self._build_schema(columns)

        # Generate data
        logger.info(f"Generating {config.num_rows:,} rows in {config.num_partitions} partitions")

        # Create RDD with synthetic data
        if columns is None:
            raise ValueError("Columns must be specified")
        data_rdd = self._generate_rdd(config, columns, schema)

        # Convert to DataFrame
        df = self.spark.createDataFrame(data_rdd, schema)

        # Apply data skew if requested
        if config.skew_factor > 1.0:
            if columns is None:
                raise ValueError("Columns must be specified")
            df = self._apply_skew(df, columns, config.skew_factor)

        # Repartition
        df = df.repartition(config.num_partitions)

        # Write data
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = df.write.mode("overwrite")

        # Apply format-specific options
        options = self.FORMAT_OPTIONS.get(config.format, {})
        if config.compression:
            options["compression"] = config.compression

        for key, value in options.items():
            writer = writer.option(key, value)

        # Write based on format
        if config.format == "delta":
            writer.format("delta").save(str(output_path))
        elif config.format == "csv":
            writer.option("header", "true").csv(str(output_path))
        else:
            writer.format(config.format).save(str(output_path))

        logger.info(f"Data generated at: {output_path}")

        return output_path

    def generate_tabular(
        self,
        output_path: str | Path,
        num_rows: int = 10000,
        num_cols: int = 10,
        **kwargs: Any,
    ) -> Path:
        """Generate simple tabular data.

        Args:
            output_path: Output directory.
            num_rows: Number of rows.
            num_cols: Number of columns.
            **kwargs: Additional options for DataGeneratorConfig.

        Returns:
            Path to generated data.

        """
        # Create column specs
        columns = []
        for i in range(num_cols):
            if i == 0:
                # ID column
                col = ColumnSpec(
                    name="id",
                    data_type="int",
                    min_value=1,
                    max_value=num_rows * 10,
                    nullable=False,
                )
            elif i == 1:
                # Category column
                col = ColumnSpec(
                    name="category",
                    data_type="string",
                    cardinality=10,
                )
            elif i < num_cols // 2:
                # String columns
                col = ColumnSpec(
                    name=f"str_col_{i}",
                    data_type="string",
                    cardinality=1000,
                )
            else:
                # Numeric columns
                col = ColumnSpec(
                    name=f"num_col_{i}",
                    data_type="double",
                    min_value=0.0,
                    max_value=1000.0,
                )
            columns.append(col)

        config = DataGeneratorConfig(num_rows=num_rows, **kwargs)

        return self.generate(output_path, config, columns)

    def generate_nested(
        self,
        output_path: str | Path,
        num_rows: int = 10000,
        **kwargs: Any,
    ) -> Path:
        """Generate nested JSON data.

        Args:
            output_path: Output directory.
            num_rows: Number of rows.
            **kwargs: Additional options.

        Returns:
            Path to generated data.

        """
        if not PYSPARK_AVAILABLE:
            raise RuntimeError("PySpark required for data generation")

        config = DataGeneratorConfig(num_rows=num_rows, format="json", **kwargs)

        # Create nested schema
        from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

        inner_schema = StructType(
            [
                StructField("key", StringType(), True),
                StructField("value", IntegerType(), True),
            ],
        )

        schema = StructType(
            [
                StructField("id", IntegerType(), False),
                StructField("name", StringType(), True),
                StructField("items", ArrayType(inner_schema), True),
                StructField("metadata", MapType(StringType(), StringType()), True),
            ],
        )

        return self.generate(output_path, config, schema=schema)

    def _default_columns(self) -> list[ColumnSpec]:
        """Get default column specifications.

        Returns:
            List of default columns.

        """
        return [
            ColumnSpec("id", "int", min_value=1, max_value=1000000, nullable=False),
            ColumnSpec("name", "string", cardinality=10000),
            ColumnSpec("age", "int", min_value=18, max_value=100),
            ColumnSpec("salary", "double", min_value=30000, max_value=200000),
            ColumnSpec("department", "string", cardinality=10),
            ColumnSpec("hire_date", "date"),
            ColumnSpec("is_active", "boolean"),
        ]

    def _build_schema(self, columns: list[ColumnSpec]) -> Any:
        """Build PySpark schema from column specs.

        Args:
            columns: Column specifications.

        Returns:
            PySpark StructType schema.

        """
        if not PYSPARK_AVAILABLE:
            return None

        type_mapping = {
            "string": StringType(),
            "int": IntegerType(),
            "integer": IntegerType(),
            "double": DoubleType(),
            "float": DoubleType(),
            "boolean": BooleanType(),
            "bool": BooleanType(),
            "date": DateType(),
            "timestamp": TimestampType(),
        }

        fields = []
        for col in columns:
            spark_type = type_mapping.get(col.data_type.lower(), StringType())
            field = StructField(col.name, spark_type, col.nullable)
            fields.append(field)

        return StructType(fields)

    def _generate_rdd(
        self,
        config: DataGeneratorConfig,
        columns: list[ColumnSpec],
        _schema: Any,
    ) -> Any:
        """Generate RDD with synthetic data.

        Args:
            config: Generator configuration.
            columns: Column specifications.
            schema: PySpark schema.

        Returns:
            RDD with generated data.

        """
        if self.spark is None:
            raise ValueError("Spark session not initialized")
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        def generate_row(_row_id: int) -> tuple[Any, ...]:
            """Generate a single row."""
            row = []
            for col in columns:
                value = self._generate_value(col, config)
                row.append(value)
            return tuple(row)

        # Create RDD
        rdd = self.spark.sparkContext.parallelize(
            range(config.num_rows),
            config.num_partitions,
        ).map(generate_row)

        return rdd

    def _generate_value(
        self,
        column: ColumnSpec,
        config: DataGeneratorConfig,
    ) -> Any:
        """Generate a single value for a column.

        Args:
            column: Column specification.
            config: Generator configuration.

        Returns:
            Generated value.

        """
        # Check for null
        if column.nullable and random.random() < config.null_ratio:  # nosec B311 - pseudorandom used for data generation, not crypto
            return None

        # Generate based on type
        if column.data_type in ("int", "integer"):
            min_val_int = int(column.min_value or 0)
            max_val_int = int(column.max_value or 1000000)
            return random.randint(min_val_int, max_val_int)  # nosec B311

        elif column.data_type in ("double", "float"):
            min_val_float = float(column.min_value) if column.min_value is not None else 0.0
            max_val_float = float(column.max_value) if column.max_value is not None else 1.0
            return random.uniform(min_val_float, max_val_float)  # nosec B311

        elif column.data_type == "boolean":
            return random.choice([True, False])  # nosec B311

        elif column.data_type == "date":
            from datetime import datetime, timedelta

            days = random.randint(0, 3650)  # ~10 years  # nosec B311
            return datetime.now() - timedelta(days=days)

        elif column.data_type == "timestamp":
            from datetime import datetime, timedelta

            seconds = random.randint(0, 315360000)  # ~10 years  # nosec B311
            return datetime.now() - timedelta(seconds=seconds)

        elif column.cardinality:
            # Categorical string
            return f"category_{random.randint(1, column.cardinality)}"  # nosec B311
        else:
            # Random string
            length = random.randint(5, 20)  # nosec B311
            return "".join(random.choices(string.ascii_letters, k=length))  # nosec B311

    def _apply_skew(
        self,
        df: Any,
        columns: list[ColumnSpec],
        _skew_factor: float,
    ) -> Any:
        """Apply data skew to DataFrame.

        Args:
            df: DataFrame to modify.
            columns: Column specifications.
            skew_factor: Skew intensity.

        Returns:
            Modified DataFrame.

        """
        # Find categorical columns to skew
        categorical_cols = [col.name for col in columns if col.data_type == "string" and col.cardinality]

        if not categorical_cols:
            return df

        # Apply skew to first categorical column
        skew_col = categorical_cols[0]

        # Create skewed distribution
        from pyspark.sql.functions import lit, rand, when

        # Make 80% of values the same (hot key)
        skewed_df = df.withColumn(
            skew_col,
            when(rand() < 0.8, lit("HOT_KEY")).otherwise(df[skew_col]),
        )

        return skewed_df

    def get_supported_formats(self) -> list[str]:
        """Get list of supported data formats.

        Returns:
            List of format names.

        """
        return self.SUPPORTED_FORMATS.copy()

    def estimate_size(
        self,
        num_rows: int,
        num_cols: int,
        avg_col_size_bytes: int = 20,
        format: str = "parquet",
        compression: str = "snappy",
    ) -> dict[str, float]:
        """Estimate generated data size.

        Args:
            num_rows: Number of rows.
            num_cols: Number of columns.
            avg_col_size_bytes: Average column size.
            format: Data format.
            compression: Compression codec.

        Returns:
            Dictionary with size estimates.

        """
        # Raw size
        raw_bytes = num_rows * num_cols * avg_col_size_bytes

        # Format compression ratios
        compression_ratios = {
            "none": 1.0,
            "snappy": 0.7,
            "gzip": 0.5,
            "lz4": 0.75,
            "zstd": 0.55,
            "zlib": 0.6,
        }

        format_overhead = {
            "parquet": 1.0,
            "delta": 1.1,
            "json": 1.5,  # JSON is verbose
            "csv": 1.2,
            "orc": 0.9,
        }

        compression_ratio = compression_ratios.get(compression, 0.7)
        overhead = format_overhead.get(format, 1.0)

        estimated_bytes = raw_bytes * overhead * compression_ratio

        return {
            "raw_gb": raw_bytes / (1024**3),
            "estimated_gb": estimated_bytes / (1024**3),
            "rows": num_rows,
            "columns": num_cols,
        }
