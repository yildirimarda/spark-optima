# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Data sampling strategies for Spark datasets.

This module provides various sampling strategies for extracting representative
subsets of data for testing and profiling purposes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# Optional PySpark import

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, rand, row_number
    from pyspark.sql.window import Window

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, rand, row_number
    from pyspark.sql.window import Window

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


@dataclass
class SampleConfig:
    """Configuration for data sampling.

    Attributes:
        sample_size: Number of rows to sample (or ratio if < 1).
        strategy: Sampling strategy (random, stratified, reservoir).
        seed: Random seed for reproducibility.
        columns: Columns to include (None = all).

    """

    sample_size: float = 0.1
    strategy: str = "random"
    seed: int = 42
    columns: list[str] | None = None


class Sampler(ABC):
    """Abstract base class for data samplers."""

    @abstractmethod
    def sample(self, data_path: str | Path, output_path: str | Path, config: SampleConfig) -> Path:
        """Sample data from source to destination.

        Args:
            data_path: Source data path.
            output_path: Output path for sample.
            config: Sampling configuration.

        Returns:
            Path to sampled data.

        """


class RandomSampler(Sampler):
    """Random sampling strategy.

    Samples rows randomly with uniform probability.
    """

    def __init__(self, spark: Any | None = None) -> None:
        """Initialize sampler.

        Args:
            spark: SparkSession (optional).

        """
        self.spark = spark
        self._own_spark = False

    def sample(self, data_path: str | Path, output_path: str | Path, config: SampleConfig) -> Path:
        """Sample data using random sampling."""
        if not PYSPARK_AVAILABLE:
            raise RuntimeError("PySpark required")

        if self.spark is None:
            self.spark = SparkSession.builder.appName("SparkOptimaSampler").master("local[*]").getOrCreate()
            self._own_spark = True

        data_path = Path(data_path)
        output_path = Path(output_path)

        # Load data
        df = self._load_data(data_path)

        # Select columns if specified
        if config.columns:
            df = df.select(*config.columns)

        # Apply random sampling
        if config.sample_size < 1.0:
            # Fraction sampling
            sample_df = df.sample(False, config.sample_size, config.seed)
        else:
            # Fixed size sampling
            total = df.count()
            fraction = min(config.sample_size / total, 1.0)
            sample_df = df.sample(False, fraction, config.seed)
            sample_df = sample_df.limit(int(config.sample_size))

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.write.mode("overwrite").parquet(str(output_path))

        logger.info(f"Random sample saved to: {output_path}")
        return output_path

    def _load_data(self, data_path: Path) -> Any:
        """Load data from path."""
        if self.spark is None:
            raise ValueError("Spark session not initialized")
        # Infer format
        suffix = data_path.suffix.lower()
        if suffix == ".csv":
            return self.spark.read.option("header", "true").csv(str(data_path))
        elif suffix == ".json":
            return self.spark.read.json(str(data_path))
        else:
            return self.spark.read.parquet(str(data_path))


class StratifiedSampler(Sampler):
    """Stratified sampling strategy.

    Samples preserving the distribution of a stratification column.
    """

    def __init__(self, spark: Any | None = None) -> None:
        """Initialize sampler."""
        self.spark = spark
        self._own_spark = False

    def sample(
        self,
        data_path: str | Path,
        output_path: str | Path,
        config: SampleConfig,
        stratify_column: str = "",
    ) -> Path:
        """Sample data using stratified sampling.

        Args:
            data_path: Source data path.
            output_path: Output path.
            config: Sampling configuration.
            stratify_column: Column to stratify by.

        Returns:
            Path to sampled data.

        """
        if not PYSPARK_AVAILABLE:
            raise RuntimeError("PySpark required")

        if not stratify_column:
            raise ValueError("stratify_column required for stratified sampling")

        if self.spark is None:
            self.spark = SparkSession.builder.appName("SparkOptimaSampler").master("local[*]").getOrCreate()
            self._own_spark = True

        data_path = Path(data_path)
        output_path = Path(output_path)

        # Load data
        df = self._load_data(data_path)

        # Get strata counts
        strata_counts = df.groupBy(stratify_column).count().collect()
        total_count = sum(row["count"] for row in strata_counts)

        # Calculate sample size per stratum
        sample_dfs = []
        for row in strata_counts:
            stratum_value = row[stratify_column]
            stratum_count = row["count"]

            # Calculate sample size for this stratum
            if config.sample_size < 1.0:
                stratum_sample_size = int(stratum_count * config.sample_size)
            else:
                stratum_sample_size = int(config.sample_size * (stratum_count / total_count))

            # Sample from stratum
            stratum_df = df.filter(col(stratify_column) == stratum_value)
            fraction = min(stratum_sample_size / stratum_count, 1.0)
            sampled = stratum_df.sample(False, fraction, config.seed)
            sampled = sampled.limit(stratum_sample_size)

            sample_dfs.append(sampled)

        # Union all samples
        if sample_dfs:
            sample_df = sample_dfs[0]
            for df in sample_dfs[1:]:
                sample_df = sample_df.union(df)
        else:
            sample_df = df.limit(0)  # Empty DataFrame

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.write.mode("overwrite").parquet(str(output_path))

        logger.info(f"Stratified sample saved to: {output_path}")
        return output_path

    def _load_data(self, data_path: Path) -> Any:
        """Load data from path."""
        if self.spark is None:
            raise ValueError("Spark session not initialized")
        suffix = data_path.suffix.lower()
        if suffix == ".csv":
            return self.spark.read.option("header", "true").csv(str(data_path))
        elif suffix == ".json":
            return self.spark.read.json(str(data_path))
        else:
            return self.spark.read.parquet(str(data_path))


class ReservoirSampler(Sampler):
    """Reservoir sampling strategy.

    Samples a fixed number of rows using reservoir sampling algorithm.
    Suitable for streaming or very large datasets.
    """

    def __init__(self, spark: Any | None = None) -> None:
        """Initialize sampler."""
        self.spark = spark
        self._own_spark = False

    def sample(self, data_path: str | Path, output_path: str | Path, config: SampleConfig) -> Path:
        """Sample data using reservoir sampling.

        For large datasets, this uses a two-pass approach:
        1. First pass: count total rows
        2. Second pass: select rows with probability k/n

        """
        if not PYSPARK_AVAILABLE:
            raise RuntimeError("PySpark required")

        if config.sample_size >= 1.0:
            sample_size = int(config.sample_size)
        else:
            raise ValueError("Reservoir sampling requires sample_size >= 1 (absolute count)")

        if self.spark is None:
            self.spark = SparkSession.builder.appName("SparkOptimaSampler").master("local[*]").getOrCreate()
            self._own_spark = True

        data_path = Path(data_path)
        output_path = Path(output_path)

        # Load data
        df = self._load_data(data_path)

        # Count total rows
        total_count = df.count()

        if total_count <= sample_size:
            # Use all data if smaller than sample size
            sample_df = df
        else:
            # Reservoir sampling via random selection
            # Add random column and select top k
            df_with_rand = df.withColumn("_rand", rand(config.seed))

            # Use window function to get top k
            window = Window.orderBy(col("_rand"))
            df_ranked = df_with_rand.withColumn("_rank", row_number().over(window))
            sample_df = df_ranked.filter(col("_rank") <= sample_size).drop("_rand", "_rank")

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.write.mode("overwrite").parquet(str(output_path))

        logger.info(f"Reservoir sample ({sample_size} rows) saved to: {output_path}")
        return output_path

    def _load_data(self, data_path: Path) -> Any:
        """Load data from path."""
        if self.spark is None:
            raise ValueError("Spark session not initialized")
        suffix = data_path.suffix.lower()
        if suffix == ".csv":
            return self.spark.read.option("header", "true").csv(str(data_path))
        elif suffix == ".json":
            return self.spark.read.json(str(data_path))
        else:
            return self.spark.read.parquet(str(data_path))


class DataSampler:
    """Factory class for data sampling operations.

    Provides a unified interface for different sampling strategies.

    Example:
        >>> sampler = DataSampler(spark)
        >>> config = SampleConfig(sample_size=1000, strategy="random")
        >>> sample_path = sampler.sample("./data.parquet", "./sample.parquet", config)

    """

    STRATEGIES: dict[str, type[Sampler]] = {
        "random": RandomSampler,
        "stratified": StratifiedSampler,
        "reservoir": ReservoirSampler,
    }

    def __init__(self, spark: Any | None = None) -> None:
        """Initialize sampler factory.

        Args:
            spark: SparkSession (optional).

        """
        self.spark = spark
        self._samplers: dict[str, Sampler] = {}

    def sample(
        self,
        data_path: str | Path,
        output_path: str | Path,
        config: SampleConfig,
        stratify_column: str = "",
    ) -> Path:
        """Sample data using specified strategy.

        Args:
            data_path: Source data path.
            output_path: Output path.
            config: Sampling configuration.
            stratify_column: Column for stratified sampling.

        Returns:
            Path to sampled data.

        """
        strategy = config.strategy.lower()

        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. Use one of: {list(self.STRATEGIES.keys())}",
            )

        # Get or create sampler
        if strategy not in self._samplers:
            self._samplers[strategy] = self.STRATEGIES[strategy](self.spark)  # type: ignore[call-arg]

        sampler = self._samplers[strategy]

        # Call appropriate method
        if strategy == "stratified":
            return sampler.sample(data_path, output_path, config, stratify_column)  # type: ignore[call-arg]
        else:
            return sampler.sample(data_path, output_path, config)

    def get_supported_strategies(self) -> list[str]:
        """Get list of supported sampling strategies."""
        return list(self.STRATEGIES.keys())

    def estimate_sample_size(
        self,
        total_rows: int,
        confidence: float = 0.95,
        margin_error: float = 0.05,
    ) -> int:
        """Estimate required sample size using statistical formula.

        Uses the formula for sample size with finite population correction:
        n = (Z^2 * p * (1-p)) / (e^2)

        where:
        - Z is the Z-score for confidence level
        - p is the estimated proportion (0.5 for maximum variance)
        - e is the margin of error

        Args:
            total_rows: Total population size.
            confidence: Confidence level (0.9, 0.95, 0.99).
            margin_error: Acceptable margin of error.

        Returns:
            Recommended sample size.

        """
        # Z-scores for common confidence levels
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)

        # Use p = 0.5 for maximum variance
        p = 0.5

        # Calculate sample size
        n = (z**2 * p * (1 - p)) / (margin_error**2)

        # Apply finite population correction
        if total_rows > 0:
            n = n / (1 + (n - 1) / total_rows)

        return int(min(n, total_rows))
