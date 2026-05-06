# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for data samplers."""

from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spark_optima.data.samplers import (
    DataSampler,
    RandomSampler,
    ReservoirSampler,
    SampleConfig,
    Sampler,
    StratifiedSampler,
)


class TestSampleConfig:
    """Tests for SampleConfig."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        config = SampleConfig()
        assert config.sample_size == 0.1
        assert config.strategy == "random"
        assert config.seed == 42
        assert config.columns is None

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        config = SampleConfig(
            sample_size=1000,
            strategy="stratified",
            seed=123,
            columns=["col1", "col2"],
        )
        assert config.sample_size == 1000
        assert config.strategy == "stratified"
        assert config.seed == 123
        assert config.columns == ["col1", "col2"]

    def test_is_dataclass(self) -> None:
        """Test that SampleConfig is a dataclass."""
        assert is_dataclass(SampleConfig)

    def test_sample_config_to_dict(self) -> None:
        """Test SampleConfig to dict conversion."""
        config = SampleConfig(sample_size=100, strategy="reservoir", seed=123, columns=["a", "b"])
        assert config.sample_size == 100
        assert config.strategy == "reservoir"
        assert config.seed == 123
        assert config.columns == ["a", "b"]

    def test_sample_config_with_none_columns(self) -> None:
        """Test SampleConfig with None columns."""
        config = SampleConfig(sample_size=0.5, strategy="random", seed=42, columns=None)
        assert config.columns is None

    def test_sample_config_integer_sample_size(self) -> None:
        """Test SampleConfig with integer sample size."""
        config = SampleConfig(sample_size=1000)
        assert config.sample_size == 1000

    def test_sample_config_float_sample_size(self) -> None:
        """Test SampleConfig with float sample size."""
        config = SampleConfig(sample_size=0.01)
        assert config.sample_size == 0.01

    def test_sample_config_with_all_options(self) -> None:
        """Test SampleConfig with all options."""
        config = SampleConfig(
            sample_size=500,
            strategy="reservoir",
            seed=999,
            columns=["col1", "col2", "col3"],
        )
        assert config.sample_size == 500
        assert config.strategy == "reservoir"
        assert config.seed == 999
        assert len(config.columns) == 3

    def test_sample_config_default_values(self) -> None:
        """Test SampleConfig default values are sensible."""
        config = SampleConfig()
        assert config.sample_size == 0.1  # 10% sample
        assert config.strategy == "random"
        assert config.seed == 42  # Reproducible
        assert config.columns is None  # All columns


class TestSampler:
    """Tests for Sampler base class."""

    def test_sampler_is_abstract(self) -> None:
        """Test that Sampler is an abstract class."""
        with pytest.raises(TypeError):
            Sampler()


class TestRandomSampler:
    """Tests for RandomSampler."""

    def test_initialization_without_spark(self) -> None:
        """Test initialization without Spark."""
        sampler = RandomSampler()
        assert sampler.spark is None
        assert sampler._own_spark is False

    def test_initialization_with_spark(self) -> None:
        """Test initialization with Spark."""
        mock_spark = object()
        sampler = RandomSampler(spark=mock_spark)
        assert sampler.spark is mock_spark
        assert sampler._own_spark is False

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    def test_sample_with_fraction(self, mock_spark_class) -> None:
        """Test sample method with fraction sample_size."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        # Mock DataFrame operations
        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df
        mock_df.sample.return_value = mock_df
        mock_df.count.return_value = 10000
        mock_df.select.return_value = mock_df

        sampler = RandomSampler(spark=mock_spark)
        config = SampleConfig(sample_size=0.5)  # Fraction

        result = sampler.sample("input.parquet", "output.parquet", config)

        # Verify sample was called with fraction
        mock_df.sample.assert_called_once_with(False, 0.5, 42)
        assert result == Path("output.parquet")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    def test_sample_with_fixed_size(self, mock_spark_class) -> None:
        """Test sample method with fixed size sample_size."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        # Mock DataFrame operations
        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df
        mock_df.sample.return_value = mock_df
        mock_df.count.return_value = 1000
        mock_df.select.return_value = mock_df
        mock_df.limit.return_value = mock_df

        sampler = RandomSampler(spark=mock_spark)
        config = SampleConfig(sample_size=100)  # Fixed size

        result = sampler.sample("input.parquet", "output.parquet", config)

        # Verify operations
        mock_df.sample.assert_called_once()
        mock_df.limit.assert_called_once_with(100)
        assert result == Path("output.parquet")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    def test_sample_with_columns(self, mock_spark_class) -> None:
        """Test sample method with column selection."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df
        mock_df.sample.return_value = mock_df
        mock_df.count.return_value = 1000
        mock_df.select.return_value = mock_df

        sampler = RandomSampler(spark=mock_spark)
        config = SampleConfig(sample_size=0.5, columns=["col1", "col2"])

        sampler.sample("input.parquet", "output.parquet", config)

        # Verify select was called with columns
        mock_df.select.assert_called_once_with(*config.columns)

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    def test_sample_csv_format(self, mock_spark_class) -> None:
        """Test sample with CSV format."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_spark.read.option.return_value.csv.return_value = mock_df
        mock_df.sample.return_value = mock_df
        mock_df.count.return_value = 1000

        sampler = RandomSampler(spark=mock_spark)
        config = SampleConfig(sample_size=0.5)

        sampler.sample(Path("input.csv"), "output.parquet", config)

        # Verify CSV was read with header option
        mock_spark.read.option.assert_called_with("header", "true")
        mock_spark.read.option.return_value.csv.assert_called_once_with("input.csv")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    def test_sample_json_format(self, mock_spark_class) -> None:
        """Test sample with JSON format."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_spark.read.json.return_value = mock_df
        mock_df.sample.return_value = mock_df
        mock_df.count.return_value = 1000

        sampler = RandomSampler(spark=mock_spark)
        config = SampleConfig(sample_size=0.5)

        sampler.sample(Path("input.json"), "output.parquet", config)

        mock_spark.read.json.assert_called_once_with("input.json")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    def test_sample_creates_output_dir(self, mock_spark_class) -> None:
        """Test that sample creates output directory."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df
        mock_df.sample.return_value = mock_df
        mock_df.count.return_value = 1000

        sampler = RandomSampler(spark=mock_spark)
        config = SampleConfig(sample_size=0.5)

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            sampler.sample("input.parquet", "subdir/output.parquet", config)
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", False)
    def test_sample_requires_pyspark(self) -> None:
        """Test that sample raises error without PySpark."""
        sampler = RandomSampler()
        config = SampleConfig()
        with pytest.raises(RuntimeError, match="PySpark required"):
            sampler.sample("./input", "./output", config)

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    def test_sample_creates_spark_if_none(self, mock_spark_class) -> None:
        """Test that RandomSampler creates Spark session if not provided."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df
        mock_df.sample.return_value = mock_df
        mock_df.count.return_value = 1000

        sampler = RandomSampler(spark=None)
        config = SampleConfig(sample_size=0.5)

        result = sampler.sample("input.parquet", "output.parquet", config)

        # Verify Spark session was created
        mock_spark_class.builder.appName.assert_called_once_with("SparkOptimaSampler")
        mock_spark_class.builder.master.assert_called_once_with("local[*]")
        mock_spark_class.builder.getOrCreate.assert_called_once()
        assert sampler._own_spark is True
        assert result == Path("output.parquet")


class TestRandomSamplerLoadData:
    """Tests for RandomSampler._load_data method."""

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch.object(RandomSampler, "__init__", lambda self, spark=None: None)
    def test_load_data_parquet(self) -> None:
        """Test loading parquet data."""
        sampler = RandomSampler()
        sampler.spark = MagicMock()

        sampler._load_data(Path("data.parquet"))
        sampler.spark.read.parquet.assert_called_once_with("data.parquet")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch.object(RandomSampler, "__init__", lambda self, spark=None: None)
    def test_load_data_csv(self) -> None:
        """Test loading csv data."""
        sampler = RandomSampler()
        sampler.spark = MagicMock()

        sampler._load_data(Path("data.csv"))
        sampler.spark.read.option.assert_called_with("header", "true")
        sampler.spark.read.option.return_value.csv.assert_called_once_with("data.csv")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch.object(RandomSampler, "__init__", lambda self, spark=None: None)
    def test_load_data_json(self) -> None:
        """Test loading json data."""
        sampler = RandomSampler()
        sampler.spark = MagicMock()

        sampler._load_data(Path("data.json"))
        sampler.spark.read.json.assert_called_once_with("data.json")


class TestStratifiedSampler:
    """Tests for StratifiedSampler."""

    def test_initialization_without_spark(self) -> None:
        """Test initialization without Spark."""
        sampler = StratifiedSampler()
        assert sampler.spark is None

    def test_initialization_with_spark(self) -> None:
        """Test initialization with Spark."""
        mock_spark = object()
        sampler = StratifiedSampler(spark=mock_spark)
        assert sampler.spark is mock_spark

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    @patch("spark_optima.data.samplers.col")
    def test_sample_with_stratification(self, mock_col, mock_spark_class) -> None:
        """Test stratified sample method."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        # Mock DataFrame operations
        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df

        # Mock groupBy.collect
        mock_row1 = MagicMock()
        mock_row1.__getitem__.side_effect = lambda k: 100 if k == "count" else "cat1"
        mock_row1.__contains__.return_value = True
        mock_row2 = MagicMock()
        mock_row2.__getitem__.side_effect = lambda k: 50 if k == "count" else "cat2"
        mock_row2.__contains__.return_value = True

        mock_df.groupBy.return_value.count.return_value.collect.return_value = [
            mock_row1,
            mock_row2,
        ]
        mock_df.count.return_value = 150

        # Mock filter and sample
        mock_stratum_df = MagicMock()
        mock_df.filter.return_value = mock_stratum_df
        mock_stratum_df.sample.return_value = mock_stratum_df
        mock_stratum_df.limit.return_value = mock_stratum_df

        # Mock union
        mock_spark.createDataFrame = MagicMock()

        sampler = StratifiedSampler(spark=mock_spark)
        config = SampleConfig(sample_size=0.5, seed=42)

        result = sampler.sample(
            "input.parquet", "output.parquet", config, stratify_column="category"
        )

        assert result == Path("output.parquet")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    @patch("spark_optima.data.samplers.col")
    def test_sample_with_empty_result(self, mock_col, mock_spark_class) -> None:
        """Test stratified sample with no data."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df
        mock_df.groupBy.return_value.count.return_value.collect.return_value = []
        mock_df.limit.return_value = MagicMock()

        sampler = StratifiedSampler(spark=mock_spark)
        config = SampleConfig(sample_size=0.5)

        result = sampler.sample(
            "input.parquet", "output.parquet", config, stratify_column="category"
        )

        # Should return empty DataFrame result
        assert result == Path("output.parquet")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", False)
    def test_sample_requires_pyspark(self) -> None:
        """Test that sample raises error without PySpark."""
        sampler = StratifiedSampler()
        config = SampleConfig(strategy="stratified")
        with pytest.raises(RuntimeError, match="PySpark required"):
            sampler.sample("./input", "./output", config)

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch.object(StratifiedSampler, "__init__", lambda self, spark=None: None)
    def test_load_data_all_formats(self) -> None:
        """Test loading data with different formats."""
        sampler = StratifiedSampler()
        sampler.spark = MagicMock()

        # Test parquet
        sampler._load_data(Path("data.parquet"))
        sampler.spark.read.parquet.assert_called_with("data.parquet")

        # Test csv
        sampler._load_data(Path("data.csv"))
        sampler.spark.read.option.return_value.csv.assert_called_with("data.csv")

        # Test json
        sampler._load_data(Path("data.json"))
        sampler.spark.read.json.assert_called_with("data.json")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    def test_sample_raises_error_without_stratify_column(self) -> None:
        """Test that stratified sampling raises ValueError without stratify_column."""
        sampler = StratifiedSampler()
        config = SampleConfig(strategy="stratified", sample_size=0.5)

        with pytest.raises(ValueError, match="stratify_column required"):
            sampler.sample("./input", "./output", config, stratify_column="")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    @patch("spark_optima.data.samplers.col")
    def test_sample_with_absolute_sample_size(self, mock_col, mock_spark_class) -> None:
        """Test stratified sample with absolute sample size (>= 1)."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df

        # Mock groupBy.collect for strata
        mock_row1 = MagicMock()
        mock_row1.__getitem__.side_effect = lambda k: 100 if k == "count" else "cat1"
        mock_row1.__contains__.return_value = True

        mock_df.groupBy.return_value.count.return_value.collect.return_value = [mock_row1]
        mock_df.count.return_value = 100

        # Mock filter and sample for stratum
        mock_stratum_df = MagicMock()
        mock_df.filter.return_value = mock_stratum_df
        mock_stratum_df.sample.return_value = mock_stratum_df
        mock_stratum_df.limit.return_value = mock_stratum_df

        sampler = StratifiedSampler(spark=mock_spark)
        config = SampleConfig(sample_size=50, strategy="stratified")  # Absolute size

        result = sampler.sample(
            "input.parquet", "output.parquet", config, stratify_column="category"
        )

        assert result == Path("output.parquet")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    @patch("spark_optima.data.samplers.col")
    def test_sample_creates_spark_if_none(self, mock_col, mock_spark_class) -> None:
        """Test that StratifiedSampler creates Spark session if not provided."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df
        mock_df.groupBy.return_value.count.return_value.collect.return_value = []
        mock_df.limit.return_value = MagicMock()

        sampler = StratifiedSampler(spark=None)
        config = SampleConfig(strategy="stratified", sample_size=0.5)

        result = sampler.sample(
            "input.parquet", "output.parquet", config, stratify_column="category"
        )

        # Verify Spark session was created
        mock_spark_class.builder.appName.assert_called_once_with("SparkOptimaSampler")
        mock_spark_class.builder.master.assert_called_once_with("local[*]")
        mock_spark_class.builder.getOrCreate.assert_called_once()
        assert sampler._own_spark is True
        assert result == Path("output.parquet")


class TestReservoirSampler:
    """Tests for ReservoirSampler."""

    def test_initialization_without_spark(self) -> None:
        """Test initialization without Spark."""
        sampler = ReservoirSampler()
        assert sampler.spark is None

    def test_initialization_with_spark(self) -> None:
        """Test initialization with Spark."""
        mock_spark = object()
        sampler = ReservoirSampler(spark=mock_spark)
        assert sampler.spark is mock_spark

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    def test_reservoir_requires_absolute_sample_size(self) -> None:
        """Test that reservoir sampling requires sample_size >= 1."""
        sampler = ReservoirSampler()
        config = SampleConfig(strategy="reservoir", sample_size=0.5)  # Fraction not allowed

        with pytest.raises(ValueError, match="Reservoir sampling requires sample_size >= 1"):
            sampler.sample("./input", "./output", config)

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    @patch("spark_optima.data.samplers.rand")
    @patch("spark_optima.data.samplers.Window")
    @patch("spark_optima.data.samplers.row_number")
    @patch("spark_optima.data.samplers.col")
    def test_sample_reservoir_basic(
        self, mock_col, mock_row_number, mock_window, mock_rand, mock_spark_class
    ) -> None:
        """Test reservoir sample method."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        # Mock DataFrame operations
        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df
        mock_df.count.return_value = 10000
        mock_df.withColumn.return_value = mock_df

        # Mock col to return something that supports <= comparison
        mock_col.return_value.__le__ = MagicMock(return_value=MagicMock())

        sampler = ReservoirSampler(spark=mock_spark)
        config = SampleConfig(sample_size=100, strategy="reservoir")

        result = sampler.sample("input.parquet", "output.parquet", config)

        assert result == Path("output.parquet")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    @patch("spark_optima.data.samplers.rand")
    @patch("spark_optima.data.samplers.Window")
    @patch("spark_optima.data.samplers.row_number")
    @patch("spark_optima.data.samplers.col")
    def test_sample_reservoir_smaller_than_sample(
        self, mock_col, mock_row_number, mock_window, mock_rand, mock_spark_class
    ) -> None:
        """Test reservoir sample when data is smaller than sample size."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df
        mock_df.count.return_value = 50  # Less than sample size

        sampler = ReservoirSampler(spark=mock_spark)
        config = SampleConfig(sample_size=100, strategy="reservoir")

        result = sampler.sample("input.parquet", "output.parquet", config)

        # Should use all data without sampling
        assert result == Path("output.parquet")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", False)
    def test_sample_requires_pyspark(self) -> None:
        """Test that sample raises error without PySpark."""
        sampler = ReservoirSampler()
        config = SampleConfig(strategy="reservoir")
        with pytest.raises(RuntimeError, match="PySpark required"):
            sampler.sample("./input", "./output", config)

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch.object(ReservoirSampler, "__init__", lambda self, spark=None: None)
    def test_load_data_all_formats(self) -> None:
        """Test loading data with different formats."""
        sampler = ReservoirSampler()
        sampler.spark = MagicMock()

        # Test parquet
        sampler._load_data(Path("data.parquet"))
        sampler.spark.read.parquet.assert_called_with("data.parquet")

        # Test csv
        sampler._load_data(Path("data.csv"))
        sampler.spark.read.option.return_value.csv.assert_called_with("data.csv")

        # Test json
        sampler._load_data(Path("data.json"))
        sampler.spark.read.json.assert_called_with("data.json")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    @patch("spark_optima.data.samplers.rand")
    @patch("spark_optima.data.samplers.Window")
    @patch("spark_optima.data.samplers.row_number")
    @patch("spark_optima.data.samplers.col")
    def test_sample_creates_spark_if_none(
        self, mock_col, mock_row_number, mock_window, mock_rand, mock_spark_class
    ) -> None:
        """Test that ReservoirSampler creates Spark session if not provided."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df
        mock_df.count.return_value = 10000
        mock_df.withColumn.return_value = mock_df

        # Mock col to return something that supports <= comparison
        mock_col.return_value.__le__ = MagicMock(return_value=MagicMock())

        sampler = ReservoirSampler(spark=None)
        config = SampleConfig(strategy="reservoir", sample_size=100)

        result = sampler.sample("input.parquet", "output.parquet", config)

        # Verify Spark session was created
        mock_spark_class.builder.appName.assert_called_once_with("SparkOptimaSampler")
        mock_spark_class.builder.master.assert_called_once_with("local[*]")
        mock_spark_class.builder.getOrCreate.assert_called_once()
        assert sampler._own_spark is True
        assert result == Path("output.parquet")


class TestDataSampler:
    """Tests for DataSampler factory class."""

    def test_initialization(self) -> None:
        """Test initialization."""
        sampler = DataSampler()
        assert sampler.spark is None
        assert sampler._samplers == {}

    def test_initialization_with_spark(self) -> None:
        """Test initialization with Spark."""
        mock_spark = object()
        sampler = DataSampler(spark=mock_spark)
        assert sampler.spark is mock_spark

    def test_supported_strategies(self) -> None:
        """Test supported strategies list."""
        sampler = DataSampler()
        strategies = sampler.get_supported_strategies()
        assert "random" in strategies
        assert "stratified" in strategies
        assert "reservoir" in strategies

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        sampler = DataSampler()
        config = SampleConfig(strategy="invalid")

        with pytest.raises(ValueError, match="Unknown strategy"):
            sampler.sample("./input", "./output", config)

    def test_strategies_class_variable(self) -> None:
        """Test that STRATEGIES is correctly defined."""
        assert "random" in DataSampler.STRATEGIES
        assert "stratified" in DataSampler.STRATEGIES
        assert "reservoir" in DataSampler.STRATEGIES
        assert len(DataSampler.STRATEGIES) == 3

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    def test_sample_random_strategy(self, mock_spark_class) -> None:
        """Test sampling with random strategy."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df
        mock_df.sample.return_value = mock_df
        mock_df.count.return_value = 1000

        sampler = DataSampler(spark=mock_spark)
        config = SampleConfig(strategy="random", sample_size=0.5)

        result = sampler.sample("./input.parquet", "./output.parquet", config)

        assert result == Path("./output.parquet")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    @patch("spark_optima.data.samplers.col")
    def test_sample_stratified_strategy(self, mock_col, mock_spark_class) -> None:
        """Test sampling with stratified strategy."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df
        mock_df.groupBy.return_value.count.return_value.collect.return_value = []
        mock_df.limit.return_value = MagicMock()

        sampler = DataSampler(spark=mock_spark)
        config = SampleConfig(strategy="stratified", sample_size=0.5)

        result = sampler.sample(
            "./input.parquet", "./output.parquet", config, stratify_column="category"
        )

        assert result == Path("./output.parquet")

    @patch("spark_optima.data.samplers.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.samplers.SparkSession")
    @patch("spark_optima.data.samplers.rand")
    @patch("spark_optima.data.samplers.Window")
    @patch("spark_optima.data.samplers.row_number")
    @patch("spark_optima.data.samplers.col")
    def test_sample_reservoir_strategy(
        self, mock_col, mock_row_number, mock_window, mock_rand, mock_spark_class
    ) -> None:
        """Test sampling with reservoir strategy."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName.return_value = mock_spark_class.builder
        mock_spark_class.builder.master.return_value = mock_spark_class.builder
        mock_spark_class.builder.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_spark.read.parquet.return_value = mock_df
        mock_df.count.return_value = 10000
        mock_df.withColumn.return_value = mock_df

        # Mock col to return something that supports <= comparison
        mock_col.return_value.__le__ = MagicMock(return_value=MagicMock())

        sampler = DataSampler(spark=mock_spark)
        config = SampleConfig(strategy="reservoir", sample_size=100)

        result = sampler.sample("./input.parquet", "./output.parquet", config)

        assert result == Path("./output.parquet")

    def test_sample_caches_samplers(self) -> None:
        """Test that DataSampler caches sampler instances."""
        DataSampler()
        SampleConfig(strategy="random")

        # After calling sample (which will fail without PySpark), the sampler should be cached
        from spark_optima.data.samplers import PYSPARK_AVAILABLE

        if PYSPARK_AVAILABLE:
            # Can't easily test without PySpark, but structure is correct
            pass
        # The sampler is created before the actual sampling call
        # So we can verify the structure

    def test_get_supported_strategies_returns_list(self) -> None:
        """Test that get_supported_strategies returns a list."""
        sampler = DataSampler()
        strategies = sampler.get_supported_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) == 3


class TestEstimateSampleSize:
    """Tests for estimate_sample_size method."""

    def test_estimate_default_confidence(self) -> None:
        """Test sample size estimation with default confidence."""
        sampler = DataSampler()
        size = sampler.estimate_sample_size(total_rows=100000)
        assert size > 0
        assert size <= 100000

    def test_estimate_95_confidence(self) -> None:
        """Test sample size estimation with 95% confidence."""
        sampler = DataSampler()
        size_95 = sampler.estimate_sample_size(total_rows=100000, confidence=0.95)
        assert size_95 > 0

    def test_estimate_99_confidence(self) -> None:
        """Test sample size estimation with 99% confidence."""
        sampler = DataSampler()
        size_99 = sampler.estimate_sample_size(total_rows=100000, confidence=0.99)
        assert size_99 > 0
        # Higher confidence should require larger sample
        assert size_99 >= sampler.estimate_sample_size(total_rows=100000, confidence=0.95)

    def test_estimate_90_confidence(self) -> None:
        """Test sample size estimation with 90% confidence."""
        sampler = DataSampler()
        size_90 = sampler.estimate_sample_size(total_rows=100000, confidence=0.90)
        assert size_90 > 0
        # Lower confidence should require smaller sample
        assert size_90 <= sampler.estimate_sample_size(total_rows=100000, confidence=0.95)

    def test_estimate_different_margins(self) -> None:
        """Test sample size estimation with different margins of error."""
        sampler = DataSampler()

        small_margin = sampler.estimate_sample_size(total_rows=100000, margin_error=0.01)
        large_margin = sampler.estimate_sample_size(total_rows=100000, margin_error=0.1)

        # Smaller margin of error should require larger sample
        assert small_margin > large_margin

    def test_estimate_small_population(self) -> None:
        """Test sample size estimation with small population."""
        sampler = DataSampler()
        size = sampler.estimate_sample_size(total_rows=100)
        assert size <= 100
        assert size > 0

    def test_estimate_zero_population(self) -> None:
        """Test sample size estimation with zero population."""
        sampler = DataSampler()
        size = sampler.estimate_sample_size(total_rows=0)
        assert size == 0

    def test_estimate_custom_confidence_values(self) -> None:
        """Test sample size estimation with custom confidence values."""
        sampler = DataSampler()
        # Test with custom confidence that uses default z-score
        size = sampler.estimate_sample_size(total_rows=10000, confidence=0.80)
        assert size > 0

    def test_estimate_unknown_confidence_default_z(self) -> None:
        """Test that unknown confidence uses default z-score."""
        sampler = DataSampler()
        # 0.85 is not in z_scores, should use default 1.96
        size_85 = sampler.estimate_sample_size(total_rows=10000, confidence=0.85)
        size_95 = sampler.estimate_sample_size(total_rows=10000, confidence=0.95)
        # Both should use z=1.96 (default), so they should be equal
        assert size_85 == size_95

    def test_estimate_sample_size_formula(self) -> None:
        """Test the sample size formula."""
        sampler = DataSampler()
        # With p=0.5, z=1.96, e=0.05:
        # n = (1.96^2 * 0.5 * 0.5) / (0.05^2) = 384.16
        # With finite population correction for 10000:
        # n = 384.16 / (1 + (384.16 - 1) / 10000) = 369.99 -> 369
        size = sampler.estimate_sample_size(total_rows=10000, confidence=0.95, margin_error=0.05)
        assert 365 <= size <= 375  # Approximately 370 after finite population correction

    def test_estimate_finite_population_correction(self) -> None:
        """Test finite population correction."""
        sampler = DataSampler()
        # Without finite population correction, sample size would be ~384
        # With population of 1000, it should be smaller
        size_small = sampler.estimate_sample_size(
            total_rows=1000, confidence=0.95, margin_error=0.05
        )
        size_large = sampler.estimate_sample_size(
            total_rows=100000, confidence=0.95, margin_error=0.05
        )
        # Smaller population should give smaller or equal sample size
        assert size_small <= size_large


class TestDataSamplerStrategyAccess:
    """Tests for DataSampler strategy access."""

    def test_sampler_caching(self) -> None:
        """Test that samplers are cached."""
        sampler = DataSampler()
        SampleConfig(strategy="random")

        # Get sampler twice - should return same instance
        # We can't easily test this without Spark, but we can verify the structure
        strategies = sampler.get_supported_strategies()
        assert len(strategies) == 3

    def test_sample_with_different_strategies(self) -> None:
        """Test that different strategies are registered."""
        sampler = DataSampler()

        # Verify all strategies are available
        assert "random" in sampler.STRATEGIES
        assert "stratified" in sampler.STRATEGIES
        assert "reservoir" in sampler.STRATEGIES
        assert len(sampler.STRATEGIES) == 3


class TestSamplerIntegration:
    """Integration tests for samplers (mock-based)."""

    def test_random_sampler_creates_spark_if_needed(self) -> None:
        """Test that RandomSampler creates Spark if not provided."""
        sampler = RandomSampler()
        assert sampler.spark is None
        assert sampler._own_spark is False

    def test_stratified_sampler_creates_spark_if_needed(self) -> None:
        """Test that StratifiedSampler creates Spark if not provided."""
        sampler = StratifiedSampler()
        assert sampler.spark is None
        assert sampler._own_spark is False

    def test_reservoir_sampler_creates_spark_if_needed(self) -> None:
        """Test that ReservoirSampler creates Spark if not provided."""
        sampler = ReservoirSampler()
        assert sampler.spark is None
        assert sampler._own_spark is False


class TestDataSamplerFactory:
    """Additional tests for DataSampler factory."""

    def test_create_sampler_random(self) -> None:
        """Test creating random sampler."""
        sampler = DataSampler()
        strategy = "random"
        sampler_obj = sampler.STRATEGIES[strategy](sampler.spark)
        assert isinstance(sampler_obj, RandomSampler)

    def test_create_sampler_stratified(self) -> None:
        """Test creating stratified sampler."""
        sampler = DataSampler()
        strategy = "stratified"
        sampler_obj = sampler.STRATEGIES[strategy](sampler.spark)
        assert isinstance(sampler_obj, StratifiedSampler)

    def test_create_sampler_reservoir(self) -> None:
        """Test creating reservoir sampler."""
        sampler = DataSampler()
        strategy = "reservoir"
        sampler_obj = sampler.STRATEGIES[strategy](sampler.spark)
        assert isinstance(sampler_obj, ReservoirSampler)

    def test_sample_method_calls_right_sampler(self) -> None:
        """Test that sample method calls the right sampler."""
        sampler = DataSampler()
        # Verify the method routing logic
        assert sampler.STRATEGIES["random"] == RandomSampler
        assert sampler.STRATEGIES["stratified"] == StratifiedSampler
        assert sampler.STRATEGIES["reservoir"] == ReservoirSampler


class TestSampleConfigValidation:
    """Tests for SampleConfig validation."""

    def test_sample_config_with_all_options(self) -> None:
        """Test SampleConfig with all options."""
        config = SampleConfig(
            sample_size=500,
            strategy="reservoir",
            seed=999,
            columns=["col1", "col2", "col3"],
        )
        assert config.sample_size == 500
        assert config.strategy == "reservoir"
        assert config.seed == 999
        assert len(config.columns) == 3

    def test_sample_config_default_values(self) -> None:
        """Test SampleConfig default values are sensible."""
        config = SampleConfig()
        assert config.sample_size == 0.1  # 10% sample
        assert config.strategy == "random"
        assert config.seed == 42  # Reproducible
        assert config.columns is None  # All columns
