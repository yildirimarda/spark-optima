# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for data generators."""

from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spark_optima.data.generators import (
    ColumnSpec,
    DataGenerator,
    DataGeneratorConfig,
)


class TestDataGeneratorConfig:
    """Tests for DataGeneratorConfig."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        config = DataGeneratorConfig()
        assert config.num_rows == 10000
        assert config.num_partitions == 4
        assert config.format == "parquet"
        assert config.compression == "snappy"
        assert config.null_ratio == 0.05
        assert config.skew_factor == 1.0
        assert config.random_seed == 42

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        config = DataGeneratorConfig(
            num_rows=50000,
            num_partitions=8,
            format="json",
            compression="gzip",
            null_ratio=0.1,
            skew_factor=2.0,
            random_seed=123,
        )
        assert config.num_rows == 50000
        assert config.num_partitions == 8
        assert config.format == "json"
        assert config.compression == "gzip"
        assert config.null_ratio == 0.1
        assert config.skew_factor == 2.0
        assert config.random_seed == 123

    def test_is_dataclass(self) -> None:
        """Test that DataGeneratorConfig is a dataclass."""
        assert is_dataclass(DataGeneratorConfig)


class TestColumnSpec:
    """Tests for ColumnSpec."""

    def test_required_fields(self) -> None:
        """Test required fields only."""
        spec = ColumnSpec(name="test_col")
        assert spec.name == "test_col"
        assert spec.data_type == "string"
        assert spec.nullable is True
        assert spec.cardinality is None
        assert spec.min_value is None
        assert spec.max_value is None

    def test_full_initialization(self) -> None:
        """Test full initialization."""
        spec = ColumnSpec(
            name="id",
            data_type="int",
            nullable=False,
            cardinality=1000,
            min_value=1,
            max_value=1000000,
        )
        assert spec.name == "id"
        assert spec.data_type == "int"
        assert spec.nullable is False
        assert spec.cardinality == 1000
        assert spec.min_value == 1
        assert spec.max_value == 1000000

    def test_is_dataclass(self) -> None:
        """Test that ColumnSpec is a dataclass."""
        assert is_dataclass(ColumnSpec)


class TestDataGenerator:
    """Tests for DataGenerator."""

    def test_initialization_without_spark(self) -> None:
        """Test initialization without Spark."""
        generator = DataGenerator()
        assert generator.spark is None
        assert generator._own_spark is False

    def test_initialization_with_spark(self) -> None:
        """Test initialization with Spark."""
        mock_spark = object()
        generator = DataGenerator(spark=mock_spark)
        assert generator.spark is mock_spark
        assert generator._own_spark is False

    def test_supported_formats(self) -> None:
        """Test supported formats list."""
        generator = DataGenerator()
        formats = generator.get_supported_formats()
        assert "parquet" in formats
        assert "delta" in formats
        assert "json" in formats
        assert "csv" in formats
        assert "orc" in formats

    def test_format_options(self) -> None:
        """Test format-specific options."""
        assert "parquet" in DataGenerator.FORMAT_OPTIONS
        assert "delta" in DataGenerator.FORMAT_OPTIONS
        assert "json" in DataGenerator.FORMAT_OPTIONS
        assert "csv" in DataGenerator.FORMAT_OPTIONS
        assert "orc" in DataGenerator.FORMAT_OPTIONS

    def test_supported_formats_returns_copy(self) -> None:
        """Test that get_supported_formats returns a copy."""
        generator = DataGenerator()
        formats = generator.get_supported_formats()
        formats.append("new_format")  # Modify the returned list
        assert "new_format" not in generator.get_supported_formats()


class TestEstimateSize:
    """Tests for estimate_size method."""

    def test_estimate_parquet_size(self) -> None:
        """Test size estimation for Parquet format."""
        generator = DataGenerator()
        result = generator.estimate_size(
            num_rows=1000000,
            num_cols=10,
            avg_col_size_bytes=20,
            format="parquet",
            compression="snappy",
        )
        assert "raw_gb" in result
        assert "estimated_gb" in result
        assert result["rows"] == 1000000
        assert result["columns"] == 10

    def test_estimate_json_size(self) -> None:
        """Test size estimation for JSON format."""
        generator = DataGenerator()
        result = generator.estimate_size(
            num_rows=100000,
            num_cols=5,
            format="json",
            compression="gzip",
        )
        assert result["estimated_gb"] > 0
        assert "raw_gb" in result

    def test_estimate_csv_size(self) -> None:
        """Test size estimation for CSV format."""
        generator = DataGenerator()
        result = generator.estimate_size(
            num_rows=50000,
            num_cols=8,
            format="csv",
            compression="none",
        )
        assert result["estimated_gb"] > 0

    def test_estimate_delta_size(self) -> None:
        """Test size estimation for Delta format."""
        generator = DataGenerator()
        result = generator.estimate_size(
            num_rows=1000000,
            num_cols=10,
            format="delta",
            compression="snappy",
        )
        assert result["estimated_gb"] > 0

    def test_estimate_different_compressions(self) -> None:
        """Test different compression ratios."""
        generator = DataGenerator()

        snappy = generator.estimate_size(100000, 10, compression="snappy")
        gzip = generator.estimate_size(100000, 10, compression="gzip")

        # Gzip should have smaller size than snappy
        assert gzip["estimated_gb"] < snappy["estimated_gb"]

    def test_estimate_none_compression(self) -> None:
        """Test size estimation with no compression."""
        generator = DataGenerator()
        result = generator.estimate_size(
            num_rows=1000,
            num_cols=5,
            format="parquet",
            compression="none",
        )
        assert result["estimated_gb"] > 0

    def test_estimate_lz4_compression(self) -> None:
        """Test size estimation with lz4 compression."""
        generator = DataGenerator()
        result = generator.estimate_size(
            num_rows=100000,
            num_cols=10,
            format="parquet",
            compression="lz4",
        )
        assert "estimated_gb" in result

    def test_estimate_zstd_compression(self) -> None:
        """Test size estimation with zstd compression."""
        generator = DataGenerator()
        result = generator.estimate_size(
            num_rows=100000,
            num_cols=10,
            format="parquet",
            compression="zstd",
        )
        assert "estimated_gb" in result

    def test_estimate_zlib_compression(self) -> None:
        """Test size estimation with zlib compression."""
        generator = DataGenerator()
        result = generator.estimate_size(
            num_rows=100000,
            num_cols=10,
            format="parquet",
            compression="zlib",
        )
        assert "estimated_gb" in result

    def test_estimate_orc_format(self) -> None:
        """Test size estimation for ORC format."""
        generator = DataGenerator()
        result = generator.estimate_size(
            num_rows=50000,
            num_cols=8,
            format="orc",
            compression="zstd",
        )
        assert result["estimated_gb"] > 0

    def test_estimate_with_custom_avg_col_size(self) -> None:
        """Test size estimation with custom average column size."""
        generator = DataGenerator()
        result = generator.estimate_size(
            num_rows=1000,
            num_cols=5,
            avg_col_size_bytes=50,
            format="parquet",
        )
        assert result["raw_gb"] == (1000 * 5 * 50) / (1024**3)

    def test_estimate_all_format_overhead(self) -> None:
        """Test all format overhead values."""
        generator = DataGenerator()
        formats = ["parquet", "delta", "json", "csv", "orc"]
        for fmt in formats:
            result = generator.estimate_size(1000, 5, format=fmt)
            assert "estimated_gb" in result

    def test_estimate_unknown_compression(self) -> None:
        """Test size estimation with unknown compression (uses default)."""
        generator = DataGenerator()
        result = generator.estimate_size(
            num_rows=1000,
            num_cols=5,
            compression="unknown_compression",
        )
        assert result["estimated_gb"] > 0

    def test_estimate_unknown_format(self) -> None:
        """Test size estimation with unknown format (uses default overhead)."""
        generator = DataGenerator()
        result = generator.estimate_size(
            num_rows=1000,
            num_cols=5,
            format="unknown_format",
        )
        assert result["estimated_gb"] > 0

    def test_estimate_compression_ratios(self) -> None:
        """Test all compression ratios."""
        generator = DataGenerator()
        compressions = ["none", "snappy", "gzip", "lz4", "zstd", "zlib"]
        for comp in compressions:
            result = generator.estimate_size(1000, 5, compression=comp)
            assert result["estimated_gb"] > 0

    def test_estimate_raw_bytes_calculation(self) -> None:
        """Test raw bytes calculation."""
        generator = DataGenerator()
        result = generator.estimate_size(num_rows=100, num_cols=10, avg_col_size_bytes=20)
        expected_raw = 100 * 10 * 20
        assert result["raw_gb"] == expected_raw / (1024**3)

    def test_estimate_with_zero_rows(self) -> None:
        """Test estimate with zero rows."""
        generator = DataGenerator()
        result = generator.estimate_size(num_rows=0, num_cols=10)
        assert result["raw_gb"] == 0
        assert result["estimated_gb"] == 0
        assert result["rows"] == 0

    def test_estimate_with_zero_cols(self) -> None:
        """Test estimate with zero columns."""
        generator = DataGenerator()
        result = generator.estimate_size(num_rows=100, num_cols=0)
        assert result["raw_gb"] == 0
        assert result["columns"] == 0


class TestGenerateTabular:
    """Tests for generate_tabular method."""

    def test_generate_tabular_method_exists(self) -> None:
        """Test that generate_tabular method exists."""
        generator = DataGenerator()
        assert hasattr(generator, "generate_tabular")

    def test_generate_tabular_creates_correct_columns(self) -> None:
        """Test that generate_tabular creates correct columns."""
        generator = DataGenerator()
        # Mock the generate method to inspect what's passed
        with patch.object(generator, "generate") as mock_generate:
            mock_generate.return_value = Path("./output")
            generator.generate_tabular("./output", num_rows=100, num_cols=5)
            mock_generate.assert_called_once()
            # Check that columns were created and passed (3rd positional arg)
            call_args = mock_generate.call_args
            columns = call_args.args[2]  # columns is 3rd positional arg
            assert len(columns) == 5
            # Check first column is id
            assert columns[0].name == "id"
            assert columns[0].data_type == "int"

    def test_generate_tabular_kwargs_passthrough(self) -> None:
        """Test that kwargs are passed to DataGeneratorConfig."""
        generator = DataGenerator()
        with patch.object(generator, "generate") as mock_generate:
            mock_generate.return_value = Path("./output")
            generator.generate_tabular("./output", num_rows=500, num_cols=3, compression="gzip")
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            # config is the 2nd positional arg (args[1])
            config = call_args.args[1]
            assert config.num_rows == 500
            assert config.compression == "gzip"


class TestGenerateNested:
    """Tests for generate_nested method."""

    def test_generate_nested_requires_pyspark(self) -> None:
        """Test that generate_nested requires PySpark."""
        from spark_optima.data.generators import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            generator = DataGenerator()
            with pytest.raises(RuntimeError, match="PySpark required"):
                generator.generate_nested("./output")

    def test_generate_nested_calls_generate(self) -> None:
        """Test that generate_nested calls generate with correct params."""
        generator = DataGenerator()
        with patch.object(generator, "generate") as mock_generate:
            mock_generate.return_value = Path("./output")
            generator.generate_nested("./output", num_rows=5000)
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            # Check output_path is passed correctly
            assert str(call_args.args[0]) == "./output"
            # config is the 2nd positional arg (args[1])
            config = call_args.args[1]
            assert config.num_rows == 5000
            assert config.format == "json"


class TestGeneratorEdgeCases:
    """Tests for edge cases in DataGenerator."""

    def test_config_with_zero_rows(self) -> None:
        """Test DataGeneratorConfig with zero rows."""
        config = DataGeneratorConfig(num_rows=0)
        assert config.num_rows == 0

    def test_config_with_large_partitions(self) -> None:
        """Test DataGeneratorConfig with large number of partitions."""
        config = DataGeneratorConfig(num_partitions=1000)
        assert config.num_partitions == 1000

    def test_config_null_ratio_boundaries(self) -> None:
        """Test DataGeneratorConfig with null ratio boundaries."""
        config = DataGeneratorConfig(null_ratio=0.0)
        assert config.null_ratio == 0.0

        config = DataGeneratorConfig(null_ratio=1.0)
        assert config.null_ratio == 1.0

    def test_column_spec_with_all_numeric_types(self) -> None:
        """Test ColumnSpec with various numeric types."""
        types = [
            "int",
            "integer",
            "bigint",
            "smallint",
            "tinyint",
            "float",
            "double",
            "decimal",
            "long",
            "short",
        ]
        for dt in types:
            spec = ColumnSpec(name="num_col", data_type=dt)
            assert spec.data_type == dt

    def test_column_spec_string_types(self) -> None:
        """Test ColumnSpec with string types."""
        types = ["string", "varchar", "text", "char"]
        for dt in types:
            spec = ColumnSpec(name="str_col", data_type=dt)
            assert spec.data_type == dt

    def test_column_spec_boolean_type(self) -> None:
        """Test ColumnSpec with boolean type."""
        spec = ColumnSpec(name="bool_col", data_type="boolean")
        assert spec.data_type == "boolean"


class TestDefaultColumns:
    """Tests for _default_columns method."""

    def test_default_columns_returns_list(self) -> None:
        """Test that _default_columns returns a list."""
        generator = DataGenerator()
        columns = generator._default_columns()
        assert isinstance(columns, list)
        assert len(columns) > 0

    def test_default_columns_types(self) -> None:
        """Test default column specifications."""
        generator = DataGenerator()
        columns = generator._default_columns()
        for col in columns:
            assert isinstance(col, ColumnSpec)

    def test_default_columns_has_id(self) -> None:
        """Test that default columns include an id column."""
        generator = DataGenerator()
        columns = generator._default_columns()
        names = [c.name for c in columns]
        assert "id" in names

    def test_default_columns_count(self) -> None:
        """Test expected number of default columns."""
        generator = DataGenerator()
        columns = generator._default_columns()
        assert len(columns) == 7

    def test_default_columns_id_spec(self) -> None:
        """Test default id column specification."""
        generator = DataGenerator()
        columns = generator._default_columns()
        id_col = columns[0]
        assert id_col.name == "id"
        assert id_col.data_type == "int"
        assert id_col.nullable is False
        assert id_col.min_value == 1
        assert id_col.max_value == 1000000

    def test_default_columns_name_spec(self) -> None:
        """Test default name column specification."""
        generator = DataGenerator()
        columns = generator._default_columns()
        name_col = columns[1]
        assert name_col.name == "name"
        assert name_col.data_type == "string"
        assert name_col.cardinality == 10000

    def test_default_columns_date_spec(self) -> None:
        """Test default hire_date column specification."""
        generator = DataGenerator()
        columns = generator._default_columns()
        date_col = columns[5]  # hire_date
        assert date_col.name == "hire_date"
        assert date_col.data_type == "date"

    def test_default_columns_boolean_spec(self) -> None:
        """Test default is_active column specification."""
        generator = DataGenerator()
        columns = generator._default_columns()
        bool_col = columns[6]  # is_active
        assert bool_col.name == "is_active"
        assert bool_col.data_type == "boolean"


class TestBuildSchema:
    """Tests for _build_schema method."""

    def test_build_schema_returns_none_without_pyspark(self) -> None:
        """Test that _build_schema returns None when PySpark unavailable."""
        from spark_optima.data.generators import PYSPARK_AVAILABLE

        generator = DataGenerator()
        columns = [ColumnSpec(name="test", data_type="string")]

        if not PYSPARK_AVAILABLE:
            result = generator._build_schema(columns)
            assert result is None
        else:
            result = generator._build_schema(columns)
            assert result is not None

    def test_build_schema_with_various_types(self) -> None:
        """Test schema building with different data types."""
        from spark_optima.data.generators import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        generator = DataGenerator()
        columns = [
            ColumnSpec(name="str_col", data_type="string"),
            ColumnSpec(name="int_col", data_type="int"),
            ColumnSpec(name="dbl_col", data_type="double"),
            ColumnSpec(name="bool_col", data_type="boolean"),
        ]
        schema = generator._build_schema(columns)
        assert schema is not None
        assert len(schema.fields) == 4

    def test_build_schema_with_nullable(self) -> None:
        """Test that nullable is respected in schema."""
        from spark_optima.data.generators import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        generator = DataGenerator()
        columns = [
            ColumnSpec(name="nullable_col", data_type="string", nullable=True),
            ColumnSpec(name="non_nullable_col", data_type="int", nullable=False),
        ]
        schema = generator._build_schema(columns)
        assert schema.fields[0].nullable is True
        assert schema.fields[1].nullable is False

    def test_build_schema_integer_type(self) -> None:
        """Test schema building with 'integer' type mapping."""
        from spark_optima.data.generators import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        generator = DataGenerator()
        columns = [ColumnSpec(name="int_col", data_type="integer")]
        schema = generator._build_schema(columns)
        from pyspark.sql.types import IntegerType

        assert isinstance(schema.fields[0].dataType, IntegerType)

    def test_build_schema_float_type(self) -> None:
        """Test schema building with 'float' type mapping."""
        from spark_optima.data.generators import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        generator = DataGenerator()
        columns = [ColumnSpec(name="float_col", data_type="float")]
        schema = generator._build_schema(columns)
        from pyspark.sql.types import DoubleType

        assert isinstance(schema.fields[0].dataType, DoubleType)

    def test_build_schema_bool_type(self) -> None:
        """Test schema building with 'bool' type mapping."""
        from spark_optima.data.generators import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        generator = DataGenerator()
        columns = [ColumnSpec(name="bool_col", data_type="bool")]
        schema = generator._build_schema(columns)
        from pyspark.sql.types import BooleanType

        assert isinstance(schema.fields[0].dataType, BooleanType)

    def test_build_schema_date_type(self) -> None:
        """Test schema building with 'date' type mapping."""
        from spark_optima.data.generators import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        generator = DataGenerator()
        columns = [ColumnSpec(name="date_col", data_type="date")]
        schema = generator._build_schema(columns)
        from pyspark.sql.types import DateType

        assert isinstance(schema.fields[0].dataType, DateType)

    def test_build_schema_timestamp_type(self) -> None:
        """Test schema building with 'timestamp' type mapping."""
        from spark_optima.data.generators import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        generator = DataGenerator()
        columns = [ColumnSpec(name="ts_col", data_type="timestamp")]
        schema = generator._build_schema(columns)
        from pyspark.sql.types import TimestampType

        assert isinstance(schema.fields[0].dataType, TimestampType)

    def test_build_schema_unknown_type_defaults_to_string(self) -> None:
        """Test that unknown types default to StringType."""
        from spark_optima.data.generators import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        generator = DataGenerator()
        columns = [ColumnSpec(name="unknown_col", data_type="unknown_type")]
        schema = generator._build_schema(columns)
        from pyspark.sql.types import StringType

        assert isinstance(schema.fields[0].dataType, StringType)

    def test_build_schema_case_insensitive(self) -> None:
        """Test that type mapping is case insensitive."""
        from spark_optima.data.generators import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        generator = DataGenerator()
        columns = [ColumnSpec(name="col", data_type="INT")]
        schema = generator._build_schema(columns)
        from pyspark.sql.types import IntegerType

        assert isinstance(schema.fields[0].dataType, IntegerType)


class TestGenerateValue:
    """Tests for _generate_value method."""

    def test_generate_string_value(self) -> None:
        """Test generating string values."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="string", cardinality=10)
        value = generator._generate_value(col, DataGeneratorConfig())
        assert isinstance(value, str)
        assert value.startswith("category_")

    def test_generate_int_value(self) -> None:
        """Test generating integer values."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="int", min_value=1, max_value=100)
        config = DataGeneratorConfig(null_ratio=0.0)  # Ensure not null
        value = generator._generate_value(col, config)
        assert isinstance(value, int)
        assert 1 <= value <= 100

    def test_generate_integer_value(self) -> None:
        """Test generating integer values with 'integer' type."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="integer", min_value=1, max_value=100)
        config = DataGeneratorConfig(null_ratio=0.0)  # Ensure not null
        value = generator._generate_value(col, config)
        assert isinstance(value, int), f"Expected int, got {type(value)}"
        assert 1 <= value <= 100

    def test_generate_float_value(self) -> None:
        """Test generating float values with 'float' type."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="float", min_value=0.0, max_value=1.0)
        config = DataGeneratorConfig(null_ratio=0.0)  # Ensure not null
        value = generator._generate_value(col, config)
        assert isinstance(value, float), f"Expected float, got {type(value)}"
        assert 0.0 <= value <= 1.0

    def test_generate_boolean_value(self) -> None:
        """Test generating boolean values."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="boolean")
        config = DataGeneratorConfig(null_ratio=0.0)  # Ensure not null
        value = generator._generate_value(col, config)
        assert isinstance(value, bool)

    def test_generate_bool_value_falls_to_string(self) -> None:
        """Test that 'bool' type (not 'boolean') generates string."""
        # Note: _generate_value handles "boolean" but not "bool"
        # "bool" falls through to the else clause and generates a string
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="bool", cardinality=5)
        config = DataGeneratorConfig(null_ratio=0.0)
        value = generator._generate_value(col, config)
        assert isinstance(value, str)

    def test_generate_date_value(self) -> None:
        """Test generating date values."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="date")
        config = DataGeneratorConfig(null_ratio=0.0)  # Ensure not null
        value = generator._generate_value(col, config)
        from datetime import datetime

        assert isinstance(value, datetime)

    def test_generate_timestamp_value(self) -> None:
        """Test generating timestamp values."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="timestamp")
        config = DataGeneratorConfig(null_ratio=0.0)  # Ensure not null
        value = generator._generate_value(col, config)
        from datetime import datetime

        assert isinstance(value, datetime), f"Expected datetime, got {type(value)}"

    def test_generate_null_value(self) -> None:
        """Test generating null values based on null_ratio."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="string", nullable=True)
        config = DataGeneratorConfig(null_ratio=1.0)
        value = generator._generate_value(col, config)
        assert value is None

    def test_generate_value_no_nullable(self) -> None:
        """Test generating non-nullable values."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="int", nullable=False, min_value=1, max_value=10)
        config = DataGeneratorConfig(null_ratio=1.0)
        value = generator._generate_value(col, config)
        assert value is not None  # Should not be null since nullable=False

    def test_generate_string_without_cardinality(self) -> None:
        """Test generating random string without cardinality."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="string", cardinality=None)
        config = DataGeneratorConfig(null_ratio=0.0)
        value = generator._generate_value(col, config)
        assert isinstance(value, str)
        assert len(value) >= 5 and len(value) <= 20

    def test_generate_int_with_default_min_max(self) -> None:
        """Test generating int with default min/max values."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="int", min_value=None, max_value=None)
        config = DataGeneratorConfig(null_ratio=0.0)
        value = generator._generate_value(col, config)
        assert isinstance(value, int)
        assert 0 <= value <= 1000000  # Default max

    def test_generate_double_with_default_min_max(self) -> None:
        """Test generating double with default min/max values."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="double", min_value=None, max_value=None)
        config = DataGeneratorConfig(null_ratio=0.0)
        value = generator._generate_value(col, config)
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0  # Default range

    def test_generate_value_unknown_type_is_string(self) -> None:
        """Test that unknown types generate string values."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="unknown_type", cardinality=5)
        config = DataGeneratorConfig(null_ratio=0.0)
        value = generator._generate_value(col, config)
        assert isinstance(value, str)
        assert value.startswith("category_")

    def test_generate_value_with_all_types(self) -> None:
        """Test generating values with all supported types."""
        from datetime import datetime

        generator = DataGenerator()
        config = DataGeneratorConfig(null_ratio=0.0)

        # Test each type
        types_and_checks = [
            ("string", str),
            ("int", int),
            ("integer", int),
            ("double", float),
            ("float", float),
            ("boolean", bool),
            ("date", datetime),
            ("timestamp", datetime),
        ]

        for data_type, expected_type in types_and_checks:
            if data_type in ("date", "timestamp"):
                col = ColumnSpec(name="test", data_type=data_type)
            else:
                col = ColumnSpec(name="test", data_type=data_type, cardinality=5)
            value = generator._generate_value(col, config)
            assert isinstance(value, expected_type), f"Failed for type: {data_type}"


class TestUnsupportedFormat:
    """Tests for unsupported format handling."""

    def test_generate_unsupported_format(self) -> None:
        """Test that unsupported format raises ValueError."""
        from spark_optima.data.generators import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        generator = DataGenerator()
        config = DataGeneratorConfig(format="unsupported_format")
        with pytest.raises(ValueError, match="Unsupported format"):
            generator.generate("./output", config=config)


class TestGeneratorCleanup:
    """Tests for generator cleanup (__del__ method)."""

    def test_del_stops_spark_when_own_spark(self) -> None:
        """Test that __del__ stops Spark when we own it."""
        generator = DataGenerator()
        mock_spark = MagicMock()
        generator.spark = mock_spark
        generator._own_spark = True
        generator.__del__()
        mock_spark.stop.assert_called_once()

    def test_del_does_not_stop_spark_when_not_own(self) -> None:
        """Test that __del__ does not stop Spark when we don't own it."""
        generator = DataGenerator()
        mock_spark = MagicMock()
        generator.spark = mock_spark
        generator._own_spark = False
        generator.__del__()
        mock_spark.stop.assert_not_called()

    def test_del_handles_exception(self) -> None:
        """Test that __del__ handles exceptions gracefully."""
        generator = DataGenerator()
        mock_spark = MagicMock()
        mock_spark.stop.side_effect = Exception("Test exception")
        generator.spark = mock_spark
        generator._own_spark = True
        # Should not raise exception
        generator.__del__()


class TestGenerateRDD:
    """Tests for _generate_rdd method."""

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.generators.SparkSession")
    def test_generate_rdd_creates_rdd(self, mock_spark_class) -> None:
        """Test that _generate_rdd creates an RDD correctly."""
        mock_spark = MagicMock()
        mock_spark_class.builder.appName().master().getOrCreate.return_value = mock_spark

        generator = DataGenerator(spark=mock_spark)
        config = DataGeneratorConfig(num_rows=100, num_partitions=2)
        columns = [ColumnSpec(name="test", data_type="int", min_value=1, max_value=100)]

        # Call _generate_rdd
        generator._generate_rdd(config, columns, MagicMock())

        # Verify parallelize was called with correct args
        mock_spark.sparkContext.parallelize.assert_called_once_with(range(100), 2)


class TestGenerateIntegration:
    """Integration tests for generators."""

    def test_generator_has_all_required_methods(self) -> None:
        """Test that DataGenerator has all required methods."""
        generator = DataGenerator()
        required_methods = [
            "generate_tabular",
            "generate_nested",
            "estimate_size",
            "get_supported_formats",
            "generate",
            "_build_schema",
            "_generate_value",
        ]
        for method in required_methods:
            assert hasattr(generator, method), f"Missing method: {method}"

    def test_format_options_completeness(self) -> None:
        """Test that all formats have options defined."""
        for fmt in DataGenerator.FORMAT_OPTIONS:
            assert isinstance(DataGenerator.FORMAT_OPTIONS[fmt], dict)

    def test_estimate_size_keys(self) -> None:
        """Test that estimate_size returns all expected keys."""
        generator = DataGenerator()
        result = generator.estimate_size(100, 10)
        expected_keys = {"raw_gb", "estimated_gb", "rows", "columns"}
        assert expected_keys.issubset(set(result.keys()))

    def test_data_generator_is_not_abstract(self) -> None:
        """Test that DataGenerator can be instantiated."""
        generator = DataGenerator()
        assert generator is not None

    def test_format_options_have_compression(self) -> None:
        """Test that format options include compression where expected."""
        # Parquet, json, csv should have compression in options
        assert "compression" in DataGenerator.FORMAT_OPTIONS.get("parquet", {})
        assert "compression" in DataGenerator.FORMAT_OPTIONS.get("json", {})
        assert "compression" in DataGenerator.FORMAT_OPTIONS.get("csv", {})


class TestDataGeneratorConfigColumns:
    """Tests for DataGeneratorConfig with columns field."""

    def test_config_with_columns(self) -> None:
        """Test DataGeneratorConfig with columns parameter."""
        columns = [
            ColumnSpec(name="id", data_type="int"),
            ColumnSpec(name="name", data_type="string"),
        ]
        config = DataGeneratorConfig(num_rows=100, columns=columns)
        assert config.columns is not None
        assert len(config.columns) == 2
        assert config.columns[0].name == "id"
        assert config.columns[1].name == "name"

    def test_config_columns_none_by_default(self) -> None:
        """Test that columns is None by default."""
        config = DataGeneratorConfig()
        assert config.columns is None


class TestGenerateMethodPySparkNotAvailable:
    """Tests for generate method when PySpark is not available."""

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", False)
    def test_generate_raises_runtime_error_when_pyspark_not_available(self) -> None:
        """Test that generate raises RuntimeError when PySpark not available (line 167)."""
        generator = DataGenerator()
        config = DataGeneratorConfig()
        with pytest.raises(RuntimeError, match="PySpark required"):
            generator.generate("./output", config=config)


class TestGenerateMethod:
    """Tests for generate method with PySpark available."""

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_initializes_spark_if_none(self) -> None:
        """Test that generate initializes Spark if not provided (lines 180-185)."""
        generator = DataGenerator(spark=None)
        mock_spark = MagicMock()
        mock_df = MagicMock()
        mock_spark.createDataFrame.return_value = mock_df
        mock_df.repartition.return_value = mock_df
        mock_df.write = MagicMock()
        mock_df.write.mode.return_value = mock_df.write

        with (patch("spark_optima.data.generators.SparkSession") as mock_spark_class,
              patch.object(generator, "_generate_rdd") as mock_rdd,
              patch.object(generator, "_build_schema") as mock_schema,
              patch.object(generator, "_default_columns") as mock_default):
            mock_spark_class.builder.appName().master().getOrCreate.return_value = mock_spark
            mock_rdd.return_value = MagicMock()
            mock_schema.return_value = MagicMock()
            mock_default.return_value = [
                ColumnSpec(name="id", data_type="int", min_value=1, max_value=100),
            ]

            generator.generate("./output", config=DataGeneratorConfig())

            # Verify Spark was initialized
            mock_spark_class.builder.appName().master().getOrCreate.assert_called_once()
            assert generator._own_spark is True

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_with_default_columns(self) -> None:
        """Test generate uses default columns when none provided (lines 188-191)."""
        generator = DataGenerator()
        generator.spark = MagicMock()
        mock_df = MagicMock()
        generator.spark.createDataFrame.return_value = mock_df
        mock_df.repartition.return_value = mock_df
        mock_df.write = MagicMock()
        mock_df.write.mode.return_value = mock_df.write

        with (patch.object(generator, "_generate_rdd") as mock_rdd,
              patch.object(generator, "_build_schema") as mock_schema,
              patch.object(generator, "_default_columns") as mock_default):
            mock_rdd.return_value = MagicMock()
            mock_schema.return_value = MagicMock()
            mock_default.return_value = [
                ColumnSpec(
                    name="id",
                    data_type="int",
                    min_value=1,
                    max_value=1000000,
                    nullable=False,
                ),
            ]

            generator.generate("./output", config=DataGeneratorConfig())

            # Verify default columns were used
            mock_default.assert_called_once()
            mock_schema.assert_called_once_with(mock_default.return_value)

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_with_skew_factor(self) -> None:
        """Test generate applies skew when skew_factor > 1.0 (lines 206-208)."""
        generator = DataGenerator()
        generator.spark = MagicMock()
        mock_df = MagicMock()
        generator.spark.createDataFrame.return_value = mock_df
        mock_df_repartitioned = MagicMock()
        mock_df.repartition.return_value = mock_df_repartitioned
        mock_df_repartitioned.write = MagicMock()
        mock_df_repartitioned.write.mode.return_value = mock_df_repartitioned.write

        columns = [
            ColumnSpec(name="category", data_type="string", cardinality=10),
        ]

        with (patch.object(generator, "_generate_rdd") as mock_rdd,
              patch.object(generator, "_build_schema") as mock_schema,
              patch.object(generator, "_apply_skew") as mock_skew):
            mock_rdd.return_value = MagicMock()
            mock_schema.return_value = MagicMock()
            mock_skew.return_value = MagicMock()

            config = DataGeneratorConfig(skew_factor=2.0)
            generator.generate("./output", config=config, columns=columns)

            # Verify _apply_skew was called
            mock_skew.assert_called_once()

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_writes_parquet(self) -> None:
        """Test generate writes data in parquet format (lines 226-232)."""
        generator = DataGenerator()
        generator.spark = MagicMock()
        mock_df = MagicMock()
        generator.spark.createDataFrame.return_value = mock_df
        mock_df_repartitioned = MagicMock()
        mock_df.repartition.return_value = mock_df_repartitioned

        # Mock the writer chain
        mock_writer = MagicMock()
        mock_df_repartitioned.write = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.option.return_value = mock_writer

        with patch.object(generator, "_generate_rdd") as mock_rdd, \
            patch.object(generator, "_build_schema") as mock_schema:
                mock_rdd.return_value = MagicMock()
                mock_schema.return_value = MagicMock()

                config = DataGeneratorConfig(format="parquet", compression="snappy")
                columns = [ColumnSpec(name="id", data_type="int", min_value=1, max_value=100)]
                generator.generate("./output", config=config, columns=columns)

                # Verify parquet format was used
                mock_writer.format.assert_called_with("parquet")
                mock_writer.format().save.assert_called_once()

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_writes_delta(self) -> None:
        """Test generate writes data in delta format."""
        generator = DataGenerator()
        generator.spark = MagicMock()
        mock_df = MagicMock()
        generator.spark.createDataFrame.return_value = mock_df
        mock_df_repartitioned = MagicMock()
        mock_df.repartition.return_value = mock_df_repartitioned

        mock_writer = MagicMock()
        mock_df_repartitioned.write = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.option.return_value = mock_writer

        with patch.object(generator, "_generate_rdd") as mock_rdd, \
            patch.object(generator, "_build_schema") as mock_schema:
                mock_rdd.return_value = MagicMock()
                mock_schema.return_value = MagicMock()

                config = DataGeneratorConfig(format="delta")
                columns = [ColumnSpec(name="id", data_type="int", min_value=1, max_value=100)]
                generator.generate("./output", config=config, columns=columns)

                mock_writer.format.assert_called_with("delta")

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_writes_csv(self) -> None:
        """Test generate writes data in csv format."""
        generator = DataGenerator()
        generator.spark = MagicMock()
        mock_df = MagicMock()
        generator.spark.createDataFrame.return_value = mock_df
        mock_df_repartitioned = MagicMock()
        mock_df.repartition.return_value = mock_df_repartitioned

        mock_writer = MagicMock()
        mock_df_repartitioned.write = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.option.return_value = mock_writer

        with patch.object(generator, "_generate_rdd") as mock_rdd, \
            patch.object(generator, "_build_schema") as mock_schema:
                mock_rdd.return_value = MagicMock()
                mock_schema.return_value = MagicMock()

                config = DataGeneratorConfig(format="csv")
                columns = [ColumnSpec(name="id", data_type="int", min_value=1, max_value=100)]
                generator.generate("./output", config=config, columns=columns)

                mock_writer.option.assert_any_call("header", "true")

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_writes_orc(self) -> None:
        """Test generate writes data in orc format."""
        generator = DataGenerator()
        generator.spark = MagicMock()
        mock_df = MagicMock()
        generator.spark.createDataFrame.return_value = mock_df
        mock_df_repartitioned = MagicMock()
        mock_df.repartition.return_value = mock_df_repartitioned

        mock_writer = MagicMock()
        mock_df_repartitioned.write = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.option.return_value = mock_writer

        with patch.object(generator, "_generate_rdd") as mock_rdd, \
            patch.object(generator, "_build_schema") as mock_schema:
                mock_rdd.return_value = MagicMock()
                mock_schema.return_value = MagicMock()

                config = DataGeneratorConfig(format="orc")
                columns = [ColumnSpec(name="id", data_type="int", min_value=1, max_value=100)]
                generator.generate("./output", config=config, columns=columns)

                mock_writer.format.assert_called_with("orc")


class TestGenerateTabularEdgeCases:
    """Tests for generate_tabular edge cases."""

    def test_generate_tabular_string_columns(self) -> None:
        """Test generate_tabular creates string columns for i < num_cols // 2 (line 278)."""
        generator = DataGenerator()
        with patch.object(generator, "generate") as mock_generate:
            mock_generate.return_value = Path("./output")
            # Use 6 columns so some fall in the i < num_cols // 2 range (i < 3)
            generator.generate_tabular("./output", num_rows=100, num_cols=6)
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            columns = call_args.args[2]
            # Check that string columns were created
            str_cols = [
                c for c in columns if c.data_type == "string" and c.name.startswith("str_col_")
            ]
            assert len(str_cols) > 0

    def test_generate_tabular_numeric_columns(self) -> None:
        """Test generate_tabular creates numeric columns for i >= num_cols // 2."""
        generator = DataGenerator()
        with patch.object(generator, "generate") as mock_generate:
            mock_generate.return_value = Path("./output")
            generator.generate_tabular("./output", num_rows=100, num_cols=6)
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            columns = call_args.args[2]
            # Check that numeric columns were created
            num_cols = [c for c in columns if c.data_type == "double"]
            assert len(num_cols) > 0

    def test_generate_tabular_category_column(self) -> None:
        """Test generate_tabular creates category column at index 1."""
        generator = DataGenerator()
        with patch.object(generator, "generate") as mock_generate:
            mock_generate.return_value = Path("./output")
            generator.generate_tabular("./output", num_rows=100, num_cols=5)
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            columns = call_args.args[2]
            # Check category column
            assert columns[1].name == "category"
            assert columns[1].data_type == "string"
            assert columns[1].cardinality == 10

    def test_generate_tabular_with_all_kwargs(self) -> None:
        """Test generate_tabular passes all kwargs to DataGeneratorConfig."""
        generator = DataGenerator()
        with patch.object(generator, "generate") as mock_generate:
            mock_generate.return_value = Path("./output")
            generator.generate_tabular(
                "./output",
                num_rows=1000,
                num_cols=8,
                format="json",
                compression="gzip",
                null_ratio=0.1,
                skew_factor=2.0,
                random_seed=123,
            )
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            config = call_args.args[1]
            assert config.num_rows == 1000
            assert config.format == "json"
            assert config.compression == "gzip"
            assert config.null_ratio == 0.1
            assert config.skew_factor == 2.0
            assert config.random_seed == 123


class TestGenerateNestedEdgeCases:
    """Tests for generate_nested edge cases."""

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", False)
    def test_generate_nested_raises_when_pyspark_not_available(self) -> None:
        """Test generate_nested raises RuntimeError when PySpark not available (line 315)."""
        generator = DataGenerator()
        with pytest.raises(RuntimeError, match="PySpark required"):
            generator.generate_nested("./output")

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_nested_creates_nested_schema(self) -> None:
        """Test generate_nested creates correct nested schema."""
        generator = DataGenerator()
        with patch.object(generator, "generate") as mock_generate:
            mock_generate.return_value = Path("./output")
            generator.generate_nested("./output", num_rows=5000)
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            # Check that schema was passed
            assert call_args.kwargs.get("schema") is not None

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_nested_passes_kwargs(self) -> None:
        """Test generate_nested passes kwargs to DataGeneratorConfig."""
        generator = DataGenerator()
        with patch.object(generator, "generate") as mock_generate:
            mock_generate.return_value = Path("./output")
            generator.generate_nested("./output", num_rows=2000, compression="gzip")
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            config = call_args.args[1]
            assert config.num_rows == 2000
            assert config.compression == "gzip"
            assert config.format == "json"  # Default for nested


class TestBuildSchemaPySparkNotAvailable:
    """Tests for _build_schema when PySpark is not available."""

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", False)
    def test_build_schema_returns_none_when_pyspark_not_available(self) -> None:
        """Test _build_schema returns None when PySpark not available (line 368)."""
        generator = DataGenerator()
        columns = [ColumnSpec(name="test", data_type="string")]
        result = generator._build_schema(columns)
        assert result is None


class TestGenerateRDDDetailed:
    """Detailed tests for _generate_rdd method."""

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_rdd_generates_correct_number_of_rows(self) -> None:
        """Test _generate_rdd generates correct number of rows."""
        mock_spark = MagicMock()
        mock_rdd = MagicMock()
        mock_spark.sparkContext.parallelize.return_value = mock_rdd
        mock_rdd.map.return_value = mock_rdd

        generator = DataGenerator(spark=mock_spark)
        config = DataGeneratorConfig(num_rows=50, num_partitions=2, random_seed=42)
        columns = [ColumnSpec(name="id", data_type="int", min_value=1, max_value=100)]

        generator._generate_rdd(config, columns, MagicMock())

        # Verify parallelize was called with correct range
        mock_spark.sparkContext.parallelize.assert_called_once_with(range(50), 2)
        # Verify map was called to generate rows
        mock_rdd.map.assert_called_once()

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_rdd_calls_generate_row(self) -> None:
        """Test _generate_rdd calls generate_row for each row (lines 411-417)."""
        mock_spark = MagicMock()
        mock_rdd = MagicMock()
        mock_spark.sparkContext.parallelize.return_value = mock_rdd

        generator = DataGenerator(spark=mock_spark)
        config = DataGeneratorConfig(num_rows=10, random_seed=42)
        columns = [
            ColumnSpec(name="id", data_type="int", min_value=1, max_value=100),
        ]

        # Track calls to _generate_value
        gen_value_calls = []

        def mock_generate_value(col, cfg):
            gen_value_calls.append((col, cfg))
            return 42

        # Mock _generate_value
        with patch.object(generator, "_generate_value", side_effect=mock_generate_value):
            # Make map actually call the function
            def mock_map(func):
                # Call func for each row to simulate the map operation
                for i in range(config.num_rows):
                    func(i)
                return mock_rdd

            mock_rdd.map = mock_map
            generator._generate_rdd(config, columns, MagicMock())

            # Verify _generate_value was called (once for each row)
            assert len(gen_value_calls) == config.num_rows

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_rdd_with_multiple_columns(self) -> None:
        """Test _generate_rdd with multiple columns."""
        mock_spark = MagicMock()

        generator = DataGenerator(spark=mock_spark)
        config = DataGeneratorConfig(num_rows=5, random_seed=42)
        columns = [
            ColumnSpec(name="col1", data_type="int", min_value=1, max_value=10),
            ColumnSpec(name="col2", data_type="string", cardinality=5),
            ColumnSpec(name="col3", data_type="double", min_value=0.0, max_value=1.0),
        ]

        # Mock the RDD and map function
        mock_rdd = MagicMock()
        mock_spark.sparkContext.parallelize.return_value = mock_rdd

        generator._generate_rdd(config, columns, MagicMock())

        mock_spark.sparkContext.parallelize.assert_called_once_with(range(5), config.num_partitions)
        mock_rdd.map.assert_called_once()


class TestApplySkew:
    """Tests for _apply_skew method."""

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_apply_skew_modifies_dataframe(self) -> None:
        """Test _apply_skew modifies the DataFrame correctly (lines 505-525)."""
        generator = DataGenerator()
        mock_df = MagicMock()
        mock_skewed_df = MagicMock()
        mock_df.withColumn.return_value = mock_skewed_df

        columns = [
            ColumnSpec(name="category", data_type="string", cardinality=10),
        ]

        # Mock the pyspark functions used in _apply_skew
        with patch("pyspark.sql.functions.rand") as mock_rand:
            mock_rand.return_value = 0.5  # Ensure the condition is a number
            with patch("pyspark.sql.functions.lit"), \
                patch("pyspark.sql.functions.when") as mock_when:
                    mock_when.return_value = mock_when
                    mock_when.otherwise.return_value = "skewed_col"

                    result = generator._apply_skew(mock_df, columns, 2.0)

                    # Verify withColumn was called to apply skew
                    mock_df.withColumn.assert_called_once()
                    assert result == mock_skewed_df

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_apply_skew_returns_df_when_no_categorical_cols(self) -> None:
        """Test _apply_skew returns original DF when no categorical columns (line 510-511)."""
        generator = DataGenerator()
        mock_df = MagicMock()

        columns = [
            ColumnSpec(name="id", data_type="int"),
            ColumnSpec(name="score", data_type="double"),
        ]

        result = generator._apply_skew(mock_df, columns, 2.0)

        # Should return original df when no categorical columns
        assert result == mock_df
        mock_df.withColumn.assert_not_called()

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_apply_skew_uses_first_categorical_column(self) -> None:
        """Test _apply_skew uses the first categorical column found."""
        generator = DataGenerator()
        mock_df = MagicMock()
        mock_skewed_df = MagicMock()
        mock_df.withColumn.return_value = mock_skewed_df

        columns = [
            ColumnSpec(name="id", data_type="int"),
            ColumnSpec(name="category1", data_type="string", cardinality=10),
            ColumnSpec(name="category2", data_type="string", cardinality=20),
        ]

        # Mock the pyspark functions
        with patch("pyspark.sql.functions.rand") as mock_rand:
            mock_rand.return_value = 0.5  # Ensure the condition is a number
            with patch("pyspark.sql.functions.lit"), \
                patch("pyspark.sql.functions.when") as mock_when:
                    mock_when.return_value = mock_when
                    mock_when.otherwise.return_value = "skewed_col"

                    generator._apply_skew(mock_df, columns, 2.0)

                    # Verify withColumn was called with first categorical column name
                    call_args = mock_df.withColumn.call_args
                    assert call_args[0][0] == "category1"

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_apply_skew_filters_string_with_cardinality(self) -> None:
        """Test _apply_skew only considers string columns with cardinality (lines 505-508)."""
        generator = DataGenerator()
        mock_df = MagicMock()
        mock_skewed_df = MagicMock()
        mock_df.withColumn.return_value = mock_skewed_df

        columns = [
            ColumnSpec(name="str_no_cardinality", data_type="string"),  # Should NOT be used
            ColumnSpec(
                name="int_with_cardinality", data_type="int", cardinality=10
            ),  # Should NOT be used
            ColumnSpec(name="valid_category", data_type="string", cardinality=10),  # SHOULD be used
        ]

        # Mock the pyspark functions
        with patch("pyspark.sql.functions.rand") as mock_rand:
            mock_rand.return_value = 0.5  # Ensure the condition is a number
            with patch("pyspark.sql.functions.lit"), \
                patch("pyspark.sql.functions.when") as mock_when:
                    mock_when.return_value = mock_when
                    mock_when.otherwise.return_value = "skewed_col"

                    generator._apply_skew(mock_df, columns, 2.0)

                    # Verify withColumn was called with the valid category column
                    call_args = mock_df.withColumn.call_args
                    assert call_args[0][0] == "valid_category"


class TestPySparkImportFailure:
    """Tests for PySpark import failure path (lines 40-42)."""

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", False)
    def test_pyspark_not_available_warning_logged(self) -> None:
        """Test that PySpark not available warning would be logged (lines 40-42)."""
        # The warning is logged at module import time when PYSPARK_AVAILABLE is False
        # We can verify the behavior by checking that methods raise appropriate errors
        generator = DataGenerator()

        # _build_schema should return None
        columns = [ColumnSpec(name="test", data_type="string")]
        result = generator._build_schema(columns)
        assert result is None

        # generate should raise RuntimeError
        with pytest.raises(RuntimeError, match="PySpark required"):
            generator.generate("./output")

        # generate_nested should raise RuntimeError
        with pytest.raises(RuntimeError, match="PySpark required"):
            generator.generate_nested("./output")

    def test_pyspark_import_error_path(self) -> None:
        """Test the actual import error path (lines 40-42) by reloading
        module with pyspark mocked."""
        import builtins
        import importlib

        # Save the original __import__
        original_import = builtins.__import__

        # Mock __import__ to raise ImportError for pyspark
        def mock_import(name, *args, **kwargs):
            if name.startswith("pyspark"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = mock_import

            # Reload the module to trigger the import error path
            import spark_optima.data.generators

            importlib.reload(spark_optima.data.generators)

            # Check that PYSPARK_AVAILABLE is False
            assert spark_optima.data.generators.PYSPARK_AVAILABLE is False
        finally:
            builtins.__import__ = original_import
            # Reload again to restore the original state
            importlib.reload(spark_optima.data.generators)


class TestGenerateValueEdgeCases:
    """Additional tests for _generate_value edge cases."""

    def test_generate_value_boolean_type(self) -> None:
        """Test generating boolean values specifically."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="boolean")
        config = DataGeneratorConfig(null_ratio=0.0)
        value = generator._generate_value(col, config)
        assert isinstance(value, bool)

    def test_generate_value_timestamp_type(self) -> None:
        """Test generating timestamp values."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="timestamp")
        config = DataGeneratorConfig(null_ratio=0.0)
        from datetime import datetime

        value = generator._generate_value(col, config)
        assert isinstance(value, datetime)

    def test_generate_value_random_string_length(self) -> None:
        """Test that random strings have correct length."""
        generator = DataGenerator()
        col = ColumnSpec(name="test", data_type="string", cardinality=None)
        config = DataGeneratorConfig(null_ratio=0.0)

        # Generate several values and check lengths
        lengths = set()
        for _ in range(100):
            value = generator._generate_value(col, config)
            lengths.add(len(value))

        # All lengths should be between 5 and 20
        for length in lengths:
            assert 5 <= length <= 20


class TestGenerateMethodWriteEdgeCases:
    """Tests for generate method write edge cases."""

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_creates_output_directory(self) -> None:
        """Test that generate creates output directory if it doesn't exist."""
        generator = DataGenerator()
        generator.spark = MagicMock()
        mock_df = MagicMock()
        generator.spark.createDataFrame.return_value = mock_df
        mock_df_repartitioned = MagicMock()
        mock_df.repartition.return_value = mock_df_repartitioned

        mock_writer = MagicMock()
        mock_df_repartitioned.write = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.option.return_value = mock_writer

        with (patch.object(generator, "_generate_rdd") as mock_rdd,
              patch.object(generator, "_build_schema") as mock_schema,
              patch("pathlib.Path.mkdir") as mock_mkdir):
            mock_rdd.return_value = MagicMock()
            mock_schema.return_value = MagicMock()

            columns = [ColumnSpec(name="id", data_type="int", min_value=1, max_value=100)]
            generator.generate("./output/subdir", DataGeneratorConfig(), columns=columns)

            # Verify mkdir was called
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_with_custom_compression(self) -> None:
        """Test generate with custom compression option."""
        generator = DataGenerator()
        generator.spark = MagicMock()
        mock_df = MagicMock()
        generator.spark.createDataFrame.return_value = mock_df
        mock_df_repartitioned = MagicMock()
        mock_df.repartition.return_value = mock_df_repartitioned

        mock_writer = MagicMock()
        mock_df_repartitioned.write = mock_writer
        mock_writer.mode.return_value = mock_writer

        with patch.object(generator, "_generate_rdd") as mock_rdd, \
            patch.object(generator, "_build_schema") as mock_schema:
                mock_rdd.return_value = MagicMock()
                mock_schema.return_value = MagicMock()

                columns = [ColumnSpec(name="id", data_type="int", min_value=1, max_value=100)]
                config = DataGeneratorConfig(compression="gzip")
                generator.generate("./output", config=config, columns=columns)

                # Verify compression was set
                mock_writer.option.assert_any_call("compression", "gzip")

    @patch("spark_optima.data.generators.PYSPARK_AVAILABLE", True)
    def test_generate_with_no_compression_in_format_options(self) -> None:
        """Test generate when format has no compression in options."""
        generator = DataGenerator()
        generator.spark = MagicMock()
        mock_df = MagicMock()
        generator.spark.createDataFrame.return_value = mock_df
        mock_df_repartitioned = MagicMock()
        mock_df.repartition.return_value = mock_df_repartitioned

        # Track option calls
        option_calls = []

        class MockWriter:
            def mode(self, _mode):
                return self

            def option(self, key, value):
                option_calls.append((key, value))
                return self

            def format(self, _fmt):
                return self

            def save(self, _path):
                return None

        mock_df_repartitioned.write = MockWriter()

        with patch.object(generator, "_generate_rdd") as mock_rdd, \
            patch.object(generator, "_build_schema") as mock_schema:
                mock_rdd.return_value = MagicMock()
                mock_schema.return_value = MagicMock()

                columns = [ColumnSpec(name="id", data_type="int", min_value=1, max_value=100)]
                # delta format has overwriteSchema but no default compression
                config = DataGeneratorConfig(format="delta", compression="zstd")
                generator.generate("./output", config=config, columns=columns)

                # Verify compression was set even though not in default options
                assert ("compression", "zstd") in option_calls
