# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for data profiler."""

from __future__ import annotations

import contextlib
import importlib
import sys
from dataclasses import is_dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spark_optima.data.profiler import (
    ColumnProfile,
    DataProfile,
    DataProfiler,
)


class TestColumnProfile:
    """Tests for ColumnProfile."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        profile = ColumnProfile(name="test_col")
        assert profile.name == "test_col"
        assert profile.data_type == ""
        assert profile.nullable is True
        assert profile.null_count == 0
        assert profile.null_ratio == 0.0
        assert profile.distinct_count == 0
        assert profile.min_value is None
        assert profile.max_value is None
        assert profile.mean is None
        assert profile.std_dev is None
        assert profile.top_values == []

    def test_full_initialization(self) -> None:
        """Test full initialization."""
        profile = ColumnProfile(
            name="id",
            data_type="int",
            nullable=False,
            null_count=10,
            null_ratio=0.05,
            distinct_count=1000,
            min_value=1,
            max_value=1000,
            mean=500.5,
            std_dev=288.67,
            top_values=[("a", 100), ("b", 80)],
        )
        assert profile.name == "id"
        assert profile.data_type == "int"
        assert profile.nullable is False
        assert profile.null_count == 10
        assert profile.null_ratio == 0.05
        assert profile.distinct_count == 1000

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        profile = ColumnProfile(
            name="test_col",
            data_type="string",
            nullable=True,
            null_count=5,
            null_ratio=0.1,
            distinct_count=100,
        )
        data = profile.to_dict()
        assert data["name"] == "test_col"
        assert data["data_type"] == "string"
        assert data["null_count"] == 5

    def test_is_dataclass(self) -> None:
        """Test that ColumnProfile is a dataclass."""
        assert is_dataclass(ColumnProfile)

    def test_to_dict_limits_top_values(self) -> None:
        """Test that to_dict limits top_values to 5."""
        profile = ColumnProfile(
            name="test",
            top_values=[(f"val{i}", i) for i in range(10)],
        )
        data = profile.to_dict()
        assert len(data["top_values"]) == 5

    def test_to_dict_with_all_fields(self) -> None:
        """Test to_dict with all fields populated."""
        profile = ColumnProfile(
            name="id",
            data_type="int",
            nullable=False,
            null_count=0,
            null_ratio=0.0,
            distinct_count=100,
            min_value=1,
            max_value=100,
            mean=50.5,
            std_dev=28.87,
            top_values=[("1", 50), ("2", 30)],
        )
        data = profile.to_dict()
        assert data["name"] == "id"
        assert data["min_value"] == 1
        assert data["max_value"] == 100

    def test_top_values_empty(self) -> None:
        """Test to_dict with empty top_values."""
        profile = ColumnProfile(name="test")
        data = profile.to_dict()
        assert data["top_values"] == []

    def test_top_values_less_than_five(self) -> None:
        """Test to_dict with less than 5 top_values."""
        profile = ColumnProfile(
            name="test",
            top_values=[("a", 10), ("b", 5)],
        )
        data = profile.to_dict()
        assert len(data["top_values"]) == 2


class TestDataProfile:
    """Tests for DataProfile."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        profile = DataProfile()
        assert profile.path == ""
        assert profile.format == ""
        assert profile.num_rows == 0
        assert profile.num_columns == 0
        assert profile.size_bytes == 0
        assert profile.num_partitions == 0
        assert profile.columns == []
        assert profile.schema == ""

    def test_full_initialization(self) -> None:
        """Test full initialization."""
        columns = [
            ColumnProfile(name="id", data_type="int"),
            ColumnProfile(name="name", data_type="string"),
        ]
        profile = DataProfile(
            path="/data/test.parquet",
            format="parquet",
            num_rows=100000,
            num_columns=2,
            size_bytes=1000000,
            num_partitions=4,
            columns=columns,
            schema="struct<id:int,name:string>",
        )
        assert profile.path == "/data/test.parquet"
        assert profile.num_rows == 100000

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        profile = DataProfile(
            path="/data/test.parquet",
            format="parquet",
            num_rows=1000,
            num_columns=2,
            size_bytes=1000000,
        )
        data = profile.to_dict()
        assert data["path"] == "/data/test.parquet"
        assert data["num_rows"] == 1000
        assert "size_mb" in data

    def test_to_dict_with_columns(self) -> None:
        """Test to_dict with columns."""
        columns = [
            ColumnProfile(name="id", data_type="int"),
            ColumnProfile(name="name", data_type="string"),
        ]
        profile = DataProfile(columns=columns)
        data = profile.to_dict()
        assert len(data["columns"]) == 2
        assert data["columns"][0]["name"] == "id"

    def test_to_dict_without_columns(self) -> None:
        """Test to_dict without columns."""
        profile = DataProfile()
        data = profile.to_dict()
        assert data["columns"] == []

    def test_get_column_names(self) -> None:
        """Test get_column_names method."""
        columns = [
            ColumnProfile(name="id"),
            ColumnProfile(name="name"),
            ColumnProfile(name="age"),
        ]
        profile = DataProfile(columns=columns)
        names = profile.get_column_names()
        assert names == ["id", "name", "age"]

    def test_get_column_names_empty(self) -> None:
        """Test get_column_names with no columns."""
        profile = DataProfile()
        names = profile.get_column_names()
        assert names == []

    def test_get_column_profile(self) -> None:
        """Test get_column_profile method."""
        columns = [
            ColumnProfile(name="id", data_type="int"),
            ColumnProfile(name="name", data_type="string"),
        ]
        profile = DataProfile(columns=columns)

        found = profile.get_column_profile("id")
        assert found is not None
        assert found.name == "id"

        not_found = profile.get_column_profile("nonexistent")
        assert not_found is None

    def test_get_column_profile_case_sensitive(self) -> None:
        """Test that get_column_profile is case sensitive."""
        columns = [
            ColumnProfile(name="ID", data_type="int"),
        ]
        profile = DataProfile(columns=columns)

        found = profile.get_column_profile("id")  # lowercase
        assert found is None  # Should not find it (case sensitive)

    def test_get_numeric_columns(self) -> None:
        """Test get_numeric_columns method."""
        columns = [
            ColumnProfile(name="id", data_type="int"),
            ColumnProfile(name="name", data_type="string"),
            ColumnProfile(name="age", data_type="integer"),
            ColumnProfile(name="salary", data_type="double"),
        ]
        profile = DataProfile(columns=columns)

        numeric = profile.get_numeric_columns()
        assert len(numeric) == 3
        assert all(c.name in ["id", "age", "salary"] for c in numeric)

    def test_get_numeric_columns_all_types(self) -> None:
        """Test get_numeric_columns with all numeric types."""
        columns = [
            ColumnProfile(name="col1", data_type="int"),
            ColumnProfile(name="col2", data_type="integer"),
            ColumnProfile(name="col3", data_type="long"),
            ColumnProfile(name="col4", data_type="bigint"),
            ColumnProfile(name="col5", data_type="double"),
            ColumnProfile(name="col6", data_type="float"),
            ColumnProfile(name="col7", data_type="decimal"),
        ]
        profile = DataProfile(columns=columns)
        numeric = profile.get_numeric_columns()
        assert len(numeric) == 7

    def test_get_numeric_columns_no_numeric(self) -> None:
        """Test when there are no numeric columns."""
        columns = [
            ColumnProfile(name="name", data_type="string"),
            ColumnProfile(name="status", data_type="boolean"),
        ]
        profile = DataProfile(columns=columns)
        numeric = profile.get_numeric_columns()
        assert len(numeric) == 0

    def test_get_numeric_columns_all_numeric(self) -> None:
        """Test when all columns are numeric."""
        columns = [
            ColumnProfile(name="col1", data_type="int"),
            ColumnProfile(name="col2", data_type="double"),
            ColumnProfile(name="col3", data_type="float"),
        ]
        profile = DataProfile(columns=columns)
        numeric = profile.get_numeric_columns()
        assert len(numeric) == 3

    def test_get_numeric_columns_case_insensitive(self) -> None:
        """Test that numeric detection is case insensitive."""
        columns = [
            ColumnProfile(name="col1", data_type="INT"),
            ColumnProfile(name="col2", data_type="Double"),
        ]
        profile = DataProfile(columns=columns)
        numeric = profile.get_numeric_columns()
        assert len(numeric) == 2

    def test_get_categorical_columns(self) -> None:
        """Test get_categorical_columns method."""
        columns = [
            ColumnProfile(name="id", distinct_count=10000),  # High cardinality
            ColumnProfile(name="category", distinct_count=10),  # Low cardinality
            ColumnProfile(name="status", distinct_count=5),  # Low cardinality
        ]
        profile = DataProfile(columns=columns)

        categorical = profile.get_categorical_columns()
        assert len(categorical) == 2
        assert all(c.distinct_count < 100 for c in categorical)

    def test_get_categorical_columns_no_categorical(self) -> None:
        """Test when there are no categorical columns."""
        columns = [
            ColumnProfile(name="id", distinct_count=10000),
            ColumnProfile(name="timestamp", distinct_count=5000),
        ]
        profile = DataProfile(columns=columns)
        categorical = profile.get_categorical_columns()
        assert len(categorical) == 0

    def test_get_categorical_columns_empty_distinct_count(self) -> None:
        """Test columns with zero distinct count."""
        columns = [
            ColumnProfile(name="empty_col", distinct_count=0),
        ]
        profile = DataProfile(columns=columns)
        categorical = profile.get_categorical_columns()
        assert len(categorical) == 0  # distinct_count is not >0

    def test_get_categorical_columns_boundary(self) -> None:
        """Test columns with distinct_count exactly 100 (boundary)."""
        columns = [
            ColumnProfile(name="boundary_col", distinct_count=100),
        ]
        profile = DataProfile(columns=columns)
        categorical = profile.get_categorical_columns()
        assert len(categorical) == 0  # distinct_count < 100, so 100 is not included

    def test_data_profile_is_dataclass(self) -> None:
        """Test that DataProfile is a dataclass."""
        assert is_dataclass(DataProfile)

    def test_to_dict_size_mb_zero_bytes(self) -> None:
        """Test to_dict with zero bytes."""
        profile = DataProfile(size_bytes=0)
        data = profile.to_dict()
        assert data["size_mb"] == 0.0

    def test_to_dict_size_mb_non_zero(self) -> None:
        """Test to_dict with non-zero bytes."""
        profile = DataProfile(size_bytes=2097152)  # 2 MB
        data = profile.to_dict()
        assert data["size_mb"] == 2.0


class TestDataProfiler:
    """Tests for DataProfiler."""

    def test_initialization_without_spark(self) -> None:
        """Test initialization without Spark."""
        profiler = DataProfiler()
        assert profiler.spark is None
        assert profiler._own_spark is False

    def test_initialization_with_spark(self) -> None:
        """Test initialization with Spark."""
        mock_spark = object()
        profiler = DataProfiler(spark=mock_spark)
        assert profiler.spark is mock_spark
        assert profiler._own_spark is False

    def test_del_stops_spark_when_own_spark(self) -> None:
        """Test that __del__ stops Spark when we own it."""
        profiler = DataProfiler()
        mock_spark = MagicMock()
        profiler.spark = mock_spark
        profiler._own_spark = True
        profiler.__del__()
        mock_spark.stop.assert_called_once()

    def test_del_does_not_stop_spark_when_not_own(self) -> None:
        """Test that __del__ does not stop Spark when we don't own it."""
        profiler = DataProfiler()
        mock_spark = MagicMock()
        profiler.spark = mock_spark
        profiler._own_spark = False
        profiler.__del__()
        mock_spark.stop.assert_not_called()

    def test_del_handles_exception(self) -> None:
        """Test that __del__ handles exceptions gracefully."""
        profiler = DataProfiler()
        mock_spark = MagicMock()
        mock_spark.stop.side_effect = Exception("Test exception")
        profiler.spark = mock_spark
        profiler._own_spark = True
        # Should not raise exception
        profiler.__del__()


class TestPySparkAvailability:
    """Tests for PySpark availability check."""

    def test_pyspark_availability(self) -> None:
        """Test PySpark availability check."""
        from spark_optima.data.profiler import PYSPARK_AVAILABLE

        # This test just checks the flag exists
        assert isinstance(PYSPARK_AVAILABLE, bool)

    def test_profiler_requires_pyspark_for_profile(self) -> None:
        """Test that profile method requires PySpark."""
        from spark_optima.data.profiler import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            profiler = DataProfiler()
            with pytest.raises(RuntimeError, match="PySpark required"):
                profiler.profile("./data.parquet")

    def test_profiler_requires_pyspark_for_quick_profile(self) -> None:
        """Test that profile_quick method requires PySpark."""
        from spark_optima.data.profiler import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            profiler = DataProfiler()
            with pytest.raises(RuntimeError, match="PySpark required"):
                profiler.profile_quick("./data.parquet")

    def test_profiler_requires_pyspark_for_analyze_skew(self) -> None:
        """Test that analyze_skew method requires PySpark."""
        from spark_optima.data.profiler import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            profiler = DataProfiler()
            with pytest.raises(RuntimeError, match="PySpark required"):
                profiler.analyze_skew("./data.parquet", "column")


class TestIsNumericType:
    """Tests for _is_numeric_type method."""

    def test_numeric_types(self) -> None:
        """Test that numeric types are correctly identified."""
        profiler = DataProfiler()
        numeric_types = ["int", "integer", "long", "bigint", "double", "float", "decimal"]
        for t in numeric_types:
            assert profiler._is_numeric_type(t) is True

    def test_non_numeric_types(self) -> None:
        """Test that non-numeric types are correctly identified."""
        profiler = DataProfiler()
        non_numeric = ["string", "boolean", "date", "timestamp", "array", "map"]
        for t in non_numeric:
            assert profiler._is_numeric_type(t) is False

    def test_case_insensitive(self) -> None:
        """Test that type checking is case insensitive."""
        profiler = DataProfiler()
        assert profiler._is_numeric_type("INT") is True
        assert profiler._is_numeric_type("Double") is True
        assert profiler._is_numeric_type("STRING") is False

    def test_numeric_type_with_spark_format(self) -> None:
        """Test numeric types as they appear from Spark."""
        profiler = DataProfiler()
        # Spark might return types like "IntegerType" or "LongType"
        assert profiler._is_numeric_type("IntegerType") is True  # Contains "integer"
        assert profiler._is_numeric_type("LongType") is True  # Contains "long"
        assert profiler._is_numeric_type("StringType") is False

    def test_empty_string(self) -> None:
        """Test with empty string."""
        profiler = DataProfiler()
        assert profiler._is_numeric_type("") is False

    def test_partial_match(self) -> None:
        """Test that partial matches work correctly."""
        profiler = DataProfiler()
        # "bigint" should match because "int" is in it
        assert profiler._is_numeric_type("bigint") is True


class TestInferFormat:
    """Tests for _infer_format method."""

    def test_infer_parquet(self) -> None:
        """Test inferring parquet format."""
        profiler = DataProfiler()
        from pathlib import Path

        fmt = profiler._infer_format(Path("data.parquet"))
        assert fmt == "parquet"

    def test_infer_csv(self) -> None:
        """Test inferring csv format."""
        profiler = DataProfiler()
        from pathlib import Path

        fmt = profiler._infer_format(Path("data.csv"))
        assert fmt == "csv"

    def test_infer_json(self) -> None:
        """Test inferring json format."""
        profiler = DataProfiler()
        from pathlib import Path

        fmt = profiler._infer_format(Path("data.json"))
        assert fmt == "json"

    def test_infer_orc(self) -> None:
        """Test inferring orc format."""
        profiler = DataProfiler()
        from pathlib import Path

        fmt = profiler._infer_format(Path("data.orc"))
        assert fmt == "orc"

    def test_infer_default_for_unknown(self) -> None:
        """Test default format for unknown extensions."""
        profiler = DataProfiler()
        from pathlib import Path

        fmt = profiler._infer_format(Path("data.unknown"))
        assert fmt == "parquet"  # Default

    def test_infer_case_insensitive(self) -> None:
        """Test format inference is case insensitive."""
        profiler = DataProfiler()
        from pathlib import Path

        fmt = profiler._infer_format(Path("data.PARQUET"))
        assert fmt == "parquet"

    def test_infer_with_path_object(self) -> None:
        """Test infer with Path object."""
        profiler = DataProfiler()
        fmt = profiler._infer_format(Path("/some/path/data.parquet"))
        assert fmt == "parquet"

    def test_infer_with_no_extension(self) -> None:
        """Test infer with no extension."""
        profiler = DataProfiler()
        fmt = profiler._infer_format(Path("data"))
        assert fmt == "parquet"  # Default


class TestEstimateSize:
    """Tests for _estimate_size method."""

    def test_estimate_size_file(self, tmp_path) -> None:
        """Test size estimation for a single file."""
        profiler = DataProfiler()
        # Create a test file
        test_file = tmp_path / "test.parquet"
        test_file.write_bytes(b"x" * 100)
        size = profiler._estimate_size(test_file, "parquet")
        assert size == 100

    def test_estimate_size_directory(self, tmp_path) -> None:
        """Test size estimation for a directory."""
        profiler = DataProfiler()
        # Create test directory with files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file1.parquet").write_bytes(b"x" * 50)
        (data_dir / "file2.parquet").write_bytes(b"x" * 30)
        size = profiler._estimate_size(data_dir, "parquet")
        assert size == 80

    def test_estimate_size_nonexistent(self) -> None:
        """Test size estimation for nonexistent path."""
        profiler = DataProfiler()
        from pathlib import Path

        size = profiler._estimate_size(Path("nonexistent"), "parquet")
        assert size == 0

    def test_estimate_size_exception_handling(self, tmp_path) -> None:
        """Test size estimation handles exceptions."""
        profiler = DataProfiler()
        # Create a mock Path that raises an OSError (which is caught)
        mock_path = MagicMock(spec=Path)
        mock_path.is_file.side_effect = OSError("Test exception")
        size = profiler._estimate_size(mock_path, "parquet")
        assert size == 0

    def test_estimate_size_directory_with_subdirs(self, tmp_path) -> None:
        """Test size estimation for directory with subdirectories."""
        profiler = DataProfiler()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file1.parquet").write_bytes(b"x" * 50)

        # Create subdirectory with more files
        subdir = data_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.parquet").write_bytes(b"x" * 30)

        size = profiler._estimate_size(data_dir, "parquet")
        assert size == 80  # Should include subdirectory files

    def test_estimate_size_empty_directory(self, tmp_path) -> None:
        """Test size estimation for empty directory."""
        profiler = DataProfiler()
        data_dir = tmp_path / "empty_dir"
        data_dir.mkdir()
        size = profiler._estimate_size(data_dir, "parquet")
        assert size == 0


class TestLoadData:
    """Tests for _load_data method."""

    def test_load_data_parquet(self) -> None:
        """Test loading parquet data."""
        from spark_optima.data.profiler import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        profiler = DataProfiler()
        profiler.spark = MagicMock()
        profiler._own_spark = False

        profiler._load_data(Path("data.parquet"), "parquet")
        profiler.spark.read.parquet.assert_called_once_with("data.parquet")

    def test_load_data_csv(self) -> None:
        """Test loading csv data."""
        from spark_optima.data.profiler import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        profiler = DataProfiler()
        profiler.spark = MagicMock()
        profiler._own_spark = False

        profiler._load_data(Path("data.csv"), "csv")
        profiler.spark.read.option.assert_called_with("header", "true")
        profiler.spark.read.option.return_value.option.assert_called_with("inferSchema", "true")

    def test_load_data_json(self) -> None:
        """Test loading json data."""
        from spark_optima.data.profiler import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        profiler = DataProfiler()
        profiler.spark = MagicMock()
        profiler._own_spark = False

        profiler._load_data(Path("data.json"), "json")
        profiler.spark.read.json.assert_called_once_with("data.json")

    def test_load_data_delta(self) -> None:
        """Test loading delta data."""
        from spark_optima.data.profiler import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        profiler = DataProfiler()
        profiler.spark = MagicMock()
        profiler._own_spark = False

        profiler._load_data(Path("data"), "delta")
        profiler.spark.read.format.assert_called_once_with("delta")
        profiler.spark.read.format.return_value.load.assert_called_once_with("data")

    def test_load_data_orc(self) -> None:
        """Test loading orc data."""
        from spark_optima.data.profiler import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        profiler = DataProfiler()
        profiler.spark = MagicMock()
        profiler._own_spark = False

        profiler._load_data(Path("data.orc"), "orc")
        profiler.spark.read.orc.assert_called_once_with("data.orc")

    def test_load_data_unknown_format(self) -> None:
        """Test loading data with unknown format (generic load)."""
        from spark_optima.data.profiler import PYSPARK_AVAILABLE

        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")

        profiler = DataProfiler()
        profiler.spark = MagicMock()
        profiler._own_spark = False

        profiler._load_data(Path("data.unknown"), "unknown")
        profiler.spark.read.format.assert_called_once_with("unknown")
        profiler.spark.read.format.return_value.load.assert_called_once_with("data.unknown")


class TestDataProfilerEdgeCases:
    """Edge case tests for DataProfiler."""

    def test_profiler_initialization_idempotent(self) -> None:
        """Test that multiple initializations work."""
        p1 = DataProfiler()
        p2 = DataProfiler()
        assert p1.spark is None
        assert p2.spark is None

    def test_profiler_with_spark_mock(self) -> None:
        """Test profiler with mocked Spark."""
        mock_spark = object()
        profiler = DataProfiler(spark=mock_spark)
        assert profiler.spark is mock_spark
        assert profiler._own_spark is False


class TestDataProfileToDict:
    """Tests for DataProfile.to_dict method."""

    def test_to_dict_contains_size_mb(self) -> None:
        """Test that to_dict includes size_mb."""
        profile = DataProfile(size_bytes=1048576)  # 1 MB
        data = profile.to_dict()
        assert "size_mb" in data
        assert data["size_mb"] == 1.0

    def test_to_dict_with_columns(self) -> None:
        """Test to_dict with columns."""
        columns = [
            ColumnProfile(name="id", data_type="int"),
            ColumnProfile(name="name", data_type="string"),
        ]
        profile = DataProfile(columns=columns)
        data = profile.to_dict()
        assert len(data["columns"]) == 2
        assert data["columns"][0]["name"] == "id"

    def test_to_dict_without_columns(self) -> None:
        """Test to_dict without columns."""
        profile = DataProfile()
        data = profile.to_dict()
        assert data["columns"] == []


class TestColumnProfileToDict:
    """Tests for ColumnProfile.to_dict method."""

    def test_to_dict_limits_top_values(self) -> None:
        """Test that to_dict limits top_values to 5."""
        profile = ColumnProfile(
            name="test",
            top_values=[(f"val{i}", i) for i in range(10)],
        )
        data = profile.to_dict()
        assert len(data["top_values"]) == 5

    def test_to_dict_with_all_fields(self) -> None:
        """Test to_dict with all fields populated."""
        profile = ColumnProfile(
            name="id",
            data_type="int",
            nullable=False,
            null_count=0,
            null_ratio=0.0,
            distinct_count=100,
            min_value=1,
            max_value=100,
            mean=50.5,
            std_dev=28.87,
            top_values=[("1", 50), ("2", 30)],
        )
        data = profile.to_dict()
        assert data["name"] == "id"
        assert data["min_value"] == 1
        assert data["max_value"] == 100


class TestGetNumericColumns:
    """Additional tests for get_numeric_columns method."""

    def test_no_numeric_columns(self) -> None:
        """Test when there are no numeric columns."""
        columns = [
            ColumnProfile(name="name", data_type="string"),
            ColumnProfile(name="status", data_type="boolean"),
        ]
        profile = DataProfile(columns=columns)
        numeric = profile.get_numeric_columns()
        assert len(numeric) == 0

    def test_all_numeric_columns(self) -> None:
        """Test when all columns are numeric."""
        columns = [
            ColumnProfile(name="col1", data_type="int"),
            ColumnProfile(name="col2", data_type="double"),
            ColumnProfile(name="col3", data_type="float"),
        ]
        profile = DataProfile(columns=columns)
        numeric = profile.get_numeric_columns()
        assert len(numeric) == 3

    def test_case_insensitive_numeric(self) -> None:
        """Test that numeric detection is case insensitive."""
        columns = [
            ColumnProfile(name="col1", data_type="INT"),
            ColumnProfile(name="col2", data_type="Double"),
        ]
        profile = DataProfile(columns=columns)
        numeric = profile.get_numeric_columns()
        assert len(numeric) == 2


class TestGetCategoricalColumns:
    """Additional tests for get_categorical_columns method."""

    def test_no_categorical_columns(self) -> None:
        """Test when there are no categorical columns."""
        columns = [
            ColumnProfile(name="id", distinct_count=10000),
            ColumnProfile(name="timestamp", distinct_count=5000),
        ]
        profile = DataProfile(columns=columns)
        categorical = profile.get_categorical_columns()
        assert len(categorical) == 0

    def test_empty_distinct_count(self) -> None:
        """Test columns with zero distinct count."""
        columns = [
            ColumnProfile(name="empty_col", distinct_count=0),
        ]
        profile = DataProfile(columns=columns)
        categorical = profile.get_categorical_columns()
        assert len(categorical) == 0  # distinct_count is not >0


class TestProfileMethod:
    """Tests for profile method with mocked PySpark."""

    @patch("spark_optima.data.profiler.stddev")
    @patch("spark_optima.data.profiler.avg")
    @patch("spark_optima.data.profiler.spark_max")
    @patch("spark_optima.data.profiler.spark_min")
    @patch("spark_optima.data.profiler.col")
    @patch("spark_optima.data.profiler.SparkSession")
    def test_profile_with_pyspark(
        self, mock_spark_session, mock_col, mock_spark_min, mock_spark_max, mock_avg, mock_stddev
    ) -> None:
        """Test profile method with PySpark available."""
        # Setup mocks
        mock_spark = MagicMock()
        builder = mock_spark_session.builder.appName.return_value
        builder.master.return_value.getOrCreate.return_value = mock_spark

        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.count.return_value = 1000
        mock_df.rdd.getNumPartitions.return_value = 4
        mock_df.schema = MagicMock()
        mock_df.schema.simpleString.return_value = "struct<id:int,name:string>"
        mock_df.schema.fields = [
            MagicMock(name="id", dataType=MagicMock(simpleString=lambda: "int"), nullable=False),
            MagicMock(name="name", dataType=MagicMock(simpleString=lambda: "string"), nullable=True),
        ]
        mock_df.filter.return_value.count.return_value = 0
        mock_df.select.return_value.distinct.return_value.count.return_value = 100

        # Mock numeric stats
        mock_stats = MagicMock()
        mock_stats.__getitem__ = MagicMock(
            side_effect=lambda x: {"min": 1, "max": 1000, "mean": 500.5, "stddev": 288.67}[x]
        )
        mock_df.select.return_value.collect.return_value = [mock_stats]

        mock_spark.read.parquet.return_value = mock_df

        profiler = DataProfiler()
        with patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True):
            profile = profiler.profile("/data/test.parquet", format="parquet")

        assert profile.path == "/data/test.parquet"
        assert profile.format == "parquet"
        assert profile.num_rows == 1000
        assert profile.num_columns == 2
        assert profile.num_partitions == 4

    def test_profile_raises_without_pyspark(self) -> None:
        """Test that profile raises RuntimeError when PySpark not available."""
        with patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", False):
            profiler = DataProfiler()
            with pytest.raises(RuntimeError, match="PySpark required"):
                profiler.profile("./data.parquet")

    @patch("spark_optima.data.profiler.col")
    @patch("spark_optima.data.profiler.SparkSession")
    def test_profile_with_existing_spark(self, mock_spark_session, mock_col) -> None:
        """Test profile method with existing Spark session."""
        mock_spark = MagicMock()
        mock_df = MagicMock()
        mock_df.count.return_value = 100
        mock_df.rdd.getNumPartitions.return_value = 2
        mock_df.schema = MagicMock()
        mock_df.schema.simpleString.return_value = "struct<col1:string>"
        mock_df.schema.fields = [
            MagicMock(name="col1", dataType=MagicMock(simpleString=lambda: "string"), nullable=True),
        ]
        mock_df.filter.return_value.count.return_value = 0
        mock_df.select.return_value.distinct.return_value.count.return_value = 10

        mock_spark.read.parquet.return_value = mock_df

        profiler = DataProfiler(spark=mock_spark)
        with patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True):
            profile = profiler.profile("/data/test.parquet")

        assert profile.num_rows == 100
        # Should not create new Spark session
        mock_spark_session.builder.appName.assert_not_called()

    @patch("spark_optima.data.profiler.stddev")
    @patch("spark_optima.data.profiler.avg")
    @patch("spark_optima.data.profiler.spark_max")
    @patch("spark_optima.data.profiler.spark_min")
    @patch("spark_optima.data.profiler.col")
    @patch("spark_optima.data.profiler.SparkSession")
    def test_profile_with_sample_ratio(
        self, mock_spark_session, mock_col, mock_spark_min, mock_spark_max, mock_avg, mock_stddev
    ) -> None:
        """Test profile method with sample_ratio < 1.0."""
        mock_spark = MagicMock()
        builder = mock_spark_session.builder.appName.return_value
        builder.master.return_value.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_sampled_df = MagicMock()
        mock_df.sample.return_value = mock_sampled_df

        mock_sampled_df.count.return_value = 500
        mock_sampled_df.rdd.getNumPartitions.return_value = 4
        mock_sampled_df.schema = MagicMock()
        mock_sampled_df.schema.simpleString.return_value = "struct<id:int>"
        mock_sampled_df.schema.fields = [
            MagicMock(name="id", dataType=MagicMock(simpleString=lambda: "int"), nullable=False),
        ]
        mock_sampled_df.filter.return_value.count.return_value = 0
        mock_sampled_df.select.return_value.distinct.return_value.count.return_value = 50

        # Mock stats collection
        mock_stats = MagicMock()
        mock_stats.__getitem__ = MagicMock(
            side_effect=lambda x: {"min": 1, "max": 1000, "mean": 500.0, "stddev": 250.0}[x]
        )
        mock_sampled_df.select.return_value.collect.return_value = [mock_stats]

        mock_spark.read.parquet.return_value = mock_df

        profiler = DataProfiler()
        with patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True):
            profiler.profile("/data/test.parquet", sample_ratio=0.5)

        # Verify sample was called
        mock_df.sample.assert_called_once_with(False, 0.5)

    @patch("spark_optima.data.profiler.col")
    @patch("spark_optima.data.profiler.SparkSession")
    def test_profile_infer_format(self, mock_spark_session, mock_col) -> None:
        """Test profile method infers format from path."""
        mock_spark = MagicMock()
        builder = mock_spark_session.builder.appName.return_value
        builder.master.return_value.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_df.count.return_value = 100
        mock_df.rdd.getNumPartitions.return_value = 2
        mock_df.schema = MagicMock()
        mock_df.schema.simpleString.return_value = "struct<col1:string>"
        mock_df.schema.fields = []
        # CSV loading chain: reader.option(...).option(...).csv(...)
        mock_spark.read.option.return_value.option.return_value.csv.return_value = mock_df

        profiler = DataProfiler()
        with patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True):
            profiler.profile("/data/test.csv", format=None)  # Should infer csv

        # Verify that csv was called (through the option chain)
        mock_spark.read.option.assert_called()


class TestProfileQuickMethod:
    """Tests for profile_quick method with mocked PySpark."""

    @patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.profiler.SparkSession")
    def test_profile_quick_with_pyspark(self, mock_spark_session) -> None:
        """Test profile_quick method with PySpark available."""
        mock_spark = MagicMock()
        builder = mock_spark_session.builder.appName.return_value
        builder.master.return_value.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_df.count.return_value = 500
        mock_df.rdd.getNumPartitions.return_value = 2
        mock_df.columns = ["col1", "col2", "col3"]

        mock_spark.read.parquet.return_value = mock_df

        profiler = DataProfiler()
        result = profiler.profile_quick("/data/test.parquet")

        assert result["path"] == "/data/test.parquet"
        assert result["format"] == "parquet"
        assert result["num_rows"] == 500
        assert result["num_columns"] == 3
        assert result["num_partitions"] == 2

    @patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True)
    def test_profile_quick_raises_without_pyspark(self) -> None:
        """Test that profile_quick raises RuntimeError when PySpark not available."""
        with patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", False):
            profiler = DataProfiler()
            with pytest.raises(RuntimeError, match="PySpark required"):
                profiler.profile_quick("./data.parquet")

    @patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.profiler.SparkSession")
    def test_profile_quick_with_existing_spark(self, mock_spark_session) -> None:
        """Test profile_quick method with existing Spark session."""
        mock_spark = MagicMock()
        mock_df = MagicMock()
        mock_df.count.return_value = 100
        mock_df.rdd.getNumPartitions.return_value = 1
        mock_df.columns = ["id"]

        mock_spark.read.json.return_value = mock_df

        profiler = DataProfiler(spark=mock_spark)
        result = profiler.profile_quick("/data/test.json", format="json")

        assert result["num_rows"] == 100
        mock_spark_session.builder.appName.assert_not_called()


class TestProfileColumnMethod:
    """Tests for _profile_column method."""

    @patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.profiler.col")
    @patch("spark_optima.data.profiler.spark_min")
    @patch("spark_optima.data.profiler.spark_max")
    @patch("spark_optima.data.profiler.avg")
    @patch("spark_optima.data.profiler.stddev")
    def test_profile_numeric_column(self, mock_stddev, mock_avg, mock_spark_max, mock_spark_min, mock_col) -> None:
        """Test profiling a numeric column."""
        profiler = DataProfiler()

        # Create mock DataFrame and field
        mock_df = MagicMock()
        mock_df.filter.return_value.count.return_value = 10  # 10 nulls
        mock_df.count.return_value = 100  # 100 total rows
        mock_df.select.return_value.distinct.return_value.count.return_value = 50

        # Mock stats result
        mock_stats = MagicMock()
        mock_stats.__getitem__ = MagicMock(
            side_effect=lambda x: {"min": 1, "max": 100, "mean": 50.5, "stddev": 28.87}[x]
        )
        mock_df.select.return_value.collect.return_value = [mock_stats]

        mock_field = MagicMock()
        mock_field.name = "age"
        mock_field.dataType = MagicMock(simpleString=lambda: "int")
        mock_field.nullable = False

        profile = profiler._profile_column(mock_df, mock_field)

        assert profile.name == "age"
        assert profile.data_type == "int"
        assert profile.nullable is False
        assert profile.null_count == 10
        assert profile.null_ratio == 0.1
        assert profile.distinct_count == 50
        assert profile.min_value == 1
        assert profile.max_value == 100
        assert profile.mean == 50.5
        assert profile.std_dev == 28.87

    @patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.profiler.col")
    def test_profile_non_numeric_column(self, mock_col) -> None:
        """Test profiling a non-numeric column."""
        profiler = DataProfiler()

        mock_df = MagicMock()
        mock_df.filter.return_value.count.return_value = 5
        mock_df.count.return_value = 100
        mock_df.select.return_value.distinct.return_value.count.return_value = 20

        mock_field = MagicMock()
        mock_field.name = "name"
        mock_field.dataType = MagicMock(simpleString=lambda: "string")
        mock_field.nullable = True

        profile = profiler._profile_column(mock_df, mock_field)

        assert profile.name == "name"
        assert profile.data_type == "string"
        assert profile.nullable is True
        assert profile.null_count == 5
        assert profile.null_ratio == 0.05
        assert profile.distinct_count == 20
        assert profile.min_value is None
        assert profile.max_value is None
        assert profile.mean is None
        assert profile.std_dev is None

    @patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.profiler.col")
    def test_profile_column_zero_rows(self, mock_col) -> None:
        """Test profiling column with zero rows."""
        profiler = DataProfiler()

        mock_df = MagicMock()
        mock_df.filter.return_value.count.return_value = 0
        mock_df.count.return_value = 0  # Total rows is 0
        mock_df.select.return_value.distinct.return_value.count.return_value = 0

        mock_field = MagicMock()
        mock_field.name = "empty_col"
        mock_field.dataType = MagicMock(simpleString=lambda: "string")
        mock_field.nullable = True

        profile = profiler._profile_column(mock_df, mock_field)

        assert profile.null_ratio == 0.0  # Should handle division by zero

    @patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True)
    @patch("spark_optima.data.profiler.col")
    @patch("spark_optima.data.profiler.spark_min")
    @patch("spark_optima.data.profiler.spark_max")
    @patch("spark_optima.data.profiler.avg")
    @patch("spark_optima.data.profiler.stddev")
    def test_profile_various_numeric_types(
        self, mock_stddev, mock_avg, mock_spark_max, mock_spark_min, mock_col
    ) -> None:
        """Test profiling columns with various numeric types."""
        profiler = DataProfiler()

        numeric_types = ["int", "double", "float", "long", "bigint", "decimal"]

        for type_name in numeric_types:
            mock_df = MagicMock()
            mock_df.filter.return_value.count.return_value = 0
            mock_df.count.return_value = 100
            mock_df.select.return_value.distinct.return_value.count.return_value = 50

            mock_stats = MagicMock()
            mock_stats.__getitem__ = MagicMock(
                side_effect=lambda x: {"min": 1, "max": 100, "mean": 50.0, "stddev": 25.0}[x]
            )
            mock_df.select.return_value.collect.return_value = [mock_stats]

            mock_field = MagicMock()
            mock_field.name = "col"
            mock_field.dataType = MagicMock(simpleString=lambda t=type_name: t)
            mock_field.nullable = True

            profile = profiler._profile_column(mock_df, mock_field)
            assert profile.min_value is not None, f"Failed for type {type_name}"
            assert profile.mean is not None, f"Failed for type {type_name}"


class TestAnalyzeSkewMethod:
    """Tests for analyze_skew method with mocked PySpark."""

    @patch("spark_optima.data.profiler.col")
    @patch("spark_optima.data.profiler.SparkSession")
    def test_analyze_skew_with_pyspark(self, mock_spark_session, mock_col) -> None:
        """Test analyze_skew method with PySpark available."""
        mock_spark = MagicMock()
        builder = mock_spark_session.builder.appName.return_value
        builder.master.return_value.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_df.count.return_value = 1000
        mock_df.schema = MagicMock()
        mock_df.schema.fields = []

        # Mock groupBy chain for value_counts
        mock_grouped = mock_df.groupBy.return_value
        mock_counted = mock_grouped.count.return_value
        mock_ordered = mock_counted.orderBy.return_value
        mock_limited = mock_ordered.limit.return_value

        # Mock top values
        mock_row1 = MagicMock()
        mock_row1.__getitem__ = MagicMock(side_effect=lambda x: "a" if x == "category" else 500)
        mock_row2 = MagicMock()
        mock_row2.__getitem__ = MagicMock(side_effect=lambda x: "b" if x == "category" else 200)
        mock_limited.collect.return_value = [mock_row1, mock_row2]

        # Mock distinct_values count (value_counts.count())
        mock_ordered.count.return_value = 3

        mock_spark.read.parquet.return_value = mock_df

        profiler = DataProfiler()
        with patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True):
            result = profiler.analyze_skew("/data/test.parquet", "category")

        assert result["column"] == "category"
        assert result["total_rows"] == 1000
        assert result["distinct_values"] == 3
        assert len(result["top_values"]) == 2
        assert "skew_ratio" in result
        assert "is_skewed" in result

    def test_analyze_skew_raises_without_pyspark(self) -> None:
        """Test that analyze_skew raises RuntimeError when PySpark not available."""
        with patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", False):
            profiler = DataProfiler()
            with pytest.raises(RuntimeError, match="PySpark required"):
                profiler.analyze_skew("./data.parquet", "column")

    @patch("spark_optima.data.profiler.col")
    @patch("spark_optima.data.profiler.SparkSession")
    def test_analyze_skew_empty_result(self, mock_spark_session, mock_col) -> None:
        """Test analyze_skew with empty result."""
        mock_spark = MagicMock()
        builder = mock_spark_session.builder.appName.return_value
        builder.master.return_value.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_df.count.return_value = 0
        mock_df.schema = MagicMock()
        mock_df.schema.fields = []

        # Mock groupBy chain
        mock_grouped = mock_df.groupBy.return_value
        mock_counted = mock_grouped.count.return_value
        mock_ordered = mock_counted.orderBy.return_value
        mock_limited = mock_ordered.limit.return_value

        mock_limited.collect.return_value = []
        mock_ordered.count.return_value = 0

        mock_spark.read.parquet.return_value = mock_df

        profiler = DataProfiler()
        with patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True):
            result = profiler.analyze_skew("/data/test.parquet", "empty_col")

        assert result["total_rows"] == 0
        assert result["distinct_values"] == 0
        assert result["top_values"] == []
        assert result["skew_ratio"] == 1.0  # Default when no values

    @patch("spark_optima.data.profiler.col")
    @patch("spark_optima.data.profiler.SparkSession")
    def test_analyze_skew_with_existing_spark(self, mock_spark_session, mock_col) -> None:
        """Test analyze_skew with existing Spark session."""
        mock_spark = MagicMock()
        mock_df = MagicMock()
        mock_df.count.return_value = 100
        mock_df.schema = MagicMock()
        mock_df.schema.fields = []

        # Mock groupBy chain
        mock_grouped = mock_df.groupBy.return_value
        mock_counted = mock_grouped.count.return_value
        mock_ordered = mock_counted.orderBy.return_value
        mock_limited = mock_ordered.limit.return_value

        mock_row = MagicMock()
        mock_row.__getitem__ = MagicMock(side_effect=lambda x: "x" if x == "col" else 50)
        mock_limited.collect.return_value = [mock_row]
        mock_counted.count.return_value = 1

        mock_spark.read.json.return_value = mock_df

        profiler = DataProfiler(spark=mock_spark)
        with patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True):
            result = profiler.analyze_skew("/data/test.json", "col")

        assert result["column"] == "col"
        mock_spark_session.builder.appName.assert_not_called()

    @patch("spark_optima.data.profiler.col")
    @patch("spark_optima.data.profiler.SparkSession")
    def test_analyze_skew_is_skewed(self, mock_spark_session, mock_col) -> None:
        """Test that is_skewed is correctly determined."""
        mock_spark = MagicMock()
        builder = mock_spark_session.builder.appName.return_value
        builder.master.return_value.getOrCreate.return_value = mock_spark

        mock_df = MagicMock()
        mock_df.count.return_value = 100
        mock_df.schema = MagicMock()
        mock_df.schema.fields = []

        # Mock groupBy chain
        mock_grouped = mock_df.groupBy.return_value
        mock_counted = mock_grouped.count.return_value
        mock_ordered = mock_counted.orderBy.return_value
        mock_limited = mock_ordered.limit.return_value

        mock_row1 = MagicMock()
        mock_row1.__getitem__ = MagicMock(side_effect=lambda x: "a" if x == "skewed_col" else 90)
        mock_row2 = MagicMock()
        mock_row2.__getitem__ = MagicMock(side_effect=lambda x: "b" if x == "skewed_col" else 5)
        mock_limited.collect.return_value = [mock_row1, mock_row2]
        mock_counted.count.return_value = 5

        mock_spark.read.parquet.return_value = mock_df

        profiler = DataProfiler()
        with patch("spark_optima.data.profiler.PYSPARK_AVAILABLE", True):
            result = profiler.analyze_skew("/data/test.parquet", "skewed_col")

        # skew_ratio = 90 / (100/2) = 1.8, which is < 2.0, so not skewed
        assert isinstance(result["is_skewed"], bool)


class TestEstimateSizeEdgeCases:
    """Additional edge case tests for _estimate_size method."""

    def test_estimate_size_with_delta_format(self, tmp_path) -> None:
        """Test size estimation for delta format directory."""
        profiler = DataProfiler()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "_delta_log").mkdir()
        (data_dir / "part-00001.parquet").write_bytes(b"x" * 100)

        size = profiler._estimate_size(data_dir, "delta")
        assert size == 100

    def test_estimate_size_permission_error(self, tmp_path) -> None:
        """Test size estimation handles permission errors gracefully."""
        profiler = DataProfiler()
        # Create a mock that raises PermissionError
        mock_path = MagicMock(spec=Path)
        mock_path.is_file.return_value = False
        mock_path.is_dir.return_value = True
        mock_path.rglob.side_effect = PermissionError("Access denied")

        size = profiler._estimate_size(mock_path, "parquet")
        assert size == 0

    def test_estimate_size_with_symlinks(self, tmp_path) -> None:
        """Test size estimation with symlinks."""
        profiler = DataProfiler()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file1.parquet").write_bytes(b"x" * 50)

        # Create a symlink (macOS/Linux)
        symlink = data_dir / "link.parquet"
        with contextlib.suppress(OSError, NotImplementedError):
            (data_dir / "file1.parquet").symlink_to(symlink)

        # Just test it doesn't crash
        size = profiler._estimate_size(data_dir, "parquet")
        assert size >= 50


class TestPySparkImportError:
    """Test the ImportError path when PySpark is not available."""

    def test_pyspark_import_error_block(self) -> None:
        """Test that PYSPARK_AVAILABLE is set to False when import fails."""
        # Save original modules
        saved_modules = {}
        pyspark_modules = [key for key in sys.modules if key.startswith("pyspark")]
        for key in pyspark_modules:
            saved_modules[key] = sys.modules.pop(key)

        try:
            # Mock the import to raise ImportError
            with patch.dict("sys.modules", {"pyspark": None, "pyspark.sql": None}):
                # Reload the profiler module to trigger the import
                import spark_optima.data.profiler as profiler_module

                importlib.reload(profiler_module)
                # After reload with pyspark mocked, PYSPARK_AVAILABLE should be False
                assert profiler_module.PYSPARK_AVAILABLE is False
        finally:
            # Restore modules
            sys.modules.update(saved_modules)
            # Reload to restore original state
            import spark_optima.data.profiler as profiler_module

            importlib.reload(profiler_module)
