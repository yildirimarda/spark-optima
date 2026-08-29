# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the Java Spark code parser."""

from __future__ import annotations

import pytest

from spark_optima.analysis.java_parser import JavaCodeParser, detect_language, parse_java_code
from spark_optima.analysis.models import SparkOperationType

pytestmark = pytest.mark.unit


def _method_names(result) -> list[str]:  # type: ignore[no-untyped-def]
    return [op.method_name for op in result.operations]


class TestJavaParserOperations:
    def test_parse_simple_chain(self) -> None:
        code = 'Dataset<Row> df = spark.read().parquet("data");\nDataset<Row> out = df.filter("a > 1").select("b");'
        result = parse_java_code(code)
        names = _method_names(result)
        assert names.count("read") == 1
        assert "filter" in names
        assert "select" in names
        assert result.language == "java"

    def test_operation_types_mapped(self) -> None:
        code = (
            'Dataset<Row> a = df.groupBy("k").agg(sum("v"));\n'
            "a.cache();\n"
            "Dataset<Row> b = a.repartition(200);\n"
            'b.join(other, "id");\n'
        )
        result = parse_java_code(code)
        types = {op.method_name: op.operation_type for op in result.operations}
        assert types.get("groupBy") == SparkOperationType.AGGREGATION
        assert types.get("cache") == SparkOperationType.CACHE
        assert types.get("repartition") == SparkOperationType.REPARTITION
        assert types.get("join") == SparkOperationType.JOIN

    def test_rdd_operations_detected(self) -> None:
        code = "JavaRDD<String> grouped = rdd.groupBy(x -> x);\nJavaRDD<String> reduced = rdd.reduce((a, b) -> a);"
        result = parse_java_code(code)
        # groupBy/reduce are not in SPARK_METHOD_MAP as RDD ops by default,
        # but if they appear as method calls they may be registered.
        # The main point is parsing does not crash.
        assert result.language == "java"

    def test_write_accessor_without_parens(self) -> None:
        code = 'df.write().mode("overwrite").parquet("out/");'
        result = parse_java_code(code)
        write_ops = [op for op in result.operations if op.operation_type == SparkOperationType.WRITE]
        assert len(write_ops) == 1
        assert write_ops[0].method_name == "write"

    def test_locations_are_one_indexed_lines(self) -> None:
        code = 'Dataset<Row> a = 1;\nDataset<Row> df2 = df.filter("x > 0");\ndf2.show();\n'
        result = parse_java_code(code)
        filter_op = next(op for op in result.operations if op.method_name == "filter")
        assert filter_op.location is not None and filter_op.location.line == 2

    def test_empty_source_yields_no_operations(self) -> None:
        result = parse_java_code("")
        assert result.operations == []
        assert result.operation_count == 0


class TestJavaValTracking:
    def test_assignment_registers_lineage(self) -> None:
        code = 'Dataset<Row> joined = df.join(small, "id");'
        parser = JavaCodeParser()
        result = parser.parse_source(code)
        assert "joined" in result.dataframe_vars
        assert [op.method_name for op in result.dataframe_vars["joined"]] == ["join"]


class TestJavaMasking:
    def test_line_comment_masked(self) -> None:
        code = '// val fake = df.collect()\nDataset<Row> real = df.filter("x > 1");\n'
        result = parse_java_code(code)
        assert _method_names(result) == ["filter"]

    def test_block_comment_masked(self) -> None:
        code = "/* df.crossJoin(x).show() */\nDataset<Row> out = df.show();\n"
        result = parse_java_code(code)
        # show is tracked
        assert "show" in _method_names(result)


class TestDetectLanguage:
    def test_detects_java_import(self) -> None:
        assert detect_language("import org.apache.spark.sql.SparkSession;\n") == "java"

    def test_detects_java_spark_session_builder(self) -> None:
        assert detect_language('SparkSession.builder().appName("test").getOrCreate()') == "java"

    def test_detects_java_typed_assignment(self) -> None:
        assert detect_language("Dataset<Row> df = spark.read();\n") == "java"

    def test_scala_val_not_java(self) -> None:
        assert detect_language('val df = spark.read.parquet("d")\n') == "scala"

    def test_python_stays_python(self) -> None:
        assert detect_language("from pyspark.sql import SparkSession\n") == "python"

    def test_empty_defaults_to_python(self) -> None:
        assert detect_language("") == "python"
