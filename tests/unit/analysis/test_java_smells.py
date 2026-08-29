# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for smell detection and recommendations on Java sources."""

from __future__ import annotations

import pytest

from spark_optima.analysis.smell_detector import SmellDetector, detect_smells

pytestmark = pytest.mark.unit


def _smell_types(result) -> set[str]:  # type: ignore[no-untyped-def]
    return {smell.smell_type for smell in result.smells}


class TestJavaOperationBasedSmells:
    def test_cartesian_join_detected(self) -> None:
        code = "Dataset<Row> out = df1.crossJoin(df2);"
        result = detect_smells(code, language="java")
        assert "cartesian_join" in _smell_types(result)

    def test_missing_broadcast_hint_detected(self) -> None:
        code = 'Dataset<Row> out = df1.join(df2, "id");'
        result = detect_smells(code, language="java")
        assert "missing_broadcast_hint" in _smell_types(result)

    def test_broadcast_hint_suppresses_smell(self) -> None:
        code = 'Dataset<Row> out = df1.join(broadcast(df2), "id");'
        result = detect_smells(code, language="java")
        assert "missing_broadcast_hint" not in _smell_types(result)

    def test_select_star_detected(self) -> None:
        code = 'Dataset<Row> out = df.select("*");'
        result = detect_smells(code, language="java")
        assert "select_star" in _smell_types(result)

    def test_metadata_reports_language(self) -> None:
        result = detect_smells("Dataset<Row> df = spark.read();", language="java")
        assert result.metadata["language"] == "java"


class TestJavaSqlStringSmells:
    def test_triple_quoted_sql_smells(self) -> None:
        code = 'Dataset<Row> top = spark.sql("""\n  SELECT * FROM events ORDER BY ts\n""");'
        result = detect_smells(code, language="java")
        types = _smell_types(result)
        assert "select_star" in types
        assert "sql_orderby_without_limit" in types


class TestJavaEquivalentDetectors:
    def test_collect_without_limit_flagged(self) -> None:
        code = "Dataset<Row> rows = df.collect();"
        result = detect_smells(code, language="java")
        assert "large_collect" in _smell_types(result)

    def test_single_partition_write_fluent_chain(self) -> None:
        code = 'df.coalesce(1).write().mode("overwrite").parquet("out/");'
        result = detect_smells(code, language="java")
        assert "single_partition_write" in _smell_types(result)

    def test_no_stale_python_ast_leaks_into_java_analysis(self) -> None:
        detector = SmellDetector()
        java_result = detector.analyze_source("Dataset<Row> df = spark.read();", language="java")
        java_result = detector.analyze_source("Dataset<Row> df = spark.read();", language="java")
        # Python-only detectors should not run on java source
        assert "count_for_empty_check" not in _smell_types(java_result)
        assert "infer_schema" not in _smell_types(java_result)
