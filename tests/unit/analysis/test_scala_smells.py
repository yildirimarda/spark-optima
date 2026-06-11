# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for smell detection and recommendations on Scala sources."""

from __future__ import annotations

import pytest

from spark_optima.analysis.models import SeverityLevel
from spark_optima.analysis.recommender import RecommendationEngine, analyze_code
from spark_optima.analysis.smell_detector import SmellDetector, detect_smells

pytestmark = pytest.mark.unit


def _smell_types(result) -> set[str]:  # type: ignore[no-untyped-def]
    """Return the set of detected smell types."""
    return {smell.smell_type for smell in result.smells}


class TestScalaOperationBasedSmells:
    """Existing operation-based detectors on Scala parse results."""

    def test_cartesian_join_detected(self) -> None:
        """crossJoin() in Scala is flagged as a cartesian join."""
        code = "val out = df1.crossJoin(df2)\nout.show()\n"
        result = detect_smells(code, language="scala")

        assert "cartesian_join" in _smell_types(result)
        smell = result.get_smells_by_type("cartesian_join")[0]
        assert smell.severity == SeverityLevel.HIGH

    def test_missing_broadcast_hint_detected(self) -> None:
        """Joins without broadcast hints are flagged."""
        code = 'val out = df1.join(df2, Seq("id"))'
        result = detect_smells(code, language="scala")
        assert "missing_broadcast_hint" in _smell_types(result)

    def test_broadcast_hint_suppresses_smell(self) -> None:
        """Joins with broadcast() are not flagged."""
        code = 'val out = df1.join(broadcast(df2), Seq("id"))'
        result = detect_smells(code, language="scala")
        assert "missing_broadcast_hint" not in _smell_types(result)

    def test_caching_issue_detected(self) -> None:
        """cache() without enough reuse is flagged."""
        code = "df.cache()\ndf.show()\n"
        result = detect_smells(code, language="scala")
        assert "caching_issue" in _smell_types(result)

    def test_select_star_detected(self) -> None:
        """select("*") in Scala is flagged."""
        code = 'val out = df.select("*")\nout.show()\n'
        result = detect_smells(code, language="scala")
        assert "select_star" in _smell_types(result)

    def test_metadata_reports_language(self) -> None:
        """Analysis metadata carries the source language."""
        result = detect_smells("df.show()", language="scala")
        assert result.metadata["language"] == "scala"


class TestScalaSqlStringSmells:
    """spark.sql("...") strings flow into the SQL analyzer."""

    def test_triple_quoted_sql_smells(self) -> None:
        """Triple-quoted SQL is analyzed for SELECT * and unbounded ORDER BY."""
        code = 'val top = spark.sql("""\n  SELECT * FROM events ORDER BY ts\n""")\ntop.show()\n'
        result = detect_smells(code, language="scala")

        types = _smell_types(result)
        assert "select_star" in types
        assert "sql_orderby_without_limit" in types

    def test_interpolated_sql_skipped(self) -> None:
        """Interpolated SQL strings are skipped without errors."""
        code = 'val out = spark.sql(s"SELECT * FROM ${table}")'
        result = detect_smells(code, language="scala")
        assert "select_star" not in _smell_types(result)


class TestScalaEquivalentDetectors:
    """Scala equivalents of Python-AST-based detectors."""

    def test_collect_without_limit_flagged(self) -> None:
        """collect() with no preceding limit() is flagged."""
        code = "val rows = df.collect()"
        result = detect_smells(code, language="scala")
        assert "large_collect" in _smell_types(result)

    def test_collect_after_fluent_limit_not_flagged(self) -> None:
        """df.limit(N).collect() is not flagged."""
        code = "val rows = df.limit(100).collect()"
        result = detect_smells(code, language="scala")
        assert "large_collect" not in _smell_types(result)

    def test_collect_after_assigned_limit_not_flagged(self) -> None:
        """val limited = df.limit(N); limited.collect() is not flagged."""
        code = "val limited = df.limit(100)\nval rows = limited.collect()\n"
        result = detect_smells(code, language="scala")
        assert "large_collect" not in _smell_types(result)

    def test_single_partition_write_fluent_chain(self) -> None:
        """coalesce(1).write... is flagged."""
        code = 'df.coalesce(1).write.parquet("out/")'
        result = detect_smells(code, language="scala")
        assert "single_partition_write" in _smell_types(result)

    def test_single_partition_write_two_statements(self) -> None:
        """val single = df.repartition(1); single.write... is flagged."""
        code = 'val single = df.repartition(1)\nsingle.write.parquet("out/")\n'
        result = detect_smells(code, language="scala")
        assert "single_partition_write" in _smell_types(result)

    def test_multi_partition_write_not_flagged(self) -> None:
        """coalesce(8) before a write is not a single-partition write."""
        code = 'df.coalesce(8).write.parquet("out/")'
        result = detect_smells(code, language="scala")
        assert "single_partition_write" not in _smell_types(result)

    def test_write_of_original_df_not_flagged(self) -> None:
        """Writing the original DataFrame is fine even if a repartition(1) val exists.

        Regression test: only the assigned variable (or the same fluent
        chain) carries the single-partition bottleneck, not the source df.
        """
        code = 'val single = df.repartition(1)\ndf.write.parquet("out/")\n'
        result = detect_smells(code, language="scala")
        assert "single_partition_write" not in _smell_types(result)

    def test_multiline_chain_with_writer_options_flagged(self) -> None:
        """coalesce(1) chains spanning lines and .mode(...) glue are flagged."""
        code = 'df.coalesce(1)\n  .write.mode("overwrite").parquet("out/")\n'
        result = detect_smells(code, language="scala")
        assert "single_partition_write" in _smell_types(result)

    def test_collect_with_limit_and_unrelated_ops_between(self) -> None:
        """Unrelated operations between limit() and collect() do not matter.

        Regression test: the limit window must be computed within the same
        DataFrame lineage bucket, not over global chain positions.
        """
        code = (
            'val top = df.orderBy(desc("amount")).limit(10)\n'
            'a.select("c1").filter("c2 > 1").show()\n'
            'b.select("c1").filter("c2 > 1").show()\n'
            'c.select("c1").show()\n'
            "val rows = top.collect()\n"
        )
        result = detect_smells(code, language="scala")
        assert "large_collect" not in _smell_types(result)

    def test_collect_without_limit_still_flagged_with_unrelated_ops(self) -> None:
        """Unrelated lineages never suppress a genuine unbounded collect()."""
        code = 'other.limit(10).show()\na.select("c1").show()\nval rows = df.collect()\n'
        result = detect_smells(code, language="scala")
        assert "large_collect" in _smell_types(result)

    def test_ast_based_detectors_skip_gracefully(self) -> None:
        """Python-AST detectors produce no smells and no crashes on Scala."""
        code = """
        val df = spark.read.option("inferSchema", "true").csv("data.csv")
        if (df.count() == 0) {
          println("empty")
        }
        for (c <- columns) {
          out = out.withColumn(c, trim(col(c)))
        }
        """
        result = detect_smells(code, language="scala")

        types = _smell_types(result)
        # These detectors walk the Python AST and must skip for Scala input
        assert "count_for_empty_check" not in types
        assert "infer_schema" not in types
        assert "withcolumn_in_loop" not in types

    def test_no_stale_python_ast_leaks_into_scala_analysis(self) -> None:
        """A prior Python analysis must not leak AST smells into Scala results."""
        detector = SmellDetector()
        python_code = "if df.count() == 0:\n    handle_empty()\n"
        python_result = detector.analyze_source(python_code, language="python")
        assert "count_for_empty_check" in _smell_types(python_result)

        scala_result = detector.analyze_source("df.show()", language="scala")
        assert "count_for_empty_check" not in _smell_types(scala_result)


class TestScalaSmallFileProblem:
    """small_file_problem on Scala write chains.

    Regression tests: the bare ``write`` accessor carries no arguments in
    Scala, so partitioning must be detected from the surrounding chain.
    """

    def test_partitioned_write_not_flagged(self) -> None:
        """df.write.partitionBy(...).parquet(...) is a partitioned write."""
        code = 'df.write.partitionBy("dt").parquet("out/")'
        result = detect_smells(code, language="scala")
        assert "small_file_problem" not in _smell_types(result)

    def test_unpartitioned_write_flagged(self) -> None:
        """df.write.parquet(...) without partitioning is still flagged."""
        code = 'df.write.parquet("out/")'
        result = detect_smells(code, language="scala")
        assert "small_file_problem" in _smell_types(result)

    def test_partitioned_write_with_writer_options_not_flagged(self) -> None:
        """Unrecorded chained calls like .mode(...) do not break the chain."""
        code = 'df.write.mode("overwrite").partitionBy("dt").parquet("out/")'
        result = detect_smells(code, language="scala")
        assert "small_file_problem" not in _smell_types(result)

    def test_assigned_partitioned_writer_not_flagged(self) -> None:
        """A partitioned writer assigned to a val suppresses the smell on save."""
        code = 'val writer = df.write.partitionBy("dt")\nwriter.save("out/")\n'
        result = detect_smells(code, language="scala")
        assert "small_file_problem" not in _smell_types(result)

    def test_other_partitioned_write_does_not_suppress(self) -> None:
        """A partitionBy in another statement on the same df does not leak."""
        code = 'df.write.partitionBy("dt").parquet("a/")\ndf.write.parquet("b/")\n'
        result = detect_smells(code, language="scala")
        assert "small_file_problem" in _smell_types(result)


class TestGroupByKeySmell:
    """New groupbykey_usage smell on both language paths."""

    def test_scala_groupbykey_flagged_high(self) -> None:
        """Scala rdd.groupByKey() is flagged with HIGH severity."""
        code = "val grouped = rdd.groupByKey().mapValues(_.sum)"
        result = detect_smells(code, language="scala")

        smells = result.get_smells_by_type("groupbykey_usage")
        assert len(smells) == 1
        assert smells[0].severity == SeverityLevel.HIGH

    def test_python_groupbykey_flagged_high(self) -> None:
        """Python rdd.groupByKey() is flagged with HIGH severity."""
        code = "result = rdd.groupByKey().mapValues(sum)\n"
        result = detect_smells(code, language="python")

        smells = result.get_smells_by_type("groupbykey_usage")
        assert len(smells) == 1
        assert smells[0].severity == SeverityLevel.HIGH

    def test_dataframe_groupby_not_flagged(self) -> None:
        """DataFrame groupBy() is not mistaken for RDD groupByKey()."""
        python_result = detect_smells('df.groupBy("k").count()\n', language="python")
        scala_result = detect_smells('df.groupBy("k").count()', language="scala")

        assert "groupbykey_usage" not in _smell_types(python_result)
        assert "groupbykey_usage" not in _smell_types(scala_result)

    def test_groupbykey_recommendation_has_snippets(self) -> None:
        """The recommender produces before/after snippets for groupByKey."""
        result = analyze_code("val grouped = rdd.groupByKey()", language="scala")

        recs = [r for r in result.recommendations if r.smell.smell_type == "groupbykey_usage"]
        assert len(recs) == 1
        assert "groupByKey" in recs[0].before_code
        assert "reduceByKey" in recs[0].after_code
        assert recs[0].effort == "medium"


class TestLanguageRouting:
    """Entry-point language parameter and auto-detection."""

    def test_analyze_code_auto_detects_scala(self) -> None:
        """analyze_code(language='auto') routes clear Scala source correctly."""
        code = (
            "import org.apache.spark.sql.SparkSession\n"
            "object Job {\n  def main(args: Array[String]): Unit = {\n    df1.crossJoin(df2).collect()\n  }\n}\n"
        )
        result = analyze_code(code)

        assert result.metadata["language"] == "scala"
        assert "cartesian_join" in _smell_types(result)

    def test_analyze_code_defaults_to_python(self) -> None:
        """Ambiguous source still goes through the Python path."""
        result = analyze_code("df1.crossJoin(df2).collect()\n")
        assert result.metadata["language"] == "python"
        assert "cartesian_join" in _smell_types(result)

    def test_engine_analyze_file_routes_by_suffix(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """RecommendationEngine.analyze_file auto-routes .scala files."""
        scala_file = tmp_path / "Job.scala"
        scala_file.write_text("val out = df1.crossJoin(df2)\n", encoding="utf-8")

        engine = RecommendationEngine()
        result = engine.analyze_file(str(scala_file))

        assert result.metadata["language"] == "scala"
        assert "cartesian_join" in _smell_types(result)

    def test_engine_analyze_file_python_unchanged(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Python files keep going through the Python parser."""
        py_file = tmp_path / "job.py"
        py_file.write_text("result = df1.crossJoin(df2)\n", encoding="utf-8")

        engine = RecommendationEngine()
        result = engine.analyze_file(str(py_file))

        assert result.metadata["language"] == "python"
        assert "cartesian_join" in _smell_types(result)

    def test_explicit_language_overrides_detection(self) -> None:
        """language='python' forces the Python parser (raising on Scala syntax)."""
        with pytest.raises(SyntaxError):
            detect_smells("object Job { val x = df.show() }", language="python")

    def test_invalid_language_rejected(self) -> None:
        """Unknown language values raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported language"):
            SmellDetector().analyze_source("df.show()", language="java")

    def test_scala_path_never_flags_python_only_detectors(self) -> None:
        """A realistic Scala job analyzes end-to-end without crashing."""
        code = """
        import org.apache.spark.sql.SparkSession
        import org.apache.spark.sql.functions.broadcast

        object EtlJob {
          def main(args: Array[String]): Unit = {
            val spark = SparkSession.builder().appName("etl").getOrCreate()
            val events = spark.read.parquet("s3://bucket/events")
            val users = spark.read.parquet("s3://bucket/users")
            val joined = events.join(users, Seq("user_id"))
            val agg = joined.groupBy("user_id").agg(sum("amount"))
            agg.orderBy(desc("amount")).show()
            agg.coalesce(1).write.parquet("s3://bucket/out")
            val legacy = rdd.groupByKey().mapValues(_.size)
            spark.sql(\"\"\"SELECT * FROM staging ORDER BY ts\"\"\").collect()
          }
        }
        """
        result = analyze_code(code, language="scala")

        types = _smell_types(result)
        assert "groupbykey_usage" in types
        assert "single_partition_write" in types
        assert "select_star" in types
        assert len(result.recommendations) >= len(result.smells) - 1
        assert result.metadata["language"] == "scala"
