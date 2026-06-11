# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the Scala Spark code parser."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest

from spark_optima.analysis.models import SparkOperationType
from spark_optima.analysis.parser import parse_spark_code
from spark_optima.analysis.scala_parser import (
    ScalaCodeParser,
    detect_language,
    parse_scala_code,
)

if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.unit


def _method_names(result) -> list[str]:  # type: ignore[no-untyped-def]
    """Return detected method names in chain order."""
    return [op.method_name for op in result.operations]


class TestScalaParserOperations:
    """Operation detection in Scala sources."""

    def test_parse_simple_chain(self) -> None:
        """Detects read, join, and select in a simple program."""
        code = """
        val df = spark.read.parquet("data.parquet")
        val small = spark.read.parquet("small.parquet")
        val out = df.join(small, Seq("id")).select("a", "b")
        """
        result = parse_scala_code(code)

        names = _method_names(result)
        assert names.count("read") == 2
        assert "join" in names
        assert "select" in names
        assert result.language == "scala"
        assert result.operation_count == len(result.operations)

    def test_operation_types_mapped(self) -> None:
        """Method names map to the same operation types as the Python parser."""
        code = """
        val a = df.groupBy("k").agg(sum("v"))
        a.cache()
        val b = a.repartition(200)
        b.join(other, Seq("id"))
        b.collect()
        """
        result = parse_scala_code(code)
        types = {op.method_name: op.operation_type for op in result.operations}

        assert types["groupBy"] == SparkOperationType.AGGREGATION
        assert types["cache"] == SparkOperationType.CACHE
        assert types["repartition"] == SparkOperationType.REPARTITION
        assert types["join"] == SparkOperationType.JOIN
        assert types["collect"] == SparkOperationType.ACTION

    def test_rdd_operations_detected(self) -> None:
        """RDD operations groupByKey/reduceByKey/mapPartitions are detected."""
        code = """
        val grouped = rdd.groupByKey()
        val reduced = rdd.reduceByKey(_ + _)
        val mapped = rdd.mapPartitions(iter => iter.map(transform))
        """
        result = parse_scala_code(code)
        types = {op.method_name: op.operation_type for op in result.operations}

        assert types["groupByKey"] == SparkOperationType.AGGREGATION
        assert types["reduceByKey"] == SparkOperationType.AGGREGATION
        assert types["mapPartitions"] == SparkOperationType.TRANSFORMATION

    def test_chain_positions_follow_source_order(self) -> None:
        """Chain positions increase monotonically in source order."""
        code = """
        val x = df.filter(cond).limit(10)
        x.collect()
        """
        result = parse_scala_code(code)

        names = _method_names(result)
        assert names == ["filter", "limit", "collect"]
        positions = [op.chain_position for op in result.operations]
        assert positions == sorted(positions)

    def test_chained_calls_resolve_root_variable(self) -> None:
        """Fluent chains resolve the root DataFrame variable."""
        code = 'df.join(small, Seq("id")).select("a").filter(cond)'
        result = parse_scala_code(code)

        assert {op.dataframe_var for op in result.operations} == {"df"}

    def test_multiline_chain_resolves_root_variable(self) -> None:
        """Chains split across lines still resolve the root variable."""
        code = """
        val out = df
          .join(small, Seq("id"))
          .select("a")
        """
        result = parse_scala_code(code)

        join_op = next(op for op in result.operations if op.method_name == "join")
        select_op = next(op for op in result.operations if op.method_name == "select")
        assert join_op.dataframe_var == "df"
        assert select_op.dataframe_var == "df"

    def test_write_accessor_without_parens(self) -> None:
        """df.write.parquet(...) records a WRITE operation."""
        code = 'df.write.mode("overwrite").parquet("out/")'
        result = parse_scala_code(code)

        write_ops = [op for op in result.operations if op.operation_type == SparkOperationType.WRITE]
        assert len(write_ops) == 1
        assert write_ops[0].method_name == "write"
        assert write_ops[0].dataframe_var == "df"

    def test_standalone_broadcast_call(self) -> None:
        """A bare broadcast(...) call is tracked like the Python parser does."""
        code = 'val out = df.join(broadcast(small), Seq("id"))'
        result = parse_scala_code(code)

        broadcast_ops = [op for op in result.operations if op.method_name == "broadcast"]
        assert len(broadcast_ops) == 1
        assert broadcast_ops[0].dataframe_var == "broadcast_call"

    def test_locations_are_one_indexed_lines(self) -> None:
        """Operation locations carry correct 1-indexed line numbers."""
        code = "val a = 1\nval df2 = df.filter(cond)\ndf2.collect()\n"
        result = parse_scala_code(code)

        filter_op = next(op for op in result.operations if op.method_name == "filter")
        collect_op = next(op for op in result.operations if op.method_name == "collect")
        assert filter_op.location is not None and filter_op.location.line == 2
        assert collect_op.location is not None and collect_op.location.line == 3

    def test_empty_source_yields_no_operations(self) -> None:
        """Empty source parses to an empty result."""
        result = parse_scala_code("")
        assert result.operations == []
        assert result.operation_count == 0


class TestScalaValTracking:
    """val/var assignment tracking for DataFrame lineage."""

    def test_val_assignment_registers_lineage(self) -> None:
        """val x = df.join(...) registers the join under x."""
        code = 'val joined = df.join(small, Seq("id"))'
        parser = ScalaCodeParser()
        result = parser.parse_source(code)

        assert "joined" in result.dataframe_vars
        assert [op.method_name for op in result.dataframe_vars["joined"]] == ["join"]
        # The operation itself still records the chain root
        assert result.dataframe_vars["joined"][0].dataframe_var == "df"

    def test_multiline_assignment_lineage(self) -> None:
        """Multi-line chains are attributed to the assigned variable."""
        code = """
        val result = df
          .join(small, Seq("id"))
          .select("a")
        result.show()
        """
        result = parse_scala_code(code)

        names = [op.method_name for op in result.dataframe_vars["result"]]
        assert "join" in names
        assert "select" in names
        assert "show" in names

    def test_var_and_lazy_val_assignments(self) -> None:
        """var and lazy val assignments are tracked too."""
        code = """
        var mutable = df.filter(cond)
        lazy val deferred = df.limit(5)
        """
        result = parse_scala_code(code)

        assert [op.method_name for op in result.dataframe_vars["mutable"]] == ["filter"]
        assert [op.method_name for op in result.dataframe_vars["deferred"]] == ["limit"]

    def test_get_dataframe_lineage(self) -> None:
        """get_dataframe_lineage mirrors the Python parser helper."""
        parser = ScalaCodeParser()
        parser.parse_source("val out = df.filter(cond)\nout.collect()")

        lineage = parser.get_dataframe_lineage("out")
        assert [op.method_name for op in lineage] == ["filter", "collect"]
        assert parser.get_dataframe_lineage("unknown") == []

    def test_assignment_without_spark_ops_is_ignored(self) -> None:
        """Plain value assignments do not pollute the lineage map."""
        result = parse_scala_code('val n = 42\nval s = "text"')
        assert "n" not in result.dataframe_vars
        assert "s" not in result.dataframe_vars


class TestScalaMasking:
    """Comment and string masking before scanning."""

    def test_line_comment_masked(self) -> None:
        """Method calls inside // comments never match."""
        code = "// val fake = df.collect()\nval real = df.filter(cond)\n"
        result = parse_scala_code(code)
        assert _method_names(result) == ["filter"]

    def test_block_comment_masked(self) -> None:
        """Method calls inside /* */ comments never match."""
        code = "/* df.crossJoin(other).collect() */\ndf.show()\n"
        result = parse_scala_code(code)
        assert _method_names(result) == ["show"]

    def test_nested_block_comment_masked(self) -> None:
        """Scala block comments nest; inner terminators do not end masking."""
        code = "/* outer /* df.collect() */ still comment df.crossJoin(x) */\ndf.count()\n"
        result = parse_scala_code(code)
        assert _method_names(result) == ["count"]

    def test_method_name_inside_string_masked(self) -> None:
        """Method names inside string literals never match."""
        code = 'val msg = "please call df.collect() and df.join(x) soon"\ndf.show()\n'
        result = parse_scala_code(code)
        assert _method_names(result) == ["show"]

    def test_interpolated_string_masked(self) -> None:
        """Interpolated strings (including ${...} expressions) are masked."""
        code = 'val msg = s"rows: ${df.collect().length} via .join("\ndf.show()\n'
        result = parse_scala_code(code)
        assert _method_names(result) == ["show"]

    def test_triple_quoted_string_masked(self) -> None:
        """Triple-quoted string contents are masked."""
        code = 'val doc = """contains df.crossJoin(other) and ).collect()"""\ndf.show()\n'
        result = parse_scala_code(code)
        assert _method_names(result) == ["show"]

    def test_escaped_quote_inside_string(self) -> None:
        """Escaped quotes do not terminate string masking early."""
        code = 'val msg = "he said \\".collect()\\" loudly"\ndf.show()\n'
        result = parse_scala_code(code)
        assert _method_names(result) == ["show"]

    def test_char_literal_with_double_quote(self) -> None:
        """A '"' char literal does not start a bogus string region."""
        code = "val quote = '\"'\ndf.filter(cond).collect()\n"
        result = parse_scala_code(code)
        assert _method_names(result) == ["filter", "collect"]

    def test_comment_inside_string_not_treated_as_comment(self) -> None:
        """A // sequence inside a string does not mask the rest of the line."""
        code = 'val url = "http://example.com"; df.show()\n'
        result = parse_scala_code(code)
        assert _method_names(result) == ["show"]

    def test_string_literal_inside_interpolation_block_masked(self) -> None:
        """A nested string inside ${...} does not terminate the outer string."""
        code = 's"result: ${run("df.collect()")}"'
        result = parse_scala_code(code)
        assert result.operations == []

    def test_interpolation_block_with_nested_braces_and_strings(self) -> None:
        """Nested braces and quotes inside ${...} are consumed as one block."""
        code = 'val msg = s"got ${fmt({ "df.crossJoin(x)" })} rows"\ndf.show()\n'
        result = parse_scala_code(code)
        assert _method_names(result) == ["show"]

    def test_triple_quoted_interpolation_with_nested_string_masked(self) -> None:
        """${...} blocks inside triple-quoted interpolated strings are masked."""
        code = 'val q = s"""${run("df.collect()")}"""\ndf.show()\n'
        result = parse_scala_code(code)
        assert _method_names(result) == ["show"]

    def test_code_after_interpolated_string_still_scanned(self) -> None:
        """Masking the ${...} block does not swallow trailing real code."""
        code = 'val msg = s"rows: ${run("x.collect()")}"\ndf.filter(cond).collect()\n'
        result = parse_scala_code(code)
        assert _method_names(result) == ["filter", "collect"]

    def test_escaped_dollar_in_interpolated_string(self) -> None:
        """$$ escapes do not start an interpolation block."""
        code = 'val msg = s"costs $$5 ${tag}"\ndf.show()\n'
        result = parse_scala_code(code)
        assert _method_names(result) == ["show"]


class TestScalaSqlExtraction:
    """spark.sql() literal recovery for SQL analysis."""

    def test_plain_sql_literal_recovered(self) -> None:
        """A plain SQL string argument is recoverable via ast.literal_eval."""
        code = 'val out = spark.sql("SELECT id FROM events WHERE id > 1")'
        result = parse_scala_code(code)

        sql_op = next(op for op in result.operations if op.method_name == "sql")
        assert ast.literal_eval(sql_op.arguments[0]) == "SELECT id FROM events WHERE id > 1"

    def test_triple_quoted_sql_recovered(self) -> None:
        """Triple-quoted SQL contents are recovered with newlines intact."""
        code = 'val out = spark.sql("""\n  SELECT *\n  FROM events\n""")'
        result = parse_scala_code(code)

        sql_op = next(op for op in result.operations if op.method_name == "sql")
        recovered = ast.literal_eval(sql_op.arguments[0])
        assert "SELECT *" in recovered
        assert "FROM events" in recovered

    def test_interpolated_sql_not_treated_as_literal(self) -> None:
        """s-interpolated SQL keeps its raw text and is not a plain literal."""
        code = 'val out = spark.sql(s"SELECT * FROM ${table}")'
        result = parse_scala_code(code)

        sql_op = next(op for op in result.operations if op.method_name == "sql")
        assert sql_op.arguments[0].startswith('s"')

    def test_escaped_sql_literal_unescaped(self) -> None:
        """Escape sequences in plain strings are resolved before recovery."""
        code = 'spark.sql("SELECT \\"a\\" FROM t")'
        result = parse_scala_code(code)

        sql_op = next(op for op in result.operations if op.method_name == "sql")
        assert ast.literal_eval(sql_op.arguments[0]) == 'SELECT "a" FROM t'


class TestScalaArguments:
    """Balanced-parenthesis argument extraction."""

    def test_top_level_commas_split_arguments(self) -> None:
        """Commas inside nested calls do not split arguments."""
        code = 'df.join(broadcast(small), Seq("id", "key"))'
        result = parse_scala_code(code)

        join_op = next(op for op in result.operations if op.method_name == "join")
        assert len(join_op.arguments) == 2
        assert join_op.arguments[0] == "broadcast(small)"
        assert join_op.arguments[1] == 'Seq("id", "key")'

    def test_string_arguments_use_python_repr(self) -> None:
        """Plain string arguments match the Python parser's repr format."""
        code = 'df.select("*")'
        result = parse_scala_code(code)

        select_op = next(op for op in result.operations if op.method_name == "select")
        assert select_op.arguments == ["'*'"]

    def test_numeric_argument_text(self) -> None:
        """Numeric arguments keep their literal text."""
        code = "df.repartition(200)"
        result = parse_scala_code(code)

        op = next(op for op in result.operations if op.method_name == "repartition")
        assert op.arguments == ["200"]

    def test_no_arguments(self) -> None:
        """Arity-0 calls produce empty argument lists."""
        code = "df.cache()\ndf.count( )"
        result = parse_scala_code(code)

        assert all(op.arguments == [] for op in result.operations)

    def test_multiline_arguments_collapsed(self) -> None:
        """Arguments spanning lines are whitespace-collapsed."""
        code = 'df.filter(\n  col("a") > 1 &&\n  col("b") < 2\n)'
        result = parse_scala_code(code)

        op = next(op for op in result.operations if op.method_name == "filter")
        assert op.arguments == ['col("a") > 1 && col("b") < 2']


class TestScalaParseFile:
    """File-based parsing."""

    def test_parse_file(self, tmp_path: Path) -> None:
        """parse_file reads and parses a .scala file."""
        scala_file = tmp_path / "Job.scala"
        scala_file.write_text('val df = spark.read.parquet("d")\ndf.collect()\n', encoding="utf-8")

        result = ScalaCodeParser().parse_file(scala_file)
        assert "collect" in _method_names(result)
        assert result.language == "scala"

    def test_parse_file_not_found(self) -> None:
        """parse_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            ScalaCodeParser().parse_file("/non/existent/Job.scala")


class TestDetectLanguage:
    """Language auto-detection heuristic."""

    def test_detects_scala_object(self) -> None:
        """Scala objects with Spark imports are detected."""
        code = "import org.apache.spark.sql.SparkSession\nobject Job {\n}\n"
        assert detect_language(code) == "scala"

    def test_detects_scala_val_assignments(self) -> None:
        """val assignments are a Scala marker."""
        code = 'val df = spark.read.parquet("d")\ndf.show()\n'
        assert detect_language(code) == "scala"

    def test_weak_scala_markers_default_to_python(self) -> None:
        """object/def-with-brace alone is no longer enough to pick Scala."""
        code = "object Job {\n  def main(args: Array[String]): Unit = {\n    df.show()\n  }\n}\n"
        assert detect_language(code) == "python"

    def test_strong_marker_inside_object_detected(self) -> None:
        """A val assignment inside an object classifies as Scala."""
        code = 'object Job {\n  def main(args: Array[String]): Unit = {\n    val df = spark.table("t")\n  }\n}\n'
        assert detect_language(code) == "scala"

    def test_python_docstring_with_scala_example_stays_python(self) -> None:
        """Scala markers inside a Python docstring never flip detection."""
        code = (
            '"""Utility script.\n'
            "\n"
            "Scala equivalent:\n"
            '    val df = spark.read.parquet("x")\n'
            "    import org.apache.spark.sql.SparkSession\n"
            '"""\n'
            'df = spark.read.parquet("x")\n'
            "df.show()\n"
        )
        assert detect_language(code) == "python"

    def test_scala_markers_in_strings_and_comments_ignored(self) -> None:
        """val markers inside string literals or # comments do not count."""
        assert detect_language('text = "val df = 1"\nprint(text)\n') == "python"
        assert detect_language("# val df = 1\nx = 5\n") == "python"

    def test_python_markers_in_scala_strings_ignored(self) -> None:
        """Python markers inside Scala string literals do not flip detection."""
        code = 'val note = "import pyspark"\nval df = spark.read.parquet("x")\n'
        assert detect_language(code) == "scala"

    def test_python_import_wins(self) -> None:
        """pyspark imports always classify as Python."""
        code = "from pyspark.sql import SparkSession\ndf.show()\n"
        assert detect_language(code) == "python"

    def test_python_def_detected(self) -> None:
        """Python function definitions classify as Python."""
        code = "def main():\n    df.show()\n"
        assert detect_language(code) == "python"

    def test_ambiguous_defaults_to_python(self) -> None:
        """Ambiguous source defaults to Python (unchanged behavior)."""
        assert detect_language("x = 5\nprint(x)\n") == "python"
        assert detect_language("") == "python"


class TestParseResultLanguage:
    """ParseResult language tagging."""

    def test_scala_result_is_tagged(self) -> None:
        """Scala parse results carry language='scala'."""
        assert parse_scala_code("df.show()").language == "scala"

    def test_python_result_defaults_to_python(self) -> None:
        """Python parse results keep the default language tag."""
        assert parse_spark_code("df.show()").language == "python"
