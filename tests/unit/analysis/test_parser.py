# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the Spark code parser."""

import ast

import pytest

from spark_optima.analysis.models import SparkOperationType
from spark_optima.analysis.parser import (
    SparkCodeParser,
    _SparkVisitor,
    parse_spark_code,
)


class TestSparkCodeParser:
    """Test cases for SparkCodeParser."""

    def test_empty_code(self):
        """Test parsing empty code."""
        parser = SparkCodeParser()
        result = parser.parse_source("")
        assert len(result.operations) == 0

    def test_simple_join_detection(self):
        """Test detecting a simple join operation."""
        code = """
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
df1 = spark.read.parquet("table1.parquet")
df2 = spark.read.parquet("table2.parquet")
result = df1.join(df2, "id")
result.show()
"""
        parser = SparkCodeParser()
        result = parser.parse_source(code)

        assert len(result.operations) >= 1
        join_ops = [op for op in result.operations if op.method_name == "join"]
        assert len(join_ops) == 1
        assert join_ops[0].operation_type == SparkOperationType.JOIN

    def test_multiple_operations(self):
        """Test detecting multiple operations."""
        code = """
df = spark.read.parquet("data.parquet")
df_filtered = df.filter(df.age > 18)
df_grouped = df_filtered.groupBy("city").count()
df_grouped.show()
"""
        parser = SparkCodeParser()
        result = parser.parse_source(code)

        assert len(result.operations) >= 3

    def test_dataframe_variable_tracking(self):
        """Test that DataFrame variables are tracked."""
        code = """
df = spark.read.parquet("data.parquet")
df2 = df.filter(df.age > 18)
df3 = df2.groupBy("city").count()
"""
        parser = SparkCodeParser()
        result = parser.parse_source(code)

        # Check that DataFrame variables were tracked
        assert len(result.dataframe_vars) >= 1
        assert "df" in result.dataframe_vars or "df2" in result.dataframe_vars

    def test_get_join_operations(self):
        """Test getting join operations."""
        code = """
df1.join(df2, "id")
df3.join(df4, "key")
df1.filter(df1.x > 0)
"""
        parser = SparkCodeParser()
        parser.parse_source(code)

        joins = parser.get_join_operations()
        assert len(joins) == 2

    def test_get_shuffle_operations(self):
        """Test getting shuffle operations."""
        code = """
df.groupBy("col").count()
df.orderBy("col")
df.repartition(10)
"""
        parser = SparkCodeParser()
        parser.parse_source(code)

        shuffles = parser.get_shuffle_operations()
        assert len(shuffles) >= 2

    def test_get_operations_without_broadcast(self):
        """Test detecting joins without broadcast hints."""
        code = """
from pyspark.sql.functions import broadcast
df1.join(df2, "id")
df1.join(broadcast(df2), "id")
"""
        parser = SparkCodeParser()
        parser.parse_source(code)

        without_broadcast = parser.get_operations_without_broadcast()
        # Should find the join without broadcast hint
        assert len(without_broadcast) >= 1


class TestParseResult:
    """Test cases for ParseResult."""

    def test_get_summary(self):
        """Test summary generation."""
        code = """
df = spark.read.parquet("data.parquet")
df.join(df2, "id")
df.groupBy("col").count()
"""
        result = parse_spark_code(code)
        summary = result.get_summary()

        assert summary["total_operations"] > 0
        assert "operations_by_type" in summary


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_parse_spark_code(self):
        """Test the parse_spark_code convenience function."""
        code = """
df = spark.read.parquet("data.parquet")
df.filter(df.x > 0)
"""
        result = parse_spark_code(code)
        assert isinstance(result.operations, list)


class TestParserEdgeCases:
    """Test edge cases for SparkCodeParser."""

    def test_parse_file_not_found(self):
        """Test parse_file with non-existent file."""
        parser = SparkCodeParser()
        with pytest.raises(FileNotFoundError, match="File not found"):
            parser.parse_file("/non/existent/file.py")

    def test_parse_source_syntax_error(self):
        """Test parse_source with invalid syntax."""
        parser = SparkCodeParser()
        with pytest.raises(SyntaxError):
            parser.parse_source("this is not valid python %%@")

    def test_get_dataframe_variable_unsupported_node(self):
        """Test _get_dataframe_variable with unsupported node type."""
        parser = SparkCodeParser()
        visitor = _SparkVisitor(parser)
        # Create a node type that's not handled (ast.List is just an example)
        node = ast.List(elts=[], ctx=ast.Load())
        result = visitor._get_dataframe_variable(node)
        assert result is None

    def test_get_assignment_targets_tuple(self):
        """Test _get_assignment_targets with tuple assignment."""
        parser = SparkCodeParser()
        visitor = _SparkVisitor(parser)
        # Create a Tuple assignment target: a, b = some_func()
        tuple_node = ast.Tuple(
            elts=[ast.Name(id="a", ctx=ast.Store()), ast.Name(id="b", ctx=ast.Store())],
            ctx=ast.Store(),
        )
        targets = visitor._get_assignment_targets([tuple_node])
        assert "a" in targets
        assert "b" in targets

    def test_extract_arguments_with_keywords(self):
        """Test _extract_arguments with keyword arguments."""
        parser = SparkCodeParser()
        visitor = _SparkVisitor(parser)
        # Create a call node with keyword arguments
        node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="df", ctx=ast.Load()),
                attr="join",
                ctx=ast.Load(),
            ),
            args=[ast.Name(id="df2", ctx=ast.Load())],
            keywords=[ast.keyword(arg="on", value=ast.Constant(value="id"))],
        )
        args = visitor._extract_arguments(node)
        assert any("on=" in arg for arg in args)

    def test_node_to_string_list(self):
        """Test _node_to_string with List node."""
        parser = SparkCodeParser()
        visitor = _SparkVisitor(parser)
        node = ast.List(
            elts=[ast.Constant(value=1), ast.Constant(value=2)],
            ctx=ast.Load(),
        )
        result = visitor._node_to_string(node)
        assert result == "[1, 2]"

    def test_node_to_string_lambda(self):
        """Test _node_to_string with Lambda node."""
        parser = SparkCodeParser()
        visitor = _SparkVisitor(parser)
        # Create a simple lambda: lambda x: x + 1
        node = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[], args=[ast.arg(arg="x")], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=ast.BinOp(
                left=ast.Name(id="x", ctx=ast.Load()),
                op=ast.Add(),
                right=ast.Constant(value=1),
            ),
        )
        result = visitor._node_to_string(node)
        assert result == "lambda..."

    def test_node_to_string_dict(self):
        """Test _node_to_string with Dict node."""
        parser = SparkCodeParser()
        visitor = _SparkVisitor(parser)
        node = ast.Dict(
            keys=[ast.Constant(value="key")],
            values=[ast.Constant(value="value")],
        )
        result = visitor._node_to_string(node)
        assert result == "{...}"

    def test_get_location_no_lineno(self):
        """Test _get_location with node missing lineno."""
        parser = SparkCodeParser()
        visitor = _SparkVisitor(parser)
        # ast.Pass doesn't have lineno set in some cases
        node = ast.Pass()
        # Manually remove lineno if present
        if hasattr(node, "lineno"):
            delattr(node, "lineno")
        result = visitor._get_location(node)
        assert result is None

    def test_get_operations_by_type(self):
        """Test get_operations_by_type method."""
        code = """
df = spark.read("data.parquet")
df.filter(df.x > 0)
df.show()
"""
        parser = SparkCodeParser()
        parser.parse_source(code)
        # The get_operations_by_type method itself is what we're testing
        # Just verify it returns a list
        ops = parser.get_operations_by_type(SparkOperationType.READ)
        assert isinstance(ops, list)
        ops = parser.get_operations_by_type(SparkOperationType.JOIN)
        assert isinstance(ops, list)

    def test_parse_result_get_summary_with_dataframe_vars(self):
        """Test ParseResult.get_summary includes dataframe variables."""
        code = """
df1 = spark.read.parquet("data")
df2 = df1.filter(df1.x > 0)
"""
        result = parse_spark_code(code)
        summary = result.get_summary()
        assert "dataframe_variables" in summary
        assert len(summary["dataframe_variables"]) > 0

    def test_node_to_string_str(self):
        """Test _node_to_string with ast.Constant for string values."""
        parser = SparkCodeParser()
        visitor = _SparkVisitor(parser)
        # Use ast.Constant instead of deprecated ast.Str (removed in Python 3.14+)
        node = ast.Constant(value="test_string")
        result = visitor._node_to_string(node)
        assert result == "'test_string'"

    def test_node_to_string_num(self):
        """Test _node_to_string with ast.Constant for numeric values."""
        parser = SparkCodeParser()
        visitor = _SparkVisitor(parser)
        # Use ast.Constant instead of deprecated ast.Num (removed in Python 3.14+)
        node = ast.Constant(value=42)
        result = visitor._node_to_string(node)
        assert result == "42"
