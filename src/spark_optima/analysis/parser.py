# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""AST-based parser for PySpark code analysis.

This module provides the SparkCodeParser class that analyzes Python source code
to detect Spark DataFrame operations using the AST module.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from spark_optima.analysis.models import (
    CodeLocation,
    SparkOperation,
    SparkOperationType,
)

logger = logging.getLogger(__name__)


# Map of PySpark method names to their operation types
SPARK_METHOD_MAP: dict[str, SparkOperationType] = {
    # Read operations
    "read": SparkOperationType.READ,
    "readStream": SparkOperationType.READ,
    "table": SparkOperationType.READ,
    "sql": SparkOperationType.READ,
    # Write operations
    "write": SparkOperationType.WRITE,
    "writeStream": SparkOperationType.WRITE,
    "save": SparkOperationType.WRITE,
    "saveAsTable": SparkOperationType.WRITE,
    "insertInto": SparkOperationType.WRITE,
    # Join operations
    "join": SparkOperationType.JOIN,
    "crossJoin": SparkOperationType.JOIN,
    "joinWith": SparkOperationType.JOIN,
    # Aggregation operations
    "groupBy": SparkOperationType.AGGREGATION,
    "rollup": SparkOperationType.AGGREGATION,
    "cube": SparkOperationType.AGGREGATION,
    "pivot": SparkOperationType.AGGREGATION,
    "agg": SparkOperationType.AGGREGATION,
    "reduceByKey": SparkOperationType.AGGREGATION,
    # Cache operations
    "cache": SparkOperationType.CACHE,
    "persist": SparkOperationType.CACHE,
    "unpersist": SparkOperationType.CACHE,
    # Repartition operations
    "repartition": SparkOperationType.REPARTITION,
    "coalesce": SparkOperationType.REPARTITION,
    "partitionBy": SparkOperationType.REPARTITION,
    # UDF operations
    "udf": SparkOperationType.UDF,
    "pandas_udf": SparkOperationType.UDF,
    # Window operations
    "window": SparkOperationType.WINDOW,
    "Window": SparkOperationType.WINDOW,
    # Transformations
    "select": SparkOperationType.TRANSFORMATION,
    "selectExpr": SparkOperationType.TRANSFORMATION,
    "filter": SparkOperationType.TRANSFORMATION,
    "where": SparkOperationType.TRANSFORMATION,
    "withColumn": SparkOperationType.TRANSFORMATION,
    "withColumnRenamed": SparkOperationType.TRANSFORMATION,
    "drop": SparkOperationType.TRANSFORMATION,
    "dropDuplicates": SparkOperationType.TRANSFORMATION,
    "distinct": SparkOperationType.TRANSFORMATION,
    "orderBy": SparkOperationType.TRANSFORMATION,
    "sort": SparkOperationType.TRANSFORMATION,
    "union": SparkOperationType.TRANSFORMATION,
    "unionAll": SparkOperationType.TRANSFORMATION,
    "intersect": SparkOperationType.TRANSFORMATION,
    "exceptAll": SparkOperationType.TRANSFORMATION,
    "map": SparkOperationType.TRANSFORMATION,
    "flatMap": SparkOperationType.TRANSFORMATION,
    "mapPartitions": SparkOperationType.TRANSFORMATION,
    # Actions
    "show": SparkOperationType.ACTION,
    "collect": SparkOperationType.ACTION,
    "count": SparkOperationType.ACTION,
    "take": SparkOperationType.ACTION,
    "first": SparkOperationType.ACTION,
    "head": SparkOperationType.ACTION,
    "foreach": SparkOperationType.ACTION,
    "foreachPartition": SparkOperationType.ACTION,
    "reduce": SparkOperationType.ACTION,
    "toPandas": SparkOperationType.ACTION,
}

# Methods that indicate potential shuffle operations
SHUFFLE_METHODS: set[str] = {
    "join",
    "crossJoin",
    "groupBy",
    "rollup",
    "cube",
    "repartition",
    "distinct",
    "dropDuplicates",
    "orderBy",
    "sort",
    "reduceByKey",
}

# Methods that accept broadcast hints
BROADCAST_METHODS: set[str] = {"join", "crossJoin"}


class SparkCodeParser:
    """Parser for analyzing PySpark code using Python AST.

    This class parses Python source code to detect Spark DataFrame operations,
    track variable assignments, and build an operation graph for analysis.

    Attributes:
        source_code: The original source code being parsed.
        ast_tree: The parsed AST tree.
        operations: List of detected Spark operations.
        dataframe_vars: Mapping of variable names to DataFrame origins.

    Example:
        >>> parser = SparkCodeParser()
        >>> result = parser.parse_file("spark_job.py")
        >>> for op in result.operations:
        ...     print(f"{op.method_name} at line {op.location.line}")

    """

    def __init__(self) -> None:
        """Initialize the parser."""
        self.source_code: str = ""
        self.ast_tree: ast.AST | None = None
        self.operations: list[SparkOperation] = []
        self.dataframe_vars: dict[str, list[SparkOperation]] = {}
        self._chain_counter: int = 0

    def parse_file(self, file_path: str | Path) -> ParseResult:
        """Parse a Python file containing Spark code.

        Args:
            file_path: Path to the Python file.

        Returns:
            ParseResult containing operations and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            SyntaxError: If the file contains invalid Python syntax.

        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        source = file_path.read_text(encoding="utf-8")
        return self.parse_source(source)

    def parse_source(self, source_code: str) -> ParseResult:
        """Parse Python source code containing Spark operations.

        Args:
            source_code: Python source code string.

        Returns:
            ParseResult containing operations and metadata.

        Raises:
            SyntaxError: If the source contains invalid Python syntax.

        """
        self.source_code = source_code
        self.operations = []
        self.dataframe_vars = {}
        self._chain_counter = 0

        try:
            self.ast_tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.error(f"Syntax error in source code: {e}")
            raise

        # Visit all nodes in the AST
        visitor = _SparkVisitor(self)
        visitor.visit(self.ast_tree)

        return ParseResult(
            operations=self.operations,
            dataframe_vars=self.dataframe_vars,
            operation_count=len(self.operations),
        )

    def _add_operation(
        self,
        method_name: str,
        dataframe_var: str,
        arguments: list[str] | None = None,
        location: CodeLocation | None = None,
    ) -> SparkOperation:
        """Add a detected Spark operation.

        Args:
            method_name: Name of the Spark method.
            dataframe_var: Variable name of the DataFrame.
            arguments: List of argument representations.
            location: Source code location.

        Returns:
            The created SparkOperation.

        """
        op_type = SPARK_METHOD_MAP.get(method_name, SparkOperationType.TRANSFORMATION)

        operation = SparkOperation(
            operation_type=op_type,
            method_name=method_name,
            dataframe_var=dataframe_var,
            arguments=arguments or [],
            location=location,
            chain_position=self._chain_counter,
        )

        self._chain_counter += 1
        self.operations.append(operation)

        # Track DataFrame variable lineage
        if dataframe_var not in self.dataframe_vars:
            self.dataframe_vars[dataframe_var] = []
        self.dataframe_vars[dataframe_var].append(operation)

        return operation

    def get_operations_by_type(
        self,
        op_type: SparkOperationType,
    ) -> list[SparkOperation]:
        """Get operations filtered by type.

        Args:
            op_type: Operation type to filter by.

        Returns:
            List of matching operations.

        """
        return [op for op in self.operations if op.operation_type == op_type]

    def get_join_operations(self) -> list[SparkOperation]:
        """Get all join operations.

        Returns:
            List of join operations.

        """
        return self.get_operations_by_type(SparkOperationType.JOIN)

    def get_aggregation_operations(self) -> list[SparkOperation]:
        """Get all aggregation operations.

        Returns:
            List of aggregation operations.

        """
        return self.get_operations_by_type(SparkOperationType.AGGREGATION)

    def get_shuffle_operations(self) -> list[SparkOperation]:
        """Get all operations that cause shuffling.

        Returns:
            List of shuffle-causing operations.

        """
        return [op for op in self.operations if op.method_name in SHUFFLE_METHODS]

    def get_operations_without_broadcast(self) -> list[SparkOperation]:
        """Get join operations without broadcast hints.

        Returns:
            List of join operations missing broadcast hints.

        """
        joins = self.get_join_operations()
        return [op for op in joins if not any("broadcast" in arg.lower() for arg in op.arguments)]

    def get_dataframe_lineage(self, var_name: str) -> list[SparkOperation]:
        """Get the operation lineage for a DataFrame variable.

        Args:
            var_name: Name of the DataFrame variable.

        Returns:
            List of operations in order of execution.

        """
        return self.dataframe_vars.get(var_name, [])


class ParseResult:
    """Result of parsing Spark code.

    Attributes:
        operations: List of detected Spark operations.
        dataframe_vars: Mapping of variable names to their operations.
        operation_count: Total number of operations detected.

    """

    def __init__(
        self,
        operations: list[SparkOperation],
        dataframe_vars: dict[str, list[SparkOperation]],
        operation_count: int,
    ) -> None:
        """Initialize parse result."""
        self.operations = operations
        self.dataframe_vars = dataframe_vars
        self.operation_count = operation_count

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the parse result.

        Returns:
            Dictionary with operation counts by type.

        """
        type_counts: dict[str, int] = {}
        for op in self.operations:
            type_name = op.operation_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "total_operations": self.operation_count,
            "dataframe_variables": list(self.dataframe_vars.keys()),
            "operations_by_type": type_counts,
        }


class _SparkVisitor(ast.NodeVisitor):
    """AST visitor for detecting Spark operations.

    This visitor traverses the AST to find Spark DataFrame method calls
    and track variable assignments.

    """

    def __init__(self, parser: SparkCodeParser) -> None:
        """Initialize the visitor.

        Args:
            parser: The parent parser instance.

        """
        self.parser = parser
        self.current_df_var: str | None = None

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Visit function/method call nodes.

        Args:
            node: AST Call node.

        """
        # Check if this is a DataFrame method call (e.g., df.join(...))
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            df_var = self._get_dataframe_variable(node.func.value)

            if df_var and method_name in SPARK_METHOD_MAP:
                location = self._get_location(node)
                arguments = self._extract_arguments(node)

                self.parser._add_operation(
                    method_name=method_name,
                    dataframe_var=df_var,
                    arguments=arguments,
                    location=location,
                )

        # Check for standalone function calls like broadcast(), udf(), pandas_udf()
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == "broadcast":
                location = self._get_location(node)
                # Track broadcast usage
                self.parser._add_operation(
                    method_name="broadcast",
                    dataframe_var="broadcast_call",
                    arguments=[],
                    location=location,
                )
            elif func_name in ("udf", "pandas_udf"):
                location = self._get_location(node)
                # Track UDF usage
                arguments = self._extract_arguments(node)
                self.parser._add_operation(
                    method_name=func_name,
                    dataframe_var="udf_call",
                    arguments=arguments,
                    location=location,
                )

        # Continue visiting child nodes
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        """Visit assignment nodes to track DataFrame variables.

        Args:
            node: AST Assign node.

        """
        # Track variable assignments from Spark operations
        if isinstance(node.value, ast.Call):
            target_names = self._get_assignment_targets(node.targets)

            # Check if the right-hand side is a Spark operation
            if isinstance(node.value.func, ast.Attribute):
                method_name = node.value.func.attr
                if method_name in SPARK_METHOD_MAP:
                    # The assigned variable inherits the DataFrame lineage
                    for target in target_names:
                        self.current_df_var = target

        self.generic_visit(node)

    def _get_dataframe_variable(self, node: ast.expr) -> str | None:
        """Extract DataFrame variable name from an AST node.

        Args:
            node: AST expression node.

        Returns:
            Variable name if it's a simple name, None otherwise.

        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            # Handle chained calls like df.join().select()
            return self._get_dataframe_variable(node.func)
        elif isinstance(node, ast.Attribute):
            return self._get_dataframe_variable(node.value)
        return None

    def _get_assignment_targets(self, targets: list[ast.expr]) -> list[str]:
        """Extract variable names from assignment targets.

        Args:
            targets: List of assignment target nodes.

        Returns:
            List of variable names.

        """
        names = []
        for target in targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        names.append(elt.id)
        return names

    def _extract_arguments(self, node: ast.Call) -> list[str]:
        """Extract argument representations from a call node.

        Args:
            node: AST Call node.

        Returns:
            List of argument string representations.

        """
        args = []

        # Positional arguments
        for arg in node.args:
            args.append(self._node_to_string(arg))

        # Keyword arguments
        for kw in node.keywords:
            if kw.arg:
                args.append(f"{kw.arg}={self._node_to_string(kw.value)}")

        return args

    def _node_to_string(self, node: ast.AST) -> str:
        """Convert an AST node to a string representation.

        Args:
            node: AST node.

        Returns:
            String representation of the node.

        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str | int | float):
                return repr(node.value)
            else:
                return repr(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self._node_to_string(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            func_str = self._node_to_string(node.func)
            args_str = ", ".join(self._node_to_string(arg) for arg in node.args)
            return f"{func_str}({args_str})"
        elif isinstance(node, ast.List):
            elems = ", ".join(self._node_to_string(e) for e in node.elts)
            return f"[{elems}]"
        elif isinstance(node, ast.Dict):
            return "{...}"  # Simplified dict representation
        elif isinstance(node, ast.Lambda):
            return "lambda..."
        else:
            return ast.dump(node)[:50]  # Fallback

    def _get_location(self, node: ast.AST) -> CodeLocation | None:
        """Extract source location from an AST node.

        Args:
            node: AST node.

        Returns:
            CodeLocation if line info is available, None otherwise.

        """
        if hasattr(node, "lineno") and node.lineno is not None:
            return CodeLocation(
                line=node.lineno,
                column=getattr(node, "col_offset", 0),
                end_line=getattr(node, "end_lineno", None),
                end_column=getattr(node, "end_col_offset", None),
            )
        return None


def parse_spark_code(source_code: str) -> ParseResult:
    """Convenience function to parse Spark code.

    Args:
        source_code: Python source code string.

    Returns:
        ParseResult containing detected operations.

    Example:
        >>> code = '''
        ... from pyspark.sql import SparkSession
        ... spark = SparkSession.builder.getOrCreate()
        ... df = spark.read.parquet("data.parquet")
        ... result = df.groupBy("col").count()
        ... result.show()
        ... '''
        >>> result = parse_spark_code(code)
        >>> print(f"Found {len(result.operations)} operations")

    """
    parser = SparkCodeParser()
    return parser.parse_source(source_code)
