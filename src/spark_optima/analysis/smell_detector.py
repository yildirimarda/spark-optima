# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Code smell detection for Spark applications.

This module provides the SmellDetector class that analyzes parsed Spark code
to detect various performance anti-patterns and code smells.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from spark_optima.analysis.models import (
    AnalysisResult,
    CodeLocation,
    CodeSmell,
    SeverityLevel,
    SparkOperationType,
)
from spark_optima.analysis.parser import (
    SHUFFLE_METHODS,
    ParseResult,
    SparkCodeParser,
)

logger = logging.getLogger(__name__)

# Action methods that trigger a Spark job when chained after a transformation
_CHAINED_ACTION_METHODS: frozenset[str] = frozenset(
    {
        "show",
        "collect",
        "count",
        "take",
        "first",
        "head",
        "toPandas",
        "foreach",
        "foreachPartition",
        "reduce",
    }
)

# Methods that start or perform a write when chained after a transformation
_WRITE_CHAIN_METHODS: frozenset[str] = frozenset({"write", "writeStream", "save", "saveAsTable", "insertInto"})

# Argument representations that select every column (as produced by the parser)
_STAR_ARGUMENTS: frozenset[str] = frozenset({"'*'", '"*"', "col('*')", 'col("*")'})

# Regexes for lightweight SQL string analysis (case-insensitive)
_SQL_SELECT_STAR_RE = re.compile(r"\bSELECT\s+\*", re.IGNORECASE)
_SQL_CROSS_JOIN_RE = re.compile(r"\bCROSS\s+JOIN\b", re.IGNORECASE)


class SmellDetector:
    """Detector for Spark code smells and anti-patterns.

    This class analyzes parsed Spark code to detect various performance issues
    such as missing broadcast hints, unnecessary shuffles, caching problems,
    UDF usage, and data skew potential.

    Attributes:
        parser: The SparkCodeParser instance used for parsing.
        detected_smells: List of detected code smells.

    Example:
        >>> detector = SmellDetector()
        >>> result = detector.analyze_file("spark_job.py")
        >>> for smell in result.smells:
        ...     print(f"{smell.smell_type}: {smell.description}")

    """

    def __init__(self) -> None:
        """Initialize the smell detector."""
        self.parser = SparkCodeParser()
        self.detected_smells: list[CodeSmell] = []
        self._detection_rules: list[Callable[[ParseResult], list[CodeSmell]]] = [
            self._detect_missing_broadcast_hints,
            self._detect_unnecessary_shuffles,
            self._detect_caching_issues,
            self._detect_udf_usage,
            self._detect_data_skew_potential,
            self._detect_repartition_issues,
            self._detect_small_file_problems,
            self._detect_serialization_issues,
            self._detect_large_collect,
            self._detect_cartesian_join,
            self._detect_topandas_usage,
            self._detect_count_for_empty_check,
            self._detect_single_partition_write,
            self._detect_infer_schema,
            self._detect_withcolumn_in_loop,
            self._detect_select_star,
            self._detect_orderby_without_limit,
            self._detect_sql_antipatterns,
        ]

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyze a Python file for code smells.

        Args:
            file_path: Path to the Python file.

        Returns:
            AnalysisResult containing detected smells.

        """
        parse_result = self.parser.parse_file(file_path)
        return self._analyze_parse_result(parse_result)

    def analyze_source(self, source_code: str) -> AnalysisResult:
        """Analyze Python source code for code smells.

        Args:
            source_code: Python source code string.

        Returns:
            AnalysisResult containing detected smells.

        """
        parse_result = self.parser.parse_source(source_code)
        return self._analyze_parse_result(parse_result)

    def _analyze_parse_result(self, parse_result: ParseResult) -> AnalysisResult:
        """Analyze a parse result for code smells.

        Args:
            parse_result: Result from parsing Spark code.

        Returns:
            AnalysisResult with all detected smells.

        """
        self.detected_smells = []

        # Run all detection rules
        for rule in self._detection_rules:
            try:
                smells = rule(parse_result)
                self.detected_smells.extend(smells)
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Detection rule {rule.__name__} failed: {e}")

        # Create analysis result
        return AnalysisResult(
            operations=parse_result.operations,
            smells=self.detected_smells,
            metadata={
                "total_operations": parse_result.operation_count,
                "smell_count": len(self.detected_smells),
                "dataframe_variables": list(parse_result.dataframe_vars.keys()),
            },
        )

    def _detect_missing_broadcast_hints(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Detect join operations without broadcast hints.

        Small DataFrames should be broadcasted to avoid shuffling.

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        joins = parse_result.operations
        joins = [op for op in joins if op.operation_type == SparkOperationType.JOIN]

        for join_op in joins:
            # Check if broadcast hint is already present
            has_broadcast = any("broadcast" in arg.lower() or "hint" in arg.lower() for arg in join_op.arguments)

            if not has_broadcast:
                smell = CodeSmell(
                    smell_type="missing_broadcast_hint",
                    description=(
                        f"Join operation at {join_op.location} does not use "
                        "broadcast hint. Consider broadcasting the smaller DataFrame "
                        "to avoid expensive shuffle operations."
                    ),
                    severity=SeverityLevel.MEDIUM,
                    location=join_op.location,
                    affected_operation=join_op,
                    impact=(
                        "Without broadcast, both DataFrames are shuffled across the "
                        "network, causing significant performance degradation for "
                        "small-large table joins."
                    ),
                )
                smells.append(smell)

        return smells

    def _detect_unnecessary_shuffles(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Detect operations that may cause unnecessary shuffling.

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        shuffle_ops = [op for op in parse_result.operations if op.method_name in SHUFFLE_METHODS]

        # Group consecutive shuffle operations
        sorted_ops = sorted(shuffle_ops, key=lambda x: x.chain_position)

        for i, op in enumerate(sorted_ops):
            # Check for consecutive repartitions
            if op.method_name == "repartition":
                # Check if there's another repartition shortly after
                for next_op in sorted_ops[i + 1 :]:
                    if next_op.chain_position - op.chain_position <= 3 and next_op.method_name in [
                        "repartition",
                        "coalesce",
                    ]:
                        smell = CodeSmell(
                            smell_type="unnecessary_shuffle",
                            description=(
                                f"Multiple repartition/coalesce operations detected "
                                f"near line {op.location.line if op.location else 'unknown'}. "
                                "Consider combining them into a single operation."
                            ),
                            severity=SeverityLevel.HIGH,
                            location=op.location,
                            affected_operation=op,
                            impact=(
                                "Each repartition triggers a full shuffle of data "
                                "across the network. Multiple repartitions waste "
                                "resources and significantly increase execution time."
                            ),
                        )
                        smells.append(smell)
                        break

            # Check for repartition before sort (sort already shuffles)
            if op.method_name in ["orderBy", "sort"]:
                for prev_op in sorted_ops[:i]:
                    if op.chain_position - prev_op.chain_position <= 2 and prev_op.method_name == "repartition":
                        smell = CodeSmell(
                            smell_type="unnecessary_shuffle",
                            description=(
                                f"Repartition followed by sort at line "
                                f"{op.location.line if op.location else 'unknown'}. "
                                "Sort already triggers a shuffle; repartition is redundant."
                            ),
                            severity=SeverityLevel.MEDIUM,
                            location=op.location,
                            affected_operation=op,
                            impact=(
                                "Sort operations already redistribute data across "
                                "partitions. Prior repartition wastes a full shuffle."
                            ),
                        )
                        smells.append(smell)
                        break

        return smells

    def _detect_caching_issues(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Detect caching/persist strategy issues.

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        cache_ops = [op for op in parse_result.operations if op.method_name in ["cache", "persist"]]

        # Check for cache without subsequent action
        for cache_op in cache_ops:
            # Find subsequent operations on same DataFrame
            subsequent_ops = [
                op
                for op in parse_result.operations
                if (op.chain_position > cache_op.chain_position and op.dataframe_var == cache_op.dataframe_var)
            ]

            # Check if cache is used multiple times
            action_count = sum(1 for op in subsequent_ops if op.operation_type == SparkOperationType.ACTION)

            if action_count < 2:
                smell = CodeSmell(
                    smell_type="caching_issue",
                    description=(
                        f"Cache/persist at line "
                        f"{cache_op.location.line if cache_op.location else 'unknown'} "
                        f"is used only {action_count} time(s). Caching has overhead and "
                        "should only be used when DataFrame is reused."
                    ),
                    severity=SeverityLevel.LOW,
                    location=cache_op.location,
                    affected_operation=cache_op,
                    impact=(
                        "Unnecessary caching wastes memory and adds serialization overhead without providing benefits."
                    ),
                )
                smells.append(smell)

        # Check for missing cache on reused DataFrames
        df_usage_counts: dict[str, int] = {}
        for op in parse_result.operations:
            df_usage_counts[op.dataframe_var] = df_usage_counts.get(op.dataframe_var, 0) + 1

        for df_var, count in df_usage_counts.items():
            if count > 3:  # Used in more than 3 operations
                # Check if it's cached
                df_ops = parse_result.dataframe_vars.get(df_var, [])
                has_cache = any(op.method_name in ["cache", "persist"] for op in df_ops)

                if not has_cache:
                    # Find the first operation to use as location
                    first_op = df_ops[0] if df_ops else None
                    smell = CodeSmell(
                        smell_type="caching_issue",
                        description=(
                            f"DataFrame '{df_var}' is used {count} times but not cached. "
                            "Consider caching if it's expensive to compute."
                        ),
                        severity=SeverityLevel.LOW,
                        location=first_op.location if first_op else None,
                        affected_operation=first_op,
                        impact=(
                            "Without caching, transformations are recomputed for each "
                            "action, wasting CPU and I/O resources."
                        ),
                    )
                    smells.append(smell)

        return smells

    def _detect_udf_usage(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Detect UDF usage that prevents Catalyst optimization.

        Plain Python UDFs are flagged as HIGH severity because they require
        row-by-row serialization between the JVM and Python. Pandas UDFs are
        flagged as MEDIUM severity: they are vectorized via Arrow but still
        slower than built-in functions and opaque to the Catalyst optimizer.

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        udf_ops = [op for op in parse_result.operations if op.operation_type == SparkOperationType.UDF]

        for udf_op in udf_ops:
            line = udf_op.location.line if udf_op.location else "unknown"
            if udf_op.method_name == "pandas_udf":
                smell = CodeSmell(
                    smell_type="pandas_udf_usage",
                    description=(
                        f"pandas_udf usage detected at line {line}. "
                        "Pandas UDFs are vectorized but still slower than built-in "
                        "functions and invisible to the Catalyst optimizer. "
                        "Prefer built-in Spark SQL functions when an equivalent exists."
                    ),
                    severity=SeverityLevel.MEDIUM,
                    location=udf_op.location,
                    affected_operation=udf_op,
                    impact=(
                        "Pandas UDFs transfer data between JVM and Python via Arrow, "
                        "which is efficient but still adds overhead and prevents "
                        "Catalyst optimizations such as predicate pushdown."
                    ),
                )
            else:
                smell = CodeSmell(
                    smell_type="udf_usage",
                    description=(
                        f"UDF usage detected at line {line}. "
                        "UDFs prevent Catalyst optimizer from optimizing the query. "
                        "Consider using built-in functions or Pandas UDFs."
                    ),
                    severity=SeverityLevel.HIGH,
                    location=udf_op.location,
                    affected_operation=udf_op,
                    impact=(
                        "Regular Python UDFs run in separate Python processes, "
                        "preventing Catalyst optimizations and causing serialization "
                        "overhead. Use Spark SQL functions or Pandas UDFs when possible."
                    ),
                )
            smells.append(smell)

        return smells

    def _detect_data_skew_potential(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Detect operations that may cause data skew.

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []

        # Check joins and aggregations
        skew_sensitive_ops = [
            op
            for op in parse_result.operations
            if op.operation_type in [SparkOperationType.JOIN, SparkOperationType.AGGREGATION]
        ]

        for op in skew_sensitive_ops:
            # Check if arguments suggest potential skew. Empty arguments simply
            # mean no column signal — the large-dataset heuristic below must
            # still apply, so do not skip the operation entirely.
            args_str = " ".join(op.arguments).lower()

            # Common high-cardinality / frequently-skewed column names
            skew_column_indicators = [
                "user_id",
                "customer_id",
                "null",
                "none",
                "0",
            ]

            has_skew_column = bool(op.arguments) and any(indicator in args_str for indicator in skew_column_indicators)

            # Retrieve the data size if available in the operation metadata
            data_size_gb = getattr(op, "data_size_gb", None) or 0
            is_large_dataset = data_size_gb > 1

            # Only flag when there is a concrete skew signal:
            # either a known high-skew column name, or the dataset is large.
            if has_skew_column or is_large_dataset:
                smell = CodeSmell(
                    smell_type="data_skew_potential",
                    description=(
                        f"{op.operation_type.name} operation at line "
                        f"{op.location.line if op.location else 'unknown'} "
                        f"may be susceptible to data skew. "
                        f"Consider using salting or adaptive query execution."
                    ),
                    severity=SeverityLevel.MEDIUM,
                    location=op.location,
                    affected_operation=op,
                    impact=(
                        "Data skew can cause some partitions to be much larger "
                        "than others, leading to out-of-memory errors or "
                        "significantly increased execution time."
                    ),
                )
                smells.append(smell)

        return smells

    def _detect_repartition_issues(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Detect repartition vs coalesce misuse.

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        repartition_ops = [op for op in parse_result.operations if op.method_name == "repartition"]

        for op in repartition_ops:
            # Check if decreasing partitions (should use coalesce)
            args_str = " ".join(op.arguments)

            # Simple heuristic: repartition with small number might be decreasing
            import re

            numbers = re.findall(r"\d+", args_str)
            if numbers:
                partition_count = int(numbers[0])
                if partition_count < 10:  # Likely decreasing partitions
                    smell = CodeSmell(
                        smell_type="repartition_vs_coalesce",
                        description=(
                            f"repartition({partition_count}) at line "
                            f"{op.location.line if op.location else 'unknown'} "
                            "should use coalesce() instead to avoid full shuffle."
                        ),
                        severity=SeverityLevel.MEDIUM,
                        location=op.location,
                        affected_operation=op,
                        impact=(
                            "repartition() always triggers a full shuffle. "
                            "coalesce() avoids shuffling when reducing partitions, "
                            "significantly improving performance."
                        ),
                    )
                    smells.append(smell)

        return smells

    def _detect_small_file_problems(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Detect potential small file problems.

        Args:
            parse_result: parse_result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []

        # Check for write operations without partitioning
        write_ops = [op for op in parse_result.operations if op.operation_type == SparkOperationType.WRITE]

        for op in write_ops:
            args_str = " ".join(op.arguments).lower()

            # Check if partitioned write
            is_partitioned = "partitionby" in args_str or "bucket" in args_str

            if not is_partitioned:
                smell = CodeSmell(
                    smell_type="small_file_problem",
                    description=(
                        f"Write operation at line "
                        f"{op.location.line if op.location else 'unknown'} "
                        "does not use partitioning. This may create many small "
                        "files which hurt read performance."
                    ),
                    severity=SeverityLevel.LOW,
                    location=op.location,
                    affected_operation=op,
                    impact=(
                        "Many small files cause metadata overhead and slow down "
                        "subsequent reads. Use partitioning for large datasets."
                    ),
                )
                smells.append(smell)

        return smells

    def _detect_serialization_issues(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Detect potential serialization configuration issues.

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []

        # Check for operations that benefit from Kryo serialization
        serialization_heavy_ops = [
            op for op in parse_result.operations if op.method_name in ["map", "flatMap", "mapPartitions", "filter"]
        ]

        if len(serialization_heavy_ops) > 5:
            first_op = serialization_heavy_ops[0]
            smell = CodeSmell(
                smell_type="serialization_issue",
                description=(
                    f"Found {len(serialization_heavy_ops)} operations that involve "
                    "data serialization. Ensure Kryo serialization is enabled "
                    "for better performance."
                ),
                severity=SeverityLevel.LOW,
                location=first_op.location,
                affected_operation=first_op,
                impact=(
                    "Java serialization is slow and creates large byte arrays. "
                    "Kryo serialization is faster and more compact."
                ),
            )
            smells.append(smell)

        return smells

    def _detect_large_collect(self, parse_result: ParseResult) -> list[CodeSmell]:  # noqa: ARG002
        """Detect .collect() calls without a preceding .limit() that could OOM the driver.

        Args:
            parse_result: Parse result to analyze (unused; AST is read from the parser).

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []

        tree = self.parser.ast_tree
        if tree is None:
            return smells

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "collect":
                # Walk up the call chain (up to 5 levels) looking for .limit()
                has_limit = False
                current = node.func.value
                for _ in range(5):
                    if isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
                        if current.func.attr == "limit":
                            has_limit = True
                            break
                        current = current.func.value
                    else:
                        break

                if not has_limit:
                    smells.append(
                        CodeSmell(
                            smell_type="large_collect",
                            description=("collect() without limit() can cause driver OOM on large datasets"),
                            severity=SeverityLevel.MEDIUM,
                            location=self._make_location(node),
                            affected_operation=None,
                            impact=(
                                "Calling collect() pulls all data into driver memory. "
                                "Use .limit(N).collect() or write to storage instead."
                            ),
                        )
                    )

        return smells

    def _detect_cartesian_join(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Detect crossJoin() calls that produce cartesian products.

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        cross_joins = [op for op in parse_result.operations if op.method_name == "crossJoin"]

        for op in cross_joins:
            smells.append(
                CodeSmell(
                    smell_type="cartesian_join",
                    description=(
                        f"crossJoin() at line {op.location.line if op.location else 'unknown'} "
                        "produces a cartesian product. Result size is the product of both "
                        "input row counts, which explodes for non-trivial inputs."
                    ),
                    severity=SeverityLevel.HIGH,
                    location=op.location,
                    affected_operation=op,
                    impact=(
                        "A cartesian product multiplies the row counts of both inputs, "
                        "causing massive shuffle volumes, executor OOM, and runaway "
                        "execution times. Use an equi-join with a key, or broadcast "
                        "the smaller side if the cross join is intentional and tiny."
                    ),
                )
            )

        return smells

    def _detect_topandas_usage(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Detect toPandas() calls that pull the full dataset into the driver.

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        topandas_ops = [op for op in parse_result.operations if op.method_name == "toPandas"]

        for op in topandas_ops:
            smells.append(
                CodeSmell(
                    smell_type="topandas_usage",
                    description=(
                        f"toPandas() at line {op.location.line if op.location else 'unknown'} "
                        "materializes the entire DataFrame in driver memory. Limit or "
                        "sample the data first, or keep processing distributed."
                    ),
                    severity=SeverityLevel.HIGH,
                    location=op.location,
                    affected_operation=op,
                    impact=(
                        "toPandas() collects all rows to the driver as a pandas "
                        "DataFrame, which can crash the driver with OOM on large "
                        "datasets. Enable Arrow and reduce the data volume before "
                        "converting."
                    ),
                )
            )

        return smells

    def _detect_count_for_empty_check(self, parse_result: ParseResult) -> list[CodeSmell]:  # noqa: ARG002
        """Detect count() compared against zero used as an emptiness check.

        Matches comparisons such as ``df.count() == 0``, ``df.count() > 0``,
        ``df.count() != 0`` and the reversed forms (``0 == df.count()``).

        Args:
            parse_result: Parse result to analyze (unused; AST is read from the parser).

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        tree = self.parser.ast_tree
        if tree is None:
            return smells

        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
                continue

            left, right = node.left, node.comparators[0]
            count_side = left if self._is_count_call(left) else right if self._is_count_call(right) else None
            other_side = right if count_side is left else left

            if count_side is None or not self._is_int_constant(other_side, 0):
                continue

            smells.append(
                CodeSmell(
                    smell_type="count_for_empty_check",
                    description=(
                        f"count() compared against 0 at line {node.lineno} is used as an "
                        "emptiness check. Use df.isEmpty() (Spark 3.3+) or "
                        "len(df.take(1)) == 0 instead."
                    ),
                    severity=SeverityLevel.MEDIUM,
                    location=self._make_location(node),
                    affected_operation=None,
                    impact=(
                        "count() scans every partition of the dataset just to decide "
                        "whether it is empty. isEmpty() or take(1) stop after the first "
                        "row is found, which is dramatically cheaper on large data."
                    ),
                )
            )

        return smells

    def _detect_single_partition_write(self, parse_result: ParseResult) -> list[CodeSmell]:  # noqa: ARG002
        """Detect repartition(1)/coalesce(1) followed by a write.

        Flags fluent chains such as ``df.coalesce(1).write.parquet(...)`` as
        well as the two-statement form where the single-partition DataFrame is
        assigned to a variable that is later written.

        Args:
            parse_result: Parse result to analyze (unused; AST is read from the parser).

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        tree = self.parser.ast_tree
        if tree is None:
            return smells

        parent_map = self._build_parent_map(tree)

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("repartition", "coalesce")
            ):
                continue
            if not (node.args and self._is_int_constant(node.args[0], 1)):
                continue

            chained, top = self._collect_chained_methods(node, parent_map)
            # Only treat it as a write bottleneck when the write starts within
            # a few chain positions of the repartition/coalesce call.
            followed_by_write = any(method in _WRITE_CHAIN_METHODS for method in chained[:4])
            if not followed_by_write:
                followed_by_write = self._assigned_var_is_written(top, parent_map, tree)

            if followed_by_write:
                smells.append(
                    CodeSmell(
                        smell_type="single_partition_write",
                        description=(
                            f"{node.func.attr}(1) before a write at line {node.lineno} "
                            "forces the entire output through a single task."
                        ),
                        severity=SeverityLevel.MEDIUM,
                        location=self._make_location(node),
                        affected_operation=None,
                        impact=(
                            "Writing through one partition serializes the whole output "
                            "on a single executor core, creating a bottleneck and a "
                            "single large file. Let Spark write multiple files, or "
                            "compact afterwards if a single file is truly required."
                        ),
                    )
                )

        return smells

    def _detect_infer_schema(self, parse_result: ParseResult) -> list[CodeSmell]:  # noqa: ARG002
        """Detect schema inference in read operations.

        Matches both the ``inferSchema=True`` keyword argument and
        ``.option("inferSchema", "true")`` builder calls.

        Args:
            parse_result: Parse result to analyze (unused; AST is read from the parser).

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        tree = self.parser.ast_tree
        if tree is None:
            return smells

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            flagged = any(kw.arg == "inferSchema" and self._is_truthy_literal(kw.value) for kw in node.keywords)
            if (
                not flagged
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "option"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "inferSchema"
                and self._is_truthy_literal(node.args[1])
            ):
                flagged = True

            if flagged:
                smells.append(
                    CodeSmell(
                        smell_type="infer_schema",
                        description=(
                            f"Schema inference enabled at line {node.lineno}. "
                            "Provide an explicit schema instead of inferSchema=True."
                        ),
                        severity=SeverityLevel.MEDIUM,
                        location=self._make_location(node),
                        affected_operation=None,
                        impact=(
                            "Schema inference triggers an extra full pass over the input "
                            "data before the actual read, doubling I/O. It can also infer "
                            "wrong or unstable types. An explicit StructType schema avoids "
                            "both problems."
                        ),
                    )
                )

        return smells

    def _detect_withcolumn_in_loop(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Detect withColumn() calls inside for/while loops.

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        loop_ops = [op for op in parse_result.operations if op.method_name == "withColumn" and op.in_loop]

        for op in loop_ops:
            smells.append(
                CodeSmell(
                    smell_type="withcolumn_in_loop",
                    description=(
                        f"withColumn() inside a loop at line "
                        f"{op.location.line if op.location else 'unknown'}. Each call adds "
                        "a projection to the logical plan; use a single select() with all "
                        "column expressions instead."
                    ),
                    severity=SeverityLevel.HIGH,
                    location=op.location,
                    affected_operation=op,
                    impact=(
                        "Calling withColumn() repeatedly in a loop grows the logical plan "
                        "by one projection per iteration, causing exponential analysis "
                        "time in the Catalyst optimizer and potential StackOverflowError. "
                        "A single select() with multiple expressions builds one projection."
                    ),
                )
            )

        return smells

    def _detect_select_star(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Detect select("*") calls that defeat column pruning.

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        select_ops = [op for op in parse_result.operations if op.method_name == "select"]

        for op in select_ops:
            if not any(arg.strip() in _STAR_ARGUMENTS for arg in op.arguments):
                continue
            smells.append(
                CodeSmell(
                    smell_type="select_star",
                    description=(
                        f"select('*') at line {op.location.line if op.location else 'unknown'} "
                        "reads every column. Select only the columns you need so Spark can "
                        "prune the rest."
                    ),
                    severity=SeverityLevel.LOW,
                    location=op.location,
                    affected_operation=op,
                    impact=(
                        "Selecting all columns prevents column pruning, increasing I/O, "
                        "memory, and shuffle volume — especially costly for wide tables "
                        "stored in columnar formats like Parquet."
                    ),
                )
            )

        return smells

    def _detect_orderby_without_limit(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Detect orderBy/sort that feeds an action or write without a limit.

        A global sort that is consumed in full is one of the most expensive
        operations in Spark. The check covers fluent chains such as
        ``df.orderBy(...).show()`` and the two-statement form where the sorted
        DataFrame is assigned and used later.

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        tree = self.parser.ast_tree
        if tree is None:
            return smells

        parent_map = self._build_parent_map(tree)

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("orderBy", "sort")
            ):
                continue

            chained, top = self._collect_chained_methods(node, parent_map)
            if "limit" in chained:
                continue

            followed_by_consumer = any(
                method in _CHAINED_ACTION_METHODS or method in _WRITE_CHAIN_METHODS for method in chained
            )
            if not followed_by_consumer:
                followed_by_consumer = self._sorted_var_reaches_action(top, parent_map, tree)

            if followed_by_consumer:
                smells.append(
                    CodeSmell(
                        smell_type="orderby_without_limit",
                        description=(
                            f"{node.func.attr}() at line {node.lineno} performs a full "
                            "global sort that is consumed without a limit()."
                        ),
                        severity=SeverityLevel.MEDIUM,
                        location=self._make_location(node),
                        affected_operation=None,
                        impact=(
                            "A global sort shuffles and range-partitions the entire "
                            "dataset. Without a limit, every row must be sorted, which is "
                            "expensive and often unnecessary. Add limit(N) for top-N "
                            "queries, or use sortWithinPartitions() when global order is "
                            "not required."
                        ),
                    )
                )

        return smells

    def _detect_sql_antipatterns(self, parse_result: ParseResult) -> list[CodeSmell]:
        """Inspect spark.sql() string literals for SQL anti-patterns.

        Only plain string literals are analyzed; f-strings, variables, and
        concatenations are skipped gracefully. Detects ``SELECT *`` (column
        pruning) and ``CROSS JOIN`` (cartesian product), case-insensitively.

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells: list[CodeSmell] = []
        sql_ops = [op for op in parse_result.operations if op.method_name == "sql" and op.arguments]

        for op in sql_ops:
            sql_text = self._extract_string_literal(op.arguments[0])
            if sql_text is None:
                continue

            line = op.location.line if op.location else "unknown"

            if _SQL_SELECT_STAR_RE.search(sql_text):
                smells.append(
                    CodeSmell(
                        smell_type="select_star",
                        description=(
                            f"SELECT * in spark.sql() at line {line} reads every column. "
                            "List only the columns you need so Spark can prune the rest."
                        ),
                        severity=SeverityLevel.LOW,
                        location=op.location,
                        affected_operation=op,
                        impact=(
                            "SELECT * prevents column pruning, increasing I/O, memory, "
                            "and shuffle volume — especially costly for wide tables "
                            "stored in columnar formats like Parquet."
                        ),
                    )
                )

            if _SQL_CROSS_JOIN_RE.search(sql_text):
                smells.append(
                    CodeSmell(
                        smell_type="cartesian_join",
                        description=(
                            f"CROSS JOIN in spark.sql() at line {line} produces a "
                            "cartesian product. Result size is the product of both "
                            "input row counts."
                        ),
                        severity=SeverityLevel.HIGH,
                        location=op.location,
                        affected_operation=op,
                        impact=(
                            "A cartesian product multiplies the row counts of both "
                            "inputs, causing massive shuffle volumes, executor OOM, and "
                            "runaway execution times. Use an equi-join with a key "
                            "whenever possible."
                        ),
                    )
                )

        return smells

    # ------------------------------------------------------------------
    # AST helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _make_location(node: ast.AST) -> CodeLocation | None:
        """Build a CodeLocation from an AST node if line info is available.

        Args:
            node: AST node.

        Returns:
            CodeLocation if the node carries line information, None otherwise.

        """
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            return None
        return CodeLocation(
            line=lineno,
            column=getattr(node, "col_offset", 0),
            end_line=getattr(node, "end_lineno", None),
            end_column=getattr(node, "end_col_offset", None),
        )

    @staticmethod
    def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
        """Map every AST node to its parent node.

        Args:
            tree: Root of the AST.

        Returns:
            Dictionary mapping each child node to its parent.

        """
        return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    @staticmethod
    def _collect_chained_methods(
        node: ast.AST,
        parent_map: dict[ast.AST, ast.AST],
        max_depth: int = 12,
    ) -> tuple[list[str], ast.AST]:
        """Collect method names applied after a node in the same fluent chain.

        Walks upward through enclosing Attribute/Call nodes. For
        ``df.orderBy("c").limit(10).show()`` starting at the orderBy call it
        returns ``(["limit", "show"], <outermost call node>)``.

        Args:
            node: AST node to start from (typically a Call).
            parent_map: Child-to-parent node mapping.
            max_depth: Maximum number of upward steps to take.

        Returns:
            Tuple of (method names in execution order, topmost chain node).

        """
        methods: list[str] = []
        current: ast.AST = node
        for _ in range(max_depth):
            parent = parent_map.get(current)
            if isinstance(parent, ast.Attribute) and parent.value is current:
                methods.append(parent.attr)
                current = parent
            elif isinstance(parent, ast.Call) and parent.func is current:
                current = parent
            else:
                break
        return methods, current

    @staticmethod
    def _assigned_var_is_written(
        top: ast.AST,
        parent_map: dict[ast.AST, ast.AST],
        tree: ast.AST,
    ) -> bool:
        """Check whether the variable assigned from a chain is later written.

        Args:
            top: Topmost node of the originating call chain.
            parent_map: Child-to-parent node mapping.
            tree: Root of the AST to search for write accesses.

        Returns:
            True if the assigned variable is used in a write chain later.

        """
        parent = parent_map.get(top)
        if not isinstance(parent, ast.Assign):
            return False

        target_names = {target.id for target in parent.targets if isinstance(target, ast.Name)}
        if not target_names:
            return False

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in _WRITE_CHAIN_METHODS
                and isinstance(node.value, ast.Name)
                and node.value.id in target_names
            ):
                return True
        return False

    @classmethod
    def _sorted_var_reaches_action(
        cls,
        top: ast.AST,
        parent_map: dict[ast.AST, ast.AST],
        tree: ast.AST,
    ) -> bool:
        """Check whether a sorted DataFrame variable feeds an action or write.

        Args:
            top: Topmost node of the orderBy/sort call chain.
            parent_map: Child-to-parent node mapping.
            tree: Root of the AST to search for variable accesses.

        Returns:
            True if the variable reaches an action/write with no limit() seen.

        """
        parent = parent_map.get(top)
        if not isinstance(parent, ast.Assign):
            return False

        target_names = {target.id for target in parent.targets if isinstance(target, ast.Name)}
        if not target_names:
            return False

        reaches_consumer = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or cls._root_name(node.value) not in target_names:
                continue
            # A limit() anywhere on the sorted variable makes the sort bounded.
            if node.attr == "limit":
                return False
            if node.attr in _CHAINED_ACTION_METHODS or node.attr in _WRITE_CHAIN_METHODS:
                reaches_consumer = True
        return reaches_consumer

    @staticmethod
    def _root_name(node: ast.AST) -> str | None:
        """Resolve the root variable name of an attribute/call chain.

        For ``s.filter(...).show`` the root name is ``s``.

        Args:
            node: AST node at any point in the chain.

        Returns:
            The root variable name, or None if the chain has no simple root.

        """
        current = node
        while True:
            if isinstance(current, ast.Attribute):
                current = current.value
            elif isinstance(current, ast.Call):
                current = current.func
            elif isinstance(current, ast.Name):
                return current.id
            else:
                return None

    @staticmethod
    def _is_count_call(node: ast.AST) -> bool:
        """Check whether a node is a .count() call with no arguments.

        Args:
            node: AST node.

        Returns:
            True if the node is an argument-less .count() method call.

        """
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "count"
            and not node.args
        )

    @staticmethod
    def _is_int_constant(node: ast.AST, value: int) -> bool:
        """Check whether a node is a specific integer constant.

        Args:
            node: AST node.
            value: Expected integer value.

        Returns:
            True if the node is the given int literal (booleans excluded).

        """
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
            and node.value == value
        )

    @staticmethod
    def _is_truthy_literal(node: ast.AST) -> bool:
        """Check whether a node is a True-ish literal (True or 'true').

        Args:
            node: AST node.

        Returns:
            True if the node is the constant True or a 'true' string.

        """
        if not isinstance(node, ast.Constant):
            return False
        if node.value is True:
            return True
        return isinstance(node.value, str) and node.value.lower() == "true"

    @staticmethod
    def _extract_string_literal(argument: str) -> str | None:
        """Recover a plain string literal from a repr-style argument.

        The parser stores arguments as their repr; plain string literals
        therefore arrive quoted. Anything else (f-strings, variables,
        concatenations) is skipped by returning None.

        Args:
            argument: Argument representation produced by the parser.

        Returns:
            The literal string value, or None if not a plain string literal.

        """
        text = argument.strip()
        if not text.startswith(("'", '"')):
            return None
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        return value if isinstance(value, str) else None


def detect_smells(source_code: str) -> AnalysisResult:
    """Convenience function to detect smells in Spark code.

    Args:
        source_code: Python source code string.

    Returns:
        AnalysisResult containing detected smells.

    Example:
        >>> code = '''
        ... df1 = spark.read.parquet("large_table.parquet")
        ... df2 = spark.read.parquet("small_table.parquet")
        ... result = df1.join(df2, "id")  # Missing broadcast hint
        ... result.show()
        ... '''
        >>> result = detect_smells(code)
        >>> for smell in result.smells:
        ...     print(f"{smell.smell_type}: {smell.description}")

    """
    detector = SmellDetector()
    return detector.analyze_source(source_code)
