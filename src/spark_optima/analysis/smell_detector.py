# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Code smell detection for Spark applications.

This module provides the SmellDetector class that analyzes parsed Spark code
to detect various performance anti-patterns and code smells.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from spark_optima.analysis.models import (
    AnalysisResult,
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
        smells = []
        joins = parse_result.operations
        joins = [op for op in joins if op.operation_type == SparkOperationType.JOIN]

        for join_op in joins:
            # Check if broadcast hint is already present
            has_broadcast = any(
                "broadcast" in arg.lower() or "hint" in arg.lower() for arg in join_op.arguments
            )

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
        smells = []
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
                    if (
                        op.chain_position - prev_op.chain_position <= 2
                        and prev_op.method_name == "repartition"
                    ):
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
        smells = []
        cache_ops = [op for op in parse_result.operations if op.method_name in ["cache", "persist"]]

        # Check for cache without subsequent action
        for cache_op in cache_ops:
            # Find subsequent operations on same DataFrame
            subsequent_ops = [
                op
                for op in parse_result.operations
                if (
                    op.chain_position > cache_op.chain_position
                    and op.dataframe_var == cache_op.dataframe_var
                )
            ]

            # Check if cache is used multiple times
            action_count = sum(
                1 for op in subsequent_ops if op.operation_type == SparkOperationType.ACTION
            )

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
                        "Unnecessary caching wastes memory and adds serialization "
                        "overhead without providing benefits."
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

        Args:
            parse_result: Parse result to analyze.

        Returns:
            List of detected smells.

        """
        smells = []
        udf_ops = [
            op for op in parse_result.operations if op.operation_type == SparkOperationType.UDF
        ]

        for udf_op in udf_ops:
            smell = CodeSmell(
                smell_type="udf_usage",
                description=(
                    f"UDF usage detected at line "
                    f"{udf_op.location.line if udf_op.location else 'unknown'}. "
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
        smells = []

        # Check joins and aggregations
        skew_sensitive_ops = [
            op
            for op in parse_result.operations
            if op.operation_type in [SparkOperationType.JOIN, SparkOperationType.AGGREGATION]
        ]

        for op in skew_sensitive_ops:
            # Check if arguments suggest potential skew
            args_str = " ".join(op.arguments).lower()

            # Common high-cardinality columns that often cause skew
            skew_indicators = [
                "user_id",
                "id",
                "key",
                "customer_id",
                "null",
                "none",
                "",
                "0",
            ]

            has_skew_indicator = any(indicator in args_str for indicator in skew_indicators)

            if has_skew_indicator or not op.arguments:
                # No explicit hint for handling skew
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
        smells = []
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
        smells = []

        # Check for write operations without partitioning
        write_ops = [
            op for op in parse_result.operations if op.operation_type == SparkOperationType.WRITE
        ]

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
        smells = []

        # Check for operations that benefit from Kryo serialization
        serialization_heavy_ops = [
            op
            for op in parse_result.operations
            if op.method_name in ["map", "flatMap", "mapPartitions", "filter"]
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
