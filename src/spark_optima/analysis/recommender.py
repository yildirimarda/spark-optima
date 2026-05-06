# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Recommendation engine for Spark code improvements.

This module provides the RecommendationEngine class that generates
actionable code improvement suggestions based on detected code smells.
"""

from __future__ import annotations

import logging
from typing import Any

from spark_optima.analysis.models import (
    AnalysisResult,
    CodeRecommendation,
    CodeSmell,
    SeverityLevel,
)
from spark_optima.analysis.smell_detector import SmellDetector

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Engine for generating code improvement recommendations.

    This class analyzes detected code smells and generates specific,
    actionable recommendations with before/after code examples.

    Attributes:
        detector: SmellDetector instance for code analysis.

    Example:
        >>> engine = RecommendationEngine()
        >>> result = engine.analyze("spark_job.py")
        >>> for rec in result.recommendations:
        ...     print(f"{rec.suggestion}")
        ...     print(f"Before: {rec.before_code}")
        ...     print(f"After: {rec.after_code}")

    """

    # Mapping of smell types to recommendation generators
    _recommendation_generators: dict[str, Any] = {}

    def __init__(self) -> None:
        """Initialize the recommendation engine."""
        self.detector = SmellDetector()
        self._register_generators()

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyze a file and generate recommendations.

        Args:
            file_path: Path to the Python file.

        Returns:
            AnalysisResult with smells and recommendations.

        """
        analysis_result = self.detector.analyze_file(file_path)
        analysis_result.recommendations = self._generate_recommendations(analysis_result.smells)
        return analysis_result

    def analyze_source(self, source_code: str) -> AnalysisResult:
        """Analyze source code and generate recommendations.

        Args:
            source_code: Python source code string.

        Returns:
            AnalysisResult with smells and recommendations.

        """
        analysis_result = self.detector.analyze_source(source_code)
        analysis_result.recommendations = self._generate_recommendations(analysis_result.smells)
        return analysis_result

    def _register_generators(self) -> None:
        """Register recommendation generators for each smell type."""
        self._recommendation_generators = {
            "missing_broadcast_hint": self._gen_broadcast_recommendation,
            "unnecessary_shuffle": self._gen_shuffle_recommendation,
            "caching_issue": self._gen_caching_recommendation,
            "udf_usage": self._gen_udf_recommendation,
            "data_skew_potential": self._gen_skew_recommendation,
            "repartition_vs_coalesce": self._gen_coalesce_recommendation,
            "small_file_problem": self._gen_partitioning_recommendation,
            "serialization_issue": self._gen_serialization_recommendation,
        }

    def _generate_recommendations(
        self,
        smells: list[CodeSmell],
    ) -> list[CodeRecommendation]:
        """Generate recommendations from detected smells.

        Args:
            smells: List of detected code smells.

        Returns:
            List of code recommendations.

        """
        recommendations = []

        for smell in smells:
            generator = self._recommendation_generators.get(smell.smell_type)
            if generator:
                try:
                    rec = generator(smell)
                    if rec:
                        recommendations.append(rec)
                except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"Failed to generate recommendation for {smell.smell_type}: {e}")
            else:
                # Generic recommendation for unknown smell types
                rec = self._gen_generic_recommendation(smell)
                recommendations.append(rec)

        return recommendations

    def _gen_broadcast_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for missing broadcast hint.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        op = smell.affected_operation
        if not op:
            return self._gen_generic_recommendation(smell)

        df_var = op.dataframe_var
        args = op.arguments
        args_str = ", ".join(args) if args else ""

        return CodeRecommendation(
            smell=smell,
            suggestion="Add broadcast hint to the smaller DataFrame in the join",
            before_code=f"{df_var}.join({args_str})",
            after_code=f"{df_var}.join(broadcast(small_df), {args_str})",
            explanation=(
                "Broadcasting sends a copy of the entire small DataFrame to all "
                "executor nodes, eliminating the need to shuffle the large DataFrame. "
                "This significantly speeds up joins where one table is much smaller "
                "than the other. Use broadcast() from pyspark.sql.functions."
            ),
            effort="low",
        )

    def _gen_shuffle_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for unnecessary shuffle.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        op = smell.affected_operation
        if not op:
            return self._gen_generic_recommendation(smell)

        method = op.method_name

        if method == "repartition":
            return CodeRecommendation(
                smell=smell,
                suggestion="Remove redundant repartition or combine with other operations",
                before_code="df.repartition(100).repartition(50)",
                after_code="df.repartition(50)  # Single repartition with final partition count",
                explanation=(
                    "Multiple repartitions cause multiple full shuffles of data. "
                    "Each shuffle is expensive as it requires network I/O and disk "
                    "serialization. Always use a single repartition with the target "
                    "number of partitions."
                ),
                effort="low",
            )
        else:
            return CodeRecommendation(
                smell=smell,
                suggestion="Remove repartition before sort/orderBy operations",
                before_code="df.repartition(100).orderBy('col')",
                after_code="df.orderBy('col')  # Sort already shuffles data",
                explanation=(
                    "orderBy and sort operations already trigger a shuffle to "
                    "redistribute data for sorting. Adding a repartition before "
                    "them causes an unnecessary additional shuffle."
                ),
                effort="low",
            )

    def _gen_caching_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for caching issues.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        description = smell.description.lower()

        if "only" in description and "time" in description:
            # Cache used only once
            return CodeRecommendation(
                smell=smell,
                suggestion="Remove unnecessary cache() call",
                before_code="df.cache().show()  # Used only once",
                after_code="df.show()  # Remove cache for single use",
                explanation=(
                    "Caching has overhead: it serializes data to memory/disk and "
                    "tracks lineage. If the DataFrame is used only once, caching "
                    "adds this overhead without any benefit. Remove cache() calls "
                    "for DataFrames that are not reused."
                ),
                effort="low",
            )
        else:
            # Missing cache on reused DataFrame
            return CodeRecommendation(
                smell=smell,
                suggestion="Add cache() to frequently used DataFrame",
                before_code="df = expensive_transformations(source)\n"
                "df.write.parquet('out1')\n"
                "df.write.parquet('out2')",
                after_code="df = expensive_transformations(source).cache()\n"
                "df.write.parquet('out1')\n"
                "df.write.parquet('out2')\n"
                "df.unpersist()  # Clean up when done",
                explanation=(
                    "When a DataFrame is used multiple times, transformations are "
                    "recomputed for each action. Caching stores the result in memory "
                    "(or disk) after the first computation, avoiding redundant work. "
                    "Remember to unpersist() when done to free memory."
                ),
                effort="low",
            )

    def _gen_udf_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for UDF usage.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Replace Python UDF with built-in functions or Pandas UDF",
            before_code="from pyspark.sql.functions import udf\n"
            "@udf\n"
            "def my_func(col):\n"
            "    return col * 2\n"
            "df.withColumn('doubled', my_func(df.value))",
            after_code="# Option 1: Use built-in functions\n"
            "from pyspark.sql.functions import col\n"
            "df.withColumn('doubled', col('value') * 2)\n\n"
            "# Option 2: Use Pandas UDF for complex logic\n"
            "from pyspark.sql.functions import pandas_udf\n"
            "@pandas_udf('double')\n"
            "def my_func(col):\n"
            "    return col * 2",
            explanation=(
                "Regular Python UDFs run in separate Python processes, requiring "
                "data serialization between JVM and Python. This prevents Catalyst "
                "optimizer from optimizing the query. Built-in functions run in JVM "
                "and are fully optimized. Pandas UDFs use Arrow for efficient "
                "data transfer and are much faster than regular UDFs."
            ),
            effort="medium",
        )

    def _gen_skew_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for data skew potential.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        op = smell.affected_operation
        if not op:
            return self._gen_generic_recommendation(smell)

        op_type = op.operation_type.name.lower()

        return CodeRecommendation(
            smell=smell,
            suggestion="Use salting technique or enable Adaptive Query Execution",
            before_code=f"# {op_type} on skewed column\n" f'df1.join(df2, "user_id")',
            after_code="# Option 1: Enable AQE (Spark 3.0+)\n"
            "spark.conf.set('spark.sql.adaptive.enabled', 'true')\n"
            "spark.conf.set('spark.sql.adaptive.skewJoin.enabled', 'true')\n\n"
            "# Option 2: Manual salting\n"
            "salt_count = 10\n"
            'df1_salted = df1.withColumn("salt", (rand() * salt_count).cast("int"))\n'
            'df2_salted = df2.withColumn("salt", '
            "explode(array([lit(i) for i in range(salt_count)])))\n"
            'df1_salted.join(df2_salted, ["user_id", "salt"])',
            explanation=(
                "Data skew occurs when some keys have significantly more records "
                "than others, causing some partitions to be much larger. Adaptive "
                "Query Execution (AQE) automatically detects and handles skew. "
                "Manual salting duplicates skewed keys with random salts to "
                "distribute the load evenly across partitions."
            ),
            effort="high",
        )

    def _gen_coalesce_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for repartition vs coalesce.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Use coalesce() instead of repartition() when reducing partitions",
            before_code="df.repartition(4)  # Reduces partitions but shuffles",
            after_code="df.coalesce(4)  # Reduces without shuffling",
            explanation=(
                "repartition() always triggers a full shuffle of data across the "
                "network, even when reducing the number of partitions. coalesce() "
                "is optimized for reducing partitions: it minimizes data movement "
                "by combining existing partitions without shuffling. Use coalesce() "
                "when writing to few files; use repartition() only when you need "
                "to increase partitions or require data to be evenly distributed."
            ),
            effort="low",
        )

    def _gen_partitioning_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for small file problems.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Add partitioning to write operations for large datasets",
            before_code='df.write.parquet("output/")  # Many small files',
            after_code="# Partition by date for time-series data\n"
            'df.write.partitionBy("date").parquet("output/")\n\n'
            "# Or use bucketing for joins\n"
            'df.write.bucketBy(100, "user_id").saveAsTable("bucketed_table")',
            explanation=(
                "Writing large datasets without partitioning creates many small "
                "files (often one per partition), which hurts read performance due "
                "to metadata overhead. Partitioning organizes data into directory "
                "hierarchies, enabling partition pruning during reads. Bucketing "
                "is ideal for optimizing join performance on specific columns."
            ),
            effort="medium",
        )

    def _gen_serialization_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for serialization issues.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Enable Kryo serialization for better performance",
            before_code="# Using default Java serialization (slower)",
            after_code="# In SparkSession configuration:\n"
            "SparkSession.builder\n"
            "    .appName('MyApp')\n"
            "    .config('spark.serializer', "
            "'org.apache.spark.serializer.KryoSerializer')\n"
            "    .config('spark.kryo.registrator', 'my.package.MyRegistrator')\n"
            "    .getOrCreate()",
            explanation=(
                "Kryo serialization is significantly faster and more compact than "
                "Java serialization. It reduces both CPU time and memory/network "
                "usage during shuffles and caching. For custom classes, register "
                "them with a KryoRegistrator for best performance."
            ),
            effort="low",
        )

    def _gen_generic_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate a generic recommendation for unknown smell types.

        Args:
            smell: The detected code smell.

        Returns:
            Generic code recommendation.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion=f"Address {smell.smell_type} issue",
            before_code="# See original code at detected location",
            after_code="# Implement recommended fix",
            explanation=smell.impact or smell.description,
            effort="medium",
        )

    def get_priority_recommendations(
        self,
        analysis_result: AnalysisResult,
        max_recommendations: int = 10,
    ) -> list[CodeRecommendation]:
        """Get recommendations sorted by priority.

        Args:
            analysis_result: Analysis result with recommendations.
            max_recommendations: Maximum number of recommendations to return.

        Returns:
            List of priority-sorted recommendations.

        """

        # Sort by severity and effort
        def priority_score(rec: CodeRecommendation) -> tuple[int, int]:
            severity_order = {
                SeverityLevel.CRITICAL: 0,
                SeverityLevel.HIGH: 1,
                SeverityLevel.MEDIUM: 2,
                SeverityLevel.LOW: 3,
            }
            effort_order = {"low": 0, "medium": 1, "high": 2}

            sev_score = severity_order.get(rec.smell.severity, 4)
            eff_score = effort_order.get(rec.effort, 1)

            return (sev_score, eff_score)

        sorted_recs = sorted(analysis_result.recommendations, key=priority_score)
        return sorted_recs[:max_recommendations]


def analyze_code(source_code: str) -> AnalysisResult:
    """Convenience function to analyze code and get recommendations.

    Args:
        source_code: Python source code string.

    Returns:
        AnalysisResult with smells and recommendations.

    Example:
        >>> code = '''
        ... df1 = spark.read.parquet("large.parquet")
        ... df2 = spark.read.parquet("small.parquet")
        ... result = df1.join(df2, "id")
        ... result.show()
        ... '''
        >>> result = analyze_code(code)
        >>> for rec in result.get_high_priority_recommendations():
        ...     print(f"{rec.suggestion}")

    """
    engine = RecommendationEngine()
    return engine.analyze_source(source_code)
