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
            "pandas_udf_usage": self._gen_pandas_udf_recommendation,
            "data_skew_potential": self._gen_skew_recommendation,
            "repartition_vs_coalesce": self._gen_coalesce_recommendation,
            "small_file_problem": self._gen_partitioning_recommendation,
            "serialization_issue": self._gen_serialization_recommendation,
            "large_collect": self._gen_large_collect_recommendation,
            "cartesian_join": self._gen_cartesian_join_recommendation,
            "topandas_usage": self._gen_topandas_recommendation,
            "count_for_empty_check": self._gen_empty_check_recommendation,
            "single_partition_write": self._gen_single_partition_write_recommendation,
            "infer_schema": self._gen_infer_schema_recommendation,
            "withcolumn_in_loop": self._gen_withcolumn_loop_recommendation,
            "select_star": self._gen_select_star_recommendation,
            "orderby_without_limit": self._gen_orderby_limit_recommendation,
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
            before_code=f'# {op_type} on skewed column\ndf1.join(df2, "user_id")',
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

    def _gen_pandas_udf_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for pandas_udf usage.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Replace pandas_udf with built-in functions when an equivalent exists",
            before_code="from pyspark.sql.functions import pandas_udf\n"
            "@pandas_udf('double')\n"
            "def doubled(s):\n"
            "    return s * 2\n"
            "df.withColumn('doubled', doubled(df.value))",
            after_code="# Built-in expressions run inside the JVM and are fully optimized\n"
            "from pyspark.sql.functions import col\n"
            "df.withColumn('doubled', col('value') * 2)",
            explanation=(
                "Pandas UDFs are vectorized via Arrow and much faster than regular "
                "Python UDFs, but they still transfer data out of the JVM and hide "
                "logic from the Catalyst optimizer. When the same computation can be "
                "expressed with built-in Spark SQL functions, prefer those; keep "
                "pandas_udf only for logic with no built-in equivalent."
            ),
            effort="medium",
        )

    def _gen_large_collect_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for collect() without limit().

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Bound collect() with limit() or keep the data distributed",
            before_code="rows = df.collect()  # Pulls the entire dataset to the driver",
            after_code="# Option 1: Bound the result\n"
            "rows = df.limit(1000).collect()\n\n"
            "# Option 2: Keep processing distributed\n"
            'df.write.parquet("output/")',
            explanation=(
                "collect() materializes every row in driver memory, which crashes "
                "the driver on large datasets. Bound it with limit(N) when you only "
                "need a sample, or write results to storage and keep processing "
                "distributed."
            ),
            effort="low",
        )

    def _gen_cartesian_join_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for cartesian/cross joins.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Replace the cartesian product with a keyed join (or broadcast a tiny side)",
            before_code="result = df1.crossJoin(df2)  # |df1| x |df2| rows",
            after_code="# Option 1: Join on a key whenever one exists\n"
            'result = df1.join(df2, "id")\n\n'
            "# Option 2: If the cross join is intentional and one side is tiny\n"
            "from pyspark.sql.functions import broadcast\n"
            "result = df1.crossJoin(broadcast(tiny_df))",
            explanation=(
                "A cartesian product multiplies the row counts of both inputs: even "
                "two modest 100k-row tables explode into 10 billion rows. Most cross "
                "joins are accidental and should be keyed joins. When a cross join is "
                "genuinely needed (e.g. against a handful of constants), broadcast "
                "the tiny side to avoid shuffling the large one."
            ),
            effort="medium",
        )

    def _gen_topandas_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for toPandas() usage.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Limit or sample before toPandas() and enable Arrow",
            before_code="pdf = df.toPandas()  # Entire dataset into driver memory",
            after_code="# Option 1: Bound the data first\n"
            "pdf = df.limit(100_000).toPandas()\n\n"
            "# Option 2: Sample a fraction\n"
            "pdf = df.sample(fraction=0.01).toPandas()\n\n"
            "# Always enable Arrow for the transfer\n"
            'spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")',
            explanation=(
                "toPandas() collects the full DataFrame into the driver as a pandas "
                "object, which easily exhausts driver memory. Reduce the volume with "
                "limit() or sample() first, and enable Arrow so the transfer is "
                "columnar and vectorized instead of row-by-row pickling."
            ),
            effort="low",
        )

    def _gen_empty_check_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for count()-based emptiness checks.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Use isEmpty() or take(1) instead of count() for emptiness checks",
            before_code="if df.count() == 0:\n    handle_empty()",
            after_code="# Spark 3.3+\n"
            "if df.isEmpty():\n"
            "    handle_empty()\n\n"
            "# Older Spark versions\n"
            "if len(df.take(1)) == 0:\n"
            "    handle_empty()",
            explanation=(
                "count() computes the exact number of rows, scanning every partition "
                "of the dataset just to answer a yes/no question. isEmpty() (Spark "
                "3.3+) and take(1) short-circuit after finding the first row, turning "
                "a full scan into a near-instant check."
            ),
            effort="low",
        )

    def _gen_single_partition_write_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for single-partition writes.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Avoid coalesce(1)/repartition(1) before writes; let Spark write in parallel",
            before_code='df.coalesce(1).write.parquet("output/")  # Single-task write',
            after_code="# Option 1: Parallel write (preferred)\n"
            'df.write.parquet("output/")\n\n'
            "# Option 2: Fewer, larger files without a single-task bottleneck\n"
            'df.coalesce(8).write.parquet("output/")\n\n'
            "# Option 3: If exactly one file is required, compact afterwards\n"
            "# (e.g. a downstream job or filesystem-level merge)",
            explanation=(
                "coalesce(1) or repartition(1) before a write funnels the entire "
                "output through a single task on one executor core, serializing the "
                "write and risking OOM. Write in parallel and, if a single file is a "
                "hard requirement, produce it in a cheap post-processing step instead."
            ),
            effort="low",
        )

    def _gen_infer_schema_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for schema inference in reads.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Provide an explicit schema instead of inferSchema=True",
            before_code='df = spark.read.csv("data.csv", header=True, inferSchema=True)',
            after_code="from pyspark.sql.types import StructType, StructField, StringType, IntegerType\n\n"
            "schema = StructType([\n"
            '    StructField("id", IntegerType(), False),\n'
            '    StructField("name", StringType(), True),\n'
            "])\n"
            'df = spark.read.csv("data.csv", header=True, schema=schema)',
            explanation=(
                "inferSchema=True makes Spark read the input twice: once to sample "
                "and infer types, once for the actual load. On large inputs this "
                "doubles I/O, and inferred types can drift between runs. An explicit "
                "StructType schema skips the inference pass and guarantees stable, "
                "correct types."
            ),
            effort="medium",
        )

    def _gen_withcolumn_loop_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for withColumn() inside loops.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Replace per-iteration withColumn() with a single select()",
            before_code="for c in columns:\n    df = df.withColumn(c + '_clean', trim(col(c)))",
            after_code="# Build all expressions first, then project once\n"
            "exprs = [trim(col(c)).alias(c + '_clean') for c in columns]\n"
            "df = df.select('*', *exprs)\n\n"
            "# Spark 3.3+: withColumns accepts a mapping in one call\n"
            "df = df.withColumns({c + '_clean': trim(col(c)) for c in columns})",
            explanation=(
                "Each withColumn() call adds a new projection node to the logical "
                "plan. Inside a loop this grows the plan linearly, and Catalyst "
                "analysis cost grows much faster — long loops cause minutes of "
                "planning time or StackOverflowError. Building the expressions in "
                "Python and projecting once keeps the plan flat."
            ),
            effort="medium",
        )

    def _gen_select_star_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for select('*') / SELECT * usage.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Select only the columns you actually need",
            before_code='df.select("*")  # or spark.sql("SELECT * FROM events")',
            after_code='df.select("user_id", "event_time", "amount")\n'
            '# or: spark.sql("SELECT user_id, event_time, amount FROM events")',
            explanation=(
                "Selecting every column disables column pruning: Spark must read, "
                "deserialize, and shuffle data you never use. For columnar formats "
                "like Parquet/ORC, naming only the needed columns lets Spark skip "
                "entire column chunks on disk, often cutting I/O dramatically."
            ),
            effort="low",
        )

    def _gen_orderby_limit_recommendation(self, smell: CodeSmell) -> CodeRecommendation:
        """Generate recommendation for orderBy without limit.

        Args:
            smell: The detected code smell.

        Returns:
            Code recommendation with fix.

        """
        return CodeRecommendation(
            smell=smell,
            suggestion="Add limit() for top-N queries or avoid the global sort",
            before_code='df.orderBy(col("amount").desc()).show()  # Full global sort',
            after_code="# Option 1: Top-N — Spark optimizes orderBy+limit into TakeOrdered\n"
            'df.orderBy(col("amount").desc()).limit(100).show()\n\n'
            "# Option 2: Order within partitions when global order is not needed\n"
            'df.sortWithinPartitions("amount")',
            explanation=(
                "orderBy()/sort() performs a global sort: the whole dataset is "
                "range-partitioned and shuffled. When followed by limit(N), Spark "
                "rewrites the query into a cheap TakeOrdered operation that never "
                "sorts the full data. If you only need ordering inside each output "
                "file or partition, sortWithinPartitions() avoids the shuffle "
                "entirely."
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
