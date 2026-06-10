# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the recommendation engine."""

import pytest

from spark_optima.analysis.models import (
    AnalysisResult,
    CodeRecommendation,
    CodeSmell,
    SeverityLevel,
    SparkOperation,
    SparkOperationType,
)
from spark_optima.analysis.recommender import (
    RecommendationEngine,
    analyze_code,
)


class TestRecommendationEngine:
    """Test cases for RecommendationEngine."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = RecommendationEngine()
        assert engine.detector is not None

    def test_analyze_source(self):
        """Test analyzing source code."""
        code = """
df1 = spark.read.parquet("large.parquet")
df2 = spark.read.parquet("small.parquet")
result = df1.join(df2, "id")
result.show()
"""
        engine = RecommendationEngine()
        result = engine.analyze_source(code)

        assert isinstance(result.operations, list)
        assert isinstance(result.smells, list)
        assert isinstance(result.recommendations, list)

    def test_recommendation_generation(self):
        """Test that recommendations are generated."""
        code = """
df1 = spark.read.parquet("large.parquet")
df2 = spark.read.parquet("small.parquet")
result = df1.join(df2, "id")
result.show()
"""
        engine = RecommendationEngine()
        result = engine.analyze_source(code)

        # Should have recommendations for detected smells
        if result.smells:
            assert len(result.recommendations) > 0

    def test_broadcast_recommendation(self):
        """Test broadcast hint recommendation."""
        code = """
df1 = spark.read.parquet("large.parquet")
df2 = spark.read.parquet("small.parquet")
result = df1.join(df2, "id")
result.show()
"""
        engine = RecommendationEngine()
        result = engine.analyze_source(code)

        # Check for broadcast recommendation
        broadcast_recs = [r for r in result.recommendations if "broadcast" in r.suggestion.lower()]
        if broadcast_recs:
            assert broadcast_recs[0].before_code
            assert broadcast_recs[0].after_code
            assert broadcast_recs[0].explanation

    def test_priority_recommendations(self):
        """Test getting priority recommendations."""
        code = """
df = spark.read.parquet("data.parquet")
df.cache()
df.show()
"""
        engine = RecommendationEngine()
        result = engine.analyze_source(code)

        priority_recs = engine.get_priority_recommendations(result)
        # Should return recommendations sorted by priority
        assert isinstance(priority_recs, list)

    def test_recommendation_effort_levels(self):
        """Test that recommendations have valid effort levels."""
        code = """
df1 = spark.read.parquet("large.parquet")
df2 = spark.read.parquet("small.parquet")
result = df1.join(df2, "id")
result.show()
"""
        engine = RecommendationEngine()
        result = engine.analyze_source(code)

        valid_efforts = {"low", "medium", "high"}
        for rec in result.recommendations:
            assert rec.effort in valid_efforts


class TestAnalyzeCodeFunction:
    """Test cases for the analyze_code convenience function."""

    def test_analyze_code(self):
        """Test the analyze_code function."""
        code = """
df = spark.read.parquet("data.parquet")
df.filter(df.x > 0)
"""
        result = analyze_code(code)
        assert isinstance(result.operations, list)
        assert isinstance(result.smells, list)
        assert isinstance(result.recommendations, list)


class TestRecommendationEngineEdgeCases:
    """Edge case tests for RecommendationEngine."""

    def test_analyze_file(self, tmp_path):
        """Test analyze_file method."""
        # Create a test file
        test_file = tmp_path / "test_code.py"
        test_file.write_text(
            """
df = spark.read.parquet("data.parquet")
df.filter(df.x > 0)
df.show()
"""
        )
        engine = RecommendationEngine()
        result = engine.analyze_file(str(test_file))
        assert isinstance(result.operations, list)
        assert isinstance(result.smells, list)

    def test_analyze_file_not_found(self):
        """Test analyze_file with non-existent file."""
        engine = RecommendationEngine()
        with pytest.raises(FileNotFoundError):
            engine.analyze_file("/non/existent/file.py")

    def test_broadcast_recommendation_no_op(self):
        """Test broadcast recommendation when operation is None."""
        engine = RecommendationEngine()
        smell = CodeSmell(
            smell_type="missing_broadcast_hint",
            description="Test",
            severity=SeverityLevel.MEDIUM,
        )
        # Create a smell without affected_operation
        smell.affected_operation = None
        rec = engine._gen_broadcast_recommendation(smell)
        assert rec is not None
        assert "broadcast" in rec.suggestion.lower() or "Address" in rec.suggestion

    def test_shuffle_recommendation_no_op(self):
        """Test shuffle recommendation when operation is None."""
        engine = RecommendationEngine()
        smell = CodeSmell(
            smell_type="unnecessary_shuffle",
            description="Test",
            severity=SeverityLevel.HIGH,
        )
        smell.affected_operation = None
        rec = engine._gen_shuffle_recommendation(smell)
        assert rec is not None

    def test_shuffle_recommendation_sort_method(self):
        """Test shuffle recommendation for sort method."""
        engine = RecommendationEngine()
        smell = CodeSmell(
            smell_type="unnecessary_shuffle",
            description="Test",
            severity=SeverityLevel.MEDIUM,
        )
        # Create an operation with sort method
        op = SparkOperation(
            operation_type=SparkOperationType.TRANSFORMATION,
            method_name="sort",
            dataframe_var="df",
        )
        smell.affected_operation = op
        rec = engine._gen_shuffle_recommendation(smell)
        assert rec is not None
        assert "repartition" in rec.before_code.lower() or "sort" in rec.before_code.lower()

    def test_caching_recommendation_missing_cache(self):
        """Test caching recommendation for missing cache case."""
        engine = RecommendationEngine()
        smell = CodeSmell(
            smell_type="caching_issue",
            description="DataFrame used multiple times but not cached",
            severity=SeverityLevel.LOW,
        )
        # Create an operation
        op = SparkOperation(
            operation_type=SparkOperationType.READ,
            method_name="read",
            dataframe_var="df",
        )
        smell.affected_operation = op
        rec = engine._gen_caching_recommendation(smell)
        assert rec is not None
        assert "cache" in rec.suggestion.lower()

    def test_udf_recommendation(self):
        """Test UDF recommendation generation."""
        engine = RecommendationEngine()
        smell = CodeSmell(
            smell_type="udf_usage",
            description="UDF usage detected",
            severity=SeverityLevel.HIGH,
        )
        op = SparkOperation(
            operation_type=SparkOperationType.UDF,
            method_name="udf",
            dataframe_var="df",
        )
        smell.affected_operation = op
        rec = engine._gen_udf_recommendation(smell)
        assert rec is not None
        assert "UDF" in rec.suggestion or "udf" in rec.suggestion.lower()

    def test_skew_recommendation_no_op(self):
        """Test skew recommendation when operation is None."""
        engine = RecommendationEngine()
        smell = CodeSmell(
            smell_type="data_skew_potential",
            description="Potential data skew",
            severity=SeverityLevel.MEDIUM,
        )
        smell.affected_operation = None
        rec = engine._gen_skew_recommendation(smell)
        assert rec is not None
        assert "skew" in rec.suggestion.lower() or "Address" in rec.suggestion

    def test_skew_recommendation_with_op(self):
        """Test skew recommendation with operation."""
        engine = RecommendationEngine()
        smell = CodeSmell(
            smell_type="data_skew_potential",
            description="Potential data skew in join",
            severity=SeverityLevel.MEDIUM,
        )
        op = SparkOperation(
            operation_type=SparkOperationType.JOIN,
            method_name="join",
            dataframe_var="df1",
        )
        smell.affected_operation = op
        rec = engine._gen_skew_recommendation(smell)
        assert rec is not None
        assert "salt" in rec.after_code.lower() or "aqe" in rec.after_code.lower()

    def test_coalesce_recommendation(self):
        """Test coalesce recommendation generation."""
        engine = RecommendationEngine()
        smell = CodeSmell(
            smell_type="repartition_vs_coalesce",
            description="Should use coalesce",
            severity=SeverityLevel.MEDIUM,
        )
        rec = engine._gen_coalesce_recommendation(smell)
        assert rec is not None
        assert "coalesce" in rec.suggestion.lower()

    def test_partitioning_recommendation(self):
        """Test partitioning recommendation generation."""
        engine = RecommendationEngine()
        smell = CodeSmell(
            smell_type="small_file_problem",
            description="Small file problem",
            severity=SeverityLevel.LOW,
        )
        rec = engine._gen_partitioning_recommendation(smell)
        assert rec is not None
        assert "partition" in rec.suggestion.lower() or "Partition" in rec.suggestion

    def test_serialization_recommendation(self):
        """Test serialization recommendation generation."""
        engine = RecommendationEngine()
        smell = CodeSmell(
            smell_type="serialization_issue",
            description="Serialization issue",
            severity=SeverityLevel.LOW,
        )
        rec = engine._gen_serialization_recommendation(smell)
        assert rec is not None
        assert "kryo" in rec.suggestion.lower() or "Kryo" in rec.suggestion

    def test_generic_recommendation(self):
        """Test generic recommendation generation."""
        engine = RecommendationEngine()
        smell = CodeSmell(
            smell_type="unknown_smell",
            description="Unknown issue",
            severity=SeverityLevel.MEDIUM,
        )
        rec = engine._gen_generic_recommendation(smell)
        assert rec is not None
        assert "Address" in rec.suggestion
        assert rec.effort == "medium"

    def test_generate_recommendations_exception(self):
        """Test exception handling in _generate_recommendations."""
        engine = RecommendationEngine()
        # Create a smell that will cause an exception
        smell = CodeSmell(
            smell_type="test_smell",
            description="Test",
            severity=SeverityLevel.LOW,
        )
        # The generator lookup will return None for unknown type
        # and it will use generic recommendation
        result = engine._generate_recommendations([smell])
        assert len(result) == 1

    def test_generate_recommendations_with_exception_in_generator(self):
        """Test exception handling when generator raises an exception."""
        engine = RecommendationEngine()

        # Create a generator function that raises a specific exception
        def failing_generator(smell):
            raise RuntimeError("Test exception")

        engine._recommendation_generators["test_smell"] = failing_generator
        smell = CodeSmell(
            smell_type="test_smell",
            description="Test",
            severity=SeverityLevel.LOW,
        )
        # Should not raise, should log warning and continue
        result = engine._generate_recommendations([smell])
        assert len(result) == 0

    def test_shuffle_recommendation_repartition_method(self):
        """Test shuffle recommendation for repartition method."""
        engine = RecommendationEngine()
        smell = CodeSmell(
            smell_type="unnecessary_shuffle",
            description="Test",
            severity=SeverityLevel.HIGH,
        )
        # Create an operation with repartition method
        op = SparkOperation(
            operation_type=SparkOperationType.REPARTITION,
            method_name="repartition",
            dataframe_var="df",
        )
        smell.affected_operation = op
        rec = engine._gen_shuffle_recommendation(smell)
        assert rec is not None
        assert "repartition" in rec.before_code.lower()

    def test_get_priority_recommendations_sorted(self):
        """Test that priority recommendations are properly sorted."""
        engine = RecommendationEngine()
        smells = [
            CodeSmell("s1", "desc1", SeverityLevel.LOW),
            CodeSmell("s2", "desc2", SeverityLevel.CRITICAL),
            CodeSmell("s3", "desc3", SeverityLevel.HIGH),
            CodeSmell("s4", "desc4", SeverityLevel.MEDIUM),
        ]
        recs = [
            CodeRecommendation(smell=smells[0], suggestion="low", effort="low"),
            CodeRecommendation(smell=smells[1], suggestion="critical", effort="high"),
            CodeRecommendation(smell=smells[2], suggestion="high", effort="medium"),
            CodeRecommendation(smell=smells[3], suggestion="medium", effort="low"),
        ]
        result = AnalysisResult(recommendations=recs)
        priority = engine.get_priority_recommendations(result, max_recommendations=10)
        # Should be sorted by severity (critical first)
        assert priority[0].smell.severity == SeverityLevel.CRITICAL
        assert priority[-1].smell.severity == SeverityLevel.LOW

    def test_get_priority_recommendations_max(self):
        """Test priority recommendations with max limit."""
        engine = RecommendationEngine()
        smells = [CodeSmell(f"s{i}", f"desc{i}", SeverityLevel.LOW) for i in range(5)]
        recs = [CodeRecommendation(smell=s, suggestion=f"rec{i}") for i, s in enumerate(smells)]
        result = AnalysisResult(recommendations=recs)
        priority = engine.get_priority_recommendations(result, max_recommendations=2)
        assert len(priority) == 2


class TestNewSmellRecommendations:
    """Tests for recommendation generators of the v1.1 smells."""

    @pytest.mark.parametrize(
        ("smell_type", "severity", "expected_text"),
        [
            ("cartesian_join", SeverityLevel.HIGH, "join"),
            ("topandas_usage", SeverityLevel.HIGH, "toPandas"),
            ("count_for_empty_check", SeverityLevel.MEDIUM, "isEmpty"),
            ("single_partition_write", SeverityLevel.MEDIUM, "coalesce"),
            ("infer_schema", SeverityLevel.MEDIUM, "schema"),
            ("withcolumn_in_loop", SeverityLevel.HIGH, "select"),
            ("select_star", SeverityLevel.LOW, "column"),
            ("orderby_without_limit", SeverityLevel.MEDIUM, "limit"),
            ("pandas_udf_usage", SeverityLevel.MEDIUM, "built-in"),
            ("large_collect", SeverityLevel.MEDIUM, "limit"),
            ("sql_orderby_without_limit", SeverityLevel.MEDIUM, "LIMIT"),
            ("sql_union_instead_of_union_all", SeverityLevel.LOW, "UNION ALL"),
            ("sql_leading_wildcard_like", SeverityLevel.MEDIUM, "LIKE"),
            ("sql_in_subquery", SeverityLevel.LOW, "EXISTS"),
        ],
    )
    def test_generator_registered_with_code_examples(self, smell_type, severity, expected_text):
        """Test that each new smell type has a generator with before/after code."""
        engine = RecommendationEngine()
        smell = CodeSmell(smell_type=smell_type, description="Test", severity=severity)
        recs = engine._generate_recommendations([smell])
        assert len(recs) == 1
        rec = recs[0]
        # A dedicated generator (not the generic fallback) must be registered
        assert smell_type in engine._recommendation_generators
        assert rec.before_code
        assert rec.after_code
        assert rec.explanation
        combined = (rec.suggestion + rec.before_code + rec.after_code + rec.explanation).lower()
        assert expected_text.lower() in combined

    def test_cross_join_end_to_end(self):
        """Test that crossJoin code yields a cartesian join recommendation."""
        code = """
result = df1.crossJoin(df2)
result.show()
"""
        result = analyze_code(code)
        recs = [r for r in result.recommendations if r.smell.smell_type == "cartesian_join"]
        assert len(recs) == 1
        assert "broadcast" in recs[0].after_code or "join" in recs[0].after_code

    def test_sql_findings_end_to_end(self):
        """Test that SQL anti-patterns in spark.sql() literals yield recommendations."""
        code = """
df = spark.sql("SELECT id FROM a UNION SELECT id FROM b ORDER BY id")
"""
        result = analyze_code(code)
        union_recs = [r for r in result.recommendations if r.smell.smell_type == "sql_union_instead_of_union_all"]
        order_recs = [r for r in result.recommendations if r.smell.smell_type == "sql_orderby_without_limit"]
        assert len(union_recs) == 1
        assert "UNION ALL" in union_recs[0].after_code
        assert len(order_recs) == 1
        assert "LIMIT" in order_recs[0].after_code

    def test_sql_select_star_end_to_end_uses_existing_generator(self):
        """Test that SELECT * in SQL flows through the existing select_star generator."""
        code = """
df = spark.sql("SELECT * FROM events")
"""
        result = analyze_code(code)
        recs = [r for r in result.recommendations if r.smell.smell_type == "select_star"]
        assert len(recs) == 1
        assert "select" in recs[0].after_code.lower()

    def test_topandas_end_to_end(self):
        """Test that toPandas code yields a limit/sampling recommendation."""
        code = """
pdf = df.toPandas()
"""
        result = analyze_code(code)
        recs = [r for r in result.recommendations if r.smell.smell_type == "topandas_usage"]
        assert len(recs) == 1
        assert "limit" in recs[0].after_code or "sample" in recs[0].after_code

    def test_withcolumn_loop_end_to_end(self):
        """Test that withColumn-in-loop code yields a select() recommendation."""
        code = """
for c in columns:
    df = df.withColumn(c, df[c] * 2)
"""
        result = analyze_code(code)
        recs = [r for r in result.recommendations if r.smell.smell_type == "withcolumn_in_loop"]
        assert len(recs) == 1
        assert "select" in recs[0].after_code

    def test_pandas_udf_recommendation_distinct_from_plain_udf(self):
        """Test that pandas_udf gets its own MEDIUM-severity recommendation."""
        code = """
my_pudf = pandas_udf(lambda s: s * 2, "double")
"""
        result = analyze_code(code)
        pandas_recs = [r for r in result.recommendations if r.smell.smell_type == "pandas_udf_usage"]
        assert len(pandas_recs) == 1
        assert pandas_recs[0].smell.severity == SeverityLevel.MEDIUM
        assert "built-in" in pandas_recs[0].suggestion.lower()
