# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for analysis models."""

import pytest

from spark_optima.analysis.models import (
    AnalysisResult,
    CodeLocation,
    CodeRecommendation,
    CodeSmell,
    SeverityLevel,
    SparkOperation,
    SparkOperationType,
)
from spark_optima.analysis.parser import ParseResult


class TestCodeLocation:
    """Test cases for CodeLocation."""

    def test_valid_location(self):
        """Test creating a valid location."""
        loc = CodeLocation(line=10, column=5)
        assert loc.line == 10
        assert loc.column == 5
        assert loc.end_line is None

    def test_invalid_line_number(self):
        """Test that invalid line numbers raise error."""
        with pytest.raises(ValueError, match="Line number must be >= 1"):
            CodeLocation(line=0, column=0)

    def test_invalid_column(self):
        """Test that invalid column numbers raise error."""
        with pytest.raises(ValueError, match="Column number must be >= 0"):
            CodeLocation(line=1, column=-1)

    def test_string_representation(self):
        """Test string representation."""
        loc = CodeLocation(line=10, column=5)
        assert str(loc) == "line 10:5"


class TestSparkOperation:
    """Test cases for SparkOperation."""

    def test_operation_creation(self):
        """Test creating a Spark operation."""
        op = SparkOperation(
            operation_type=SparkOperationType.JOIN,
            method_name="join",
            dataframe_var="df1",
            arguments=["df2", '"id"'],
        )
        assert op.method_name == "join"
        assert op.dataframe_var == "df1"
        assert len(op.arguments) == 2

    def test_operation_string(self):
        """Test operation string representation."""
        op = SparkOperation(
            operation_type=SparkOperationType.READ,
            method_name="read",
            dataframe_var="spark",
            arguments=['"parquet"'],
        )
        assert str(op) == 'spark.read("parquet")'


class TestCodeSmell:
    """Test cases for CodeSmell."""

    def test_smell_creation(self):
        """Test creating a code smell."""
        smell = CodeSmell(
            smell_type="missing_broadcast",
            description="Broadcast hint missing",
            severity=SeverityLevel.MEDIUM,
            impact="Causes shuffle",
        )
        assert smell.smell_type == "missing_broadcast"
        assert smell.severity == SeverityLevel.MEDIUM

    def test_invalid_severity(self):
        """Test that invalid severity raises error."""
        with pytest.raises(ValueError, match="Invalid severity level"):
            CodeSmell(
                smell_type="test",
                description="test",
                severity="invalid",  # type: ignore
            )

    def test_to_dict(self):
        """Test converting smell to dictionary."""
        loc = CodeLocation(line=10, column=0)
        smell = CodeSmell(
            smell_type="test",
            description="Test smell",
            severity=SeverityLevel.HIGH,
            location=loc,
            impact="Test impact",
        )
        data = smell.to_dict()
        assert data["smell_type"] == "test"
        assert data["severity"] == "high"
        assert data["location"]["line"] == 10


class TestCodeRecommendation:
    """Test cases for CodeRecommendation."""

    def test_recommendation_creation(self):
        """Test creating a recommendation."""
        smell = CodeSmell(
            smell_type="test",
            description="test",
            severity=SeverityLevel.LOW,
        )
        rec = CodeRecommendation(
            smell=smell,
            suggestion="Fix this",
            before_code="old",
            after_code="new",
            effort="low",
        )
        assert rec.suggestion == "Fix this"
        assert rec.before_code == "old"

    def test_invalid_effort(self):
        """Test that invalid effort level raises error."""
        smell = CodeSmell(
            smell_type="test",
            description="test",
            severity=SeverityLevel.LOW,
        )
        with pytest.raises(ValueError, match="Effort must be one of"):
            CodeRecommendation(
                smell=smell,
                suggestion="fix",
                effort="invalid",
            )


class TestAnalysisResult:
    """Test cases for AnalysisResult."""

    def test_empty_result(self):
        """Test creating an empty result."""
        result = AnalysisResult()
        assert len(result.operations) == 0
        assert len(result.smells) == 0
        assert not result.has_smells()

    def test_get_smells_by_severity(self):
        """Test filtering smells by severity."""
        smells = [
            CodeSmell("s1", "desc1", SeverityLevel.HIGH),
            CodeSmell("s2", "desc2", SeverityLevel.LOW),
            CodeSmell("s3", "desc3", SeverityLevel.HIGH),
        ]
        result = AnalysisResult(smells=smells)
        high_smells = result.get_smells_by_severity(SeverityLevel.HIGH)
        assert len(high_smells) == 2

    def test_get_critical_smells(self):
        """Test getting critical smells."""
        smells = [
            CodeSmell("s1", "desc1", SeverityLevel.CRITICAL),
            CodeSmell("s2", "desc2", SeverityLevel.HIGH),
        ]
        result = AnalysisResult(smells=smells)
        critical = result.get_critical_smells()
        assert len(critical) == 1
        assert critical[0].severity == SeverityLevel.CRITICAL

    def test_get_high_priority_recommendations(self):
        """Test sorting recommendations by priority."""
        smells = [
            CodeSmell("s1", "desc1", SeverityLevel.LOW),
            CodeSmell("s2", "desc2", SeverityLevel.CRITICAL),
        ]
        recs = [
            CodeRecommendation(smell=smells[0], suggestion="low"),
            CodeRecommendation(smell=smells[1], suggestion="critical"),
        ]
        result = AnalysisResult(recommendations=recs)
        sorted_recs = result.get_high_priority_recommendations()
        assert sorted_recs[0].smell.severity == SeverityLevel.CRITICAL

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = AnalysisResult(
            operations=[],
            smells=[CodeSmell("test", "test", SeverityLevel.LOW)],
            recommendations=[],
        )
        data = result.to_dict()
        assert data["smells_count"] == 1
        assert data["operations_count"] == 0


class TestParseResult:
    """Test cases for ParseResult."""

    def test_parse_result_creation(self):
        """Test creating a parse result."""
        operations = [
            SparkOperation(
                operation_type=SparkOperationType.JOIN,
                method_name="join",
                dataframe_var="df",
            ),
        ]
        result = ParseResult(
            operations=operations,
            dataframe_vars={"df": operations},
            operation_count=1,
        )
        assert result.operation_count == 1
        assert len(result.dataframe_vars) == 1

    def test_get_summary(self):
        """Test getting parse result summary."""
        operations = [
            SparkOperation(
                operation_type=SparkOperationType.JOIN,
                method_name="join",
                dataframe_var="df1",
            ),
            SparkOperation(
                operation_type=SparkOperationType.JOIN,
                method_name="join",
                dataframe_var="df2",
            ),
        ]
        result = ParseResult(
            operations=operations,
            dataframe_vars={},
            operation_count=2,
        )
        summary = result.get_summary()
        assert summary["total_operations"] == 2
        assert summary["operations_by_type"]["JOIN"] == 2


class TestCodeLocationExtended:
    """Extended tests for CodeLocation."""

    def test_string_representation_multiline(self):
        """Test string representation with end_line different from line."""
        loc = CodeLocation(line=10, column=5, end_line=15, end_column=10)
        str_repr = str(loc)
        assert "15" in str_repr
        assert "10:5" in str_repr

    def test_string_representation_no_end_line(self):
        """Test string representation without end_line."""
        loc = CodeLocation(line=10, column=5)
        str_repr = str(loc)
        assert str_repr == "line 10:5"

    def test_code_location_with_end_column_only(self):
        """Test CodeLocation with end_line but no end_column."""
        loc = CodeLocation(line=10, column=5, end_line=15, end_column=None)
        str_repr = str(loc)
        # When end_line != line, should show the range
        assert "line 10:5 - 15:" in str_repr or "line 10:5" in str_repr


class TestSparkOperationExtended:
    """Extended tests for SparkOperation."""

    def test_operation_string_without_arguments(self):
        """Test string representation without arguments."""
        op = SparkOperation(
            operation_type=SparkOperationType.READ,
            method_name="read",
            dataframe_var="spark",
        )
        assert str(op) == "spark.read()"

    def test_operation_string_with_multiple_arguments(self):
        """Test string representation with multiple arguments."""
        op = SparkOperation(
            operation_type=SparkOperationType.JOIN,
            method_name="join",
            dataframe_var="df1",
            arguments=["df2", '"id"', "how='inner'"],
        )
        str_repr = str(op)
        assert "join" in str_repr
        assert "df2" in str_repr


class TestCodeSmellExtended:
    """Extended tests for CodeSmell."""

    def test_to_dict_without_location(self):
        """Test to_dict without location."""
        smell = CodeSmell(
            smell_type="test",
            description="Test",
            severity=SeverityLevel.MEDIUM,
        )
        data = smell.to_dict()
        assert data["location"] is None

    def test_to_dict_with_impact(self):
        """Test to_dict with impact."""
        loc = CodeLocation(line=5, column=0)
        smell = CodeSmell(
            smell_type="test",
            description="Test",
            severity=SeverityLevel.HIGH,
            location=loc,
            impact="Test impact description",
        )
        data = smell.to_dict()
        assert data["impact"] == "Test impact description"
        assert data["location"]["line"] == 5


class TestCodeRecommendationExtended:
    """Extended tests for CodeRecommendation."""

    def test_to_dict(self):
        """Test converting recommendation to dictionary."""
        smell = CodeSmell(
            smell_type="test",
            description="test",
            severity=SeverityLevel.LOW,
        )
        rec = CodeRecommendation(
            smell=smell,
            suggestion="Fix this",
            before_code="old_code",
            after_code="new_code",
            explanation="Do this fix",
            effort="low",
        )
        data = rec.to_dict()
        assert data["suggestion"] == "Fix this"
        assert data["before_code"] == "old_code"
        assert data["after_code"] == "new_code"
        assert data["explanation"] == "Do this fix"
        assert data["effort"] == "low"
        assert "smell" in data

    def test_recommendation_defaults(self):
        """Test recommendation with default values."""
        smell = CodeSmell(
            smell_type="test",
            description="test",
            severity=SeverityLevel.LOW,
        )
        rec = CodeRecommendation(
            smell=smell,
            suggestion="Fix",
        )
        assert rec.before_code == ""
        assert rec.after_code == ""
        assert rec.explanation == ""
        assert rec.effort == "low"


class TestAnalysisResultExtended:
    """Extended tests for AnalysisResult."""

    def test_get_smells_by_type(self):
        """Test filtering smells by type."""
        smells = [
            CodeSmell("missing_broadcast_hint", "desc1", SeverityLevel.MEDIUM),
            CodeSmell("udf_usage", "desc2", SeverityLevel.HIGH),
            CodeSmell("missing_broadcast_hint", "desc3", SeverityLevel.LOW),
        ]
        result = AnalysisResult(smells=smells)
        broadcast_smells = result.get_smells_by_type("missing_broadcast_hint")
        assert len(broadcast_smells) == 2

    def test_has_smells_true(self):
        """Test has_smells with smells present."""
        smells = [CodeSmell("s1", "desc1", SeverityLevel.HIGH)]
        result = AnalysisResult(smells=smells)
        assert result.has_smells() is True

    def test_has_smells_false(self):
        """Test has_smells with no smells."""
        result = AnalysisResult(smells=[])
        assert result.has_smells() is False

    def test_to_dict_with_data(self):
        """Test converting result to dictionary with data."""
        op = SparkOperation(
            operation_type=SparkOperationType.READ,
            method_name="read",
            dataframe_var="df",
        )
        smell = CodeSmell("test", "desc", SeverityLevel.LOW)
        rec = CodeRecommendation(smell=smell, suggestion="Fix")
        result = AnalysisResult(
            operations=[op],
            smells=[smell],
            recommendations=[rec],
            metadata={"source": "test.py"},
        )
        data = result.to_dict()
        assert data["operations_count"] == 1
        assert data["smells_count"] == 1
        assert data["recommendations_count"] == 1
        assert len(data["smells"]) == 1
        assert len(data["recommendations"]) == 1
        assert data["metadata"]["source"] == "test.py"

    def test_str_representation(self):
        """Test string representation of AnalysisResult."""
        result = AnalysisResult(
            operations=[
                SparkOperation(
                    operation_type=SparkOperationType.READ,
                    method_name="read",
                    dataframe_var="df",
                )
            ],
            smells=[CodeSmell("test", "desc", SeverityLevel.LOW)],
            recommendations=[
                CodeRecommendation(
                    smell=CodeSmell("test", "desc", SeverityLevel.LOW),
                    suggestion="Fix",
                )
            ],
        )
        str_repr = str(result)
        assert "AnalysisResult" in str_repr
        assert "operations=1" in str_repr or "operations= 1" in str_repr
        assert "smells=1" in str_repr or "smells= 1" in str_repr
