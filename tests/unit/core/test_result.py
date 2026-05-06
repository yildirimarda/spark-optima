# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for optimization result classes.

This module contains tests for OptimizationResult and CodeSuggestion classes
including validation, serialization, and utility methods.
"""

from __future__ import annotations

import pytest

from spark_optima.core.result import CodeSuggestion, OptimizationResult


class TestCodeSuggestion:
    """Test cases for CodeSuggestion class."""

    def test_code_suggestion_creation(self) -> None:
        """Test basic code suggestion creation."""
        suggestion = CodeSuggestion(
            line_number=10,
            issue_type="broadcast",
            description="Missing broadcast hint for small table",
            suggestion="Use broadcast(small_df) for better performance",
        )

        assert suggestion.line_number == 10
        assert suggestion.issue_type == "broadcast"
        assert suggestion.severity == "medium"  # Default

    def test_code_suggestion_all_severities(self) -> None:
        """Test code suggestion with all valid severity levels."""
        severities = ["low", "medium", "high", "critical"]

        for severity in severities:
            suggestion = CodeSuggestion(
                line_number=1,
                issue_type="test",
                description="Test",
                suggestion="Test",
                severity=severity,
            )
            assert suggestion.severity == severity

    def test_code_suggestion_invalid_severity(self) -> None:
        """Test code suggestion with invalid severity raises error."""
        with pytest.raises(ValueError, match="Severity must be one of"):
            CodeSuggestion(
                line_number=1,
                issue_type="test",
                description="Test",
                suggestion="Test",
                severity="invalid",
            )

    def test_code_suggestion_zero_line_number(self) -> None:
        """Test code suggestion with line number 0 (valid for file-level)."""
        suggestion = CodeSuggestion(
            line_number=0,
            issue_type="config",
            description="Configuration issue",
            suggestion="Update config",
        )
        assert suggestion.line_number == 0

    def test_code_suggestion_negative_line_number(self) -> None:
        """Test code suggestion with negative line number."""
        # Negative line numbers might be allowed for special cases
        suggestion = CodeSuggestion(
            line_number=-1,
            issue_type="general",
            description="General issue",
            suggestion="Fix it",
        )
        assert suggestion.line_number == -1


class TestOptimizationResultCreation:
    """Test cases for OptimizationResult creation."""

    def test_optimization_result_default_values(self) -> None:
        """Test optimization result with default values."""
        result = OptimizationResult()

        assert result.configuration == {}
        assert result.estimated_time_minutes == 0.0
        assert result.confidence_score == 0.0
        assert result.code_suggestions == []
        assert result.platform_specific == {}
        assert result.metadata == {}

    def test_optimization_result_custom_values(self) -> None:
        """Test optimization result with custom values."""
        config = {
            "spark.executor.memory": "4g",
            "spark.executor.cores": "4",
        }
        suggestions = [
            CodeSuggestion(10, "broadcast", "Missing hint", "Add broadcast"),
        ]

        result = OptimizationResult(
            configuration=config,
            estimated_time_minutes=15.5,
            confidence_score=0.85,
            code_suggestions=suggestions,
            platform_specific={"platform": "local"},
            metadata={"trials": 50},
        )

        assert result.configuration == config
        assert result.estimated_time_minutes == 15.5
        assert result.confidence_score == 0.85
        assert len(result.code_suggestions) == 1
        assert result.platform_specific["platform"] == "local"
        assert result.metadata["trials"] == 50

    def test_optimization_result_empty_suggestions(self) -> None:
        """Test optimization result with empty suggestions list."""
        result = OptimizationResult(code_suggestions=[])
        assert result.code_suggestions == []
        assert result.get_top_suggestions() == []


class TestOptimizationResultValidation:
    """Test cases for OptimizationResult validation."""

    def test_confidence_score_boundary_zero(self) -> None:
        """Test confidence score at lower boundary (0.0)."""
        result = OptimizationResult(confidence_score=0.0)
        assert result.confidence_score == 0.0

    def test_confidence_score_boundary_one(self) -> None:
        """Test confidence score at upper boundary (1.0)."""
        result = OptimizationResult(confidence_score=1.0)
        assert result.confidence_score == 1.0

    def test_confidence_score_invalid_negative(self) -> None:
        """Test confidence score with negative value raises error."""
        with pytest.raises(ValueError, match="confidence_score must be between"):
            OptimizationResult(confidence_score=-0.1)

    def test_confidence_score_invalid_over_one(self) -> None:
        """Test confidence score with value over 1.0 raises error."""
        with pytest.raises(ValueError, match="confidence_score must be between"):
            OptimizationResult(confidence_score=1.1)

    def test_estimated_time_zero(self) -> None:
        """Test estimated time at lower boundary (0.0)."""
        result = OptimizationResult(estimated_time_minutes=0.0)
        assert result.estimated_time_minutes == 0.0

    def test_estimated_time_positive(self) -> None:
        """Test estimated time with positive value."""
        result = OptimizationResult(estimated_time_minutes=30.5)
        assert result.estimated_time_minutes == 30.5

    def test_estimated_time_invalid_negative(self) -> None:
        """Test estimated time with negative value raises error."""
        with pytest.raises(ValueError, match="estimated_time_minutes must be non-negative"):
            OptimizationResult(estimated_time_minutes=-1.0)


class TestOptimizationResultSerialization:
    """Test cases for OptimizationResult serialization."""

    def test_to_dict_empty_result(self) -> None:
        """Test conversion to dictionary for empty result."""
        result = OptimizationResult()
        result_dict = result.to_dict()

        assert result_dict["configuration"] == {}
        assert result_dict["estimated_time_minutes"] == 0.0
        assert result_dict["confidence_score"] == 0.0
        assert result_dict["code_suggestions"] == []
        assert result_dict["platform_specific"] == {}
        assert result_dict["metadata"] == {}

    def test_to_dict_full_result(self) -> None:
        """Test conversion to dictionary for complete result."""
        suggestions = [
            CodeSuggestion(10, "broadcast", "Missing hint", "Add broadcast", "high"),
        ]
        result = OptimizationResult(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.9,
            code_suggestions=suggestions,
            platform_specific={"platform": "local"},
            metadata={"trials": 50},
        )

        result_dict = result.to_dict()

        assert result_dict["configuration"]["spark.executor.memory"] == "4g"
        assert result_dict["estimated_time_minutes"] == 10.0
        assert result_dict["confidence_score"] == 0.9
        assert len(result_dict["code_suggestions"]) == 1
        assert result_dict["code_suggestions"][0]["issue_type"] == "broadcast"
        assert result_dict["code_suggestions"][0]["severity"] == "high"
        assert result_dict["platform_specific"]["platform"] == "local"
        assert result_dict["metadata"]["trials"] == 50

    def test_to_dict_preserves_types(self) -> None:
        """Test that to_dict preserves value types correctly."""
        result = OptimizationResult(
            configuration={"key": "value", "number": 42},
            estimated_time_minutes=5.5,
            confidence_score=0.75,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict["configuration"], dict)
        assert isinstance(result_dict["estimated_time_minutes"], float)
        assert isinstance(result_dict["confidence_score"], float)


class TestOptimizationResultGetTopSuggestions:
    """Test cases for get_top_suggestions method."""

    def test_get_top_suggestions_empty(self) -> None:
        """Test getting top suggestions when empty."""
        result = OptimizationResult()
        top = result.get_top_suggestions()
        assert top == []

    def test_get_top_suggestions_single(self) -> None:
        """Test getting top suggestions with single item."""
        suggestions = [
            CodeSuggestion(1, "type1", "desc1", "sugg1", "medium"),
        ]
        result = OptimizationResult(code_suggestions=suggestions)

        top = result.get_top_suggestions()

        assert len(top) == 1
        assert top[0].issue_type == "type1"

    def test_get_top_suggestions_sorted_by_severity(self) -> None:
        """Test suggestions sorted by severity."""
        suggestions = [
            CodeSuggestion(1, "type1", "desc1", "sugg1", "low"),
            CodeSuggestion(2, "type2", "desc2", "sugg2", "critical"),
            CodeSuggestion(3, "type3", "desc3", "sugg3", "high"),
            CodeSuggestion(4, "type4", "desc4", "sugg4", "medium"),
        ]
        result = OptimizationResult(code_suggestions=suggestions)

        top = result.get_top_suggestions()

        assert len(top) == 4
        assert top[0].severity == "critical"
        assert top[1].severity == "high"
        assert top[2].severity == "medium"
        assert top[3].severity == "low"

    def test_get_top_suggestions_with_limit(self) -> None:
        """Test getting limited number of suggestions."""
        suggestions = [
            CodeSuggestion(1, "type1", "desc1", "sugg1", "low"),
            CodeSuggestion(2, "type2", "desc2", "sugg2", "critical"),
            CodeSuggestion(3, "type3", "desc3", "sugg3", "high"),
        ]
        result = OptimizationResult(code_suggestions=suggestions)

        top = result.get_top_suggestions(max_suggestions=2)

        assert len(top) == 2
        assert top[0].severity == "critical"
        assert top[1].severity == "high"

    def test_get_top_suggestions_limit_zero(self) -> None:
        """Test getting top suggestions with limit 0."""
        suggestions = [
            CodeSuggestion(1, "type1", "desc1", "sugg1", "high"),
        ]
        result = OptimizationResult(code_suggestions=suggestions)

        top = result.get_top_suggestions(max_suggestions=0)

        assert top == []

    def test_get_top_suggestions_limit_greater_than_count(self) -> None:
        """Test getting top suggestions with limit greater than count."""
        suggestions = [
            CodeSuggestion(1, "type1", "desc1", "sugg1", "high"),
        ]
        result = OptimizationResult(code_suggestions=suggestions)

        top = result.get_top_suggestions(max_suggestions=10)

        assert len(top) == 1

    def test_get_top_suggestions_same_severity_ordering(self) -> None:
        """Test ordering of suggestions with same severity."""
        suggestions = [
            CodeSuggestion(1, "type1", "desc1", "sugg1", "high"),
            CodeSuggestion(2, "type2", "desc2", "sugg2", "high"),
        ]
        result = OptimizationResult(code_suggestions=suggestions)

        top = result.get_top_suggestions()

        # Same severity, should maintain original order
        assert len(top) == 2
        assert top[0].line_number == 1
        assert top[1].line_number == 2


class TestOptimizationResultStringRepresentation:
    """Test cases for string representation."""

    def test_str_empty_result(self) -> None:
        """Test string representation of empty result."""
        result = OptimizationResult()
        str_repr = str(result)

        assert "OptimizationResult" in str_repr
        assert "config_keys=0" in str_repr
        assert "est_time=0.0min" in str_repr
        assert "confidence=0.0%" in str_repr
        assert "suggestions=0" in str_repr

    def test_str_full_result(self) -> None:
        """Test string representation of complete result."""
        suggestions = [CodeSuggestion(1, "type", "desc", "sugg", "medium")]
        result = OptimizationResult(
            configuration={"key1": "value1", "key2": "value2"},
            estimated_time_minutes=15.5,
            confidence_score=0.85,
            code_suggestions=suggestions,
        )

        str_repr = str(result)

        assert "OptimizationResult" in str_repr
        assert "config_keys=2" in str_repr
        assert "est_time=15.5min" in str_repr
        assert "confidence=85.0%" in str_repr
        assert "suggestions=1" in str_repr

    def test_str_rounding(self) -> None:
        """Test proper rounding in string representation."""
        result = OptimizationResult(
            estimated_time_minutes=10.12345,
            confidence_score=0.12345,
        )

        str_repr = str(result)

        # Should show reasonable precision
        assert "est_time=10.1min" in str_repr or "est_time=10.12min" in str_repr
        assert "confidence=12.3%" in str_repr or "confidence=12.35%" in str_repr


class TestOptimizationResultEdgeCases:
    """Test edge cases."""

    def test_large_configuration(self) -> None:
        """Test result with very large configuration."""
        config = {f"spark.config.{i}": f"value{i}" for i in range(1000)}
        result = OptimizationResult(configuration=config)

        assert len(result.configuration) == 1000
        result_dict = result.to_dict()
        assert len(result_dict["configuration"]) == 1000

    def test_many_suggestions(self) -> None:
        """Test result with many suggestions."""
        suggestions = [
            CodeSuggestion(i, f"type{i}", f"desc{i}", f"sugg{i}", "medium") for i in range(100)
        ]
        result = OptimizationResult(code_suggestions=suggestions)

        assert len(result.code_suggestions) == 100
        top = result.get_top_suggestions(max_suggestions=50)
        assert len(top) == 50

    def test_nested_metadata(self) -> None:
        """Test result with nested metadata structure."""
        metadata = {
            "level1": {
                "level2": {
                    "level3": "deep_value",
                },
            },
            "list": [1, 2, 3],
        }
        result = OptimizationResult(metadata=metadata)

        assert result.metadata["level1"]["level2"]["level3"] == "deep_value"
        assert result.metadata["list"] == [1, 2, 3]

        # Should serialize correctly
        result_dict = result.to_dict()
        assert result_dict["metadata"]["level1"]["level2"]["level3"] == "deep_value"

    def test_unicode_content(self) -> None:
        """Test result with unicode content."""
        suggestion = CodeSuggestion(
            line_number=1,
            issue_type="unicode_test",
            description="Description with unicode: ñ, 中文, 🚀",
            suggestion="Suggestion with unicode: émojis 🎉",
        )
        result = OptimizationResult(
            configuration={"key": "value with unicode: äöü"},
            code_suggestions=[suggestion],
        )

        result_dict = result.to_dict()
        assert "中文" in result_dict["code_suggestions"][0]["description"]
        assert "🎉" in result_dict["code_suggestions"][0]["suggestion"]

    def test_none_values_in_metadata(self) -> None:
        """Test result with None values in metadata."""
        metadata = {
            "null_value": None,
            "valid_value": "test",
        }
        result = OptimizationResult(metadata=metadata)

        assert result.metadata["null_value"] is None
        result_dict = result.to_dict()
        assert result_dict["metadata"]["null_value"] is None
