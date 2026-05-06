# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the Optimizer class.

This module contains tests for the core Optimizer functionality.
"""

from pathlib import Path

import pytest

from spark_optima.core.optimizer import Optimizer
from spark_optima.core.result import CodeSuggestion, OptimizationResult


class TestOptimizer:
    """Test cases for the Optimizer class."""

    def test_optimizer_initialization_valid_platform(self) -> None:
        """Test optimizer initialization with valid platform."""
        optimizer = Optimizer(platform="local", spark_version="3.5.0")
        assert optimizer.platform == "local"
        assert optimizer.spark_version == "3.5.0"
        assert optimizer.optimization_mode == "simulation"

    def test_optimizer_initialization_invalid_platform(self) -> None:
        """Test optimizer initialization with invalid platform raises error."""
        with pytest.raises(ValueError, match="Invalid platform"):
            Optimizer(platform="invalid_platform")

    def test_optimizer_initialization_invalid_mode(self) -> None:
        """Test optimizer initialization with invalid mode raises error."""
        with pytest.raises(ValueError, match="Invalid mode"):
            Optimizer(platform="local", optimization_mode="invalid_mode")

    def test_optimizer_initialization_all_valid_platforms(self) -> None:
        """Test optimizer initialization with all valid platforms."""
        valid_platforms = ["local", "databricks", "aws_glue", "azure_synapse"]
        for platform in valid_platforms:
            optimizer = Optimizer(platform=platform)
            assert optimizer.platform == platform

    def test_optimizer_repr(self) -> None:
        """Test optimizer string representation."""
        optimizer = Optimizer(platform="databricks", spark_version="3.5.0")
        repr_str = repr(optimizer)
        assert "Optimizer" in repr_str
        assert "databricks" in repr_str
        assert "3.5.0" in repr_str

    def test_optimize_nonexistent_file(self, tmp_path: Path) -> None:
        """Test optimize with non-existent file raises error."""
        optimizer = Optimizer(platform="local")
        nonexistent_file = tmp_path / "nonexistent.py"

        with pytest.raises(FileNotFoundError, match="Code file not found"):
            optimizer.optimize(code_path=nonexistent_file)

    def test_optimize_with_existing_file(self, tmp_path: Path) -> None:
        """Test optimize with existing file returns result."""
        optimizer = Optimizer(platform="local")
        test_file = tmp_path / "test_spark_job.py"
        test_file.write_text("# Test Spark job\n")

        result = optimizer.optimize(code_path=test_file)

        assert isinstance(result, OptimizationResult)
        assert isinstance(result.configuration, dict)
        assert isinstance(result.code_suggestions, list)


class TestOptimizationResult:
    """Test cases for the OptimizationResult class."""

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
        config = {"spark.executor.memory": "4g"}
        result = OptimizationResult(
            configuration=config,
            estimated_time_minutes=15.5,
            confidence_score=0.85,
        )

        assert result.configuration == config
        assert result.estimated_time_minutes == 15.5
        assert result.confidence_score == 0.85

    def test_optimization_result_invalid_confidence(self) -> None:
        """Test optimization result with invalid confidence score."""
        with pytest.raises(ValueError, match="confidence_score must be between"):
            OptimizationResult(confidence_score=1.5)

        with pytest.raises(ValueError, match="confidence_score must be between"):
            OptimizationResult(confidence_score=-0.1)

    def test_optimization_result_invalid_time(self) -> None:
        """Test optimization result with negative time."""
        with pytest.raises(ValueError, match="estimated_time_minutes must be non-negative"):
            OptimizationResult(estimated_time_minutes=-1.0)

    def test_optimization_result_to_dict(self) -> None:
        """Test conversion to dictionary."""
        suggestion = CodeSuggestion(
            line_number=10,
            issue_type="broadcast",
            description="Missing broadcast hint",
            suggestion="Add broadcast hint",
            severity="medium",
        )
        result = OptimizationResult(
            configuration={"key": "value"},
            estimated_time_minutes=10.0,
            confidence_score=0.9,
            code_suggestions=[suggestion],
        )

        result_dict = result.to_dict()

        assert result_dict["configuration"] == {"key": "value"}
        assert result_dict["estimated_time_minutes"] == 10.0
        assert result_dict["confidence_score"] == 0.9
        assert len(result_dict["code_suggestions"]) == 1
        assert result_dict["code_suggestions"][0]["issue_type"] == "broadcast"

    def test_get_top_suggestions(self) -> None:
        """Test getting top suggestions sorted by severity."""
        suggestions = [
            CodeSuggestion(1, "type1", "desc1", "sugg1", "low"),
            CodeSuggestion(2, "type2", "desc2", "sugg2", "critical"),
            CodeSuggestion(3, "type3", "desc3", "sugg3", "high"),
            CodeSuggestion(4, "type4", "desc4", "sugg4", "medium"),
        ]
        result = OptimizationResult(code_suggestions=suggestions)

        top = result.get_top_suggestions(max_suggestions=2)

        assert len(top) == 2
        assert top[0].severity == "critical"
        assert top[1].severity == "high"

    def test_optimization_result_str(self) -> None:
        """Test string representation."""
        result = OptimizationResult(
            configuration={"key1": "value1", "key2": "value2"},
            estimated_time_minutes=15.5,
            confidence_score=0.85,
            code_suggestions=[CodeSuggestion(1, "type", "desc", "sugg", "low")],
        )

        str_repr = str(result)

        assert "OptimizationResult" in str_repr
        assert "config_keys=2" in str_repr
        assert "est_time=15.5min" in str_repr
        assert "confidence=85.0%" in str_repr
        assert "suggestions=1" in str_repr


class TestCodeSuggestion:
    """Test cases for the CodeSuggestion class."""

    def test_code_suggestion_default_severity(self) -> None:
        """Test code suggestion with default severity."""
        suggestion = CodeSuggestion(
            line_number=10,
            issue_type="broadcast",
            description="Missing broadcast hint",
            suggestion="Add broadcast hint",
        )

        assert suggestion.severity == "medium"

    def test_code_suggestion_invalid_severity(self) -> None:
        """Test code suggestion with invalid severity raises error."""
        with pytest.raises(ValueError, match="Severity must be one of"):
            CodeSuggestion(
                line_number=10,
                issue_type="broadcast",
                description="desc",
                suggestion="sugg",
                severity="invalid",
            )

    def test_code_suggestion_valid_severities(self) -> None:
        """Test code suggestion with all valid severities."""
        valid_severities = ["low", "medium", "high", "critical"]
        for severity in valid_severities:
            suggestion = CodeSuggestion(
                line_number=10,
                issue_type="broadcast",
                description="desc",
                suggestion="sugg",
                severity=severity,
            )
            assert suggestion.severity == severity
