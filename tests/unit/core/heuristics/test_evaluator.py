# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the formula evaluator."""

from __future__ import annotations

import pytest

from spark_optima.core.heuristics.evaluator import FormulaError, FormulaEvaluator


class TestFormulaEvaluator:
    """Test cases for FormulaEvaluator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = FormulaEvaluator()

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert self.evaluator.evaluate("2 + 3") == 5
        assert self.evaluator.evaluate("10 - 4") == 6
        assert self.evaluator.evaluate("3 * 4") == 12
        assert self.evaluator.evaluate("15 / 3") == 5.0

    def test_advanced_operations(self):
        """Test advanced operations."""
        assert self.evaluator.evaluate("2 ** 3") == 8
        assert self.evaluator.evaluate("17 // 5") == 3
        assert self.evaluator.evaluate("17 % 5") == 2

    def test_functions(self):
        """Test mathematical functions."""
        assert self.evaluator.evaluate("min(5, 3)") == 3
        assert self.evaluator.evaluate("max(5, 3)") == 5
        assert self.evaluator.evaluate("abs(-5)") == 5
        assert self.evaluator.evaluate("round(3.7)") == 4

    def test_variables(self):
        """Test variable substitution."""
        variables = {"x": 10, "y": 20}
        assert self.evaluator.evaluate("x + y", variables) == 30
        assert self.evaluator.evaluate("x * 2", variables) == 20

    def test_complex_expression(self):
        """Test complex expressions."""
        variables = {"total_memory": 128, "num_executors": 4}
        result = self.evaluator.evaluate("min(total_memory / num_executors, 64)", variables)
        assert result == 32.0

    def test_comparison_operators(self):
        """Test comparison operators."""
        assert self.evaluator.evaluate("5 > 3") is True
        assert self.evaluator.evaluate("5 < 3") is False
        assert self.evaluator.evaluate("5 == 5") is True
        assert self.evaluator.evaluate("5 != 3") is True

    def test_division_by_zero(self):
        """Test division by zero handling."""
        with pytest.raises(FormulaError):
            self.evaluator.evaluate("10 / 0")

    def test_invalid_formula(self):
        """Test invalid formula handling."""
        with pytest.raises(FormulaError):
            self.evaluator.evaluate("invalid syntax here")

    def test_unknown_variable(self):
        """Test unknown variable handling."""
        with pytest.raises(FormulaError):
            self.evaluator.evaluate("unknown_var + 5")

    def test_empty_formula(self):
        """Test empty formula handling."""
        with pytest.raises(FormulaError):
            self.evaluator.evaluate("")

    def test_unsafe_operation_blocked(self):
        """Test that unsafe operations are blocked."""
        with pytest.raises(FormulaError):
            self.evaluator.evaluate("__import__('os').system('ls')")


class TestBytesParsing:
    """Test cases for byte parsing."""

    def test_parse_bytes_basic(self):
        """Test basic byte parsing."""
        assert FormulaEvaluator.parse_bytes("1024") == 1024
        assert FormulaEvaluator.parse_bytes("1k") == 1024
        assert FormulaEvaluator.parse_bytes("1m") == 1024**2
        assert FormulaEvaluator.parse_bytes("1g") == 1024**3

    def test_parse_bytes_case_insensitive(self):
        """Test case insensitivity."""
        assert FormulaEvaluator.parse_bytes("1K") == 1024
        assert FormulaEvaluator.parse_bytes("1M") == 1024**2
        assert FormulaEvaluator.parse_bytes("1G") == 1024**3

    def test_parse_bytes_with_spaces(self):
        """Test parsing with spaces."""
        assert FormulaEvaluator.parse_bytes("  1g  ") == 1024**3

    def test_parse_bytes_invalid(self):
        """Test invalid byte format."""
        with pytest.raises(FormulaError):
            FormulaEvaluator.parse_bytes("invalid")

    def test_format_bytes(self):
        """Test byte formatting."""
        assert FormulaEvaluator.format_bytes(1024**3) == "1g"
        assert FormulaEvaluator.format_bytes(1024**2) == "1m"
        assert FormulaEvaluator.format_bytes(1024) == "1k"
        assert FormulaEvaluator.format_bytes(512) == "512b"


class TestDurationParsing:
    """Test cases for duration parsing."""

    def test_parse_duration_seconds(self):
        """Test seconds parsing."""
        assert FormulaEvaluator.parse_duration("30") == 30
        assert FormulaEvaluator.parse_duration("30s") == 30

    def test_parse_duration_minutes(self):
        """Test minutes parsing."""
        assert FormulaEvaluator.parse_duration("5m") == 300
        assert FormulaEvaluator.parse_duration("5min") == 300

    def test_parse_duration_hours(self):
        """Test hours parsing."""
        assert FormulaEvaluator.parse_duration("2h") == 7200
        assert FormulaEvaluator.parse_duration("1hr") == 3600

    def test_parse_duration_invalid(self):
        """Test invalid duration format."""
        with pytest.raises(FormulaError):
            FormulaEvaluator.parse_duration("invalid")

    def test_format_duration(self):
        """Test duration formatting."""
        assert FormulaEvaluator.format_duration(3600) == "1h"
        assert FormulaEvaluator.format_duration(300) == "5min"
        assert FormulaEvaluator.format_duration(30) == "30s"
