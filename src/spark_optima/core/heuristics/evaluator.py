# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Safe formula evaluator for heuristic rules.

This module provides a secure mathematical expression evaluator for parsing
and computing heuristic formulas without using unsafe eval() calls.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any


class FormulaError(Exception):
    """Exception raised for formula evaluation errors."""


class FormulaEvaluator:
    """Safe mathematical formula evaluator for heuristic rules.

    This class parses and evaluates mathematical expressions safely
    without using Python's eval(). It supports basic arithmetic,
    mathematical functions, and variable substitution.

    Supported operations:
        - Arithmetic: +, -, *, /, //, %, **
        - Comparison: <, <=, >, >=, ==, !=
        - Functions: min, max, abs, round, floor, ceil, sqrt, log, log10
        - Constants: pi, e

    Example:
        >>> evaluator = FormulaEvaluator()
        >>> result = evaluator.evaluate("min(total_memory / 4, 64)",
        ...                            {"total_memory": 128})
        >>> print(result)  # 32.0

    """

    # Allowed AST node types for safe evaluation
    ALLOWED_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Constant,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Eq,
        ast.NotEq,
        ast.USub,
        ast.UAdd,
        ast.Compare,
    )

    # Allowed mathematical functions
    ALLOWED_FUNCTIONS = {
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "pow": pow,
    }

    # Allowed constants
    ALLOWED_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "true": True,
        "false": False,
    }

    # Bytes pattern: number followed by optional unit
    BYTES_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmgtpe]?[bB]?)$", re.IGNORECASE)

    # Duration pattern: number followed by time unit
    DURATION_PATTERN = re.compile(
        r"^(\d+(?:\.\d+)?)\s*(ms|s|sec|m|min|h|hr|d|day)?$",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        """Initialize the formula evaluator."""
        self._variables: dict[str, Any] = {}

    def evaluate(self, formula: str, variables: dict[str, Any] | None = None) -> Any:
        """Evaluate a mathematical formula safely.

        Args:
            formula: Mathematical expression to evaluate.
            variables: Dictionary of variable names to values.

        Returns:
            Computed result (int, float, or bool).

        Raises:
            FormulaError: If formula is invalid or contains unsafe operations.

        Example:
            >>> evaluator = FormulaEvaluator()
            >>> evaluator.evaluate("2 + 3 * 4")
            14
            >>> evaluator.evaluate("min(a, b)", {"a": 10, "b": 5})
            5

        """
        if not formula or not formula.strip():
            raise FormulaError("Empty formula")

        # Clean up formula
        formula = formula.strip()

        # Update variables
        if variables:
            self._variables = variables.copy()

        try:
            # Parse the formula into AST
            tree = ast.parse(formula, mode="eval")
        except SyntaxError as e:
            raise FormulaError(f"Invalid formula syntax: {e}") from e

        # Validate AST nodes for safety
        self._validate_ast(tree)

        # Evaluate the AST
        try:
            result = self._eval_node(tree.body)
            return result
        except (ValueError, TypeError, AttributeError, ZeroDivisionError) as e:
            raise FormulaError(f"Evaluation error: {e}") from e

    def _validate_ast(self, tree: ast.AST) -> None:
        """Validate that AST only contains allowed node types.

        Args:
            tree: AST to validate.

        Raises:
            FormulaError: If unsafe node type is found.

        """
        for node in ast.walk(tree):
            if not isinstance(node, self.ALLOWED_NODES):
                raise FormulaError(f"Unsafe operation detected: {type(node).__name__}")

    def _eval_node(self, node: ast.AST) -> Any:
        """Recursively evaluate an AST node.

        Args:
            node: AST node to evaluate.

        Returns:
            Evaluated value.

        Raises:
            FormulaError: If node type is not supported.

        """
        if isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Name):
            name = node.id.lower()
            if name in self._variables:
                return self._variables[name]
            elif name in self.ALLOWED_CONSTANTS:
                return self.ALLOWED_CONSTANTS[name]
            else:
                raise FormulaError(f"Unknown variable: {node.id}")

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._apply_binop(left, right, node.op)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            return self._apply_unaryop(operand, node.op)

        elif isinstance(node, ast.Call):
            return self._eval_call(node)

        elif isinstance(node, ast.Compare):
            return self._eval_compare(node)

        else:
            raise FormulaError(f"Unsupported node type: {type(node).__name__}")

    def _apply_binop(self, left: Any, right: Any, op: ast.operator) -> Any:
        """Apply binary operation.

        Args:
            left: Left operand.
            right: Right operand.
            op: Binary operator.

        Returns:
            Operation result.

        """
        if isinstance(op, ast.Add):
            return left + right
        elif isinstance(op, ast.Sub):
            return left - right
        elif isinstance(op, ast.Mult):
            return left * right
        elif isinstance(op, ast.Div):
            if right == 0:
                raise FormulaError("Division by zero")
            return left / right
        elif isinstance(op, ast.FloorDiv):
            if right == 0:
                raise FormulaError("Division by zero")
            return left // right
        elif isinstance(op, ast.Mod):
            return left % right
        elif isinstance(op, ast.Pow):
            return left**right
        else:
            raise FormulaError(f"Unsupported binary operator: {type(op).__name__}")

    def _apply_unaryop(self, operand: Any, op: ast.unaryop) -> Any:
        """Apply unary operation.

        Args:
            operand: Operand value.
            op: Unary operator.

        Returns:
            Operation result.

        """
        if isinstance(op, ast.UAdd):
            return +operand
        elif isinstance(op, ast.USub):
            return -operand
        else:
            raise FormulaError(f"Unsupported unary operator: {type(op).__name__}")

    def _eval_call(self, node: ast.Call) -> Any:
        """Evaluate function call.

        Args:
            node: Call node.

        Returns:
            Function result.

        Raises:
            FormulaError: If function is not allowed.

        """
        if not isinstance(node.func, ast.Name):
            raise FormulaError("Only simple function names allowed")

        func_name = node.func.id.lower()

        if func_name not in self.ALLOWED_FUNCTIONS:
            raise FormulaError(f"Unknown function: {func_name}")

        # Evaluate arguments
        args = [self._eval_node(arg) for arg in node.args]

        # Call the function
        func = self.ALLOWED_FUNCTIONS[func_name]
        return func(*args)  # type: ignore[operator]

    def _eval_compare(self, node: ast.Compare) -> bool:
        """Evaluate comparison expression.

        Args:
            node: Compare node.

        Returns:
            Comparison result.

        """
        left = self._eval_node(node.left)

        for op, comparator in zip(node.ops, node.comparators, strict=False):
            right = self._eval_node(comparator)

            if isinstance(op, ast.Lt):
                if not (left < right):
                    return False
            elif isinstance(op, ast.LtE):
                if not (left <= right):
                    return False
            elif isinstance(op, ast.Gt):
                if not (left > right):
                    return False
            elif isinstance(op, ast.GtE):
                if not (left >= right):
                    return False
            elif isinstance(op, ast.Eq):
                if left != right:
                    return False
            elif isinstance(op, ast.NotEq):
                if left == right:
                    return False
            else:
                raise FormulaError(f"Unsupported comparison: {type(op).__name__}")

            left = right

        return True

    @staticmethod
    def parse_bytes(value_str: str) -> int:
        """Parse byte size string to bytes.

        Args:
            value_str: String like "4g", "512m", "64k", "1024".

        Returns:
            Size in bytes as integer.

        Raises:
            FormulaError: If format is invalid.

        Example:
            >>> FormulaEvaluator.parse_bytes("4g")
            4294967296
            >>> FormulaEvaluator.parse_bytes("512m")
            536870912

        """
        if not value_str or not value_str.strip():
            raise FormulaError("Empty bytes string")

        value_str = value_str.strip().lower()

        # Handle pure number
        if value_str.isdigit():
            return int(value_str)

        match = FormulaEvaluator.BYTES_PATTERN.match(value_str)
        if not match:
            raise FormulaError(f"Invalid bytes format: {value_str}")

        number = float(match.group(1))
        unit = match.group(2).lower() if match.group(2) else "b"

        multipliers = {
            "b": 1,
            "k": 1024,
            "kb": 1024,
            "m": 1024**2,
            "mb": 1024**2,
            "g": 1024**3,
            "gb": 1024**3,
            "t": 1024**4,
            "tb": 1024**4,
            "p": 1024**5,
            "pb": 1024**5,
            "e": 1024**6,
            "eb": 1024**6,
        }

        multiplier = multipliers.get(unit)
        if multiplier is None:
            raise FormulaError(f"Unknown unit: {unit}")

        return int(number * multiplier)

    @staticmethod
    def parse_duration(value_str: str) -> int:
        """Parse duration string to seconds.

        Args:
            value_str: String like "30min", "5s", "1h", "2d".

        Returns:
            Duration in seconds as integer.

        Raises:
            FormulaError: If format is invalid.

        Example:
            >>> FormulaEvaluator.parse_duration("30min")
            1800
            >>> FormulaEvaluator.parse_duration("1h")
            3600

        """
        if not value_str or not value_str.strip():
            raise FormulaError("Empty duration string")

        value_str = value_str.strip().lower()

        # Handle pure number (assume seconds)
        if value_str.isdigit():
            return int(value_str)

        match = FormulaEvaluator.DURATION_PATTERN.match(value_str)
        if not match:
            raise FormulaError(f"Invalid duration format: {value_str}")

        number = float(match.group(1))
        unit = match.group(2).lower() if match.group(2) else "s"

        multipliers = {
            "ms": 0.001,
            "s": 1,
            "sec": 1,
            "m": 60,
            "min": 60,
            "h": 3600,
            "hr": 3600,
            "d": 86400,
            "day": 86400,
        }

        multiplier = multipliers.get(unit)
        if multiplier is None:
            raise FormulaError(f"Unknown time unit: {unit}")

        return int(number * multiplier)

    @staticmethod
    def format_bytes(bytes_val: int) -> str:
        """Format bytes to human-readable string.

        Args:
            bytes_val: Size in bytes.

        Returns:
            Formatted string like "4g", "512m".

        """
        if bytes_val >= 1024**3:
            return f"{bytes_val // 1024**3}g"
        elif bytes_val >= 1024**2:
            return f"{bytes_val // 1024**2}m"
        elif bytes_val >= 1024:
            return f"{bytes_val // 1024}k"
        else:
            return f"{bytes_val}b"

    @staticmethod
    def format_duration(seconds: int) -> str:
        """Format seconds to human-readable duration string.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted string like "5min", "1h", "30s".

        """
        if seconds >= 86400:
            return f"{seconds // 86400}d"
        elif seconds >= 3600:
            return f"{seconds // 3600}h"
        elif seconds >= 60:
            return f"{seconds // 60}min"
        else:
            return f"{seconds}s"
