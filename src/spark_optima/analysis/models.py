# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Data models for code analysis module.

This module defines the data structures used to represent code analysis results,
including Spark operations, code smells, and recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class SparkOperationType(Enum):
    """Types of Spark operations that can be detected."""

    READ = auto()
    WRITE = auto()
    TRANSFORMATION = auto()
    ACTION = auto()
    JOIN = auto()
    AGGREGATION = auto()
    CACHE = auto()
    REPARTITION = auto()
    UDF = auto()
    WINDOW = auto()


class SeverityLevel(Enum):
    """Severity levels for code smells."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CodeLocation:
    """Represents a location in source code.

    Attributes:
        line: Line number (1-indexed).
        column: Column number (0-indexed).
        end_line: End line number for multi-line expressions.
        end_column: End column number.

    """

    line: int
    column: int = 0
    end_line: int | None = None
    end_column: int | None = None

    def __post_init__(self) -> None:
        """Validate location values."""
        if self.line < 1:
            raise ValueError("Line number must be >= 1")
        if self.column < 0:
            raise ValueError("Column number must be >= 0")

    def __str__(self) -> str:
        """Return string representation of location."""
        if self.end_line and self.end_line != self.line:
            return f"line {self.line}:{self.column} - {self.end_line}:{self.end_column or ''}"
        return f"line {self.line}:{self.column}"


@dataclass
class SparkOperation:
    """Represents a detected Spark operation.

    Attributes:
        operation_type: Type of Spark operation.
        method_name: Name of the method called (e.g., 'join', 'groupBy').
        dataframe_var: Variable name of the DataFrame being operated on.
        arguments: List of argument representations.
        location: Source code location.
        chain_position: Position in the transformation chain.

    """

    operation_type: SparkOperationType
    method_name: str
    dataframe_var: str
    arguments: list[str] = field(default_factory=list)
    location: CodeLocation | None = None
    chain_position: int = 0

    def __str__(self) -> str:
        """Return string representation of operation."""
        args_str = ", ".join(self.arguments) if self.arguments else ""
        return f"{self.dataframe_var}.{self.method_name}({args_str})"


@dataclass
class CodeSmell:
    """Represents a detected code smell in Spark code.

    Attributes:
        smell_type: Type of code smell (e.g., 'missing_broadcast').
        description: Human-readable description of the issue.
        severity: Severity level of the smell.
        location: Source code location.
        affected_operation: The Spark operation associated with this smell.
        impact: Description of performance impact.

    """

    smell_type: str
    description: str
    severity: SeverityLevel
    location: CodeLocation | None = None
    affected_operation: SparkOperation | None = None
    impact: str = ""

    def __post_init__(self) -> None:
        """Validate severity level."""
        if not isinstance(self.severity, SeverityLevel):
            raise ValueError(f"Invalid severity level: {self.severity}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation of the code smell.

        """
        return {
            "smell_type": self.smell_type,
            "description": self.description,
            "severity": self.severity.value,
            "location": {
                "line": self.location.line if self.location else None,
                "column": self.location.column if self.location else None,
            }
            if self.location
            else None,
            "impact": self.impact,
        }


@dataclass
class CodeRecommendation:
    """Represents a code improvement recommendation.

    Attributes:
        smell: The associated code smell.
        suggestion: Recommended fix or improvement.
        before_code: Original problematic code.
        after_code: Suggested improved code.
        explanation: Detailed explanation of the fix.
        effort: Estimated effort to apply (low/medium/high).

    """

    smell: CodeSmell
    suggestion: str
    before_code: str = ""
    after_code: str = ""
    explanation: str = ""
    effort: str = "low"

    def __post_init__(self) -> None:
        """Validate effort level."""
        valid_efforts = ["low", "medium", "high"]
        if self.effort not in valid_efforts:
            raise ValueError(f"Effort must be one of: {valid_efforts}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation of the recommendation.

        """
        return {
            "smell": self.smell.to_dict(),
            "suggestion": self.suggestion,
            "before_code": self.before_code,
            "after_code": self.after_code,
            "explanation": self.explanation,
            "effort": self.effort,
        }


@dataclass
class AnalysisResult:
    """Container for complete code analysis results.

    Attributes:
        operations: List of detected Spark operations.
        smells: List of detected code smells.
        recommendations: List of improvement recommendations.
        metadata: Additional analysis metadata.

    """

    operations: list[SparkOperation] = field(default_factory=list)
    smells: list[CodeSmell] = field(default_factory=list)
    recommendations: list[CodeRecommendation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_smells_by_severity(self, severity: SeverityLevel) -> list[CodeSmell]:
        """Get smells filtered by severity level.

        Args:
            severity: Severity level to filter by.

        Returns:
            List of code smells with the specified severity.

        """
        return [s for s in self.smells if s.severity == severity]

    def get_smells_by_type(self, smell_type: str) -> list[CodeSmell]:
        """Get smells filtered by type.

        Args:
            smell_type: Type of smell to filter by.

        Returns:
            List of code smells of the specified type.

        """
        return [s for s in self.smells if s.smell_type == smell_type]

    def get_critical_smells(self) -> list[CodeSmell]:
        """Get all critical severity smells.

        Returns:
            List of critical code smells.

        """
        return self.get_smells_by_severity(SeverityLevel.CRITICAL)

    def get_high_priority_recommendations(self) -> list[CodeRecommendation]:
        """Get recommendations sorted by priority.

        Returns:
            List of recommendations sorted by severity.

        """
        severity_order = {
            SeverityLevel.CRITICAL: 0,
            SeverityLevel.HIGH: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 3,
        }
        return sorted(
            self.recommendations,
            key=lambda r: severity_order.get(r.smell.severity, 4),
        )

    def has_smells(self) -> bool:
        """Check if any smells were detected.

        Returns:
            True if smells were detected, False otherwise.

        """
        return len(self.smells) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation of the analysis result.

        """
        return {
            "operations_count": len(self.operations),
            "smells_count": len(self.smells),
            "recommendations_count": len(self.recommendations),
            "smells": [s.to_dict() for s in self.smells],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"AnalysisResult("
            f"operations={len(self.operations)}, "
            f"smells={len(self.smells)}, "
            f"recommendations={len(self.recommendations)}"
            f")"
        )
