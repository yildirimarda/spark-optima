# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Result classes for Spark Optima optimization output.

This module defines the data structures used to represent optimization
results, including configurations, metrics, and recommendations.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeSuggestion:
    """Represents a code improvement suggestion.

    Attributes:
        line_number: Line number where the issue was detected.
        issue_type: Category of the issue (e.g., "broadcast", "caching").
        description: Human-readable description of the issue.
        suggestion: Recommended fix or improvement.
        severity: Severity level ("low", "medium", "high", "critical").

    """

    line_number: int
    issue_type: str
    description: str
    suggestion: str
    severity: str = "medium"

    def __post_init__(self) -> None:
        """Validate severity level."""
        valid_severities = ["low", "medium", "high", "critical"]
        if self.severity not in valid_severities:
            raise ValueError(f"Severity must be one of: {valid_severities}")


@dataclass
class OptimizationResult:
    """Container for Spark configuration optimization results.

    This class holds all information about an optimization run, including
    the recommended configuration, performance estimates, and code suggestions.

    Attributes:
        configuration: Dictionary of Spark configuration key-value pairs.
        estimated_time_minutes: Predicted execution time in minutes.
        confidence_score: Confidence level of the optimization (0.0 to 1.0).
        code_suggestions: List of code improvement suggestions.
        platform_specific: Platform-specific configuration details.
        metadata: Additional metadata about the optimization run.

    Example:
        >>> result = OptimizationResult(
        ...     configuration={"spark.executor.memory": "4g"},
        ...     estimated_time_minutes=15.5,
        ...     confidence_score=0.85
        ... )
        >>> print(result.to_dict())

    """

    configuration: dict[str, Any] = field(default_factory=dict)
    estimated_time_minutes: float = 0.0
    confidence_score: float = 0.0
    code_suggestions: list[CodeSuggestion] = field(default_factory=list)
    platform_specific: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result fields."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        if self.estimated_time_minutes < 0:
            raise ValueError("estimated_time_minutes must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format.

        Returns:
            Dictionary representation of the optimization result.

        """
        return {
            "configuration": self.configuration,
            "estimated_time_minutes": self.estimated_time_minutes,
            "confidence_score": self.confidence_score,
            "code_suggestions": [
                {
                    "line_number": s.line_number,
                    "issue_type": s.issue_type,
                    "description": s.description,
                    "suggestion": s.suggestion,
                    "severity": s.severity,
                }
                for s in self.code_suggestions
            ],
            "platform_specific": self.platform_specific,
            "metadata": self.metadata,
        }

    def get_top_suggestions(self, max_suggestions: int = 5) -> list[CodeSuggestion]:
        """Get top code suggestions sorted by severity.

        Args:
            max_suggestions: Maximum number of suggestions to return.

        Returns:
            List of code suggestions sorted by severity (critical first).

        """
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_suggestions = sorted(
            self.code_suggestions,
            key=lambda x: severity_order.get(x.severity, 4),
        )
        return sorted_suggestions[:max_suggestions]

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"OptimizationResult("
            f"config_keys={len(self.configuration)}, "
            f"est_time={self.estimated_time_minutes:.1f}min, "
            f"confidence={self.confidence_score:.1%}, "
            f"suggestions={len(self.code_suggestions)}"
            f")"
        )
