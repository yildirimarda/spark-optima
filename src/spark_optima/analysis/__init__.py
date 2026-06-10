# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Code analysis module for Spark Optima.

This module provides Spark code parsing, smell detection, and
optimization recommendations for Python Spark applications.

Example:
    >>> from spark_optima.analysis import analyze_code
    >>> result = analyze_code(spark_code)
    >>> for rec in result.recommendations:
    ...     print(f"{rec.suggestion}")

"""

from spark_optima.analysis.models import (
    AnalysisResult,
    CodeLocation,
    CodeRecommendation,
    CodeSmell,
    SeverityLevel,
    SparkOperation,
    SparkOperationType,
)
from spark_optima.analysis.parser import (
    ParseResult,
    SparkCodeParser,
    parse_spark_code,
)
from spark_optima.analysis.recommender import (
    RecommendationEngine,
    analyze_code,
)
from spark_optima.analysis.smell_detector import (
    SmellDetector,
    detect_smells,
)
from spark_optima.analysis.sql_analyzer import (
    SQLAnalyzer,
    SQLFinding,
)

__all__ = [
    # Models
    "AnalysisResult",
    "CodeLocation",
    "CodeRecommendation",
    "CodeSmell",
    "ParseResult",
    "SeverityLevel",
    "SparkOperation",
    "SparkOperationType",
    # Parser
    "SparkCodeParser",
    "parse_spark_code",
    # Smell Detection
    "SmellDetector",
    "detect_smells",
    # SQL Analysis
    "SQLAnalyzer",
    "SQLFinding",
    # Recommendations
    "RecommendationEngine",
    "analyze_code",
]
