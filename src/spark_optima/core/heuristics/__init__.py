# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Heuristic optimization engine for Spark configuration.

This module provides classes for calculating optimal Spark configurations
based on resource constraints, platform characteristics, and data profiles
using rule-based heuristics.

Example:
    >>> from spark_optima.core.heuristics import HeuristicEngine
    >>> from spark_optima.core.config_engine import ConfigDatabase
    >>> db = ConfigDatabase()
    >>> engine = HeuristicEngine(db.get_config_set("3.5.0"))
    >>> config = engine.evaluate(
    ...     resources={"total_memory_gb": 64, "total_cores": 16},
    ...     platform="local",
    ...     data_profile={"format": "parquet", "size_gb": 100}
    ... )

"""

from spark_optima.core.heuristics.context import DataProfile, EvaluationContext
from spark_optima.core.heuristics.engine import HeuristicEngine
from spark_optima.core.heuristics.evaluator import FormulaError, FormulaEvaluator
from spark_optima.core.heuristics.rules import HeuristicRuleDef, RuleRegistry

__all__ = [
    "HeuristicEngine",
    "FormulaEvaluator",
    "FormulaError",
    "EvaluationContext",
    "DataProfile",
    "RuleRegistry",
    "HeuristicRuleDef",
]
