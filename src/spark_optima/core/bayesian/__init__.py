# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Bayesian optimization module for Spark configuration tuning.

This module provides Bayesian optimization capabilities using Optuna
for finding optimal Spark configurations based on heuristic starting points.
"""

from spark_optima.core.bayesian.models import (
    BayesianOptimizationResult,
    SearchSpaceConfig,
    TrialResult,
)
from spark_optima.core.bayesian.optimizer import BayesianOptimizer
from spark_optima.core.bayesian.search_space import SearchSpaceBuilder
from spark_optima.core.bayesian.trial_runner import TrialRunner

__all__ = [
    "BayesianOptimizer",
    "BayesianOptimizationResult",
    "TrialResult",
    "SearchSpaceBuilder",
    "SearchSpaceConfig",
    "TrialRunner",
]
