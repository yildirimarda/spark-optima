# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Spark Configuration Engine module.

This module provides comprehensive configuration management for Apache Spark
across multiple versions (3.0-4.1) and platforms.

Example:
    >>> from spark_optima.core.config_engine import ConfigDatabase
    >>> db = ConfigDatabase()
    >>> params = db.get_parameters(version="3.5.0", category="memory")
    >>> config = db.get_recommended_config(
    ...     version="3.5.0",
    ...     platform="databricks",
    ...     resources={"memory_gb": 64, "cores": 16}
    ... )

"""

from spark_optima.core.config_engine.database import ConfigDatabase
from spark_optima.core.config_engine.loader import VersionLoader
from spark_optima.core.config_engine.models import (
    ConfigParameter,
    ConfigSet,
    HeuristicRule,
    ParameterCategory,
    ParameterType,
    PlatformSupport,
    ValidationConstraint,
)
from spark_optima.core.config_engine.validator import ConfigValidator, ValidationError

__all__ = [
    "ConfigDatabase",
    "ConfigParameter",
    "ConfigSet",
    "ConfigValidator",
    "HeuristicRule",
    "ParameterCategory",
    "ParameterType",
    "PlatformSupport",
    "ValidationConstraint",
    "ValidationError",
    "VersionLoader",
]
