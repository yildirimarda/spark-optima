# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Dependency injection for FastAPI routes.

This module provides dependency functions for injecting shared
resources into API endpoints, including the optimizer instance
and configuration database.
"""

from __future__ import annotations

import logging
from typing import Any

from spark_optima.core.config_engine.database import ConfigDatabase
from spark_optima.core.optimizer import Optimizer

logger = logging.getLogger(__name__)


class OptimizationService:
    """Service container for optimization-related dependencies.

    This class holds shared resources that can be injected into
    API endpoints, including the configuration database and
    optimizer instances.

    Attributes:
        config_database: Shared configuration database instance.
        _optimizer_cache: Cache of optimizer instances by platform.
    """

    def __init__(self) -> None:
        """Initialize the service container."""
        self.config_database = ConfigDatabase()
        self._optimizer_cache: dict[str, Optimizer] = {}

    def get_optimizer(
        self,
        platform: str,
        spark_version: str = "3.5.0",
        optimization_mode: str = "simulation",
    ) -> Optimizer:
        """Get or create an optimizer instance.

        This method implements a simple caching mechanism to reuse
        optimizer instances for the same platform/version combination.

        Args:
            platform: Target platform (local, databricks, etc.).
            spark_version: Spark version to optimize for.
            optimization_mode: Either "simulation" or "execution".

        Returns:
            Optimizer instance ready to use.

        Raises:
            ValueError: If platform or spark_version is invalid.

        """
        cache_key = f"{platform}:{spark_version}:{optimization_mode}"

        if cache_key not in self._optimizer_cache:
            logger.debug(f"Creating new optimizer for {cache_key}")
            self._optimizer_cache[cache_key] = Optimizer(
                platform=platform,
                spark_version=spark_version,
                optimization_mode=optimization_mode,
            )

        return self._optimizer_cache[cache_key]

    def clear_cache(self) -> None:
        """Clear the optimizer cache.

        This can be useful for freeing memory or ensuring fresh
        optimizer instances.
        """
        self._optimizer_cache.clear()
        logger.debug("Optimizer cache cleared")

    def get_available_spark_versions(self) -> list[str]:
        """Get list of available Spark versions.

        Returns:
            List of supported Spark version strings.
        """
        return self.config_database.get_available_versions()

    def validate_spark_version(self, version: str) -> bool:
        """Validate if a Spark version is supported.

        Args:
            version: Spark version string (e.g., "3.5.0").

        Returns:
            True if version is supported, False otherwise.
        """
        available = self.get_available_spark_versions()
        # Check for exact match or major.minor match
        if version in available:
            return True

        # Try matching major.minor (e.g., "3.5" matches "3.5.0")
        version_prefix = ".".join(version.split(".")[:2])
        return any(v.startswith(version_prefix) for v in available)


# Global service instance (singleton pattern)
_service_instance: OptimizationService | None = None


def get_optimization_service() -> OptimizationService:
    """Get the global optimization service instance.

    This function implements a singleton pattern to ensure
    a single service instance is shared across all requests.

    Returns:
        OptimizationService instance.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = OptimizationService()
    return _service_instance


def get_optimizer(
    platform: str,
    spark_version: str = "3.5.0",
    optimization_mode: str = "simulation",
) -> Optimizer:
    """Dependency function to get an optimizer instance.

    This function can be used as a FastAPI dependency to inject
    an optimizer into route handlers.

    Args:
        platform: Target platform.
        spark_version: Spark version.
        optimization_mode: Optimization mode.

    Returns:
        Configured Optimizer instance.
    """
    service = get_optimization_service()
    return service.get_optimizer(platform, spark_version, optimization_mode)


class APIMetadata:
    """API metadata and configuration.

    This class holds API-wide metadata such as version,
    title, and description.
    """

    TITLE = "Spark Optima API"
    DESCRIPTION = """
    Intelligent Apache Spark configuration optimization API.

    This API provides endpoints for:
    - Optimizing Spark configurations for specific workloads
    - Analyzing Spark code for optimization opportunities
    - Getting platform-specific recommendations

    ## Key Features

    - **Hybrid Optimization**: Combines heuristic rules with Bayesian optimization
    - **Multi-Platform**: Supports Local, AWS Glue, Databricks, Azure Synapse
    - **Code Analysis**: Detects code smells and suggests improvements
    - **Simulation Mode**: Fast performance predictions without running Spark

    ## Authentication

    Currently, the API does not require authentication. In production,
    API key or OAuth2 authentication should be implemented.

    ## Rate Limiting

    Production deployments should implement rate limiting to prevent
    abuse and ensure fair usage.
    """
    VERSION = "0.1.0"
    CONTACT = {
        "name": "Spark Optima Contributors",
        "email": "your-email@example.com",
    }
    LICENSE_INFO = {
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    }


# Platform metadata for API responses
PLATFORM_METADATA: dict[str, dict[str, Any]] = {
    "local": {
        "display_name": "Local Mode",
        "description": "Run Spark locally on a single machine",
        "supported_spark_versions": [
            "3.0.0",
            "3.1.0",
            "3.2.0",
            "3.3.0",
            "3.4.0",
            "3.5.0",
            "4.0.0",
        ],
        "supported_features": [
            "heuristic_optimization",
            "bayesian_optimization",
            "code_analysis",
            "simulation_mode",
            "execution_mode",
        ],
    },
    "aws_glue": {
        "display_name": "AWS Glue",
        "description": "AWS Glue serverless Spark ETL service",
        "supported_spark_versions": ["3.0.0", "3.1.0", "3.2.0", "3.3.0", "3.4.0", "3.5.0"],
        "supported_features": [
            "heuristic_optimization",
            "bayesian_optimization",
            "code_analysis",
            "simulation_mode",
            "cost_estimation",
        ],
    },
    "databricks": {
        "display_name": "Databricks",
        "description": "Databricks unified analytics platform",
        "supported_spark_versions": [
            "3.0.0",
            "3.1.0",
            "3.2.0",
            "3.3.0",
            "3.4.0",
            "3.5.0",
            "4.0.0",
        ],
        "supported_features": [
            "heuristic_optimization",
            "bayesian_optimization",
            "code_analysis",
            "simulation_mode",
            "cost_estimation",
            "cluster_autoscaling",
        ],
    },
    "azure_synapse": {
        "display_name": "Azure Synapse Analytics",
        "description": "Microsoft Azure Synapse Analytics Spark pools",
        "supported_spark_versions": ["3.0.0", "3.1.0", "3.2.0", "3.3.0", "3.4.0", "3.5.0"],
        "supported_features": [
            "heuristic_optimization",
            "bayesian_optimization",
            "code_analysis",
            "simulation_mode",
        ],
    },
}


def get_platform_metadata(platform: str) -> dict[str, Any] | None:
    """Get metadata for a specific platform.

    Args:
        platform: Platform identifier.

    Returns:
        Platform metadata dictionary or None if not found.
    """
    return PLATFORM_METADATA.get(platform)


def get_all_platforms_metadata() -> dict[str, dict[str, Any]]:
    """Get metadata for all supported platforms.

    Returns:
        Dictionary mapping platform names to their metadata.
    """
    return PLATFORM_METADATA.copy()
