# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Optimization endpoints for the API.

This module provides endpoints for running Spark configuration
optimization and code analysis.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status

from spark_optima.analysis import RecommendationEngine
from spark_optima.api.dependencies import get_optimization_service
from spark_optima.api.models import (
    AnalysisRequest,
    AnalysisResponse,
    CodeSuggestionResponse,
    OptimizationMetadataResponse,
    OptimizationRequest,
    OptimizationResponse,
    PlatformSpecificConfig,
)
from spark_optima.core.result import CodeSuggestion
from spark_optima.platforms.models import ResourceSpec

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Optimization"])


def _generate_optimization_id() -> str:
    """Generate a unique optimization ID.

    Returns:
        Unique identifier string.
    """
    return f"opt-{uuid.uuid4().hex[:12]}"


def _convert_code_suggestions(suggestions: list[CodeSuggestion]) -> list[CodeSuggestionResponse]:
    """Convert internal CodeSuggestion objects to API response models.

    Args:
        suggestions: List of CodeSuggestion objects.

    Returns:
        List of CodeSuggestionResponse models.
    """
    return [
        CodeSuggestionResponse(
            line_number=s.line_number,
            issue_type=s.issue_type,
            description=s.description,
            suggestion=s.suggestion,
            severity=s.severity,
        )
        for s in suggestions
    ]


@router.post(
    "/optimize",
    response_model=OptimizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Optimize Spark configuration",
    description="Run configuration optimization for Spark code.",
    responses={
        200: {
            "description": "Optimization completed successfully",
            "model": OptimizationResponse,
        },
        400: {"description": "Invalid request parameters"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def optimize(request: OptimizationRequest) -> OptimizationResponse:
    """Optimize Spark configuration for the provided code.

    This endpoint runs the full optimization pipeline including:
    - Code analysis
    - Heuristic configuration generation
    - Bayesian optimization (if enabled)
    - Performance estimation

    Args:
        request: Optimization request with code, platform, and parameters.

    Returns:
        OptimizationResponse with recommended configuration.

    Example:
        ```python
        import requests

        response = requests.post("http://localhost:8000/api/v1/optimize", json={
            "code": "from pyspark.sql import SparkSession...",
            "platform": "databricks",
            "spark_version": "3.5.0",
            "resources": {"cpu_cores": 8, "memory_gb": 32},
            "data_profile": {"size_gb": 100, "format": "parquet"}
        })
        result = response.json()
        ```
    """
    optimization_id = _generate_optimization_id()
    logger.info(f"Starting optimization {optimization_id} for platform {request.platform}")

    try:
        # Get optimization service
        service = get_optimization_service()

        # Validate Spark version
        if not service.validate_spark_version(request.spark_version):
            available = service.get_available_spark_versions()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported Spark version: {request.spark_version}. Available: {available}",
            )

        # Create resource spec
        resources = ResourceSpec(
            cpu_cores=request.resources.cpu_cores,
            memory_gb=request.resources.memory_gb,
            disk_gb=request.resources.disk_gb,
            gpu_count=request.resources.gpu_count,
        )

        # Prepare data profile
        data_profile: dict[str, Any] | None = None
        if request.data_profile:
            data_profile = {
                "size_gb": request.data_profile.size_gb,
                "format": request.data_profile.format.value,
                "schema": request.data_profile.schema,
                "compression": request.data_profile.compression,
                "partitioning": request.data_profile.partitioning,
            }

        # Prepare constraints
        constraints: dict[str, Any] = {}
        if request.constraints:
            if request.constraints.max_memory_gb:
                constraints["max_memory_gb"] = request.constraints.max_memory_gb
            if request.constraints.max_cost_per_hour:
                constraints["max_cost_per_hour"] = request.constraints.max_cost_per_hour
            if request.constraints.max_executors:
                constraints["max_executors"] = request.constraints.max_executors
            if request.constraints.timeout_minutes:
                constraints["timeout_minutes"] = request.constraints.timeout_minutes

        # Write code to temporary file for analysis
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(request.code)
            temp_code_path = Path(f.name)

        try:
            # Get optimizer
            optimizer = service.get_optimizer(
                platform=request.platform.value,
                spark_version=request.spark_version,
                optimization_mode="simulation",  # API always uses simulation mode
            )

            # Run optimization
            result = optimizer.optimize(
                code_path=temp_code_path,
                resources=resources,
                data_profile=data_profile,
                resource_constraints=constraints if constraints else None,
                use_bayesian=request.use_bayesian,
                bayesian_trials=request.bayesian_trials,
                objectives=request.objectives,
            )

            # Build response
            code_suggestions = _convert_code_suggestions(result.code_suggestions)

            metadata = OptimizationMetadataResponse(
                platform=request.platform.value,
                spark_version=request.spark_version,
                optimization_mode="simulation",
                bayesian_used=result.metadata.get("bayesian_used", False),
                bayesian_trials=result.metadata.get("bayesian_trials", 0),
                resources=result.metadata.get("resources", {}),
                data_profile=result.metadata.get("data_profile"),
                code_analysis=result.metadata.get("code_analysis"),
            )

            platform_specific = PlatformSpecificConfig(
                platform=result.platform_specific.get("platform", request.platform.value),
                spark_version=result.platform_specific.get("spark_version", request.spark_version),
                cluster_config=result.platform_specific.get("cluster_config"),
                glue_version=result.platform_specific.get("glue_version"),
                spark_pool_version=result.platform_specific.get("spark_pool_version"),
                spark_config=result.platform_specific.get("spark_config", {}),
            )

            return OptimizationResponse(
                optimization_id=optimization_id,
                status="success",
                configuration=result.configuration,
                estimated_time_minutes=result.estimated_time_minutes,
                confidence_score=result.confidence_score,
                code_suggestions=code_suggestions,
                platform_specific=platform_specific,
                metadata=metadata,
            )

        finally:
            # Clean up temp file
            with contextlib.suppress(Exception):
                temp_code_path.unlink()

    except HTTPException:
        raise
    except (RuntimeError, ValueError, KeyError, AttributeError, TypeError) as e:
        logger.exception(f"Optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {e!s}",
        ) from e


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Spark code",
    description="Analyze Spark code for optimization opportunities.",
    responses={
        200: {
            "description": "Analysis completed successfully",
            "model": AnalysisResponse,
        },
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"},
    },
)
async def analyze(request: AnalysisRequest) -> AnalysisResponse:
    """Analyze Spark code for optimization opportunities.

    This endpoint performs static code analysis to detect:
    - Code smells and anti-patterns
    - Missing optimizations (broadcast hints, caching)
    - Potential performance issues

    Args:
        request: Analysis request with Spark code.

    Returns:
        AnalysisResponse with suggestions and metrics.

    Example:
        ```python
        import requests

        response = requests.post("http://localhost:8000/api/v1/analyze", json={
            "code": "from pyspark.sql import SparkSession..."
        })
        result = response.json()
        ```
    """
    logger.info("Starting code analysis")

    try:
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(request.code)
            temp_code_path = Path(f.name)

        try:
            # Run analysis
            engine = RecommendationEngine()
            analysis = engine.analyze_file(str(temp_code_path))

            # Convert suggestions
            suggestions = _convert_code_suggestions(
                [
                    CodeSuggestion(
                        line_number=rec.smell.location.line if rec.smell.location else 0,
                        issue_type=rec.smell.smell_type,
                        description=rec.smell.description,
                        suggestion=rec.suggestion,
                        severity=rec.smell.severity.value,
                    )
                    for rec in analysis.recommendations
                ],
            )

            return AnalysisResponse(
                operations_count=len(analysis.operations),
                smells_count=len(analysis.smells),
                recommendations_count=len(analysis.recommendations),
                suggestions=suggestions,
            )

        finally:
            # Clean up temp file
            with contextlib.suppress(Exception):
                temp_code_path.unlink()

    except (RuntimeError, ValueError, KeyError, AttributeError, TypeError) as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {e!s}",
        ) from e
