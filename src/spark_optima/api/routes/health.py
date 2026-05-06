# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Health check endpoints for the API.

This module provides endpoints for monitoring the health and status
of the Spark Optima API service.
"""

from __future__ import annotations

import time
from datetime import datetime

from fastapi import APIRouter, status

from spark_optima import __version__
from spark_optima.api.dependencies import get_optimization_service
from spark_optima.api.models import HealthResponse

# Module start time for uptime calculation
_module_start_time = time.time()

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if the API service is running and healthy.",
    responses={
        200: {
            "description": "Service is healthy",
            "model": HealthResponse,
        },
        503: {
            "description": "Service is unhealthy",
            "model": HealthResponse,
        },
    },
)
async def health_check() -> HealthResponse:
    """Get the health status of the API.

    This endpoint returns the current health status of the service,
    including uptime, version, and component status.

    Returns:
        HealthResponse with service status information.

    Example:
        ```
        GET /health
        {
            "status": "healthy",
            "version": "0.1.0",
            "uptime_seconds": 3600.5,
            "components": {
                "config_database": "healthy",
                "optimizer": "healthy"
            }
        }
        ```
    """
    # Calculate uptime
    uptime = time.time() - _module_start_time

    # Check component health
    components: dict[str, str] = {}

    try:
        service = get_optimization_service()
        # Try to get available versions to verify config database
        versions = service.get_available_spark_versions()
        if versions:
            components["config_database"] = "healthy"
        else:
            components["config_database"] = "degraded"
    except (RuntimeError, ValueError, AttributeError):
        components["config_database"] = "unhealthy"

    # Determine overall status
    if all(status == "healthy" for status in components.values()):
        overall_status = "healthy"
    elif any(status == "unhealthy" for status in components.values()):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=__version__,
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat(),
        components=components,
    )


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Check if the API is ready to accept requests.",
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"},
    },
)
async def readiness_check() -> dict[str, str]:
    """Check if the service is ready to handle requests.

    This endpoint is typically used by Kubernetes or load balancers
    to determine if the pod should receive traffic.

    Returns:
        Simple status message.

    Example:
        ```
        GET /health/ready
        {"status": "ready"}
        ```
    """
    try:
        service = get_optimization_service()
        # Verify critical components are available
        _ = service.config_database
        return {"status": "ready"}
    except (RuntimeError, ValueError, AttributeError, TypeError) as e:
        return {"status": f"not_ready: {e!s}"}


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
    description="Check if the API process is alive.",
    responses={
        200: {"description": "Service is alive"},
        503: {"description": "Service is not alive"},
    },
)
async def liveness_check() -> dict[str, str]:
    """Check if the service is alive.

    This endpoint is typically used by Kubernetes to determine if
    the container should be restarted.

    Returns:
        Simple status message.

    Example:
        ```
        GET /health/live
        {"status": "alive"}
        ```
    """
    return {"status": "alive"}
