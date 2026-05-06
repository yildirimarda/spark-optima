# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Platform information endpoints for the API.

This module provides endpoints for retrieving information about
supported Spark platforms and their configurations.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from spark_optima.api.dependencies import (
    get_all_platforms_metadata,
    get_optimization_service,
    get_platform_metadata,
)
from spark_optima.api.models import PlatformInfoResponse

router = APIRouter(prefix="/api/v1/platforms", tags=["Platforms"])


@router.get(
    "",
    response_model=list[PlatformInfoResponse],
    status_code=status.HTTP_200_OK,
    summary="List supported platforms",
    description="Get a list of all supported Spark platforms.",
    responses={
        200: {
            "description": "List of supported platforms",
            "model": list[PlatformInfoResponse],
        },
    },
)
async def list_platforms() -> list[PlatformInfoResponse]:
    """List all supported Spark platforms.

    This endpoint returns information about all platforms supported
    by Spark Optima, including their capabilities and supported
    Spark versions.

    Returns:
        List of PlatformInfoResponse with platform information.

    Example:
        ```python
        import requests

        response = requests.get("http://localhost:8000/api/v1/platforms")
        platforms = response.json()
        for platform in platforms:
            print(f"{platform['display_name']}: {platform['description']}")
        ```
    """
    platforms_metadata = get_all_platforms_metadata()

    platforms = [
        PlatformInfoResponse(
            name=name,
            display_name=meta["display_name"],
            description=meta["description"],
            supported_spark_versions=meta["supported_spark_versions"],
            supported_features=meta["supported_features"],
        )
        for name, meta in platforms_metadata.items()
    ]

    return platforms


@router.get(
    "/spark-versions",
    status_code=status.HTTP_200_OK,
    summary="List supported Spark versions",
    description="Get a list of all supported Apache Spark versions.",
    responses={
        200: {"description": "List of supported Spark versions"},
    },
)
async def list_spark_versions() -> dict[str, list[str]]:
    """List all supported Spark versions.

    This endpoint returns a list of all Apache Spark versions
    that are supported by Spark Optima.

    Returns:
        Dictionary with list of supported versions.

    Example:
        ```python
        import requests

        response = requests.get("http://localhost:8000/api/v1/platforms/spark-versions")
        versions = response.json()["versions"]
        print(f"Supported versions: {versions}")
        ```
    """
    service = get_optimization_service()
    versions = service.get_available_spark_versions()

    return {"versions": versions}


@router.get(
    "/{platform_name}",
    response_model=PlatformInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get platform details",
    description="Get detailed information about a specific platform.",
    responses={
        200: {
            "description": "Platform details",
            "model": PlatformInfoResponse,
        },
        404: {"description": "Platform not found"},
    },
)
async def get_platform(platform_name: str) -> PlatformInfoResponse:
    """Get details for a specific platform.

    This endpoint returns detailed information about a specific
    platform, including its display name, description, supported
    Spark versions, and available features.

    Args:
        platform_name: Platform identifier (e.g., "databricks", "aws_glue").

    Returns:
        PlatformInfoResponse with platform details.

    Raises:
        HTTPException: If the platform is not supported.

    Example:
        ```python
        import requests

        response = requests.get("http://localhost:8000/api/v1/platforms/databricks")
        platform = response.json()
        print(f"Databricks supports: {platform['supported_spark_versions']}")
        ```
    """
    metadata = get_platform_metadata(platform_name)

    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Platform '{platform_name}' not found. "
            f"Use /api/v1/platforms to list available platforms.",
        )

    return PlatformInfoResponse(
        name=platform_name,
        display_name=metadata["display_name"],
        description=metadata["description"],
        supported_spark_versions=metadata["supported_spark_versions"],
        supported_features=metadata["supported_features"],
    )
