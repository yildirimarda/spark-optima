# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Validate and import endpoints for the API.

This module exposes `POST /api/v1/validate` and `POST /api/v1/import`
that mirror the CLI `validate` and `import` commands.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from spark_optima.core import validate_import

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Validate/Import"])


class ValidateRequest(BaseModel):
    """Request model for the validate endpoint."""

    config_file: str = Field(..., description="Path to Spark configuration file (properties or JSON)")
    platform: str = Field("", description="Target platform (optional)")
    spark_version: str = Field("3.5.0", pattern=r"^\d+\.\d+\.\d+$", description="Spark version")


class ValidateResponse(BaseModel):
    """Response model for the validate endpoint."""

    config_file: str = Field(..., description="Path to validated config file")
    spark_version: str = Field(..., description="Spark version used")
    platform: str | None = Field(None, description="Platform validated against")
    parameter_count: int = Field(..., description="Number of parameters checked")
    issues: list[dict[str, str]] = Field(..., description="Validation issues")
    error_count: int = Field(..., description="Number of errors")
    warning_count: int = Field(..., description="Number of warnings")
    valid: bool = Field(..., description="True when no errors found")


class ImportRequest(BaseModel):
    """Request model for the import endpoint."""

    config_file: str = Field(..., description="Path to existing Spark config (properties or JSON)")
    code_path: str = Field(..., description="Path to Spark application code file")
    platform: str = Field("local", description="Target platform")
    spark_version: str = Field("3.5.0", pattern=r"^\d+\.\d+\.\d+$", description="Spark version")
    data_size_gb: float = Field(0.0, ge=0.0, description="Data size in GB")
    bayesian_trials: int = Field(50, ge=1, le=500, description="Number of Bayesian trials")


class ImportResponse(BaseModel):
    """Response model for the import endpoint."""

    current: dict[str, Any] = Field(..., description="Current configuration")
    recommended: dict[str, Any] = Field(..., description="Recommended configuration")
    diff: dict[str, Any] = Field(..., description="Differences between current and recommended")
    estimated_time_minutes: float = Field(..., description="Estimated execution time with recommendation")


@router.post(
    "/validate",
    response_model=ValidateResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate Spark configuration",
    description="Validate a Spark configuration file against the parameter database.",
    responses={
        200: {"description": "Validation completed"},
        400: {"description": "Invalid parameters or missing file"},
        500: {"description": "Internal server error"},
    },
)
async def validate(request: ValidateRequest) -> ValidateResponse:
    """Validate a Spark configuration file.

    Mirrors the ``spark-optima validate`` CLI command.
    """
    try:
        result = validate_import.validate_config(
            config_path=request.config_file,
            platform=request.platform,
            spark_version=request.spark_version,
        )
    except ValueError as exc:
        logger.warning("Validation failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {exc!s}",
        ) from exc

    return ValidateResponse(**result)


@router.post(
    "/import",
    response_model=ImportResponse,
    status_code=status.HTTP_200_OK,
    summary="Import and compare Spark configuration",
    description="Import an existing Spark config, run optimization, and show differences.",
    responses={
        200: {"description": "Import completed"},
        400: {"description": "Invalid parameters or missing file"},
        500: {"description": "Internal server error"},
    },
)
async def import_config(request: ImportRequest) -> ImportResponse:
    """Import an existing Spark config and compare with optimized recommendation.

    Mirrors the ``spark-optima import`` CLI command.
    """
    try:
        result = validate_import.import_config(
            config_path=request.config_file,
            code_path=request.code_path,
            platform=request.platform,
            spark_version=request.spark_version,
            data_size_gb=request.data_size_gb,
            bayesian_trials=request.bayesian_trials,
        )
    except ValueError as exc:
        logger.warning("Import failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Import error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Import failed: {exc!s}",
        ) from exc

    return ImportResponse(**result)
