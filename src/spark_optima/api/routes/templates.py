# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Workload template endpoints for the API.

This module exposes the curated workload templates (batch ETL, streaming,
ML training, interactive analytics) over HTTP, with parity to the
``spark-optima templates`` CLI command. Access is read-only: templates are
loaded once from ``data/templates`` and served from memory.
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, HTTPException, status

from spark_optima.api.models import (
    TemplateDetailResponse,
    TemplateListResponse,
    TemplateParameterResponse,
    TemplateSummaryResponse,
)
from spark_optima.core.templates import TemplateRegistry, WorkloadTemplate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/templates", tags=["Templates"])

# Registry singleton (templates are static YAML files; load them once)
_registry_lock = threading.Lock()
_registry_instance: TemplateRegistry | None = None


def get_template_registry() -> TemplateRegistry:
    """Get the shared read-only template registry instance.

    Returns:
        The lazily created TemplateRegistry singleton.
    """
    global _registry_instance
    with _registry_lock:
        if _registry_instance is None:
            _registry_instance = TemplateRegistry()
        return _registry_instance


def _to_detail(template: WorkloadTemplate) -> TemplateDetailResponse:
    """Convert a WorkloadTemplate to its full API representation.

    Args:
        template: The internal template record.

    Returns:
        TemplateDetailResponse including config and per-parameter comments.
    """
    return TemplateDetailResponse(
        name=template.name,
        display_name=template.display_name,
        description=template.description,
        workload_traits=list(template.workload_traits),
        config={
            name: TemplateParameterResponse(value=parameter.value, comment=parameter.comment)
            for name, parameter in template.config.items()
        },
        recommended_for=list(template.recommended_for),
        not_recommended_for=list(template.not_recommended_for),
    )


@router.get(
    "",
    response_model=TemplateListResponse,
    status_code=status.HTTP_200_OK,
    summary="List workload templates",
    description="List the curated workload templates with summary information.",
    responses={
        200: {
            "description": "Template summaries sorted by name",
            "model": TemplateListResponse,
        },
    },
)
async def list_templates() -> TemplateListResponse:
    """List all curated workload templates.

    Returns:
        TemplateListResponse with one summary per template, sorted by name.

    Example:
        ```bash
        curl -s http://localhost:8000/api/v1/templates
        ```
    """
    registry = get_template_registry()
    return TemplateListResponse(
        templates=[
            TemplateSummaryResponse(
                name=template.name,
                display_name=template.display_name,
                description=template.description,
                parameter_count=len(template.config),
            )
            for template in registry.list_templates()
        ],
    )


@router.get(
    "/{name}",
    response_model=TemplateDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a workload template",
    description="Get a single workload template including its configuration and per-parameter rationale.",
    responses={
        200: {
            "description": "Full template details",
            "model": TemplateDetailResponse,
        },
        404: {"description": "Template not found"},
    },
)
async def get_template(name: str) -> TemplateDetailResponse:
    """Get a single workload template by name.

    Args:
        name: Template identifier (e.g. "etl-batch").

    Returns:
        TemplateDetailResponse with the full template including the curated
        configuration and per-parameter comments.

    Raises:
        HTTPException: 404 if no template with that name exists.

    Example:
        ```bash
        curl -s http://localhost:8000/api/v1/templates/etl-batch
        ```
    """
    registry = get_template_registry()
    try:
        template = registry.get_template(name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _to_detail(template)
