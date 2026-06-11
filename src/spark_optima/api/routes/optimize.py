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
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, status

from spark_optima.analysis import RecommendationEngine
from spark_optima.api import webhooks
from spark_optima.api.dependencies import get_optimization_service
from spark_optima.api.jobs import BaseJobStore, Job, get_job_store
from spark_optima.api.models import (
    AnalysisRequest,
    AnalysisResponse,
    CodeSuggestionResponse,
    JobSubmittedResponse,
    OptimizationMetadataResponse,
    OptimizationRequest,
    OptimizationResponse,
    PlatformSpecificConfig,
)
from spark_optima.core.result import CodeSuggestion
from spark_optima.platforms.models import ResourceSpec

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Optimization"])

#: Minimum seconds between persisted progress writes for one job. Module-level
#: so tests can patch it to observe every trial.
PROGRESS_WRITE_INTERVAL_SECONDS = 0.5


class JobProgressRecorder:
    """Forwards per-trial optimizer progress events to the job store.

    Instances are bound to a job id via :meth:`bind` (passed as the
    ``on_accepted`` hook of ``store.submit``) and then used as the
    ``progress_callback`` of the optimization run. Writes are throttled to
    at most one every ``PROGRESS_WRITE_INTERVAL_SECONDS`` seconds, except
    for the final trial which is always persisted. Runs that stop early
    (timeout, max consecutive failures) never emit that final trial, so
    :meth:`flush` must be called once the run ends to persist the last
    event that was throttled away.

    Attributes:
        _store: Job store receiving the progress snapshots.
        _job_id: Bound job identifier, None until bind() is called.
        _last_write: Monotonic timestamp of the last persisted write.
        _last_event: Most recent progress event seen, persisted or not.
    """

    def __init__(self, store: BaseJobStore) -> None:
        """Initialize the recorder.

        Args:
            store: Job store to persist progress snapshots into.
        """
        self._store = store
        self._job_id: str | None = None
        self._last_write = 0.0
        self._last_event: dict[str, Any] | None = None

    def bind(self, job_id: str) -> None:
        """Bind the recorder to a job id before the work starts.

        Args:
            job_id: Identifier assigned by the job store at submission.
        """
        self._job_id = job_id

    def __call__(self, progress: dict[str, Any]) -> None:
        """Persist a progress event, throttled to the configured interval.

        Args:
            progress: Progress event from the Bayesian optimizer.
        """
        if self._job_id is None:  # pragma: no cover — bind() runs before the work starts
            return
        self._last_event = progress
        now = time.monotonic()
        is_final = progress.get("trials_completed") == progress.get("n_trials")
        if not is_final and now - self._last_write < PROGRESS_WRITE_INTERVAL_SECONDS:
            return
        self._last_write = now
        self._store.set_progress(self._job_id, progress)

    def flush(self) -> None:
        """Unconditionally persist the most recent progress event.

        Called once the optimization run has ended (successfully or not) so
        the stored snapshot always matches the last event the optimizer
        reported, even when the run stopped early and the event had been
        throttled away. No-op when unbound or no event was ever seen.
        """
        if self._job_id is None or self._last_event is None:
            return
        self._store.set_progress(self._job_id, self._last_event)


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


def _execute_optimization(
    request: OptimizationRequest,
    optimization_id: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> OptimizationResponse:
    """Run the full optimization pipeline synchronously.

    This is the shared implementation behind both the synchronous
    ``POST /optimize`` endpoint and the asynchronous job-based
    ``POST /optimize/async`` endpoint.

    Args:
        request: Optimization request with code, platform, and parameters.
        optimization_id: Identifier to attach to the resulting response.
        progress_callback: Optional per-trial progress callback forwarded to
            the optimizer (used by async jobs to persist live progress).

    Returns:
        OptimizationResponse with the recommended configuration.

    Raises:
        HTTPException: 400 for an unsupported Spark version, 500 when the
            optimization pipeline fails.
    """
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
                "schema": request.data_profile.schema_info,
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
                progress_callback=progress_callback,
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
    return _execute_optimization(request, optimization_id)


def _build_webhook_callback(webhook_url: str | None, store: BaseJobStore) -> Callable[[Job], None] | None:
    """Build the job completion callback delivering a webhook notification.

    The callback runs on the job worker thread after the final state has
    been persisted. It POSTs the notification payload (with retries) and
    records the delivery outcome as ``webhook_status`` on the job record.
    Delivery failures are logged and never affect the job state.

    Args:
        webhook_url: Validated webhook URL from the request, or None.
        store: The job store holding the job record.

    Returns:
        The callback to pass to ``store.submit()``, or None when no
        webhook was requested.
    """
    if webhook_url is None:
        return None

    def _deliver(job: Job) -> None:
        """Deliver the webhook and record the outcome on the job."""
        delivered = webhooks.deliver_webhook(webhook_url, webhooks.build_webhook_payload(job))
        store.set_webhook_status(job.job_id, "delivered" if delivered else "failed")

    return _deliver


@router.post(
    "/optimize/async",
    response_model=JobSubmittedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an asynchronous optimization job",
    description="Queue an optimization run and return a job id for polling.",
    responses={
        202: {
            "description": "Optimization job accepted",
            "model": JobSubmittedResponse,
        },
        400: {"description": "Invalid request parameters"},
        422: {"description": "Validation error"},
    },
)
async def optimize_async(request: OptimizationRequest) -> JobSubmittedResponse:
    """Submit an optimization run as an asynchronous background job.

    The request is validated and queued on a small worker pool; the
    response returns immediately with a job id. Poll
    ``GET /api/v1/jobs/{job_id}`` until the job status is "completed"
    (the full optimization result is included) or "failed", or stream
    live per-trial progress from ``GET /api/v1/jobs/{job_id}/events``
    (Server-Sent Events). When ``webhook_url`` is provided, the API
    additionally POSTs a JSON notification to that URL once the job
    finishes; the delivery outcome is reported as ``webhook_status`` on
    the job record.

    Args:
        request: Optimization request with code, platform, and parameters.

    Returns:
        JobSubmittedResponse with the job id and polling URL.

    Raises:
        HTTPException: 400 for an unsupported Spark version.

    Example:
        ```python
        import requests

        response = requests.post("http://localhost:8000/api/v1/optimize/async", json={...})
        job = response.json()
        status = requests.get(f"http://localhost:8000{job['status_url']}").json()
        ```
    """
    # Validate upfront so obviously broken requests fail fast with 400
    # instead of being queued and failing later.
    service = get_optimization_service()
    if not service.validate_spark_version(request.spark_version):
        available = service.get_available_spark_versions()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported Spark version: {request.spark_version}. Available: {available}",
        )

    optimization_id = _generate_optimization_id()
    store = get_job_store()
    progress_recorder = JobProgressRecorder(store)

    def _work() -> dict[str, Any]:
        """Run the optimization and return the response payload."""
        try:
            return _execute_optimization(request, optimization_id, progress_callback=progress_recorder).model_dump()
        except HTTPException as exc:
            # Unwrap HTTP errors into plain messages for the job record
            raise RuntimeError(str(exc.detail)) from exc
        finally:
            # Early-stopped runs (timeout, max consecutive failures) end
            # without the always-persisted final trial; flush so the stored
            # snapshot matches the last event the optimizer reported. Runs
            # on success AND failure paths.
            progress_recorder.flush()

    on_finished = _build_webhook_callback(request.webhook_url, store)
    job_id = store.submit(
        _work,
        platform=request.platform.value,
        spark_version=request.spark_version,
        on_finished=on_finished,
        on_accepted=progress_recorder.bind,
    )
    job = store.get(job_id)
    current_status = job.status if job is not None else "pending"
    logger.info(f"Accepted async optimization job {job_id} (optimization {optimization_id})")

    return JobSubmittedResponse(
        job_id=job_id,
        status=current_status,
        status_url=f"/api/v1/jobs/{job_id}",
    )


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
