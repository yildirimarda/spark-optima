# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Job status endpoints for asynchronous optimization runs.

This module provides endpoints to poll the status and results of jobs
submitted via ``POST /api/v1/optimize/async``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, status

from spark_optima.api.jobs import Job, get_job_store
from spark_optima.api.models import JobDetailResponse, JobListResponse, JobSummaryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])


def _to_summary(job: Job) -> JobSummaryResponse:
    """Convert a Job to its API summary representation.

    Args:
        job: The internal job record.

    Returns:
        JobSummaryResponse for list endpoints.
    """
    return JobSummaryResponse(
        job_id=job.job_id,
        status=job.status,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        platform=job.platform,
        spark_version=job.spark_version,
    )


def _to_detail(job: Job) -> JobDetailResponse:
    """Convert a Job to its full API representation.

    Args:
        job: The internal job record.

    Returns:
        JobDetailResponse including result or error payloads.
    """
    return JobDetailResponse(
        job_id=job.job_id,
        status=job.status,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        platform=job.platform,
        spark_version=job.spark_version,
        result=job.result,
        error=job.error,
    )


@router.get(
    "",
    response_model=JobListResponse,
    status_code=status.HTTP_200_OK,
    summary="List optimization jobs",
    description="List recently submitted asynchronous optimization jobs, newest first.",
    responses={
        200: {
            "description": "Job summaries, newest first",
            "model": JobListResponse,
        },
    },
)
async def list_jobs(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of jobs to return"),
) -> JobListResponse:
    """List recently submitted optimization jobs.

    Finished jobs are retained in memory for a limited time and evicted
    afterwards, so older jobs may no longer appear.

    Args:
        limit: Maximum number of jobs to return (newest first).

    Returns:
        JobListResponse with job summaries.

    Example:
        ```python
        import requests

        response = requests.get("http://localhost:8000/api/v1/jobs?limit=10")
        for job in response.json()["jobs"]:
            print(job["job_id"], job["status"])
        ```
    """
    jobs = get_job_store().list_jobs(limit=limit)
    return JobListResponse(jobs=[_to_summary(job) for job in jobs])


@router.get(
    "/{job_id}",
    response_model=JobDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get optimization job status",
    description="Get the status and result of an asynchronous optimization job.",
    responses={
        200: {
            "description": "Job details",
            "model": JobDetailResponse,
        },
        404: {"description": "Job not found"},
    },
)
async def get_job(job_id: str) -> JobDetailResponse:
    """Get the status and result of an asynchronous optimization job.

    Poll this endpoint after submitting via ``POST /api/v1/optimize/async``.
    When the status is "completed" the full optimization result is included
    in the ``result`` field; when "failed" the ``error`` field explains why.

    Args:
        job_id: Job identifier returned at submission time.

    Returns:
        JobDetailResponse with the current job state.

    Raises:
        HTTPException: 404 if the job is unknown (or already evicted).

    Example:
        ```python
        import requests

        response = requests.get("http://localhost:8000/api/v1/jobs/abc123")
        job = response.json()
        if job["status"] == "completed":
            print(job["result"]["configuration"])
        ```
    """
    job = get_job_store().get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found. It may have expired or never existed.",
        )
    return _to_detail(job)
