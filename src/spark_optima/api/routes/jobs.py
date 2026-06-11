# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Job status endpoints for asynchronous optimization runs.

This module provides endpoints to poll the status and results of jobs
submitted via ``POST /api/v1/optimize/async``, plus a Server-Sent Events
stream (``GET /api/v1/jobs/{job_id}/events``) that pushes live optimization
progress until the job reaches a terminal state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from spark_optima.api.jobs import Job, get_job_store
from spark_optima.api.models import JobDetailResponse, JobListResponse, JobSummaryResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])

#: Seconds between job-store polls inside the SSE event loop. Module-level
#: so tests can patch it for fast, deterministic streaming.
EVENTS_POLL_INTERVAL_SECONDS = 0.5

#: Seconds of silence after which an SSE heartbeat comment is emitted to
#: keep intermediaries from closing the connection.
EVENTS_HEARTBEAT_INTERVAL_SECONDS = 10.0

#: Job statuses that end the SSE stream with a final "done" event.
TERMINAL_JOB_STATES = ("completed", "failed")


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
        webhook_status=job.webhook_status,
        progress=job.progress,
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


def _sse_frame(event: str, data: dict[str, Any]) -> str:
    """Format a Server-Sent Events frame.

    Args:
        event: Event name (e.g. "progress" or "done").
        data: JSON-serializable event payload.

    Returns:
        Wire-format SSE frame terminated by a blank line.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get(
    "/{job_id}/events",
    status_code=status.HTTP_200_OK,
    summary="Stream optimization job progress (SSE)",
    description=(
        "Stream live job progress as Server-Sent Events until the job reaches a terminal state. "
        "Emits 'progress' events whenever the progress snapshot changes, comment heartbeats "
        "(': keep-alive') during quiet periods, and a final 'done' event with the job status."
    ),
    responses={
        200: {
            "description": "Server-Sent Events stream of job progress",
            "content": {"text/event-stream": {"example": 'event: progress\ndata: {"trials_completed": 5}\n\n'}},
        },
        404: {"description": "Job not found"},
    },
)
async def stream_job_events(job_id: str) -> StreamingResponse:
    """Stream live progress of an asynchronous optimization job over SSE.

    The stream polls the job store and emits:

    - ``event: progress`` frames whenever the job's progress snapshot
      changes (the data is the JSON progress payload, the same object as
      the ``progress`` field of ``GET /api/v1/jobs/{job_id}``),
    - comment heartbeats (``: keep-alive``) roughly every 10 seconds of
      silence so proxies do not close the connection,
    - a final ``event: done`` frame with the terminal job status
      (``completed`` or ``failed``), after which the stream closes.

    Args:
        job_id: Job identifier returned at submission time.

    Returns:
        StreamingResponse with ``text/event-stream`` media type.

    Raises:
        HTTPException: 404 if the job is unknown (or already evicted).

    Example:
        ```bash
        curl -N http://localhost:8000/api/v1/jobs/abc123/events
        ```
    """
    store = get_job_store()
    # Store lookups are synchronous (and the sqlite backend performs cleanup
    # writes inside get()), so run them on a worker thread to keep the event
    # loop responsive.
    if await asyncio.to_thread(store.get, job_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found. It may have expired or never existed.",
        )

    async def event_stream() -> AsyncGenerator[str, None]:
        """Poll the job store and yield SSE frames until a terminal state."""
        last_progress: dict[str, Any] | None = None
        last_sent = time.monotonic()
        while True:
            # Never call the blocking store directly on the event loop: the
            # sqlite get() does TTL-cleanup WRITES and contends with WAL
            # writers, which would stall every other request while polling.
            job = await asyncio.to_thread(store.get, job_id)
            if job is None:
                # Evicted mid-stream (TTL): report and close instead of 404
                yield _sse_frame("done", {"job_id": job_id, "status": "expired", "error": "job expired"})
                return
            if job.progress is not None and job.progress != last_progress:
                last_progress = job.progress
                last_sent = time.monotonic()
                yield _sse_frame("progress", job.progress)
            if job.status in TERMINAL_JOB_STATES:
                yield _sse_frame("done", {"job_id": job.job_id, "status": job.status, "error": job.error})
                return
            if time.monotonic() - last_sent >= EVENTS_HEARTBEAT_INTERVAL_SECONDS:
                last_sent = time.monotonic()
                yield ": keep-alive\n\n"
            await asyncio.sleep(EVENTS_POLL_INTERVAL_SECONDS)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable proxy buffering (nginx)
        },
    )
