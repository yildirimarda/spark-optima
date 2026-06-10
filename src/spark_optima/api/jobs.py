# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""In-memory job store for asynchronous optimization runs.

This module provides a thread-safe, in-memory job registry backed by a
small thread pool. Optimization work is synchronous CPU-bound code, so it
is executed on a ``ThreadPoolExecutor`` instead of FastAPI background
tasks (which would serialize work on the event loop).

The store is process-local: jobs are lost on restart and are not shared
across workers. Finished jobs are evicted opportunistically once they are
older than the configured TTL — no background cleanup threads are used.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

JobStatus = Literal["pending", "running", "completed", "failed"]

#: Default time-to-live for finished (completed/failed) jobs, in hours.
DEFAULT_JOB_TTL_HOURS = 6.0

#: Shared worker pool for job execution. Optimization is synchronous
#: CPU-bound work, so a small dedicated pool keeps the API responsive
#: without oversubscribing the host.
_EXECUTOR: Executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="spark-optima-job")


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 formatted UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    """A single asynchronous optimization job.

    Attributes:
        job_id: Unique job identifier (uuid4 hex).
        platform: Target platform requested for the optimization.
        spark_version: Spark version requested for the optimization.
        status: Current lifecycle status.
        submitted_at: UTC ISO timestamp when the job was accepted.
        started_at: UTC ISO timestamp when execution began, if started.
        finished_at: UTC ISO timestamp when execution ended, if finished.
        result: Optimization result payload when status is "completed".
        error: Error message when status is "failed".
        submitted_ts: Monotonic-ish epoch seconds used for ordering.
        finished_ts: Epoch seconds when the job finished, used for TTL eviction.
    """

    job_id: str
    platform: str
    spark_version: str
    status: JobStatus = "pending"
    submitted_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    submitted_ts: float = field(default=0.0, repr=False)
    finished_ts: float | None = field(default=None, repr=False)


class JobStore:
    """Thread-safe in-memory registry of asynchronous optimization jobs.

    Jobs are executed on a shared module-level thread pool (or an injected
    executor) and tracked in a plain dict guarded by a lock. Finished jobs
    older than the TTL are removed opportunistically whenever the store is
    accessed.

    Attributes:
        _jobs: Mapping of job_id to Job.
        _lock: Lock guarding all access to the job mapping.
        _ttl_seconds: TTL for finished jobs, in seconds.
        _executor: Optional executor override (used in tests).
    """

    def __init__(self, ttl_hours: float = DEFAULT_JOB_TTL_HOURS, executor: Executor | None = None) -> None:
        """Initialize the job store.

        Args:
            ttl_hours: Hours to retain finished (completed/failed) jobs.
            executor: Optional executor override; defaults to the shared
                module-level thread pool when None.
        """
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_hours * 3600.0
        self._executor = executor

    def submit(self, work: Callable[[], dict[str, Any]], platform: str, spark_version: str) -> str:
        """Register a new job and schedule it for execution.

        Args:
            work: Zero-argument callable performing the optimization and
                returning the result payload. Any exception it raises marks
                the job as failed with the exception message.
            platform: Target platform (stored as request summary).
            spark_version: Spark version (stored as request summary).

        Returns:
            The job identifier for status polling.
        """
        job_id = uuid.uuid4().hex
        job = Job(
            job_id=job_id,
            platform=platform,
            spark_version=spark_version,
            submitted_at=_utc_now_iso(),
            submitted_ts=time.time(),
        )
        with self._lock:
            self._cleanup_locked()
            self._jobs[job_id] = job

        executor = self._executor if self._executor is not None else _EXECUTOR
        executor.submit(self._run, job_id, work)
        logger.info(f"Submitted optimization job {job_id} (platform={platform}, spark_version={spark_version})")
        return job_id

    def get(self, job_id: str) -> Job | None:
        """Look up a job by identifier.

        Args:
            job_id: Job identifier returned by submit().

        Returns:
            The Job if known, None otherwise (including after TTL eviction).
        """
        with self._lock:
            self._cleanup_locked()
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[Job]:
        """List known jobs, newest first.

        Args:
            limit: Maximum number of jobs to return.

        Returns:
            Jobs ordered by submission time descending.
        """
        with self._lock:
            self._cleanup_locked()
            ordered = sorted(self._jobs.values(), key=lambda job: job.submitted_ts, reverse=True)
            return ordered[:limit]

    def _run(self, job_id: str, work: Callable[[], dict[str, Any]]) -> None:
        """Execute a job on the worker pool and record its outcome.

        Args:
            job_id: Identifier of the job to execute.
            work: The callable performing the actual optimization.
        """
        self._mark_running(job_id)
        try:
            result = work()
        except Exception as exc:  # noqa: BLE001 — jobs must never crash the worker pool
            logger.exception(f"Optimization job {job_id} failed")
            self._mark_failed(job_id, str(exc))
        else:
            self._mark_completed(job_id, result)

    def _mark_running(self, job_id: str) -> None:
        """Transition a job to the running state.

        Args:
            job_id: Identifier of the job to update.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "running"
                job.started_at = _utc_now_iso()

    def _mark_completed(self, job_id: str, result: dict[str, Any]) -> None:
        """Transition a job to the completed state.

        Args:
            job_id: Identifier of the job to update.
            result: The optimization result payload.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "completed"
                job.result = result
                job.finished_at = _utc_now_iso()
                job.finished_ts = time.time()

    def _mark_failed(self, job_id: str, error: str) -> None:
        """Transition a job to the failed state.

        Args:
            job_id: Identifier of the job to update.
            error: Human-readable failure message.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "failed"
                job.error = error
                job.finished_at = _utc_now_iso()
                job.finished_ts = time.time()

    def _cleanup_locked(self) -> None:
        """Evict finished jobs older than the TTL.

        Must be called with the lock held. Runs opportunistically on every
        store access so no background threads are needed.
        """
        if self._ttl_seconds <= 0:
            return
        cutoff = time.time() - self._ttl_seconds
        expired = [
            job_id for job_id, job in self._jobs.items() if job.finished_ts is not None and job.finished_ts < cutoff
        ]
        for job_id in expired:
            del self._jobs[job_id]
        if expired:
            logger.debug(f"Evicted {len(expired)} expired job(s) from the job store")


# Global store instance (singleton pattern, mirrors get_optimization_service)
_store_lock = threading.Lock()
_store_instance: JobStore | None = None


def get_job_store() -> JobStore:
    """Get the global job store instance.

    Returns:
        The shared JobStore singleton.
    """
    global _store_instance
    with _store_lock:
        if _store_instance is None:
            _store_instance = JobStore()
        return _store_instance


def reset_job_store() -> None:
    """Discard the global job store instance.

    Primarily intended for tests that need isolated job state.
    """
    global _store_instance
    with _store_lock:
        _store_instance = None
